#!/usr/bin/env python3
"""Lance FineVision full-layer G-token DPO with shuffle-KL decay and U masking."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mmap
import os
import random
import shutil
import sys
import tempfile
import time
from array import array
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torchvision.transforms import functional as TF

WIDTH = 2048
WORK_WIDTH = 1024
TOKENS = 256
BF16 = torch.bfloat16
TRAIN_AMP_MODE = os.environ.get("LANCE_TRAIN_AMP_DTYPE", "bf16")
if TRAIN_AMP_MODE not in {"bf16", "fp16", "fp32"}:
    raise ValueError("LANCE_TRAIN_AMP_DTYPE must be bf16, fp16, or fp32")
TRAIN_AMP_DTYPE = torch.float16 if TRAIN_AMP_MODE == "fp16" else BF16
TRAIN_AMP_ENABLED = TRAIN_AMP_MODE != "fp32"
MODEL_RUNTIME_DTYPE = torch.float32 if TRAIN_AMP_MODE == "fp32" else BF16
SAMPLER_LAYERS = tuple(range(36))
RATING_FIELDS = (
    "visual_dependency_ratings",
    "image_correspondence_ratings",
    "relevance_ratings",
)
LOSS_SCALE = 1.0
LAMBDA_META = 0.5
ARCHITECTURE_ID = "gafu_paper_main_lance_v1"
CHECKPOINT_SCHEMA = "gafu_paper_main_lance_checkpoint_v1"
RUN_SCHEMA = "gafu_paper_main_lance_run_v1"


if os.environ.get("LANCE_BLAS_LIBRARY"):
    blas_library = os.environ["LANCE_BLAS_LIBRARY"]
    if blas_library != "cublas":
        raise ValueError("LANCE_BLAS_LIBRARY must be cublas when set")
    torch.backends.cuda.preferred_blas_library(blas_library)


class InstructionManifest:
    def __init__(self, path: str, image_root_from: str = "", image_root_to: str = "") -> None:
        self.path = Path(path)
        self.image_root_from = image_root_from.rstrip("/")
        self.image_root_to = image_root_to.rstrip("/")
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise RuntimeError(f"empty manifest: {self.path}")
        self._handle = None
        self._mapped: mmap.mmap | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, index: int) -> dict[str, Any]:
        if self._mapped is None:
            self._handle = self.path.open("rb")
            self._mapped = mmap.mmap(
                self._handle.fileno(), length=0, access=mmap.ACCESS_READ,
            )
        self._mapped.seek(self.offsets[index % len(self.offsets)])
        row = json.loads(self._mapped.readline())
        conversations = row.get("conversations", [])
        human = next((x for x in conversations if x.get("from") in {"human", "user"}), None)
        assistant = next((x for x in conversations if x.get("from") in {"gpt", "assistant"}), None)
        prompt = str(row.get("question", row.get("prompt", human.get("value", "") if human else "")))
        target = str(row.get("answer", row.get("fullAnswer", assistant.get("value", "") if assistant else "")))
        prompt = prompt.replace("<image>", "").strip()
        target = target.strip()
        image_path = str(row.get("image", row.get("image_path", "")))
        if self.image_root_from and image_path.startswith(self.image_root_from + "/"):
            image_path = self.image_root_to + image_path[len(self.image_root_from):]
        if not prompt or not target or not image_path:
            raise ValueError(f"invalid caption row: {row.get('id')}")
        return {
            "id": str(row.get("id", index)),
            "image_path": image_path,
            "prompt": prompt,
            "answer": target,
            "caption_words": len(target.split()),
        }


class FineVisionManifest:
    """Lazy row access for FineVision Parquet shards with embedded image bytes."""

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        base = self.manifest_path.parent
        self.files: list[tuple[Path, int]] = []
        for item in payload.get("files", []):
            path = base / item["path"]
            if not path.exists():
                path = base.parent / item["path"]
            if "text_only" in path.parts or not path.exists():
                continue
            self.files.append((path, int(item["rows"])))
        if not self.files:
            raise RuntimeError(f"no FineVision Parquet shards in {self.manifest_path}")
        self.ends: list[int] = []
        total = 0
        for _, rows in self.files:
            total += rows
            self.ends.append(total)
        self._file_index: int | None = None
        self._parquet: pq.ParquetFile | None = None
        self._row_group: int | None = None
        self._table: Any = None

    def __len__(self) -> int:
        return self.ends[-1]

    def _row(self, index: int) -> dict[str, Any]:
        index %= len(self)
        file_index = bisect_right(self.ends, index)
        start = 0 if file_index == 0 else self.ends[file_index - 1]
        local_index = index - start
        path, _ = self.files[file_index]
        if self._file_index != file_index:
            self._file_index = file_index
            self._parquet = pq.ParquetFile(path)
            self._row_group = None
        assert self._parquet is not None
        group_start = 0
        for group_index in range(self._parquet.num_row_groups):
            group_rows = self._parquet.metadata.row_group(group_index).num_rows
            if local_index < group_start + group_rows:
                if self._row_group != group_index:
                    self._row_group = group_index
                    self._table = self._parquet.read_row_group(group_index)
                assert self._table is not None
                row_index = local_index - group_start
                return {
                    name: self._table[name][row_index].as_py()
                    for name in self._table.column_names
                }
            group_start += group_rows
        raise IndexError(index)

    def get(self, index: int) -> dict[str, Any]:
        row = self._row(index)
        images = row.get("images") or []
        texts = row.get("texts") or []
        if not images or not texts:
            raise ValueError(f"invalid FineVision row {index}")
        ratings = {field: row.get(field) for field in RATING_FIELDS}
        if any(values is None or len(values) != len(texts) for values in ratings.values()):
            raise ValueError(f"FineVision rating alignment mismatch at row {index}")
        selected: tuple[str, str, int] | None = None
        for turn_index, turn in enumerate(texts):
            user = str((turn or {}).get("user", "")).replace("<image>", "").strip()
            assistant = str((turn or {}).get("assistant", "")).strip()
            if user and assistant:
                selected = (user, assistant, turn_index)
        if selected is None:
            raise ValueError(f"FineVision row {index} has no user/assistant turn")
        prompt, answer, turn_index = selected
        image_bytes = (images[0] or {}).get("bytes")
        if not image_bytes:
            raise ValueError(f"FineVision row {index} has no embedded image bytes")
        result = {
            "id": f"finevision-{index}",
            "image_bytes": image_bytes,
            "prompt": prompt,
            "answer": answer,
            "caption_words": len(answer.split()),
            "visual_dependency": int(ratings["visual_dependency_ratings"][turn_index]),
            "image_correspondence": int(ratings["image_correspondence_ratings"][turn_index]),
            "relevance": int(ratings["relevance_ratings"][turn_index]),
            "turn_index": turn_index,
        }
        if row.get("donor_index") is not None:
            result["donor_index"] = int(row["donor_index"])
        if row.get("toy_code") is not None:
            result["toy_code"] = str(row["toy_code"])
        if row.get("donor_code") is not None:
            result["donor_code"] = str(row["donor_code"])
        return result


def load_manifest(
    path: str, image_root_from: str = "", image_root_to: str = "",
) -> InstructionManifest | FineVisionManifest:
    if Path(path).name == "DATASET_MANIFEST.json":
        return FineVisionManifest(path)
    return InstructionManifest(path, image_root_from, image_root_to)


def image_source(row: dict[str, Any]) -> str | bytes | bytearray:
    if "image_bytes" in row:
        return row["image_bytes"]
    return row["image_path"]


def correspondence_gate(rating: int) -> float:
    if rating <= 2:
        return 0.0
    if rating == 3:
        return 0.5
    return 1.0


def metadata_weight(row: dict[str, Any]) -> float:
    if "visual_dependency" not in row or "image_correspondence" not in row:
        return 1.0
    visual_score = (int(row["visual_dependency"]) - 1) / 4.0
    return 1.0 + LAMBDA_META * visual_score * correspondence_gate(
        int(row["image_correspondence"]),
    )


def lm_head_logits(lm_head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    vocab_chunk_size = int(os.environ.get("LANCE_LM_HEAD_VOCAB_CHUNK", "0"))
    if vocab_chunk_size <= 0:
        return lm_head(hidden)
    if not isinstance(lm_head, nn.Linear):
        raise TypeError("vocabulary-chunked lm_head requires nn.Linear")
    bias = lm_head.bias
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return torch.cat([
            F.linear(
                hidden,
                lm_head.weight[start:start + vocab_chunk_size],
                None if bias is None else bias[start:start + vocab_chunk_size],
            )
            for start in range(0, lm_head.out_features, vocab_chunk_size)
        ], dim=-1)


def token_cross_entropy(
    lm_head: nn.Module,
    selected_hidden: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None or selected_hidden.shape[0] <= chunk_size:
        return F.cross_entropy(lm_head_logits(lm_head, selected_hidden), labels, reduction="none")
    return torch.cat([
        F.cross_entropy(lm_head_logits(lm_head, hidden_chunk), label_chunk, reduction="none")
        for hidden_chunk, label_chunk in zip(
            selected_hidden.split(chunk_size),
            labels.split(chunk_size),
            strict=True,
        )
    ])


def select_rows(
    manifest: InstructionManifest | FineVisionManifest,
    base_serial: int,
    mix_a: int,
    mix_b: int,
    micro_batch_size: int,
    max_answer_words: int,
    max_microbatch_words: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    total_words = 0
    cursor = base_serial
    scanned = 0
    while len(selected) < micro_batch_size:
        if scanned >= len(manifest):
            raise RuntimeError(
                "caption budget cannot form a microbatch: "
                f"selected={len(selected)} total_words={total_words} "
                f"budget={max_microbatch_words}"
            )
        row = manifest.get((mix_a * cursor + mix_b) % len(manifest))
        cursor += 1
        scanned += 1
        words = int(row["caption_words"])
        if words > max_answer_words:
            continue
        remaining_slots = micro_batch_size - len(selected) - 1
        if total_words + words + remaining_slots > max_microbatch_words:
            continue
        selected.append(row)
        total_words += words
    return selected, scanned - len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--flexae-checkpoint", required=True)
    parser.add_argument("--flexae-root", required=True)
    parser.add_argument("--lance-runtime", required=True)
    parser.add_argument("--lance-model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("stage2",), required=True)
    parser.add_argument("--native-lora-sft", action="store_true")
    parser.add_argument("--init-bridge-checkpoint", default=None)
    parser.add_argument("--init-stage2-checkpoint", default=None)
    parser.add_argument("--init-step", type=int, default=0)
    parser.add_argument("--image-root-from", default="")
    parser.add_argument("--image-root-to", default="")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--mask-u-prob", type=float, default=0.0)
    parser.add_argument("--mask-u-schedule", default="")
    parser.add_argument("--mask-u-schedule-boundaries", default="")
    parser.add_argument("--mask-u-schedule-horizon", type=int, default=None)
    parser.add_argument(
        "--mask-u-granularity",
        choices=("per_sample_per_vit_token", "per_sample_whole_u"),
        default="per_sample_per_vit_token",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--g-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--g-pair-temperature", type=float, default=0.07)
    parser.add_argument("--real-ce-loss-weight", type=float, default=5.0)
    parser.add_argument("--answer-use-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-use-beta", type=float, default=0.1)
    parser.add_argument("--dpo-real-weight", type=float, default=4.0)
    parser.add_argument("--dpo-negative-weight", type=float, default=1.0)
    parser.add_argument("--dpo-negative-grad-cap", type=float, default=0.25)
    parser.add_argument(
        "--answer-use-reference",
        choices=("zero_delta",),
        default="zero_delta",
    )
    parser.add_argument("--shuffle-kl-weight", type=float, default=0.0)
    parser.add_argument("--shuffle-kl-schedule-steps", default="")
    parser.add_argument("--shuffle-kl-schedule-weights", default="")
    parser.add_argument("--drop-kl-weight", type=float, default=0.0)
    parser.add_argument("--train-kl-chunk-size", type=int, default=32)
    parser.add_argument("--projector-lr", type=float, default=2e-5)
    parser.add_argument("--fuser-lr", type=float, default=2e-5)
    parser.add_argument("--sampler-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--extractor-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-master-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bridge-master-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=8)
    parser.add_argument("--final-val-limit", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--max-answer-words", type=int, default=256)
    parser.add_argument("--max-microbatch-words", type=int, default=300)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-packed-tokens", type=int, default=None)
    parser.add_argument("--prefetch-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--balanced-order", default=None)
    parser.add_argument("--encoder-unfreeze-step", type=int, default=2501)
    parser.add_argument("--tower-handoff-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--schedule-max-steps", type=int, default=None)
    parser.add_argument("--resume-ramp-start-step", type=int, default=None)
    parser.add_argument("--resume-ramp-steps", type=int, default=0)
    parser.add_argument("--gradient-audit-steps", default="")
    parser.add_argument("--gradient-audit-dir", default=None)
    parser.add_argument("--mix-seed", type=int, default=20260808)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-latest", type=int, default=2)
    return parser.parse_args()


def parse_mask_u_schedule(
    values_text: str,
    boundaries_text: str,
    max_steps: int,
) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    if not values_text and not boundaries_text:
        return None
    if not values_text or not boundaries_text:
        raise ValueError("mask-U schedule values and boundaries must be provided together")
    values = tuple(float(value.strip()) for value in values_text.split(",") if value.strip())
    boundaries = tuple(int(value.strip()) for value in boundaries_text.split(",") if value.strip())
    if not values or len(values) != len(boundaries):
        raise ValueError("mask-U schedule values and boundaries must have equal non-zero lengths")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"mask-U schedule probabilities must be in [0, 1]: {values}")
    if any(boundary < 1 for boundary in boundaries):
        raise ValueError(f"mask-U schedule boundaries must be positive: {boundaries}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"mask-U schedule boundaries must be strictly increasing: {boundaries}")
    if boundaries[-1] != max_steps:
        raise ValueError(
            f"final mask-U schedule boundary must equal max_steps: {boundaries[-1]} != {max_steps}"
        )
    return values, boundaries


def mask_u_probability_for_step(
    step: int,
    schedule: tuple[tuple[float, ...], tuple[int, ...]] | None,
    fallback: float,
) -> float:
    if schedule is None:
        return fallback
    values, boundaries = schedule
    index = bisect_left(boundaries, step)
    if index >= len(values):
        raise ValueError(f"step {step} exceeds mask-U schedule ending at {boundaries[-1]}")
    return values[index]


def parse_shuffle_kl_schedule(
    steps_text: str,
    weights_text: str,
    max_steps: int,
) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    if not steps_text and not weights_text:
        return None
    if not steps_text or not weights_text:
        raise ValueError("shuffle-KL schedule steps and weights must be provided together")
    steps = tuple(int(value.strip()) for value in steps_text.split(",") if value.strip())
    weights = tuple(float(value.strip()) for value in weights_text.split(",") if value.strip())
    if not steps or len(steps) != len(weights):
        raise ValueError("shuffle-KL schedule steps and weights must have equal non-zero lengths")
    if steps[0] != 1 or steps[-1] != max_steps:
        raise ValueError(f"shuffle-KL schedule must span step 1 through {max_steps}: {steps}")
    if any(left >= right for left, right in zip(steps, steps[1:])):
        raise ValueError(f"shuffle-KL schedule steps must be strictly increasing: {steps}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"shuffle-KL schedule weights must be non-negative: {weights}")
    return steps, weights


def piecewise_shuffle_kl_weight_for_step(
    step: int,
    schedule: tuple[tuple[int, ...], tuple[float, ...]],
) -> float:
    steps, weights = schedule
    if step < steps[0] or step > steps[-1]:
        raise ValueError(f"step {step} is outside shuffle-KL schedule [{steps[0]}, {steps[-1]}]")
    right = bisect_left(steps, step)
    if right < len(steps) and steps[right] == step:
        return weights[right]
    left = right - 1
    alpha = (step - steps[left]) / (steps[right] - steps[left])
    return weights[left] + alpha * (weights[right] - weights[left])


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pil(value: str | bytes | bytearray) -> Image.Image:
    stream = io.BytesIO(value) if isinstance(value, (bytes, bytearray)) else value
    with Image.open(stream) as source:
        return source.convert("RGB")


def vae_tensor(value: Image.Image) -> torch.Tensor:
    return TF.pil_to_tensor(value.resize((512, 512), Image.Resampling.BICUBIC)).float().div_(127.5).sub_(1.0)


def atomic_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def token_id(new_token_ids: dict[str, Any], key: str) -> int:
    value = new_token_ids[key]
    return int(value.item()) if torch.is_tensor(value) else int(value)


def cosine_lr_factor(
    step: int, warmup_steps: int, schedule_max_steps: int, min_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, schedule_max_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, progress))
    )


def resumed_lr_factor(
    step: int,
    warmup_steps: int,
    schedule_max_steps: int,
    min_lr_ratio: float,
    ramp_start_step: int | None,
    ramp_steps: int,
) -> float:
    target = cosine_lr_factor(step, warmup_steps, schedule_max_steps, min_lr_ratio)
    if ramp_start_step is None or not (ramp_start_step < step <= ramp_start_step + ramp_steps):
        return target
    alpha = (step - ramp_start_step) / ramp_steps
    return min_lr_ratio + alpha * (target - min_lr_ratio)


class GlobalConditionedResidualAdapter(nn.Module):
    def __init__(self, bottleneck: int = WORK_WIDTH) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(WIDTH, eps=1e-6)
        self.down = nn.Linear(WIDTH, bottleneck, bias=False)
        self.context = nn.Linear(bottleneck, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, WIDTH, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.down(self.norm(tokens)), approximate="tanh")
        gate = 1.0 + torch.tanh(self.context(hidden.mean(1, keepdim=True)))
        return tokens + self.up(hidden * gate)


class GProjector(nn.Module):
    """Trainable projector whose initial mapping is exactly raw_g -> raw_g."""

    def __init__(self) -> None:
        super().__init__()
        self.calibrator = nn.Linear(WIDTH, WIDTH, bias=True)
        self.adapter = GlobalConditionedResidualAdapter(WORK_WIDTH)
        with torch.no_grad():
            self.calibrator.weight.copy_(torch.eye(WIDTH))
            self.calibrator.bias.zero_()
            self.adapter.up.weight.zero_()

    def forward(self, raw_g: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.calibrator(raw_g))


class GlobalAttentionBase(nn.Module):
    """Global attention over all 256 G tokens, with no positional bias."""

    def __init__(self) -> None:
        super().__init__()

    def global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, query_len, _ = q.shape
        if k.shape[:2] != (batch, TOKENS) or v.shape[:2] != (batch, TOKENS):
            raise ValueError(f"expected K/V [B,{TOKENS},D], got {tuple(k.shape)}/{tuple(v.shape)}")
        q = q.view(batch, query_len, 16, 64).transpose(1, 2)
        k = k.view(batch, TOKENS, 16, 64).transpose(1, 2)
        v = v.view(batch, TOKENS, 16, 64).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        return mixed.transpose(1, 2).reshape(batch, query_len, WORK_WIDTH)


def smooth_bidirectional_pair_score(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Score Q/K compatibility using the projections that drive cross-attention."""
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise ValueError(f"invalid pair-score shapes: q={tuple(q.shape)} k={tuple(k.shape)}")
    if q.shape[-1] != WORK_WIDTH or k.shape[-1] != WORK_WIDTH:
        raise ValueError(f"pair-score width must be {WORK_WIDTH}")
    if temperature <= 0.0:
        raise ValueError("pair-score temperature must be positive")
    batch, q_len, _ = q.shape
    k_len = k.shape[1]
    q_heads = q.reshape(batch, q_len, 16, 64).transpose(1, 2)
    k_heads = k.reshape(batch, k_len, 16, 64).transpose(1, 2)
    q_heads = F.normalize(q_heads.float(), dim=-1)
    k_heads = F.normalize(k_heads.float(), dim=-1)
    similarity = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / temperature
    q_to_k = torch.logsumexp(similarity, dim=-1) - math.log(k_len)
    k_to_q = torch.logsumexp(similarity, dim=-2) - math.log(q_len)
    return 0.5 * (q_to_k.mean(dim=(-1, -2)) + k_to_q.mean(dim=(-1, -2)))


def pairwise_logistic_loss(real_score: torch.Tensor, shuffled_score: torch.Tensor) -> torch.Tensor:
    if real_score.shape != shuffled_score.shape:
        raise ValueError(
            f"pair-score shape mismatch: real={tuple(real_score.shape)} "
            f"shuffled={tuple(shuffled_score.shape)}"
        )
    return F.softplus(shuffled_score - real_score).mean()


def answer_use_copo_loss(
    real_ce: torch.Tensor,
    shuffled_ce: torch.Tensor,
    beta: float,
    reference_gap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Increase matched-G answer advantage relative to a fixed reference gap."""
    if real_ce.shape != shuffled_ce.shape:
        raise ValueError(
            f"answer-use CE shape mismatch: real={tuple(real_ce.shape)} "
            f"shuffle={tuple(shuffled_ce.shape)}"
        )
    if beta <= 0.0:
        raise ValueError("answer-use beta must be positive")
    if reference_gap is None:
        reference_gap = torch.zeros_like(real_ce)
    if reference_gap.shape != real_ce.shape:
        raise ValueError("answer-use reference gap shape mismatch")
    current_gap = shuffled_ce - real_ce
    return F.softplus(-beta * (current_gap - reference_gap)).mean()


def forward_kl_per_token(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-token KL(reference || current), with no reference gradient."""
    if current_logits.shape != reference_logits.shape or current_logits.ndim != 2:
        raise ValueError(
            f"logit shape mismatch: current={tuple(current_logits.shape)} "
            f"reference={tuple(reference_logits.shape)}"
        )
    reference_log_probs = F.log_softmax(reference_logits.detach().float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    current_log_probs = F.log_softmax(current_logits.float(), dim=-1)
    return (reference_probs * (reference_log_probs - current_log_probs)).sum(dim=-1)


def forward_kl_loss(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None or current_logits.shape[0] <= chunk_size:
        return forward_kl_per_token(current_logits, reference_logits).mean()
    if chunk_size < 1:
        raise ValueError("KL chunk size must be positive")
    total = current_logits.new_zeros((), dtype=torch.float32)
    for current_chunk, reference_chunk in zip(
        current_logits.split(chunk_size),
        reference_logits.split(chunk_size),
        strict=True,
    ):
        per_token = torch_checkpoint(
            forward_kl_per_token,
            current_chunk,
            reference_chunk,
            use_reentrant=False,
        )
        total = total + per_token.sum()
    return total / current_logits.shape[0]


def forward_kl_from_hidden_per_token(
    lm_head: nn.Module,
    current_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute validation KL without materializing full-sequence vocabulary logits."""
    if current_hidden.shape != reference_hidden.shape or current_hidden.ndim != 2:
        raise ValueError(
            f"hidden shape mismatch: current={tuple(current_hidden.shape)} "
            f"reference={tuple(reference_hidden.shape)}"
        )
    if chunk_size < 1:
        raise ValueError("KL hidden chunk size must be positive")
    chunks = []
    for current_chunk, reference_chunk in zip(
        current_hidden.split(chunk_size),
        reference_hidden.split(chunk_size),
        strict=True,
    ):
        current_logits = lm_head(current_chunk)
        reference_logits = lm_head(reference_chunk)
        chunks.append(forward_kl_per_token(current_logits, reference_logits))
    return torch.cat(chunks)


def trainable_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    named = {
        f"bridge.{name}": parameter
        for name, parameter in model.bridge.named_parameters()
    }
    named.update({
        f"lora.{name}": parameter
        for name, parameter in understanding_lora_named_parameters(model.bagel)
    })
    return named


def synchronized_branch_gradients(
    loss: torch.Tensor,
    parameters: list[nn.Parameter],
    retain_graph: bool,
    world_size: int,
) -> list[torch.Tensor]:
    active_parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, active_parameters, retain_graph=retain_graph, allow_unused=True,
    )
    values = [
        torch.zeros_like(parameter, dtype=torch.float32) if gradient is None
        else gradient.detach().float()
        for parameter, gradient in zip(active_parameters, gradients, strict=True)
    ]
    if world_size > 1:
        for value in values:
            dist.all_reduce(value)
            value.div_(world_size)
    by_id = {id(parameter): value for parameter, value in zip(active_parameters, values, strict=True)}
    return [
        by_id.get(id(parameter), torch.zeros(parameter.numel(), dtype=torch.float32).view_as(parameter))
        for parameter in parameters
    ]


def global_gradient_norm(values: list[torch.Tensor]) -> float:
    return math.sqrt(math.fsum(float(value.norm()) ** 2 for value in values))


@contextmanager
def use_reference_parameters(
    model: nn.Module,
    reference_state: dict[str, torch.Tensor],
):
    """Temporarily swap only bridge/LoRA tensors without duplicating Lance."""
    named = trainable_named_parameters(model)
    if set(named) != set(reference_state):
        raise RuntimeError(
            f"reference state mismatch: missing={sorted(set(named) - set(reference_state))[:3]} "
            f"unexpected={sorted(set(reference_state) - set(named))[:3]}"
        )
    current = {
        name: parameter.detach().clone()
        for name, parameter in named.items()
    }
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(reference_state[name].to(parameter.device, parameter.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(current[name])


@contextmanager
def disable_understanding_lora(bagel: nn.Module):
    modules = [module for module in bagel.modules() if isinstance(module, UnderstandingLoRALinear)]
    scales = [module.scale for module in modules]
    for module in modules:
        module.scale = 0.0
    try:
        yield
    finally:
        for module, scale in zip(modules, scales, strict=True):
            module.scale = scale


class EntrySpatialFuser(GlobalAttentionBase):
    """Flamingo-style gated cross-attention from raw U queries into projected G."""

    def __init__(self) -> None:
        super().__init__()
        self.u_norm = nn.RMSNorm(WIDTH, eps=1e-6)
        self.g_norm = nn.RMSNorm(WIDTH, eps=1e-6)
        self.q = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.k = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.v = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.out = nn.Linear(WORK_WIDTH, WIDTH, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, u: torch.Tensor, projected_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.g_norm(projected_g)
        mixed = self.global_attention(self.q(self.u_norm(u)), self.k(g), self.v(g))
        delta = torch.tanh(self.gate) * self.out(mixed)
        return u + delta, delta


class TextHiddenGSampler(GlobalAttentionBase):
    """Use contextual frozen-LLM language states as queries into G memory."""

    def __init__(self) -> None:
        super().__init__()
        self.h_norm = nn.RMSNorm(WIDTH, eps=1e-6)
        self.g_norm = nn.RMSNorm(WIDTH, eps=1e-6)
        self.z_norm = nn.RMSNorm(WORK_WIDTH, eps=1e-6)
        self.q = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.k = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.v = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.o = nn.Linear(WORK_WIDTH, WORK_WIDTH, bias=False)
        self.mid = nn.Linear(WORK_WIDTH, WORK_WIDTH, bias=False)
        self.up = nn.Linear(WORK_WIDTH, WIDTH, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self, hidden: torch.Tensor, projected_g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(self.h_norm(hidden))
        g = self.g_norm(projected_g)
        k = self.k(g)
        attended = self.o(self.global_attention(q, k, self.v(g)))
        middle = F.gelu(self.mid(self.z_norm(attended)), approximate="tanh")
        delta = self.up(middle)
        return hidden + delta, delta, q, k


class Stage3Bridge(nn.Module):
    def __init__(self, pair_temperature: float) -> None:
        super().__init__()
        if pair_temperature <= 0.0:
            raise ValueError("pair temperature must be positive")
        self.pair_temperature = float(pair_temperature)
        self.projector = GProjector()
        self.deep_samplers = nn.ModuleList([
            TextHiddenGSampler() for _ in SAMPLER_LAYERS
        ])
        self._context: tuple[list[torch.Tensor], torch.Tensor, bool] | None = None
        self._capture_pair_scores = False
        self.last_entry_ratio = 0.0
        self.last_deep_ratios = [0.0] * len(SAMPLER_LAYERS)
        self.last_deep_queries: list[list[torch.Tensor] | None] = [None] * len(SAMPLER_LAYERS)
        self.last_deep_pair_scores: list[torch.Tensor | None] = [None] * len(SAMPLER_LAYERS)

    def begin(
        self,
        u: torch.Tensor,
        raw_g: torch.Tensor,
        language_indexes: list[torch.Tensor],
        use_entry_g: bool,
        use_deep_g: bool,
        position_embedding: torch.Tensor,
        capture_pair_scores: bool = False,
    ) -> torch.Tensor:
        if not use_deep_g:
            self._context = (language_indexes, raw_g, False)
            self.last_entry_ratio = 0.0
            self.last_deep_ratios = [0.0] * len(SAMPLER_LAYERS)
            self._capture_pair_scores = False
            self.last_deep_queries = [None] * len(SAMPLER_LAYERS)
            self.last_deep_pair_scores = [None] * len(SAMPLER_LAYERS)
            return u
        projected_g = self.projector(raw_g)
        positioned_g = projected_g + position_embedding.to(
            device=projected_g.device, dtype=projected_g.dtype,
        )
        self._context = (language_indexes, positioned_g, use_deep_g)
        self._capture_pair_scores = bool(capture_pair_scores)
        self.last_entry_ratio = 0.0
        self.last_deep_ratios = [0.0] * len(SAMPLER_LAYERS)
        self.last_deep_queries = [None] * len(SAMPLER_LAYERS)
        self.last_deep_pair_scores = [None] * len(SAMPLER_LAYERS)
        return u

    def clear_context(self) -> None:
        self._context = None
        self._capture_pair_scores = False
        self.last_deep_queries = [None] * len(SAMPLER_LAYERS)
        self.last_deep_pair_scores = [None] * len(SAMPLER_LAYERS)

    def apply_deep_sampler(self, sampler_index: int, packed_sequence: torch.Tensor) -> torch.Tensor:
        if self._context is None:
            raise RuntimeError("deep sampler called without active bridge context")
        language_indexes, projected_g, use_g = self._context
        if not use_g:
            return packed_sequence
        updated = packed_sequence.clone()
        delta_square_sum = packed_sequence.new_zeros((), dtype=torch.float32)
        hidden_square_sum = packed_sequence.new_zeros((), dtype=torch.float32)
        element_count = 0
        pair_queries = []
        pair_scores = []
        for batch_index, indexes in enumerate(language_indexes):
            hidden = packed_sequence[indexes].unsqueeze(0)
            sampled, delta, q, k = self.deep_samplers[sampler_index](
                hidden, projected_g[batch_index:batch_index + 1],
            )
            # Deep samplers keep FP32 master weights, while Lance hidden states
            # are commonly BF16 under autocast. Match the destination dtype at
            # the hook boundary before writing the residual update back.
            updated[indexes] = sampled.squeeze(0).to(dtype=updated.dtype)
            delta_square_sum = delta_square_sum + delta.float().square().sum()
            hidden_square_sum = hidden_square_sum + hidden.float().square().sum()
            element_count += hidden.numel()
            if self._capture_pair_scores:
                pair_queries.append(q)
                pair_scores.append(smooth_bidirectional_pair_score(q, k, self.pair_temperature))
        self.last_deep_ratios[sampler_index] = float(
            (delta_square_sum / element_count).sqrt() /
            (hidden_square_sum / element_count).sqrt().clamp_min(1e-12)
        )
        if self._capture_pair_scores:
            self.last_deep_queries[sampler_index] = pair_queries
            self.last_deep_pair_scores[sampler_index] = torch.cat(pair_scores)
        return updated

    def deep_pair_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        if any(score is None for score in self.last_deep_pair_scores):
            raise RuntimeError("deep pair scores are incomplete")
        by_layer = torch.stack([
            score for score in self.last_deep_pair_scores if score is not None
        ])
        return by_layer.mean(dim=0), by_layer

    def shuffled_deep_pair_scores(
        self, raw_g: torch.Tensor, position_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if any(queries is None for queries in self.last_deep_queries):
            raise RuntimeError("deep pair queries are incomplete")
        projected_g = self.projector(raw_g)
        positioned_g = projected_g + position_embedding.to(
            device=projected_g.device, dtype=projected_g.dtype,
        )
        layer_scores = []
        for sampler, query_list in zip(self.deep_samplers, self.last_deep_queries, strict=True):
            assert query_list is not None
            g = sampler.g_norm(positioned_g)
            k = sampler.k(g)
            scores = [
                smooth_bidirectional_pair_score(
                    query, k[index:index + 1], self.pair_temperature,
                )
                for index, query in enumerate(query_list)
            ]
            layer_scores.append(torch.cat(scores))
        by_layer = torch.stack(layer_scores)
        return by_layer.mean(dim=0), by_layer


class UnderstandingLoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, master_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(
            (rank, base.in_features), device=base.weight.device, dtype=master_dtype,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            (base.out_features, rank), device=base.weight.device, dtype=master_dtype,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_dtype = self.base.weight.dtype
        base_output = self.base(values.to(dtype=base_dtype)).to(dtype=values.dtype)
        lora_a = self.lora_A.to(dtype=values.dtype)
        lora_b = self.lora_B.to(dtype=values.dtype)
        return base_output + F.linear(F.linear(values, lora_a), lora_b) * self.scale


UNDERSTANDING_LORA_PATHS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def install_understanding_lora(
    bagel: nn.Module, rank: int, alpha: float, master_dtype: torch.dtype,
) -> tuple[str, ...]:
    installed = []
    for layer_index, layer in enumerate(bagel._get_base_language_model().model.layers):
        for path in UNDERSTANDING_LORA_PATHS:
            parent = layer
            parts = path.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            name = parts[-1]
            linear = getattr(parent, name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"LoRA target is not linear: layer={layer_index} path={path}")
            setattr(parent, name, UnderstandingLoRALinear(linear, rank, alpha, master_dtype))
            installed.append(f"model.layers.{layer_index}.{path}")
    return tuple(installed)


def understanding_lora_named_parameters(bagel: nn.Module) -> list[tuple[str, nn.Parameter]]:
    result = []
    for module_name, module in bagel._get_base_language_model().named_modules():
        if isinstance(module, UnderstandingLoRALinear):
            result.extend(((f"{module_name}.lora_A", module.lora_A), (f"{module_name}.lora_B", module.lora_B)))
    return result


def understanding_lora_state_dict(bagel: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in understanding_lora_named_parameters(bagel)}


def load_understanding_lora_state(bagel: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = dict(understanding_lora_named_parameters(bagel))
    if set(state) != set(expected):
        raise RuntimeError(f"LoRA state mismatch: missing={len(set(expected)-set(state))} unexpected={len(set(state)-set(expected))}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def bridge_parameter_groups(bridge: Stage3Bridge, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"params": list(bridge.projector.parameters()), "lr": args.projector_lr, "name": "projector"},
        {"params": list(bridge.deep_samplers.parameters()), "lr": args.sampler_lr, "name": "deep_samplers"},
    ]


class Stage3CaptionModel(nn.Module):
    def __init__(
        self, bagel: nn.Module, tokenizer: Any, position_embedding: torch.Tensor,
        max_prompt_tokens: int = 512,
        mask_u_prob: float = 0.0,
        mask_u_granularity: str = "per_sample_per_vit_token",
        pair_temperature: float = 0.07,
        native_lora_sft: bool = False,
    ) -> None:
        super().__init__()
        self.bagel = bagel
        self.bridge = Stage3Bridge(pair_temperature)
        self.tokenizer = tokenizer
        if max_prompt_tokens < 4:
            raise ValueError("max_prompt_tokens must be at least 4")
        if not 0.0 <= mask_u_prob <= 1.0:
            raise ValueError(f"mask_u_prob must be in [0, 1], got {mask_u_prob}")
        self.max_prompt_tokens = max_prompt_tokens
        self.mask_u_prob = float(mask_u_prob)
        if mask_u_granularity not in {"per_sample_per_vit_token", "per_sample_whole_u"}:
            raise ValueError(f"unknown mask_u_granularity: {mask_u_granularity}")
        self.mask_u_granularity = mask_u_granularity
        self.native_lora_sft = bool(native_lora_sft)
        self.last_u_mask_ratio = 0.0
        self.last_prompt_truncations = 0
        self.register_buffer("position_embedding", position_embedding, persistent=False)
        base = bagel._get_base_language_model()
        if len(base.model.layers) <= SAMPLER_LAYERS[-1]:
            raise RuntimeError(f"Lance has only {len(base.model.layers)} layers")
        self._hook_handles = []
        for sampler_index, layer_index in (() if self.native_lora_sft else enumerate(SAMPLER_LAYERS)):
            def hook(_module, _args, output, index=sampler_index):
                return self.bridge.apply_deep_sampler(index, output)
            self._hook_handles.append(base.model.layers[layer_index].register_forward_hook(hook))

    def forward(
        self,
        raw_u: torch.Tensor,
        raw_g: torch.Tensor,
        prompts: list[str],
        answers: list[str],
        mode: str = "real",
        ce_chunk_size: int | None = None,
        capture_pair_scores: bool = False,
        fixed_u_mask: torch.Tensor | None = None,
        return_answer_logits: bool = False,
        return_answer_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(prompts)
        if raw_u.shape[0] != batch or len(answers) != batch:
            raise ValueError("tower/text batch mismatch")
        bridge_dtype = self.bridge.projector.calibrator.weight.dtype
        raw_u = raw_u.to(dtype=bridge_dtype)
        raw_g = raw_g.to(dtype=bridge_dtype)
        self.last_u_mask_ratio = 0.0
        if fixed_u_mask is not None:
            if fixed_u_mask.shape != raw_u.shape[:2] or fixed_u_mask.dtype != torch.bool:
                raise ValueError(
                    f"fixed U mask must be bool {tuple(raw_u.shape[:2])}, "
                    f"got {fixed_u_mask.dtype} {tuple(fixed_u_mask.shape)}"
                )
            token_mask = fixed_u_mask.to(device=raw_u.device)
            raw_u = raw_u.masked_fill(token_mask.unsqueeze(-1), 0.0)
            self.last_u_mask_ratio = float(token_mask.float().mean().detach())
        elif self.mask_u_prob > 0.0 and torch.is_grad_enabled():
            if self.mask_u_granularity == "per_sample_per_vit_token":
                token_mask = torch.rand(raw_u.shape[:2], device=raw_u.device) < self.mask_u_prob
            else:
                sample_mask = torch.rand(raw_u.shape[0], 1, device=raw_u.device) < self.mask_u_prob
                token_mask = sample_mask.expand(raw_u.shape[0], raw_u.shape[1])
            raw_u = raw_u.masked_fill(token_mask.unsqueeze(-1), 0.0)
            self.last_u_mask_ratio = float(token_mask.float().mean().detach())
        start_id = token_id(self.bagel.new_token_ids, "start_of_image")
        end_id = token_id(self.bagel.new_token_ids, "end_of_image")
        bos_id = token_id(self.bagel.new_token_ids, "bos_token_id")
        eos_id = token_id(self.bagel.new_token_ids, "eos_token_id")
        text_ids, text_indexes, labels, ce_indexes = [], [], [], []
        image_indexes, language_indexes, position_ids, masks, sample_lens = [], [], [], [], []
        cursor = 0
        prompt_truncations = 0
        for prompt, answer in zip(prompts, answers, strict=True):
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].to(raw_u.device)
            prompt_budget = self.max_prompt_tokens - 2
            if prompt_ids.numel() > prompt_budget:
                head = prompt_budget // 2
                prompt_ids = torch.cat([prompt_ids[:head], prompt_ids[-(prompt_budget - head):]])
                prompt_truncations += 1
            prompt_ids = torch.cat([prompt_ids.new_tensor([bos_id]), prompt_ids, prompt_ids.new_tensor([eos_id])])
            target_ids = self.tokenizer.encode(answer, return_tensors="pt")[0].to(raw_u.device)
            target_ids = torch.cat([target_ids, target_ids.new_tensor([eos_id])])
            assistant = torch.cat([prompt_ids, prompt_ids.new_tensor([bos_id]), target_ids[:-1]])
            image_len = TOKENS + 2
            seq_len = image_len + int(assistant.numel())
            assistant_index = cursor + image_len + int(prompt_ids.numel())
            local_text_indexes = torch.cat([
                torch.tensor([cursor, cursor + TOKENS + 1], dtype=torch.long, device=raw_u.device),
                torch.arange(cursor + image_len, cursor + seq_len, dtype=torch.long, device=raw_u.device),
            ])
            text_ids.append(torch.cat([prompt_ids.new_tensor([start_id, end_id]), assistant]))
            text_indexes.append(local_text_indexes)
            labels.append(target_ids)
            ce = torch.zeros(seq_len, dtype=torch.bool, device=raw_u.device)
            ce[assistant_index - cursor: assistant_index - cursor + target_ids.numel()] = True
            ce_indexes.append(ce)
            image_indexes.append(torch.arange(cursor + 1, cursor + 1 + TOKENS, device=raw_u.device))
            language_indexes.append(torch.arange(
                cursor + image_len, cursor + seq_len,
                dtype=torch.long, device=raw_u.device,
            ))
            position_ids.append(torch.cat([
                torch.zeros(image_len, dtype=torch.long, device=raw_u.device),
                torch.arange(1, 1 + assistant.numel(), dtype=torch.long, device=raw_u.device),
            ]))
            allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=BF16, device=raw_u.device))
            allowed[:image_len, :image_len] = 1
            masks.append((1.0 - allowed) * torch.finfo(BF16).min)
            sample_lens.append(seq_len)
            cursor += seq_len
        self.last_prompt_truncations = prompt_truncations

        packed_text_ids = torch.cat(text_ids)
        packed_text_indexes = torch.cat(text_indexes)
        packed_image_indexes = torch.cat(image_indexes)
        packed_text_embedding = self.bagel._embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((cursor, WIDTH))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        if mode not in {"real", "entry_only", "deep_only", "drop_g"}:
            raise ValueError(f"unknown injection mode: {mode}")
        fused_u = raw_u if self.native_lora_sft else self.bridge.begin(
            raw_u,
            raw_g,
            language_indexes,
            use_entry_g=mode in {"real", "entry_only"},
            use_deep_g=mode in {"real", "deep_only"},
            position_embedding=self.position_embedding,
            capture_pair_scores=capture_pair_scores,
        )
        # Lance Qwen-ViT outputs already include their native spatial encoding.
        visual_tokens = fused_u
        packed_sequence[packed_image_indexes] = visual_tokens.to(
            dtype=packed_sequence.dtype
        ).reshape(-1, WIDTH)
        packed_und_indexes = torch.cat([packed_text_indexes, packed_image_indexes])
        empty = packed_image_indexes.new_empty((0,))
        hidden = self.bagel._lm_forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=masks,
            packed_position_ids=torch.cat(position_ids),
            packed_und_token_indexes=packed_und_indexes,
            packed_gen_token_indexes=empty,
        )
        selected_hidden = hidden[torch.cat(ce_indexes)]
        flat_labels = torch.cat(labels)
        lm_head = self.bagel._get_base_language_model().lm_head
        if return_answer_logits and return_answer_hidden:
            raise ValueError("answer logits and hidden states cannot both be returned")
        answer_logits = None
        if return_answer_logits:
            if ce_chunk_size is not None:
                raise ValueError("answer logits are incompatible with chunked CE")
            answer_logits = lm_head(selected_hidden)
            token_ce = F.cross_entropy(answer_logits, flat_labels, reduction="none")
        else:
            token_ce = token_cross_entropy(lm_head, selected_hidden, flat_labels, ce_chunk_size)
        lengths = [int(target.numel()) for target in labels]
        if token_ce.numel() != sum(lengths):
            raise RuntimeError(f"CE length mismatch: {token_ce.numel()} != {sum(lengths)}")
        ce_sums = torch.stack([chunk.sum() for chunk in token_ce.split(lengths)])
        answer_lengths = torch.tensor(lengths, device=ce_sums.device, dtype=torch.float32)
        if answer_logits is not None:
            return ce_sums, answer_lengths, answer_logits
        if return_answer_hidden:
            return ce_sums, answer_lengths, selected_hidden
        return ce_sums, answer_lengths


def atomic_checkpoint(
    output: Path,
    step: int,
    model: Stage3CaptionModel | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    validation: dict[str, Any],
    rng_states: list[dict[str, Any]],
    stage: str,
    mask_u_probability: float,
    mask_u_schedule: tuple[tuple[float, ...], tuple[int, ...]] | None,
    flexae: nn.Module | None = None,
    native_lora_sft: bool = False,
) -> None:
    final = output / f"checkpoint-{step}"
    if (final / "complete.json").is_file():
        return
    temporary = output / f".checkpoint-{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in module.bridge.state_dict().items()},
        str(temporary / "bridge.safetensors"),
    )
    if stage in {"stage2", "stage3"}:
        save_file(understanding_lora_state_dict(module.bagel), str(temporary / "understanding_lora.safetensors"))
    if stage == "stage2":
        if flexae is None:
            raise RuntimeError("Stage 2 checkpoint requires the FlexAE state")
        flexae_module = flexae.module if isinstance(flexae, DistributedDataParallel) else flexae
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in flexae_module.state_dict().items()
            },
            str(temporary / "flexae.safetensors"),
        )
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "optimizer.pt",
    )
    for rng_rank, rng_state in enumerate(rng_states):
        torch.save(rng_state, temporary / f"rng_rank{rng_rank}.pt")
    (temporary / "complete.json").write_text(json.dumps({
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "stage": stage,
        "step": step,
        "training_mode": "native_lora_sft" if native_lora_sft else "gtoken_stage2",
        "native_lora_sft": native_lora_sft,
        "trainable_parameters": (
            sum(parameter.numel() for _, parameter in understanding_lora_named_parameters(module.bagel))
            if native_lora_sft else
            sum(p.numel() for p in module.bridge.parameters())
        ),
        "projector_parameters": sum(p.numel() for p in module.bridge.projector.parameters()),
        "entry_fuser_parameters": 0,
        "deep_sampler_parameters": sum(p.numel() for p in module.bridge.deep_samplers.parameters()),
        "sampler_layers": [] if native_lora_sft else list(SAMPLER_LAYERS),
        "model_width": WIDTH,
        "work_width": WORK_WIDTH,
        "attention_heads": 16,
        "g_tokens": TOKENS,
        "attention_scope": "global_256",
        "deep_query_source": "contextual_llm_language_hidden",
        "entry_fuser": False,
        "entry_query_source": None,
        "deep_post_attention_input": "attention_only",
        "projected_g_position_embedding": "fixed_bagel_vit_2d",
        "projector_init": "identity_from_stage1_raw_g",
        "lora": stage in {"stage2", "stage3"},
        "flexae_trainable": stage == "stage3",
        "flexae_parameters": 0 if flexae is None else sum(
            parameter.numel() for parameter in flexae.parameters()
        ),
        "understanding_lora_parameters": sum(p.numel() for _, p in understanding_lora_named_parameters(module.bagel)),
        "understanding_lora_tensors": len(understanding_lora_named_parameters(module.bagel)),
        "rng_ranks": len(rng_states),
        "position_embedding_additions": 2,
        "mask_u_probability": mask_u_probability,
        "mask_u_schedule": None if mask_u_schedule is None else {
            "probabilities": list(mask_u_schedule[0]),
            "boundaries": list(mask_u_schedule[1]),
        },
        "validation": validation,
    }, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, final)


def complete_checkpoint(path: Path, stage: str, world_size: int) -> tuple[int, dict[str, Any]] | None:
    required = ["bridge.safetensors", "optimizer.pt", "complete.json"]
    if stage in {"stage2", "stage3"}:
        required.append("understanding_lora.safetensors")
    if stage == "stage2":
        required.append("flexae.safetensors")
    required.extend(f"rng_rank{rank}.pt" for rank in range(world_size))
    if any(not (path / name).is_file() for name in required):
        return None
    try:
        step = int(path.name.rsplit("-", 1)[-1])
        marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        if (
            marker.get("schema") != CHECKPOINT_SCHEMA
            or marker.get("architecture_id") != ARCHITECTURE_ID
            or marker.get("step") != step
            or marker.get("stage") != stage
            or marker.get("rng_ranks") != world_size
        ):
            return None
        return step, marker
    except (ValueError, OSError, json.JSONDecodeError):
        return None


def retain_checkpoints(output: Path, stage: str, world_size: int, keep_latest: int) -> dict[str, Any]:
    complete = []
    for path in output.glob("checkpoint-*"):
        result = complete_checkpoint(path, stage, world_size)
        if result is not None:
            complete.append((result[0], path, result[1]))
    complete.sort()
    latest = {step for step, _, _ in complete[-keep_latest:]}
    scored = [
        (float(marker["validation"]["real_shuffle_gap"]), step)
        for step, _, marker in complete
        if isinstance(marker.get("validation"), dict) and "real_shuffle_gap" in marker["validation"]
    ]
    best_step = max(scored)[1] if scored else None
    keep = latest | ({best_step} if best_step is not None else set())
    removed = []
    for step, path, _ in complete:
        if step not in keep:
            shutil.rmtree(path)
            removed.append(step)
    decision = {"stage": stage, "best_gap_step": best_step, "latest_steps": sorted(latest), "kept_steps": sorted(keep), "removed_steps": removed}
    temporary = output / ".retention.json.tmp"
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "retention.json")
    return decision


def length_bin(length: int) -> str:
    for upper in (4, 16, 64, 128, 256, 512):
        if length <= upper:
            return f"le_{upper}"
    return "gt_512"


def summarize_gap_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("distributed validation produced no records")

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            "gap_token_mean": sum(row["gap_token_mean"] for row in rows) / len(rows),
            "gap_ce_sum": sum(row["gap_ce_sum"] for row in rows) / len(rows),
            "gap_sqrt_length": sum(row["gap_sqrt_length"] for row in rows) / len(rows),
            "positive_gap_fraction": sum(row["gap_token_mean"] > 0 for row in rows) / len(rows),
        }

    bins: dict[str, dict[str, dict[str, float | int]]] = {}
    keys = {
        "answer_length": lambda row: length_bin(int(row["answer_tokens"])),
        "visual_dependency": lambda row: str(row.get("visual_dependency", "na")),
        "image_correspondence": lambda row: str(row.get("image_correspondence", "na")),
        "relevance": lambda row: str(row.get("relevance", "na")),
    }
    for family, key_fn in keys.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            grouped.setdefault(key_fn(row), []).append(row)
        bins[family] = {key: summarize(rows) for key, rows in sorted(grouped.items())}
    return {"per_sample_summary": summarize(records), "bins": bins}


def validate_distributed(
    model: Stage3CaptionModel,
    manifest: InstructionManifest | FineVisionManifest,
    sample_count: int,
    extract_towers: Any,
    batch_size: int,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    if batch_size != 2 or sample_count % (world_size * batch_size):
        raise RuntimeError(
            "validation requires complete global batches with batch_size_per_rank=2"
        )
    plan_container: list[dict[str, Any] | None] = [None]
    if rank == 0:
        packed_lengths = []
        for index in range(sample_count):
            row = manifest.get(index)
            prompt_tokens = min(
                len(model.tokenizer.encode(row["prompt"])),
                model.max_prompt_tokens - 2,
            ) + 2
            answer_tokens = len(model.tokenizer.encode(row["answer"])) + 1
            packed_lengths.append(TOKENS + 2 + prompt_tokens + answer_tokens)
        sorted_indices = sorted(range(sample_count), key=lambda index: (packed_lengths[index], index))
        pairs = [
            (sorted_indices[offset], sorted_indices[-1 - offset])
            for offset in range(sample_count // 2)
        ]
        rank_indices = [[] for _ in range(world_size)]
        for pair_index, pair in enumerate(pairs):
            rank_indices[pair_index % world_size].extend(pair)
        plan_container[0] = {
            "rank_indices": rank_indices,
            "max_sample_packed_tokens": max(packed_lengths),
            "max_pair_packed_tokens": max(
                packed_lengths[first] + packed_lengths[second]
                for first, second in pairs
            ),
        }
    dist.broadcast_object_list(plan_container, src=0)
    plan = plan_container[0]
    assert plan is not None
    local_indices = plan["rank_indices"][rank]
    local_records: list[dict[str, Any]] = []
    real_entry_ratio_batches: list[float] = []
    real_deep_ratio_batches: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(local_indices), batch_size):
            indexes = local_indices[start:start + batch_size]
            rows = [manifest.get(index) for index in indexes]
            donor_indexes = [(index + 1) % sample_count for index in indexes]
            donor_rows = [manifest.get(index) for index in donor_indexes]
            raw_u, raw_g = extract_towers([image_source(row) for row in rows])
            _, donor_g = extract_towers([image_source(row) for row in donor_rows])
            prompts = [row["prompt"] for row in rows]
            answers = [row["answer"] for row in rows]
            masked_u = torch.ones(
                raw_u.shape[:2], dtype=torch.bool, device=raw_u.device,
            )
            outputs: dict[str, tuple[torch.Tensor, ...]] = {}
            for name, condition_g, condition_mask in (
                ("clean_real", raw_g, None),
                ("clean_shuffle", donor_g, None),
                ("clean_zero", torch.zeros_like(raw_g), None),
                ("masked_real", raw_g, masked_u),
                ("masked_shuffle", donor_g, masked_u),
                ("masked_zero", torch.zeros_like(raw_g), masked_u),
            ):
                with torch.autocast("cuda", dtype=BF16):
                    outputs[name] = model(
                        raw_u, condition_g, prompts, answers, "real",
                        ce_chunk_size=256,
                        fixed_u_mask=condition_mask,
                    )
                if name == "clean_real":
                    real_entry_ratio_batches.append(model.bridge.last_entry_ratio)
                    real_deep_ratio_batches.append(list(model.bridge.last_deep_ratios))
                model.bridge.clear_context()
            for name, condition_mask in (
                ("clean_drop", None),
                ("masked_drop", masked_u),
            ):
                with torch.autocast("cuda", dtype=BF16):
                    outputs[name] = model(
                        raw_u, raw_g, prompts, answers, "drop_g",
                        ce_chunk_size=256,
                        fixed_u_mask=condition_mask,
                        return_answer_hidden=True,
                    )
                model.bridge.clear_context()
            with disable_understanding_lora(model.bagel):
                for name, condition_mask in (
                    ("reference_clean_shuffle", None),
                    ("reference_masked_shuffle", masked_u),
                ):
                    with torch.autocast("cuda", dtype=BF16):
                        outputs[name] = model(
                            raw_u, donor_g, prompts, answers, "drop_g",
                            ce_chunk_size=256,
                            fixed_u_mask=condition_mask,
                            return_answer_hidden=True,
                        )
                    model.bridge.clear_context()
            outputs["reference_clean_real"] = outputs["reference_clean_shuffle"]
            outputs["reference_masked_real"] = outputs["reference_masked_shuffle"]
            lengths = outputs["clean_real"][1].detach().cpu().tolist()
            sums = {
                name: value[0].detach().float().cpu().tolist()
                for name, value in outputs.items()
            }
            int_lengths = [int(value) for value in lengths]
            lm_head = model.bagel._get_base_language_model().lm_head
            clean_drop_kl = [
                float(chunk.mean().detach())
                for chunk in forward_kl_from_hidden_per_token(
                    lm_head,
                    outputs["clean_drop"][2], outputs["reference_clean_shuffle"][2],
                    chunk_size=32,
                ).split(int_lengths)
            ]
            masked_drop_kl = [
                float(chunk.mean().detach())
                for chunk in forward_kl_from_hidden_per_token(
                    lm_head,
                    outputs["masked_drop"][2], outputs["reference_masked_shuffle"][2],
                    chunk_size=32,
                ).split(int_lengths)
            ]
            for offset, (index, donor_index, row, answer_tokens) in enumerate(
                zip(indexes, donor_indexes, rows, lengths, strict=True)
            ):
                length = int(answer_tokens)
                clean_real_sum = float(sums["clean_real"][offset])
                clean_shuffle_sum = float(sums["clean_shuffle"][offset])
                clean_zero_sum = float(sums["clean_zero"][offset])
                masked_real_sum = float(sums["masked_real"][offset])
                masked_shuffle_sum = float(sums["masked_shuffle"][offset])
                masked_zero_sum = float(sums["masked_zero"][offset])
                clean_drop_sum = float(sums["clean_drop"][offset])
                masked_drop_sum = float(sums["masked_drop"][offset])
                reference_clean_real_sum = float(sums["reference_clean_real"][offset])
                reference_clean_shuffle_sum = float(sums["reference_clean_shuffle"][offset])
                reference_masked_real_sum = float(sums["reference_masked_real"][offset])
                reference_masked_shuffle_sum = float(sums["reference_masked_shuffle"][offset])
                local_records.append({
                    "index": index,
                    "id": row["id"],
                    "donor_index": donor_index,
                    "answer_tokens": length,
                    "visual_dependency": row.get("visual_dependency"),
                    "image_correspondence": row.get("image_correspondence"),
                    "relevance": row.get("relevance"),
                    "real_ce": clean_real_sum / length,
                    "shuffle_ce": clean_shuffle_sum / length,
                    "zero_ce": clean_zero_sum / length,
                    "drop_ce": clean_drop_sum / length,
                    "drop_reference_kl": clean_drop_kl[offset],
                    "gap_token_mean": (clean_shuffle_sum - clean_real_sum) / length,
                    "support_token_mean": (clean_zero_sum - clean_real_sum) / length,
                    "rejection_token_mean": (clean_shuffle_sum - clean_zero_sum) / length,
                    "gap_ce_sum": clean_shuffle_sum - clean_real_sum,
                    "gap_sqrt_length": (clean_shuffle_sum - clean_real_sum) / math.sqrt(length),
                    "masked_real_ce": masked_real_sum / length,
                    "masked_shuffle_ce": masked_shuffle_sum / length,
                    "masked_zero_ce": masked_zero_sum / length,
                    "masked_drop_ce": masked_drop_sum / length,
                    "masked_drop_reference_kl": masked_drop_kl[offset],
                    "reference_real_ce": reference_clean_real_sum / length,
                    "reference_shuffle_ce": reference_clean_shuffle_sum / length,
                    "shuffle_ce_drift": (
                        clean_shuffle_sum - reference_clean_shuffle_sum
                    ) / length,
                    "reference_masked_real_ce": reference_masked_real_sum / length,
                    "reference_masked_shuffle_ce": reference_masked_shuffle_sum / length,
                    "masked_shuffle_ce_drift": (
                        masked_shuffle_sum - reference_masked_shuffle_sum
                    ) / length,
                    "masked_gap_token_mean": (
                        masked_shuffle_sum - masked_real_sum
                    ) / length,
                    "masked_support_token_mean": (
                        masked_zero_sum - masked_real_sum
                    ) / length,
                    "masked_rejection_token_mean": (
                        masked_shuffle_sum - masked_zero_sum
                    ) / length,
                })
    gathered: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    gathered_ratios: list[list[list[float]] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(real_deep_ratio_batches, gathered_ratios, dst=0)
    gathered_entry_ratios: list[list[float] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(real_entry_ratio_batches, gathered_entry_ratios, dst=0)
    if rank != 0:
        return None, None
    assert gathered is not None and gathered_ratios is not None and gathered_entry_ratios is not None
    records = sorted(
        [record for rank_records in gathered if rank_records for record in rank_records],
        key=lambda row: row["index"],
    )
    if len(records) != sample_count:
        raise RuntimeError(f"validation count mismatch: {len(records)} != {sample_count}")
    ratio_batches = [
        ratios
        for rank_batches in gathered_ratios
        if rank_batches is not None
        for ratios in rank_batches
    ]
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
        "global_batch_size": world_size * batch_size,
        "shuffle_contract": "fixed_next_sample_donor_g",
        "validation_order_contract": "packed_length_long_short_pairs_rank_balanced",
        "validation_ce_chunk_size": 256,
        "validation_kl_chunk_size": 32,
        "validation_drop_contract": "true_drop_g_with_current_lora_no_bridge",
        "validation_drop_kl_reference": "bridge_bypassed_and_lora_scale_zero_base_teacher",
        "validation_u_contract": "dual_clean_u_and_whole_sample_zero_u",
        "max_sample_packed_tokens": plan["max_sample_packed_tokens"],
        "max_pair_packed_tokens": plan["max_pair_packed_tokens"],
        "real_ce": sum(row["real_ce"] for row in records) / sample_count,
        "shuffle_ce": sum(row["shuffle_ce"] for row in records) / sample_count,
        "zero_ce": sum(row["zero_ce"] for row in records) / sample_count,
        "drop_ce": sum(row["drop_ce"] for row in records) / sample_count,
        "drop_reference_kl": sum(row["drop_reference_kl"] for row in records) / sample_count,
        "masked_real_ce": sum(row["masked_real_ce"] for row in records) / sample_count,
        "masked_shuffle_ce": sum(row["masked_shuffle_ce"] for row in records) / sample_count,
        "masked_zero_ce": sum(row["masked_zero_ce"] for row in records) / sample_count,
        "masked_drop_ce": sum(row["masked_drop_ce"] for row in records) / sample_count,
        "masked_drop_reference_kl": sum(row["masked_drop_reference_kl"] for row in records) / sample_count,
        "reference_real_ce": sum(row["reference_real_ce"] for row in records) / sample_count,
        "reference_shuffle_ce": sum(row["reference_shuffle_ce"] for row in records) / sample_count,
        "shuffle_ce_drift": sum(row["shuffle_ce_drift"] for row in records) / sample_count,
        "reference_masked_real_ce": sum(row["reference_masked_real_ce"] for row in records) / sample_count,
        "reference_masked_shuffle_ce": sum(row["reference_masked_shuffle_ce"] for row in records) / sample_count,
        "masked_shuffle_ce_drift": sum(row["masked_shuffle_ce_drift"] for row in records) / sample_count,
        "real_entry_residual_rms_ratio_batch_mean": sum(
            ratio for rank_ratios in gathered_entry_ratios if rank_ratios for ratio in rank_ratios
        ) / sum(len(rank_ratios) for rank_ratios in gathered_entry_ratios if rank_ratios),
        "real_deep_residual_rms_ratios_batch_mean": [
            sum(ratios[layer] for ratios in ratio_batches) / len(ratio_batches)
            for layer in range(len(SAMPLER_LAYERS))
        ],
    }
    result["real_shuffle_gap"] = result["shuffle_ce"] - result["real_ce"]
    result["reference_real_shuffle_gap"] = (
        result["reference_shuffle_ce"] - result["reference_real_ce"]
    )
    result["dpo_gap_improvement"] = (
        result["real_shuffle_gap"] - result["reference_real_shuffle_gap"]
    )
    result["real_zero_support"] = result["zero_ce"] - result["real_ce"]
    result["zero_shuffle_rejection"] = result["shuffle_ce"] - result["zero_ce"]
    result["masked_real_shuffle_gap"] = (
        result["masked_shuffle_ce"] - result["masked_real_ce"]
    )
    result["reference_masked_real_shuffle_gap"] = (
        result["reference_masked_shuffle_ce"] - result["reference_masked_real_ce"]
    )
    result["masked_dpo_gap_improvement"] = (
        result["masked_real_shuffle_gap"]
        - result["reference_masked_real_shuffle_gap"]
    )
    result["masked_real_zero_support"] = (
        result["masked_zero_ce"] - result["masked_real_ce"]
    )
    result["masked_zero_shuffle_rejection"] = (
        result["masked_shuffle_ce"] - result["masked_zero_ce"]
    )
    result.update(summarize_gap_records(records))
    return result, records


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device=device)


def parameter_update_audit(
    bridge: Stage3Bridge,
    initial_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    zero_update_tensors: list[str] = []
    for name, parameter in bridge.named_parameters():
        current = parameter.detach().cpu()
        initial = initial_state[name]
        if current.shape != initial.shape or current.dtype != initial.dtype:
            raise RuntimeError(f"initial-state mismatch for {name}")
        changed = current.ne(initial)
        delta = current.float() - initial.float()
        component = name.split(".", 1)[0]
        stats = components.setdefault(component, {
            "elements": 0, "changed_elements": 0, "delta_l2_sq": 0.0,
            "delta_abs_sum": 0.0, "delta_abs_max": 0.0,
        })
        stats["elements"] += current.numel()
        stats["changed_elements"] += int(changed.sum())
        stats["delta_l2_sq"] += float(delta.square().sum())
        stats["delta_abs_sum"] += float(delta.abs().sum())
        stats["delta_abs_max"] = max(stats["delta_abs_max"], float(delta.abs().max()))
        if not bool(changed.any()):
            zero_update_tensors.append(name)
    for stats in components.values():
        elements = stats["elements"]
        stats["changed_fraction"] = stats["changed_elements"] / elements
        stats["delta_rms"] = math.sqrt(stats.pop("delta_l2_sq") / elements)
        stats["delta_abs_mean"] = stats.pop("delta_abs_sum") / elements

    moment_dtypes: dict[str, int] = {}
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                dtype = str(value.dtype).replace("torch.", "")
                moment_dtypes[dtype] = moment_dtypes.get(dtype, 0) + 1
    parameter_dtypes: dict[str, int] = {}
    for parameter in bridge.parameters():
        dtype = str(parameter.dtype).replace("torch.", "")
        parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + 1
    return {
        "components": components,
        "zero_update_tensor_count": len(zero_update_tensors),
        "zero_update_tensors": zero_update_tensors,
        "parameter_tensor_dtypes": parameter_dtypes,
        "adam_moment_tensor_dtypes": moment_dtypes,
    }


def enable_layer_activation_checkpointing(bagel: nn.Module) -> int:
    layers = bagel._get_base_language_model().model.layers
    enabled = 0
    for layer in layers:
        if getattr(layer, "_gtoken_activation_checkpointing", False):
            continue
        original_forward = layer.forward

        def checkpointed_forward(*args, _forward=original_forward, **kwargs):
            if torch.is_grad_enabled():
                return torch_checkpoint(
                    _forward, *args, use_reentrant=False, **kwargs,
                )
            return _forward(*args, **kwargs)

        layer.forward = checkpointed_forward
        layer._gtoken_activation_checkpointing = True
        enabled += 1
    return enabled


def main() -> None:
    args = parse_args()
    stop_after_step = args.stop_after_step or args.max_steps
    if not 1 <= stop_after_step <= args.max_steps:
        raise ValueError("stop-after-step must lie within the training horizon")
    mask_u_schedule_horizon = args.mask_u_schedule_horizon or args.max_steps
    if args.max_steps > mask_u_schedule_horizon:
        raise ValueError("max_steps cannot exceed mask-U schedule horizon")
    mask_u_schedule = parse_mask_u_schedule(
        args.mask_u_schedule,
        args.mask_u_schedule_boundaries,
        mask_u_schedule_horizon,
    )
    shuffle_kl_schedule = parse_shuffle_kl_schedule(
        args.shuffle_kl_schedule_steps,
        args.shuffle_kl_schedule_weights,
        args.max_steps,
    )
    if mask_u_schedule is not None and args.mask_u_prob != mask_u_schedule[0][0]:
        raise ValueError(
            "mask-u-prob must equal the first scheduled probability: "
            f"{args.mask_u_prob} != {mask_u_schedule[0][0]}"
        )
    if args.keep_latest != 2:
        raise ValueError("this experiment requires keep-latest=2 for latest-2 plus best-gap retention")
    if args.stage == "stage2" and args.init_stage2_checkpoint:
        raise ValueError("Stage2 cannot use --init-stage2-checkpoint")
    if args.stage == "stage3" and not args.init_stage2_checkpoint:
        raise ValueError("Stage3 requires --init-stage2-checkpoint")
    if args.stage == "stage3" and args.init_bridge_checkpoint:
        raise ValueError("Stage3 initializes bridge and LoRA from --init-stage2-checkpoint")
    if args.native_lora_sft:
        if args.stage != "stage2":
            raise ValueError("native LoRA SFT is a Stage2-only baseline")
        disabled = {
            "mask_u_prob": args.mask_u_prob,
            "g_pair_loss_weight": args.g_pair_loss_weight,
            "answer_use_loss_weight": args.answer_use_loss_weight,
            "shuffle_kl_weight": args.shuffle_kl_weight,
            "drop_kl_weight": args.drop_kl_weight,
        }
        enabled = {name: value for name, value in disabled.items() if value != 0.0}
        if enabled or mask_u_schedule is not None or shuffle_kl_schedule is not None:
            raise ValueError(f"native LoRA SFT forbids G-token objectives and masking: {enabled}")
        if args.init_bridge_checkpoint or args.init_stage2_checkpoint:
            raise ValueError("native LoRA SFT starts from the frozen native Lance model")
    if args.g_pair_loss_weight < 0.0:
        raise ValueError("g-pair-loss-weight must be non-negative")
    if args.g_pair_temperature <= 0.0:
        raise ValueError("g-pair-temperature must be positive")
    if args.answer_use_loss_weight < 0.0:
        raise ValueError("answer-use loss weight must be non-negative")
    if args.answer_use_beta <= 0.0:
        raise ValueError("answer-use beta must be positive")
    if args.shuffle_kl_weight < 0.0:
        raise ValueError("shuffle-kl-weight must be non-negative")
    maximum_shuffle_kl_weight = max(
        (args.shuffle_kl_weight,)
        + (() if shuffle_kl_schedule is None else shuffle_kl_schedule[1])
    )
    if maximum_shuffle_kl_weight > 0.0 and args.answer_use_loss_weight <= 0.0:
        raise ValueError("shuffle KL requires reference-relative answer-use DPO")
    if args.drop_kl_weight < 0.0:
        raise ValueError("drop-kl-weight must be non-negative")
    if args.drop_kl_weight > 0.0 and args.answer_use_loss_weight <= 0.0:
        raise ValueError("drop KL requires the fixed-base answer-use teacher")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    for path in (args.flexae_root, args.lance_runtime):
        if path not in sys.path:
            sys.path.insert(0, path)

    from flexae_model import BagelFlexAE256
    from lance_backend import (
        FrozenWanEncoder,
        extract_qwen_u,
        fixed_g_position_embedding,
        flatten_wan_latents,
        load_lance_native,
        preprocess_qwen_vit,
    )

    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group("nccl", device_id=device)
        else:
            fd, group_file = tempfile.mkstemp(prefix="lance_stage23_bridge_")
            os.close(fd)
            os.unlink(group_file)
            dist.init_process_group("nccl", init_method=f"file://{group_file}", rank=0, world_size=1, device_id=device)

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest = load_manifest(args.manifest, args.image_root_from, args.image_root_to)
    val_manifest = load_manifest(args.val_manifest, args.image_root_from, args.image_root_to)
    if not isinstance(manifest, (InstructionManifest, FineVisionManifest)) or not isinstance(
        val_manifest, (InstructionManifest, FineVisionManifest)
    ):
        raise TypeError("FineVision fixed-base DPO requires embedded-image Parquet manifests")
    bagel = load_lance_native(args.lance_runtime, args.lance_model_root, device)
    tokenizer = bagel.tokenizer
    bagel.train()
    bagel.requires_grad_(False)
    activation_checkpoint_layers = 0
    if args.activation_checkpoint:
        base_lm = bagel._get_base_language_model()
        base_lm.gradient_checkpointing = True
        base_lm.config.use_cache = False
        activation_checkpoint_layers = enable_layer_activation_checkpointing(bagel)
    for module in bagel.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    wan = None if args.native_lora_sft else FrozenWanEncoder(
        args.lance_runtime,
        str(Path(args.lance_model_root) / "Wan2.2_VAE.pth"),
        device,
    )
    flexae = BagelFlexAE256(
        patch_dim=48,
        bagel_dim=WIDTH,
        gradient_checkpointing=True,
    ).to(device=device, dtype=torch.float32)
    flexae.load_state_dict(load_file(args.flexae_checkpoint), strict=True)
    flexae.requires_grad_(False)
    flexae.eval()
    encoder_parameters = [
        parameter
        for name in ("source_embed", "encoder", "g_projector")
        for parameter in getattr(flexae, name).parameters()
    ] + [flexae.encoder_query]
    encoder_trainable = False
    position_embedding = fixed_g_position_embedding(device)

    def preprocess_towers(
        sources: list[str | bytes | bytearray],
    ) -> tuple[list[Image.Image], list[torch.Tensor], torch.Tensor]:
        images = [load_pil(source) for source in sources]
        vit_patches, grids = preprocess_qwen_vit(images)
        return images, vit_patches, grids

    def extract_preprocessed(
        inputs: tuple[list[Image.Image], list[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, vit_patches, grids = inputs
        with torch.inference_mode(), torch.autocast("cuda", dtype=BF16):
            raw_u = extract_qwen_u(bagel, vit_patches, grids, device)
            if args.native_lora_sft:
                patches = raw_u.new_empty((raw_u.shape[0], 48, 32, 32))
            else:
                assert wan is not None
                latent = wan.encode(images)
                patches = flatten_wan_latents(latent)
        if args.tower_handoff_dtype == "bf16":
            return raw_u.to(dtype=BF16).clone(), patches.to(dtype=BF16).clone()
        return raw_u.float().clone(), patches.float().clone()

    def extract_towers(sources: list[str | bytes | bytearray]) -> tuple[torch.Tensor, torch.Tensor]:
        raw_u, patches = extract_preprocessed(preprocess_towers(sources))
        if args.native_lora_sft:
            return raw_u, raw_u.new_zeros((raw_u.shape[0], TOKENS, WIDTH))
        with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
            raw_g, _ = flexae.encode(patches)
        return raw_u, raw_g.detach()

    model = Stage3CaptionModel(
        bagel, tokenizer, position_embedding,
        max_prompt_tokens=args.max_prompt_tokens,
        mask_u_prob=args.mask_u_prob,
        mask_u_granularity=args.mask_u_granularity,
        pair_temperature=args.g_pair_temperature,
        native_lora_sft=args.native_lora_sft,
    ).to(device=device, dtype=MODEL_RUNTIME_DTYPE)
    model.bagel.requires_grad_(False)
    model.bridge.requires_grad_(not args.native_lora_sft)
    if args.stage == "stage2":
        if args.init_bridge_checkpoint:
            bridge_artifact = Path(args.init_bridge_checkpoint) / "bridge.safetensors"
            if not bridge_artifact.is_file():
                raise FileNotFoundError(f"missing bridge initializer: {bridge_artifact}")
            model.bridge.load_state_dict(load_file(bridge_artifact), strict=True)
    bridge_master_dtype = BF16 if args.bridge_master_dtype == "bf16" else torch.float32
    model.bridge.to(dtype=bridge_master_dtype)
    lora_master_dtype = BF16 if args.lora_master_dtype == "bf16" else torch.float32
    if args.stage in {"stage2", "stage3"}:
        lora_targets = install_understanding_lora(
            model.bagel, args.lora_rank, args.lora_alpha, lora_master_dtype,
        )
    else:
        lora_targets = ()
    lora_named = understanding_lora_named_parameters(model.bagel)
    lora_parameters = [parameter for _, parameter in lora_named]
    lora_ids = {id(parameter) for parameter in lora_parameters}
    expected_targets = len(model.bagel._get_base_language_model().model.layers) * len(UNDERSTANDING_LORA_PATHS)
    if len(lora_targets) != expected_targets or len(lora_named) != expected_targets * 2:
        raise RuntimeError(f"LoRA coverage mismatch: targets={len(lora_targets)} tensors={len(lora_named)}")

    if args.stage == "stage3":
        stage2_checkpoint = Path(args.init_stage2_checkpoint)
        stage2_bridge = stage2_checkpoint / "bridge.safetensors"
        stage2_lora = stage2_checkpoint / "understanding_lora.safetensors"
        stage2_marker = stage2_checkpoint / "complete.json"
        for artifact in (stage2_bridge, stage2_lora, stage2_marker):
            if not artifact.is_file():
                raise FileNotFoundError(f"missing Stage2 initializer artifact: {artifact}")
        marker = json.loads(stage2_marker.read_text(encoding="utf-8"))
        if marker.get("stage") != "stage2" or marker.get("architecture_id") != ARCHITECTURE_ID:
            raise RuntimeError(f"invalid Stage2 initializer marker: {marker}")
        if int(marker.get("step", -1)) != args.init_step:
            raise RuntimeError(
                f"Stage2 initializer step mismatch: {marker.get('step')} != {args.init_step}"
            )
        model.bridge.load_state_dict(load_file(stage2_bridge), strict=True)
        load_understanding_lora_state(model.bagel, load_file(stage2_lora))

    reference_path = output / "reference_trainables.safetensors"
    if rank == 0 and not reference_path.is_file():
        save_file(
            {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in trainable_named_parameters(model).items()
            },
            str(reference_path),
        )
    dist.barrier()
    reference_state_cpu = load_file(str(reference_path))
    current_named = trainable_named_parameters(model)
    if set(reference_state_cpu) != set(current_named):
        raise RuntimeError("frozen reference does not match current bridge/LoRA layout")
    with torch.no_grad():
        for name, parameter in current_named.items():
            parameter.copy_(reference_state_cpu[name].to(device=device, dtype=parameter.dtype))
    frozen_modules = (bagel,) + (() if wan is None else (wan.vae.model,)) + (
        (flexae,) if args.stage == "stage2" else ()
    )
    frozen_grad_count = sum(
        p.requires_grad for module in frozen_modules for p in module.parameters()
        if id(p) not in lora_ids
    )
    if frozen_grad_count:
        raise RuntimeError(f"freeze audit failed: {frozen_grad_count} frozen parameters require grad")
    parameter_groups = [] if args.native_lora_sft else bridge_parameter_groups(model.bridge, args)
    if lora_parameters:
        parameter_groups.append({"params": lora_parameters, "lr": args.lora_lr, "name": "understanding_lora"})
    extractor_parameters = encoder_parameters
    parameter_groups.append({
        "params": extractor_parameters,
        "lr": args.extractor_lr,
        "betas": (0.9, 0.95),
        "name": "flexae_encoder",
    })
    optimizer = torch.optim.AdamW([
        *parameter_groups,
    ], weight_decay=0.0, fused=args.fused_adamw)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    extractor_ids = {id(parameter) for parameter in extractor_parameters}
    bridge_trainable_ids = {
        id(parameter) for parameter in model.bridge.parameters() if parameter.requires_grad
    }
    trainable_ids = bridge_trainable_ids | lora_ids | extractor_ids
    if sum(len(group["params"]) for group in parameter_groups) != len(optimizer_ids):
        raise RuntimeError("duplicate bridge parameter registration across optimizer groups")
    if optimizer_ids != trainable_ids:
        raise RuntimeError(f"optimizer/trainable mismatch: optimizer={len(optimizer_ids)} expected={len(trainable_ids)}")

    schedule_max_steps = args.schedule_max_steps or args.max_steps
    if schedule_max_steps < args.max_steps:
        raise ValueError("schedule_max_steps cannot be less than max_steps")
    if args.resume_ramp_steps < 0:
        raise ValueError("resume_ramp_steps cannot be negative")
    if (args.resume_ramp_start_step is None) != (args.resume_ramp_steps == 0):
        raise ValueError("resume ramp requires both a start step and a positive length")

    def lr_lambda(step: int) -> float:
        return resumed_lr_factor(
            step + (args.init_step if args.stage == "stage3" else 0),
            args.warmup_steps,
            schedule_max_steps,
            args.min_lr_ratio,
            args.resume_ramp_start_step,
            args.resume_ramp_steps,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = args.init_step if args.stage == "stage3" else 0
    if start_step < 0 or start_step >= args.max_steps:
        raise ValueError("init-step must be in [0, max-steps) for Stage3 initialization")
    if args.resume:
        candidates: list[tuple[int, Path]] = []
        for checkpoint in output.glob("checkpoint-*"):
            result = complete_checkpoint(checkpoint, args.stage, world_size)
            if result is not None:
                candidates.append((result[0], checkpoint))
        if candidates:
            start_step, checkpoint = max(candidates)
            model.bridge.load_state_dict(load_file(checkpoint / "bridge.safetensors"), strict=True)
            if args.stage in {"stage2", "stage3"}:
                load_understanding_lora_state(model.bagel, load_file(checkpoint / "understanding_lora.safetensors"))
            if args.stage == "stage2":
                flexae.load_state_dict(load_file(checkpoint / "flexae.safetensors"), strict=True)
            state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng_state = torch.load(checkpoint / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
            random.setstate(rng_state["python"])
            torch.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device)
            if rank == 0:
                print(json.dumps({"event": "resume", "step": start_step, "checkpoint": str(checkpoint)}), flush=True)

    def enable_flexae_encoder() -> None:
        nonlocal encoder_trainable
        if encoder_trainable:
            return
        for name in ("source_embed", "encoder", "g_projector"):
            getattr(flexae, name).requires_grad_(True)
        flexae.encoder_query.requires_grad_(True)
        flexae.train()
        for name in ("local_head", "g_to_decoder", "decoder", "patch_head"):
            getattr(flexae, name).eval()
        encoder_trainable = True
        if rank == 0:
            print(json.dumps({"event": "flexae_encoder_unfrozen", "step": args.encoder_unfreeze_step}), flush=True)

    if start_step >= args.encoder_unfreeze_step:
        enable_flexae_encoder()

    balanced_indices: array | None = None
    balanced_metadata: dict[str, Any] | None = None
    if args.balanced_order:
        prefix = Path(args.balanced_order)
        binary = prefix.with_suffix(".bin")
        metadata_path = prefix.with_suffix(".json")
        balanced_metadata = json.loads(metadata_path.read_text())
        base_step = int(balanced_metadata["resume_step"])
        balanced_indices = array("I")
        with binary.open("rb") as handle:
            balanced_indices.fromfile(handle, int(balanced_metadata["order_entries"]))
        if rank == 0:
            print(json.dumps({
                "event": "launch_shuffle_loaded",
                "base_step": base_step,
                "start_step": start_step,
                "entries": len(balanced_indices),
            }), flush=True)

    trainable_parameters = (
        sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)
        + sum(p.numel() for p in lora_parameters)
        + sum(p.numel() for p in extractor_parameters)
    )
    if rank == 0:
        config = {
            **vars(args),
            "schema": RUN_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "world_size": world_size,
            "global_effective_batch": world_size * args.micro_batch_size * args.gradient_accumulation,
            "manifest_rows": len(manifest),
            "val_rows": len(val_manifest),
            "trainable_parameters": trainable_parameters,
            "training_mode": "native_lora_sft" if args.native_lora_sft else "gtoken_stage2",
            "sampler_layers": [] if args.native_lora_sft else list(SAMPLER_LAYERS),
            "model_width": WIDTH,
            "work_width": WORK_WIDTH,
            "attention_heads": 16,
            "g_tokens": TOKENS,
            "attention_scope": "global_256",
            "activation_checkpoint_layers": activation_checkpoint_layers,
            "component_parameters": {
                name: sum(p.numel() for p in getattr(model.bridge, name).parameters())
                for name in ("projector", "deep_samplers")
            },
            "flexae_sha256": sha256(args.flexae_checkpoint),
            "bagel_frozen": True,
            "vit_frozen": True,
            "flexae_frozen": args.stage == "stage2",
            "flexae_trainable_parameters": sum(p.numel() for p in extractor_parameters),
            "stage3_encoder_only": args.stage == "stage3",
            "raw_g_projected": not args.native_lora_sft,
            "projector_init": "identity_from_stage1_raw_g",
            "deep_query_source": "contextual_llm_language_hidden",
            "answer_use": {
                "enabled": args.answer_use_loss_weight > 0.0,
                "negative": "half_manifest_donor_g",
                "score": "mean_gold_answer_token_log_probability",
                "weight": args.answer_use_loss_weight,
                "beta": args.answer_use_beta,
                "reference": "frozen_initial_policy_exact_zero_delta_base_teacher",
                "llm_forwards_per_microbatch": (
                    4 if args.drop_kl_weight > 0.0 else (3 if args.answer_use_loss_weight > 0.0 else 1)
                ),
                "gradient": "full_real_and_shuffle_dpo_plus_shuffle_and_drop_forward_kl_anchors",
                "shuffle_kl_weight": args.shuffle_kl_weight,
                "shuffle_kl_schedule": (
                    [
                        {"step": step, "weight": weight}
                        for step, weight in zip(*shuffle_kl_schedule)
                    ]
                    if shuffle_kl_schedule is not None else None
                ),
                "shuffle_kl_direction": "KL(reference_shuffle || current_shuffle)",
                "drop_kl_weight": args.drop_kl_weight,
                "drop_kl_direction": "KL(reference_base || current_drop_g)",
                "drop_g_training_term": args.drop_kl_weight > 0.0,
                "reference_implementation": "bridge_bypassed_and_lora_scale_zero_only_for_teacher_forward",
                "reference_trainables_path": str(reference_path),
                "reference_trainables_sha256": sha256(str(reference_path)),
            },
            "deep_pairing": {
                "enabled": args.g_pair_loss_weight > 0.0,
                "negative": "half_manifest_donor_g",
                "score": "mean_smooth_bidirectional_query_key_alignment_across_10_layers",
                "weight": args.g_pair_loss_weight,
                "temperature": args.g_pair_temperature,
            },
            "entry_fuser": False,
            "entry_fuser_contract": None,
            "joint_lora_from_step_one": args.stage == "stage2" and args.init_bridge_checkpoint is None,
            "deep_post_attention_input": "attention_only",
            "projected_g_position_embedding": "fixed_bagel_vit_2d",
            "mask_u": {
                "enabled": args.mask_u_prob > 0.0,
                "probability": args.mask_u_prob,
                "schedule": None if mask_u_schedule is None else {
                    "probabilities": list(mask_u_schedule[0]),
                    "boundaries": list(mask_u_schedule[1]),
                    "horizon": mask_u_schedule_horizon,
                    "step_semantics": "optimizer_step_inclusive_upper_bound",
                },
                "granularity": args.mask_u_granularity,
                "train_only": False,
                "masked_value": "zero_raw_u_keep_position_embedding",
                "validation_masked": True,
                "validation_views": ["clean_u", "whole_sample_zero_u"],
            },
            "lora": args.stage in {"stage2", "stage3"},
            "native_lora_sft": args.native_lora_sft,
            "understanding_lora_parameters": sum(p.numel() for p in lora_parameters),
            "understanding_lora_tensors": len(lora_parameters),
            "bridge_master_dtype_actual": str(
                model.bridge.projector.calibrator.weight.dtype
            ).replace("torch.", ""),
            "lora_master_dtype_actual": (
                str(lora_parameters[0].dtype).replace("torch.", "") if lora_parameters else None
            ),
            "position_embedding_additions": 2,
            "deep_updated_positions": "prompt_and_teacher_forced_answer_language_tokens",
            "loss": {
                "formula": (
                    "mean answer-token CE"
                    if args.native_lora_sft else
                    "0.1 * matched_sft + dpo_weight * softplus(-beta * ((shuffle_ce-real_ce)-(ref_shuffle_ce-ref_real_ce))) + shuffle_kl_weight * KL(base||current_shuffle) + drop_kl_weight * KL(base||current_drop_g)"
                ),
                "lambda_CE": args.real_ce_loss_weight,
                "lambda_DPO": args.answer_use_loss_weight,
                "lambda_KL_base": args.shuffle_kl_weight,
                "metadata_weighting": False,
                "relevance_filter": ">=3 via audited FineVision balanced order",
                "answer_tokens_include_eos": True,
            },
            "validation_memory_contract": {
                "ce_chunk_size": 256,
                "drop_kl_logits": "answer_tokens_only_unchunked",
                "optimizer_state_device": "cpu_during_validation",
                "sample_order": "packed_length_long_short_pairs_rank_balanced",
            },
            "balanced_order_metadata": balanced_metadata,
            "schedule_max_steps": schedule_max_steps,
        }
        (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    initial_path = output / "bridge_initial.safetensors"
    if rank == 0 and not initial_path.is_file():
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.bridge.state_dict().items()},
            str(initial_path),
        )
    dist.barrier()
    initial_state = load_file(str(initial_path)) if rank == 0 else {}
    trainable = model

    class EncoderOnlyFlexAE(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            return self.base.encode(patches)[0]

    encoder_forward = EncoderOnlyFlexAE(flexae)
    trainable_flexae: nn.Module | None = encoder_forward

    manifest_size = len(manifest)
    mix_a = (2 * args.mix_seed + 1) % manifest_size or 1
    while math.gcd(mix_a, manifest_size) != 1:
        mix_a = (mix_a + 2) % manifest_size or 1
    mix_b = args.mix_seed % manifest_size
    final_val_limit = args.final_val_limit or args.val_limit
    if final_val_limit < args.val_limit:
        raise ValueError("final_val_limit cannot be less than val_limit")
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    timing = {"preprocess_wait": 0.0, "tower": 0.0, "model": 0.0}
    prompt_truncations = 0
    prefetch_pool = ThreadPoolExecutor(max_workers=args.prefetch_workers, thread_name_prefix="image-prefetch")
    progress_root_value = os.environ.get("GTOKEN_RANK_PROGRESS_DIR", "")
    progress_root = Path(progress_root_value) if progress_root_value else None
    if progress_root is not None:
        progress_root.mkdir(parents=True, exist_ok=True)

    def mark_rank_progress(event: str, step: int, micro: int | None = None) -> None:
        if progress_root is None:
            return
        payload = {
            "event": event,
            "step": step,
            "micro": micro,
            "rank": rank,
            "time": time.time(),
        }
        destination = progress_root / f"rank-{rank}.json"
        temporary = progress_root / f".rank-{rank}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    gradient_audit_steps = {
        int(value) for value in args.gradient_audit_steps.split(",") if value.strip()
    }
    gradient_audit_dir = Path(args.gradient_audit_dir) if args.gradient_audit_dir else None
    if gradient_audit_steps and gradient_audit_dir is None:
        raise ValueError("gradient_audit_dir is required when gradient_audit_steps is set")
    if rank == 0 and gradient_audit_dir is not None:
        gradient_audit_dir.mkdir(parents=True, exist_ok=True)
    telemetry = output / "train.jsonl"
    skipped_telemetry = output / "skipped_steps.jsonl"

    def estimate_packed_tokens(rows: list[dict[str, Any]]) -> int:
        total = 0
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], return_tensors="pt")[0]
            prompt_budget = args.max_prompt_tokens - 2
            if prompt_ids.numel() > prompt_budget:
                prompt_ids = prompt_ids[: prompt_budget // 2]
                tail = tokenizer.encode(row["prompt"], return_tensors="pt")[0]
                prompt_ids = torch.cat([prompt_ids, tail[-(prompt_budget - prompt_budget // 2):]])
            answer_ids = tokenizer.encode(row["answer"], return_tensors="pt")[0]
            # BOS/EOS around prompt, BOS before answer, EOS after answer.
            total += TOKENS + 2 + int(prompt_ids.numel()) + 2 + int(answer_ids.numel()) + 1
        return total

    for step in range(start_step + 1, stop_after_step + 1):
        if step >= args.encoder_unfreeze_step:
            enable_flexae_encoder()
        model.mask_u_prob = mask_u_probability_for_step(step, mask_u_schedule, args.mask_u_prob)
        effective_shuffle_kl_weight = (
            args.shuffle_kl_weight
            if shuffle_kl_schedule is None
            else piecewise_shuffle_kl_weight_for_step(step, shuffle_kl_schedule)
        )
        mark_rank_progress("step_start", step)
        ce_total = 0.0
        objective_total = 0.0
        metadata_weight_total = 0.0
        answer_tokens_total = 0.0
        selected_words = 0
        skipped_rows = 0
        mask_u_ratio_total = 0.0
        deep_pair_loss_total = 0.0
        real_deep_pair_score_total = 0.0
        shuffled_deep_pair_score_total = 0.0
        deep_pair_accuracy_total = 0.0
        answer_use_loss_total = 0.0
        shuffle_kl_loss_total = 0.0
        drop_kl_loss_total = 0.0
        reference_gap_total = 0.0
        train_shuffle_ce_total = 0.0
        train_gap_total = 0.0
        optimized_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        positive_grad_accum: list[torch.Tensor] | None = None
        negative_grad_accum: list[torch.Tensor] | None = None
        micro_rows = []
        micro_donor_rows = []
        for micro in range(args.gradient_accumulation):
            if balanced_indices is not None and balanced_metadata is not None:
                base_step = int(balanced_metadata["resume_step"])
                position = (
                    ((step - base_step - 1) * args.gradient_accumulation + micro)
                    * world_size * args.micro_batch_size + rank * args.micro_batch_size
                )
                indexes = balanced_indices[position:position + args.micro_batch_size]
                if len(indexes) != args.micro_batch_size:
                    raise RuntimeError(f"balanced-order exhausted at step={step} micro={micro} rank={rank}")
                rows = [manifest.get(index) for index in indexes]
                donor_rows = [
                    manifest.get((int(index) + manifest_size // 2) % manifest_size)
                    for index in indexes
                ] if (args.g_pair_loss_weight > 0.0 or args.answer_use_loss_weight > 0.0) else []
                skipped = 0
            else:
                serial = ((step - 1) * args.gradient_accumulation + micro) * world_size * args.micro_batch_size + rank * args.micro_batch_size
                rows, skipped = select_rows(
                    manifest, serial, mix_a, mix_b, args.micro_batch_size,
                    args.max_answer_words, args.max_microbatch_words,
                )
                donor_rows = [
                    manifest.get((mix_a * (serial + offset + 1) + mix_b) % len(manifest))
                    for offset in range(len(rows))
                ]
            selected_words += sum(row["caption_words"] for row in rows)
            skipped_rows += skipped
            micro_rows.append(rows)
            micro_donor_rows.append(donor_rows)
        mark_rank_progress("rows_selected", step)
        if args.max_packed_tokens is not None:
            local_lengths = torch.tensor(
                [estimate_packed_tokens(rows) for rows in micro_rows],
                dtype=torch.int64, device=device,
            )
            local_max = int(local_lengths.max().item())
            global_max_tensor = torch.tensor(local_max, dtype=torch.int64, device=device)
            mark_rank_progress("packed_preflight_before", step)
            dist.all_reduce(global_max_tensor, op=dist.ReduceOp.MAX)
            mark_rank_progress("packed_preflight_after", step)
            global_max = int(global_max_tensor.item())
            if global_max > args.max_packed_tokens:
                skip_record = {
                    "event": "optimizer_step_skipped",
                    "step": step,
                    "max_packed_tokens": args.max_packed_tokens,
                    "local_max_packed_tokens": local_max,
                    "global_max_packed_tokens": global_max,
                    "rank": rank,
                    "reason": "packed_length_exceeds_safety_threshold",
                }
                raise RuntimeError(json.dumps(skip_record, sort_keys=True))
        futures = [(
            prefetch_pool.submit(preprocess_towers, [image_source(row) for row in rows]),
            prefetch_pool.submit(preprocess_towers, [image_source(row) for row in donor_rows])
            if donor_rows else None,
        ) for rows, donor_rows in zip(micro_rows, micro_donor_rows, strict=True)]
        for micro, (rows, future_pair) in enumerate(zip(micro_rows, futures, strict=True)):
            clock = time.monotonic()
            mark_rank_progress("preprocess_wait", step, micro)
            preprocessed = future_pair[0].result(timeout=args.prefetch_timeout_seconds)
            donor_preprocessed = (
                future_pair[1].result(timeout=args.prefetch_timeout_seconds)
                if future_pair[1] is not None else None
            )
            mark_rank_progress("preprocess_ready", step, micro)
            timing["preprocess_wait"] += time.monotonic() - clock
            clock = time.monotonic()
            raw_u, patches = extract_preprocessed(preprocessed)
            donor_g = None
            if donor_preprocessed is not None:
                _, donor_patches = extract_preprocessed(donor_preprocessed)
                with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
                    donor_g, _ = flexae.encode(donor_patches)
                donor_g = donor_g.detach()
            if not encoder_trainable and args.stage == "stage2" and not args.native_lora_sft:
                with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
                    raw_g, _ = flexae.encode(patches)
            elif args.native_lora_sft:
                raw_g = raw_u.new_zeros((raw_u.shape[0], TOKENS, WIDTH))
            mark_rank_progress("tower_ready", step, micro)
            timing["tower"] += time.monotonic() - clock
            sync = trainable.no_sync() if isinstance(trainable, DistributedDataParallel) and micro + 1 < args.gradient_accumulation else nullcontext()
            extractor_sync = (
                trainable_flexae.no_sync()
                if isinstance(trainable_flexae, DistributedDataParallel)
                and micro + 1 < args.gradient_accumulation
                else nullcontext()
            )
            clock = time.monotonic()
            with sync, extractor_sync, torch.autocast(
                "cuda", dtype=TRAIN_AMP_DTYPE, enabled=TRAIN_AMP_ENABLED,
            ):
                if encoder_trainable and not args.native_lora_sft:
                    assert trainable_flexae is not None
                    raw_g = trainable_flexae(patches)
                if model.mask_u_granularity == "per_sample_per_vit_token":
                    fixed_u_mask = (
                        torch.rand(raw_u.shape[:2], device=raw_u.device)
                        < model.mask_u_prob
                    )
                else:
                    fixed_u_mask = (
                        torch.rand(raw_u.shape[0], 1, device=raw_u.device)
                        < model.mask_u_prob
                    ).expand(raw_u.shape[0], raw_u.shape[1])
                prompts = [row["prompt"] for row in rows]
                answers = [row["answer"] for row in rows]
                if args.answer_use_loss_weight > 0.0:
                    assert donor_g is not None
                    with disable_understanding_lora(model.bagel), torch.no_grad():
                        (
                            reference_shuffle_ce_sums,
                            reference_shuffle_lengths,
                            reference_shuffle_logits,
                        ) = model(
                            raw_u, donor_g, prompts, answers,
                            mode="drop_g",
                            fixed_u_mask=fixed_u_mask,
                            return_answer_logits=True,
                        )
                        model.bridge.clear_context()
                    reference_gap = raw_u.new_zeros(raw_u.shape[0])
                else:
                    reference_gap = raw_u.new_zeros(raw_u.shape[0])
                    reference_shuffle_logits = None

                if args.drop_kl_weight > 0.0:
                    assert reference_shuffle_logits is not None
                    model.bridge.clear_context()
                    (
                        _current_drop_ce_sums,
                        current_drop_lengths,
                        current_drop_logits,
                    ) = trainable(
                        raw_u, raw_g, prompts, answers,
                        mode="drop_g",
                        fixed_u_mask=fixed_u_mask,
                        return_answer_logits=True,
                    )
                    if not torch.equal(reference_shuffle_lengths, current_drop_lengths):
                        raise RuntimeError("base/current-drop answer lengths changed")
                    drop_kl_loss = forward_kl_loss(
                        current_drop_logits, reference_shuffle_logits,
                        chunk_size=args.train_kl_chunk_size,
                    )
                    model.bridge.clear_context()
                else:
                    drop_kl_loss = raw_u.new_zeros(())

                need_answer_logits = args.answer_use_loss_weight > 0.0 or args.drop_kl_weight > 0.0 or effective_shuffle_kl_weight > 0.0
                ce_sums, answer_lengths = trainable(
                    raw_u, raw_g,
                    prompts, answers,
                    capture_pair_scores=args.g_pair_loss_weight > 0.0,
                    fixed_u_mask=fixed_u_mask,
                    ce_chunk_size=None if need_answer_logits else args.train_kl_chunk_size,
                )
                real_deep_ratios = list(model.bridge.last_deep_ratios)
                weights = torch.tensor(
                    [metadata_weight(row) for row in rows],
                    device=ce_sums.device, dtype=ce_sums.dtype,
                )
                # Relevance filtering is part of dataset preparation, not launch-time validation.
                token_mean_ce = ce_sums / answer_lengths
                objective = (
                    token_mean_ce.mean()
                    if args.native_lora_sft else
                    args.real_ce_loss_weight * token_mean_ce.mean()
                )
                if args.g_pair_loss_weight > 0.0:
                    assert donor_g is not None
                    real_deep_pair_score, real_deep_by_layer = model.bridge.deep_pair_scores()
                    shuffled_deep_pair_score, shuffled_deep_by_layer = (
                        model.bridge.shuffled_deep_pair_scores(
                            donor_g.to(dtype=model.bridge.projector.calibrator.weight.dtype),
                            model.position_embedding,
                        )
                    )
                    deep_pair_loss = pairwise_logistic_loss(
                        real_deep_pair_score, shuffled_deep_pair_score,
                    )
                else:
                    deep_pair_loss = ce_sums.new_zeros(())
                    real_deep_pair_score = ce_sums.new_zeros(raw_u.shape[0])
                    shuffled_deep_pair_score = ce_sums.new_zeros(raw_u.shape[0])
                    real_deep_by_layer = ce_sums.new_zeros(len(SAMPLER_LAYERS), raw_u.shape[0])
                    shuffled_deep_by_layer = real_deep_by_layer
                if args.answer_use_loss_weight > 0.0:
                    assert donor_g is not None
                    model.bridge.clear_context()
                    shuffled_ce_sums, shuffled_lengths, shuffled_logits = trainable(
                        raw_u, donor_g,
                        prompts, answers,
                        capture_pair_scores=False,
                        fixed_u_mask=fixed_u_mask,
                        return_answer_logits=True,
                    )
                    if not torch.equal(answer_lengths, shuffled_lengths):
                        raise RuntimeError("real/shuffle answer lengths changed")
                    shuffled_token_mean_ce = shuffled_ce_sums / shuffled_lengths
                    shuffled_deep_ratios = list(model.bridge.last_deep_ratios)
                    dpo_signal = args.answer_use_beta * torch.sigmoid(
                        -args.answer_use_beta * (
                            shuffled_token_mean_ce - token_mean_ce - reference_gap
                        )
                    ).detach()
                    dpo_real_loss = (dpo_signal * token_mean_ce).mean()
                    dpo_negative_loss = -(dpo_signal * shuffled_token_mean_ce).mean()
                    answer_use_loss = (
                        args.dpo_real_weight * dpo_real_loss
                        + args.dpo_negative_weight * dpo_negative_loss
                    )
                    assert reference_shuffle_logits is not None
                    shuffle_kl_loss = forward_kl_loss(
                        shuffled_logits, reference_shuffle_logits,
                        chunk_size=args.train_kl_chunk_size,
                    )
                else:
                    shuffled_token_mean_ce = token_mean_ce.detach()
                    dpo_real_loss = ce_sums.new_zeros(())
                    dpo_negative_loss = ce_sums.new_zeros(())
                    answer_use_loss = ce_sums.new_zeros(())
                    shuffle_kl_loss = ce_sums.new_zeros(())
                    shuffled_deep_ratios = [0.0] * len(SAMPLER_LAYERS)
                loss = (
                    objective
                    + args.g_pair_loss_weight * deep_pair_loss
                    + args.answer_use_loss_weight * answer_use_loss
                    + effective_shuffle_kl_weight * shuffle_kl_loss
                    + args.drop_kl_weight * drop_kl_loss
                ) / args.gradient_accumulation
                positive_loss = (
                    objective
                    + args.g_pair_loss_weight * deep_pair_loss
                    + args.answer_use_loss_weight * args.dpo_real_weight * dpo_real_loss
                    + effective_shuffle_kl_weight * shuffle_kl_loss
                    + args.drop_kl_weight * drop_kl_loss
                ) / args.gradient_accumulation
                positive_values = synchronized_branch_gradients(
                    positive_loss, optimized_parameters, True, world_size,
                )
                if args.answer_use_loss_weight > 0.0:
                    negative_loss = (
                        args.answer_use_loss_weight * args.dpo_negative_weight * dpo_negative_loss
                    ) / args.gradient_accumulation
                    negative_values = synchronized_branch_gradients(
                        negative_loss, optimized_parameters, False, world_size,
                    )
                else:
                    negative_values = [torch.zeros_like(value) for value in positive_values]
                if positive_grad_accum is None:
                    positive_grad_accum = positive_values
                    negative_grad_accum = negative_values
                else:
                    for target, value in zip(positive_grad_accum, positive_values, strict=True):
                        target.add_(value)
                    assert negative_grad_accum is not None
                    for target, value in zip(negative_grad_accum, negative_values, strict=True):
                        target.add_(value)
                model.bridge.clear_context()
            timing["model"] += time.monotonic() - clock
            prompt_truncations += model.last_prompt_truncations
            mask_u_ratio_total += model.last_u_mask_ratio / args.gradient_accumulation
            ce_total += float(token_mean_ce.detach().mean()) / args.gradient_accumulation
            objective_total += float(objective.detach()) / args.gradient_accumulation
            deep_pair_loss_total += float(deep_pair_loss.detach()) / args.gradient_accumulation
            real_deep_pair_score_total += float(
                real_deep_pair_score.detach().mean()
            ) / args.gradient_accumulation
            shuffled_deep_pair_score_total += float(
                shuffled_deep_pair_score.detach().mean()
            ) / args.gradient_accumulation
            deep_pair_accuracy_total += float(
                (real_deep_pair_score.detach() > shuffled_deep_pair_score.detach()).float().mean()
            ) / args.gradient_accumulation
            answer_use_loss_total += float(answer_use_loss.detach()) / args.gradient_accumulation
            shuffle_kl_loss_total += float(shuffle_kl_loss.detach()) / args.gradient_accumulation
            drop_kl_loss_total += float(drop_kl_loss.detach()) / args.gradient_accumulation
            reference_gap_total += float(reference_gap.detach().mean()) / args.gradient_accumulation
            train_shuffle_ce_total += float(
                shuffled_token_mean_ce.detach().mean()
            ) / args.gradient_accumulation
            train_gap_total += float(
                (shuffled_token_mean_ce.detach() - token_mean_ce.detach()).mean()
            ) / args.gradient_accumulation
            metadata_weight_total += float(weights.detach().mean()) / args.gradient_accumulation
            answer_tokens_total += float(answer_lengths.detach().mean()) / args.gradient_accumulation

        assert positive_grad_accum is not None and negative_grad_accum is not None
        cap_scale = min(
            1.0,
            args.dpo_negative_grad_cap * global_gradient_norm(positive_grad_accum)
            / max(global_gradient_norm(negative_grad_accum), 1e-12),
        )
        for parameter, positive, negative in zip(
            optimized_parameters, positive_grad_accum, negative_grad_accum, strict=True,
        ):
            parameter.grad = (positive + cap_scale * negative).to(
                device=parameter.device, dtype=parameter.dtype,
            )

        frozen_bagel_gradient_count = sum(
            parameter.grad is not None for parameter in model.bagel.parameters() if id(parameter) not in lora_ids
        )
        lora_gradient_count = sum(parameter.grad is not None for parameter in lora_parameters)
        extractor_gradient_count = sum(
            parameter.grad is not None for parameter in extractor_parameters
        )
        if frozen_bagel_gradient_count:
            raise RuntimeError(f"frozen Lance received {frozen_bagel_gradient_count} parameter gradients")
        if lora_gradient_count != len(lora_parameters):
            raise RuntimeError(f"LoRA gradient coverage mismatch: {lora_gradient_count}/{len(lora_parameters)}")
        if encoder_trainable and extractor_gradient_count == 0:
            raise RuntimeError(
                "FlexAE encoder received no gradient after the paper unfreeze step"
            )

        def component_grad_norm(component: nn.Module) -> float:
            terms = [p.grad.detach().float().square().sum() for p in component.parameters() if p.grad is not None]
            return float(torch.sqrt(torch.stack(terms).sum())) if terms else 0.0

        component_grad_norms = {
            "projector": component_grad_norm(model.bridge.projector),
            "deep_samplers": component_grad_norm(model.bridge.deep_samplers),
        }
        if lora_parameters:
            terms = [p.grad.detach().float().square().sum() for p in lora_parameters if p.grad is not None]
            component_grad_norms["understanding_lora"] = float(torch.sqrt(torch.stack(terms).sum()))
        if extractor_parameters:
            terms = [
                parameter.grad.detach().float().square().sum()
                for parameter in extractor_parameters if parameter.grad is not None
            ]
            component_grad_norms["flexae_encoder"] = float(torch.sqrt(torch.stack(terms).sum()))
        sampler_up_grads = [sampler.up.weight.grad for sampler in model.bridge.deep_samplers]
        if not args.native_lora_sft and any(
            value is None or float(value.detach().float().norm()) <= 0.0
            for value in sampler_up_grads
        ):
            sampler_up_grad_norms = [
                None if value is None else float(value.detach().float().norm())
                for value in sampler_up_grads
            ]
            raise RuntimeError(
                "one or more deep sampler zero-init outputs did not receive gradient: "
                f"{sampler_up_grad_norms}"
            )
        if not args.native_lora_sft and step >= 3 and any(
            component_grad_norms[name] <= 0.0
            for name in ("projector", "deep_samplers")
        ):
            raise RuntimeError(f"bridge gradient chain still locked at step {step}: {component_grad_norms}")
        if rank == 0 and step in gradient_audit_steps and gradient_audit_dir is not None:
            gradient_path = gradient_audit_dir / f"gradients-step-{step}.safetensors"
            save_file({
                name: parameter.grad.detach().cpu().contiguous()
                for name, parameter in model.bridge.named_parameters()
                if parameter.grad is not None
            }, str(gradient_path))
            audit_record = {
                "event": "gradient_audit_saved",
                "step": step,
                "path": str(gradient_path),
                "sha256": sha256(str(gradient_path)),
                "component_grad_norms": component_grad_norms,
                "frozen_bagel_gradient_count": frozen_bagel_gradient_count,
                "understanding_lora_gradient_count": lora_gradient_count,
                "flexae_encoder_gradient_count": extractor_gradient_count,
            }
            atomic_jsonl(gradient_audit_dir / "audit.jsonl", audit_record)
            print(json.dumps(audit_record, sort_keys=True), flush=True)
        optimized_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(optimized_parameters, args.max_grad_norm))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        mark_rank_progress("optimizer_done", step)
        values = torch.tensor([
            ce_total, objective_total, grad_norm,
            metadata_weight_total, answer_tokens_total, mask_u_ratio_total,
            deep_pair_loss_total, real_deep_pair_score_total,
            shuffled_deep_pair_score_total, deep_pair_accuracy_total,
            answer_use_loss_total, train_shuffle_ce_total, train_gap_total,
            shuffle_kl_loss_total, reference_gap_total, drop_kl_loss_total,
        ], dtype=torch.float64, device=device)
        dist.all_reduce(values)
        values.div_(world_size)
        if rank == 0 and (step == 1 or step % args.logging_steps == 0):
            record = {
                "event": "train", "step": step,
                "caption_ce": float(values[0]),
                "weighted_sqrt_length_objective": float(values[1]),
                "grad_norm": float(values[2]),
                "metadata_weight_mean": float(values[3]),
                "answer_tokens_mean": float(values[4]),
                "mask_u_ratio": float(values[5]),
                "mask_u_probability_requested": model.mask_u_prob,
                "deep_pair_loss": float(values[6]),
                "real_deep_pair_score": float(values[7]),
                "shuffled_deep_pair_score": float(values[8]),
                "deep_pair_accuracy": float(values[9]),
                "real_deep_pair_scores_by_layer": [
                    float(value) for value in real_deep_by_layer.detach().mean(dim=1)
                ],
                "shuffled_deep_pair_scores_by_layer": [
                    float(value) for value in shuffled_deep_by_layer.detach().mean(dim=1)
                ],
                "answer_use_loss": float(values[10]),
                "shuffle_caption_ce": float(values[11]),
                "train_real_shuffle_gap": float(values[12]),
                "shuffle_reference_kl": float(values[13]),
                "effective_shuffle_kl_weight": effective_shuffle_kl_weight,
                "reference_real_shuffle_gap": float(values[14]),
                "dpo_gap_improvement": float(values[12] - values[14]),
                "drop_reference_kl": float(values[15]),
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "component_grad_norms": component_grad_norms,
                "entry_residual_rms_ratio": model.bridge.last_entry_ratio,
                "deep_residual_rms_ratios": real_deep_ratios,
                "shuffle_deep_residual_rms_ratios": shuffled_deep_ratios,
                "frozen_bagel_gradient_count": frozen_bagel_gradient_count,
                "understanding_lora_gradient_count": lora_gradient_count,
                "flexae_encoder_gradient_count": extractor_gradient_count,
                "prompt_truncations": prompt_truncations,
                "timing_seconds": timing,
                "selected_caption_words": selected_words, "selection_skips": skipped_rows,
                "seconds": time.monotonic() - start_time,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            atomic_jsonl(telemetry, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            start_time = time.monotonic()
            timing = {"preprocess_wait": 0.0, "tower": 0.0, "model": 0.0}
            prompt_truncations = 0

        if step % args.save_steps == 0 or step == stop_after_step:
            dist.barrier()
            move_optimizer_state(optimizer, "cpu")
            torch.cuda.empty_cache()
            current_val_count = min(
                final_val_limit if step == args.max_steps else args.val_limit,
                len(val_manifest),
            )
            validation, validation_records = validate_distributed(
                model, val_manifest, current_val_count, extract_towers,
                args.val_batch_size, rank, world_size,
            )
            if rank == 0:
                assert validation is not None and validation_records is not None
                validation = {"event": "validation", "step": step, **validation}
                validation["update_audit"] = (
                    {"mode": "native_lora_sft", "bridge_trainable": False}
                    if args.native_lora_sft else
                    parameter_update_audit(model.bridge, initial_state, optimizer)
                )
                samples_path = output / f"validation-step-{step}-samples.jsonl"
                temporary_samples = samples_path.with_suffix(".jsonl.tmp")
                temporary_samples.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation_records),
                    encoding="utf-8",
                )
                os.replace(temporary_samples, samples_path)
                validation["per_sample_path"] = str(samples_path)
                validation["per_sample_sha256"] = sha256(str(samples_path))
                atomic_jsonl(output / "validation.jsonl", validation)
                print(json.dumps(validation, sort_keys=True), flush=True)
            local_rng = {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(device),
            }
            gathered_rng: list[dict[str, Any]] | None = [None] * world_size if rank == 0 else None
            dist.gather_object(local_rng, gathered_rng, dst=0)
            if rank == 0:
                assert gathered_rng is not None
                atomic_checkpoint(
                    output, step, trainable, optimizer, scheduler, validation,
                    gathered_rng, args.stage, model.mask_u_prob, mask_u_schedule,
                    flexae=flexae,
                    native_lora_sft=args.native_lora_sft,
                )
                decision = retain_checkpoints(output, args.stage, world_size, args.keep_latest)
                print(json.dumps({"event": "checkpoint_retention", **decision}, sort_keys=True), flush=True)
            dist.barrier()
            move_optimizer_state(optimizer, device)
            torch.cuda.empty_cache()

    if rank == 0 and stop_after_step == args.max_steps:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.bridge.state_dict().items()},
            str(output / "bridge_final.safetensors"),
        )
        if args.stage in {"stage2", "stage3"}:
            save_file(understanding_lora_state_dict(model.bagel), str(output / "understanding_lora_final.safetensors"))
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in flexae.state_dict().items()},
            str(output / "flexae_final.safetensors"),
        )
        (output / "complete.json").write_text(json.dumps({
            "schema": RUN_SCHEMA, "stage": args.stage, "step": args.max_steps,
            "architecture_id": ARCHITECTURE_ID,
            "training_mode": "native_lora_sft" if args.native_lora_sft else "gtoken_stage2",
            "native_lora_sft": args.native_lora_sft,
            "trainable_parameters": trainable_parameters,
            "mask_u_schedule": None if mask_u_schedule is None else {
                "probabilities": list(mask_u_schedule[0]),
                "boundaries": list(mask_u_schedule[1]),
                "horizon": mask_u_schedule_horizon,
            },
            "shuffle_kl_schedule": None if shuffle_kl_schedule is None else {
                "steps": list(shuffle_kl_schedule[0]),
                "weights": list(shuffle_kl_schedule[1]),
            },
        }, indent=2) + "\n")
    prefetch_pool.shutdown(wait=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
