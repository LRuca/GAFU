#!/usr/bin/env python3
"""Janus-Pro full-layer G-memory injection with answer-only causal training."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from array import array
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn


WIDTH, WORK_WIDTH, G_TOKENS = 2048, 1024, 256
LAYERS = tuple(range(24))
ARCHITECTURE_ID = "gafu_paper_main_janus_pro_v1"
CHECKPOINT_SCHEMA = "gafu_paper_main_janus_pro_checkpoint_v1"
RUN_SCHEMA = "gafu_paper_main_janus_pro_run_v1"
TRAINING_MODE = "ours"
RATING_FIELDS = ("visual_dependency_ratings", "image_correspondence_ratings", "relevance_ratings")


class FineVisionManifest:
    def __init__(self, manifest_path: str) -> None:
        self.path = Path(manifest_path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.files: list[tuple[Path, int]] = []
        for item in payload.get("files", []):
            path = self.path.parent / item["path"]
            if not path.exists():
                path = self.path.parent.parent / item["path"]
            if path.exists() and "text_only" not in path.parts:
                self.files.append((path, int(item["rows"])))
        if not self.files:
            raise RuntimeError("FineVision manifest has no usable shards")
        self.ends, total = [], 0
        for _path, rows in self.files:
            total += rows; self.ends.append(total)
        self._file_index = self._group_index = None
        self._parquet = self._table = None

    def __len__(self) -> int:
        return self.ends[-1]

    def get(self, index: int) -> dict[str, Any]:
        index %= len(self)
        file_index = bisect_right(self.ends, index)
        local = index - (0 if file_index == 0 else self.ends[file_index - 1])
        if file_index != self._file_index:
            self._file_index, self._group_index = file_index, None
            self._parquet = pq.ParquetFile(self.files[file_index][0])
        start = 0
        for group in range(self._parquet.num_row_groups):
            rows = self._parquet.metadata.row_group(group).num_rows
            if local < start + rows:
                if group != self._group_index:
                    self._group_index = group; self._table = self._parquet.read_row_group(group)
                row = {name: self._table[name][local - start].as_py() for name in self._table.column_names}
                break
            start += rows
        else:
            raise IndexError(index)
        texts, images = row.get("texts") or [], row.get("images") or []
        ratings = {field: row.get(field) for field in RATING_FIELDS}
        if any(value is None or len(value) != len(texts) for value in ratings.values()):
            raise ValueError(f"rating alignment mismatch at {index}")
        selected = None
        for turn_index, turn in enumerate(texts):
            prompt = str((turn or {}).get("user", "")).replace("<image>", "").strip()
            answer = str((turn or {}).get("assistant", "")).strip()
            if prompt and answer:
                selected = (prompt, answer, turn_index)
        image_bytes = (images[0] or {}).get("bytes") if images else None
        if selected is None or not image_bytes:
            raise ValueError(f"invalid FineVision row {index}")
        prompt, answer, turn = selected
        return {"index": index, "image_bytes": image_bytes, "prompt": prompt, "answer": answer,
                "visual_dependency": int(ratings["visual_dependency_ratings"][turn]),
                "image_correspondence": int(ratings["image_correspondence_ratings"][turn]),
                "relevance": int(ratings["relevance_ratings"][turn])}


def load_image(value: bytes) -> Image.Image:
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--balanced-order", required=True); parser.add_argument("--flexae-checkpoint", required=True)
    parser.add_argument("--janus-code", required=True); parser.add_argument("--janus-model", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--encoder-unfreeze-step", type=int, default=2501)
    parser.add_argument("--extractor-lr", type=float, default=1e-4)
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--schedule-max-steps", type=int, default=5000)
    parser.add_argument("--micro-batch-size", type=int, default=4); parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--projector-lr", type=float, default=2e-5); parser.add_argument("--sampler-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4); parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0); parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1); parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10); parser.add_argument("--final-val-limit", type=int, default=512)
    parser.add_argument("--mask-u-schedule", default="0.8,0.5,0.2,0.0")
    parser.add_argument("--mask-u-boundaries", default="1250,2500,3750,5000")
    parser.add_argument("--shuffle-kl-steps", default="1,1875,2188,2500,5000")
    parser.add_argument("--shuffle-kl-weights", default="1,1,0.25,0.25,0")
    parser.add_argument("--answer-use-weight", type=float, default=1.0); parser.add_argument("--answer-use-beta", type=float, default=0.1)
    parser.add_argument("--real-ce-loss-weight", type=float, default=5.0)
    parser.add_argument("--dpo-real-weight", type=float, default=4.0); parser.add_argument("--dpo-negative-weight", type=float, default=1.0)
    parser.add_argument("--dpo-negative-grad-cap", type=float, default=0.25); parser.add_argument("--drop-kl-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260825); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonformal-smoke", action="store_true")
    parser.add_argument("--training-mode", choices=("ours", "sft"), default="ours")
    return parser.parse_args()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__(); self.base = base.requires_grad_(False); self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + F.linear(F.linear(values, self.lora_A), self.lora_B) * self.scale


@contextmanager
def disable_lora(janus: nn.Module):
    modules = [module for module in janus.modules() if isinstance(module, LoRALinear)]
    scales = [module.scale for module in modules]
    for module in modules: module.scale = 0.0
    try: yield
    finally:
        for module, scale in zip(modules, scales, strict=True): module.scale = scale


def install_lora(janus: nn.Module, rank: int, alpha: float) -> None:
    for layer in janus.language_model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"unexpected pre-LoRA module: {type(module)}")
            setattr(layer.self_attn, name, LoRALinear(module, rank, alpha).to(module.weight.device))


def lora_state(janus: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in janus.state_dict().items() if ".lora_A" in name or ".lora_B" in name}


def load_lora_state(janus: nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(janus.named_parameters())
    if set(state) != {name for name in named if ".lora_A" in name or ".lora_B" in name}:
        raise RuntimeError("LoRA checkpoint key contract mismatch")
    with torch.no_grad():
        for name, value in state.items(): named[name].copy_(value.to(named[name]))


class GProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.RMSNorm(1024), nn.Linear(1024, WIDTH), nn.GELU(approximate="tanh"), nn.Linear(WIDTH, WIDTH), nn.RMSNorm(WIDTH))
    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.net(values)


class TextHiddenGSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.h_norm = nn.RMSNorm(WIDTH); self.g_norm = nn.RMSNorm(WIDTH)
        self.q = nn.Linear(WIDTH, WORK_WIDTH, bias=False); self.k = nn.Linear(WIDTH, WORK_WIDTH, bias=False)
        self.v = nn.Linear(WIDTH, WORK_WIDTH, bias=False); self.o = nn.Linear(WORK_WIDTH, WIDTH, bias=False)
        nn.init.zeros_(self.o.weight)

    def forward(self, hidden: torch.Tensor, g: torch.Tensor, language_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        q = self.q(self.h_norm(hidden)).reshape(batch, length, 16, 64).transpose(1, 2)
        normalized_g = self.g_norm(g)
        k = self.k(normalized_g).reshape(batch, G_TOKENS, 16, 64).transpose(1, 2)
        v = self.v(normalized_g).reshape(batch, G_TOKENS, 16, 64).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, length, WORK_WIDTH)
        return hidden + self.o(mixed) * language_mask.unsqueeze(-1).to(hidden.dtype)


class Stage2Bridge(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.projector = GProjector(); self.samplers = nn.ModuleList(TextHiddenGSampler() for _ in LAYERS)
        self.context: tuple[torch.Tensor, torch.Tensor] | None = None

    def begin(self, raw_g: torch.Tensor, position: torch.Tensor, language_mask: torch.Tensor) -> None:
        self.context = (self.projector(raw_g) + position.to(raw_g.device, raw_g.dtype), language_mask)

    def clear(self) -> None: self.context = None

    def hook(self, index: int, output: Any) -> Any:
        if self.context is None: return output
        hidden = output[0] if isinstance(output, tuple) else output
        g, mask = self.context; updated = self.samplers[index](hidden, g, mask)
        return (updated, *output[1:]) if isinstance(output, tuple) else updated


class JanusStage2(nn.Module):
    def __init__(self, janus: nn.Module, flexae: nn.Module, position: torch.Tensor) -> None:
        super().__init__(); self.janus = janus; self.flexae = flexae; self.bridge = Stage2Bridge(); self.register_buffer("position", position, persistent=True)
        self._handles = [layer.register_forward_hook(lambda _m, _a, output, i=i: self.bridge.hook(i, output)) for i, layer in enumerate(janus.language_model.model.layers)]

    def forward(self, u: torch.Tensor, raw_g: torch.Tensor, language: Any, mode: str, fixed_u_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from janus_backend import prepare_inputs_embeds
        masked_u = u.masked_fill(fixed_u_mask.unsqueeze(-1), 0)
        inputs = prepare_inputs_embeds(self.janus, language, masked_u)
        language_mask = language.attention_mask.bool() & ~language.images_seq_mask
        if mode == "real": self.bridge.begin(raw_g, self.position, language_mask)
        elif mode != "drop_g": raise ValueError(mode)
        hidden = self.janus.language_model.model(inputs_embeds=inputs, attention_mask=language.attention_mask, use_cache=False, return_dict=True).last_hidden_state
        self.bridge.clear()
        selected = hidden[language.answer_predictor_mask]
        logits = self.janus.language_model.lm_head(selected)
        token_ce = F.cross_entropy(logits.float(), language.answer_labels, reduction="none")
        sums = torch.zeros(u.shape[0], device=u.device).scatter_add_(0, language.answer_sample_ids, token_ce)
        lengths = torch.bincount(language.answer_sample_ids, minlength=u.shape[0]).to(sums.dtype)
        return sums, lengths, logits


def forward_kl(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    reference_log = F.log_softmax(reference.detach().float(), -1); current_log = F.log_softmax(current.float(), -1)
    return (reference_log.exp() * (reference_log - current_log)).sum(-1).mean()


def schedule_value(step: int, points: tuple[int, ...], values: tuple[float, ...]) -> float:
    right = bisect_left(points, step)
    if right < len(points) and points[right] == step: return values[right]
    left = right - 1; alpha = (step - points[left]) / (points[right] - points[left])
    return values[left] + alpha * (values[right] - values[left])


def metadata_weight(row: dict[str, Any]) -> float:
    visual = (row["visual_dependency"] - 1) / 4
    gate = 0.0 if row["image_correspondence"] <= 2 else (0.5 if row["image_correspondence"] == 3 else 1.0)
    return 1.0 + 0.5 * visual * gate


def load_order(prefix: str) -> tuple[array, dict[str, Any]]:
    binary = Path(prefix + ".bin"); values = array("I")
    with binary.open("rb") as handle: values.fromfile(handle, binary.stat().st_size // values.itemsize)
    return values, json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))


def lr_factor(step: int, warmup: int, maximum: int, floor: float) -> float:
    if step <= warmup: return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def trainable_parameters(model: JanusStage2) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def enable_flexae_encoder(flexae: nn.Module) -> list[nn.Parameter]:
    """Enable only the Stage 1 Encoder-side tensors required by the paper."""
    flexae.requires_grad_(False)
    for name in ("source_embed", "encoder", "g_projector"):
        getattr(flexae, name).requires_grad_(True)
    flexae.encoder_query.requires_grad_(True)
    flexae.train()
    for name in ("local_head", "g_to_decoder", "decoder", "patch_head"):
        getattr(flexae, name).eval()
    return [parameter for parameter in flexae.parameters() if parameter.requires_grad]


def synchronized_branch_gradients(loss: torch.Tensor, parameters: list[nn.Parameter], retain_graph: bool, world_size: int) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=False)
    values = [gradient.detach().float() for gradient in gradients]
    if world_size == 1: return values
    maximum = (128 * 1024 * 1024) // 4
    start = 0
    while start < len(values):
        end, count = start, 0
        while end < len(values) and (count == 0 or count + values[end].numel() <= maximum):
            count += values[end].numel(); end += 1
        flat = torch.cat([value.reshape(-1) for value in values[start:end]])
        dist.all_reduce(flat); flat.div_(world_size)
        offset = 0
        for index in range(start, end):
            size = values[index].numel(); values[index] = flat[offset:offset+size].view_as(values[index]); offset += size
        start = end
    return values


def global_norm(values: list[torch.Tensor]) -> float:
    return math.sqrt(math.fsum(float(value.norm()) ** 2 for value in values))


def latest_complete(output: Path, world_size: int) -> tuple[int, Path] | None:
    found = []
    for path in output.glob("checkpoint-*"):
        try: step = int(path.name.split("-")[-1]); marker = json.loads((path / "complete.json").read_text())
        except (ValueError, FileNotFoundError, json.JSONDecodeError): continue
        required = [path / "bridge.safetensors", path / "understanding_lora.safetensors", path / "flexae.safetensors", path / "optimizer.pt"] + [path / f"rng_rank{rank}.pt" for rank in range(world_size)]
        if marker.get("schema") == CHECKPOINT_SCHEMA and marker.get("step") == step and all(item.is_file() for item in required): found.append((step, path))
    return max(found) if found else None


def save_checkpoint(output: Path, step: int, model: JanusStage2, optimizer: Any, scheduler: Any, rank: int, world_size: int, validation: dict[str, float] | None) -> None:
    path = output / f"checkpoint-{step}"; path.mkdir(parents=True, exist_ok=True)
    torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, path / f"rng_rank{rank}.pt"); dist.barrier()
    if rank == 0:
        save_file({key: value.detach().cpu().contiguous() for key, value in model.bridge.state_dict().items()}, path / "bridge.safetensors")
        save_file(lora_state(model.janus), path / "understanding_lora.safetensors")
        save_file({key: value.detach().cpu().contiguous() for key, value in model.flexae.state_dict().items()}, path / "flexae.safetensors")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path / "optimizer.pt")
        marker = {"schema": CHECKPOINT_SCHEMA, "architecture_id": ARCHITECTURE_ID, "training_mode": TRAINING_MODE, "stage": "stage2", "step": step, "world_size": world_size,
                  "bridge_tensors": len(model.bridge.state_dict()), "lora_tensors": len(lora_state(model.janus)), "layers": list(LAYERS), "g_tokens": G_TOKENS,
                  "u_tokens": 576, "rng_ranks": world_size, "encoder_unfreeze_step": 2501,
                  "flexae_encoder_trainable_after_step": 2500, "validation": validation}
        (path / "complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()


@torch.no_grad()
def validate(model: JanusStage2, data: FineVisionManifest, processor: Any, limit: int, rank: int, world_size: int, device: torch.device) -> dict[str, float] | None:
    from flexae_model import flatten_vq_latents
    from janus_backend import build_language_batch, extract_u, extract_vq, preprocess_images
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    model.eval()
    for index in range(rank, min(limit, len(data)), world_size):
        row = data.get(index); pixel = preprocess_images(processor, [load_image(row["image_bytes"])], device)
        u = extract_u(model.janus, pixel); vq, _ = extract_vq(model.janus, pixel); raw_g, _ = model.flexae.encode(flatten_vq_latents(vq).float())
        language = build_language_batch(processor, [row["prompt"]], [row["answer"]]).to(device); mask = torch.zeros((1, 576), dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            real, length, _ = model(u, raw_g, language, "real", mask)
            drop, _, _ = model(u, raw_g, language, "drop_g", mask)
        totals += torch.tensor((float(real / length), float(drop / length), float((drop-real)/length), 1), dtype=torch.float64, device=device)
    dist.all_reduce(totals); model.train()
    model.janus.vision_model.eval(); model.janus.aligner.eval()
    model.janus.gen_vision_model.eval(); model.janus.gen_aligner.eval(); model.flexae.eval()
    if rank: return None
    return {"real_ce": float(totals[0]/totals[3]), "drop_ce": float(totals[1]/totals[3]), "drop_gap": float(totals[2]/totals[3]), "samples": int(totals[3])}


def main() -> None:
    args = parse_args()
    global ARCHITECTURE_ID, TRAINING_MODE
    TRAINING_MODE = args.training_mode
    if TRAINING_MODE == "sft":
        ARCHITECTURE_ID = "janus_pro_1b_vq_flexae256_full_layer_supervised_ce_sft_v1"
    local_rank, rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    topology = (world_size, args.micro_batch_size, args.gradient_accumulation)
    if not args.allow_nonformal_smoke and topology not in {(8, 4, 1), (8, 2, 2), (2, 8, 2), (2, 16, 1)}:
        raise RuntimeError(f"unsupported formal Stage2 topology: {topology}")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank); dist.init_process_group("nccl", device_id=device)
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank); torch.cuda.manual_seed(args.seed + rank)
    from flexae_model import JanusFlexAE256, flatten_vq_latents
    from janus_backend import build_language_batch, extract_u, extract_vq, fixed_g_position_embedding, load_janus, preprocess_images
    janus, processor = load_janus(args.janus_code, args.janus_model, device); janus.requires_grad_(False); install_lora(janus, args.lora_rank, args.lora_alpha)
    flexae = JanusFlexAE256().to(device=device, dtype=torch.float32); flexae.load_state_dict(load_file(args.flexae_checkpoint), strict=True); flexae.requires_grad_(False).eval()
    model = JanusStage2(janus, flexae, fixed_g_position_embedding(device)).to(device).train()
    janus.vision_model.eval(); janus.aligner.eval(); janus.gen_vision_model.eval(); janus.gen_aligner.eval(); flexae.eval()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    resumed = latest_complete(output, world_size) if args.resume else None
    resumed_step = resumed[0] if resumed else 0
    flexae_encoder_parameters: list[nn.Parameter] = []
    if resumed_step >= args.encoder_unfreeze_step:
        flexae_encoder_parameters = enable_flexae_encoder(flexae)
    parameters = trainable_parameters(model)
    projector_ids = {id(value) for value in model.bridge.projector.parameters()}
    sampler_ids = {id(value) for value in model.bridge.samplers.parameters()}
    lora_ids = {id(value) for value in janus.parameters() if value.requires_grad}
    flexae_ids = {id(value) for value in flexae_encoder_parameters}
    if {id(value) for value in parameters} != projector_ids | sampler_ids | lora_ids | flexae_ids:
        raise RuntimeError("trainable parameter ownership contract mismatch")
    optimizer_groups = [
        {"params": [value for value in parameters if id(value) in projector_ids], "lr": args.projector_lr},
        {"params": [value for value in parameters if id(value) in sampler_ids], "lr": args.sampler_lr},
        {"params": [value for value in parameters if id(value) in lora_ids], "lr": args.lora_lr},
    ]
    if flexae_encoder_parameters:
        optimizer_groups.append({"params": flexae_encoder_parameters, "lr": args.extractor_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay, fused=True)
    lr_lambda = lambda step: lr_factor(step + 1, args.warmup_steps, args.schedule_max_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda for _ in optimizer.param_groups],
    )
    start = 1
    if resumed:
        resumed_step, path = resumed; model.bridge.load_state_dict(load_file(path / "bridge.safetensors"), strict=True); load_lora_state(janus, load_file(path / "understanding_lora.safetensors")); flexae.load_state_dict(load_file(path / "flexae.safetensors"), strict=True)
        state = torch.load(path / "optimizer.pt", map_location=device, weights_only=False); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        rng = torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False); torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"]); start = resumed_step + 1
    order, order_meta = load_order(args.balanced_order); train_data, val_data = FineVisionManifest(args.manifest), FineVisionManifest(args.val_manifest)
    mask_values = tuple(float(x) for x in args.mask_u_schedule.split(",")); mask_boundaries = tuple(int(x) for x in args.mask_u_boundaries.split(","))
    kl_points = tuple(int(x) for x in args.shuffle_kl_steps.split(",")); kl_values = tuple(float(x) for x in args.shuffle_kl_weights.split(","))
    if mask_boundaries[-1] != args.max_steps or kl_points[0] != 1 or kl_points[-1] != args.max_steps: raise ValueError("schedule horizon mismatch")
    if rank == 0:
        config = {**vars(args), "schema": RUN_SCHEMA, "architecture_id": ARCHITECTURE_ID, "training_mode": TRAINING_MODE, "world_size": world_size,
                  "global_effective_batch": world_size * args.micro_batch_size * args.gradient_accumulation,
                  "stage1_decoder_frozen": True, "encoder_unfreeze_step": args.encoder_unfreeze_step,
                  "native_u_tokens": 576, "g_memory": [256, 2048], "sampler_layers": list(LAYERS), "order": order_meta,
                  "answer_use": {"lambda_CE": args.real_ce_loss_weight, "lambda_DPO": args.answer_use_weight,
                                  "real_negative_ratio": [args.dpo_real_weight, args.dpo_negative_weight],
                                  "negative_gradient_cap_relative_to_real": args.dpo_negative_grad_cap,
                                  "metadata_weighting": False}}
        (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    telemetry = output / "train.jsonl"
    stop_after = args.stop_after_step or args.max_steps
    if not 1 <= stop_after <= args.max_steps: raise ValueError("stop-after-step must lie within the fixed schedule horizon")
    for step in range(start, stop_after + 1):
        if step == args.encoder_unfreeze_step:
            flexae_encoder_parameters = enable_flexae_encoder(flexae)
            optimizer.add_param_group({
                "params": flexae_encoder_parameters,
                "lr": args.extractor_lr * lr_factor(step, args.warmup_steps, args.schedule_max_steps, args.min_lr_ratio),
                "initial_lr": args.extractor_lr,
            })
            scheduler.base_lrs.append(args.extractor_lr)
            scheduler.lr_lambdas.append(lr_lambda)
            parameters = trainable_parameters(model)
            if rank == 0:
                print(json.dumps({"event": "flexae_encoder_unfrozen", "step": step, "parameters": len(flexae_encoder_parameters)}), flush=True)
        mask_prob = mask_values[bisect_left(mask_boundaries, step)]
        optimizer.zero_grad(set_to_none=True)
        global_batch = world_size * args.micro_batch_size * args.gradient_accumulation
        kl_weight = schedule_value(step, kl_points, kl_values)
        positive_grad = negative_grad = kl_grad = None
        tracked = torch.zeros(5, dtype=torch.float64, device=device)
        for accumulation in range(args.gradient_accumulation):
            base = (step - 1) * global_batch + (accumulation * world_size + rank) * args.micro_batch_size
            indices = [int(order[(base + i) % len(order)]) for i in range(args.micro_batch_size)]
            rows = [train_data.get(index) for index in indices]; donors = [train_data.get((index + len(train_data)//2) % len(train_data)) for index in indices]
            images = [load_image(row["image_bytes"]) for row in rows]; donor_images = [load_image(row["image_bytes"]) for row in donors]
            with torch.no_grad():
                pixel = preprocess_images(processor, images, device); donor_pixel = preprocess_images(processor, donor_images, device)
                u = extract_u(janus, pixel); vq, _ = extract_vq(janus, pixel); donor_vq, _ = extract_vq(janus, donor_pixel)
            if step >= args.encoder_unfreeze_step:
                raw_g, _ = flexae.encode(flatten_vq_latents(vq).float())
                donor_g, _ = flexae.encode(flatten_vq_latents(donor_vq).float())
            else:
                with torch.no_grad():
                    raw_g, _ = flexae.encode(flatten_vq_latents(vq).float())
                    donor_g, _ = flexae.encode(flatten_vq_latents(donor_vq).float())
            language = build_language_batch(processor, [row["prompt"] for row in rows], [row["answer"] for row in rows]).to(device)
            fixed_mask = (torch.rand((args.micro_batch_size, 1), device=device) < mask_prob).expand(-1, 576)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with disable_lora(janus), torch.no_grad():
                    reference_sum, reference_length, reference_logits = model(u, donor_g, language, "drop_g", fixed_mask)
                real_sum, lengths, _ = model(u, raw_g, language, "real", fixed_mask); shuffle_sum, shuffle_lengths, shuffle_logits = model(u, donor_g, language, "real", fixed_mask)
                if not torch.equal(lengths, shuffle_lengths) or not torch.equal(lengths, reference_length): raise RuntimeError("answer length changed across branches")
                real_ce, shuffle_ce = real_sum / lengths, shuffle_sum / lengths
                # Paper definition: mean answer-token CE on the matched-G
                # branch, scaled by lambda_CE. Metadata and sqrt-length
                # weighting are excluded from paper-main.
                objective = args.real_ce_loss_weight * real_ce.mean()
                q = args.answer_use_beta * torch.sigmoid(-args.answer_use_beta * (shuffle_ce - real_ce)).detach()
                positive = objective + args.answer_use_weight * args.dpo_real_weight * (q * real_ce).mean()
                negative = -args.answer_use_weight * args.dpo_negative_weight * (q * shuffle_ce).mean()
                kl = forward_kl(shuffle_logits, reference_logits)
            micro_positive = synchronized_branch_gradients(positive, parameters, True, world_size)
            micro_negative = synchronized_branch_gradients(negative, parameters, True, world_size)
            micro_kl = synchronized_branch_gradients(kl_weight * kl, parameters, False, world_size)
            if positive_grad is None:
                positive_grad = [value / args.gradient_accumulation for value in micro_positive]
                negative_grad = [value / args.gradient_accumulation for value in micro_negative]
                kl_grad = [value / args.gradient_accumulation for value in micro_kl]
            else:
                for target, value in zip(positive_grad, micro_positive, strict=True): target.add_(value, alpha=1 / args.gradient_accumulation)
                for target, value in zip(negative_grad, micro_negative, strict=True): target.add_(value, alpha=1 / args.gradient_accumulation)
                for target, value in zip(kl_grad, micro_kl, strict=True): target.add_(value, alpha=1 / args.gradient_accumulation)
            tracked += torch.stack((real_ce.mean(), shuffle_ce.mean(), objective, kl, (shuffle_ce-real_ce).mean())).detach().double() / args.gradient_accumulation
        assert positive_grad is not None and negative_grad is not None and kl_grad is not None
        real_ce_mean, shuffle_ce_mean, objective_mean, kl_mean, gap_mean = tracked
        cap_scale = min(1.0, args.dpo_negative_grad_cap * global_norm(positive_grad) / max(global_norm(negative_grad), 1e-12))
        for parameter, pos, neg, anchor in zip(parameters, positive_grad, negative_grad, kl_grad, strict=True): parameter.grad = (pos + cap_scale * neg + anchor).to(parameter.dtype)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm); optimizer.step(); scheduler.step()
        if rank == 0 and (step == 1 or step % args.logging_steps == 0):
            record = {"step": step, "real_ce": float(real_ce_mean), "shuffle_ce": float(shuffle_ce_mean), "gap": float(gap_mean), "objective": float(objective_mean), "shuffle_kl": float(kl_mean), "shuffle_kl_weight": kl_weight, "mask_u_probability": mask_prob, "negative_cap_scale": cap_scale, "grad_norm": float(grad_norm), "lr": [group["lr"] for group in optimizer.param_groups], "peak_gib": torch.cuda.max_memory_allocated()/2**30, "time": time.time()}
            with telemetry.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step % args.save_steps == 0 or step == stop_after:
            metrics = validate(model, val_data, processor, args.final_val_limit if step == args.max_steps else min(32, args.final_val_limit), rank, world_size, device)
            save_checkpoint(output, step, model, optimizer, scheduler, rank, world_size, metrics)
    if rank == 0 and stop_after == args.max_steps:
        marker = json.loads((output / f"checkpoint-{args.max_steps}" / "complete.json").read_text()); (output / "complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True)+"\n")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__": main()
