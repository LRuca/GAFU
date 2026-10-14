#!/usr/bin/env python3
"""Stage1 Lance/Wan FlexAE training on one epoch of unique FineVision images."""

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
import tempfile
import time
from array import array
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel


BF16 = torch.bfloat16
SCHEMA = "lance3b_finevision_wan_flexae256_stage1_checkpoint_v1"
RUN_SCHEMA = "lance3b_finevision_wan_flexae256_stage1_run_v1"
ARCHITECTURE_ID = "lance3b_wan48_flexae256_relation_v1"


class FineVisionImages:
    def __init__(self, manifest: str) -> None:
        self.manifest = Path(manifest).resolve()
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.files: list[tuple[Path, int]] = []
        for item in payload.get("files", []):
            path = self.manifest.parent / item["path"]
            if not path.exists():
                path = self.manifest.parent.parent / item["path"]
            if "text_only" in path.parts or not path.exists():
                continue
            self.files.append((path, int(item["rows"])))
        if not self.files:
            raise RuntimeError(f"no FineVision image shards in {self.manifest}")
        self.ends: list[int] = []
        total = 0
        for _, rows in self.files:
            total += rows
            self.ends.append(total)
        self._file_index: int | None = None
        self._parquet: pq.ParquetFile | None = None
        self._group_index: int | None = None
        self._table: Any = None

    def __len__(self) -> int:
        return self.ends[-1]

    def image_bytes(self, index: int) -> bytes:
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self.ends, index)
        start = 0 if file_index == 0 else self.ends[file_index - 1]
        local = index - start
        if self._file_index != file_index:
            self._file_index = file_index
            self._parquet = pq.ParquetFile(self.files[file_index][0])
            self._group_index = None
        assert self._parquet is not None
        group_start = 0
        for group_index in range(self._parquet.num_row_groups):
            rows = self._parquet.metadata.row_group(group_index).num_rows
            if local < group_start + rows:
                if self._group_index != group_index:
                    self._group_index = group_index
                    self._table = self._parquet.read_row_group(group_index, columns=["images"])
                images = self._table["images"][local - group_start].as_py() or []
                value = (images[0] or {}).get("bytes") if images else None
                if not value:
                    raise ValueError(f"FineVision row {index} has no embedded image")
                return value
            group_start += rows
        raise IndexError(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--unique-order", required=True)
    parser.add_argument("--lance-runtime", required=True)
    parser.add_argument("--lance-model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--native-loss-weight", type=float, default=0.25)
    parser.add_argument("--binding-loss-weight", type=float, default=1.0)
    parser.add_argument("--global-relation-weight", type=float, default=0.137)
    parser.add_argument("--local-relation-weight", type=float, default=1.34)
    parser.add_argument("--relation-ramp-steps", type=int, default=100)
    parser.add_argument("--relation-initial-scale", type=float, default=0.2)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(value: bytes) -> Image.Image:
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB")


def gather_nodes(nodes: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return nodes
    return torch.cat(tuple(differentiable_all_gather(nodes)), dim=0)


def pool_nodes(tokens: torch.Tensor, size: int) -> torch.Tensor:
    side = int(math.isqrt(tokens.shape[1]))
    if tokens.ndim != 3 or side * side != tokens.shape[1]:
        raise ValueError(f"expected square token sequence, got {tuple(tokens.shape)}")
    values = tokens.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)
    values = F.adaptive_avg_pool2d(values, (size, size))
    return values.permute(0, 2, 3, 1).reshape(tokens.shape[0], size * size, -1)


def relation_values(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(nodes.float(), dim=-1)
    values = (normalized @ normalized.T)[mask]
    return values - values.mean()


def relation_distance(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    left = relation_values(student, mask)
    right = relation_values(teacher.detach(), mask)
    return 1.0 - (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-8)


def multiscale_relation_losses(
    g_tokens: torch.Tensor,
    teacher: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    global_g = gather_nodes(pool_nodes(g_tokens, 1).squeeze(1), world_size)
    global_t = gather_nodes(pool_nodes(teacher, 1).squeeze(1), world_size)
    global_mask = ~torch.eye(global_g.shape[0], dtype=torch.bool, device=global_g.device)
    global_loss = relation_distance(global_g, global_t, global_mask)
    local = {}
    for size in (4, 8, 16):
        student = gather_nodes(pool_nodes(g_tokens, size), world_size)
        target = gather_nodes(pool_nodes(teacher, size), world_size)
        batch, count, width = student.shape
        student = student.reshape(batch * count, width)
        target = target.reshape(batch * count, width)
        image_ids = torch.arange(batch, device=student.device).repeat_interleave(count)
        diagonal = torch.eye(batch * count, dtype=torch.bool, device=student.device)
        intra = image_ids[:, None].eq(image_ids[None, :]) & ~diagonal
        inter = image_ids[:, None].ne(image_ids[None, :])
        local[size] = 0.5 * (
            relation_distance(student, target, intra)
            + relation_distance(student, target, inter)
        )
    return global_loss, local


def reconstruction_losses(
    result: dict[str, torch.Tensor],
    patches: torch.Tensor,
    native_weight: torch.Tensor,
    native_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from flexae_model import group_local_patch_targets

    reconstructed = result["reconstructed_patches"]
    patch_mse = (reconstructed.float() - patches.float()).square().mean()
    native_target = F.linear(patches.float(), native_weight, native_bias).detach()
    native_reconstructed = F.linear(reconstructed.float(), native_weight, native_bias)
    native_nmse = (
        (native_reconstructed - native_target).square().mean()
        / native_target.square().mean().detach().clamp_min(1e-8)
    )
    local_target = group_local_patch_targets(patches.float())
    local_error = (result["local_prediction"].float() - local_target).square().mean(dim=-1)
    local_energy = local_target.square().mean(dim=-1).detach().clamp_min(1e-8)
    binding_nmse = (local_error / local_energy).mean()
    return {
        "patch_mse": patch_mse,
        "native_nmse": native_nmse,
        "binding_nmse": binding_nmse,
    }


def relation_scale(step: int, args: argparse.Namespace) -> float:
    if args.relation_ramp_steps <= 0:
        return 1.0
    progress = min(1.0, step / args.relation_ramp_steps)
    return args.relation_initial_scale + (1.0 - args.relation_initial_scale) * progress


def complete_checkpoint(path: Path, world_size: int) -> tuple[int, dict[str, Any]] | None:
    required = ["flexae.safetensors", "optimizer.pt", "complete.json"]
    required.extend(f"rng_rank{rank}.pt" for rank in range(world_size))
    if any(not (path / item).is_file() for item in required):
        return None
    marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if marker.get("schema") != SCHEMA or int(marker.get("world_size", -1)) != world_size:
        return None
    step = int(marker["step"])
    if path.name != f"checkpoint-{step}":
        return None
    return step, marker


def save_checkpoint(
    output: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_states: list[dict[str, Any]],
    validation: dict[str, float],
    world_size: int,
) -> None:
    final = output / f"checkpoint-{step}"
    if complete_checkpoint(final, world_size) is not None:
        return
    temporary = output / f".checkpoint-{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    module = model.module if isinstance(model, DistributedDataParallel) else model
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()},
        str(temporary / "flexae.safetensors"),
    )
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "optimizer.pt",
    )
    for rank, state in enumerate(rng_states):
        torch.save(state, temporary / f"rng_rank{rank}.pt")
    marker = {
        "schema": SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "step": step,
        "world_size": world_size,
        "validation": validation,
        "rng_ranks": len(rng_states),
    }
    (temporary / "complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, final)


def retain_checkpoints(output: Path, world_size: int) -> dict[str, Any]:
    complete = []
    for path in output.glob("checkpoint-*"):
        result = complete_checkpoint(path, world_size)
        if result is not None:
            complete.append((result[0], path, result[1]))
    complete.sort()
    latest = {step for step, _, _ in complete[-2:]}
    scored = []
    for step, _, marker in complete:
        validation = marker.get("validation", {})
        if all(name in validation for name in ("patch_mse", "native_nmse", "binding_nmse")):
            score = (
                float(validation["patch_mse"])
                + 0.25 * float(validation["native_nmse"])
                + float(validation["binding_nmse"])
            )
            scored.append((score, step))
    best_step = min(scored)[1] if scored else None
    keep = latest | ({best_step} if best_step is not None else set())
    removed = []
    for step, path, _ in complete:
        if step not in keep:
            shutil.rmtree(path)
            removed.append(step)
    decision = {
        "latest_steps": sorted(latest), "best_validation_step": best_step,
        "kept_steps": sorted(keep), "removed_steps": removed,
    }
    temporary = output / ".retention.json.tmp"
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "retention.json")
    return decision


@torch.no_grad()
def validate(
    model: nn.Module,
    teacher: nn.Module,
    wan: Any,
    native_weight: torch.Tensor,
    native_bias: torch.Tensor,
    data: FineVisionImages,
    limit: int,
    rank: int,
    world_size: int,
    device: torch.device,
    preprocess_qwen_vit: Any,
    extract_qwen_u: Any,
    flatten_wan_latents: Any,
) -> dict[str, float] | None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for index in range(rank, min(limit, len(data)), world_size):
        image = load_image(data.image_bytes(index))
        vit_patches, grids = preprocess_qwen_vit([image])
        with torch.autocast("cuda", dtype=BF16):
            u = extract_qwen_u(teacher, vit_patches, grids, device)
            patches = flatten_wan_latents(wan.encode([image]))
            result = module(patches)
        losses = reconstruction_losses(result, patches, native_weight, native_bias)
        cosine = F.cosine_similarity(
            F.normalize(result["g_tokens"].float(), dim=-1),
            F.normalize(u.float(), dim=-1),
            dim=-1,
        ).mean()
        sums += torch.tensor([
            float(losses["patch_mse"]), float(losses["native_nmse"]),
            float(losses["binding_nmse"]), float(cosine),
        ], dtype=torch.float64, device=device)
    count = torch.tensor(
        sum(1 for index in range(rank, min(limit, len(data)), world_size)),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(sums)
    dist.all_reduce(count)
    module.train()
    if rank != 0:
        return None
    values = sums / count.clamp_min(1)
    return {
        "patch_mse": float(values[0]),
        "native_nmse": float(values[1]),
        "binding_nmse": float(values[2]),
        "g_u_token_cosine": float(values[3]),
        "samples": int(count),
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    for path in (str(Path(__file__).resolve().parent), args.lance_runtime):
        if path not in os.sys.path:
            os.sys.path.insert(0, path)
    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group("nccl", device_id=device)
        else:
            fd, group_file = tempfile.mkstemp(prefix="lance_stage1_")
            os.close(fd)
            os.unlink(group_file)
            dist.init_process_group(
                "nccl", init_method=f"file://{group_file}", rank=0, world_size=1,
                device_id=device,
            )

    from flexae_model import BagelFlexAE256
    from lance_backend import (
        FrozenWanEncoder,
        extract_qwen_u,
        flatten_wan_latents,
        load_native_vae_projection,
        load_qwen_vit_teacher,
        preprocess_qwen_vit,
    )

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    train_data = FineVisionImages(args.manifest)
    val_data = FineVisionImages(args.val_manifest)
    prefix = Path(args.unique_order)
    order_meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    binary = prefix.with_suffix(".bin")
    expected = {
        "schema": "finevision_unique_image_order_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "world_size": world_size,
        "micro_batch_size": args.micro_batch_size,
    }
    mismatch = {key: (order_meta.get(key), value) for key, value in expected.items()
                if order_meta.get(key) != value}
    if mismatch or sha256(binary) != order_meta.get("order_sha256"):
        raise RuntimeError(f"unique image order contract mismatch: {mismatch}")
    order = array("I")
    with binary.open("rb") as handle:
        order.fromfile(handle, int(order_meta["order_entries"]))
    max_steps = int(order_meta["max_steps"])

    teacher = load_qwen_vit_teacher(args.lance_runtime, args.lance_model_root, device)
    wan = FrozenWanEncoder(
        args.lance_runtime,
        str(Path(args.lance_model_root) / "Wan2.2_VAE.pth"),
        device,
    )
    native_weight, native_bias = load_native_vae_projection(args.lance_model_root, device)
    model = BagelFlexAE256(
        patch_dim=48, bagel_dim=2048, gradient_checkpointing=True,
    ).to(device=device, dtype=torch.float32).train()
    trainable: nn.Module = DistributedDataParallel(
        model, device_ids=[local_rank], gradient_as_bucket_view=True,
    ) if world_size > 1 else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), fused=True,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-8, (step + 1) / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume:
        candidates = []
        for checkpoint in output.glob("checkpoint-*"):
            result = complete_checkpoint(checkpoint, world_size)
            if result is not None:
                candidates.append((result[0], checkpoint))
        if candidates:
            start_step, checkpoint = max(candidates)
            model.load_state_dict(load_file(checkpoint / "flexae.safetensors"), strict=True)
            state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng = torch.load(checkpoint / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch_cpu"])
            torch.cuda.set_rng_state(rng["torch_cuda"], device=device)
            if rank == 0:
                print(json.dumps({"event": "resume", "step": start_step,
                                  "checkpoint": str(checkpoint)}), flush=True)

    if rank == 0:
        config = {
            **vars(args), "schema": RUN_SCHEMA, "architecture_id": ARCHITECTURE_ID,
            "world_size": world_size, "global_batch": world_size * args.micro_batch_size,
            "max_steps": max_steps, "unique_images": order_meta["unique_images"],
            "loss": "patch_mse + 0.25*native_nmse + binding_nmse + ramp*(0.137*global_relation + 1.34*mean(local_4_8_16))",
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        (output / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dist.barrier()
    telemetry = output / "train.jsonl"
    started = time.monotonic()
    global_batch = world_size * args.micro_batch_size
    for step in range(start_step + 1, max_steps + 1):
        offset = (step - 1) * global_batch + rank * args.micro_batch_size
        images = [load_image(train_data.image_bytes(int(order[offset + index])))
                  for index in range(args.micro_batch_size)]
        vit_patches, grids = preprocess_qwen_vit(images)
        with torch.inference_mode(), torch.autocast("cuda", dtype=BF16):
            teacher_tokens = extract_qwen_u(teacher, vit_patches, grids, device)
            patches = flatten_wan_latents(wan.encode(images))
        with torch.autocast("cuda", dtype=BF16):
            result = trainable(patches)
            losses = reconstruction_losses(result, patches, native_weight, native_bias)
            global_relation, local_relations = multiscale_relation_losses(
                result["g_tokens"], teacher_tokens, world_size,
            )
            local_relation = torch.stack(tuple(local_relations.values())).mean()
            scale = relation_scale(step, args)
            loss = (
                losses["patch_mse"]
                + args.native_loss_weight * losses["native_nmse"]
                + args.binding_loss_weight * losses["binding_nmse"]
                + scale * (
                    args.global_relation_weight * global_relation
                    + args.local_relation_weight * local_relation
                )
            )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        values = torch.tensor([
            float(loss.detach()), float(losses["patch_mse"].detach()),
            float(losses["native_nmse"].detach()), float(losses["binding_nmse"].detach()),
            float(global_relation.detach()), float(local_relation.detach()), grad_norm,
        ], dtype=torch.float64, device=device)
        dist.all_reduce(values)
        values.div_(world_size)
        if rank == 0 and (step == 1 or step % args.logging_steps == 0):
            record = {
                "event": "train", "step": step, "loss": float(values[0]),
                "patch_mse": float(values[1]), "native_nmse": float(values[2]),
                "binding_nmse": float(values[3]), "global_relation": float(values[4]),
                "local_relation": float(values[5]), "relation_scale": scale,
                "grad_norm": float(values[6]), "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with telemetry.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % args.save_steps == 0 or step == max_steps:
            validation = validate(
                trainable, teacher, wan, native_weight, native_bias, val_data,
                args.val_limit, rank, world_size, device, preprocess_qwen_vit,
                extract_qwen_u, flatten_wan_latents,
            )
            rng = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                   "torch_cuda": torch.cuda.get_rng_state(device)}
            gathered: list[dict[str, Any]] | None = [None] * world_size if rank == 0 else None
            dist.gather_object(rng, gathered, dst=0)
            if rank == 0:
                assert validation is not None and gathered is not None
                record = {"event": "validation", "step": step, **validation}
                with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                save_checkpoint(
                    output, step, trainable, optimizer, scheduler, gathered,
                    validation, world_size,
                )
                print(json.dumps({
                    "event": "checkpoint_retention",
                    **retain_checkpoints(output, world_size),
                }, sort_keys=True), flush=True)
                print(json.dumps(record, sort_keys=True), flush=True)
            dist.barrier()

    if rank == 0:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
            str(output / "flexae_final.safetensors"),
        )
        (output / "complete.json").write_text(json.dumps({
            "schema": RUN_SCHEMA, "architecture_id": ARCHITECTURE_ID,
            "step": max_steps, "world_size": world_size,
            "flexae": "flexae_final.safetensors",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
