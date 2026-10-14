#!/usr/bin/env python3
"""Train Janus FlexAE on frozen native VQ and SigLIP routes."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import time
from array import array
from bisect import bisect_right
from contextlib import nullcontext
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


SCHEMA = "janus_pro_1b_vq_flexae256_stage1_checkpoint_v1"
RUN_SCHEMA = "janus_pro_1b_vq_flexae256_stage1_run_v1"
ARCHITECTURE_ID = "janus_pro_vq24_flexae256_semantic_spatial_v1"


class FineVisionImages:
    def __init__(self, manifest: str) -> None:
        self.manifest = Path(manifest).resolve()
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.files: list[tuple[Path, int]] = []
        for item in payload.get("files", []):
            path = self.manifest.parent / item["path"]
            if not path.exists():
                path = self.manifest.parent.parent / item["path"]
            if path.exists() and "text_only" not in path.parts:
                self.files.append((path, int(item["rows"])))
        if not self.files:
            raise RuntimeError("FineVision manifest has no embedded-image shards")
        self.ends, total = [], 0
        for _path, rows in self.files:
            total += rows
            self.ends.append(total)
        self._file_index = self._group_index = None
        self._parquet = self._table = None

    def __len__(self) -> int:
        return self.ends[-1]

    def image(self, index: int) -> Image.Image:
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
                    self._group_index = group
                    self._table = self._parquet.read_row_group(group, columns=["images"])
                images = self._table["images"][local - start].as_py() or []
                value = (images[0] or {}).get("bytes") if images else None
                if not value:
                    raise ValueError(f"row {index} has no image")
                with Image.open(io.BytesIO(value)) as image:
                    return image.convert("RGB")
            start += rows
        raise IndexError(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--unique-order", required=True)
    parser.add_argument("--janus-code", required=True)
    parser.add_argument("--janus-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
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
    parser.add_argument("--keep-latest-checkpoints", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonformal-smoke", action="store_true")
    return parser.parse_args()


def pool_nodes(tokens: torch.Tensor, size: int) -> torch.Tensor:
    side = math.isqrt(tokens.shape[1])
    values = tokens.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(values, (size, size)).permute(0, 2, 3, 1).reshape(tokens.shape[0], size * size, -1)


def gather_nodes(tokens: torch.Tensor, world_size: int) -> torch.Tensor:
    return tokens if world_size == 1 else torch.cat(tuple(differentiable_all_gather(tokens)), dim=0)


def relation_values(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(nodes.float(), dim=-1)
    values = (normalized @ normalized.T)[mask]
    return values - values.mean()


def relation_distance(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    left, right = relation_values(student, mask), relation_values(teacher.detach(), mask)
    return 1.0 - (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-8)


def relation_losses(g: torch.Tensor, u: torch.Tensor, world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_g = gather_nodes(pool_nodes(g, 1).squeeze(1), world_size)
    global_u = gather_nodes(pool_nodes(u, 1).squeeze(1), world_size)
    mask = ~torch.eye(global_g.shape[0], dtype=torch.bool, device=g.device)
    global_loss = relation_distance(global_g, global_u, mask)
    local = []
    for size in (4, 8, 16):
        student, teacher = gather_nodes(pool_nodes(g, size), world_size), gather_nodes(pool_nodes(u, size), world_size)
        batch, count, _ = student.shape
        student, teacher = student.reshape(batch * count, -1), teacher.reshape(batch * count, -1)
        image_ids = torch.arange(batch, device=g.device).repeat_interleave(count)
        diagonal = torch.eye(batch * count, dtype=torch.bool, device=g.device)
        intra = image_ids[:, None].eq(image_ids[None]) & ~diagonal
        inter = image_ids[:, None].ne(image_ids[None])
        local.append(0.5 * (relation_distance(student, teacher, intra) + relation_distance(student, teacher, inter)))
    return global_loss, torch.stack(local).mean()


def cosine_factor(step: int, warmup: int, maximum: int, floor: float) -> float:
    if step <= warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_order(prefix: str) -> tuple[array, dict[str, Any]]:
    binary, metadata = Path(prefix + ".bin"), Path(prefix + ".json")
    values = array("I")
    with binary.open("rb") as handle:
        values.fromfile(handle, binary.stat().st_size // values.itemsize)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if len(values) != int(meta["order_entries"]):
        raise RuntimeError("unique order length contract mismatch")
    return values, meta


def latest_complete(output: Path, world_size: int) -> tuple[int, Path] | None:
    found = []
    for path in output.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
            marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            continue
        required = [path / "flexae.safetensors", path / "optimizer.pt"] + [path / f"rng_rank{rank}.pt" for rank in range(world_size)]
        if marker.get("schema") == SCHEMA and marker.get("step") == step and all(item.is_file() for item in required):
            found.append((step, path))
    return max(found) if found else None


def prune_complete_checkpoints(output: Path, keep: int, world_size: int) -> None:
    if keep <= 0:
        return
    output = output.resolve()
    expected = {"complete.json", "flexae.safetensors", "optimizer.pt"}
    expected.update(f"rng_rank{rank}.pt" for rank in range(world_size))
    complete: list[tuple[int, Path, int]] = []
    for path in output.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved.parent != output:
            raise RuntimeError(f"checkpoint retention boundary violation: {resolved}")
        entries = list(resolved.iterdir())
        if any(entry.is_symlink() for entry in entries) or {entry.name for entry in entries} != expected:
            continue
        marker = json.loads((resolved / "complete.json").read_text(encoding="utf-8"))
        step = int(match.group(1))
        if marker.get("schema") != SCHEMA or marker.get("step") != step or marker.get("world_size") != world_size:
            continue
        complete.append((step, resolved, sum(entry.stat().st_size for entry in entries)))
    obsolete = sorted(complete)[:-keep]
    if not obsolete:
        return
    audit = output / "retention.jsonl"
    planned = {
        "event": "checkpoint_retention_plan",
        "keep": keep,
        "world_size": world_size,
        "paths": [str(path) for _step, path, _size in obsolete],
        "bytes": sum(size for _step, _path, size in obsolete),
        "time": time.time(),
    }
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(planned, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    deleted = []
    for _step, path, size in obsolete:
        for name in sorted(expected):
            (path / name).unlink()
        path.rmdir()
        deleted.append({"path": str(path), "bytes": size})
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "event": "checkpoint_retention_complete",
            "deleted": deleted,
            "time": time.time(),
        }, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(
    output: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, rank: int, world_size: int, validation: dict[str, float] | None,
    keep_latest_checkpoints: int,
) -> None:
    path = output / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, path / f"rng_rank{rank}.pt")
    dist.barrier()
    if rank == 0:
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, path / "flexae.safetensors")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path / "optimizer.pt")
        marker = {
            "schema": SCHEMA, "architecture_id": ARCHITECTURE_ID, "stage": "stage1", "step": step,
            "world_size": world_size, "source_grid": [24, 24], "g_grid": [16, 16],
            "source_dim": 8, "g_dim": 1024, "rng_ranks": world_size, "validation": validation,
        }
        (path / "complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()
    if rank == 0:
        prune_complete_checkpoints(output, keep_latest_checkpoints, world_size)
    dist.barrier()


@torch.no_grad()
def validate(model: nn.Module, janus: nn.Module, processor: Any, data: FineVisionImages, limit: int, rank: int, world_size: int, device: torch.device) -> dict[str, float] | None:
    from flexae_model import flatten_vq_latents, spatial_pool_targets
    from janus_backend import extract_vq, native_generation_embedding, preprocess_images
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    count = 0
    for index in range(rank, min(limit, len(data)), world_size):
        pixel = preprocess_images(processor, [data.image(index)], device)
        vq, _ = extract_vq(janus, pixel)
        source = flatten_vq_latents(vq).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(source)
            reconstruction = result["reconstructed_tokens"]
            patch = F.mse_loss(reconstruction.float(), source)
            native = F.mse_loss(native_generation_embedding(janus, reconstruction), native_generation_embedding(janus, source).detach()) / native_generation_embedding(janus, source).float().square().mean().clamp_min(1e-8)
            target = spatial_pool_targets(source)
            binding = ((result["local_prediction"].float() - target).square().mean(-1) / target.square().mean(-1).clamp_min(1e-8)).mean()
        sums += torch.tensor((float(patch), float(native), float(binding), 1.0), device=device, dtype=torch.float64)
        count += 1
    dist.all_reduce(sums)
    model.train()
    if rank:
        return None
    return {"patch_mse": float(sums[0] / sums[3]), "native_nmse": float(sums[1] / sums[3]), "binding_nmse": float(sums[2] / sums[3]), "samples": int(sums[3])}


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    from flexae_model import JanusFlexAE256, flatten_vq_latents, spatial_pool_targets
    from janus_backend import extract_u, extract_vq, load_janus, native_generation_embedding, preprocess_images

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    order, order_meta = load_order(args.unique_order)
    global_batch = world_size * args.micro_batch_size * args.gradient_accumulation
    maximum = args.max_steps or len(order) // global_batch
    topology = (world_size, args.micro_batch_size, args.gradient_accumulation)
    if not args.allow_nonformal_smoke and topology not in {(8, 4, 1), (2, 8, 2)}:
        raise RuntimeError(f"unsupported formal Stage1 topology: {topology}")
    janus, processor = load_janus(args.janus_code, args.janus_model, device)
    janus.requires_grad_(False)
    model = JanusFlexAE256().to(device=device, dtype=torch.float32).train()
    ddp = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda current: cosine_factor(current + 1, args.warmup_steps, maximum, args.min_lr_ratio))
    start = 1
    resumed = latest_complete(output, world_size) if args.resume else None
    if resumed:
        step, path = resumed
        model.load_state_dict(load_file(path / "flexae.safetensors"), strict=True)
        state = torch.load(path / "optimizer.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng = torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
        start = step + 1
    train_data, val_data = FineVisionImages(args.manifest), FineVisionImages(args.val_manifest)
    if rank == 0:
        config = {**vars(args), "schema": RUN_SCHEMA, "architecture_id": ARCHITECTURE_ID, "world_size": world_size, "global_effective_batch": global_batch, "max_steps": maximum, "order": order_meta,
                  "relation_batch_contract": "global_effective_batch",
                  "loss": "vq_mse + .25*native_nmse + binding + ramp*(.137*global_relation + 1.34*local_relation)"}
        (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    telemetry = output / "train.jsonl"
    stop_after = args.stop_after_step or maximum
    if not 1 <= stop_after <= maximum:
        raise ValueError("stop-after-step must lie within the fixed schedule horizon")
    for step in range(start, stop_after + 1):
        optimizer.zero_grad(set_to_none=True)
        tracked = torch.zeros(6, dtype=torch.float64, device=device)
        exact_relation_accumulation = args.gradient_accumulation > 1
        base_losses: list[torch.Tensor] = []
        base_metrics: list[torch.Tensor] = []
        accumulated_g: list[torch.Tensor] = []
        accumulated_u: list[torch.Tensor] = []
        for accumulation in range(args.gradient_accumulation):
            offset = (step - 1) * global_batch + (accumulation * world_size + rank) * args.micro_batch_size
            images = [train_data.image(int(order[offset + index])) for index in range(args.micro_batch_size)]
            pixel = preprocess_images(processor, images, device)
            with torch.no_grad():
                u = extract_u(janus, pixel)
                vq, _ = extract_vq(janus, pixel)
                source = flatten_vq_latents(vq).float()
            sync = nullcontext() if exact_relation_accumulation else (ddp.no_sync() if accumulation + 1 < args.gradient_accumulation else nullcontext())
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                result = ddp.module(source) if exact_relation_accumulation else ddp(source)
                reconstructed = result["reconstructed_tokens"]
                patch_mse = F.mse_loss(reconstructed.float(), source)
                native_target = native_generation_embedding(janus, source).detach()
                native_nmse = F.mse_loss(native_generation_embedding(janus, reconstructed), native_target) / native_target.float().square().mean().clamp_min(1e-8)
                local_target = spatial_pool_targets(source)
                binding = ((result["local_prediction"].float() - local_target).square().mean(-1) / local_target.square().mean(-1).clamp_min(1e-8)).mean()
                progress = min(1.0, step / max(1, args.relation_ramp_steps))
                relation_scale = args.relation_initial_scale + (1.0 - args.relation_initial_scale) * progress
                base_loss = patch_mse + args.native_loss_weight * native_nmse + args.binding_loss_weight * binding
                if exact_relation_accumulation:
                    base_losses.append(base_loss)
                    base_metrics.append(torch.stack((patch_mse, native_nmse, binding)))
                    accumulated_g.append(result["g_tokens"])
                    accumulated_u.append(u)
                else:
                    global_relation, local_relation = relation_losses(result["g_tokens"], u, world_size)
                    loss = base_loss + relation_scale * (args.global_relation_weight * global_relation + args.local_relation_weight * local_relation)
                    (loss / args.gradient_accumulation).backward()
            if not exact_relation_accumulation:
                tracked += torch.stack((loss, patch_mse, native_nmse, binding, global_relation, local_relation)).detach().double() / args.gradient_accumulation
        if exact_relation_accumulation:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                global_relation, local_relation = relation_losses(torch.cat(accumulated_g), torch.cat(accumulated_u), world_size)
                base_loss = torch.stack(base_losses).mean()
                loss = base_loss + relation_scale * (args.global_relation_weight * global_relation + args.local_relation_weight * local_relation)
                loss.backward()
            if world_size > 1:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(world_size)
            patch_mse, native_nmse, binding = torch.stack(base_metrics).mean(0)
            tracked = torch.stack((loss, patch_mse, native_nmse, binding, global_relation, local_relation)).detach().double()
        loss, patch_mse, native_nmse, binding, global_relation, local_relation = tracked
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step(); scheduler.step()
        if rank == 0 and (step == 1 or step % args.logging_steps == 0):
            record = {"step": step, "loss": float(loss), "patch_mse": float(patch_mse), "native_nmse": float(native_nmse), "binding_nmse": float(binding), "global_relation": float(global_relation), "local_relation": float(local_relation), "relation_scale": relation_scale, "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"], "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "time": time.time()}
            with telemetry.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step % args.save_steps == 0 or step == stop_after:
            metrics = validate(model, janus, processor, val_data, args.val_limit, rank, world_size, device)
            save_checkpoint(
                output, step, model, optimizer, scheduler, rank, world_size, metrics,
                args.keep_latest_checkpoints,
            )
    if rank == 0 and stop_after == maximum:
        final = json.loads((output / f"checkpoint-{maximum}" / "complete.json").read_text(encoding="utf-8"))
        (output / "complete.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
