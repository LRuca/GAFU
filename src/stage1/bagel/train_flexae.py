from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

from flexae_model import (
    BagelFlexAE256,
    group_local_patch_targets,
    patchify_latents,
    unpatchify_latents,
)


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    bagel_root: str
    bagel_model_path: str
    max_steps: int = 3000
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 250
    logging_steps: int = 10
    val_limit: int = 64
    train_limit: int = 0
    num_workers: int = 4
    seed: int = 20260805
    native_loss_weight: float = 0.25
    local_loss_weight: float = 0.25
    image_size: int = 512
    depth: int = 6
    hidden_size: int = 1024
    num_heads: int = 8
    ffn_size: int = 4096
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    deduplicate_images: bool = True
    posterior_mode: str = "mean"


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str,
        *,
        image_size: int,
        limit: int,
        deduplicate: bool,
    ) -> None:
        records: list[tuple[str, str]] = []
        seen: set[str] = set()
        manifest_root = Path(manifest).resolve().parent
        with open(manifest, "r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                payload = json.loads(line)
                image_path = Path(str(payload["image_path"]))
                if not image_path.is_absolute():
                    image_path = manifest_root / image_path
                path = str(image_path)
                if deduplicate and path in seen:
                    continue
                seen.add(path)
                if Path(path).is_file():
                    records.append((str(payload.get("id", line_index)), path))
                if limit > 0 and len(records) >= limit:
                    break
        if not records:
            raise RuntimeError(f"manifest has no readable images: {manifest}")
        self.records = records
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, path = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
            tensor = TF.pil_to_tensor(image).float().div_(127.5).sub_(1.0)
        return {"id": sample_id, "path": path, "image": tensor}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for field in TrainConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if field.default is field.default_factory:  # pragma: no cover
            parser.add_argument(name, required=True)
            continue
        if field.default is not field.default_factory and field.default is not None:
            if isinstance(field.default, bool):
                parser.add_argument(name, action=argparse.BooleanOptionalAction, default=field.default)
            else:
                parser.add_argument(name, type=type(field.default), default=field.default)
        else:
            parser.add_argument(name, required=True)
    values = vars(parser.parse_args())
    return TrainConfig(**values)


def distributed_setup() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def load_frozen_vae(config: TrainConfig, device: torch.device) -> nn.Module:
    sys.path.insert(0, config.bagel_root)
    from modeling.autoencoder import load_ae

    vae, _ = load_ae(str(Path(config.bagel_model_path) / "ae.safetensors"))
    vae.reg.sample = config.posterior_mode == "sample"
    vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
    return vae


def load_native_projection(config: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path = str(Path(config.bagel_model_path) / "ema.safetensors")
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("vae2llm.weight")
        bias = handle.get_tensor("vae2llm.bias")
    if weight.shape[1] != 64:
        raise RuntimeError(f"unexpected BAGEL vae2llm shape: {tuple(weight.shape)}")
    return weight.to(device=device, dtype=torch.bfloat16), bias.to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def encode_images(vae: nn.Module, images: torch.Tensor, posterior_mode: str) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        moments = vae.encoder(images.to(dtype=torch.bfloat16))
        mean, logvar = torch.chunk(moments, 2, dim=1)
        if posterior_mode == "mean":
            latent = mean
        elif posterior_mode == "sample":
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            raise ValueError(f"unsupported posterior mode: {posterior_mode}")
        latent = vae.scale_factor * (latent - vae.shift_factor)
    return latent.float()


def native_content(
    patches: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return F.linear(patches.to(weight.dtype), weight, bias)


def learning_rate_multiplier(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return max(1, step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    floor = config.min_learning_rate / config.learning_rate
    return floor + (1.0 - floor) * cosine


def reduce_sums(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / "complete.json").is_file():
            continue
        try:
            candidates.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    rank: int,
    world_size: int,
    step: int,
    epoch: int,
    next_batch_in_epoch: int,
    best_val: float,
) -> None:
    temporary = output_dir / f".checkpoint-{step}.tmp"
    final = output_dir / f"checkpoint-{step}"
    if rank == 0:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    barrier(world_size)
    torch.save(
        {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
        temporary / f"rng_rank{rank:05d}.pt",
    )
    barrier(world_size)
    if rank == 0:
        save_file(model_state(model), str(temporary / "flexae.safetensors"))
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        state = {
            "global_step": step,
            "epoch": epoch,
            "next_batch_in_epoch": next_batch_in_epoch,
            "best_val_patch_nmse": best_val,
        }
        (temporary / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "complete.json").write_text(
            json.dumps({"step": step, "files": files}, indent=2, sort_keys=True) + "\n"
        )
        if final.exists():
            raise RuntimeError(f"refusing to overwrite checkpoint: {final}")
        temporary.rename(final)
        (output_dir / "latest").write_text(final.name + "\n")
    barrier(world_size)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
) -> dict[str, Any]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    missing, unexpected = module.load_state_dict(load_file(str(path / "flexae.safetensors")), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
    rng = torch.load(path / f"rng_rank{rank:05d}.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return json.loads((path / "trainer_state.json").read_text())


@torch.no_grad()
def validate(
    model: nn.Module,
    vae: nn.Module,
    loader: DataLoader,
    native_weight: torch.Tensor,
    native_bias: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    rank: int,
    world_size: int,
    step: int,
    output_dir: Path,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(11, device=device, dtype=torch.float64)
    visual_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        latent = encode_images(vae, images, config.posterior_mode)
        patches = patchify_latents(latent)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(patches)
            reconstructed = result["reconstructed_patches"].float()
            shuffled = (model.module if isinstance(model, DistributedDataParallel) else model).decode(
                result["g_tokens"].roll(1, dims=0)
            ).float()
            zero = (model.module if isinstance(model, DistributedDataParallel) else model).decode(
                torch.zeros_like(result["g_tokens"])
            ).float()
        target_energy = patches.square().mean(dim=(1, 2)).clamp_min(1e-8)
        local_target = group_local_patch_targets(patches).detach()
        local_error = (result["local_prediction"].float() - local_target).square().mean(dim=-1)
        local_energy = local_target.square().mean(dim=-1).clamp_min(1e-8)
        real_mse = (reconstructed - patches).square().mean(dim=(1, 2))
        shuffle_mse = (shuffled - patches).square().mean(dim=(1, 2))
        zero_mse = (zero - patches).square().mean(dim=(1, 2))
        native_target = native_content(patches, native_weight, native_bias).float()
        native_recon = native_content(reconstructed, native_weight, native_bias).float()
        native_nmse = (
            (native_recon - native_target).square().mean(dim=(1, 2))
            / native_target.square().mean(dim=(1, 2)).clamp_min(1e-8)
        )
        batch_size = images.shape[0]
        totals += torch.tensor(
            [
                real_mse.sum().item(),
                (real_mse / target_energy).sum().item(),
                shuffle_mse.sum().item(),
                zero_mse.sum().item(),
                native_nmse.sum().item(),
                result["g_tokens"].float().square().mean().sqrt().item() * batch_size,
                batch_size,
                (real_mse < shuffle_mse).sum().item(),
                (real_mse < zero_mse).sum().item(),
                local_error.mean(dim=1).sum().item(),
                (local_error / local_energy).mean(dim=1).sum().item(),
            ],
            device=device,
            dtype=torch.float64,
        )
        if visual_batch is None:
            visual_batch = (images[:4], latent[:4], reconstructed[:4], shuffled[:4])
    totals = reduce_sums(totals, world_size)
    count = max(1.0, totals[6].item())
    metrics = {
        "val_patch_mse": totals[0].item() / count,
        "val_patch_nmse": totals[1].item() / count,
        "val_shuffle_mse": totals[2].item() / count,
        "val_zero_mse": totals[3].item() / count,
        "val_native_nmse": totals[4].item() / count,
        "val_g_rms": totals[5].item() / count,
        "val_real_better_shuffle": totals[7].item() / count,
        "val_real_better_zero": totals[8].item() / count,
        "val_local_mse": totals[9].item() / count,
        "val_local_nmse": totals[10].item() / count,
    }
    if rank == 0 and visual_batch is not None:
        images, latent, reconstructed, shuffled = visual_batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vae_recon = vae.decode(latent.to(torch.bfloat16)).float()
            compressed = vae.decode(unpatchify_latents(reconstructed).to(torch.bfloat16)).float()
            shuffled_image = vae.decode(unpatchify_latents(shuffled).to(torch.bfloat16)).float()
        image_mse = (compressed - images).square().mean().item()
        vae_mse = (vae_recon - images).square().mean().item()
        metrics["val_image_psnr"] = 10.0 * math.log10(4.0 / max(image_mse, 1e-12))
        metrics["val_vae_upper_psnr"] = 10.0 * math.log10(4.0 / max(vae_mse, 1e-12))
        rows = []
        for index in range(images.shape[0]):
            rows.extend([images[index], vae_recon[index], compressed[index], shuffled_image[index]])
        grid = make_grid(torch.stack(rows).add(1).div(2).clamp(0, 1), nrow=4)
        save_image(grid, output_dir / f"reconstruction_step{step:06d}.png")
    model.train()
    return metrics


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    config = parse_args()
    rank, local_rank, world_size, device = distributed_setup()
    seed_everything(config.seed, rank)
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    barrier(world_size)

    train_dataset = ManifestImageDataset(
        config.train_manifest,
        image_size=config.image_size,
        limit=config.train_limit,
        deduplicate=config.deduplicate_images,
    )
    val_dataset = ManifestImageDataset(
        config.val_manifest,
        image_size=config.image_size,
        limit=config.val_limit,
        deduplicate=config.deduplicate_images,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        sampler=val_sampler,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    vae = load_frozen_vae(config, device)
    native_weight, native_bias = load_native_projection(config, device)
    model = BagelFlexAE256(
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        ffn_size=config.ffn_size,
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(step, config),
    )

    global_step = 0
    epoch = 0
    resume_batch = 0
    best_val = float("inf")
    checkpoint_path = latest_checkpoint(output_dir)
    if checkpoint_path is not None:
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank)
        global_step = int(state["global_step"])
        epoch = int(state["epoch"])
        resume_batch = int(state["next_batch_in_epoch"])
        best_val = float(state["best_val_patch_nmse"])
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "startup",
                    "world_size": world_size,
                    "train_images": len(train_dataset),
                    "val_images": len(val_dataset),
                    "global_batch": world_size * config.micro_batch_size * config.gradient_accumulation_steps,
                    "resume": str(checkpoint_path) if checkpoint_path else None,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    running = torch.zeros(5, device=device, dtype=torch.float64)
    last_time = time.perf_counter()
    while global_step < config.max_steps:
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if batch_index < resume_batch:
                continue
            images = batch["image"].to(device, non_blocking=True)
            latent = encode_images(vae, images, config.posterior_mode)
            patches = patchify_latents(latent)
            sync_step = accumulation + 1 == config.gradient_accumulation_steps
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not sync_step
                else torch.enable_grad()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(patches)
                    reconstructed = result["reconstructed_patches"]
                    patch_mse = (reconstructed.float() - patches).square().mean()
                    target_native = native_content(patches, native_weight, native_bias)
                    reconstructed_native = native_content(reconstructed, native_weight, native_bias)
                    native_nmse = (
                        (reconstructed_native.float() - target_native.float()).square().mean()
                        / target_native.float().square().mean().detach().clamp_min(1e-8)
                    )
                    local_target = group_local_patch_targets(patches).detach()
                    local_error = (
                        result["local_prediction"].float() - local_target
                    ).square().mean(dim=-1)
                    local_energy = local_target.square().mean(dim=-1).detach().clamp_min(1e-8)
                    local_nmse = (local_error / local_energy).mean()
                    loss = (
                        patch_mse
                        + config.native_loss_weight * native_nmse
                        + config.local_loss_weight * local_nmse
                    )
                    scaled_loss = loss / config.gradient_accumulation_steps
                scaled_loss.backward()
            accumulation += 1
            running += torch.tensor(
                [loss.item(), patch_mse.item(), native_nmse.item(), local_nmse.item(), 1.0],
                device=device,
                dtype=torch.float64,
            )
            if not sync_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            global_step += 1
            next_batch = batch_index + 1
            checkpoint_epoch = epoch
            if next_batch >= len(train_loader):
                next_batch = 0
                checkpoint_epoch = epoch + 1
            if global_step % config.logging_steps == 0:
                reduced = reduce_sums(running.clone(), world_size)
                elapsed = time.perf_counter() - last_time
                payload = {
                    "event": "train",
                    "step": global_step,
                    "loss": reduced[0].item() / max(1.0, reduced[4].item()),
                    "patch_mse": reduced[1].item() / max(1.0, reduced[4].item()),
                    "native_nmse": reduced[2].item() / max(1.0, reduced[4].item()),
                    "local_nmse": reduced[3].item() / max(1.0, reduced[4].item()),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "step_seconds": elapsed / config.logging_steps,
                }
                if rank == 0:
                    append_jsonl(output_dir / "train_history.jsonl", payload)
                    print(json.dumps(payload, sort_keys=True), flush=True)
                running.zero_()
                last_time = time.perf_counter()
            if global_step % config.eval_steps == 0 or global_step == config.max_steps:
                metrics = validate(
                    model,
                    vae,
                    val_loader,
                    native_weight,
                    native_bias,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    output_dir,
                )
                if rank == 0:
                    payload = {"event": "validation", "step": global_step, **metrics}
                    append_jsonl(output_dir / "val_history.jsonl", payload)
                    print(json.dumps(payload, sort_keys=True), flush=True)
                    best_val = min(best_val, metrics["val_patch_nmse"])
                best_tensor = torch.tensor(best_val, device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.broadcast(best_tensor, src=0)
                best_val = best_tensor.item()
            if global_step % config.save_steps == 0 or global_step == config.max_steps:
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    rank=rank,
                    world_size=world_size,
                    step=global_step,
                    epoch=checkpoint_epoch,
                    next_batch_in_epoch=next_batch,
                    best_val=best_val,
                )
            if global_step >= config.max_steps:
                break
        epoch += 1
        resume_batch = 0
    barrier(world_size)
    if rank == 0:
        print(json.dumps({"event": "complete", "step": global_step}), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
