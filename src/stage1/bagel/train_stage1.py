from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid, save_image

from preflight_relational_stage1 import (
    FrozenBagelVisionTeacher,
    gather_nodes,
    multiscale_relation_losses,
    pool_nodes,
)


@dataclass(frozen=True)
class Config:
    flexae_root: str
    bagel_root: str
    bagel_model: str
    train_manifest: str
    val_manifest: str
    output_dir: str
    init_checkpoint: str = ""
    init_optimizer: str = ""
    initial_global_step: int = 0
    initial_epoch: int = 0
    initial_batch_offset: int = 0
    max_steps: int = 200
    micro_batch_size: int = 32
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    native_loss_weight: float = 0.25
    binding_loss_weight: float = 1.0
    global_relation_weight: float = 0.0
    local_relation_weight: float = 0.0
    local_relation_scales: str = "4,8,16"
    relation_ramp_steps: int = 100
    relation_initial_scale: float = 0.2
    eval_steps: int = 25
    save_steps: int = 25
    logging_steps: int = 10
    val_limit: int = 64
    num_workers: int = 4
    seed: int = 20260806
    image_size: int = 512
    vit_image_size: int = 378
    teacher_space: str = "input"


def args() -> Config:
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if field.default is MISSING:
            parser.add_argument(name, type=str, required=True)
        else:
            parser.add_argument(name, type=type(field.default), default=field.default)
    values = vars(parser.parse_args())
    if values["teacher_space"] not in {"content", "input"}:
        raise ValueError("teacher-space must be content or input")
    if values["save_steps"] != values["eval_steps"]:
        raise ValueError("transactional checkpoints require save-steps == eval-steps")
    scales = tuple(int(value) for value in values["local_relation_scales"].split(","))
    if scales != (4, 8, 16):
        raise ValueError("local-relation-scales must be exactly 4,8,16")
    return Config(**values)


def configured_local_scales(config: Config) -> tuple[int, ...]:
    return tuple(int(value) for value in config.local_relation_scales.split(","))


def coordinate_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Align content coordinates after removing the batch-wide generic direction."""
    target_side = int(math.isqrt(student.shape[1]))
    teacher_nodes = pool_nodes(teacher, target_side)
    student_nodes = student.reshape(student.shape[0], target_side * target_side, -1).float()
    student_nodes = student_nodes - student_nodes.mean(dim=(0, 1), keepdim=True)
    teacher_nodes = teacher_nodes - teacher_nodes.mean(dim=(0, 1), keepdim=True)
    return 1.0 - F.cosine_similarity(student_nodes, teacher_nodes.detach(), dim=-1).mean()


def setup(seed: int) -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(world_size)
    return value


def relation_scale(step: int, config: Config) -> float:
    if config.relation_ramp_steps <= 0:
        return 1.0
    progress = min(1.0, step / config.relation_ramp_steps)
    return config.relation_initial_scale + (1.0 - config.relation_initial_scale) * progress


def effective_rank(features: torch.Tensor) -> float:
    centered = features.double() - features.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def dataset_instance(dataset_type: type, manifest: str, **kwargs: Any):
    try:
        return dataset_type(manifest, verify_paths=True, **kwargs)
    except TypeError as error:
        if "verify_paths" not in str(error):
            raise
        return dataset_type(manifest, **kwargs)


@torch.no_grad()
def validate(
    model: nn.Module,
    vae: nn.Module,
    teacher: FrozenBagelVisionTeacher,
    loader: DataLoader,
    native_weight: torch.Tensor,
    native_bias: torch.Tensor,
    g_position: torch.Tensor,
    config: Config,
    device: torch.device,
    world_size: int,
    rank: int,
    step: int,
    output: Path,
) -> dict[str, float]:
    from flexae_model import group_local_patch_targets, patchify_latents, unpatchify_latents
    from train_flexae import encode_images, native_content

    model.eval()
    scales = configured_local_scales(config)
    sums = torch.zeros(11 + len(scales), device=device, dtype=torch.float64)
    ranks = []
    visual_rows: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        latent = encode_images(vae, images, "mean")
        patches = patchify_latents(latent)
        teacher_features = teacher(images, config.vit_image_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(patches)
        reconstructed = result["reconstructed_patches"]
        patch_error = (reconstructed.float() - patches).square().mean(dim=(1, 2))
        patch_energy = patches.square().mean(dim=(1, 2)).clamp_min(1e-8)
        native_target = native_content(patches, native_weight, native_bias)
        native_recon = native_content(reconstructed, native_weight, native_bias)
        native_nmse = (
            (native_recon.float() - native_target.float()).square().mean(dim=(1, 2))
            / native_target.float().square().mean(dim=(1, 2)).clamp_min(1e-8)
        )
        local_target = group_local_patch_targets(patches)
        local_error = (result["local_prediction"].float() - local_target).square().mean(dim=-1)
        local_energy = local_target.square().mean(dim=-1).clamp_min(1e-8)
        binding_nmse = (local_error / local_energy).mean(dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vae_images = vae.decode(latent.to(torch.bfloat16)).float()
            recon_images = vae.decode(
                unpatchify_latents(reconstructed.float()).to(torch.bfloat16)
            ).float()
        image_mse = (recon_images - images).square().mean(dim=(1, 2, 3))
        vae_mse = (vae_images - images).square().mean(dim=(1, 2, 3))
        g_tokens = result["g_tokens"].float()
        g_input = g_tokens.reshape(images.shape[0], 16, 16, -1)
        global_relation, local_relations = multiscale_relation_losses(
            g_input,
            teacher_features["content"],
            scales,
            world_size,
        )
        local_relation = torch.stack(tuple(local_relations.values())).mean()
        coord = coordinate_loss(g_tokens, teacher_features["content"])
        global_nodes = gather_nodes(pool_nodes(g_input, 1).squeeze(1), world_size)
        ranks.append(effective_rank(global_nodes))
        count = images.shape[0]
        sums += torch.tensor(
            [
                patch_error.sum().item(),
                (patch_error / patch_energy).sum().item(),
                native_nmse.sum().item(),
                binding_nmse.sum().item(),
                float(global_relation) * count,
                float(local_relation) * count,
                float(coord) * count,
                *[float(local_relations[size]) * count for size in scales],
                global_nodes.var(dim=0, unbiased=False).mean().item() * count,
                image_mse.sum().item(),
                vae_mse.sum().item(),
                count,
            ],
            dtype=torch.float64,
            device=device,
        )
        if rank == 0 and not visual_rows:
            module = model.module if isinstance(model, DistributedDataParallel) else model
            with torch.autocast("cuda", dtype=torch.bfloat16):
                shuffled = module.decode(result["g_tokens"].roll(1, dims=0)).float()
                shuffled_images = vae.decode(
                    unpatchify_latents(shuffled).to(torch.bfloat16)
                ).float()
            for index in range(min(4, images.shape[0])):
                visual_rows.extend(
                    [
                        images[index].cpu(),
                        vae_images[index].cpu(),
                        recon_images[index].cpu(),
                        shuffled_images[index].cpu(),
                    ]
                )
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = max(1.0, sums[-1].item())
    result = {
        "patch_mse": sums[0].item() / count,
        "patch_nmse": sums[1].item() / count,
        "native_nmse": sums[2].item() / count,
        "binding_nmse": sums[3].item() / count,
        "global_relation": sums[4].item() / count,
        "local_relation": sums[5].item() / count,
        "coordinate_loss": sums[6].item() / count,
        **{
            f"relation_{size}x{size}": sums[7 + index].item() / count
            for index, size in enumerate(scales)
        },
        "global_feature_variance": sums[7 + len(scales)].item() / count,
        "image_psnr": 10.0 * math.log10(
            4.0 / max(sums[8 + len(scales)].item() / count, 1e-12)
        ),
        "vae_upper_psnr": 10.0 * math.log10(
            4.0 / max(sums[9 + len(scales)].item() / count, 1e-12)
        ),
        "global_effective_rank": sum(ranks) / max(1, len(ranks)),
    }
    if rank == 0 and visual_rows:
        grid = make_grid(torch.stack(visual_rows).add(1).div(2).clamp(0, 1), nrow=4)
        save_image(grid, output / f"reconstruction_step{step:06d}.png")
    model.train()
    return result


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_transaction(
    output: Path,
    step: int,
    epoch: int,
    next_batch_index: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    checkpoint = output / f"checkpoint-{step}"
    if (checkpoint / "complete.json").is_file():
        return
    temporary = output / f".checkpoint-{step}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=False)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()},
        str(temporary / "flexae.safetensors"),
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
        },
        temporary / "optimizer.pt",
    )
    (temporary / "complete.json").write_text(
        json.dumps(
            {
                "schema": "stage1_global_local_multiscale_relation_v1",
                "step": step,
                "epoch": epoch,
                "next_batch_index": next_batch_index,
                "objective": "reconstruction_plus_global_and_local_4_8_16_relation",
                "calibrator_in_stage1": False,
                "teacher_space": "content",
                "position_added_to_relation": False,
                "local_vit_relation_in_loss": True,
                "local_relation_scales": [4, 8, 16],
                "coordinate_in_loss": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint)


def main() -> None:
    config = args()
    rank, local_rank, world_size, device = setup(config.seed)
    for path in (config.flexae_root, config.bagel_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    from flexae_model import BagelFlexAE256, group_local_patch_targets, patchify_latents
    from train_flexae import (
        ManifestImageDataset,
        TrainConfig,
        encode_images,
        learning_rate_multiplier,
        load_frozen_vae,
        load_native_projection,
        native_content,
    )

    output = Path(config.output_dir)
    if rank == 0:
        if (output / "complete.json").is_file():
            raise RuntimeError(f"refusing to overwrite complete run: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    barrier(world_size)

    train_data = dataset_instance(
        ManifestImageDataset,
        config.train_manifest,
        image_size=config.image_size,
        limit=0,
        deduplicate=True,
    )
    val_data = dataset_instance(
        ManifestImageDataset,
        config.val_manifest,
        image_size=config.image_size,
        limit=config.val_limit,
        deduplicate=True,
    )
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.micro_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.micro_batch_size,
        sampler=val_sampler,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=True,
        drop_last=True,
    )

    base_config = TrainConfig(
        train_manifest=config.train_manifest,
        val_manifest=config.val_manifest,
        output_dir=config.output_dir,
        bagel_root=config.bagel_root,
        bagel_model_path=config.bagel_model,
        max_steps=config.max_steps,
        micro_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
    )
    vae = load_frozen_vae(base_config, device)
    native_weight, native_bias = load_native_projection(base_config, device)
    teacher = FrozenBagelVisionTeacher(config.bagel_root, config.bagel_model, device)
    core_model = BagelFlexAE256(gradient_checkpointing=True)
    if config.init_checkpoint:
        checkpoint = Path(config.init_checkpoint)
        missing, unexpected = core_model.load_state_dict(load_file(str(checkpoint)), strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
            )
    model = core_model.to(device)
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
        lambda step: learning_rate_multiplier(step, base_config),
    )
    if config.init_optimizer:
        resume_state = torch.load(config.init_optimizer, map_location=device)
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        expected = (config.initial_global_step, config.initial_epoch, config.initial_batch_offset)
        actual = (
            int(resume_state["step"]),
            int(resume_state["epoch"]),
            int(resume_state["next_batch_index"]),
        )
        if actual != expected:
            raise RuntimeError(f"resume cursor mismatch: checkpoint={actual} config={expected}")
    g_position = teacher.g_position(device)

    metrics_path = output / "metrics.jsonl"
    initial = validate(
        model,
        vae,
        teacher,
        val_loader,
        native_weight,
        native_bias,
        g_position,
        config,
        device,
        world_size,
        rank,
        config.initial_global_step,
        output,
    )
    if rank == 0:
        append_jsonl(metrics_path, {"event": "validation", "step": config.initial_global_step, **initial})
        print(json.dumps({"event": "validation", "step": config.initial_global_step, **initial}, sort_keys=True), flush=True)

    global_step = config.initial_global_step
    epoch = config.initial_epoch
    resume_epoch = config.initial_epoch
    resume_batch = config.initial_batch_offset
    last_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while global_step < config.max_steps:
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == resume_epoch and batch_index < resume_batch:
                continue
            images = batch["image"].to(device, non_blocking=True)
            latent = encode_images(vae, images, "mean")
            patches = patchify_latents(latent)
            teacher_features = teacher(images, config.vit_image_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(patches)
            reconstructed = result["reconstructed_patches"]
            patch_mse = (reconstructed.float() - patches).square().mean()
            native_target = native_content(patches, native_weight, native_bias)
            native_recon = native_content(reconstructed, native_weight, native_bias)
            native_nmse = (
                (native_recon.float() - native_target.float()).square().mean()
                / native_target.float().square().mean().detach().clamp_min(1e-8)
            )
            local_target = group_local_patch_targets(patches).detach()
            local_error = (result["local_prediction"].float() - local_target).square().mean(dim=-1)
            local_energy = local_target.square().mean(dim=-1).detach().clamp_min(1e-8)
            binding_nmse = (local_error / local_energy).mean()
            g_tokens = result["g_tokens"].float()
            g_input = g_tokens.reshape(images.shape[0], 16, 16, -1)
            scales = configured_local_scales(config)
            global_relation, local_relations = multiscale_relation_losses(
                g_input,
                teacher_features["content"],
                scales,
                world_size,
            )
            local_relation = torch.stack(tuple(local_relations.values())).mean()
            ramp = relation_scale(global_step, config)
            base_loss = (
                patch_mse
                + config.native_loss_weight * native_nmse
                + config.binding_loss_weight * binding_nmse
            )
            weighted_global = ramp * config.global_relation_weight * global_relation
            weighted_local = ramp * config.local_relation_weight * local_relation
            relation_loss = weighted_global + weighted_local
            loss = base_loss + relation_loss

            if global_step == 0:
                trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
                component_norms = {}
                for name, component in (
                    ("base", base_loss),
                    ("global_relation", weighted_global),
                    ("local_relation", weighted_local),
                    ("combined_relation", relation_loss),
                ):
                    grads = torch.autograd.grad(component, trainable, retain_graph=True, allow_unused=True)
                    squared = sum(
                        grad.float().square().sum() for grad in grads if grad is not None
                    )
                    component_norms[name] = float(squared.sqrt())
                if rank == 0:
                    append_jsonl(metrics_path, {"event": "gradient_audit", **component_norms})
                    print(json.dumps({"event": "gradient_audit", **component_norms}), flush=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.logging_steps == 0 or global_step == 1:
                now = time.perf_counter()
                values = torch.tensor(
                    [
                        float(patch_mse),
                        float(native_nmse),
                        float(binding_nmse),
                        float(global_relation),
                        float(local_relation),
                        float(base_loss),
                        float(weighted_global),
                        float(weighted_local),
                        float(relation_loss),
                        float(grad_norm),
                        *[float(local_relations[size]) for size in scales],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                values = reduce_mean(values, world_size)
                payload = {
                    "event": "train",
                    "step": global_step,
                    "patch_mse": values[0].item(),
                    "native_nmse": values[1].item(),
                    "binding_nmse": values[2].item(),
                    "global_relation": values[3].item(),
                    "local_relation": values[4].item(),
                    "base_loss": values[5].item(),
                    "weighted_global_relation": values[6].item(),
                    "weighted_local_relation": values[7].item(),
                    "weighted_relation_loss": values[8].item(),
                    "grad_norm_preclip": values[9].item(),
                    **{
                        f"relation_{size}x{size}": values[10 + index].item()
                        for index, size in enumerate(scales)
                    },
                    "relation_scale": ramp,
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds_per_step_window": (now - last_time) / (
                        1 if global_step == 1 else config.logging_steps
                    ),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                last_time = now
                if rank == 0:
                    append_jsonl(metrics_path, payload)
                    print(json.dumps(payload, sort_keys=True), flush=True)

            if global_step % config.eval_steps == 0 or global_step == config.max_steps:
                metrics = validate(
                    model,
                    vae,
                    teacher,
                    val_loader,
                    native_weight,
                    native_bias,
                    g_position,
                    config,
                    device,
                    world_size,
                    rank,
                    global_step,
                    output,
                )
                if rank == 0:
                    payload = {"event": "validation", "step": global_step, **metrics}
                    append_jsonl(metrics_path, payload)
                    print(json.dumps(payload, sort_keys=True), flush=True)
                barrier(world_size)
                if rank == 0:
                    next_batch = batch_index + 1
                    next_epoch = epoch
                    if next_batch >= len(train_loader):
                        next_epoch += 1
                        next_batch = 0
                    save_transaction(
                        output,
                        global_step,
                        next_epoch,
                        next_batch,
                        model,
                        optimizer,
                        scheduler,
                    )
                barrier(world_size)
            if global_step >= config.max_steps:
                break
        epoch += 1

    if rank == 0:
        module = model.module if isinstance(model, DistributedDataParallel) else model
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in module.state_dict().items()
            },
            str(output / "flexae_final.safetensors"),
        )
        final = {
            "alignment_mode": "direct_g",
            "global_step": global_step,
            "last_validation": metrics,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        (output / "complete.json").write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "complete", **final}, sort_keys=True), flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
