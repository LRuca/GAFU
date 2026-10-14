from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.utils.data import DataLoader, DistributedSampler


@dataclass(frozen=True)
class Args:
    flexae_root: str
    bagel_root: str
    bagel_model: str
    train_manifest: str
    output: str
    micro_batch_size: int = 2
    audit_batches: int = 1
    num_workers: int = 2
    seed: int = 20260806
    image_size: int = 512
    vit_image_size: int = 378
    native_loss_weight: float = 0.25
    binding_loss_weight: float = 1.0
    teacher_space: str = "input"


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flexae-root", required=True)
    parser.add_argument("--bagel-root", required=True)
    parser.add_argument("--bagel-model", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--audit-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--vit-image-size", type=int, default=378)
    parser.add_argument("--native-loss-weight", type=float, default=0.25)
    parser.add_argument("--binding-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-space", choices=("content", "input"), default="input")
    values = vars(parser.parse_args())
    return Args(**values)


def setup_distributed(seed: int) -> tuple[int, int, int, torch.device]:
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


class FrozenBagelVisionTeacher(nn.Module):
    """Only BAGEL's frozen SigLIP, connector, and LLM-input position table."""

    def __init__(self, bagel_root: str, model_path: str, device: torch.device) -> None:
        super().__init__()
        if bagel_root not in sys.path:
            sys.path.insert(0, bagel_root)
        from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
        from modeling.bagel.modeling_utils import MLPconnector, PositionEmbedding

        config = SiglipVisionConfig.from_json_file(str(Path(model_path) / "vit_config.json"))
        config.rope = False
        config.num_hidden_layers -= 1
        self.patch_size = int(config.patch_size)
        self.max_patches_per_side = 70
        self.vit = SiglipVisionModel(config)
        embeddings = self.vit.vision_model.embeddings
        if hasattr(embeddings, "convert_conv2d_to_linear"):
            embeddings.convert_conv2d_to_linear(config, meta=False)
        self.connector = MLPconnector(config.hidden_size, 3584, "gelu_pytorch_tanh")
        self.position = PositionEmbedding(self.max_patches_per_side, 3584)

        checkpoint = str(Path(model_path) / "ema.safetensors")
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            vit_state = {
                key[len("vit_model.") :]: handle.get_tensor(key)
                for key in handle.keys()
                if key.startswith("vit_model.")
            }
            connector_state = {
                key[len("connector.") :]: handle.get_tensor(key)
                for key in handle.keys()
                if key.startswith("connector.")
            }
            position_state = {
                key[len("vit_pos_embed.") :]: handle.get_tensor(key)
                for key in handle.keys()
                if key.startswith("vit_pos_embed.")
            }
        self.vit.load_state_dict(vit_state, strict=True)
        self.connector.load_state_dict(connector_state, strict=True)
        self.position.load_state_dict(position_state, strict=True)
        self.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)

    def position_ids(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        from data.data_utils import get_flattened_position_ids_extrapolate

        return get_flattened_position_ids_extrapolate(
            height,
            width,
            self.patch_size,
            self.max_patches_per_side,
        ).to(device)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, size: int) -> dict[str, torch.Tensor]:
        from data.data_utils import patchify

        if size % self.patch_size:
            raise ValueError(f"ViT image size {size} is not divisible by patch {self.patch_size}")
        resized = F.interpolate(
            images.float(),
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1, 1)
        batch = resized.shape[0]
        one_position = self.position_ids(size, size, resized.device)
        tokens_per_image = one_position.numel()
        positions = one_position.repeat(batch)
        pixels = torch.cat([patchify(image, self.patch_size) for image in resized], dim=0)
        pixels = pixels.to(
            dtype=self.vit.vision_model.embeddings.patch_embedding.weight.dtype,
        )
        sequence_lengths = torch.full(
            (batch,), tokens_per_image, dtype=torch.int32, device=resized.device
        )
        cumulative = F.pad(torch.cumsum(sequence_lengths, dim=0), (1, 0)).to(
            dtype=torch.int32
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.vit(
                packed_pixel_values=pixels,
                packed_flattened_position_ids=positions,
                cu_seqlens=cumulative,
                max_seqlen=tokens_per_image,
            )
            content = self.connector(hidden)
            position = self.position(positions).to(content.dtype)
        grid = size // self.patch_size
        return {
            "content": content.reshape(batch, grid, grid, -1).float(),
            "input": (content + position).reshape(batch, grid, grid, -1).float(),
            "position": position.reshape(batch, grid, grid, -1).float(),
        }

    @torch.inference_mode()
    def g_position(self, device: torch.device) -> torch.Tensor:
        size = 16 * self.patch_size
        ids = self.position_ids(size, size, device)
        return self.position(ids).reshape(1, 16, 16, -1).float()


def pool_nodes(tokens: torch.Tensor, size: int) -> torch.Tensor:
    if tokens.ndim == 3:
        side = int(math.isqrt(tokens.shape[1]))
        if side * side != tokens.shape[1]:
            raise ValueError(f"token count is not square: {tokens.shape[1]}")
        tokens = tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1])
    values = tokens.permute(0, 3, 1, 2)
    values = F.adaptive_avg_pool2d(values, (size, size))
    return values.permute(0, 2, 3, 1).reshape(tokens.shape[0], size * size, -1)


def gather_nodes(nodes: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return nodes
    return torch.cat(tuple(differentiable_all_gather(nodes)), dim=0)


def relation_values(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    nodes = F.normalize(nodes.float(), dim=-1)
    similarity = nodes @ nodes.T
    values = similarity[mask]
    return values - values.mean()


def relation_distance(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    student_values = relation_values(student, mask)
    teacher_values = relation_values(teacher.detach(), mask)
    alignment = (student_values * teacher_values).sum() / (
        student_values.norm() * teacher_values.norm()
    ).clamp_min(1e-8)
    return 1.0 - alignment


def relation_losses(
    g_input: torch.Tensor,
    teacher: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    global_g = gather_nodes(pool_nodes(g_input, 1).squeeze(1), world_size)
    global_t = gather_nodes(pool_nodes(teacher, 1).squeeze(1), world_size)
    batch = global_g.shape[0]
    global_mask = ~torch.eye(batch, dtype=torch.bool, device=global_g.device)
    global_loss = relation_distance(global_g, global_t, global_mask)

    local_g = gather_nodes(pool_nodes(g_input, 4), world_size)
    local_t = gather_nodes(pool_nodes(teacher, 4), world_size)
    flat_g = local_g.reshape(batch * 16, -1)
    flat_t = local_t.reshape(batch * 16, -1)
    image_ids = torch.arange(batch, device=flat_g.device).repeat_interleave(16)
    diagonal = torch.eye(batch * 16, dtype=torch.bool, device=flat_g.device)
    intra_mask = image_ids[:, None].eq(image_ids[None, :]) & ~diagonal
    inter_mask = image_ids[:, None].ne(image_ids[None, :])
    intra = relation_distance(flat_g, flat_t, intra_mask)
    inter = relation_distance(flat_g, flat_t, inter_mask)
    local_loss = 0.5 * (intra + inter)
    return global_loss, local_loss, {
        "global_g": global_g,
        "global_t": global_t,
        "local_g": local_g,
        "local_t": local_t,
        "global_mask": global_mask,
        "intra_mask": intra_mask,
        "inter_mask": inter_mask,
    }


def spatial_relation_loss(
    g_input: torch.Tensor,
    teacher: torch.Tensor,
    size: int,
    world_size: int,
) -> torch.Tensor:
    """Match the complete intra- and inter-image relation graph at one grid scale."""
    if size < 2:
        raise ValueError("spatial relation scale must be at least 2")
    g_nodes = gather_nodes(pool_nodes(g_input, size), world_size)
    t_nodes = gather_nodes(pool_nodes(teacher, size), world_size)
    batch, node_count, width = g_nodes.shape
    flat_g = g_nodes.reshape(batch * node_count, width)
    flat_t = t_nodes.reshape(batch * node_count, width)
    image_ids = torch.arange(batch, device=flat_g.device).repeat_interleave(node_count)
    diagonal = torch.eye(batch * node_count, dtype=torch.bool, device=flat_g.device)
    intra_mask = image_ids[:, None].eq(image_ids[None, :]) & ~diagonal
    inter_mask = image_ids[:, None].ne(image_ids[None, :])
    intra = relation_distance(flat_g, flat_t, intra_mask)
    inter = relation_distance(flat_g, flat_t, inter_mask)
    return 0.5 * (intra + inter)


def multiscale_relation_losses(
    g_input: torch.Tensor,
    teacher: torch.Tensor,
    local_scales: tuple[int, ...],
    world_size: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    global_g = gather_nodes(pool_nodes(g_input, 1).squeeze(1), world_size)
    global_t = gather_nodes(pool_nodes(teacher, 1).squeeze(1), world_size)
    batch = global_g.shape[0]
    global_mask = ~torch.eye(batch, dtype=torch.bool, device=global_g.device)
    global_loss = relation_distance(global_g, global_t, global_mask)
    local = {
        size: spatial_relation_loss(g_input, teacher, size, world_size)
        for size in local_scales
    }
    return global_loss, local


def vector_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(1e-12)).item())


def cosine_summary(nodes: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    values = (F.normalize(nodes.float(), dim=-1) @ F.normalize(nodes.float(), dim=-1).T)[mask]
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], device=values.device))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def teacher_audit(
    features: dict[str, torch.Tensor],
    g_position: torch.Tensor,
    world_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pooled: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in ("content", "input", "position"):
        global_nodes = gather_nodes(pool_nodes(features[name], 1).squeeze(1), world_size)
        local_nodes = gather_nodes(pool_nodes(features[name], 4), world_size)
        pooled[name] = global_nodes, local_nodes
    batch = pooled["input"][0].shape[0]
    global_mask = ~torch.eye(batch, dtype=torch.bool, device=g_position.device)
    image_ids = torch.arange(batch, device=g_position.device).repeat_interleave(16)
    region_ids = torch.arange(16, device=g_position.device).repeat(batch)
    diagonal = torch.eye(batch * 16, dtype=torch.bool, device=g_position.device)
    intra = image_ids[:, None].eq(image_ids[None, :]) & ~diagonal
    inter = image_ids[:, None].ne(image_ids[None, :])
    same_position = inter & region_ids[:, None].eq(region_ids[None, :])
    different_position = inter & region_ids[:, None].ne(region_ids[None, :])
    for name, (global_nodes, local_nodes) in pooled.items():
        flat = local_nodes.reshape(batch * 16, -1)
        result[name] = {
            "global": cosine_summary(global_nodes, global_mask),
            "local_intra": cosine_summary(flat, intra),
            "local_inter": cosine_summary(flat, inter),
            "local_inter_same_position": cosine_summary(flat, same_position),
            "local_inter_different_position": cosine_summary(flat, different_position),
        }
    input_flat = pooled["input"][1].reshape(batch * 16, -1)
    content_flat = pooled["content"][1].reshape(batch * 16, -1)
    position_flat = pooled["position"][1].reshape(batch * 16, -1)
    input_relation = F.normalize(input_flat.float(), dim=-1) @ F.normalize(input_flat.float(), dim=-1).T
    content_relation = F.normalize(content_flat.float(), dim=-1) @ F.normalize(content_flat.float(), dim=-1).T
    position_relation = F.normalize(position_flat.float(), dim=-1) @ F.normalize(position_flat.float(), dim=-1).T
    result["relation_correlations"] = {
        "input_vs_content_intra": vector_correlation(input_relation[intra], content_relation[intra]),
        "input_vs_position_intra": vector_correlation(input_relation[intra], position_relation[intra]),
        "input_vs_content_inter": vector_correlation(input_relation[inter], content_relation[inter]),
        "input_vs_position_inter": vector_correlation(input_relation[inter], position_relation[inter]),
    }
    g_pos4 = pool_nodes(g_position, 4).squeeze(0)
    result["g_position_4x4_rms"] = float(g_pos4.square().mean().sqrt())
    return result


def allreduce_grads(
    grads: Iterable[torch.Tensor | None], world_size: int
) -> tuple[torch.Tensor | None, ...]:
    output = []
    for grad in grads:
        if grad is None:
            output.append(None)
            continue
        value = grad.detach().clone()
        if world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(world_size)
        output.append(value)
    return tuple(output)


def gradients(
    loss: torch.Tensor,
    parameters: list[nn.Parameter],
    world_size: int,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return allreduce_grads(values, world_size)


def module_group(name: str) -> str:
    for prefix in (
        "source_embed",
        "encoder_query",
        "encoder",
        "g_projector",
        "local_head",
        "g_to_decoder",
        "decoder_query",
        "decoder",
        "patch_head",
    ):
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return "other"


def grad_stats(
    names: list[str], grads: tuple[torch.Tensor | None, ...]
) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    maximum: dict[str, float] = defaultdict(float)
    nonzero = 0
    finite = True
    for name, grad in zip(names, grads):
        if grad is None:
            continue
        group = module_group(name)
        totals[group] += float(grad.double().square().sum())
        maximum[group] = max(maximum[group], float(grad.abs().max()))
        nonzero += int(torch.count_nonzero(grad).item())
        finite = finite and bool(torch.isfinite(grad).all())
    return {
        "norm": math.sqrt(sum(totals.values())),
        "module_norms": {key: math.sqrt(value) for key, value in sorted(totals.items())},
        "module_max_abs": dict(sorted(maximum.items())),
        "nonzero_elements": nonzero,
        "finite": finite,
    }


def grad_dot(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    names: list[str],
    *,
    shared_only: bool,
) -> tuple[float, float, float]:
    dot = left_norm = right_norm = 0.0
    shared = {"source_embed", "encoder_query", "encoder", "g_projector"}
    for name, lgrad, rgrad in zip(names, left, right):
        if lgrad is None or rgrad is None:
            continue
        if shared_only and module_group(name) not in shared:
            continue
        dot += float((lgrad.double() * rgrad.double()).sum())
        left_norm += float(lgrad.double().square().sum())
        right_norm += float(rgrad.double().square().sum())
    return dot, math.sqrt(left_norm), math.sqrt(right_norm)


def grad_cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    names: list[str],
) -> float:
    dot, left_norm, right_norm = grad_dot(left, right, names, shared_only=True)
    return dot / max(1e-30, left_norm * right_norm)


def candidate_weights(
    base: tuple[torch.Tensor | None, ...],
    global_grad: tuple[torch.Tensor | None, ...],
    local_grad: tuple[torch.Tensor | None, ...],
    names: list[str],
) -> dict[str, Any]:
    _, base_norm, _ = grad_dot(base, base, names, shared_only=True)
    _, global_norm, _ = grad_dot(global_grad, global_grad, names, shared_only=True)
    _, local_norm, _ = grad_dot(local_grad, local_grad, names, shared_only=True)
    global_local_dot, _, _ = grad_dot(global_grad, local_grad, names, shared_only=True)
    output = {}
    for target in (0.10, 0.25, 0.50):
        lambda_global = target * base_norm / max(1e-30, 3.0 * global_norm)
        lambda_local = 2.0 * target * base_norm / max(1e-30, 3.0 * local_norm)
        combined_sq = (
            (lambda_global * global_norm) ** 2
            + (lambda_local * local_norm) ** 2
            + 2.0 * lambda_global * lambda_local * global_local_dot
        )
        actual = math.sqrt(max(0.0, combined_sq)) / max(1e-30, base_norm)
        correction = target / max(1e-30, actual)
        lambda_global *= correction
        lambda_local *= correction
        output[str(target)] = {
            "lambda_global": lambda_global,
            "lambda_local": lambda_local,
            "pre_correction_ratio": actual,
            "corrected_target_ratio": target,
        }
    return {
        "shared_base_norm": base_norm,
        "shared_global_norm": global_norm,
        "shared_local_norm": local_norm,
        "weights": output,
    }


def invariance_audit(
    graph: dict[str, torch.Tensor],
) -> dict[str, float]:
    global_g = graph["global_g"]
    global_t = graph["global_t"]
    local_g = graph["local_g"]
    local_t = graph["local_t"]
    global_mask = graph["global_mask"]
    intra_mask = graph["intra_mask"]
    inter_mask = graph["inter_mask"]
    baseline_global = relation_distance(global_g, global_t, global_mask)
    flat_g = local_g.flatten(0, 1)
    flat_t = local_t.flatten(0, 1)
    baseline_local = 0.5 * (
        relation_distance(flat_g, flat_t, intra_mask)
        + relation_distance(flat_g, flat_t, inter_mask)
    )
    permutation = torch.arange(global_g.shape[-1] - 1, -1, -1, device=global_g.device)
    signs = torch.ones(global_g.shape[-1], device=global_g.device)
    signs[1::2] = -1
    transformed_global = global_g[:, permutation] * signs
    transformed_local = flat_g[:, permutation] * signs
    rotated_global = relation_distance(transformed_global, global_t, global_mask)
    rotated_local = 0.5 * (
        relation_distance(transformed_local, flat_t, intra_mask)
        + relation_distance(transformed_local, flat_t, inter_mask)
    )
    shuffled_global_t = global_t.roll(1, dims=0)
    shuffled_local_t = local_t.roll(1, dims=0).flatten(0, 1)
    shuffled_global = relation_distance(global_g, shuffled_global_t, global_mask)
    shuffled_local = 0.5 * (
        relation_distance(flat_g, shuffled_local_t, intra_mask)
        + relation_distance(flat_g, shuffled_local_t, inter_mask)
    )
    region_permutation = torch.arange(15, -1, -1, device=local_t.device)
    region_shuffled_t = local_t[:, region_permutation].flatten(0, 1)
    region_shuffled = 0.5 * (
        relation_distance(flat_g, region_shuffled_t, intra_mask)
        + relation_distance(flat_g, region_shuffled_t, inter_mask)
    )
    collapsed_global = global_g.mean(dim=0, keepdim=True).expand_as(global_g)
    collapsed_local = flat_g.mean(dim=0, keepdim=True).expand_as(flat_g)
    collapsed_global_loss = relation_distance(collapsed_global, global_t, global_mask)
    collapsed_local_loss = 0.5 * (
        relation_distance(collapsed_local, flat_t, intra_mask)
        + relation_distance(collapsed_local, flat_t, inter_mask)
    )
    collapsed_global_leaf = collapsed_global.detach().clone().requires_grad_(True)
    collapsed_local_leaf = collapsed_local.detach().clone().requires_grad_(True)
    collapsed_objective = relation_distance(collapsed_global_leaf, global_t, global_mask) + 0.5 * (
        relation_distance(collapsed_local_leaf, flat_t, intra_mask)
        + relation_distance(collapsed_local_leaf, flat_t, inter_mask)
    )
    collapsed_grads = torch.autograd.grad(
        collapsed_objective,
        (collapsed_global_leaf, collapsed_local_leaf),
    )
    teacher_self_global = relation_distance(global_t, global_t, global_mask)
    teacher_self_local = 0.5 * (
        relation_distance(flat_t, flat_t, intra_mask)
        + relation_distance(flat_t, flat_t, inter_mask)
    )
    teacher_shuffle_global = relation_distance(global_t, shuffled_global_t, global_mask)
    teacher_shuffle_local = 0.5 * (
        relation_distance(flat_t, shuffled_local_t, intra_mask)
        + relation_distance(flat_t, shuffled_local_t, inter_mask)
    )
    teacher_region_shuffle = 0.5 * (
        relation_distance(flat_t, region_shuffled_t, intra_mask)
        + relation_distance(flat_t, region_shuffled_t, inter_mask)
    )
    return {
        "baseline_global": float(baseline_global),
        "baseline_local": float(baseline_local),
        "orthogonal_delta_global": float((rotated_global - baseline_global).abs()),
        "orthogonal_delta_local": float((rotated_local - baseline_local).abs()),
        "scale_delta_global": float((relation_distance(global_g * 7.0, global_t, global_mask) - baseline_global).abs()),
        "scale_delta_local": float((0.5 * (relation_distance(flat_g * 7.0, flat_t, intra_mask) + relation_distance(flat_g * 7.0, flat_t, inter_mask)) - baseline_local).abs()),
        "image_shuffle_global": float(shuffled_global),
        "image_shuffle_local": float(shuffled_local),
        "region_shuffle_local": float(region_shuffled),
        "collapsed_global": float(collapsed_global_loss),
        "collapsed_local": float(collapsed_local_loss),
        "collapsed_global_grad_norm": float(collapsed_grads[0].float().norm()),
        "collapsed_local_grad_norm": float(collapsed_grads[1].float().norm()),
        "teacher_self_global": float(teacher_self_global),
        "teacher_self_local": float(teacher_self_local),
        "teacher_image_shuffle_global": float(teacher_shuffle_global),
        "teacher_image_shuffle_local": float(teacher_shuffle_local),
        "teacher_region_shuffle_local": float(teacher_region_shuffle),
    }


def main() -> None:
    args = parse_args()
    rank, _, world_size, device = setup_distributed(args.seed)
    for path in (args.flexae_root, args.bagel_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    from flexae_model import BagelFlexAE256, group_local_patch_targets, patchify_latents
    from train_flexae import (
        ManifestImageDataset,
        TrainConfig,
        encode_images,
        load_frozen_vae,
        load_native_projection,
        native_content,
    )

    dataset_kwargs = {
        "image_size": args.image_size,
        "limit": max(64, args.micro_batch_size * world_size * args.audit_batches),
        "deduplicate": True,
    }
    try:
        dataset = ManifestImageDataset(
            args.train_manifest,
            verify_paths=True,
            **dataset_kwargs,
        )
    except TypeError as error:
        if "verify_paths" not in str(error):
            raise
        dataset = ManifestImageDataset(args.train_manifest, **dataset_kwargs)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    config = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.train_manifest,
        output_dir=str(Path(args.output).parent),
        bagel_root=args.bagel_root,
        bagel_model_path=args.bagel_model,
    )
    vae = load_frozen_vae(config, device)
    native_weight, native_bias = load_native_projection(config, device)
    teacher = FrozenBagelVisionTeacher(args.bagel_root, args.bagel_model, device)
    model = BagelFlexAE256(gradient_checkpointing=False).to(device)
    g_position = teacher.g_position(device)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]

    reports = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.audit_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        latent = encode_images(vae, images, "mean")
        patches = patchify_latents(latent)
        teacher_features = teacher(images, args.vit_image_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_result = model(patches)
            reconstructed = model_result["reconstructed_patches"]
        g_input = model_result["g_tokens"].float().reshape(images.shape[0], 16, 16, -1) + g_position
        teacher_target = teacher_features[args.teacher_space]
        global_loss, local_loss, graph = relation_losses(g_input, teacher_target, world_size)

        patch_mse = (reconstructed.float() - patches).square().mean()
        target_native = native_content(patches, native_weight, native_bias)
        reconstructed_native = native_content(reconstructed, native_weight, native_bias)
        native_nmse = (
            (reconstructed_native.float() - target_native.float()).square().mean()
            / target_native.float().square().mean().detach().clamp_min(1e-8)
        )
        local_target = group_local_patch_targets(patches).detach()
        local_error = (model_result["local_prediction"].float() - local_target).square().mean(dim=-1)
        local_energy = local_target.square().mean(dim=-1).detach().clamp_min(1e-8)
        binding_nmse = (local_error / local_energy).mean()
        base_loss = (
            patch_mse
            + args.native_loss_weight * native_nmse
            + args.binding_loss_weight * binding_nmse
        )

        base_grad = gradients(base_loss, parameters, world_size, retain_graph=True)
        global_grad = gradients(global_loss, parameters, world_size, retain_graph=True)
        local_grad = gradients(local_loss, parameters, world_size, retain_graph=False)
        report = {
            "batch_index": batch_index,
            "global_batch": images.shape[0] * world_size,
            "losses": {
                "patch_mse": float(patch_mse),
                "native_nmse": float(native_nmse),
                "binding_nmse": float(binding_nmse),
                "base": float(base_loss),
                "global_relation": float(global_loss),
                "local_relation": float(local_loss),
            },
            "teacher": teacher_audit(teacher_features, g_position, world_size),
            "invariance": invariance_audit(graph),
            "gradients": {
                "base": grad_stats(names, base_grad),
                "global": grad_stats(names, global_grad),
                "local": grad_stats(names, local_grad),
                "cosine_base_global": grad_cosine(base_grad, global_grad, names),
                "cosine_base_local": grad_cosine(base_grad, local_grad, names),
                "cosine_global_local": grad_cosine(global_grad, local_grad, names),
                "candidates": candidate_weights(base_grad, global_grad, local_grad, names),
            },
            "memory": {
                "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            },
        }
        reports.append(report)
        if rank == 0:
            print(json.dumps({"event": "batch_complete", **report}, sort_keys=True), flush=True)

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args),
            "world_size": world_size,
            "batches": reports,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "output": str(output)}, sort_keys=True), flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
