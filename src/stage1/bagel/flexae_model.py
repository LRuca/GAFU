from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def patchify_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(latents.shape)}")
    batch, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"latent shape {height}x{width} is not divisible by {patch_size}")
    return (
        latents.reshape(
            batch,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, (height // patch_size) * (width // patch_size), channels * patch_size**2)
    )


def unpatchify_latents(
    tokens: torch.Tensor,
    *,
    channels: int = 16,
    grid: tuple[int, int] = (32, 32),
    patch_size: int = 2,
) -> torch.Tensor:
    batch, length, width = tokens.shape
    grid_h, grid_w = grid
    expected_width = channels * patch_size**2
    if length != grid_h * grid_w or width != expected_width:
        raise ValueError(
            f"expected [B,{grid_h * grid_w},{expected_width}], got {tuple(tokens.shape)}"
        )
    return (
        tokens.reshape(batch, grid_h, grid_w, channels, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, grid_h * patch_size, grid_w * patch_size)
    )


def group_local_patch_targets(
    patches: torch.Tensor,
    *,
    source_grid: tuple[int, int] = (32, 32),
    region_size: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Concatenate each fixed spatial region into its corresponding G-slot target."""
    if patches.ndim != 3:
        raise ValueError(f"expected [B,L,D], got {tuple(patches.shape)}")
    batch, length, width = patches.shape
    grid_h, grid_w = source_grid
    region_h, region_w = region_size
    if length != grid_h * grid_w:
        raise ValueError(f"expected {grid_h * grid_w} patches, got {length}")
    if grid_h % region_h or grid_w % region_w:
        raise ValueError(f"source grid {source_grid} is not divisible by {region_size}")
    return (
        patches.reshape(
            batch,
            grid_h // region_h,
            region_h,
            grid_w // region_w,
            region_w,
            width,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, (grid_h // region_h) * (grid_w // region_w), -1)
    )


def grid_coordinates(height: int, width: int, *, stride: int = 1, offset: int = 0) -> torch.Tensor:
    rows = torch.arange(height, dtype=torch.float32) * stride + offset
    cols = torch.arange(width, dtype=torch.float32) * stride + offset
    y, x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1)


def _rotate_axis(values: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    axis_dim = values.shape[-1]
    if axis_dim % 2:
        raise ValueError("RoPE axis width must be even")
    inv_freq = torch.exp(
        -math.log(base)
        * torch.arange(0, axis_dim, 2, device=values.device, dtype=torch.float32)
        / axis_dim
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().to(values.dtype).unsqueeze(0).unsqueeze(0)
    sin = angles.sin().to(values.dtype).unsqueeze(0).unsqueeze(0)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def apply_2d_rope(values: torch.Tensor, coordinates: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"expected [B,H,L,D], got {tuple(values.shape)}")
    if coordinates.shape != (values.shape[2], 2):
        raise ValueError(
            f"coordinate shape must be [{values.shape[2]},2], got {tuple(coordinates.shape)}"
        )
    head_dim = values.shape[-1]
    if head_dim % 4:
        raise ValueError("2D RoPE requires head_dim divisible by four")
    half = head_dim // 2
    y = _rotate_axis(values[..., :half], coordinates[:, 0], base)
    x = _rotate_axis(values[..., half:], coordinates[:, 1], base)
    return torch.cat((y, x), dim=-1)


class JointSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (hidden_size // num_heads) % 4:
            raise ValueError("attention head width must be divisible by four")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = float(dropout)

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_2d_rope(q.transpose(1, 2), coordinates)
        k = apply_2d_rope(k.transpose(1, 2), coordinates)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, length, width))


class FlexBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = JointSelfAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_size, hidden_size),
        )
        self.dropout = float(dropout)

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = hidden + F.dropout(
            self.attn(self.attn_norm(hidden), coordinates),
            p=self.dropout,
            training=self.training,
        )
        hidden = hidden + F.dropout(
            self.ffn(self.ffn_norm(hidden)),
            p=self.dropout,
            training=self.training,
        )
        return hidden


class FlexStack(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FlexBlock(hidden_size, num_heads, ffn_size, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, coordinates, use_reentrant=False)
            else:
                hidden = layer(hidden, coordinates)
        return self.final_norm(hidden)


class BagelFlexAE256(nn.Module):
    """LoViC-style spatial latent autoencoder for BAGEL fixed-512 VAE tokens."""

    def __init__(
        self,
        *,
        patch_dim: int = 64,
        hidden_size: int = 1024,
        bagel_dim: int = 3584,
        source_grid: tuple[int, int] = (32, 32),
        g_grid: tuple[int, int] = (16, 16),
        depth: int = 6,
        num_heads: int = 8,
        ffn_size: int = 4096,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if source_grid != (g_grid[0] * 2, g_grid[1] * 2):
            raise ValueError("pilot requires a uniform 2x2 source region per G token")
        self.patch_dim = patch_dim
        self.hidden_size = hidden_size
        self.bagel_dim = bagel_dim
        self.source_grid = source_grid
        self.g_grid = g_grid
        self.num_source_tokens = source_grid[0] * source_grid[1]
        self.num_g_tokens = g_grid[0] * g_grid[1]

        self.source_embed = nn.Sequential(
            nn.Linear(patch_dim, hidden_size),
            nn.RMSNorm(hidden_size, eps=1e-6),
        )
        self.encoder_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.encoder = FlexStack(
            depth,
            hidden_size,
            num_heads,
            ffn_size,
            dropout,
            gradient_checkpointing,
        )
        self.g_projector = nn.Sequential(
            nn.RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, bagel_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(bagel_dim, bagel_dim),
            nn.RMSNorm(bagel_dim, eps=1e-6),
        )
        self.local_head = nn.Linear(bagel_dim, patch_dim * 4)
        self.g_to_decoder = nn.Linear(bagel_dim, hidden_size)
        self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.decoder = FlexStack(
            depth,
            hidden_size,
            num_heads,
            ffn_size,
            dropout,
            gradient_checkpointing,
        )
        self.patch_head = nn.Linear(hidden_size, patch_dim)

        source_coordinates = grid_coordinates(*source_grid)
        g_coordinates = grid_coordinates(*g_grid, stride=2, offset=1)
        self.register_buffer("source_coordinates", source_coordinates, persistent=True)
        self.register_buffer("g_coordinates", g_coordinates, persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_query, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_query, mean=0.0, std=0.02)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.patch_head.bias)

    def encode(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.shape[1:] != (self.num_source_tokens, self.patch_dim):
            raise ValueError(
                f"expected [B,{self.num_source_tokens},{self.patch_dim}], got {tuple(patches.shape)}"
            )
        batch = patches.shape[0]
        source = self.source_embed(patches)
        queries = self.encoder_query.expand(batch, self.num_g_tokens, -1)
        hidden = torch.cat((source, queries), dim=1)
        coordinates = torch.cat((self.source_coordinates, self.g_coordinates), dim=0)
        encoded = self.encoder(hidden, coordinates)
        pre_projector = encoded[:, self.num_source_tokens :]
        return self.g_projector(pre_projector), pre_projector

    def decode(self, g_tokens: torch.Tensor) -> torch.Tensor:
        if g_tokens.shape[1:] != (self.num_g_tokens, self.bagel_dim):
            raise ValueError(
                f"expected [B,{self.num_g_tokens},{self.bagel_dim}], got {tuple(g_tokens.shape)}"
            )
        batch = g_tokens.shape[0]
        g_hidden = self.g_to_decoder(g_tokens)
        queries = self.decoder_query.expand(batch, self.num_source_tokens, -1)
        hidden = torch.cat((g_hidden, queries), dim=1)
        coordinates = torch.cat((self.g_coordinates, self.source_coordinates), dim=0)
        decoded = self.decoder(hidden, coordinates)
        return self.patch_head(decoded[:, self.num_g_tokens :])

    def forward(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        g_tokens, pre_projector = self.encode(patches)
        reconstructed = self.decode(g_tokens)
        return {
            "g_tokens": g_tokens,
            "pre_projector": pre_projector,
            "local_prediction": self.local_head(g_tokens),
            "reconstructed_patches": reconstructed,
        }


__all__ = [
    "BagelFlexAE256",
    "apply_2d_rope",
    "grid_coordinates",
    "group_local_patch_targets",
    "patchify_latents",
    "unpatchify_latents",
]
