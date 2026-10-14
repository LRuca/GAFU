from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


SOURCE_GRID = (24, 24)
G_GRID = (16, 16)


def flatten_vq_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[1:] != (8, 24, 24):
        raise ValueError(f"expected [B,8,24,24], got {tuple(latents.shape)}")
    return latents.permute(0, 2, 3, 1).reshape(latents.shape[0], 24 * 24, 8)


def unflatten_vq_latents(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1:] != (24 * 24, 8):
        raise ValueError(f"expected [B,576,8], got {tuple(tokens.shape)}")
    return tokens.reshape(tokens.shape[0], 24, 24, 8).permute(0, 3, 1, 2)


def spatial_pool_targets(
    tokens: torch.Tensor,
    source_grid: tuple[int, int] = SOURCE_GRID,
    target_grid: tuple[int, int] = G_GRID,
) -> torch.Tensor:
    """Deterministically pool native source cells into the fixed G-token grid."""
    if tokens.ndim != 3 or tokens.shape[1] != source_grid[0] * source_grid[1]:
        raise ValueError(f"source token shape mismatch: {tuple(tokens.shape)}")
    image = tokens.reshape(tokens.shape[0], *source_grid, tokens.shape[-1]).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(image.float(), target_grid).to(tokens.dtype)
    return pooled.permute(0, 2, 3, 1).reshape(tokens.shape[0], target_grid[0] * target_grid[1], -1)


def grid_coordinates(height: int, width: int) -> torch.Tensor:
    # A shared normalized coordinate frame lets 24x24 source cells and 16x16 G
    # queries interact without resampling the codebook embeddings.
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    cols = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    y, x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1) * 32.0


def _rotate_axis(values: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    width = values.shape[-1]
    inv_freq = torch.exp(
        -math.log(base)
        * torch.arange(0, width, 2, device=values.device, dtype=torch.float32)
        / width
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().to(values.dtype).unsqueeze(0).unsqueeze(0)
    sin = angles.sin().to(values.dtype).unsqueeze(0).unsqueeze(0)
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def apply_2d_rope(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or coordinates.shape != (values.shape[2], 2):
        raise ValueError("invalid RoPE tensor/coordinate contract")
    if values.shape[-1] % 4:
        raise ValueError("2D RoPE head width must be divisible by four")
    half = values.shape[-1] // 2
    return torch.cat(
        (_rotate_axis(values[..., :half], coordinates[:, 0], 10000.0),
         _rotate_axis(values[..., half:], coordinates[:, 1], 10000.0)),
        dim=-1,
    )


class JointSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads or (width // heads) % 4:
            raise ValueError("attention width/head contract is invalid")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q = apply_2d_rope(q.transpose(1, 2), coordinates)
        k = apply_2d_rope(k.transpose(1, 2), coordinates)
        v = v.transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out(mixed.transpose(1, 2).reshape(batch, length, width))


class FlexBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(width, eps=1e-6)
        self.attn = JointSelfAttention(width, heads)
        self.ffn_norm = nn.RMSNorm(width, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(approximate="tanh"), nn.Linear(ffn, width))

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), coordinates)
        return hidden + self.ffn(self.ffn_norm(hidden))


class FlexStack(nn.Module):
    def __init__(self, depth: int, width: int, heads: int, ffn: int, gradient_checkpointing: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FlexBlock(width, heads, ffn) for _ in range(depth))
        self.final_norm = nn.RMSNorm(width, eps=1e-6)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = checkpoint(layer, hidden, coordinates, use_reentrant=False) \
                if self.training and self.gradient_checkpointing else layer(hidden, coordinates)
        return self.final_norm(hidden)


class JanusFlexAE256(nn.Module):
    """Compress native Janus 24x24x8 VQ embeddings into 16x16x1024 G tokens."""

    def __init__(
        self,
        *,
        source_dim: int = 8,
        hidden_size: int = 1024,
        g_dim: int = 1024,
        source_grid: tuple[int, int] = SOURCE_GRID,
        g_grid: tuple[int, int] = G_GRID,
        depth: int = 6,
        num_heads: int = 8,
        ffn_size: int = 4096,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.source_dim = source_dim
        self.hidden_size = hidden_size
        self.g_dim = g_dim
        self.source_grid = source_grid
        self.g_grid = g_grid
        self.num_source_tokens = source_grid[0] * source_grid[1]
        self.num_g_tokens = g_grid[0] * g_grid[1]
        self.source_embed = nn.Sequential(nn.Linear(source_dim, hidden_size), nn.RMSNorm(hidden_size))
        self.encoder_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.encoder = FlexStack(depth, hidden_size, num_heads, ffn_size, gradient_checkpointing)
        self.g_projector = nn.Sequential(
            nn.RMSNorm(hidden_size), nn.Linear(hidden_size, g_dim), nn.GELU(approximate="tanh"),
            nn.Linear(g_dim, g_dim), nn.RMSNorm(g_dim),
        )
        self.local_head = nn.Linear(g_dim, source_dim)
        self.g_to_decoder = nn.Linear(g_dim, hidden_size)
        self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.decoder = FlexStack(depth, hidden_size, num_heads, ffn_size, gradient_checkpointing)
        self.source_head = nn.Linear(hidden_size, source_dim)
        self.register_buffer("source_coordinates", grid_coordinates(*source_grid), persistent=True)
        self.register_buffer("g_coordinates", grid_coordinates(*g_grid), persistent=True)
        nn.init.normal_(self.encoder_query, std=0.02)
        nn.init.normal_(self.decoder_query, std=0.02)

    def encode(self, source_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if source_tokens.shape[1:] != (self.num_source_tokens, self.source_dim):
            raise ValueError(f"source shape mismatch: {tuple(source_tokens.shape)}")
        batch = source_tokens.shape[0]
        hidden = torch.cat(
            (self.source_embed(source_tokens), self.encoder_query.expand(batch, self.num_g_tokens, -1)), dim=1,
        )
        encoded = self.encoder(hidden, torch.cat((self.source_coordinates, self.g_coordinates)))
        pre_projector = encoded[:, self.num_source_tokens:]
        return self.g_projector(pre_projector), pre_projector

    def decode(self, g_tokens: torch.Tensor) -> torch.Tensor:
        if g_tokens.shape[1:] != (self.num_g_tokens, self.g_dim):
            raise ValueError(f"G shape mismatch: {tuple(g_tokens.shape)}")
        batch = g_tokens.shape[0]
        hidden = torch.cat(
            (self.g_to_decoder(g_tokens), self.decoder_query.expand(batch, self.num_source_tokens, -1)), dim=1,
        )
        decoded = self.decoder(hidden, torch.cat((self.g_coordinates, self.source_coordinates)))
        return self.source_head(decoded[:, self.num_g_tokens:])

    def forward(self, source_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        g_tokens, pre_projector = self.encode(source_tokens)
        return {
            "g_tokens": g_tokens,
            "pre_projector": pre_projector,
            "local_prediction": self.local_head(g_tokens),
            "reconstructed_tokens": self.decode(g_tokens),
        }


__all__ = ["G_GRID", "SOURCE_GRID", "JanusFlexAE256", "flatten_vq_latents", "spatial_pool_targets", "unflatten_vq_latents"]
