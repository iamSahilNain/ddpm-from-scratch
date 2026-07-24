"""A small UNet noise-prediction network for DDPM.

Structurally this follows Ho et al. (2020) — residual blocks conditioned on a
sinusoidal timestep embedding, self-attention at the coarser resolutions, and
skip connections from every downsampling block into the matching upsampling
block — but sized for MNIST and for a CPU or a single GPU rather than for
CIFAR on a TPU pod.

The network's contract with :mod:`diffusion` is narrow: ``model(x_t, t)`` takes
a batch of noised images and a batch of integer timesteps and returns a tensor
of the same shape as ``x_t``, interpreted as predicted noise.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["UNet", "timestep_embedding", "count_parameters"]


def _group_norm(channels: int, max_groups: int = 32, min_per_group: int = 4) -> nn.GroupNorm:
    """GroupNorm with a group count that divides ``channels`` at any width.

    A hardcoded ``num_groups=32`` breaks on the narrow models the test suite
    uses, so take the largest divisor that still leaves several channels per
    group. Groups of one channel would make this instance normalisation, whose
    statistics over a twenty-eight square map are noisy.
    """
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0 and channels // groups >= min_per_group:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding of a batch of integer timesteps.

    The same construction as positional encodings in transformers: a bank of
    ``dim // 2`` frequencies spaced geometrically between one and
    ``1 / max_period``, each evaluated as a cosine and a sine of the timestep.
    Nearby timesteps land at nearby embeddings, so the network can generalise
    across ``t`` instead of learning each of the thousand steps from scratch.

    Args:
        t: Shape ``(B,)``, integer or float. The timestep of each batch element.
        dim: Width of the returned embedding. Odd widths are zero-padded.
        max_period: Period of the slowest frequency in the bank.

    Returns:
        Shape ``(B, dim)``, dtype float32, on ``t``'s device.
    """
    half = dim // 2
    exponents = torch.arange(half, dtype=torch.float32, device=t.device) / half
    freqs = torch.exp(-math.log(max_period) * exponents)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block conditioned on the timestep by a learned scale and shift.

    The timestep embedding is projected to two vectors per channel and applied
    as an affine transform *after* the second normalisation — the adaptive group
    norm of Nichol & Dhariwal (2021) rather than the plain addition before the
    norm used in the original DDPM code.

    The difference is not cosmetic. A per-channel constant added *before* a
    normalisation layer is partly, and at narrow widths entirely, subtracted
    straight back out by that layer: with one channel per group the conditioning
    signal vanishes exactly and the network silently becomes unconditional,
    which trains to a plausible-looking loss and samples pure mush. Applying the
    conditioning after the norm makes it survive at any width.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(temb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-headed-or-more self-attention over the spatial positions.

    Applied only at the coarser resolutions, where the sequence length is short
    enough that the quadratic cost is irrelevant. It gives the network a way to
    make globally coherent decisions — which digit is being drawn — that a stack
    of three-by-three convolutions reaches only slowly.
    """

    def __init__(self, channels: int, num_heads: Optional[int] = None):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, channels // 32)
        while channels % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        b, c, h, w = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.num_heads, head_dim, h * w).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, HW, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ResAttnBlock(nn.Module):
    """A residual block optionally followed by attention, as one skip source."""

    def __init__(self, res: ResBlock, attn: Optional[AttentionBlock]):
        super().__init__()
        self.res = res
        self.attn = attn

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        x = self.res(x, temb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample(nn.Module):
    """Halve both spatial dimensions with a strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Double both spatial dimensions: nearest-neighbour then a convolution.

    Nearest-then-conv rather than a transposed convolution, which is the usual
    fix for the checkerboard artefacts transposed convolutions produce.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """UNet epsilon-prediction network.

    Args:
        in_channels: Channels of the input image. One for MNIST.
        out_channels: Channels of the prediction. Defaults to ``in_channels``,
            which is what epsilon-prediction requires — the output lives in the
            same space as the noise that was added.
        base_channels: Width at the finest resolution. Everything else scales
            off this, so it is the single knob for model size.
        channel_mults: Width multiplier per resolution level. The length sets
            the number of levels, and there is one downsample between
            consecutive levels — so the default of three levels takes MNIST's
            twenty-eight down to fourteen and then to seven.
        num_res_blocks: Residual blocks per level on the way down. The way up
            uses one more per level, to consume the extra skip left by the
            downsampling op.
        attention_levels: Indices of the levels that get self-attention, zero
            being the finest. Attention at the finest level is expensive and
            not worth it here.
        dropout: Dropout inside each residual block.

    The input's spatial dimensions must be divisible by two to the power of
    ``len(channel_mults) - 1``, since each downsample halves them and each
    upsample doubles them back. Twenty-eight over four is seven, which is why
    three levels is the maximum for MNIST without padding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        attention_levels: Sequence[int] = (1, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        time_dim = base_channels * 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.downsample_factor = 2 ** (len(channel_mults) - 1)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        def maybe_attn(channels: int, level: int) -> Optional[AttentionBlock]:
            return AttentionBlock(channels) if level in set(attention_levels) else None

        # Down path. Every entry produces exactly one skip tensor, which keeps
        # the bookkeeping in forward() to a single stack.
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]  # the stem's output
        channels = base_channels
        for level, mult in enumerate(channel_mults):
            level_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResAttnBlock(
                        ResBlock(channels, level_channels, time_dim, dropout),
                        maybe_attn(level_channels, level),
                    )
                )
                channels = level_channels
                skip_channels.append(channels)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(channels))
                skip_channels.append(channels)

        self.mid_block1 = ResBlock(channels, channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResBlock(channels, channels, time_dim, dropout)

        # Up path. Mirror of the down path; each ResAttnBlock consumes one skip.
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            level_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResAttnBlock(
                        ResBlock(
                            channels + skip_channels.pop(), level_channels, time_dim, dropout
                        ),
                        maybe_attn(level_channels, level),
                    )
                )
                channels = level_channels
            if level != 0:
                self.up_blocks.append(Upsample(channels))
        assert not skip_channels, "down and up paths disagree about skip count"

        self.out_norm = _group_norm(channels)
        self.out_conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the noise in ``x`` at timestep ``t``.

        Args:
            x: Shape ``(B, in_channels, H, W)``. Both spatial dimensions must be
                divisible by ``self.downsample_factor``.
            t: Shape ``(B,)``, integer timesteps. A zero-dimensional tensor or a
                Python int is accepted and broadcast over the batch, which is
                convenient inside a sampling loop.

        Returns:
            Shape ``(B, out_channels, H, W)``.
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"t has batch {t.shape[0]} but x has batch {x.shape[0]}")
        if x.shape[-1] % self.downsample_factor or x.shape[-2] % self.downsample_factor:
            raise ValueError(
                f"spatial dims {tuple(x.shape[-2:])} must be divisible by "
                f"{self.downsample_factor} for {len(self.channel_mults)} levels"
            )

        temb = self.time_mlp(timestep_embedding(t, self.base_channels).to(x.dtype))

        h = self.stem(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, temb)
            skips.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h, temb)
            else:
                h = block(torch.cat([h, skips.pop()], dim=1), temb)

        return self.out_conv(F.silu(self.out_norm(h)))


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters, for sanity-checking model size."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
