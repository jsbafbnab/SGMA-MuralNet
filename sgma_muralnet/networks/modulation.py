"""Structure-conditioned modulation (Eqs. 3-5) and the A4/A5/A7 control heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_channel_norm(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalize across the channel dimension at each spatial position."""
    mu = z.mean(dim=1, keepdim=True)
    var = z.var(dim=1, keepdim=True, unbiased=False)
    return (z - mu) / torch.sqrt(var + eps)


class ModulationHead(nn.Module):
    """Conditioning subnetwork H_i: S_i -> (a_i, b_i).

    3x3 conv (C -> C/2) -> ReLU -> 1x1 conv (C/2 -> 2C) -> split.
    The 1x1 conv is zero-initialized (weight = 0, bias = 0).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 2, channels * 2, kernel_size=1)
        # Zero-initialize the final projection.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, s: torch.Tensor):
        h = self.relu(self.conv1(s))
        out = self.conv2(h)
        a, b = out.chunk(2, dim=1)
        return a, b


class StructureModulation(nn.Module):
    """A1/A3/A8: affine modulation of channel-local-normalized features (Eq. 5)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.head = ModulationHead(channels)
        self.eps = eps

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z_norm = local_channel_norm(z, self.eps)
        a, b = self.head(s)
        return F.relu((1.0 + a) * z_norm + b, inplace=True)


class LocalNormOnly(nn.Module):
    """A4 control: local channel normalization with no affine parameters."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        return F.relu(local_channel_norm(z, self.eps), inplace=True)


class ConcatModulation(nn.Module):
    """A5 control: concatenate image and structural features, then 1x1 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(torch.cat([z, s], dim=1)), inplace=True)


class TSGLModulation(nn.Module):
    """A7 control: combined global + local normalization with affine conditioning.

    Applies instance normalization (global, per-channel across H/W) and
    channel-local normalization (per-position across C), sums the two, then
    modulates with line-derived affine parameters.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.head = ModulationHead(channels)
        self.in_norm = nn.InstanceNorm2d(channels, track_running_stats=False)
        self.eps = eps

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z_global = self.in_norm(z)
        z_local = local_channel_norm(z, self.eps)
        z_hat = z_global + z_local
        a, b = self.head(s)
        return F.relu((1.0 + a) * z_hat + b, inplace=True)


def build_modulation(mode: str, channels: int) -> nn.Module:
    """Instantiate the conditioning head for a given ablation mode.

    Args:
        mode: 'affine' | 'norm_only' | 'concat' | 'tsgl'.
    """
    if mode == "affine":
        return StructureModulation(channels)
    if mode == "norm_only":
        return LocalNormOnly(channels)
    if mode == "concat":
        return ConcatModulation(channels)
    if mode == "tsgl":
        return TSGLModulation(channels)
    raise ValueError(f"unknown modulation mode {mode!r}")
