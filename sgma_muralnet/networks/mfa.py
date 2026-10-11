"""Multi-scale feature aggregation (MFA) adapter and the A6 matched-capacity conv control."""
from __future__ import annotations

import torch
import torch.nn as nn

from .modules import ConvBlock, ResnetBlock


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel recalibration (SENet)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MFA(nn.Module):
    def __init__(self, channels: int = 256, dilations=(1, 3, 5, 7),
                 se_reduction: int = 16, use_se: bool = True):
        super().__init__()
        assert channels % len(dilations) == 0
        out_channels = channels // len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3,
                          padding=d, dilation=d),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self.use_se = use_se
        self.se = SEBlock(channels, reduction=se_reduction) if use_se else nn.Identity()
        # Zero-initialized residual projection (stable training dynamics).
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        u = torch.cat([branch(f) for branch in self.branches], dim=1)
        v = self.se(u)
        return f + self.proj(v)


class MatchedCapacityConv(nn.Module):
    """A6 control: ordinary residual + convolution capacity in place of MFA.

    Adds a comparable number of parameters to the line encoder + conditioning
    heads + MFA combination using standard residual blocks, to control for raw
    model capacity.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.residual = ResnetBlock(channels, dilation=2)
        self.conv = ConvBlock(channels, channels, kernel_size=3, padding=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv(self.residual(f))
