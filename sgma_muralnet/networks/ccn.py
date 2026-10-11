"""Color Correction Network: a U-Net with self-attention that predicts a signed residual R in [-1, 1] from [bIs, M]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvBlock, ResnetBlock, SelfAttention


class ColorCorrectionNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 32,
                 residual_blocks: int = 4, use_attention: bool = True):
        super().__init__()
        c = base

        # Encoder
        self.stem = ConvBlock(in_channels, c, kernel_size=7, padding=0, reflection_pad=3)
        self.down1 = ConvBlock(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = ConvBlock(c * 4, c * 8, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.attention = SelfAttention(c * 8) if use_attention else nn.Identity()
        self.residuals = nn.Sequential(
            *[ResnetBlock(c * 8, dilation=2) for _ in range(residual_blocks)]
        )

        # Decoder (matched-scale skips)
        self.pre_conv = ConvBlock(c * 8, c * 4, kernel_size=3, padding=1)
        self.up3 = ConvBlock((c * 4) + (c * 4), c * 2, kernel_size=3, padding=1)
        self.up2 = ConvBlock((c * 2) + (c * 2), c, kernel_size=3, padding=1)
        self.up1 = ConvBlock(c + c, c // 2, kernel_size=3, padding=1)
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c // 2, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [bIs, M]; returns the signed residual R in [-1, 1].
        t0 = self.stem(x)
        t1 = self.down1(t0)
        t2 = self.down2(t1)
        t3 = self.down3(t2)

        f = self.residuals(self.attention(t3))

        y = self.pre_conv(f)
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat([y, t2], dim=1)
        y = self.up3(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat([y, t1], dim=1)
        y = self.up2(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat([y, t0], dim=1)
        y = self.up1(y)

        return self.out_conv(y)
