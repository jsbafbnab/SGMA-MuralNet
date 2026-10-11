"""Complete-line encoder: 7x7 stem + three stride-2 convs (1 -> 32 -> 64 -> 128 -> 256)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LineEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32):
        super().__init__()
        # 7x7 reflection-padded stem  (1 -> 32)
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
        )
        # Three stride-2 convolution blocks  (32 -> 64 -> 128 -> 256)
        self.block1 = self._down(base, base * 2)
        self.block2 = self._down(base * 2, base * 4)
        self.block3 = self._down(base * 4, base * 8)

    @staticmethod
    def _down(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, line: torch.Tensor):
        """Return the structural features (S0, S1, S2, S3)."""
        s0 = self.stem(line)
        s1 = self.block1(s0)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        return s0, s1, s2, s3
