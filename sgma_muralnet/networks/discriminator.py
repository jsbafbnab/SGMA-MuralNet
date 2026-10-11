"""70x70 PatchGAN discriminator: five 4x4 convs (64-128-256-512-1), spectral norm, pre-activation logits."""
from __future__ import annotations

import torch
import torch.nn as nn

from .modules import spectral_norm


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, use_spectral_norm: bool = True):
        super().__init__()

        def conv(in_ch, out_ch, stride, sn=True):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                                        padding=1, bias=not (sn and use_spectral_norm)),
                              sn and use_spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv1 = conv(in_channels, 64, stride=2)
        self.conv2 = conv(64, 128, stride=2)
        self.conv3 = conv(128, 256, stride=2)
        self.conv4 = conv(256, 512, stride=1)
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm),
                          use_spectral_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pre-activation logits (B, 1, H/8, W/8)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)
