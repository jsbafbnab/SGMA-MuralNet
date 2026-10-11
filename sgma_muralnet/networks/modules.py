"""Shared building blocks: residual block, non-local self-attention, weight init."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module: nn.Module, enabled: bool = True) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class ConvBlock(nn.Module):
    """conv -> InstanceNorm -> ReLU (used throughout the SRN/CCN encoders and decoders)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 reflection_pad: int = 0, norm: bool = True, activation: str = "relu"):
        super().__init__()
        layers = []
        if reflection_pad > 0:
            layers.append(nn.ReflectionPad2d(reflection_pad))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, track_running_stats=False))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "none":
            pass
        else:
            raise ValueError(f"unknown activation {activation!r}")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResnetBlock(nn.Module):
    """Residual block with two 3x3 convolutions and instance normalization.

    The first convolution supports a dilation rate (the MuralNet base uses
    dilation=2 at the bottleneck). There is no ReLU after the residual add.
    """

    def __init__(self, dim: int, dilation: int = 1, use_spectral_norm: bool = False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                    dilation=dilation, bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                    dilation=1, bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class SelfAttention(nn.Module):
    """Non-local self-attention (embedded Gaussian), used inside the CCN.

    Query/key/value 1x1 projections followed by a scaled-dot-product attention
    with a learnable, zero-initialized gamma.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


def init_weights(model: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """Weight initialization matching MuralNet/EdgeConnect conventions."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
