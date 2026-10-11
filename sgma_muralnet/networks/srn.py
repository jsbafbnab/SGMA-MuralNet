"""Structure Reconstruction Network: a U-Net with a line encoder, structure-
conditioned modulation at two encoder scales, and a bottleneck MFA adapter.
``variant`` selects an ablation configuration (A0-A8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .line_encoder import LineEncoder
from .mfa import MFA, MatchedCapacityConv
from .modulation import StructureModulation, TSGLModulation, build_modulation
from .modules import ConvBlock, ResnetBlock

# variant -> (modulation mode, bottleneck-adapter mode, concat line to input)
VARIANT_SPECS = {
    "A0": ("none", "none", True),
    "A1": ("affine", "none", True),
    "A2": ("none", "mfa", True),
    "A3": ("affine", "mfa", True),
    "A4": ("norm_only", "none", True),
    "A5": ("concat", "none", True),
    "A6": ("none", "conv", True),
    "A7": ("tsgl", "none", True),
    "A8": ("affine", "mfa", False),
}
_MOD_MODES = ("affine", "norm_only", "concat", "tsgl")


class _EncoderStage(nn.Module):
    """A stride-2 encoder stage: conv, then either modulation or IN + ReLU."""

    def __init__(self, in_ch, out_ch, mod_mode, scale, modulation_scales, use_mod):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.mod = build_modulation(mod_mode, out_ch) if use_mod else None
        self.fallback_in = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.apply_mod = use_mod and (scale in modulation_scales)

    def forward(self, x, s):
        z = self.conv(x)
        if self.apply_mod:
            return self.mod(z, s)
        return F.relu(self.fallback_in(z), inplace=True)


class StructureReconstructionNetwork(nn.Module):
    def __init__(self, in_channels=4, base=32, bottleneck=256,
                 residual_blocks=4, dilations=(1, 3, 5, 7),
                 se_reduction=16, modulation_scales=(2, 3),
                 variant="A3", use_se=True):
        super().__init__()
        if variant not in VARIANT_SPECS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANT_SPECS)}")
        self.variant = variant
        self.modulation_scales = set(modulation_scales)
        mod_mode, mfa_mode, input_concat_line = VARIANT_SPECS[variant]
        self.input_concat_line = input_concat_line
        self.use_mod = mod_mode in _MOD_MODES
        self.use_line_encoder = mod_mode in ("affine", "concat", "tsgl")
        c = base

        self.line_encoder = LineEncoder(in_channels=1, base=c) if self.use_line_encoder else None

        actual_in = in_channels if input_concat_line else (in_channels - 1)
        self.stem = ConvBlock(actual_in, c, kernel_size=7, padding=0, reflection_pad=3)
        self.down1 = ConvBlock(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.stage2 = _EncoderStage(c * 2, c * 4, mod_mode, 2, self.modulation_scales, self.use_mod)
        self.stage3 = _EncoderStage(c * 4, c * 8, mod_mode, 3, self.modulation_scales, self.use_mod)

        self.residuals = nn.Sequential(
            *[ResnetBlock(bottleneck, dilation=2) for _ in range(residual_blocks)]
        )
        if mfa_mode == "mfa":
            self.adapter = MFA(bottleneck, dilations=dilations,
                               se_reduction=se_reduction, use_se=use_se)
        elif mfa_mode == "conv":
            self.adapter = MatchedCapacityConv(bottleneck)
        else:
            self.adapter = nn.Identity()

        self.pre_conv = ConvBlock(bottleneck, c * 4, kernel_size=3, padding=1)
        self.up3 = ConvBlock((c * 4) + (c * 4), c * 2, kernel_size=3, padding=1)
        self.up2 = ConvBlock((c * 2) + (c * 2), c, kernel_size=3, padding=1)
        self.up1 = ConvBlock(c + c, c // 2, kernel_size=3, padding=1)
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c // 2, 3, kernel_size=7, padding=0),
        )

    def forward(self, x, line):
        # x: damaged image with/without the line channel (4ch, or 3ch for A8);
        # line: (B, 1, H, W) registered complete line drawing.
        if self.use_line_encoder:
            _, _, s2, s3 = self.line_encoder(line)
        else:
            s2 = s3 = None

        t0 = self.stem(x)
        t1 = self.down1(t0)
        t2 = self.stage2(t1, s2)
        t3 = self.stage3(t2, s3)

        f = self.residuals(t3)
        f_out = self.adapter(f)

        return self._decode(f_out, t0, t1, t2)

    def affine_maps(self, line):
        """Learned affine parameters (a, b) per modulation scale, used by visualize.py."""
        if not self.use_line_encoder:
            return {}
        _, _, s2, s3 = self.line_encoder(line)
        maps = {}
        for scale, stage, s in ((2, self.stage2, s2), (3, self.stage3, s3)):
            if scale not in self.modulation_scales:
                continue
            mod = stage.mod
            if isinstance(mod, (StructureModulation, TSGLModulation)):
                a, b = mod.head(s)
                maps[(scale, "a")] = a
                maps[(scale, "b")] = b
        return maps

    def _decode(self, f_out, t0, t1, t2):
        x = self.pre_conv(f_out)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, t2], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, t1], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, t0], dim=1)
        x = self.up1(x)

        x = self.out_conv(x)
        x = (torch.tanh(x) + 1.0) / 2.0
        return x.clamp(0.0, 1.0)
