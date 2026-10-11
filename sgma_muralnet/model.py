"""SGMA-MuralNet: the two-stage generator. The SRN reconstructs structure, the
CCN predicts a signed color residual, and mask compositing preserves observed
pixels.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .networks import ColorCorrectionNetwork, PatchGANDiscriminator
from .networks import StructureReconstructionNetwork


class SGMAMuralNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variant = config.get("VARIANT", "A3")

        self.srn = StructureReconstructionNetwork(
            in_channels=config.IN_CHANNELS,
            base=config.BASE_CHANNELS,
            bottleneck=config.BOTTLENECK,
            residual_blocks=config.RESIDUAL_BLOCKS,
            dilations=tuple(config.DILATIONS),
            se_reduction=config.SE_REDUCTION,
            modulation_scales=tuple(config.MODULATION_SCALES),
            variant=self.variant,
            use_se=config.get("USE_SE", True),
        )
        self.ccn = ColorCorrectionNetwork(
            in_channels=config.IN_CHANNELS,
            base=config.BASE_CHANNELS,
            residual_blocks=config.RESIDUAL_BLOCKS,
        )

    def forward(self, io: torch.Tensor, line: torch.Tensor, mask: torch.Tensor,
                return_all: bool = False):
        # io: damaged mural in [0,1]; line: registered drawing; mask: 1=hole.
        # Stage 1: structure reconstruction
        if self.srn.input_concat_line:
            srn_input = torch.cat([io, line], dim=1)
        else:
            srn_input = io
        js = self.srn(srn_input, line)
        b_is = io + mask * js

        # Stage 2: color correction
        ccn_input = torch.cat([b_is, mask], dim=1)
        r = self.ccn(ccn_input)
        jf = (b_is + r).clamp(0.0, 1.0)
        b_i = io + mask * jf

        if return_all:
            return js, b_is, r, jf, b_i
        return b_i


def build_generator(config) -> SGMAMuralNet:
    return SGMAMuralNet(config)


def build_discriminator(config) -> PatchGANDiscriminator:
    return PatchGANDiscriminator(in_channels=3,
                                 use_spectral_norm=config.USE_SPECTRAL_NORM)
