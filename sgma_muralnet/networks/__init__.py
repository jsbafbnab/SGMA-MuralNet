"""Network components for SGMA-MuralNet."""
from .ccn import ColorCorrectionNetwork
from .discriminator import PatchGANDiscriminator
from .line_encoder import LineEncoder
from .mfa import MFA, SEBlock, MatchedCapacityConv
from .modulation import (ConcatModulation, LocalNormOnly, ModulationHead,
                         StructureModulation, TSGLModulation,
                         build_modulation, local_channel_norm)
from .modules import (ConvBlock, ResnetBlock, SelfAttention, init_weights,
                      spectral_norm)
from .srn import VARIANT_SPECS, StructureReconstructionNetwork

__all__ = [
    "ColorCorrectionNetwork",
    "PatchGANDiscriminator",
    "LineEncoder",
    "MFA",
    "SEBlock",
    "MatchedCapacityConv",
    "ModulationHead",
    "StructureModulation",
    "LocalNormOnly",
    "ConcatModulation",
    "TSGLModulation",
    "build_modulation",
    "local_channel_norm",
    "ConvBlock",
    "ResnetBlock",
    "SelfAttention",
    "init_weights",
    "spectral_norm",
    "VARIANT_SPECS",
    "StructureReconstructionNetwork",
]
