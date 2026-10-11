"""SGMA-MuralNet: Structure-Guided Texture Enhancement and Multi-Scale Feature
Fusion for Progressive Mural Inpainting.

Two-stage SRN-CCN generator with line-conditioned feature modulation and
multi-scale feature aggregation, plus training, evaluation and analysis tools.
"""
from .config import Config
from .model import SGMAMuralNet, build_discriminator, build_generator

__version__ = "0.2.0"

__all__ = ["Config", "SGMAMuralNet", "build_generator", "build_discriminator"]
