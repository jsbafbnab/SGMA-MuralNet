"""Loss functions for SGMA-MuralNet."""
from .adversarial import RelativisticAverageLoss
from .criterion import Criterion
from .histogram import HistogramLoss
from .perceptual import ContentLoss, StyleLoss
from .vgg import VGG19

__all__ = [
    "RelativisticAverageLoss",
    "Criterion",
    "HistogramLoss",
    "ContentLoss",
    "StyleLoss",
    "VGG19",
]
