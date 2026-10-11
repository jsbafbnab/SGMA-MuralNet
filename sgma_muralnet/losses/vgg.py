"""Frozen ImageNet-pretrained VGG-19 feature extractor (named ReLU feature maps)."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as tv_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _select_layers() -> Dict[str, int]:
    """Map layer name -> index of the feature-module boundary in vgg19.features."""
    # Convolution+ReLU pairs as (module index range) for each named ReLU output.
    return {
        "relu1_1": 2, "relu1_2": 4,
        "relu2_1": 7, "relu2_2": 9,
        "relu3_1": 12, "relu3_2": 14, "relu3_3": 16, "relu3_4": 18,
        "relu4_1": 21, "relu4_2": 23, "relu4_3": 25, "relu4_4": 27,
        "relu5_1": 30, "relu5_2": 32, "relu5_3": 34, "relu5_4": 36,
    }


class VGG19(nn.Module):
    """VGG-19 whose ``forward`` returns a dict of named ReLU feature maps."""

    def __init__(self, requires_grad: bool = False, normalize: bool = True,
                 layers: List[str] | None = None):
        super().__init__()
        weights = tv_models.VGG19_Weights.IMAGENET1K_V1
        vgg = tv_models.vgg19(weights=weights).features.eval()
        self.normalize = normalize
        self.layers = layers or list(_select_layers().keys())

        # Keep only the prefix needed for the deepest requested layer.
        bounds = _select_layers()
        max_index = max(bounds[l] for l in self.layers)
        self.slice = vgg[:max_index]

        for p in self.parameters():
            p.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.normalize:
            mean = IMAGENET_MEAN.to(x.device, x.dtype)
            std = IMAGENET_STD.to(x.device, x.dtype)
            x = (x - mean) / std

        bounds = _select_layers()
        out: Dict[str, torch.Tensor] = {}
        prev = 0
        # Names must be traversed in index order, not arbitrary dict order.
        ordered = sorted(self.layers, key=lambda l: bounds[l])
        for name in ordered:
            idx = bounds[name]
            for i in range(prev, idx):
                x = self.slice[i](x)
            out[name] = x
            prev = idx
        return out
