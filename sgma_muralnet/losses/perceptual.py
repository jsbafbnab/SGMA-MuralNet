"""VGG-19 content and Gram-matrix style losses (L1 on relu features / Gram matrices)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .vgg import VGG19

DEFAULT_CONTENT_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
DEFAULT_STYLE_LAYERS = ["relu2_2", "relu3_4", "relu4_4", "relu5_2"]


def _gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    return f.bmm(f.transpose(1, 2)) / (h * w * c)


class ContentLoss(nn.Module):
    def __init__(self, layers: List[str] | None = None, normalize: bool = True,
                 vgg: VGG19 | None = None):
        super().__init__()
        self.layers = layers or DEFAULT_CONTENT_LAYERS
        self.vgg = vgg or VGG19(normalize=normalize, layers=self.layers)
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_feats = self.vgg(pred)
        t_feats = self.vgg(target)
        return sum(self.criterion(p_feats[l], t_feats[l]) for l in self.layers)


class StyleLoss(nn.Module):
    def __init__(self, layers: List[str] | None = None, normalize: bool = True,
                 vgg: VGG19 | None = None):
        super().__init__()
        self.layers = layers or DEFAULT_STYLE_LAYERS
        self.vgg = vgg or VGG19(normalize=normalize, layers=self.layers)
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_feats = self.vgg(pred)
        t_feats = self.vgg(target)
        return sum(self.criterion(_gram(p_feats[l]), _gram(t_feats[l])) for l in self.layers)
