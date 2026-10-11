"""Feature histogram matching loss (Eq. 14): per-channel sorted-value L2 distance."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .vgg import VGG19

DEFAULT_HIST_LAYERS = ["relu4_1", "relu4_2", "relu4_3", "relu4_4"]


class HistogramLoss(nn.Module):
    def __init__(self, layers: List[str] | None = None, normalize: bool = True,
                 vgg: VGG19 | None = None):
        super().__init__()
        self.layers = layers or DEFAULT_HIST_LAYERS
        self.vgg = vgg or VGG19(normalize=normalize, layers=self.layers)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_feats = self.vgg(pred)
        t_feats = self.vgg(target)
        loss = 0.0
        for name in self.layers:
            p = p_feats[name]
            t = t_feats[name]
            b, c, h, w = p.size()
            n = h * w
            p_sorted = torch.sort(p.flatten(2), dim=2).values
            t_sorted = torch.sort(t.flatten(2), dim=2).values
            diff2 = (p_sorted - t_sorted) ** 2
            loss = loss + diff2.sum(dim=(1, 2)).mean() / (c * n)
        return loss
