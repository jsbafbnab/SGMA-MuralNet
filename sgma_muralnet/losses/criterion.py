"""Combined training objective (Eqs. 11-17).

``train.py`` handles the discriminator forward passes and detaching:
discriminator update uses logits of ``generated.detach()``; generator update
freezes the discriminator and uses ``real.detach()``. Relativistic averages are
computed per micro-batch.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .adversarial import RelativisticAverageLoss
from .histogram import HistogramLoss
from .perceptual import ContentLoss, StyleLoss
from .vgg import VGG19, _select_layers


class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1_w = config.L1_WEIGHT
        self.c_w = config.CONTENT_WEIGHT
        self.s_w = config.STYLE_WEIGHT
        self.a_w = config.ADV_WEIGHT
        self.h_w = config.HIST_WEIGHT

        # one VGG-19 shared by the content/style/histogram losses
        bounds = _select_layers()
        all_layers = sorted(set(config.CONTENT_LAYERS + config.STYLE_LAYERS + config.HIST_LAYERS),
                            key=lambda name: bounds[name])
        vgg = VGG19(normalize=config.VGG_NORMALIZE, layers=all_layers)

        self.content = ContentLoss(config.CONTENT_LAYERS, vgg=vgg)
        self.style = StyleLoss(config.STYLE_LAYERS, vgg=vgg)
        self.hist = HistogramLoss(config.HIST_LAYERS, vgg=vgg)
        self.adv = RelativisticAverageLoss()

    def l1(self, pred, target, mask=None):
        """Weighted mean absolute error; ``mask`` selects pixels (Eq. 14)."""
        diff = (pred - target).abs()
        if mask is None:
            num = diff.sum()
            den = num.new_tensor(3.0 * pred.size(0) * pred.size(2) * pred.size(3))
        else:
            num = (diff * mask).sum()
            den = 3.0 * mask.sum()
        return num / den.clamp(min=1.0)

    def _recon(self, pred, target, mask):
        l1 = self.l1_w * self.l1(pred, target, mask)
        con = self.c_w * self.content(pred, target)
        sty = self.s_w * self.style(pred, target)
        return {"l1": l1, "content": con, "style": sty}

    def discriminator_loss(self, d_real, d_fakes):
        """Average of L_D over each (real, fake) pair (Eqs. 12, 15)."""
        terms = [self.adv.discriminator_loss(d_real, d_f) for d_f in d_fakes]
        return sum(terms) / len(terms)

    def generator_loss(self, phase, d_real, d_fakes, preds, target, mask=None):
        # preds: [Js] in phase 1, [Js, bI] in phase 2
        adv = self.a_w * sum(self.adv.generator_loss(d_real, d_f) for d_f in d_fakes) / len(d_fakes)

        recon = None
        for i, p in enumerate(preds):
            m = mask if i == 1 else None
            r = self._recon(p, target, m)
            r = {k + f"_{i}": v for k, v in r.items()}
            recon = r if recon is None else {**recon, **r}

        recon_sum = 0.5 * sum(recon.values()) if phase == 2 else sum(recon.values())
        total = recon_sum + adv

        logs = dict(recon)
        logs["adv"] = adv
        if phase == 2:
            hist = self.h_w * self.hist(preds[-1], target)
            total = total + hist
            logs["hist"] = hist
        logs["total"] = total
        return logs
