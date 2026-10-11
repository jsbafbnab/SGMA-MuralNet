"""Relativistic average logistic adversarial loss (Eqs. 12-13).

D outputs pre-activation logits; expectations average logits over the batch and
spatial dimensions of each micro-batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relativistic_pair(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    """Return (a_r, a_f) given (B, 1, H, W) pre-activation logit maps."""
    real_mean = real_logits.mean()          # scalar: E_I[D(I)]
    fake_mean = fake_logits.mean()          # scalar: E_P[D(P)]
    a_r = real_logits - fake_mean
    a_f = fake_logits - real_mean
    return a_r, a_f


class RelativisticAverageLoss(nn.Module):
    """Relativistic average logistic (RaLSGAN) loss.

    Call ``d_loss(real, fake)`` / ``g_loss(real, fake)`` with raw logits. The
    caller must detach generated images for the discriminator and freeze the
    discriminator for the generator (see ``criterion.py``).
    """

    def discriminator_loss(self, real_logits: torch.Tensor,
                           fake_logits: torch.Tensor) -> torch.Tensor:
        a_r, a_f = _relativistic_pair(real_logits, fake_logits)
        return -(F.logsigmoid(a_r).mean() + F.logsigmoid(-a_f).mean())

    def generator_loss(self, real_logits: torch.Tensor,
                       fake_logits: torch.Tensor) -> torch.Tensor:
        a_r, a_f = _relativistic_pair(real_logits, fake_logits)
        return -(F.logsigmoid(-a_r).mean() + F.logsigmoid(a_f).mean())
