"""PSNR (full / hole-only), SSIM, and LPIPS metrics."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor,
         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Returns per-sample PSNR (dB). ``mask`` 1 = hole (computes hole PSNR)."""
    mse = (pred - target).pow(2).mean(dim=1)
    if mask is not None:
        m = mask.squeeze(1)
        mse = (mse * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1.0)
    else:
        mse = mse.mean(dim=(1, 2))
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, val_range: float = 1.0) -> torch.Tensor:
    """Returns per-sample SSIM. Expects (B, C, H, W) inputs in [0, 1]."""
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    channels = pred.size(1)
    kernel = _gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).mean(dim=(1, 2, 3))


class LPIPSMetric:
    """Wrapper around the ``lpips`` package (AlexNet). Falls back to None if
    the package is unavailable."""

    def __init__(self):
        self._net = None
        try:
            import lpips
            self._net = lpips.LPIPS(net="alex", verbose=False)
        except Exception:
            self._net = None

    @property
    def available(self) -> bool:
        return self._net is not None

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("lpips is not installed; pip install lpips")
        self._net = self._net.to(pred.device)
        return self._net(pred, target).squeeze(-1).squeeze(-1).squeeze(-1)
