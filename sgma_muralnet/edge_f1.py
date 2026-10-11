"""Edge F1 metric: Canny edge maps compared under a distance tolerance."""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation, disk


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return rgb2gray(img)
    return img


def edge_f1(pred: np.ndarray, target: np.ndarray, sigma: float = 2.0,
            tolerance: int = 1) -> float:
    """F1 score between the Canny edges of two images.

    Args:
        pred, target: (H, W) or (H, W, 3) arrays in [0, 1] or [0, 255].
        sigma: Canny smoothing sigma.
        tolerance: radius (px) within which an edge matches.
    """
    pred_edge = canny(_to_gray(pred.astype(np.float64) / 255.0), sigma=sigma)
    gt_edge = canny(_to_gray(target.astype(np.float64) / 255.0), sigma=sigma)

    # Dilate ground-truth edges by the tolerance to define "matching" pixels.
    selem = disk(tolerance)
    gt_dilated = dilation(gt_edge, selem)
    pred_dilated = dilation(pred_edge, selem)

    tp = int(np.logical_and(pred_edge, gt_dilated).sum())
    fp = int(pred_edge.sum()) - tp
    # A ground-truth edge pixel is "missed" if no predicted edge is nearby.
    fn = int(np.logical_and(gt_edge, np.logical_not(pred_dilated)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1
