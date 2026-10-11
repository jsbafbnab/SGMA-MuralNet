"""Free-form irregular mask generation and area-binned sampling (binary, 1 = hole)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _draw_freeform(size: int, rng: np.random.RandomState,
                   max_vertex: int = 12, max_stroke: int = 6,
                   max_brush: int = 48, max_length: int = 120) -> np.ndarray:
    """Draw one free-form mask. Returns float32 (H, W) in {0, 1}."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    num_vertex = int(rng.randint(2, max_vertex + 1))
    angle = int(rng.randint(0, 360))
    for _ in range(num_vertex):
        x = int(rng.randint(0, size))
        y = int(rng.randint(0, size))
        strokes = int(rng.randint(1, max_stroke + 1))
        for _ in range(strokes):
            angle = (angle + int(rng.randint(-30, 31))) % 360
            length = int(rng.randint(10, max_length + 1))
            brush = int(rng.randint(4, max_brush + 1))
            rad = np.deg2rad(angle)
            x2 = int(np.clip(x + length * np.sin(rad), 0, size - 1))
            y2 = int(np.clip(y + length * np.cos(rad), 0, size - 1))
            draw.line([(x, y), (x2, y2)], fill=255, width=brush)
            x, y = x2, y2
    return (np.asarray(img).astype(np.float32) / 255.0)


def sample_freeform_mask(size: int, lo: float, hi: float,
                         rng: np.random.RandomState,
                         max_attempts: int = 2000) -> np.ndarray:
    """Generate a free-form mask whose hole fraction is in (lo, hi].

    Args:
        size: spatial size (square mask of size x size).
        lo, hi: target hole-area fraction bounds in (0, 1].
        rng: seeded RandomState for deterministic results.
    """
    # Scale brush size with the target ratio so higher bins are reachable.
    for attempt in range(max_attempts):
        # Gradually increase aggressiveness for hard (high-ratio) targets.
        brush = min(48, int(8 + hi * 80))
        mask = _draw_freeform(size, rng, max_brush=brush)
        ratio = float(mask.mean())
        if lo < ratio <= hi:
            return mask
    # Fallback: return the closest generated mask (guaranteed to be non-empty).
    best = None
    best_diff = float("inf")
    for _ in range(500):
        mask = _draw_freeform(size, rng, max_brush=min(48, int(8 + hi * 80)))
        ratio = float(mask.mean())
        diff = abs(ratio - (lo + hi) / 2)
        if diff < best_diff:
            best, best_diff = mask, diff
    return best


def generate_binned_masks(size: int, bins: List[Tuple[float, float]],
                          masks_per_bin: int, seed: int) -> List[List[np.ndarray]]:
    """Return ``masks_per_bin`` fixed masks for each area bin.

    The same masks are reused across every test image (5 fixed masks per bin).
    Deterministic given ``seed``.
    """
    masks: List[List[np.ndarray]] = []
    for bi, (lo, hi) in enumerate(bins):
        lo_f, hi_f = lo / 100.0, hi / 100.0
        bin_masks = []
        for mi in range(masks_per_bin):
            rng = np.random.RandomState(seed + bi * 1000 + mi)
            bin_masks.append(sample_freeform_mask(size, lo_f, hi_f, rng))
        masks.append(bin_masks)
    return masks


def mask_ratio(mask: np.ndarray) -> float:
    return float(mask.mean())
