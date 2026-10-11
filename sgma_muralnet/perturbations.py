"""Line-guidance perturbations (Table 12)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, map_coordinates
from skimage.morphology import label


def delete_line_pixels(line: np.ndarray, fraction: float,
                       rng: np.random.RandomState) -> np.ndarray:
    """Randomly delete ``fraction`` of foreground line pixels (10%/30%)."""
    out = line.copy()
    fg = np.argwhere(line > 0.5)
    n_del = int(round(len(fg) * fraction))
    if n_del > 0 and len(fg) > 0:
        chosen = rng.choice(len(fg), size=min(n_del, len(fg)), replace=False)
        for y, x in fg[chosen]:
            out[y, x] = 0.0
    return out


def break_contours(line: np.ndarray, rng: np.random.RandomState,
                   max_contours: int = 20, segment_len: int = 10) -> np.ndarray:
    """Delete ``segment_len``-pixel segments on up to ``max_contours`` contours.

    Connected components (contours) are labelled with 8-connectivity; for each
    selected contour a short 8-connected segment starting at a random foreground
    pixel is removed.
    """
    out = line.copy()
    binary = (line > 0.5).astype(np.int32)
    comps, n = label(binary, connectivity=2, return_num=True)
    if n == 0:
        return out
    chosen = rng.permutation(np.arange(1, n + 1))[:max_contours]
    for cid in chosen:
        ys, xs = np.where(comps == cid)
        if len(ys) < segment_len:
            continue
        # Start from a random pixel and greedily delete an 8-connected segment.
        start = int(rng.randint(len(ys)))
        frontier = [(ys[start], xs[start])]
        seen = set(frontier)
        for _ in range(segment_len):
            if not frontier:
                break
            y, x = frontier.pop(0)
            out[y, x] = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < out.shape[0] and 0 <= nx < out.shape[1]
                            and binary[ny, nx] and (ny, nx) not in seen):
                        seen.add((ny, nx))
                        frontier.append((ny, nx))
    return out


def _elastic(line: np.ndarray, alpha: float, sigma: float,
             rng: np.random.RandomState) -> np.ndarray:
    shape = line.shape
    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return map_coordinates(line, [y + dy, x + dx], order=1, mode="constant",
                           cval=0.0).reshape(shape)


def inaccurate_patterns(line: np.ndarray, rng: np.random.RandomState,
                        alpha: float = 4.0, sigma: float = 6.0) -> np.ndarray:
    """Elastic distortion of decorative motifs (Table 12, stochastic)."""
    return _elastic(line, alpha, sigma, rng)


def nonrigid_misalignment(line: np.ndarray, rng: np.random.RandomState,
                          control_spacing: int = 16, max_disp: float = 4.0) -> np.ndarray:
    """Elastic grid distortion: 16-px control spacing, 4-px max displacement."""
    sigma = control_spacing / 3.0
    return _elastic(line, max_disp, sigma, rng)


def horizontal_shift(line: np.ndarray, pixels: int) -> np.ndarray:
    """Rigid rightward shift; shifted-in pixels are zero-filled."""
    out = np.zeros_like(line)
    if pixels < line.shape[1]:
        out[:, pixels:] = line[:, :-pixels]
    return out


def stroke_thickness(line: np.ndarray) -> np.ndarray:
    """Mean of one dilation and one erosion (3x3 kernel)."""
    binary = line > 0.5
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(binary, struct)
    eroded = binary_erosion(binary, struct)
    return (dilated.astype(np.float32) + eroded.astype(np.float32)) / 2.0


def missing_facial_features(line: np.ndarray, image_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """Mask structural lines inside face bounding boxes (deterministic).

    Requires ``mediapipe`` and a visible face. If unavailable (or no face is
    detected) the line drawing is returned unchanged.
    """
    out = line.copy()
    if image_rgb is None:
        return out
    try:
        import mediapipe as mp
    except Exception:
        return out
    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=8,
                                             min_detection_confidence=0.5) as fm:
            results = fm.process((image_rgb * 255).astype(np.uint8))
        if not results.multi_face_landmarks:
            return out
        h, w = line.shape
        for lm in results.multi_face_landmarks:
            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]
            x0, x1 = max(0, min(xs) - 5), min(w, max(xs) + 5)
            y0, y1 = max(0, min(ys) - 5), min(h, max(ys) + 5)
            out[y0:y1, x0:x1] = 0.0
    except Exception:
        pass
    return out


def apply_external_sketch(line: np.ndarray, sketch_path: str) -> np.ndarray:
    """Replace the line drawing with an externally prepared sketch."""
    from PIL import Image
    with Image.open(sketch_path) as im:
        sketch = np.asarray(im.convert("L")).astype(np.float32) / 255.0
    return sketch


PERTURBATIONS = {
    "delete_10": lambda line, rng: delete_line_pixels(line, 0.10, rng),
    "delete_30": lambda line, rng: delete_line_pixels(line, 0.30, rng),
    "contour_break": lambda line, rng: break_contours(line, rng=rng),
    "inaccurate_patterns": lambda line, rng: inaccurate_patterns(line, rng),
    "nonrigid": lambda line, rng: nonrigid_misalignment(line, rng),
    "shift_2": lambda line, rng: horizontal_shift(line, 2),
    "shift_4": lambda line, rng: horizontal_shift(line, 4),
    "stroke_thickness": lambda line, rng: stroke_thickness(line),
    "missing_face": lambda line, rng: missing_facial_features(line),
}
