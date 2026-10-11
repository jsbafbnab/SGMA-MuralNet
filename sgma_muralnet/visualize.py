"""Affine modulation parameter heatmaps (mean |a| and |b| per scale, Figure 6)."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import tensor_to_image


def _magnitude(t: torch.Tensor) -> np.ndarray:
    """(B, C, H, W) -> (H, W) float32 magnitude map (mean |.| over channels)."""
    return t.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()


def _heatmap(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return (m * 255).astype(np.uint8)


def compute_affine_maps(generator, line: torch.Tensor):
    """Return {(scale, 'a'|'b'): (H, W) uint8 heatmap} upsampled to input size."""
    line = line.detach()
    maps = generator.srn.affine_maps(line)
    out = {}
    in_h, in_w = line.shape[-2:]
    for (scale, kind), t in maps.items():
        m = _heatmap(_magnitude(t))             # (H_i, W_i) -> uint8
        m_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        m_t = F.interpolate(m_t, size=(in_h, in_w), mode="nearest").squeeze().numpy()
        out[(scale, kind)] = m_t.astype(np.uint8)
    return out


def save_affine_figure(generator, line: torch.Tensor, out_dir: str,
                       name: str = "affine_maps") -> list:
    """Save scale-deviation (S) and shift (B) heatmaps with line overlay as PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    maps = compute_affine_maps(generator, line)
    line_gray = np.asarray(tensor_to_image(line.squeeze(0)).convert("L")).astype(np.float32) / 255.0

    saved = []
    for (scale, kind), m in sorted(maps.items()):
        overlay = np.stack([m, m, m], axis=-1)
        line_mask = (line_gray > 0.5)[..., None]
        overlay = np.where(line_mask, np.array([255, 0, 0], dtype=np.uint8), overlay)
        path = os.path.join(out_dir, f"{name}_scale{scale}_{kind}.png")
        Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(path)
        saved.append(path)
    return saved


if __name__ == "__main__":
    import argparse

    from .config import Config
    from .dataset import DhMurals1714
    from .model import build_generator
    from .utils import load_checkpoint

    p = argparse.ArgumentParser(description="Visualize affine modulation parameters (Figure 6)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--image-index", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./visualizations")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config.from_defaults()
    if args.data_root:
        cfg.update({"DATA_ROOT": args.data_root})
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    gen = build_generator(cfg).to(device)
    load_checkpoint(args.checkpoint, gen, None, device=str(device))
    gen.eval()

    ds = DhMurals1714(cfg, args.split)
    img, line, _ = ds[args.image_index]
    line = line.unsqueeze(0).to(device)

    with torch.no_grad():
        paths = save_affine_figure(gen, line, args.out_dir,
                                   name=f"{args.split}_{args.image_index}")
    print("saved:")
    for path in paths:
        print(f"  {path}")
