"""Benchmark evaluation: the 1250-pair protocol, guidance perturbations, and
the paired bootstrap comparison (Tables 7-9, 12).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .bootstrap import paired_bootstrap
from .config import Config
from .dataset import DhMurals1714
from .edge_f1 import edge_f1
from .masks import generate_binned_masks
from .metrics import LPIPSMetric, psnr, ssim
from .model import build_generator
from .perturbations import PERTURBATIONS
from .utils import load_checkpoint


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--baseline-checkpoint", type=str, default=None,
                   help="MuralNet (A0) checkpoint for the paired bootstrap (Table 9)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--perturb", action="store_true", help="run Table 12 perturbation study")
    p.add_argument("--bootstrap", action="store_true", help="run Table 9 paired bootstrap")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--perturb-realizations", type=int, default=5)
    return p.parse_args()


def _load_test_arrays(config) -> tuple:
    ds = DhMurals1714(config, config.EVAL_SPLIT)
    images, lines, masks = [], [], []
    for i in range(len(ds)):
        img, line, mask = ds[i]
        images.append(img.numpy())
        lines.append(line.squeeze(0).numpy())
        masks.append(mask.squeeze(0).numpy())
    return images, lines, masks


def _make_binned_masks(config) -> np.ndarray:
    """Fixed masks of shape (B, M, H, W): B bins x M masks per bin."""
    bins = [tuple(b) for b in config.MASK_BINS]
    per = config.MASKS_PER_BIN
    size = config.INPUT_SIZE
    grouped = generate_binned_masks(size, bins, per, config.EVAL_SEED)
    return np.stack([np.stack(g) for g in grouped])


@torch.no_grad()
def _forward(gen, img_np, line_np, mask_np, device):
    """img_np (3,H,W), line_np (H,W), mask_np (H,W) -> bI (1,3,H,W)."""
    img = torch.from_numpy(img_np).unsqueeze(0).to(device)
    line = torch.from_numpy(line_np).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    io = img * (1.0 - mask)
    return gen(io, line, mask)


def _fmt(x: float) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"


def _pair_metrics(b_i, img_t, mask_t, lpips_fn):
    full_psnr = psnr(b_i, img_t).item()
    hole_psnr = psnr(b_i, img_t, mask_t).item()
    ss = ssim(b_i, img_t).item()
    lp = lpips_fn(b_i, img_t).item() if lpips_fn.available else float("nan")

    b_np = (b_i[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    i_np = (img_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ef1 = edge_f1(b_np, i_np)
    return full_psnr, hole_psnr, ss, lp, ef1


def evaluate_pairs(gen, images, lines, masks_binned, device, lpips_fn):
    """Returns per-bin and overall metric dicts over all (bin, mask, image) pairs.

    masks_binned: (B, M, H, W) or None (use one mask per image).
    """
    n_bins = masks_binned.shape[0]
    n_masks = masks_binned.shape[1]
    n_img = len(images)
    metrics = {k: np.zeros((n_bins, n_masks, n_img)) for k in
               ("full_psnr", "hole_psnr", "ssim", "lpips", "edge_f1")}
    for bi in range(n_bins):
        for mi in range(n_masks):
            mask_np = masks_binned[bi, mi]
            for ii in range(n_img):
                img_np = images[ii]
                line_np = lines[ii]
                img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
                b_i = _forward(gen, img_np, line_np, mask_np, device)
                v = _pair_metrics(b_i, img_t, mask_t, lpips_fn)
                for k, val in zip(metrics, v):
                    metrics[k][bi, mi, ii] = val
    per_bin = {k: metrics[k].reshape(n_bins, -1).mean(axis=1) for k in metrics}
    overall = {k: float(metrics[k].mean()) for k in metrics}
    return overall, per_bin, metrics


def run_standard_eval(config, gen, device, lpips_fn):
    images, lines, _ = _load_test_arrays(config)
    masks_binned = _make_binned_masks(config)
    overall, per_bin, _ = evaluate_pairs(gen, images, lines, masks_binned, device, lpips_fn)
    return overall, per_bin


def run_perturbation_eval(config, gen, device, lpips_fn, realizations: int = 5):
    images, lines, _ = _load_test_arrays(config)
    masks_binned = _make_binned_masks(config)

    deterministic = {"shift_2", "shift_4", "stroke_thickness", "missing_face"}
    results = {}
    for name, fn in PERTURBATIONS.items():
        n_real = 1 if name in deterministic else realizations
        per_real = {"psnr": [], "lpips": []}
        for r in range(n_real):
            rng = np.random.RandomState(config.EVAL_SEED + 100000 + r)
            # Perturb every line drawing for this realization.
            pert_lines = [fn(l, rng) for l in lines]
            overall, _, _ = evaluate_pairs(gen, images, pert_lines, masks_binned,
                                           device, lpips_fn)
            per_real["psnr"].append(overall["full_psnr"])
            per_real["lpips"].append(overall["lpips"])
        results[name] = {
            "psnr_mean": float(np.mean(per_real["psnr"])),
            "psnr_std": float(np.std(per_real["psnr"])),
            "lpips_mean": float(np.mean(per_real["lpips"])),
            "lpips_std": float(np.std(per_real["lpips"])),
        }
    return results


def run_bootstrap(config, gen, base_gen, device, lpips_fn, n_replicates):
    images, lines, _ = _load_test_arrays(config)
    masks_binned = _make_binned_masks(config)

    def per_image_metrics(model):
        # Average over the 25 masks for each image -> per-image arrays.
        n_img = len(images)
        out = {k: np.zeros(n_img) for k in ("full_psnr", "hole_psnr", "ssim", "lpips")}
        n_bins, n_masks = masks_binned.shape[0], masks_binned.shape[1]
        for bi in range(n_bins):
            for mi in range(n_masks):
                mask_np = masks_binned[bi, mi]
                for ii in range(n_img):
                    b_i = _forward(model, images[ii], lines[ii], mask_np, device)
                    img_t = torch.from_numpy(images[ii]).unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
                    out["full_psnr"][ii] += psnr(b_i, img_t).item()
                    out["hole_psnr"][ii] += psnr(b_i, img_t, mask_t).item()
                    out["ssim"][ii] += ssim(b_i, img_t).item()
                    if lpips_fn.available:
                        out["lpips"][ii] += lpips_fn(b_i, img_t).item()
        return {k: v / (n_bins * n_masks) for k, v in out.items()}

    prop = per_image_metrics(gen)
    base = per_image_metrics(base_gen)
    results = {}
    for m in ("full_psnr", "hole_psnr", "ssim", "lpips"):
        results[m] = paired_bootstrap(prop[m], base[m], m, n_replicates,
                                      n_images=len(images), seed=config.EVAL_SEED)
    return results


def main():
    args = build_args()
    config = Config.from_yaml(args.config) if args.config else Config.from_defaults()
    if args.data_root:
        config.update({"DATA_ROOT": args.data_root})
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    gen = build_generator(config).to(device)
    load_checkpoint(args.checkpoint, gen, None, device=str(device))
    gen.eval()
    lpips_fn = LPIPSMetric()

    overall, per_bin = run_standard_eval(config, gen, device, lpips_fn)
    print(f"\n=== Standard 1250-pair evaluation (variant {config.VARIANT}) ===")
    print(f"Full PSNR : {overall['full_psnr']:.3f} dB")
    print(f"Hole PSNR : {overall['hole_psnr']:.3f} dB")
    print(f"SSIM      : {overall['ssim']:.4f}")
    print(f"LPIPS     : {_fmt(overall['lpips'])}")
    print(f"Edge F1   : {overall['edge_f1']:.4f}")
    print("\nPer-bin (Table 8):")
    print(f"{'bin %':>10} {'Full PSNR':>10} {'Hole PSNR':>10} {'SSIM':>7} {'LPIPS':>7} {'Edge F1':>8}")
    for bi, (lo, hi) in enumerate(config.MASK_BINS):
        print(f"({lo},{hi}] {per_bin['full_psnr'][bi]:>9.2f} {per_bin['hole_psnr'][bi]:>10.2f} "
              f"{per_bin['ssim'][bi]:>7.3f} {_fmt(per_bin['lpips'][bi]):>7} {per_bin['edge_f1'][bi]:>8.3f}")

    if args.perturb:
        pert = run_perturbation_eval(config, gen, device, lpips_fn,
                                     args.perturb_realizations)
        print("\n=== Guidance perturbation robustness (Table 12) ===")
        print(f"{'condition':<22} {'PSNR mean+/-std':>19} {'LPIPS mean+/-std':>19}")
        for name, r in pert.items():
            lp = _fmt(r["lpips_mean"]) + (f"+/-{r['lpips_std']:.3f}" if not np.isnan(r["lpips_mean"]) else "")
            print(f"{name:<22} {r['psnr_mean']:.2f}+/-{r['psnr_std']:.2f}  {lp:>19}")

    if args.bootstrap and args.baseline_checkpoint:
        base_config = Config.from_yaml(args.config) if args.config else Config.from_defaults()
        base_config.update({"VARIANT": "A0"})
        base_gen = build_generator(base_config).to(device)
        load_checkpoint(args.baseline_checkpoint, base_gen, None, device=str(device))
        base_gen.eval()
        boot = run_bootstrap(config, gen, base_gen, device, lpips_fn, args.n_bootstrap)
        print("\n=== Paired bootstrap vs baseline (Table 9) ===")
        print(f"{'metric':<10} {'mean_diff':>9}  {'ci_low':>9}  {'ci_high':>9}")
        for m, r in boot.items():
            print(f"{m:<10} {r['mean_diff']:>+9.4f}  {r['ci_low']:>+9.4f}  {r['ci_high']:>+9.4f}")


if __name__ == "__main__":
    main()
