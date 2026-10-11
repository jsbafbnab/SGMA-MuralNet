"""Inference and evaluation: restore a checkpoint and report PSNR / SSIM / LPIPS."""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import DhMurals1714
from .metrics import LPIPSMetric, psnr, ssim
from .model import build_generator
from .utils import create_dir, load_checkpoint, save_image


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SGMA-MuralNet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None, help="save restored images")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = build_args()
    config = Config.from_yaml(args.config) if args.config else Config.from_defaults()
    if args.data_root:
        config.update({"DATA_ROOT": args.data_root})

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = DhMurals1714(config, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    generator = build_generator(config).to(device)
    load_checkpoint(args.checkpoint, generator, None, device=str(device))
    generator.eval()

    lpips_fn = LPIPSMetric()

    full_psnr = hole_psnr = ssim_sum = lpips_sum = 0.0
    n = 0

    with torch.no_grad():
        for i, (img, line, mask) in enumerate(tqdm(loader, desc=args.split)):
            img = img.to(device); line = line.to(device); mask = mask.to(device)
            io = img * (1.0 - mask)
            _, _, _, _, b_i = generator(io, line, mask, return_all=True)

            full_psnr += psnr(b_i, img).item()
            hole_psnr += psnr(b_i, img, mask).item()
            ssim_sum += ssim(b_i, img).item()
            if lpips_fn.available:
                lpips_sum += lpips_fn(b_i, img).item()
            n += 1

            if args.out_dir:
                create_dir(args.out_dir)
                name = os.path.splitext(os.path.basename(dataset.images[i]))[0]
                save_image(b_i[0], os.path.join(args.out_dir, f"{name}.png"))

    print(f"Split: {args.split} ({n} samples)")
    print(f"Full PSNR : {full_psnr / n:.3f} dB")
    print(f"Hole PSNR : {hole_psnr / n:.3f} dB")
    print(f"SSIM      : {ssim_sum / n:.4f}")
    if lpips_fn.available:
        print(f"LPIPS     : {lpips_sum / n:.4f}")
    else:
        print("LPIPS     : skipped (pip install lpips)")


if __name__ == "__main__":
    main()
