"""Run the three-seed training protocol (Table 4).

For each seed this runs Phase 1 (SRN) then Phase 2 (SRN + CCN), writing
checkpoints to a per-seed output directory.

Usage:
    python -m scripts.run_multiseed --config config/default.yaml \
        --data-root <DhMurals1714> [--variant A3]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260913, 20260914, 20260915])
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = build_args()
    from sgma_muralnet.config import Config
    config = Config.from_yaml(args.config) if args.config else Config.from_defaults()
    base_out = config.OUTPUT_DIR
    variant = args.variant or config.VARIANT

    for seed in args.seeds:
        out = os.path.join(base_out, f"{variant}_seed{seed}")
        print(f"\n===== seed {seed} -> {out} =====", flush=True)
        for phase in (1, 2):
            cmd = [sys.executable, "-m", "sgma_muralnet.train",
                   "--phase", str(phase), "--data-root", args.data_root,
                   "--output-dir", out, "--seed", str(seed)]
            if args.config:
                cmd += ["--config", args.config]
            if args.variant:
                cmd += ["--variant", args.variant]
            if args.device:
                cmd += ["--device", args.device]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
