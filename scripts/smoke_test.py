"""One-command smoke test: runs every entry point on a tiny synthetic dataset."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_synthetic(root: str, size: int = 256):
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        for kind in ("images", "edges", "masks"):
            os.makedirs(os.path.join(root, split, kind), exist_ok=True)
        for i in range(n):
            img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(root, split, "images", f"{i}.png"))
            edge = (np.random.rand(size, size) * 255).astype(np.uint8)
            Image.fromarray(edge, "L").save(os.path.join(root, split, "edges", f"{i}.png"))
            mask = ((np.random.rand(size, size) > 0.6).astype(np.uint8) * 255)
            Image.fromarray(mask, "L").save(os.path.join(root, split, "masks", f"{i}.png"))


def _write_config(path: str, data_root: str, out_dir: str):
    cfg = {
        "DATA_ROOT": data_root,
        "OUTPUT_DIR": out_dir,
        "INPUT_SIZE": 256,
        "MASK_BINS": [[10, 20]],
        "MASKS_PER_BIN": 1,
        "PHASE1_STEPS": 2, "PHASE1_BATCH": 2, "PHASE1_MICRO": 1,
        "PHASE2_STEPS": 1, "PHASE2_BATCH": 2, "PHASE2_MICRO": 1,
        "EVAL_INTERVAL": 1, "LOG_INTERVAL": 1, "SAVE_INTERVAL": 1000,
        "NUM_WORKERS": 0, "DEVICE": "cpu",
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def run(cmd, cwd, label):
    print(f"\n--- {label} ---", flush=True)
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(proc.stdout.splitlines()[-6:])
    print(tail)
    return proc.returncode


def build_args():
    p = argparse.ArgumentParser(description="Run the SGMA-MuralNet smoke test")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = build_args()
    py = sys.executable
    tmp = tempfile.mkdtemp(prefix="sgma_smoke_")
    try:
        data_root = args.data_root or os.path.join(tmp, "data")
        if not args.data_root:
            _make_synthetic(data_root)
            print(f"synthetic dataset: {data_root}")
        out_dir = os.path.join(tmp, "runs")
        cfg_path = os.path.join(tmp, "cfg.yaml")
        _write_config(cfg_path, data_root, out_dir)

        phase2_ckpt = os.path.join(out_dir, "phase2_best.pt")
        results = []

        checks = [
            ("verify_architecture", [py, "-m", "scripts.verify_architecture"]),
            ("train phase1", [py, "-m", "sgma_muralnet.train", "--phase", "1",
                              "--config", cfg_path, "--device", args.device]),
            ("train phase2", [py, "-m", "sgma_muralnet.train", "--phase", "2",
                              "--config", cfg_path, "--device", args.device]),
            ("test", [py, "-m", "sgma_muralnet.test", "--checkpoint", phase2_ckpt,
                      "--config", cfg_path, "--device", args.device]),
            ("evaluate", [py, "-m", "sgma_muralnet.evaluate", "--checkpoint", phase2_ckpt,
                          "--config", cfg_path, "--device", args.device]),
            ("benchmark", [py, "-m", "sgma_muralnet.benchmark",
                           "--config", cfg_path, "--device", args.device]),
            ("visualize", [py, "-m", "sgma_muralnet.visualize", "--checkpoint", phase2_ckpt,
                           "--config", cfg_path, "--data-root", data_root,
                           "--out-dir", os.path.join(tmp, "viz"), "--device", args.device]),
            ("make_flist", [py, "-m", "scripts.make_flist", "--data-root", data_root,
                            "--out", os.path.join(tmp, "flist")]),
            ("sift_screening", [py, "-m", "scripts.sift_screening", "--data-root", data_root,
                                "--limit", "2", "--workers", "1"]),
        ]
        for label, cmd in checks:
            results.append((label, run(cmd, PROJECT_ROOT, label)))

        print("\n\n===== summary =====")
        failed = 0
        for label, rc in results:
            status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
            if rc != 0:
                failed += 1
            print(f"  {label:<20} {status}")
        print(f"\n{len(results) - failed}/{len(results)} checks passed")
        sys.exit(1 if failed else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
