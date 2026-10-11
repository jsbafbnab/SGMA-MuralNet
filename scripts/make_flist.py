"""Generate .flist text files for the DhMurals1714 layout.

Usage:
    python -m scripts.make_flist --data-root ./data/DhMurals1714 --out ./data/flist

Writes {train,val,test}_{images,edges,masks}.flist pointing at every file in
the corresponding subdirectory.
"""
from __future__ import annotations

import argparse
import glob
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for split in ("train", "val", "test"):
        for kind in ("images", "edges", "masks"):
            d = os.path.join(args.data_root, split, kind)
            files = []
            for ext in exts:
                files.extend(sorted(glob.glob(os.path.join(d, ext))))
            if not files:
                print(f"[skip] {d} empty")
                continue
            out = os.path.join(args.out, f"{split}_{kind}.flist")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(files) + "\n")
            print(f"[ok] {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
