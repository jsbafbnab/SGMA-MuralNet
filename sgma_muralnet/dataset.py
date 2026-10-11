"""DhMurals1714 dataset loader.

Expected layout: DATA_ROOT/{train,val,test}/{images,edges,masks} (or .flist
files). Edges are loaded as grayscale and inverted so line pixels == 1; masks
are binary with 1 == hole.
"""
from __future__ import annotations

import glob
import os
import random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_flist(path: Optional[str], fallback_dir: Optional[str]) -> List[str]:
    if path:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.*")))
        else:
            with open(path, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f if line.strip()]
        return files
    if fallback_dir and os.path.isdir(fallback_dir):
        return sorted(glob.glob(os.path.join(fallback_dir, "*.*")))
    return []


def _read(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0


def _read_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float32) / 255.0


def _read_mask(path: str) -> np.ndarray:
    return (_read_gray(path) > 0.5).astype(np.float32)


def _resize(img: np.ndarray, size: int, *, nearest: bool = False) -> np.ndarray:
    if img.ndim == 3:
        h, w = img.shape[:2]
    else:
        h, w = img.shape
    if h == size and w == size:
        return img
    mode = Image.NEAREST if nearest else Image.BICUBIC
    if img.ndim == 3:
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(arr).resize((size, size), mode)).astype(np.float32) / 255.0
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = np.asarray(Image.fromarray(arr).resize((size, size), mode)).astype(np.float32) / 255.0
    return out


class DhMurals1714(Dataset):
    def __init__(self, config, split: str = "train"):
        self.size = config.INPUT_SIZE
        self.invert_line = config.LINE_INVERT
        self.mask_type = config.MASK_TYPE
        self.augment = split == "train"
        self.split = split

        key = split.upper()
        self.images = _load_flist(config.get(f"{key}_FLIST"),
                                  os.path.join(config.DATA_ROOT, split, "images"))
        self.edges = _load_flist(config.get(f"{key}_EDGE_FLIST"),
                                 os.path.join(config.DATA_ROOT, split, "edges"))
        self.masks = _load_flist(config.get(f"{key}_MASK_FLIST"),
                                 os.path.join(config.DATA_ROOT, split, "masks"))

        assert len(self.images) > 0, f"no images found for split {split!r}"
        assert len(self.edges) == len(self.images), \
            f"image/edge count mismatch in {split}: {len(self.images)} vs {len(self.edges)}"
        if not self.masks:
            print(f"[dataset] warning: no mask files for {split}; using synthetic masks.")

    def __len__(self) -> int:
        return len(self.images)

    def _load_line(self, idx: int) -> np.ndarray:
        line = _read_gray(self.edges[idx])
        if self.invert_line:
            line = 1.0 - line
        return line

    def _load_mask(self, idx: int) -> np.ndarray:
        if self.masks:
            # Test mode uses a one-to-one mapping; train/val draw randomly.
            mi = idx if self.split == "test" else random.randint(0, len(self.masks) - 1)
            return _read_mask(self.masks[mi])
        # Synthetic fallbacks
        h = w = self.size
        mask = np.zeros((h, w), dtype=np.float32)
        if self.mask_type == "half":
            bw = w // 2
            x0 = 0 if random.random() < 0.5 else bw
            mask[:, x0:x0 + bw] = 1.0
        else:  # random_block
            bh, bw = h // 2, w // 2
            y0 = random.randint(0, h - bh)
            x0 = random.randint(0, w - bw)
            mask[y0:y0 + bh, x0:x0 + bw] = 1.0
        return mask

    def __getitem__(self, idx: int):
        img = _read(self.images[idx])
        line = self._load_line(idx)
        mask = self._load_mask(idx)

        img = _resize(img, self.size)
        line = _resize(line, self.size)
        mask = _resize(mask, self.size, nearest=True)
        mask = (mask > 0.5).astype(np.float32)

        if self.augment and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            line = np.ascontiguousarray(line[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        line_t = torch.from_numpy(line).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, line_t, mask_t
