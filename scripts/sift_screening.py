"""Cross-partition geometric screening with SIFT.

Screens every cross-partition image pair for overlapping content using SIFT
(max 5000 keypoints), FLANN 2-NN matching, Lowe's ratio 0.75, and RANSAC
homography (3.0-px reprojection). Pairs with >= 20 inliers are flagged. Runs in
parallel with multiprocessing.Pool.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm


def _list_images(partition_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(sorted(glob.glob(os.path.join(partition_dir, ext))))
    return files


def _load_gray(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


class SIFTScreener:
    def __init__(self, size: int = 256, max_keypoints: int = 5000,
                 ratio: float = 0.75, reproj: float = 3.0, min_inliers: int = 20):
        self.size = size
        self.ratio = ratio
        self.reproj = reproj
        self.min_inliers = min_inliers
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)
        flann_index = dict(algorithm=1, trees=5)   # KD-tree (SIFT float descriptors)
        flann_search = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(flann_index, flann_search)

    def screen(self, img_a: np.ndarray, img_b: np.ndarray):
        """Return (inliers, matches) for a pair of grayscale images."""
        kp_a, desc_a = self.sift.detectAndCompute(img_a, None)
        kp_b, desc_b = self.sift.detectAndCompute(img_b, None)
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return 0, 0
        knn = self.matcher.knnMatch(desc_a, desc_b, k=2)
        good = [m for m, n in knn if m.distance < self.ratio * n.distance]
        if len(good) < 4:
            return 0, len(good)
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj)
        if homography is None:
            return 0, len(good)
        return int(mask.sum()), len(good)


# Multiprocessing worker state (one SIFT/FLANN instance per worker process).
_WORKER: SIFTScreener | None = None


def _init_worker(size, max_keypoints, ratio, reproj, min_inliers):
    global _WORKER
    _WORKER = SIFTScreener(size, max_keypoints, ratio, reproj, min_inliers)


def _screen_one(task):
    """task = (combo, path_a, path_b) -> (combo, name_a, name_b, inliers, matches)."""
    combo, path_a, path_b = task
    name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)
    try:
        img_a = _load_gray(path_a, _WORKER.size)
        img_b = _load_gray(path_b, _WORKER.size)
        inliers, matches = _WORKER.screen(img_a, img_b)
    except Exception:
        return combo, name_a, name_b, -1, 0
    return combo, name_a, name_b, inliers, matches


def build_args():
    p = argparse.ArgumentParser(description="SIFT cross-partition screening (Section 4.1)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--max-keypoints", type=int, default=5000)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--reproj", type=float, default=3.0)
    p.add_argument("--min-inliers", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="cap pairs per combination (0 = all)")
    p.add_argument("--workers", type=int, default=0, help="processes (0 = cpu count)")
    p.add_argument("--out", type=str, default=None, help="write flagged pairs to file")
    return p.parse_args()


def main():
    args = build_args()
    root = args.data_root
    splits = {
        "train": _list_images(os.path.join(root, "train", "images")),
        "val": _list_images(os.path.join(root, "val", "images")),
        "test": _list_images(os.path.join(root, "test", "images")),
    }
    combos = [("train", "test"), ("train", "val"), ("val", "test")]

    tasks = []
    for a, b in combos:
        n = 0
        for pa in splits[a]:
            for pb in splits[b]:
                tasks.append(((a, b), pa, pb))
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break

    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")
    for a, b in combos:
        total = len(splits[a]) * len(splits[b])
        print(f"  {a}-{b}: {total} pairs{' (limited to %d)' % args.limit if args.limit else ''}")
    print(f"jobs: {len(tasks)}")

    workers = args.workers or os.cpu_count() or 1
    init = (args.size, args.max_keypoints, args.ratio, args.reproj, args.min_inliers)

    flagged = []
    with mp.Pool(workers, initializer=_init_worker, initargs=init) as pool:
        results = pool.imap_unordered(_screen_one, tasks, chunksize=32)
        for combo, name_a, name_b, inliers, matches in tqdm(
                results, total=len(tasks), desc="screening"):
            if inliers >= args.min_inliers:
                flagged.append((combo, name_a, name_b, inliers, matches))
                print(f"  FLAG {combo[0]}/{name_a} ~ {combo[1]}/{name_b}: "
                      f"{inliers} inliers ({matches} matches)", flush=True)

    print(f"\nTotal flagged pairs: {len(flagged)}")
    if args.out and flagged:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for (a, b), na, nb, inliers, matches in flagged:
                f.write(f"{a}\t{b}\t{na}\t{nb}\t{inliers}\t{matches}\n")
        print(f"flagged pairs written to {args.out}")


if __name__ == "__main__":
    main()
