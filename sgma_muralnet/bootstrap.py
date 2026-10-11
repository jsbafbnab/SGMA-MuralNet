"""Paired bootstrap significance testing (10,000 resamples, 95% CI).

PSNR/SSIM differences are oriented as (proposed - baseline), LPIPS as
(baseline - proposed).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def paired_bootstrap(proposed: np.ndarray, baseline: np.ndarray,
                     metric: str, n_replicates: int = 10000,
                     n_images: int = 50, seed: int = 0) -> Dict[str, float]:
    """``proposed``/``baseline`` are per-image metric arrays of equal length."""
    assert proposed.shape == baseline.shape
    if metric.lower() == "lpips":
        diff = baseline - proposed          # lower is better
    else:
        diff = proposed - baseline          # higher is better

    rng = np.random.RandomState(seed)
    means = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.randint(0, len(diff), size=n_images)
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "seed_std": None,   # filled in by the caller if multiple runs are available
    }


def paired_bootstrap_multi_run(proposed_runs: List[np.ndarray],
                               baseline_runs: List[np.ndarray],
                               metric: str, n_replicates: int = 10000,
                               n_images: int = 50, seed: int = 0) -> Dict[str, float]:
    """Bootstrap over image-level means, with between-seed std across runs.

    ``proposed_runs``/``baseline_runs`` are lists of per-image arrays, one per
    independent training run. Metrics are first averaged over masks (already
    done by the caller), then over runs.
    """
    # Pool image-level differences across runs (run-average per image).
    p = np.mean(np.stack(proposed_runs, axis=0), axis=0)
    b = np.mean(np.stack(baseline_runs, axis=0), axis=0)
    result = paired_bootstrap(p, b, metric, n_replicates, n_images, seed)

    # Between-seed variability of the run-level mean paired improvement.
    run_means = []
    for pr, br in zip(proposed_runs, baseline_runs):
        if metric.lower() == "lpips":
            run_means.append(float((br - pr).mean()))
        else:
            run_means.append(float((pr - br).mean()))
    result["seed_std"] = float(np.std(run_means))
    return result


def summarize_table9(metrics: Dict[str, Dict[str, float]]) -> str:
    """Format a Table-9-like summary from a dict of per-metric bootstrap results."""
    lines = ["metric       mean_diff   ci_low   ci_high   seed_std"]
    for m, r in metrics.items():
        sd = f"{r['seed_std']:.4f}" if r.get("seed_std") is not None else "-"
        lines.append(f"{m:<12} {r['mean_diff']:+.4f}   {r['ci_low']:+.4f}   "
                     f"{r['ci_high']:+.4f}   {sd}")
    return "\n".join(lines)
