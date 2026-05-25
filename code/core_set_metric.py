"""Adaptive per-row core-set hit-rate metric for transport plans.

For each row i of a row-stochastic plan T:

    mu_i    = mean(T[i, :])
    sigma_i = std(T[i, :], ddof=0)
    core_i  = { j : T[i, j] > mu_i + sigma_i }
    hit_i   = (gt[i] in core_i)

Two summary numbers are returned:

    hit_rate       = (#rows with hit_i = True) / n
    mean_core_size = mean over rows of |core_i|

Both numbers together are what makes the metric meaningful. A plan that
spreads mass uniformly has a huge core_i and trivially high hit_rate; a
plan that is sharp but wrong has a small core_i and a low hit_rate. The
pair (hit_rate, mean_core_size) distinguishes these regimes.

This metric only makes sense on row-stochastic plans (T whose rows are
probability distributions over targets). On raw similarity matrices
(text-only / Procrustes baselines) the threshold "mu + sigma" has no
distributional interpretation; use R@k for those instead.

Usage as a module:

    from core_set_metric import core_set_hit_rate
    hit, size = core_set_hit_rate(T, gt)

Usage as a CLI:

    python code/core_set_metric.py results/exp_c/T_transitive.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def core_set_hit_rate(
    T: np.ndarray,
    gt: np.ndarray,
    row_subset: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (hit_rate, mean_core_size) on T evaluated against gt.

    Parameters
    ----------
    T : (n_rows, n_cols) float array. Should be row-stochastic for the
        metric to be meaningful.
    gt : (n_rows,) int array. gt[i] is the correct target column for
         row i. Values must be valid column indices.
    row_subset : optional (m,) int array. If provided, the metric is
        computed only over rows in this subset. Per-row mu and sigma
        are still computed over the row's full set of columns; only
        the averaging at the end is restricted to the subset.

    Returns
    -------
    (hit_rate, mean_core_size) : tuple of floats.
    """
    if T.ndim != 2:
        raise ValueError(f"T must be 2-D, got shape {T.shape}")
    n_rows, n_cols = T.shape
    if gt.shape != (n_rows,):
        raise ValueError(f"gt must have shape ({n_rows},), got {gt.shape}")
    if gt.min() < 0 or gt.max() >= n_cols:
        raise ValueError(f"gt values out of column range [0, {n_cols - 1}]")

    mu = T.mean(axis=1, keepdims=True)
    sigma = T.std(axis=1, ddof=0, keepdims=True)
    threshold = mu + sigma
    core_mask = T > threshold
    gt_mass = T[np.arange(n_rows), gt][:, None]
    hits = (gt_mass > threshold).flatten()
    core_sizes = core_mask.sum(axis=1)

    if row_subset is not None:
        row_subset = np.asarray(row_subset, dtype=int)
        if row_subset.size == 0:
            return float("nan"), float("nan")
        hits = hits[row_subset]
        core_sizes = core_sizes[row_subset]

    return float(hits.mean()), float(core_sizes.mean())


def lift(hit_rate: float, mean_core_size: float, n_cols: int) -> float:
    """Chance-corrected lift: 1.0 = chance, >1 = above chance."""
    if mean_core_size <= 0:
        return float("nan")
    return hit_rate * n_cols / mean_core_size


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", type=Path, help="path to an .npy transport plan")
    ap.add_argument(
        "--heldout-idx",
        type=Path,
        default=None,
        help="optional .npy of column indices defining a 'heldout' "
        "row subset (rows i with i in S_compare are excluded; rows i not "
        "in S_compare are evaluated).",
    )
    args = ap.parse_args()

    T = np.load(args.plan)
    n = T.shape[0]
    gt = np.arange(n)

    print(f"plan: {args.plan}  shape={T.shape}")
    print(f"  row sums in [{T.sum(axis=1).min():.6f}, {T.sum(axis=1).max():.6f}]")
    if not np.allclose(T.sum(axis=1), T.sum(axis=1).mean(), atol=1e-3):
        print("  WARNING: rows not row-stochastic; metric interpretation is degraded.")

    hit, size = core_set_hit_rate(T, gt)
    print(f"  aggregate (all rows): hit={hit:.3f}  |core|={size:.1f}  "
          f"lift={lift(hit, size, T.shape[1]):.2f}x")

    if args.heldout_idx is not None and args.heldout_idx.exists():
        S = np.load(args.heldout_idx)
        heldout = np.setdiff1d(np.arange(n), S)
        hit_h, size_h = core_set_hit_rate(T, gt, row_subset=heldout)
        print(f"  heldout ({len(heldout)} rows):  "
              f"hit={hit_h:.3f}  |core|={size_h:.1f}  "
              f"lift={lift(hit_h, size_h, T.shape[1]):.2f}x")


if __name__ == "__main__":
    _cli()
