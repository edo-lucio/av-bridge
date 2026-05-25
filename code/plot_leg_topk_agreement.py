r"""Top-k leg agreement vs alpha. The smoking-gun diagnostic for the
sharp-vs-wide leg-composition mechanism.

For each clip i, Leg A produces a ranking over caption rows (sort
of T_iv[i, :] descending); Leg B produces a ranking over the same
caption rows from row i of T_ac (the two legs map into the same
caption space under the AVCaps clip-identity correspondence). The
top-k agreement at k is the average Jaccard-style intersection size
of their top-k caption sets, divided by k.

Reading the plot: at low alpha, leg argmaxes (k = 1) tend to
DISAGREE (each leg picks one caption, often a different one).
Composition cancels. At higher alpha, top-1 agreement may stay flat
or drop further, but top-10 agreement RISES because the smoothing
gives each leg a top-10 window that overlaps non-trivially with the
other leg's top-10 window. That overlap is what survives the matrix
product as composed-bridge retrieval.

Layout: 1 row x 2 cols, one panel per encoder regime.
  X-axis: alpha.
  Y-axis: mean top-k overlap fraction.
  Three lines per panel: k = 1, 5, 10.

Inputs (after the next HPC sbatch with the patched run_experiments):
  results/exp_c{suffix}/plans/T_iv__K{K}__a{alpha:.2f}.npy
  results/exp_c{suffix}/plans/T_ac__K{K}__a{alpha:.2f}.npy

If leg plans are not yet saved, the script renders a placeholder
panel noting that an HPC re-run is required.

Usage:
  python code/plot_leg_topk_agreement.py
  python code/plot_leg_topk_agreement.py --K 300 --ks 1 5 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

PAIRS = {
    "text-grounded": {
        "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
        "suffix": "",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "suffix": "__dinov2-large__mert-330m",
    },
}


def _topk_agreement(T_iv: np.ndarray, T_ac: np.ndarray,
                    k: int) -> float:
    """Mean fraction of overlap between Leg A's and Leg B's top-k
    caption sets, computed row-wise on the clip-identity diagonal."""
    if T_iv.shape != T_ac.shape:
        return float("nan")
    n_rows, n_cols = T_iv.shape
    k = min(k, n_cols)
    # For each row i, leg A's top-k caption indices and leg B's top-k.
    topk_iv = np.argpartition(-T_iv, k - 1, axis=1)[:, :k]
    topk_ac = np.argpartition(-T_ac, k - 1, axis=1)[:, :k]
    overlaps = np.empty(n_rows)
    for i in range(n_rows):
        overlaps[i] = (np.intersect1d(topk_iv[i], topk_ac[i]).size) / k
    return float(overlaps.mean())


def render(out_path: Path, K: int, ks: list[int]) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.8),
                             squeeze=False)

    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(ks)))
    colour_for = dict(zip(ks, cmap))

    any_data = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax = axes[0, col_idx]
        per_k_xy: dict[int, tuple[list[float], list[float]]] = {
            k: ([], []) for k in ks
        }

        for alpha in ALPHA_GRID:
            iv_path = (RES / f"exp_c{pair_spec['suffix']}" /
                       "plans" / f"T_iv__K{K}__a{alpha:.2f}.npy")
            ac_path = (RES / f"exp_c{pair_spec['suffix']}" /
                       "plans" / f"T_ac__K{K}__a{alpha:.2f}.npy")
            if not (iv_path.exists() and ac_path.exists()):
                continue
            T_iv = np.load(iv_path)
            T_ac = np.load(ac_path)
            for k in ks:
                v = _topk_agreement(T_iv, T_ac, k)
                per_k_xy[k][0].append(alpha)
                per_k_xy[k][1].append(v)

        for k in ks:
            xs, ys = per_k_xy[k]
            if not xs:
                continue
            any_data = True
            ax.plot(xs, ys, marker="o", lw=2.0, color=colour_for[k],
                    label=f"top-{k}")

        ax.set_xticks(ALPHA_GRID)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=9)
        ax.set_xlim(-0.05, 0.95)
        ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$ structural)",
                      fontsize=10)
        ax.set_ylabel(r"mean leg top-$k$ overlap fraction",
                      fontsize=10)
        ax.set_title(pair_spec["label"], fontsize=11)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    if not any_data:
        msg = (
            "Leg plans (T_iv, T_ac) per (K, alpha) are not yet saved.\n"
            "Run sbatch with the patched run_experiments.py to populate\n"
            "results/exp_c*/plans/T_iv__K*__a*.npy and T_ac__K*__a*.npy."
        )
        for ax in axes.flat:
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10,
                    color="grey", style="italic",
                    bbox=dict(facecolor="white", edgecolor="lightgrey"))

    handles = [
        Line2D([], [], color=colour_for[k], lw=2.2, marker="o",
               label=f"top-{k}")
        for k in ks
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=len(ks),
               fontsize=10, frameon=False)
    fig.suptitle(
        rf"Sharp vs wide: leg top-$k$ agreement vs $\alpha$ ($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[leg-topk] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"leg_topk_agreement__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.ks)


if __name__ == "__main__":
    main()
