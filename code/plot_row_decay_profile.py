r"""Row-sorted cumulative mass profile of the composed Transitive plan
at low / mid / high alpha. Visualises the sharp-vs-wide distinction.

For each query row of a transport plan, sort its caption-mass values
descending and accumulate them. A sharp plan reaches cumulative = 1.0
within a handful of caption rows (steep cliff at small k); a smooth
plan climbs gradually (slow ramp). Averaging this cumulative profile
across query rows gives one curve per (alpha, regime, K).

Layout: 1 row x 2 cols, one panel per encoder regime.
  X-axis: rank position k within the row (log scale, 1 .. N).
  Y-axis: cumulative mass on top-k columns of the row.
  Three lines per panel: alpha in {0.0, 0.5, 0.9}.

Inputs:
  results/exp_c{suffix}/plans/T__K{K}__a{alpha:.2f}.npy

This script uses the composed Transitive plan. If leg plans
(T_iv, T_ac) are also saved per (K, alpha) by run_experiments.py
(latest patch), the script can be extended to overlay leg curves.
For now it visualises only the composition.

Usage:
  python code/plot_row_decay_profile.py
  python code/plot_row_decay_profile.py --K 300 --alphas 0.0 0.5 0.9
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


def _mean_sorted_cumulative(T: np.ndarray) -> np.ndarray:
    """Mean over rows of sort-descending cumulative profile."""
    row_sums = T.sum(axis=1, keepdims=True)
    P = T / np.maximum(row_sums, 1e-12)
    sorted_desc = -np.sort(-P, axis=1)
    cum = np.cumsum(sorted_desc, axis=1)
    return cum.mean(axis=0)


def render(out_path: Path, K: int, alphas: list[float]) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.4 * n_cols, 4.8),
                             squeeze=False)

    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(alphas)))
    colour_for = dict(zip(alphas, cmap))

    any_data = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax = axes[0, col_idx]
        n_eff = None
        for alpha in alphas:
            path = (RES / f"exp_c{pair_spec['suffix']}" /
                    "plans" / f"T__K{K}__a{alpha:.2f}.npy")
            if not path.exists():
                print(f"  [skip] {path}")
                continue
            T = np.load(path)
            cum = _mean_sorted_cumulative(T)
            ks = np.arange(1, T.shape[1] + 1)
            if n_eff is None:
                n_eff = T.shape[1]
            any_data = True
            ax.plot(ks, cum, lw=2.0, color=colour_for[alpha],
                    label=f"$\\alpha = {alpha:.1f}$")

        if n_eff is not None:
            # Random-plan reference: uniform row → cum = k/N (linear in k).
            ax.plot(np.arange(1, n_eff + 1),
                    np.arange(1, n_eff + 1) / n_eff,
                    color="grey", ls=":", lw=1.2,
                    label="uniform row (random)")

        ax.set_xscale("log")
        ax.set_xlabel(r"rank $k$ within the row  (log scale)",
                      fontsize=10)
        ax.set_ylabel(r"cumulative mass on top-$k$ caption rows",
                      fontsize=10)
        ax.set_title(pair_spec["label"], fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[row-decay] no composed plans at K={K}; "
              "regenerate via sbatch first.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=colour_for[a], lw=2.2,
               label=f"$\\alpha = {a:.1f}$")
        for a in alphas
    ] + [
        Line2D([], [], color="grey", ls=":", lw=1.2,
               label="uniform row (random)")
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04),
               ncol=len(handles), fontsize=10, frameon=False)
    fig.suptitle(
        rf"Sharp vs wide: row-sorted cumulative mass of the composed "
        rf"Transitive plan ($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[row-decay] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 0.9])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"row_decay_profile__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.alphas)


if __name__ == "__main__":
    main()
