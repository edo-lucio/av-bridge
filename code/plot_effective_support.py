r"""Effective support size of leg and composed plans across alpha.

Converts the mean row entropy already stored in each sweep CSV into a
more interpretable count: the "effective support size" of each plan's
row, defined as exp(mean row entropy in nats). It answers "on
average, how many caption rows does each row of the plan put mass
on?" 1.0 = perfect argmax, N = uniform.

Reading the plot: sharp plans (low alpha) have small effective
support; smooth plans (high alpha) have large effective support; the
composed bridge always has higher support than either leg because the
matrix product smooths further. The figure quantifies the
sharp-vs-wide axis behind the leg-vs-composition asymmetry.

Layout: 1 row x 2 cols, one panel per encoder regime.
  X-axis: alpha
  Y-axis: effective support size (log scale)
  Three lines per panel: Leg A, Leg B, Composed Transitive bridge.

Reads:
  results/exp_a{suffix}/sweep.csv        (Leg A entropy)
  results/exp_b{suffix}/sweep.csv        (Leg B entropy)
  results/exp_c{suffix}/sweep_transitive.csv  (composed entropy)

Usage:
  python code/plot_effective_support.py
  python code/plot_effective_support.py --K 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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
        "c_suffix": "",
        "a_suffix": "",
        "b_suffix": "",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "c_suffix": "__dinov2-large__mert-330m",
        "a_suffix": "__dinov2-large",
        "b_suffix": "__mert-330m",
    },
}

COL_LEG_A = "#1f77b4"   # blue
COL_LEG_B = "#9467bd"   # purple
COL_COMP  = "#d62728"   # red


def _support(csv: Path, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alphas, effective support sizes) at K, scope='heldout'."""
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[df.scope == "heldout"]
    if "K" in df.columns:
        df = df[df.K == K]
    if "plan_row_entropy" not in df.columns or df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    ent = df["plan_row_entropy"].values.astype(float)
    return df["alpha"].values.astype(float), np.exp(ent)


def render(out_path: Path, K: int) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.6),
                             squeeze=False)

    any_data = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax = axes[0, col_idx]

        xs, ys = _support(RES / f"exp_a{pair_spec['a_suffix']}" / "sweep.csv", K)
        if xs.size:
            any_data = True
            ax.plot(xs, ys, marker="o", lw=2.0, color=COL_LEG_A,
                    label="Leg A: image $\\to$ visual-text")

        xs, ys = _support(RES / f"exp_b{pair_spec['b_suffix']}" / "sweep.csv", K)
        if xs.size:
            any_data = True
            ax.plot(xs, ys, marker="s", lw=2.0, color=COL_LEG_B,
                    label="Leg B: audio $\\to$ audio-text")

        xs, ys = _support(RES / f"exp_c{pair_spec['c_suffix']}" / "sweep_transitive.csv", K)
        if xs.size:
            any_data = True
            ax.plot(xs, ys, marker="D", lw=2.0, color=COL_COMP,
                    label="Composed Transitive bridge")

        ax.set_xticks(ALPHA_GRID)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=9)
        ax.set_xlim(-0.05, 0.95)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$ structural)",
                      fontsize=10)
        ax.set_ylabel("effective support size  (= $\\exp(\\bar H)$)",
                      fontsize=10)
        ax.set_title(pair_spec["label"], fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax)

    if not any_data:
        print("[eff-support] no plan_row_entropy data found; "
              "regenerate CSVs via sbatch first.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=COL_LEG_A, lw=2.2, marker="o",
               label="Leg A: image $\\to$ visual-text"),
        Line2D([], [], color=COL_LEG_B, lw=2.2, marker="s",
               label="Leg B: audio $\\to$ audio-text"),
        Line2D([], [], color=COL_COMP, lw=2.2, marker="D",
               label="Composed Transitive bridge"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3,
               fontsize=10, frameon=False)
    fig.suptitle(
        rf"Plan width vs $\alpha$: effective number of caption rows "
        rf"per plan row ($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[eff-support] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"effective_support__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K)


if __name__ == "__main__":
    main()
