r"""Mean diagonal mass of the composed Transitive bridge vs alpha.

For each query i, the composed plan entry T_ia[i, i] is the mass the
plan places on the CORRECT clip-identity pairing. Averaging over
queries gives a single scalar per (alpha, K, regime) that directly
drives retrieval: high diagonal mass = the correct pairing is the
plan's argmax for many queries.

Reading the plot: the peak of diagonal-mass curve across alpha should
roughly co-locate with the peak of R@10 across alpha. This closes the
loop between "sharp legs cancel under the matrix product" (intuition)
and "the composed plan puts more mass on the correct diagonal at
intermediate alpha" (measurable).

Layout: 1 row x 2 cols, one panel per encoder regime.
  Left y-axis: mean diagonal mass of composed plan T_ia.
  Right y-axis: composed bridge R@10 (from sweep_transitive.csv).
  X-axis: alpha.

Inputs:
  results/exp_c{suffix}/plans/T__K{K}__a{alpha:.2f}.npy
  results/exp_c{suffix}/sweep_transitive.csv  (for R@10 overlay)

Usage:
  python code/plot_diagonal_mass.py
  python code/plot_diagonal_mass.py --K 300
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
        "suffix": "",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "suffix": "__dinov2-large__mert-330m",
    },
}

COL_DIAG = "#1f77b4"
COL_R10  = "#d62728"


def _mean_diag(T: np.ndarray) -> float:
    # Row-normalise first (defensive — saved plans should already be).
    row_sums = T.sum(axis=1, keepdims=True)
    P = T / np.maximum(row_sums, 1e-12)
    return float(np.diag(P).mean())


def _composed_r10(suffix: str, K: int) -> tuple[np.ndarray, np.ndarray]:
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[(df.scope == "heldout") & (df.K == K)].sort_values("alpha")
    if df.empty:
        return np.array([]), np.array([])
    return df["alpha"].values.astype(float), df["R@10"].values.astype(float)


def render(out_path: Path, K: int) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.4 * n_cols, 4.8),
                             squeeze=False)

    any_data = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax = axes[0, col_idx]
        ax_r = ax.twinx()

        xs_d, ys_d = [], []
        for alpha in ALPHA_GRID:
            path = (RES / f"exp_c{pair_spec['suffix']}" /
                    "plans" / f"T__K{K}__a{alpha:.2f}.npy")
            if not path.exists():
                continue
            T = np.load(path)
            xs_d.append(alpha)
            ys_d.append(_mean_diag(T))
        if xs_d:
            any_data = True
            ax.plot(xs_d, ys_d, marker="o", lw=2.0, color=COL_DIAG,
                    label="mean diagonal mass")
            i_peak = int(np.argmax(ys_d))
            ax.scatter([xs_d[i_peak]], [ys_d[i_peak]], s=160, marker="*",
                       facecolor=COL_DIAG, edgecolor="black",
                       linewidth=0.8, zorder=6)
            ax.annotate(
                rf"diag-peak  $\alpha = {xs_d[i_peak]:.1f}$",
                xy=(xs_d[i_peak], ys_d[i_peak]),
                xytext=(0, 14), textcoords="offset points",
                color=COL_DIAG, fontsize=8.5, ha="center",
                fontweight="bold",
            )

        xs_r, ys_r = _composed_r10(pair_spec["suffix"], K)
        if xs_r.size:
            any_data = True
            ax_r.plot(xs_r, ys_r, marker="D", lw=2.0, color=COL_R10,
                      label=f"composed bridge $R@10$")
            i_peak = int(np.nanargmax(ys_r))
            ax_r.scatter([xs_r[i_peak]], [ys_r[i_peak]], s=120,
                         marker="X", facecolor=COL_R10, edgecolor="black",
                         linewidth=0.8, zorder=6)
            ax_r.annotate(
                rf"$R@10$ peak  $\alpha = {xs_r[i_peak]:.1f}$",
                xy=(xs_r[i_peak], ys_r[i_peak]),
                xytext=(0, -16), textcoords="offset points",
                color=COL_R10, fontsize=8.5, ha="center",
                fontweight="bold",
            )

        ax.set_xticks(ALPHA_GRID)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=9)
        ax.set_xlim(-0.05, 0.95)
        ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$ structural)",
                      fontsize=10)
        ax.set_ylabel(r"mean diagonal mass  $\langle T_{ia}[i,i]\rangle_i$",
                      fontsize=10, color=COL_DIAG)
        ax.tick_params(axis="y", labelcolor=COL_DIAG)
        ax_r.set_ylabel(r"composed bridge $R@10$  (heldout)",
                        fontsize=10, color=COL_R10)
        ax_r.tick_params(axis="y", labelcolor=COL_R10)
        ax.set_title(pair_spec["label"], fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax_r.grid(False)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[diag-mass] no composed plans at K={K}.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=COL_DIAG, lw=2.2, marker="o",
               label=r"mean diagonal mass  $\langle T_{ia}[i,i]\rangle$"),
        Line2D([], [], color=COL_R10, lw=2.2, marker="D",
               label=r"composed bridge $R@10$ (heldout)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2,
               fontsize=10, frameon=False)
    fig.suptitle(
        rf"Composed plan diagonal mass tracks $R@10$ across $\alpha$ "
        rf"($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag-mass] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"diagonal_mass__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K)


if __name__ == "__main__":
    main()
