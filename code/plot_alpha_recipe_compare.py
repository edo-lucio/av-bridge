r"""Retrieval vs alpha, Transitive bridge vs Caption-distance FGW.

Puts alpha directly on the x-axis and overlays the two FGW recipes so
their opposite alpha-dependence reads off a single figure:

  * Transitive Transport bridge (exp_c): no direct cross-modal feature
    term, so it RELIES on the structural smoothing -> retrieval rises to
    an interior-alpha peak (inverted-U).
  * Caption-distance FGW (exp_d): feature term M built directly from
    caption distances, so the correct pair is already sharp at alpha=0
    -> retrieval is flat then declines; the structural term is redundant.

Layout: 2 rows x 2 cols.
  rows: encoder regime (text-grounded, text-free)
  cols: retrieval metric (R@10, Cat-prec@10)
  two lines per panel: one per recipe, peak alpha starred.

Keeping each regime on its own row avoids overplotting and keeps the
noisy text-free held-out split from muddying the clean text-grounded
curves.

Inputs:
  transitive: results/exp_c{suffix}/sweep_transitive.csv  (filter K)
  caption:    results/exp_d{suffix}/sweep.csv
  both carry R@10, cat_precision_10 per (alpha, scope).

Usage:
  python code/plot_alpha_recipe_compare.py --K 300 --scope heldout
  python code/plot_alpha_recipe_compare.py --K 300 --scope aggregate
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

RECIPES = {
    "Transitive bridge": {
        "exp": "exp_c", "csv": "sweep_transitive.csv",
        "use_K": True, "color": "#2ca02c", "marker": "o",
    },
    "Caption-distance FGW": {
        "exp": "exp_d", "csv": "sweep.csv",
        "use_K": False, "color": "#9467bd", "marker": "s",
    },
}

METRIC_COLS = [
    ("R@10",             r"$R@10$"),
    ("cat_precision_10", r"Cat-prec@10"),
]


def _series(recipe: dict, suffix: str, K: int, scope: str,
            metric: str) -> tuple[np.ndarray, np.ndarray]:
    csv = RES / f"{recipe['exp']}{suffix}" / recipe["csv"]
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[df.scope == scope].copy()
    if recipe["use_K"] and "K" in df.columns:
        df = df[df.K == K]
    if df.empty or metric not in df.columns:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    return (df["alpha"].values.astype(float),
            df[metric].values.astype(float))


def render(out_path: Path, K: int, scope: str) -> None:
    n_rows, n_cols = len(PAIRS), len(METRIC_COLS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.6 * n_cols, 4.6 * n_rows),
                             squeeze=False)

    any_data = False
    for r, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        for c, (metric_col, metric_short) in enumerate(METRIC_COLS):
            ax = axes[r][c]
            for rec_name, rec in RECIPES.items():
                xs, ys = _series(rec, pair_spec["suffix"], K, scope,
                                 metric_col)
                if xs.size == 0:
                    continue
                any_data = True
                ax.plot(xs, ys, marker=rec["marker"], lw=2.0, ms=7,
                        color=rec["color"], label=rec_name,
                        markeredgecolor="black", markeredgewidth=0.5)
                i_peak = int(np.nanargmax(ys))
                ax.scatter([xs[i_peak]], [ys[i_peak]], s=230, marker="*",
                           facecolor=rec["color"], edgecolor="black",
                           linewidth=0.9, zorder=6)
                ax.annotate(rf"$\alpha^\star={xs[i_peak]:.1f}$",
                            xy=(xs[i_peak], ys[i_peak]),
                            xytext=(0, 9), textcoords="offset points",
                            ha="center", color=rec["color"],
                            fontsize=8.5, fontweight="bold")

            ax.set_xticks([0.0, 0.3, 0.5, 0.7, 0.9])
            ax.set_xlim(-0.05, 0.95)
            ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$ "
                          r"structural)", fontsize=10)
            ax.set_ylabel(metric_short, fontsize=10)
            ax.set_title(f"{pair_spec['label']}  —  {metric_short}",
                         fontsize=10.5)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[alpha-recipe] no data at K={K}, scope={scope}.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=rec["color"], lw=2.2, marker=rec["marker"],
               markersize=8, markeredgecolor="black", label=name)
        for name, rec in RECIPES.items()
    ] + [
        Line2D([], [], color="grey", lw=0, marker="*", markersize=14,
               markeredgecolor="black",
               label=r"peak $\alpha^\star$ for that metric/recipe")
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=3,
               fontsize=10, frameon=False)
    fig.suptitle(
        rf"Image$\to$audio retrieval vs $\alpha$: Transitive bridge "
        rf"relies on the structural term, Caption-distance FGW does not "
        rf"($K={K}$, {scope})",
        fontsize=12.5, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[alpha-recipe] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"alpha_recipe_compare__{args.scope}__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope)


if __name__ == "__main__":
    main()
