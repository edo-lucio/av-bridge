r"""Alpha sweep with both encoder regimes overlaid in the same panel.

For each metric (R@10, Cat-prec@10, Routes/K_cl, Pearson r):
  X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9}.
  Y-axis: metric value at held-out scope.
  Per regime (canonical blue / text-free orange):
    - Transitive Transport Bridge at K=300: solid line, markers.
    - Caption-Distance FGW: dashed line, markers.
    - Random baseline: dotted horizontal.
    - Text-only (caption-cosine ceiling): dash-dot-dot horizontal.

The point of the figure is to compare how each encoder regime's alpha
response curve sits between its own Random floor and Text-only ceiling.

Usage:
  python code/plot_alpha_sweep.py
  python code/plot_alpha_sweep.py --K 300 --scope heldout
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

# Encoder pair colours.  Same convention as plot_transitive_K_sweep's
# merged variant: blue = canonical, orange = text-free.
PAIRS = {
    "canonical": {"suffix": "",
                  "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
                  "color":  "#2a7fff"},
    "textfree":  {"suffix": "__dinov2-large__mert-330m",
                  "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
                  "color":  "#dd8452"},
}

METRICS = [
    ("R@10",            "$R@10$"),
    ("cat_precision_10",   "Cat-prec@10 (class)"),
    ("__routes_ratio",  "Routes (correct / $K_{cl}$)"),
    ("pearson_r",       "Pearson $r$"),
]

# Line styles per recipe: shared across pairs so colour identifies pair
# and line style identifies recipe.
LS_TRANSITIVE = "-"
LS_D          = "--"
LS_RANDOM     = ":"
LS_TEXT       = (0, (3, 1, 1, 1))   # dash-dot-dot


def _metric_from_row(row: pd.Series, col: str) -> float:
    if col == "__routes_ratio":
        total = float(row.get("routes_total", float("nan")))
        if not np.isfinite(total) or total == 0:
            return float("nan")
        return float(row.get("routes_correct", float("nan"))) / total
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _transitive_alpha_series(suffix: str, K: int, scope: str,
                             metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alphas, values) at the given K from sweep_transitive.csv."""
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & (df.K == K)].copy()
    if df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    ys = np.array([_metric_from_row(r, metric) for _, r in df.iterrows()])
    xs = df["alpha"].values.astype(float)
    return xs, ys


def _d_alpha_series(suffix: str, scope: str,
                    metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alphas, values) from exp_d/sweep.csv at the chosen scope."""
    csv = RES / f"exp_d{suffix}" / "sweep.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[df.scope == scope].copy()
    if df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    ys = np.array([_metric_from_row(r, metric) for _, r in df.iterrows()])
    xs = df["alpha"].values.astype(float)
    return xs, ys


def _single_recipe_value(suffix: str, exp_name: str, scope: str,
                         metric: str) -> float:
    """Single scalar value for Random / Text-only / etc. (no alpha sweep)."""
    csv = RES / f"exp_{exp_name}{suffix}" / "sweep.csv"
    if not csv.exists():
        return float("nan")
    df = pd.read_csv(csv)
    sub = df[df.scope == scope]
    if sub.empty:
        return float("nan")
    return _metric_from_row(sub.iloc[0], metric)


def render(out_path: Path, K: int, scope: str) -> None:
    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(3.6 * n_cols, 4.2),
                             squeeze=False)

    any_data = False
    for col_idx, (col, label) in enumerate(METRICS):
        ax = axes[0, col_idx]

        for pair_key, pair_spec in PAIRS.items():
            suffix = pair_spec["suffix"]
            color = pair_spec["color"]

            # Transitive Transport Bridge alpha sweep at fixed K.
            xs, ys = _transitive_alpha_series(suffix, K, scope, col)
            if xs.size:
                any_data = True
                ax.plot(xs, ys, marker="o", lw=1.8, ls=LS_TRANSITIVE,
                        color=color, zorder=3)

            # Caption-Distance FGW alpha sweep (no K).
            xs_d, ys_d = _d_alpha_series(suffix, scope, col)
            if xs_d.size:
                any_data = True
                ax.plot(xs_d, ys_d, marker="s", markersize=5, lw=1.4,
                        ls=LS_D, color=color, alpha=0.85, zorder=3)

            # Random floor.
            v = _single_recipe_value(suffix, "random", scope, col)
            if np.isfinite(v):
                ax.axhline(v, color=color, lw=1.2, ls=LS_RANDOM,
                           alpha=0.65, zorder=2)
            # Text-only ceiling.
            v = _single_recipe_value(suffix, "text", scope, col)
            if np.isfinite(v):
                ax.axhline(v, color=color, lw=1.2, ls=LS_TEXT,
                           alpha=0.75, zorder=2)

        ax.set_xticks(ALPHA_GRID)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=8)
        ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$ structural)",
                      fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[alpha-sweep] no data found at K={K}, scope={scope}.")
        plt.close(fig)
        return

    # Two-block legend: pair colour on the left, recipe line style on the right.
    pair_handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.2, label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ]
    recipe_handles = [
        Line2D([], [], color="black", lw=1.8, ls=LS_TRANSITIVE, marker="o",
               label=f"Transitive bridge ($K = {K}$)"),
        Line2D([], [], color="black", lw=1.4, ls=LS_D, marker="s",
               markersize=5, label="Caption-Distance FGW"),
        Line2D([], [], color="black", lw=1.2, ls=LS_RANDOM,
               label="Random (floor)"),
        Line2D([], [], color="black", lw=1.2, ls=LS_TEXT,
               label="Text-only (ceiling)"),
    ]
    fig.legend(handles=pair_handles, title="encoder pair (colour)",
               loc="lower left", bbox_to_anchor=(0.02, -0.06),
               ncol=1, fontsize=9, title_fontsize=10, frameon=False)
    fig.legend(handles=recipe_handles, title="recipe (line style)",
               loc="lower right", bbox_to_anchor=(0.98, -0.12),
               ncol=2, fontsize=9, title_fontsize=10, frameon=False)

    fig.suptitle(
        rf"$\alpha$ sweep, both encoder regimes overlaid  "
        rf"(Transitive at $K={K}$; held-out scope = {scope.replace('_', ' ')})",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[alpha-sweep] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="K value for the Transitive bridge alpha sweep. "
                         "Default 300 (canonical operating point).")
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Evaluation scope. Default held-out.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from K under "
                         "results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"alpha_sweep__regimes__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope)


if __name__ == "__main__":
    main()
