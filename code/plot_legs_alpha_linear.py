r"""Per-leg retrieval degradation as alpha increases, with linear fit.

A focused diagnostic plot supporting the argument that the legs'
retrieval quality decreases approximately linearly with alpha,
while the composed bridge peaks at intermediate alpha (a behaviour
documented separately by plot_transitive_alpha_sweep.py). Together
the two figures motivate the structural-vs-targeting trade-off
discussion in the chapter.

Layout: 2 rows (R@10, Cat-prec@10) x 2 cols (Leg A, Leg B).
Each panel overlays both encoder regimes (canonical blue,
text-free orange) as scatter points with a dashed linear fit.
Slope (m) and goodness of fit (R^2) of each fit are annotated
near the line's right end.

X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9}.
Y-axis: metric value at per-leg heldout scope, K = 300.

Usage:
  python code/plot_legs_alpha_linear.py
  python code/plot_legs_alpha_linear.py --K 200 --scope aggregate
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
from scipy import stats  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.9)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

PAIRS = {
    "canonical": {
        "label":    "text-grounded (CLIP-L $/$ CLAP-unfused)",
        "color":    "#2a7fff",
        "a_suffix": "",
        "b_suffix": "",
    },
    "textfree": {
        "label":    "text-free (DINOv2-L $/$ MERT-330m)",
        "color":    "#dd8452",
        "a_suffix": "__dinov2-large",
        "b_suffix": "__mert-330m",
    },
}

LEGS = [
    {"key": "a",
     "label": "Leg A  ---  image $\\to$ visual-text",
     "suffix_key": "a_suffix"},
    {"key": "b",
     "label": "Leg B  ---  audio $\\to$ audio-text",
     "suffix_key": "b_suffix"},
]

METRICS = [
    ("R@10",           "$R@10$"),
    ("cat_precision_10",  "Cat-prec@10"),
]


def _fetch(leg_key: str, suffix: str, scope: str,
           K: int) -> pd.DataFrame:
    csv = RES / f"exp_{leg_key}{suffix}" / "sweep.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & (df.K == K)]
    return df.sort_values("alpha")


def render(out_path: Path, K: int, scope: str) -> None:
    n_rows = len(METRICS)
    n_cols = len(LEGS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.4 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    any_data = False
    for row_idx, (col, label) in enumerate(METRICS):
        for col_idx, leg in enumerate(LEGS):
            ax = axes[row_idx, col_idx]

            for pair_key, pair_spec in PAIRS.items():
                suffix = pair_spec[leg["suffix_key"]]
                df = _fetch(leg["key"], suffix, scope, K)
                if df.empty:
                    continue
                if col not in df.columns:
                    continue
                any_data = True

                xs = df["alpha"].values.astype(float)
                ys = df[col].values.astype(float)

                mask = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[mask], ys[mask]
                if xs.size < 2:
                    continue

                color = pair_spec["color"]

                # Scatter of measured cells.
                ax.scatter(xs, ys, s=70, color=color,
                           edgecolor="black", linewidth=0.5,
                           zorder=4)

                # Linear fit.
                slope, intercept, r_value, _, _ = stats.linregress(xs, ys)
                x_fit = np.linspace(0.0, 1.0, 60)
                y_fit = intercept + slope * x_fit
                ax.plot(x_fit, y_fit, ls="--", lw=1.6,
                        color=color, alpha=0.85, zorder=3)

                # Annotate slope and R^2 near right end of fitted line.
                y_anno = intercept + slope * 0.95
                ax.annotate(
                    f"$m = {slope:+.2f}$\n$R^2 = {r_value**2:.2f}$",
                    xy=(0.95, y_anno),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=8, ha="left", va="center",
                    fontweight="bold",
                )

            ax.set_xticks(ALPHA_GRID)
            ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID],
                               fontsize=8)
            ax.set_xlim(-0.05, 1.25)
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\alpha$  "
                              r"(cross-modal $\leftrightarrow$ structural)",
                              fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 0:
                ax.set_title(leg["label"], fontsize=10)
            ax.grid(True, which="both", alpha=0.25)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[legs-alpha-linear] no data found at K={K}, scope={scope}.")
        plt.close(fig)
        return

    legend_handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.4,
               marker="o", markersize=7, markeredgecolor="black",
               markeredgewidth=0.5, label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ]
    fig.legend(handles=legend_handles, title="encoder regime (colour)",
               loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=2, fontsize=10, title_fontsize=11, frameon=False)

    fig.suptitle(
        rf"Per-leg retrieval degrades approximately linearly with "
        rf"$\alpha$  ($K = {K}$, scope = {scope})",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[legs-alpha-linear] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="K value at which to read each leg's alpha "
                         "sweep. Default 300 (canonical operating point).")
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Per-leg scope. exp_a / exp_b write 'heldout' "
                         "(not 'heldout'). Default held-out.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from K "
                         "under results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"legs_alpha_linear__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope)


if __name__ == "__main__":
    main()
