r"""Spread vs composed retrieval: the inverted-U mediator plot.

Reframes the alpha-elbow trade-off by putting the MEDIATOR on the
x-axis (the plan spread, measured as the geometric mean of the two
legs' effective support sizes) and the OUTCOME on the y-axis
(composed Transitive bridge retrieval at K = 300, held-out scope).

The result is an inverted-U: as the legs widen with growing alpha,
composed retrieval climbs (smooth leg plans overlap better under the
matrix product), peaks at moderate spread, then falls (too much
smoothing dilutes the cross-modal signal beyond usefulness). Each
alpha appears as one marker along the curve; alpha is no longer the
axis.

Layout: 1 row x 2 cols, one panel per retrieval metric (R@10 and
Cat-prec@10). Both encoder regimes (text-grounded blue, text-free
orange) overlaid in each panel. Peak alpha annotated.

Inputs (all already populated):
  results/exp_a{suffix}/sweep.csv          (Leg A entropy)
  results/exp_b{suffix}/sweep.csv          (Leg B entropy)
  results/exp_c{suffix}/sweep_transitive.csv  (composed R@10, Cat-prec@10)

Usage:
  python code/plot_spread_vs_retrieval.py
  python code/plot_spread_vs_retrieval.py --K 200
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
        "color":  "#2a7fff",
        "c_suffix": "",
        "a_suffix": "",
        "b_suffix": "",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "color":  "#dd8452",
        "c_suffix": "__dinov2-large__mert-330m",
        "a_suffix": "__dinov2-large",
        "b_suffix": "__mert-330m",
    },
}

METRICS = [
    ("R@10",              r"composed bridge $R@10$"),
    ("cat_precision_10",  r"composed bridge Cat-prec@10"),
]


def _alpha_keyed(csv: Path, K: int, value_col: str) -> dict[float, float]:
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    df = df[df.scope == "heldout"]
    if "K" in df.columns:
        df = df[df.K == K]
    if value_col not in df.columns or df.empty:
        return {}
    return {float(r["alpha"]): float(r[value_col]) for _, r in df.iterrows()}


def _entropy_to_support(H: float) -> float:
    return float(np.exp(H))


def render(out_path: Path, K: int) -> None:
    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols, figsize=(7.2 * n_cols, 5.6),
                             squeeze=False)

    any_data = False
    legend_pair_seen: set[str] = set()
    for col_idx, (metric_col, metric_label) in enumerate(METRICS):
        ax = axes[0, col_idx]

        for pair_key, pair_spec in PAIRS.items():
            color = pair_spec["color"]

            H_A = _alpha_keyed(
                RES / f"exp_a{pair_spec['a_suffix']}" / "sweep.csv",
                K, "plan_row_entropy")
            H_B = _alpha_keyed(
                RES / f"exp_b{pair_spec['b_suffix']}" / "sweep.csv",
                K, "plan_row_entropy")
            Y = _alpha_keyed(
                RES / f"exp_c{pair_spec['c_suffix']}" / "sweep_transitive.csv",
                K, metric_col)

            common_alphas = sorted(set(H_A) & set(H_B) & set(Y))
            if not common_alphas:
                continue

            xs = np.array([
                _entropy_to_support(0.5 * (H_A[a] + H_B[a]))
                for a in common_alphas
            ])
            ys = np.array([Y[a] for a in common_alphas])
            any_data = True

            # Connecting line, in order of alpha along the curve.
            ax.plot(xs, ys, lw=1.8, color=color, alpha=0.75, zorder=3)
            ax.scatter(xs, ys, s=90, color=color, edgecolor="black",
                       linewidth=0.7, zorder=4,
                       label=pair_spec["label"]
                             if pair_key not in legend_pair_seen else None)
            legend_pair_seen.add(pair_key)

            # Annotate each marker with its alpha.
            for a, x, y in zip(common_alphas, xs, ys):
                ax.annotate(
                    rf"$\alpha={a:.1f}$",
                    xy=(x, y),
                    xytext=(7, 5), textcoords="offset points",
                    color=color, fontsize=8, fontweight="bold",
                    alpha=0.9,
                )

            # Highlight the peak with a star.
            i_peak = int(np.nanargmax(ys))
            ax.scatter([xs[i_peak]], [ys[i_peak]], s=260, marker="*",
                       facecolor=color, edgecolor="black",
                       linewidth=0.9, zorder=6)

        ax.set_xscale("log")
        ax.set_xlabel(r"geometric mean of leg effective support  "
                      r"$\sqrt{e^{\bar H_A} \cdot e^{\bar H_B}}$",
                      fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[spread-vs-retrieval] no data at K={K}; "
              "regenerate CSVs via sbatch first.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.2, marker="o",
               markersize=9, markeredgecolor="black",
               label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ] + [
        Line2D([], [], color="grey", lw=0, marker="*", markersize=14,
               markeredgecolor="black",
               label=r"peak $\alpha^\star$ for that metric/regime")
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3,
               fontsize=10, frameon=False)

    fig.suptitle(
        rf"Spread of the legs vs composed retrieval  "
        rf"($K = {K}$; $\alpha$ traces the curve from left to right)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[spread-vs-retrieval] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"spread_vs_retrieval__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K)


if __name__ == "__main__":
    main()
