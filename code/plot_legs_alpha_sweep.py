r"""Per-leg alpha-sweep comparison across the two encoder regimes.

Two rows --- one per leg of the Transitive Transport Bridge:
  Row 1: Leg A, image -> visual-text plan T_iv from Experiment A.
  Row 2: Leg B, audio -> audio-text plan T_ac from Experiment B.

Six columns --- the retrieval and structural metrics that determine
how good each leg's plan is before composition:
  R@1, R@5, R@10, R@20, Cat-prec@10, Pearson r.

X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9} at fixed K = 300.
Y-axis: metric value at held-out scope (anchor rows excluded).

Each panel overlays both encoder regimes:
  text-grounded (CLIP-L image / CLAP-unfused audio): blue.
  text-free (DINOv2-L image / MERT-330m audio):  orange.

The chapter use of this figure is "the legs are the bottleneck for
the text-free transitive bridge". For text-aligned encoders the legs
are near-trivial because CLIP and CLAP are pre-aligned to their own
text encoders. For text-free encoders the legs have to learn the
cross-modal mapping from K anchor pairs alone, and the gap is what
caps how good the composed bridge can be.

Usage:
  python code/plot_legs_alpha_sweep.py
  python code/plot_legs_alpha_sweep.py --K 200 --scope aggregate
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
              palette="colorblind", font_scale=0.9)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

# Encoder pair colours: blue = canonical, orange = text-free.
# For the legs, "canonical" and "text-free" decompose per side, so each
# leg's pair spec carries its own per-side suffix.
PAIRS = {
    "canonical": {
        "label":      "text-grounded (CLIP-L $/$ CLAP-unfused)",
        "color":      "#2a7fff",
        "a_suffix":   "",                 # Leg A canonical = default image encoder
        "b_suffix":   "",                 # Leg B canonical = default audio encoder
    },
    "textfree": {
        "label":      "text-free (DINOv2-L $/$ MERT-330m)",
        "color":      "#dd8452",
        "a_suffix":   "__dinov2-large",   # Leg A text-free = DINOv2-large
        "b_suffix":   "__mert-330m",      # Leg B text-free = MERT-330m
    },
}

# The two legs of the transitive bridge.
LEGS = [
    {"key": "a",
     "label": "Leg A  ---  image $\\to$ visual-text  (Experiment A)",
     "suffix_key": "a_suffix"},
    {"key": "b",
     "label": "Leg B  ---  audio $\\to$ audio-text  (Experiment B)",
     "suffix_key": "b_suffix"},
]

METRICS = [
    ("R@1",            "$R@1$"),
    ("R@5",            "$R@5$"),
    ("R@10",           "$R@10$"),
    ("R@20",           "$R@20$"),
    ("cat_precision_10",  "Cat-prec@10"),
    ("pearson_r",      "Pearson $r$"),
]


def _metric_from_row(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _alpha_series(leg_key: str, suffix: str, K: int, scope: str,
                  metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alphas, values) at the given K from exp_<leg>{suffix}/sweep.csv."""
    csv = RES / f"exp_{leg_key}{suffix}" / "sweep.csv"
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


def render(out_path: Path, K: int, scope: str) -> None:
    n_rows = len(LEGS)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.3 * n_rows),
                             squeeze=False)

    any_data = False
    for row_idx, leg in enumerate(LEGS):
        for col_idx, (col, label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            for pair_key, pair_spec in PAIRS.items():
                suffix = pair_spec[leg["suffix_key"]]
                xs, ys = _alpha_series(leg["key"], suffix, K, scope, col)
                if not xs.size:
                    continue
                any_data = True
                ax.plot(xs, ys, marker="o", lw=1.8,
                        color=pair_spec["color"], zorder=3)

            ax.set_xticks(ALPHA_GRID)
            ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\alpha$", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(leg["label"] + f"\n\n{label}", fontsize=9)
            else:
                ax.set_ylabel(label, fontsize=8)
            if row_idx == 0:
                ax.set_title(label, fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[legs-alpha-sweep] no data found at K={K}, scope={scope}.")
        plt.close(fig)
        return

    # Single regime legend.
    pair_handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.2,
               marker="o", markersize=6, label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ]
    fig.legend(handles=pair_handles, title="encoder regime (colour)",
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=10, title_fontsize=11, frameon=False)

    fig.suptitle(
        rf"Per-leg quality vs $\alpha$ at $K = {K}$  "
        rf"(scope = {scope.replace('_', ' ')})",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[legs-alpha-sweep] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="K value for each leg's alpha sweep. "
                         "Default 300 (canonical operating point).")
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Per-leg scope. exp_a / exp_b write 'heldout' "
                         "(not 'heldout'). Default held-out.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from K under "
                         "results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"legs_alpha_sweep__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope)


if __name__ == "__main__":
    main()
