r"""Coarse-to-fine alignment vs alpha, all recipes overlaid.

Four panels arranged left-to-right in order of granularity:
  1. Routes (correct / K_cl)  --- coarsest, cluster routing.
  2. Cat-prec@10            --- category-level retrieval.
  3. R@10                     --- instance-level retrieval.
  4. Pearson r                --- continuous structural fidelity.

X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9} (plus the alpha = 1 point
for Pure-GW). Y-axis: metric value at the recipe's held-out scope.

Each panel overlays both encoder regimes (colour) and all recipes
that depend on alpha plus the non-alpha references (line style /
marker shape):

  - Transitive Transport Bridge (K=K):   solid,           'o'
  - Caption-Distance FGW:                dashed,          's'
  - Leg A FGW (K=K):                     dotted,          '^'
  - Leg B FGW (K=K):                     dash-dot-dot,    'v'
  - Pure-GW (alpha = 1, single point):                    '*'
  - Random:                              horizontal dotted reference.
  - Text-only:                           horizontal dash-dot reference.

Colour: text-grounded (CLIP-L x CLAP-unfused) blue, text-free (DINOv2-L
x MERT-330m) orange.

Scope note. Transitive, Caption-D, Pure-GW, Random, Text-only are
read at the shared ``heldout`` scope (so the composed-recipe
curves and the references are directly comparable cell-for-cell).
Leg A and Leg B write only the ``heldout`` scope --- a different
held-out partition with an order of magnitude more rows. The leg
curves are therefore plotted on the same axes for shape (the alpha
response curve), not for absolute lift against the composed recipes.

Usage:
  python code/plot_coarse_to_fine_alpha.py
  python code/plot_coarse_to_fine_alpha.py --K 200
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

# Encoder regimes. ``suffix`` is the cross-modal pair suffix (used by
# every recipe that consumes both modalities at once); ``leg_a`` and
# ``leg_b`` are the per-side suffixes used by exp_a / exp_b.
PAIRS = {
    "canonical": {
        "suffix":   "",
        "leg_a":    "",
        "leg_b":    "",
        "label":    "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
        "color":    "#2a7fff",
    },
    "textfree": {
        "suffix":   "__dinov2-large__mert-330m",
        "leg_a":    "__dinov2-large",
        "leg_b":    "__mert-330m",
        "label":    "text-free (DINOv2-L $\\times$ MERT-330m)",
        "color":    "#dd8452",
    },
}

# Coarse -> fine ordering.
METRICS = [
    ("__routes_ratio", "Routes (correct / $K_{cl}$)\n[cluster level]"),
    ("cat_precision_10",  "Cat-prec@10\n[category level]"),
    ("R@10",           "$R@10$\n[instance level]"),
    ("pearson_r",      "Pearson $r$\n[structural / continuous]"),
]

# Line styles per recipe: shared across regimes so colour identifies
# the regime and the (linestyle, marker) pair identifies the recipe.
LS_TRANSITIVE = "-"
LS_D          = "--"
LS_LEG_A      = ":"
LS_LEG_B      = (0, (3, 1, 1, 1))    # dash-dot-dot
LS_RANDOM     = (0, (1, 1))          # tight dotted
LS_TEXT       = (0, (5, 1, 1, 1))    # long dash-dot

MK_TRANSITIVE = "o"
MK_D          = "s"
MK_LEG_A      = "^"
MK_LEG_B      = "v"
MK_PUREGW     = "*"


def _metric_from_row(row: pd.Series, col: str) -> float:
    if col == "__routes_ratio":
        total = float(row.get("routes_total", float("nan")))
        if not np.isfinite(total) or total == 0:
            return float("nan")
        return float(row.get("routes_correct", float("nan"))) / total
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _alpha_series(csv: Path, scope: str, metric: str,
                  K: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (alphas, values) for the chosen scope (and K if applicable)."""
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    if K is not None and "K" in df.columns:
        df = df[df.K == K]
    if df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    ys = np.array([_metric_from_row(r, metric) for _, r in df.iterrows()])
    xs = df["alpha"].values.astype(float)
    return xs, ys


def _single_value(csv: Path, scope: str, metric: str) -> float:
    if not csv.exists():
        return float("nan")
    df = pd.read_csv(csv)
    sub = df[df.scope == scope]
    if sub.empty:
        return float("nan")
    return _metric_from_row(sub.iloc[0], metric)


def render(out_path: Path, K: int,
           scope_composed: str, scope_leg: str) -> None:
    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.9 * n_cols, 4.4),
                             squeeze=False)

    any_data = False
    for col_idx, (col, label) in enumerate(METRICS):
        ax = axes[0, col_idx]

        for pair_key, pair_spec in PAIRS.items():
            color = pair_spec["color"]
            suffix = pair_spec["suffix"]

            # Transitive Transport Bridge (alpha sweep at K).
            xs, ys = _alpha_series(
                RES / f"exp_c{suffix}" / "sweep_transitive.csv",
                scope_composed, col, K)
            if xs.size:
                any_data = True
                ax.plot(xs, ys, marker=MK_TRANSITIVE, markersize=8,
                        lw=1.8, ls=LS_TRANSITIVE,
                        color=color, zorder=5)

            # Caption-Distance FGW (alpha sweep, K-independent).
            xs, ys = _alpha_series(
                RES / f"exp_d{suffix}" / "sweep.csv",
                scope_composed, col, None)
            if xs.size:
                any_data = True
                ax.plot(xs, ys, marker=MK_D, markersize=7,
                        lw=1.4, ls=LS_D,
                        color=color, alpha=0.9, zorder=4)

            # Leg A FGW (alpha sweep at K). Note: per-leg scope is
            # ``heldout``, not ``heldout``.
            xs, ys = _alpha_series(
                RES / f"exp_a{pair_spec['leg_a']}" / "sweep.csv",
                scope_leg, col, K)
            if xs.size:
                any_data = True
                ax.plot(xs, ys, marker=MK_LEG_A, markersize=7,
                        lw=1.2, ls=LS_LEG_A,
                        color=color, alpha=0.85, zorder=3)

            # Leg B FGW (alpha sweep at K), same scope caveat.
            xs, ys = _alpha_series(
                RES / f"exp_b{pair_spec['leg_b']}" / "sweep.csv",
                scope_leg, col, K)
            if xs.size:
                any_data = True
                ax.plot(xs, ys, marker=MK_LEG_B, markersize=7,
                        lw=1.2, ls=LS_LEG_B,
                        color=color, alpha=0.85, zorder=3)

            # Pure-GW: single point at alpha = 1.
            v = _single_value(RES / f"exp_unsup{suffix}" / "sweep.csv",
                              scope_composed, col)
            if np.isfinite(v):
                any_data = True
                ax.plot(1.0, v, marker=MK_PUREGW, markersize=14,
                        color=color, markeredgecolor="black",
                        markeredgewidth=0.6, linestyle="None", zorder=6)

            # Random floor: horizontal reference.
            v = _single_value(RES / f"exp_random{suffix}" / "sweep.csv",
                              scope_composed, col)
            if np.isfinite(v):
                ax.axhline(v, color=color, lw=1.0, ls=LS_RANDOM,
                           alpha=0.55, zorder=1)

            # Text-only ceiling: horizontal reference.
            v = _single_value(RES / f"exp_text{suffix}" / "sweep.csv",
                              scope_composed, col)
            if np.isfinite(v):
                ax.axhline(v, color=color, lw=1.1, ls=LS_TEXT,
                           alpha=0.7, zorder=1)

        ax.set_xticks(ALPHA_GRID + [1.0])
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID] + ["1.0"],
                           fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(r"$\alpha$  (cross-modal cost $M$ "
                      r"$\leftrightarrow$  structural GW)",
                      fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[coarse-to-fine-alpha] no data found at K={K}.")
        plt.close(fig)
        return

    # Two legends: regime colour on the left, recipe shape on the right.
    pair_handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.6,
               label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ]
    recipe_handles = [
        Line2D([], [], color="black", lw=1.8, ls=LS_TRANSITIVE,
               marker=MK_TRANSITIVE,
               label=f"Transitive Transport Bridge ($K = {K}$)"),
        Line2D([], [], color="black", lw=1.4, ls=LS_D, marker=MK_D,
               label="Caption-Distance FGW"),
        Line2D([], [], color="black", lw=1.2, ls=LS_LEG_A,
               marker=MK_LEG_A,
               label=f"Leg A: image $\\to$ visual-text ($K = {K}$)"),
        Line2D([], [], color="black", lw=1.2, ls=LS_LEG_B,
               marker=MK_LEG_B,
               label=f"Leg B: audio $\\to$ audio-text ($K = {K}$)"),
        Line2D([], [], color="black", linestyle="None",
               marker=MK_PUREGW, markersize=13,
               markeredgecolor="black",
               label=r"Pure-GW ($\alpha = 1$, single point)"),
        Line2D([], [], color="black", lw=1.0, ls=LS_RANDOM,
               label="Random (floor)"),
        Line2D([], [], color="black", lw=1.1, ls=LS_TEXT,
               label="Text-only (ceiling)"),
    ]
    fig.legend(handles=pair_handles, title="encoder regime (colour)",
               loc="lower left", bbox_to_anchor=(0.02, -0.10),
               ncol=1, fontsize=9, title_fontsize=10, frameon=False)
    fig.legend(handles=recipe_handles,
               title="recipe (line style $+$ marker)",
               loc="lower right", bbox_to_anchor=(0.98, -0.18),
               ncol=4, fontsize=8.5, title_fontsize=10, frameon=False)

    fig.suptitle(
        rf"Coarse-to-fine alignment vs $\alpha$  ---  "
        rf"Transitive / legs at $K = {K}$,  "
        rf"composed scope = {scope_composed.replace('_', ' ')},  "
        rf"leg scope = {scope_leg}",
        fontsize=12, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[coarse-to-fine-alpha] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="K for the Transitive bridge and per-leg sweeps. "
                         "Default 300 (canonical operating point).")
    ap.add_argument("--scope-composed", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for Transitive / Caption-D / Pure-GW / "
                         "Random / Text-only. Default held-out (shared "
                         "across composed recipes).")
    ap.add_argument("--scope-leg", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for Leg A / Leg B (exp_a / exp_b write "
                         "'heldout', not 'heldout'). Default held-out.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from K under "
                         "results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"coarse_to_fine_alpha__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope_composed, args.scope_leg)


if __name__ == "__main__":
    main()
