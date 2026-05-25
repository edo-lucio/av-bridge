r"""Plan spread vs image->audio retrieval: the benefit-side mirror of
the row-decay profile, for either FGW recipe.

The row-decay figure (plot_row_decay_profile.py) shows the COST of the
structural term: as alpha grows, each row of the image->audio plan
spreads its mass over a wider set of audio candidates, so the single
correct audio clip holds less absolute mass. This script shows the
matching BENEFIT (or its absence): how much that same widening changes
the chance the correct audio clip falls inside the top-k retrieved
cells.

It puts the MEDIATOR on the x-axis (the plan's own spread, measured as
effective support size e^{H} = exp(mean row entropy), i.e. the number
of audio candidates each query row effectively spreads mass over) and
the OUTCOME on the y-axis (R@10 and Cat-prec@10). Each alpha is one
marker; the peak alpha is starred.

Two recipes are supported:
  --recipe transitive   the Transitive Transport bridge (exp_c).
                        The plan is the composition rownorm(T_iv @ T_ac.T);
                        it has no direct cross-modal feature term, so it
                        RELIES on the structural smoothing -> inverted-U.
  --recipe caption      Caption-Cost FGW (exp_d). The plan is a direct
                        image->audio FGW solve whose feature term M is
                        built from caption distances. The correct pair is
                        already sharp at alpha=0, so the structural term
                        adds little -> roughly flat / late decline.

Both read the plan_row_entropy column (computed on the plan itself) that
run_experiments.py now writes alongside every retrieval row, so no plan
re-loading is needed.

Inputs:
  transitive: results/exp_c{suffix}/sweep_transitive.csv  (filter K)
  caption:    results/exp_d{suffix}/sweep.csv

Usage:
  python code/plot_composed_spread_benefit.py --recipe transitive --K 300
  python code/plot_composed_spread_benefit.py --recipe caption
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
        "color":  "#2a7fff",
        "suffix": "",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "color":  "#dd8452",
        "suffix": "__dinov2-large__mert-330m",
    },
}

RECIPES = {
    "transitive": {
        "exp":       "exp_c",
        "csv":       "sweep_transitive.csv",
        "use_K":     True,
        "plan":      "composed bridge",
        "title":     r"Widening the composed plan improves "
                     r"image$\to$audio retrieval up to a point",
        "out":       "composed_spread_benefit",
    },
    "caption": {
        "exp":       "exp_d",
        "csv":       "sweep.csv",
        "use_K":     False,
        "plan":      "Caption-Cost FGW",
        "title":     r"How much the structural term ($\alpha$) buys "
                     r"Caption-Cost FGW image$\to$audio retrieval",
        "out":       "caption_cost_spread_benefit",
    },
}

METRIC_COLS = [
    ("R@10",             r"$R@10$"),
    ("cat_precision_10", r"Cat-prec@10"),
]


def _load(recipe: dict, suffix: str, K: int, scope: str) -> pd.DataFrame | None:
    csv = RES / f"{recipe['exp']}{suffix}" / recipe["csv"]
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df.scope == scope].copy()
    if recipe["use_K"] and "K" in df.columns:
        df = df[df.K == K]
    if df.empty or "plan_row_entropy" not in df.columns:
        return None
    df = df.dropna(subset=["plan_row_entropy"])
    if df.empty:
        return None
    df = df.sort_values("alpha")
    df["support"] = np.exp(df["plan_row_entropy"].astype(float))
    return df


def render(out_path: Path, recipe: dict, K: int, scope: str) -> None:
    n_cols = len(METRIC_COLS)
    fig, axes = plt.subplots(1, n_cols, figsize=(7.2 * n_cols, 5.6),
                             squeeze=False)

    any_data = False
    legend_seen: set[str] = set()
    for col_idx, (metric_col, metric_short) in enumerate(METRIC_COLS):
        ax = axes[0, col_idx]
        metric_label = rf"{recipe['plan']} {metric_short}"

        for pair_key, pair_spec in PAIRS.items():
            df = _load(recipe, pair_spec["suffix"], K, scope)
            if df is None or metric_col not in df.columns:
                continue
            xs = df["support"].values.astype(float)
            ys = df[metric_col].values.astype(float)
            alphas = df["alpha"].values.astype(float)
            color = pair_spec["color"]
            any_data = True

            ax.plot(xs, ys, lw=1.8, color=color, alpha=0.75, zorder=3)
            ax.scatter(xs, ys, s=90, color=color, edgecolor="black",
                       linewidth=0.7, zorder=4,
                       label=pair_spec["label"]
                             if pair_key not in legend_seen else None)
            legend_seen.add(pair_key)

            for a, x, y in zip(alphas, xs, ys):
                ax.annotate(rf"$\alpha={a:.1f}$", xy=(x, y),
                            xytext=(7, 5), textcoords="offset points",
                            color=color, fontsize=8, fontweight="bold",
                            alpha=0.9)

            i_peak = int(np.nanargmax(ys))
            ax.scatter([xs[i_peak]], [ys[i_peak]], s=260, marker="*",
                       facecolor=color, edgecolor="black",
                       linewidth=0.9, zorder=6)

        ax.set_xscale("log")
        ax.set_xlabel(rf"{recipe['plan']} effective support  "
                      r"$e^{\bar H}$  (audio candidates per query row)",
                      fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax)

    if not any_data:
        print(f"[spread-benefit] no populated entropy for "
              f"{recipe['exp']} at scope={scope}.")
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
    k_note = rf"$K={K}$, " if recipe["use_K"] else ""
    fig.suptitle(
        rf"{recipe['title']} ({k_note}{scope}; "
        rf"$\alpha$ traces the curve left$\to$right)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[spread-benefit] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="transitive",
                    choices=list(RECIPES.keys()))
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    default_name = f"{recipe['out']}__K{args.K}.png" \
        if recipe["use_K"] else f"{recipe['out']}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, recipe, args.K, args.scope)


if __name__ == "__main__":
    main()
