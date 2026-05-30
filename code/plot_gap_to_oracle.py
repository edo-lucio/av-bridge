r"""Recipe ladder: each method's image->audio retrieval against the
supervised Oracle ceiling and the no-transport / Random floors.

This is the proof-of-concept summary figure enabled by the two new
baselines (Oracle = supervised upper bound, NN bridge = no-transport
ablation). For each encoder regime it draws grouped bars of held-out
R@10 and Cat-prec@10 for every recipe, ordered floor -> ceiling, and
annotates each method's value as a FRACTION OF THE ORACLE -- i.e. how
much of the supervised ceiling each unsupervised/text-bridged method
recovers. The Oracle bar is the 100% reference.

Each recipe is read at its canonical operating point straight from its
own sweep CSV (robust to whatever rank_distributions.csv contains):

  Random        exp_random/sweep.csv            (single row)
  NN bridge     exp_nn/sweep.csv                (K=REUSABLE_K, alpha=0)
  Pure-GW       exp_unsup/sweep.csv             (single row)
  Caption FGW   exp_d/sweep.csv                 (alpha=REUSABLE_ALPHA)
  Transitive    exp_c/sweep_transitive.csv      (K=REUSABLE_K, alpha=REUSABLE_ALPHA)
  Text-only     exp_text/sweep.csv              (single row)
  Oracle        exp_oracle/sweep.csv            (K=REUSABLE_K, alpha=REUSABLE_ALPHA)

Usage:
  python code/plot_gap_to_oracle.py
  python code/plot_gap_to_oracle.py --scope aggregate --K 300
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

REUSABLE_K = 300
REUSABLE_ALPHA = 0.7

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

# (label, exp_dir, csv_name, selector dict | None, colour)
# selector keys are matched against CSV columns; None = single-row recipe.
RECIPES = [
    ("Random",          "exp_random", "sweep.csv",            None,
     "#777777"),
    ("NN bridge\n(no transport)", "exp_nn", "sweep.csv",
     {"K": REUSABLE_K, "alpha": 0.0}, "#8172b3"),
    ("Pure-GW",         "exp_unsup",  "sweep.csv",            None,
     "#55a868"),
    ("Caption\nFGW",    "exp_d",      "sweep.csv",
     {"alpha": REUSABLE_ALPHA}, "#dd8452"),
    ("Transitive\nbridge", "exp_c",   "sweep_transitive.csv",
     {"K": REUSABLE_K, "alpha": REUSABLE_ALPHA}, "#2a7fff"),
    ("Text-only",       "exp_text",   "sweep.csv",            None,
     "#c44e52"),
    ("Procrustes\n(rigid sup.)", "exp_procrustes", "sweep.csv",
     {"K": REUSABLE_K}, "#8c564b"),
    ("Oracle\n(supervised)", "exp_oracle", "sweep.csv",
     {"K": REUSABLE_K, "alpha": REUSABLE_ALPHA}, "#000000"),
]

METRICS = [("R@10", r"$R@10$"), ("cat_precision_10", "Cat-prec@10")]


def _value(suffix: str, exp: str, csv_name: str, sel: dict | None,
           scope: str, metric: str) -> float:
    csv = RES / f"{exp}{suffix}" / csv_name
    if not csv.exists():
        return float("nan")
    df = pd.read_csv(csv)
    sub = df[df.scope == scope]
    if sel:
        for k, v in sel.items():
            col = sub[k]
            sub = sub[np.isclose(col.astype(float), float(v))]
    if sub.empty or metric not in sub.columns:
        return float("nan")
    return float(sub.iloc[0][metric])


def render(out_path: Path, scope: str) -> None:
    n_rows, n_cols = len(PAIRS), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.8 * n_cols, 4.4 * n_rows),
                             squeeze=False)

    any_data = False
    for r, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        suffix = pair_spec["suffix"]
        for c, (metric_col, metric_label) in enumerate(METRICS):
            ax = axes[r][c]
            labels = [x[0] for x in RECIPES]
            colours = [x[4] for x in RECIPES]
            vals = [_value(suffix, exp, csvn, sel, scope, metric_col)
                    for (_, exp, csvn, sel, _) in RECIPES]
            vals = np.array(vals, dtype=float)
            if np.all(~np.isfinite(vals)):
                continue
            any_data = True

            oracle_v = vals[-1]
            xs = np.arange(len(RECIPES))
            ax.bar(xs, vals, color=colours, edgecolor="black",
                   linewidth=0.6, alpha=0.95)
            if np.isfinite(oracle_v) and oracle_v > 0:
                ax.axhline(oracle_v, color="black", lw=1.0, ls=":",
                           alpha=0.7, zorder=0)
            for x, v in zip(xs, vals):
                if not np.isfinite(v):
                    continue
                frac = (f"{100 * v / oracle_v:.0f}%"
                        if (np.isfinite(oracle_v) and oracle_v > 0) else "")
                ax.annotate(f"{v:.3f}\n{frac}", xy=(x, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_title(f"{pair_spec['label']}  —  {metric_label}",
                         fontsize=10.5)
            ax.set_ylim(0, max(0.01, np.nanmax(vals) * 1.18))
            ax.grid(True, axis="y", alpha=0.3)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[gap-to-oracle] no data at scope={scope}.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color="black", lw=1.0, ls=":",
               label="Oracle ceiling (supervised)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=1,
               fontsize=9, frameon=False)
    fig.suptitle(
        rf"Recipe ladder: image$\to$audio retrieval vs the supervised "
        rf"Oracle ceiling (scope = {scope}); % = fraction of Oracle",
        fontsize=12.5, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[gap-to-oracle] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    default_name = f"gap_to_oracle__{args.scope}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.scope)


if __name__ == "__main__":
    main()
