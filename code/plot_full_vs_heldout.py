r"""Full vs held-out comparison plot across all recipes.

For every recipe in ``RECIPE_LABELS`` (random baseline, transitive
transport bridge, caption distance FGW, Pure-GW, raw-caption text-only),
reads the aggregate-scope and held-out-scope rows from the encoder grid
sweep at the canonical encoder pair and plots them side by side as
grouped bars, one panel per metric.

The aggregate-vs-held-out gap is the in-sample-vs-generalisation
contrast. A small gap means the recipe's score reflects something
inherent to the recipe rather than memorisation of anchors used at
training time. A large gap means the score is anchor-supported in
aggregate but fragile when those anchors are held out.

Output:
  results/exp_grid/plots/full_vs_heldout.png

Optional pair argument (--pair canonical | textfree | both) lets you
restrict the plot to one encoder pair.
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

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GRID_CSV = RES / "exp_grid" / "sweep.csv"

RECIPE_LABELS = {
    "random":       "Random",
    "c-transitive": "Transitive bridge",
    "d":            "Caption-dist FGW",
    "unsup":        "Pure-GW",
    "text":         "Text-only",
    "procrustes":   "Procrustes",
    "direct":       "Direct ridge",
}

HELDOUT_SCOPE = {
    "random":       "heldout",
    "c-transitive": "heldout",
    "d":            "heldout",
    "unsup":        "heldout",
    "text":         "heldout",
    "procrustes":   "heldout",
    "direct":       "heldout",
}

# Seven metrics across the strict-to-coarse spectrum of alignment.
# Identity ($R@10$), class-level retrieval (category-recall@10), cluster
# routing (routes ratio), local-neighbourhood structural agreement
# (kNN overlap), partition agreement (AMI), pairwise-distance
# preservation (Pearson $r$), and external semantic agreement (cap-cos
# lift). Special-cased keys: ``__routes_ratio`` is computed inline as
# routes_correct / routes_total.
METRICS = [
    ("R@10",          "$R@10$",            "retrieval (identity)"),
    ("cat_precision_10", "Cat-prec@10",     "retrieval (class-level)"),
    ("__routes_ratio","Routes (correct/$K_{cl}$)", "routing (cluster-to-cluster)"),
    ("knn_overlap",   "kNN overlap",       "structural (local)"),
    ("ami",           "AMI",               "partition agreement"),
    ("pearson_r",     "Pearson $r$",       "structural (distances)"),
    ("cap_cos_lift",  "Cap-cos lift",      "semantic (external)"),
]

PAIRS = {
    "canonical": {"img": "clip-large",   "aud": "clap-unfused",
                  "label": "CLIP-L $\\times$ CLAP-unfused"},
    "textfree":  {"img": "dinov2-large", "aud": "mert-330m",
                  "label": "DINOv2-L $\\times$ MERT-330m"},
}


def _load_pair_rows(df: pd.DataFrame, img: str, aud: str) -> pd.DataFrame:
    """Return rows for one encoder pair only."""
    return df[(df.image_encoder == img) & (df.audio_encoder == aud)]


def _metric_value(df: pd.DataFrame, exp: str, scope: str,
                  metric: str) -> float:
    sub = df[(df.experiment == exp) & (df.scope == scope)]
    if sub.empty:
        return float("nan")
    r = sub.iloc[0]
    if metric == "__routes_ratio":
        total = float(r.get("routes_total", float("nan")))
        if not np.isfinite(total) or total == 0:
            return float("nan")
        return float(r.get("routes_correct", float("nan"))) / total
    if metric not in r.index:
        return float("nan")
    return float(r[metric])


def render(out_path: Path, pair_keys: list[str]) -> None:
    if not GRID_CSV.exists():
        print(f"[full-vs-heldout] skip: {GRID_CSV} not found.")
        return
    df_full = pd.read_csv(GRID_CSV)

    n_rows = len(pair_keys)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 3.4 * n_rows),
                             squeeze=False, sharey="col")

    recipe_keys = list(RECIPE_LABELS.keys())
    palette = dict(zip(recipe_keys,
                       sns.color_palette("colorblind", n_colors=len(recipe_keys))))
    x = np.arange(len(recipe_keys))
    width = 0.4

    for row_idx, pair_key in enumerate(pair_keys):
        p = PAIRS[pair_key]
        df_pair = _load_pair_rows(df_full, p["img"], p["aud"])

        for col_idx, (col, label, axis_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            agg_vals, hel_vals = [], []
            for exp in recipe_keys:
                agg_vals.append(_metric_value(df_pair, exp, "aggregate", col))
                hel_vals.append(_metric_value(df_pair, exp,
                                              HELDOUT_SCOPE[exp], col))

            ax.bar(x - width / 2,
                   [0 if np.isnan(v) else v for v in agg_vals],
                   width, label="aggregate",
                   color=[palette[k] for k in recipe_keys],
                   alpha=0.45, edgecolor="white")
            ax.bar(x + width / 2,
                   [0 if np.isnan(v) else v for v in hel_vals],
                   width, label="held-out",
                   color=[palette[k] for k in recipe_keys],
                   alpha=1.0, edgecolor="white")

            for xi, v in zip(x - width / 2, agg_vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.005, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=6.5, color="grey")
            for xi, v in zip(x + width / 2, hel_vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.005, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels([RECIPE_LABELS[k] for k in recipe_keys],
                               rotation=30, ha="right", fontsize=7.5)
            if col_idx == 0:
                ax.set_ylabel(p["label"], fontsize=9)
            if row_idx == 0:
                ax.set_title(f"{label}\n({axis_label})", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="grey",
                      alpha=a, edgecolor="white", label=lbl)
        for a, lbl in [(0.45, "aggregate (in-sample)"),
                       (1.0, "held-out (generalisation)")]
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=10, frameon=False)

    fig.suptitle(
        r"All recipes: aggregate vs held-out across six metrics",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[full-vs-heldout] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["canonical", "textfree", "both"],
                    default="both")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default: results/exp_grid/plots/...)")
    args = ap.parse_args()

    if args.pair == "both":
        keys = ["canonical", "textfree"]
        default_name = "full_vs_heldout.png"
    else:
        keys = [args.pair]
        default_name = f"full_vs_heldout__{args.pair}.png"
    out = Path(args.out) if args.out else (RES / "exp_grid" / "plots" / default_name)
    render(out, keys)


if __name__ == "__main__":
    main()
