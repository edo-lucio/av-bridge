r"""Scatter plots: encoder-pair CKA vs retrieval metrics per recipe.

For each (recipe, encoder pair) cell in the grid sweep, plot the pre-
alignment encoder CKA on the x-axis against a retrieval metric on the
y-axis. One point per encoder pair per recipe. Recipes are coloured;
per-recipe linear-regression trend lines and global / per-recipe
Pearson correlations are annotated.

The figure answers: *for a given recipe family, does increasing the
geometric similarity of the two encoder spaces (CKA) translate into
higher retrieval performance?* High positive slope -> "retrieval
piggybacks on encoder-space pre-alignment". Flat slope -> "the recipe
works the same regardless of encoder geometry". Negative slope ->
"the recipe wins on geometry-dissimilar pairs and loses on similar
pairs" (unusual, would suggest the recipe is correcting an asymmetry).

Inputs:
  results/exp_grid/sweep.csv                   (per (recipe, pair, K, alpha))
  results/exp_grid/geometric_similarity.csv    (CKA per encoder pair)

Output:
  results/exp_grid/plots/cka_vs_retrieval.png

Usage:
  python code/plot_cka_vs_retrieval.py
  python code/plot_cka_vs_retrieval.py --scope heldout
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
GRID_CSV = RES / "exp_grid" / "sweep.csv"
CKA_CSV = RES / "exp_grid" / "geometric_similarity.csv"

# Encoder-CKA-sensitive recipes only. Random is CKA-uncorrelated by
# construction (uniform plan); Raw caption cosine is encoder-independent
# (uses only the text-encoder space). Including them would clutter the
# scatter with two flat reference lines that don't carry signal.
RECIPE_LABELS = {
    "c-transitive": "Transitive Transport Bridge",
    "d":            "Caption Distance FGW",
    "unsup":        "GW (intra-modal geometry)",
}

METRICS = [
    ("R@1",              r"$R@1$  (sharp top retrieval)"),
    ("R@5",              r"$R@5$  (mid-depth retrieval)"),
    ("R@10",             r"$R@10$  (canonical operating point)"),
    ("cat_precision_10", r"Cat-prec@10  (top-$k$ category coverage)"),
]


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_merged(scope: str) -> pd.DataFrame:
    if not GRID_CSV.exists():
        raise FileNotFoundError(f"missing {GRID_CSV}")
    if not CKA_CSV.exists():
        raise FileNotFoundError(
            f"missing {CKA_CSV} (run code/analyse_geometry.py on HPC)")

    grid = pd.read_csv(GRID_CSV)
    cka = pd.read_csv(CKA_CSV)[
        ["image_encoder", "audio_encoder", "cka", "pearson_r_identity"]
    ]

    grid = grid[grid.scope == scope].copy()
    rows: list[dict] = []
    for exp, label in RECIPE_LABELS.items():
        sub = grid[grid.experiment == exp]
        for _, r in sub.iterrows():
            match = cka[(cka.image_encoder == r["image_encoder"])
                        & (cka.audio_encoder == r["audio_encoder"])]
            if match.empty:
                continue
            rows.append({
                "recipe":         label,
                "image_encoder":  r["image_encoder"],
                "audio_encoder":  r["audio_encoder"],
                "cka":            float(match.iloc[0]["cka"]),
                "id_pearson":     float(match.iloc[0]["pearson_r_identity"]),
                "R@1":            r.get("R@1", float("nan")),
                "R@5":            r.get("R@5", float("nan")),
                "R@10":           r.get("R@10", float("nan")),
                "cat_precision_10":
                                  r.get("cat_precision_10", float("nan")),
            })
    return pd.DataFrame(rows)


def render(out_path: Path, scope: str) -> None:
    df = _load_merged(scope)
    if df.empty:
        print(f"[cka-vs-retrieval] no rows at scope={scope}")
        return

    palette = dict(zip(RECIPE_LABELS.values(),
                       sns.color_palette("colorblind",
                                         n_colors=len(RECIPE_LABELS))))

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.2 * n_cols, 5.4 * n_rows),
                             squeeze=False)

    for idx, (col, title) in enumerate(METRICS):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = df.dropna(subset=["cka", col])
        if sub.empty:
            ax.set_axis_off()
            continue

        sns.scatterplot(data=sub, x="cka", y=col, hue="recipe",
                        ax=ax, palette=palette, s=70,
                        edgecolor="white", linewidth=0.6, alpha=0.9,
                        legend=False)

        for recipe, color in palette.items():
            rec_sub = sub[sub.recipe == recipe]
            if len(rec_sub) >= 3:
                sns.regplot(data=rec_sub, x="cka", y=col,
                            ax=ax, ci=None, scatter=False,
                            line_kws={"color": color, "lw": 1.3,
                                      "alpha": 0.55})

        global_r = _pearson_safe(sub["cka"].values, sub[col].values)
        annot_lines = [f"global $r$ = {global_r:+.2f}"]
        for recipe in palette.keys():
            rec_sub = sub[sub.recipe == recipe]
            r_val = _pearson_safe(rec_sub["cka"].values,
                                  rec_sub[col].values)
            short = (recipe
                     .replace("(baseline)", "")
                     .replace("(intra-modal geometry)", "")
                     .replace("(ceiling)", "")
                     .replace("Transport Bridge", "Bridge")
                     .strip())
            short = short[:22]
            annot_lines.append(
                f"  {short:22s} {r_val:+.2f}" if np.isfinite(r_val)
                else f"  {short:22s}  n/a"
            )
        ax.text(0.02, 0.98, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="lightgrey", alpha=0.9))

        ax.set_xlabel(r"linear CKA  (encoder-space similarity, "
                      r"pre-alignment)", fontsize=10)
        ax.set_ylabel(title.split("  ")[0], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    legend_handles = [
        Line2D([], [], color=color, marker="o", linestyle="None",
               markersize=9, label=label)
        for label, color in palette.items()
    ]
    fig.legend(handles=legend_handles, title="recipe",
               loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=min(len(legend_handles), 3),
               fontsize=10, title_fontsize=11, frameon=False)

    fig.suptitle(
        f"Encoder CKA vs retrieval metrics across recipes "
        f"(scope = {scope}; one point per encoder pair x recipe)",
        fontsize=13, y=1.002,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[cka-vs-retrieval] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Grid-sweep scope to read. Default heldout.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from "
                         "scope under results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"cka_vs_retrieval__{args.scope}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.scope)


if __name__ == "__main__":
    main()
