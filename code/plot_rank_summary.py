r"""Bar-plot summary of the rank distribution per recipe / encoder pair.

Reads ``results/exp_grid/rank_distributions.csv`` (produced by
analyse_ranks.py with both aggregate and heldout scopes), summarises
the per-query rank arrays into three complementary statistics, and
renders them as grouped bars with bootstrap 95\% CIs:

  1. R@10           --- Pr(rank <= 10), the standard retrieval cutoff.
  2. MRR            --- Mean reciprocal rank, 1 / rank averaged over
                        queries. Weights top ranks heavily.
  3. Median rank    --- Robust central tendency, log-scale y-axis.

Layout: 1 row x 3 cols, one panel per metric. Per panel: 5 recipes x
2 encoder pairs = 10 bars, grouped by recipe (canonical solid, text-
free hatched). 95\% bootstrap CIs are shown as black error bars.

The intent is to give the same per-recipe / per-pair comparison as
the rank-CDF plot but in a non-noisy summary form suitable for tables
or the chapter's headline figure.

Default scope is ``heldout``; pass --scope aggregate for the
in-sample view.

Usage:
  python code/plot_rank_summary.py
  python code/plot_rank_summary.py --scope aggregate
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
from matplotlib.patches import Patch  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"
RANK_CSV = RES / "exp_grid" / "rank_distributions.csv"

RECIPE_ORDER = [
    "Random (baseline)",
    "Transitive Transport Bridge",
    "Caption Distance FGW",
    "GW (intra-modal geometry)",
    "Raw caption cosine (ceiling)",
    "Procrustes (rigid supervised)",
    "Direct ridge (supervised)",
]

RECIPE_COLOURS = dict(zip(RECIPE_ORDER,
                          sns.color_palette("colorblind",
                                            n_colors=len(RECIPE_ORDER))))

PAIRS = [
    {"short": "canonical", "label": "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
     "hatch": ""},
    {"short": "text-free", "label": "text-free (DINOv2-L $\\times$ MERT-330m)",
     "hatch": "////"},
]


def _stat(ranks: np.ndarray, kind: str) -> float:
    if ranks.size == 0:
        return float("nan")
    if kind == "r10":
        return float((ranks <= 10).mean())
    if kind == "mrr":
        return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))
    if kind == "median":
        return float(np.median(ranks))
    if kind == "mean":
        return float(np.mean(np.asarray(ranks, dtype=float)))
    raise ValueError(kind)


def _bootstrap_ci(ranks: np.ndarray, kind: str,
                  n_boot: int = 1000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the statistic ``kind``."""
    if ranks.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = ranks.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = _stat(ranks[idx], kind)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


METRICS = [
    ("mean", "mean rank", "rank  (lower = better)", False),
]


def render(out_path: Path, scope: str, rank_csv: Path = RANK_CSV) -> None:
    if not rank_csv.exists():
        print(f"[rank-summary] missing input CSV: {rank_csv}")
        return
    df = pd.read_csv(rank_csv)
    if "scope" not in df.columns:
        print("[rank-summary] no 'scope' column --- regenerate "
              "rank_distributions.csv via analyse_ranks.py first.")
        return
    df = df[df.scope == scope]
    if df.empty:
        print(f"[rank-summary] no rows at scope={scope}")
        return

    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.2 * n_cols, 5.0),
                             squeeze=False)

    n_recipes = len(RECIPE_ORDER)
    bar_width = 0.36
    x_pos = np.arange(n_recipes)

    sample_sizes: dict[str, int] = {}

    for col_idx, (kind, label, ylabel, is_log) in enumerate(METRICS):
        ax = axes[0, col_idx]

        for pair_idx, pair in enumerate(PAIRS):
            offset = (pair_idx - 0.5) * bar_width
            heights, lo_err, hi_err = [], [], []
            for rec in RECIPE_ORDER:
                sub = df[(df.recipe == rec) & (df.pair == pair["short"])]
                if sub.empty:
                    heights.append(np.nan)
                    lo_err.append(np.nan)
                    hi_err.append(np.nan)
                    continue
                ranks = sub["rank"].values
                sample_sizes[pair["short"]] = int(len(ranks))
                v = _stat(ranks, kind)
                lo, hi = _bootstrap_ci(ranks, kind)
                heights.append(v)
                lo_err.append(max(0.0, v - lo))
                hi_err.append(max(0.0, hi - v))

            colours = [RECIPE_COLOURS[rec] for rec in RECIPE_ORDER]
            bars = ax.bar(
                x_pos + offset,
                [0 if np.isnan(h) else h for h in heights],
                width=bar_width,
                color=colours,
                edgecolor="black",
                linewidth=0.5,
                hatch=pair["hatch"],
                alpha=0.95,
            )
            ax.errorbar(
                x_pos + offset,
                [0 if np.isnan(h) else h for h in heights],
                yerr=[
                    [0 if np.isnan(e) else e for e in lo_err],
                    [0 if np.isnan(e) else e for e in hi_err],
                ],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5,
                alpha=0.85,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [r.replace(" (baseline)", "")
              .replace(" (ceiling)", "")
              .replace(" (intra-modal geometry)", "")
              .replace("Transport ", "")
             for r in RECIPE_ORDER],
            rotation=25, ha="right", fontsize=8.5,
        )
        ax.set_title(label, fontsize=11)
        if is_log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        sns.despine(ax=ax)

    legend_handles = [
        Patch(facecolor="lightgrey", edgecolor="black",
              hatch=PAIRS[0]["hatch"], label=PAIRS[0]["label"]),
        Patch(facecolor="lightgrey", edgecolor="black",
              hatch=PAIRS[1]["hatch"], label=PAIRS[1]["label"]),
        Line2D([], [], color="black", lw=0.9, label="95% bootstrap CI"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3,
               fontsize=10, frameon=False)

    size_note = ", ".join(
        f"{k} n={v}" for k, v in sample_sizes.items()
    ) or "n=?"
    fig.suptitle(
        f"Rank-distribution summary  (scope = {scope}; {size_note})",
        fontsize=12, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[rank-summary] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Which scope to summarise. Default heldout.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from scope "
                         "under results/exp_grid/plots/.")
    ap.add_argument("--rank-csv", type=str, default=None,
                    help="Rank-distribution CSV to read. Default "
                         "results/exp_grid/rank_distributions.csv; pass a "
                         "suffixed file (e.g. ...__a0.50.csv) to summarise "
                         "an alternative Transitive operating point.")
    args = ap.parse_args()

    rank_csv = Path(args.rank_csv) if args.rank_csv else RANK_CSV
    stem_suffix = ""
    if args.rank_csv:
        # Carry the input CSV's suffix into the default output name.
        stem = Path(args.rank_csv).stem
        stem_suffix = stem.replace("rank_distributions", "")
    default_name = f"rank_summary__{args.scope}{stem_suffix}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.scope, rank_csv)


if __name__ == "__main__":
    main()
