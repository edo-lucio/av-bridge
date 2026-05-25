r"""Scatter of the core-set metric across recipes and (K, alpha) cells.

The core-set metric is two numbers per row of a transport plan:
  core_hit       = fraction of source rows where GT lands in the row's
                   "core set" (columns whose mass exceeds the row's
                   mean + std).
  core_mean_size = mean cardinality of that core set across rows.

Neither number is interpretable alone --- a diffuse plan inflates
core_mean_size trivially, a sharp wrong plan has tiny |core| and
low hit_rate. The pair together IS the metric.

This script reads ``results/core_set_analysis/extended_table.csv``
(produced by ``code/extended_metrics_table.py``) and renders one
scatter panel per scope. Each marker is one (recipe, K, alpha)
cell; colour codes the recipe; transitive's many cells are drawn
as small markers connected by thin lines per alpha so the K-sweep
direction is visible.

Output:
  results/exp_grid/plots/core_set_scatter.png
"""
from __future__ import annotations

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
TABLE_CSV = RES / "core_set_analysis" / "extended_table.csv"
OUT = RES / "exp_grid" / "plots" / "core_set_scatter.png"


# Friendly short labels and palette anchor for each method.
METHODS = {
    "Random baseline":                                "Random",
    "GW (unsup, alpha=1.0)":                          "Pure-GW",
    "FGW direct (M = caption cos)":                   "Caption-Dist FGW",
    "FGW transitive (identity bridge)":               "Transitive bridge",
    "Text baseline (cosine sim, NOT row-stochastic)": "Text-only",
}

PALETTE = {
    "Random":            "#777777",
    "Pure-GW":           "#55a868",
    "Caption-Dist FGW":  "#dd8452",
    "Transitive bridge": "#2a7fff",
    "Text-only":         "#c44e52",
}


def _panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Render one scope's worth of points on a given axes."""
    # Filter to the core_set rows only (text baseline reports R@10/10 separately).
    sub = df[df.metric_kind == "core_set"].copy()
    if sub.empty:
        ax.set_title(f"{title}\n(no core-set rows)")
        ax.set_axis_off()
        return

    # Transitive: many (K, alpha) cells. Draw a thin line per alpha so K
    # progression is visible, and small markers.
    tr = sub[sub.method == "FGW transitive (identity bridge)"]
    if not tr.empty:
        for alpha, grp in tr.groupby("alpha"):
            grp = grp.sort_values("K")
            ax.plot(grp.core_mean_size, grp.core_hit,
                    color=PALETTE["Transitive bridge"],
                    lw=1.0, alpha=0.35, zorder=2)
            ax.scatter(grp.core_mean_size, grp.core_hit,
                       color=PALETTE["Transitive bridge"],
                       s=22, alpha=0.55, edgecolor="white", linewidth=0.4,
                       zorder=3)
        # One annotated marker at the canonical (K=300, alpha=0.7) cell.
        canon = tr[(tr.K == 300) & (np.isclose(tr.alpha, 0.7))]
        if not canon.empty:
            x = float(canon.iloc[0].core_mean_size)
            y = float(canon.iloc[0].core_hit)
            ax.scatter([x], [y], s=120,
                       facecolor=PALETTE["Transitive bridge"],
                       edgecolor="black", linewidth=1.0, zorder=5,
                       label="Transitive bridge")
            ax.annotate(r"K=300, $\alpha$=0.7", (x, y),
                        xytext=(6, -6), textcoords="offset points",
                        fontsize=7.5, alpha=0.8)

    # Caption-Distance FGW: 5 alpha cells. Connect with a line.
    d = sub[sub.method == "FGW direct (M = caption cos)"].sort_values("alpha")
    if not d.empty:
        ax.plot(d.core_mean_size, d.core_hit,
                color=PALETTE["Caption-Dist FGW"], lw=1.2, alpha=0.5,
                zorder=2)
        ax.scatter(d.core_mean_size, d.core_hit, s=70,
                   color=PALETTE["Caption-Dist FGW"],
                   edgecolor="white", linewidth=0.6, alpha=0.95, zorder=4,
                   label="Caption-Dist FGW")
        # Annotate each alpha value.
        for _, r in d.iterrows():
            ax.annotate(rf"$\alpha$={r.alpha:.1f}",
                        (r.core_mean_size, r.core_hit),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=6.5, color=PALETTE["Caption-Dist FGW"],
                        alpha=0.85)

    # Single-point recipes (one marker each).
    for method, label_short in METHODS.items():
        if method in ("FGW transitive (identity bridge)",
                      "FGW direct (M = caption cos)"):
            continue
        row = sub[sub.method == method]
        if row.empty:
            continue
        r = row.iloc[0]
        if not (np.isfinite(r.core_mean_size) and np.isfinite(r.core_hit)):
            continue
        ax.scatter([r.core_mean_size], [r.core_hit],
                   s=110, color=PALETTE[label_short],
                   edgecolor="white", linewidth=0.8, alpha=0.95, zorder=4,
                   label=label_short)

    ax.set_xscale("log")
    ax.set_xlabel(r"mean core size $\langle|C_i|\rangle$ "
                  r"(log; smaller = sharper plan rows)")
    ax.set_ylabel(r"core-set hit rate "
                  r"$\Pr[\mathrm{gt}_i \in C_i]$")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    sns.despine(ax=ax)


def main() -> None:
    if not TABLE_CSV.exists():
        print(f"[core-set plot] skip: {TABLE_CSV} not present. "
              "Run code/extended_metrics_table.py first.")
        return
    df = pd.read_csv(TABLE_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    _panel(axes[0], df[df.scope == "aggregate"],
           "Aggregate scope (all rows)")
    _panel(axes[1], df[df.scope == "heldout"],
           "Held-out scope (rows outside $S_{\\mathrm{compare}}$)")

    # Single legend from the right panel (it has more entries usually).
    handles, labels = axes[1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(by_label), 5),
               fontsize=9, frameon=False)

    fig.suptitle(
        "Core-set hit rate vs mean core size --- a sharp plan that is "
        "right sits in the upper left; a diffuse plan that hits by accident "
        "sits in the upper right",
        fontsize=12, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[core-set plot] wrote {OUT}")


if __name__ == "__main__":
    main()
