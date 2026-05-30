r"""Side-by-side category-precision@k panel for the two encoder regimes.

Reads the per-recipe category-precision curves emitted by
``analyse_levels.py`` for each encoder pair and draws them as a single
1x2 figure (text-grounded | text-free) with a shared y-axis and a
shared legend, so the two regimes can be compared at a glance in the
chapter.

Inputs (produced by ``analyse_levels.py --pair {text-grounded,text-free}``):
  results/exp_grid/levels_category_precision.csv            (text-grounded)
  results/exp_grid/levels_category_precision__text-free.csv (text-free)

Output:
  results/exp_grid/plots/levels_category_precision_panel.png

If a regime's CSV is missing the panel still renders for whichever
regime is present (the missing panel is annotated), so this can be run
before the text-free analysis has been generated on the cluster.

Usage:
  python code/plot_levels_category_precision_panel.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

REGIMES = [
    (r"Text-grounded (CLIP-L $\times$ CLAP-unfused)",
     RES / "exp_grid" / "levels_category_precision.csv"),
    (r"Text-free (DINOv2-L $\times$ MERT-330m)",
     RES / "exp_grid" / "levels_category_precision__text-free.csv"),
]

RECIPE_ORDER = [
    "Random (baseline)",
    "Transitive Transport Bridge",
    "Caption Distance FGW",
    "Pure-GW (intra-modal geom.)",
    "Raw caption cosine (ceiling)",
    "C-MCR (learned text bridge)",
    "Procrustes (rigid supervised)",
    "Direct ridge (supervised)",
]
RECIPE_COLOURS = dict(zip(RECIPE_ORDER,
                          sns.color_palette("colorblind",
                                            n_colors=len(RECIPE_ORDER))))


def _plot_panel(ax, title: str, csv: Path) -> float | None:
    """Draw one regime's curves; return its chance level (or None if absent)."""
    if not csv.exists():
        ax.text(0.5, 0.5, f"missing\n{csv.name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_title(title)
        return None
    df = pd.read_csv(csv)
    for rec in RECIPE_ORDER:
        sub = df[df.recipe == rec].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub.k, sub.category_precision, marker="o", lw=1.6,
                color=RECIPE_COLOURS[rec], label=rec)
    chance = float(df.category_precision_chance.iloc[0])
    ax.axhline(chance, color="grey", linestyle="--", lw=1.2,
               label=f"chance ({chance:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ (top-$k$ retrievals)")
    ax.set_title(title)
    sns.despine(ax=ax)
    return chance


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, csv) in zip(axes, REGIMES):
        _plot_panel(ax, title, csv)
    axes[0].set_ylabel(r"category precision$@k$  ($K_{\mathrm{cl}} = 15$)")

    handles, labels = [], []
    for ax in axes:
        h, ll = ax.get_legend_handles_labels()
        for hi, li in zip(h, ll):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Categorical precision vs retrieval depth "
                 "(held-out queries, both regimes)", fontsize=12)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "levels_category_precision_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
