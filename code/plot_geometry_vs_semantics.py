r"""Geometry vs semantics across the FGW alpha sweep.

Makes the report's claim visible in one figure: as the structural weight
alpha increases, the GEOMETRIC signal (pairwise-distance Pearson r) climbs
while the SEMANTIC signals (R@10, Cat@10) stay essentially flat. The
setting that best preserves pairwise geometry is therefore not the one
that retrieves best -- retrieval and geometry move independently.

Left axis  (semantic) : R@10 (exact) and Cat-prec@10 (coarse).
Right axis (geometric): Pearson r of source vs partner-target pairwise
                        distances.
X axis                : alpha.

Defaults to Caption-Distance FGW (exp_d) at aggregate scope, which is the
recipe/scope the report discusses (exp_d has no anchors, so aggregate is
the natural read). ``--recipe transitive`` switches to the bridge.

Usage:
  python code/plot_geometry_vs_semantics.py
  python code/plot_geometry_vs_semantics.py --recipe transitive --scope heldout
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

SEM_COLOR = "#2a7fff"
GEO_COLOR = "#c44e52"

RECIPES = {
    "caption": {
        "csv":   "exp_d{suffix}/sweep.csv",
        "use_K": False,
        "title": "Caption-Distance FGW",
        "out":   "geometry_vs_semantics__caption",
    },
    "transitive": {
        "csv":   "exp_c{suffix}/sweep_transitive.csv",
        "use_K": True,
        "title": "Transitive bridge",
        "out":   "geometry_vs_semantics__transitive",
    },
}


def render(out_path: Path, recipe: dict, suffix: str, scope: str,
           K: int) -> None:
    csv = RES / recipe["csv"].format(suffix=suffix)
    if not csv.exists():
        print(f"[geo-vs-sem] missing {csv}")
        return
    df = pd.read_csv(csv)
    df = df[df.scope == scope].copy()
    if recipe["use_K"] and "K" in df.columns:
        df = df[df.K == K]
    if df.empty:
        print(f"[geo-vs-sem] no rows at scope={scope}")
        return
    df = df.sort_values("alpha")
    x = df["alpha"].values.astype(float)
    r10 = df["R@10"].values.astype(float)
    cat = df["cat_precision_10"].values.astype(float)
    pear = df["pearson_r"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax2 = ax.twinx()

    ax.plot(x, r10, marker="o", lw=2.0, color=SEM_COLOR, label="$R@10$ (exact)")
    ax.plot(x, cat, marker="s", lw=1.8, ls="--", color=SEM_COLOR,
            alpha=0.8, label="Cat-prec@10 (coarse)")
    ax2.plot(x, pear, marker="^", lw=2.2, color=GEO_COLOR,
             label=r"Pearson $r$ (geometry)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in x])
    ax.set_xlabel(r"$\alpha$  (feature cost $\leftrightarrow$ structural GW)",
                  fontsize=11)
    ax.set_ylabel("semantic retrieval  ($R@10$, Cat-prec@10)",
                  fontsize=11, color=SEM_COLOR)
    ax2.set_ylabel(r"geometric: pairwise-distance Pearson $r$",
                   fontsize=11, color=GEO_COLOR)
    ax.tick_params(axis="y", labelcolor=SEM_COLOR)
    ax2.tick_params(axis="y", labelcolor=GEO_COLOR)
    ax.set_ylim(0, max(0.05, float(np.nanmax(np.r_[r10, cat])) * 1.25))
    ax2.set_ylim(0, max(0.05, float(np.nanmax(pear)) * 1.15))
    ax2.grid(False)

    handles = [
        Line2D([], [], color=SEM_COLOR, lw=2.0, marker="o", label="$R@10$ (exact)"),
        Line2D([], [], color=SEM_COLOR, lw=1.8, ls="--", marker="s",
               label="Cat-prec@10 (coarse)"),
        Line2D([], [], color=GEO_COLOR, lw=2.2, marker="^",
               label=r"Pearson $r$ (geometry)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    fig.suptitle(
        f"{recipe['title']}: geometry climbs while semantics stay flat\n"
        f"(scope = {scope}) — the most geometry-preserving $\\alpha$ is not "
        f"the best-retrieving one",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[geo-vs-sem] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="caption",
                    choices=list(RECIPES.keys()))
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--suffix", default="",
                    help="encoder-pair suffix, e.g. __dinov2-large__mert-330m")
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    default_name = f"{recipe['out']}__{args.scope}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, recipe, args.suffix, args.scope, args.K)


if __name__ == "__main__":
    main()
