r"""GW-lift trajectory: metric vs structural weight alpha, by encoder regime.

The bar version (``plot_gw_lift.py``) collapses alpha into a single number
(max lift over alpha>0). This view keeps alpha on the x-axis so the *shape*
of the dependence is visible: for each metric and regime we plot the metric
as a function of alpha and draw a dashed horizontal line at that regime's
alpha=0 value -- its "pure-feature" baseline. The gap between the solid curve
and its own dashed baseline is the GW lift at that alpha (the area is shaded).

The intended reading (stationarity-style: a series against its baseline
level): the text-free regime climbs above its alpha=0 line as alpha rises --
it improves the more the Gromov--Wasserstein term influences the coupling --
while the text-grounded regime stays at or below its baseline.

Reads only the sweep CSVs (no embeddings):
  --recipe caption     exp_d{suffix}/sweep.csv            (no K filter)
  --recipe transitive  exp_c{suffix}/sweep_transitive.csv (K = 300)

Output:
  results/exp_grid/plots/gw_lift_curve__{recipe}__{scope}.png

Usage:
  python code/plot_gw_lift_curve.py
  python code/plot_gw_lift_curve.py --recipe transitive
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

REGIMES = {
    "text-grounded": {"suffix": "", "color": "#2a7fff"},
    "text-free":     {"suffix": "__dinov2-large__mert-330m", "color": "#dd8452"},
}

METRICS = [("R@10", "$R@10$"), ("cat_precision_10", "Cat-prec@10")]

RECIPES = {
    "caption":    {"csv": "exp_d{suffix}/sweep.csv",
                   "use_K": False, "title": "Caption-Distance FGW"},
    "transitive": {"csv": "exp_c{suffix}/sweep_transitive.csv",
                   "use_K": True, "title": "Transitive bridge"},
}


def _series(csv: Path, use_K: bool, K: int, scope: str) -> pd.DataFrame | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    if use_K and "K" in df.columns:
        df = df[df.K == K]
    if df.empty or not np.isclose(df.alpha, 0.0).any():
        return None
    return df.sort_values("alpha")


def render(out_path: Path, recipe: dict, K: int, scope: str) -> None:
    data = {}
    for reg, spec in REGIMES.items():
        csv = RES / recipe["csv"].format(suffix=spec["suffix"])
        df = _series(csv, recipe["use_K"], K, scope)
        if df is not None:
            data[reg] = df
    if not data:
        print(f"[gw-lift-curve] no data for {recipe['title']}, scope={scope}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, (col, mlabel) in zip(axes, METRICS):
        for reg, df in data.items():
            color = REGIMES[reg]["color"]
            a = df.alpha.to_numpy(dtype=float)
            y = df[col].to_numpy(dtype=float)
            base = float(y[np.isclose(a, 0.0)][0])
            lift = y - base
            ax.plot(a, lift, "-o", color=color, lw=1.9, ms=5.5, label=reg,
                    zorder=3)
            ax.fill_between(a, 0.0, lift, color=color, alpha=0.10, zorder=0)
            j = int(np.argmax(lift))
            if lift[j] > 1e-6:
                ax.annotate(rf"$+{lift[j]:.3f}$ @ $\alpha{{=}}{a[j]:g}$",
                            xy=(a[j], lift[j]), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=8.5, color=color)
        ax.axhline(0.0, ls="--", color="grey", lw=1.3, alpha=0.9, zorder=1)
        ax.set_title(mlabel, fontsize=11.5)
        ax.set_xticks(sorted({float(v) for df in data.values()
                              for v in df.alpha}))
        ax.margins(y=0.20)
        ax.set_xlabel(r"structural weight $\alpha$  (GW influence $\to$)")
        sns.despine(ax=ax)
    axes[0].set_ylabel(
        r"GW lift  $=\ \mathrm{metric}(\alpha)\ -\ \mathrm{metric}(\alpha{=}0)$",
        fontsize=10)

    handles = [mlines.Line2D([], [], color=REGIMES[r]["color"], marker="o",
                             lw=1.9, label=r) for r in data]
    handles.append(mlines.Line2D([], [], color="grey", ls="--", lw=1.2,
                                 label=r"$\alpha=0$ baseline (per regime)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Retrieval vs. GW weight $\\alpha$ — {recipe['title']} ({scope})\n"
        "shaded gap to each dashed $\\alpha{=}0$ line is the GW lift; "
        "text-free climbs as $\\alpha$ grows, text-grounded does not",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[gw-lift-curve] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="caption", choices=list(RECIPES.keys()))
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    out = (Path(args.out) if args.out
           else PLOT_DIR / f"gw_lift_curve__{args.recipe}__{args.scope}.png")
    render(out, recipe, args.K, args.scope)


if __name__ == "__main__":
    main()
