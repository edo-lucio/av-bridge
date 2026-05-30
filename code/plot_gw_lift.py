r"""Benefit of the Gromov-Wasserstein (structural) term, by encoder regime.

For an FGW recipe, the structural weight alpha turns the GW term on. The
"GW lift" of a metric is how much the best structural setting improves it
over the pure-feature baseline (alpha = 0):

    lift(metric) = max_{alpha > 0} metric(alpha)  -  metric(alpha = 0)

A grouped barplot (one group per retrieval metric, one bar per regime)
then reads off directly which regime benefits more from the GW term, and
on which metrics. The alpha that achieves each peak is annotated, so the
reader sees where the benefit comes from and that lift = 0 means the
structural term adds nothing over pure feature matching.

Recipes (the GW term is most directly isolated in caption-cost FGW, where
alpha weights image/audio GW against the caption feature cost in
cross-modal space; in the transitive bridge alpha acts on the legs):
  --recipe caption     exp_d{suffix}/sweep.csv          (no K filter)
  --recipe transitive  exp_c{suffix}/sweep_transitive.csv (K = 300)

Held-out scope by default. Reads only the sweep CSVs (no embeddings).

Usage:
  python code/plot_gw_lift.py
  python code/plot_gw_lift.py --recipe transitive
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
PLOT_DIR = RES / "exp_grid" / "plots"

REGIMES = {
    "text-grounded": {"suffix": "", "color": "#2a7fff"},
    "text-free":     {"suffix": "__dinov2-large__mert-330m", "color": "#dd8452"},
}

METRICS = [("R@1", "$R@1$"), ("R@5", "$R@5$"), ("R@10", "$R@10$"),
           ("cat_precision_10", "Cat-prec@10")]

RECIPES = {
    "caption":    {"csv": "exp_d{suffix}/sweep.csv",
                   "use_K": False, "title": "Caption-Distance FGW"},
    "transitive": {"csv": "exp_c{suffix}/sweep_transitive.csv",
                   "use_K": True, "title": "Transitive bridge"},
}


def _lifts(csv: Path, use_K: bool, K: int, scope: str) -> dict | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    if use_K and "K" in df.columns:
        df = df[df.K == K]
    if df.empty or 0.0 not in set(df.alpha.round(6)):
        return None
    out = {}
    for col, _ in METRICS:
        base = float(df[np.isclose(df.alpha, 0.0)][col].iloc[0])
        pos = df[df.alpha > 0]
        if pos.empty:
            out[col] = (0.0, float("nan"))
            continue
        peak = float(pos[col].max())
        astar = float(pos.loc[pos[col].idxmax(), "alpha"])
        out[col] = (peak - base, astar)
    return out


def render(out_path: Path, recipe: dict, K: int, scope: str) -> None:
    data = {}
    for reg, spec in REGIMES.items():
        csv = RES / recipe["csv"].format(suffix=spec["suffix"])
        lifts = _lifts(csv, recipe["use_K"], K, scope)
        if lifts is not None:
            data[reg] = lifts
    if not data:
        print(f"[gw-lift] no data for recipe={recipe['title']}, scope={scope}")
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(data), 1)

    for i, (reg, lifts) in enumerate(data.items()):
        vals = [lifts[c][0] for c, _ in METRICS]
        astars = [lifts[c][1] for c, _ in METRICS]
        offset = (i - (len(data) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=REGIMES[reg]["color"], edgecolor="black",
                      linewidth=0.6, label=reg)
        for b, v, a in zip(bars, vals, astars):
            if not np.isfinite(v):
                continue
            tag = f"$\\alpha^*$={a:.1f}" if v > 1e-9 else "—"
            ax.annotate(f"{v:+.3f}\n{tag}",
                        xy=(b.get_x() + b.get_width() / 2,
                            v + (0.004 if v >= 0 else -0.004)),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS])
    ax.set_ylabel(r"GW lift  $=\ \max_{\alpha>0}\ \mathrm{metric}(\alpha)\ -\ "
                  r"\mathrm{metric}(\alpha{=}0)$", fontsize=10)
    ax.set_title(
        f"Benefit of the Gromov--Wasserstein term by regime — "
        f"{recipe['title']} ({scope})\n"
        f"bars above 0: structure helps; $\\alpha^*$ = peak setting",
        fontsize=11.5)
    ax.legend(title="encoder regime", fontsize=10, title_fontsize=10)
    ax.margins(y=0.18)
    sns.despine(ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[gw-lift] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="caption", choices=list(RECIPES.keys()))
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    default_name = f"gw_lift__{args.recipe}__{args.scope}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, recipe, args.K, args.scope)


if __name__ == "__main__":
    main()
