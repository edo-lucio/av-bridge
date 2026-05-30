r"""Caption-Distance FGW: synergy of M (text cost) and GW (intra-modal
structure) shown as an alpha-sweep with both endpoints made visible.

The figure makes one chapter-level point: for retrieval-style metrics
(instance and category), neither the pure text-cost endpoint
(alpha = 0, Sinkhorn on caption distances alone) nor the pure
structural endpoint (alpha = 1, equivalent to Pure-GW with no M) is
optimal. The interior of the alpha range beats both endpoints --- this
is the empirical signature of the M-and-GW *synergy*. Three retrieval-
style metrics are shown side by side so the interior optimum can be
read at three granularities (R@10, Cat-prec@10, instance Prec@10 =
R@10 / 10).

Layout: 1 row x 3 cols, one panel per metric.
  Per panel: exp_d alpha-sweep over {0.0, 0.3, 0.5, 0.7, 0.9} with a
  star marker at alpha = 1.0 showing the Pure-GW endpoint. Random
  floor and Text-only ceiling overlaid as horizontal references.
  Peak alpha for each regime annotated.

Both encoder regimes (canonical / text-free) are overlaid in the
same panel, distinguished by colour. Default scope is aggregate
(the noisy 30-row heldout obscures the interior peak); pass
--scope heldout to see the noisier version.

Usage:
  python code/plot_dfgw_synergy.py
  python code/plot_dfgw_synergy.py --scope heldout
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

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

PAIRS = {
    "text-grounded": {
        "suffix": "",
        "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
        "color":  "#2a7fff",
    },
    "text-free": {
        "suffix": "__dinov2-large__mert-330m",
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "color":  "#dd8452",
    },
}

# Special key ``__precision_10`` is computed inline as R@10 / 10 ---
# instance precision@10 under single-ground-truth-per-query.
METRICS = [
    ("R@10",              "$R@10$",                True),
    ("cat_precision_10",  "Cat-prec@10",           True),
    ("__precision_10",    "Prec@10 (instance)",    True),
]


def _resolve(row: pd.Series, metric: str) -> float:
    if metric == "__precision_10":
        r10 = float(row.get("R@10", float("nan")))
        return r10 / 10.0 if np.isfinite(r10) else float("nan")
    return float(row.get(metric, float("nan")))


def _series(csv: Path, scope: str, metric: str,
            alpha: float | None = None) -> float | tuple[np.ndarray, np.ndarray]:
    if not csv.exists():
        return (np.array([]), np.array([])) if alpha is None else float("nan")
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    if df.empty:
        return (np.array([]), np.array([])) if alpha is None else float("nan")
    if alpha is not None:
        sub = df[np.isclose(df.alpha, alpha)] if "alpha" in df.columns else df
        if sub.empty:
            sub = df  # for recipes with NaN alpha
        return _resolve(sub.iloc[0], metric)
    df = df.sort_values("alpha")
    ys = np.array([_resolve(r, metric) for _, r in df.iterrows()])
    xs = df["alpha"].values.astype(float)
    return xs, ys


def render(out_path: Path, scope: str) -> None:
    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.4 * n_cols, 4.6),
                             squeeze=False)

    any_data = False
    for col_idx, (col, label, synergy_expected) in enumerate(METRICS):
        ax = axes[0, col_idx]
        peaks: list[tuple[str, float, tuple[float, float, float]]] = []

        for pair_key, pair_spec in PAIRS.items():
            suffix = pair_spec["suffix"]
            color = pair_spec["color"]

            xs, ys = _series(
                RES / f"exp_d{suffix}" / "sweep.csv", scope, col)
            if not xs.size:
                continue
            any_data = True
            ax.plot(xs, ys, marker="o", lw=2.0, ls="-",
                    color=color, zorder=4,
                    label=f"exp_d $\\alpha$-sweep ({pair_key})")

            if np.any(np.isfinite(ys)):
                i_peak = int(np.nanargmax(ys))
                x_peak = xs[i_peak]
                y_peak = ys[i_peak]
                ax.scatter([x_peak], [y_peak], s=160, marker="*",
                           facecolor=color, edgecolor="black",
                           linewidth=0.9, zorder=6)
                peaks.append((pair_key, x_peak, color))

            v_gw = _series(
                RES / f"exp_unsup{suffix}" / "sweep.csv",
                scope, col, alpha=1.0)
            if np.isfinite(v_gw):
                ax.plot(1.0, v_gw, marker="X", markersize=12,
                        color=color, markeredgecolor="black",
                        markeredgewidth=0.7, linestyle="None",
                        zorder=5)
                ax.plot([xs[-1], 1.0], [ys[-1], v_gw],
                        color=color, lw=1.2, ls=(0, (1, 1)),
                        alpha=0.55, zorder=3)

            v_text = _series(
                RES / f"exp_text{suffix}" / "sweep.csv", scope, col)
            if isinstance(v_text, tuple):
                v_text = float(v_text[1][0]) if v_text[1].size else float("nan")
            if np.isfinite(v_text):
                ax.axhline(v_text, color=color, lw=1.0, ls="--",
                           alpha=0.5, zorder=2)

            v_rand = _series(
                RES / f"exp_random{suffix}" / "sweep.csv", scope, col)
            if isinstance(v_rand, tuple):
                v_rand = float(v_rand[1][0]) if v_rand[1].size else float("nan")
            if np.isfinite(v_rand):
                ax.axhline(v_rand, color=color, lw=1.0, ls=":",
                           alpha=0.4, zorder=1)

        ax.set_xticks(ALPHA_GRID + [1.0])
        ax.set_xticklabels(
            [f"{a:.1f}" for a in ALPHA_GRID] + ["1.0\n(Pure-GW)"],
            fontsize=8,
        )
        ax.set_xlim(-0.05, 1.08)
        ax.set_xlabel(r"$\alpha$  "
                      r"(text cost $M$ $\leftrightarrow$ structural GW)",
                      fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        sns.despine(ax=ax)

        for i, (pair_key, x_peak, color) in enumerate(peaks):
            ax.text(
                0.03, 0.04 + i * 0.07,
                rf"{pair_key}: $\alpha^\star = {x_peak:.1f}$",
                transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=color,
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5),
            )

    if not any_data:
        print(f"[dfgw-synergy] no data found at scope={scope}.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color="black", lw=1.8, marker="o",
               label=r"exp\_d $\alpha$-sweep ($\alpha \in [0, 0.9]$)"),
        Line2D([], [], color="black", linestyle="None", marker="*",
               markersize=14, markeredgecolor="black",
               label=r"peak $\alpha^\star$ for that metric / regime"),
        Line2D([], [], color="black", linestyle="None", marker="X",
               markersize=10, markeredgecolor="black",
               label=r"Pure-GW endpoint ($\alpha = 1$, no $M$)"),
        Line2D([], [], color="black", lw=1.0, ls="--",
               label="Text-only reference"),
        Line2D([], [], color="black", lw=1.0, ls=":",
               label="Random floor"),
    ]
    pair_handles = [
        Line2D([], [], color=PAIRS[k]["color"], lw=2.2,
               label=PAIRS[k]["label"])
        for k in PAIRS.keys()
    ]
    fig.legend(handles=pair_handles, title="encoder regime (colour)",
               loc="lower left", bbox_to_anchor=(0.02, -0.08),
               ncol=1, fontsize=9, title_fontsize=10, frameon=False)
    fig.legend(handles=handles, title="curve / reference type",
               loc="lower right", bbox_to_anchor=(0.98, -0.14),
               ncol=3, fontsize=9, title_fontsize=10, frameon=False)

    fig.suptitle(
        r"Caption-Distance FGW: synergy of text cost $M$ and "
        r"intra-modal GW (scope = " + scope + ")",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[dfgw-synergy] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"],
                    help="Evaluation scope. Default aggregate (less "
                         "noisy --- the synergy is clearer there). "
                         "Held-out has 30 rows and the interior peak "
                         "can be obscured by sampling noise.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from "
                         "scope under results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"dfgw_synergy__{args.scope}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.scope)


if __name__ == "__main__":
    main()
