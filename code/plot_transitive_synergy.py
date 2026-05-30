r"""Transitive Transport bridge: retrieval vs alpha, with the same
references used for the Caption-Distance FGW synergy figure.

Sibling of plot_dfgw_synergy.py. Where that figure shows the
Caption-Distance FGW (exp_d) alpha-sweep, this one shows the Transitive
bridge (exp_c, composed image->audio plan) alpha-sweep at fixed K,
plotted against the SAME recipe-independent references: the Pure-GW
direct-alignment point (exp_unsup), the Text-only reference, and the
Random floor. This lets the bridge be read on the same axes as exp_d.

The chapter-level point differs from exp_d's, by design:
  * exp_d has a direct cross-modal feature term M, so its alpha=0
    endpoint is "pure M" and the synergy is M+GW; the interior peak
    beats both endpoints.
  * The bridge has NO direct M. Its alpha blends each LEG's FGW
    (cross-modal ridge feature term vs intra-modal structural GW). Its
    alpha=0 endpoint is "pure-ridge-feature legs", and retrieval RISES
    to an interior-alpha peak because the structural smoothing is what
    makes the two caption-anchored legs reinforce on the correct pair
    under the matrix product.

Honesty notes baked into the figure:
  * The bridge sweep grid is alpha in {0, 0.3, 0.5, 0.7, 0.9}; there is
    no alpha=1 bridge cell, so the alpha=1 marker is the DIRECT Pure-GW
    recipe (exp_unsup), shown as a standalone reference -- NOT a
    continuation of the bridge curve. Hence no dotted 0.9->1.0
    connector (unlike the exp_d figure, where Pure-GW genuinely is the
    alpha=1 continuation of the same recipe).

Layout: 1 row x 3 cols, one panel per metric (R@10, Cat-prec@10,
instance Prec@10 = R@10/10). Both encoder regimes overlaid by colour.

Usage:
  python code/plot_transitive_synergy.py
  python code/plot_transitive_synergy.py --scope heldout --K 300
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

# ``__precision_10`` = R@10 / 10
# (instance precision@10 under single-ground-truth-per-query).
METRICS = [
    ("R@10",              "$R@10$"),
    ("cat_precision_10",  "Cat-prec@10"),
    ("__precision_10",    "Prec@10 (instance)"),
]


def _resolve(row: pd.Series, metric: str) -> float:
    if metric == "__precision_10":
        r10 = float(row.get("R@10", float("nan")))
        return r10 / 10.0 if np.isfinite(r10) else float("nan")
    return float(row.get(metric, float("nan")))


def _scoped(csv: Path, scope: str, K: int | None) -> pd.DataFrame | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    if K is not None and "K" in df.columns:
        df = df[df.K == K]
    return df if not df.empty else None


def _sweep(csv: Path, scope: str, metric: str,
           K: int | None) -> tuple[np.ndarray, np.ndarray]:
    df = _scoped(csv, scope, K)
    if df is None:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    ys = np.array([_resolve(r, metric) for _, r in df.iterrows()])
    xs = df["alpha"].values.astype(float)
    return xs, ys


def _point(csv: Path, scope: str, metric: str,
           alpha: float | None = None) -> float:
    """Single scalar (Pure-GW / Text / Random references)."""
    df = _scoped(csv, scope, None)
    if df is None:
        return float("nan")
    if alpha is not None and "alpha" in df.columns:
        sub = df[np.isclose(df.alpha, alpha)]
        if not sub.empty:
            df = sub
    return _resolve(df.iloc[0], metric)


def render(out_path: Path, scope: str, K: int) -> None:
    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.4 * n_cols, 4.6),
                             squeeze=False)

    any_data = False
    for col_idx, (col, label) in enumerate(METRICS):
        ax = axes[0, col_idx]
        peaks: list[tuple[str, float, str]] = []

        for pair_key, pair_spec in PAIRS.items():
            suffix = pair_spec["suffix"]
            color = pair_spec["color"]

            xs, ys = _sweep(
                RES / f"exp_c{suffix}" / "sweep_transitive.csv",
                scope, col, K)
            if not xs.size:
                continue
            any_data = True
            ax.plot(xs, ys, marker="o", lw=2.0, ls="-",
                    color=color, zorder=4,
                    label=f"bridge $\\alpha$-sweep ({pair_key})")

            if np.any(np.isfinite(ys)):
                i_peak = int(np.nanargmax(ys))
                ax.scatter([xs[i_peak]], [ys[i_peak]], s=160, marker="*",
                           facecolor=color, edgecolor="black",
                           linewidth=0.9, zorder=6)
                peaks.append((pair_key, xs[i_peak], color))

            # Pure-GW reference at alpha = 1 (DIRECT recipe, exp_unsup).
            # Shown as a standalone marker -- NOT connected to the bridge
            # curve, because it is a different recipe (see module docstring).
            v_gw = _point(
                RES / f"exp_unsup{suffix}" / "sweep.csv",
                scope, col, alpha=1.0)
            if np.isfinite(v_gw):
                ax.plot(1.0, v_gw, marker="X", markersize=12,
                        color=color, markeredgecolor="black",
                        markeredgewidth=0.7, linestyle="None",
                        zorder=5)

            v_text = _point(
                RES / f"exp_text{suffix}" / "sweep.csv", scope, col)
            if np.isfinite(v_text):
                ax.axhline(v_text, color=color, lw=1.0, ls="--",
                           alpha=0.5, zorder=2)

            v_rand = _point(
                RES / f"exp_random{suffix}" / "sweep.csv", scope, col)
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
                      r"(leg feature term $\leftrightarrow$ structural GW)",
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
        print(f"[transitive-synergy] no data at scope={scope}, K={K}.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color="black", lw=1.8, marker="o",
               label=r"bridge $\alpha$-sweep ($\alpha \in [0, 0.9]$)"),
        Line2D([], [], color="black", linestyle="None", marker="*",
               markersize=14, markeredgecolor="black",
               label=r"peak $\alpha^\star$ for that metric / regime"),
        Line2D([], [], color="black", linestyle="None", marker="X",
               markersize=10, markeredgecolor="black",
               label=r"Pure-GW reference ($\alpha=1$, direct, not bridge)"),
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

    # Scope-specific headline: the bridge shows an INTERIOR peak only
    # out-of-sample (held-out); in-sample (aggregate, anchor-dominated)
    # alpha=0 fits best and retrieval declines. That contrast is the
    # regularisation signature -- the structural term trades in-sample
    # fit for out-of-sample generalisation.
    if scope == "heldout":
        tail = ("the structural term drives an interior-$\\alpha$ peak "
                "(out-of-sample)")
    else:
        tail = ("$\\alpha=0$ fits best in-sample; the structural term "
                "trades in-sample fit for generalisation (cf. held-out)")
    fig.suptitle(
        rf"Transitive bridge: retrieval vs $\alpha$ ($K={K}$, "
        rf"scope = {scope}); {tail}",
        fontsize=12.5, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[transitive-synergy] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"],
                    help="Evaluation scope. Default aggregate (less "
                         "noisy --- the interior peak is clearer there).")
    ap.add_argument("--K", type=int, default=300,
                    help="Anchor count for the bridge legs (default 300).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    default_name = f"transitive_synergy__{args.scope}__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.scope, args.K)


if __name__ == "__main__":
    main()
