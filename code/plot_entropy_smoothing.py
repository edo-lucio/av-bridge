r"""Plan row-entropy diagnostic: visualises FGW's smoothing effect.

Argument the figure supports: per-leg retrieval degrades linearly
with alpha because the legs' plans become smoother (higher row
entropy) as the GW structural term takes over. The composed
Transitive bridge, however, *benefits* from this smoothing because
between-leg structural agreement is what survives the matrix
product. Hence the composed bridge's R@10 tends to peak at an
intermediate alpha where leg entropy is "moderate" --- not
maximally sharp, not maximally diffuse.

Layout: 1 row x 2 cols, one panel per encoder regime.
  Left y-axis: mean row entropy of T_iv (Leg A), T_ac (Leg B), and
               their geometric mean. Read from exp_a / exp_b
               sweep.csv at K = 300, heldout scope, in nats.
  Right y-axis: composed Transitive bridge R@10 at K = 300,
                heldout scope, read from
                exp_c/sweep_transitive.csv.
  X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9}.

Requires the patched run_experiments.py that writes the
``plan_row_entropy`` column to CSV. Older CSVs without that column
will render only the R@10 curve and print a warning.

Usage:
  python code/plot_entropy_smoothing.py
  python code/plot_entropy_smoothing.py --K 300
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
              palette="colorblind", font_scale=0.9)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

PAIRS = {
    "canonical": {
        "label":     "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
        "c_suffix":  "",
        "a_suffix":  "",
        "b_suffix":  "",
    },
    "textfree": {
        "label":     "text-free (DINOv2-L $\\times$ MERT-330m)",
        "c_suffix":  "__dinov2-large__mert-330m",
        "a_suffix":  "__dinov2-large",
        "b_suffix":  "__mert-330m",
    },
}

COL_LEG_A = "#1f77b4"
COL_LEG_B = "#9467bd"
COL_GEO   = "#7f7f7f"
COL_COMP  = "#d62728"


def _leg_entropy_series(leg_key: str, suffix: str, K: int,
                        scope: str) -> tuple[np.ndarray, np.ndarray]:
    csv = RES / f"exp_{leg_key}{suffix}" / "sweep.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & (df.K == K)]
    if "plan_row_entropy" not in df.columns or df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    return (df["alpha"].values.astype(float),
            df["plan_row_entropy"].values.astype(float))


def _composed_r10_series(suffix: str, K: int,
                         scope: str) -> tuple[np.ndarray, np.ndarray]:
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & (df.K == K)]
    if df.empty:
        return np.array([]), np.array([])
    df = df.sort_values("alpha")
    return (df["alpha"].values.astype(float),
            df["R@10"].values.astype(float))


def render(out_path: Path, K: int,
           scope_leg: str, scope_composed: str) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.5),
                             squeeze=False)

    any_data = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax = axes[0, col_idx]
        ax_r = ax.twinx()

        xs_a, ys_a = _leg_entropy_series(
            "a", pair_spec["a_suffix"], K, scope_leg)
        xs_b, ys_b = _leg_entropy_series(
            "b", pair_spec["b_suffix"], K, scope_leg)

        leg_data_ok = xs_a.size and xs_b.size and np.array_equal(xs_a, xs_b)
        if leg_data_ok:
            any_data = True
            ax.plot(xs_a, ys_a, marker="o", lw=1.8, color=COL_LEG_A,
                    label="Leg A entropy")
            ax.plot(xs_b, ys_b, marker="s", lw=1.8, color=COL_LEG_B,
                    label="Leg B entropy")
            geo = np.sqrt(np.maximum(ys_a, 0) * np.maximum(ys_b, 0))
            ax.plot(xs_a, geo, ls="--", marker="x", lw=1.4, color=COL_GEO,
                    alpha=0.85, label="geometric mean")

        xs_c, ys_c = _composed_r10_series(
            pair_spec["c_suffix"], K, scope_composed)
        if xs_c.size:
            any_data = True
            ax_r.plot(xs_c, ys_c, marker="D", lw=2.0, color=COL_COMP,
                      label=f"Composed bridge $R@10$ (K={K})")
            i_peak = int(np.nanargmax(ys_c))
            ax_r.axvline(xs_c[i_peak], color=COL_COMP, ls=":", lw=1.0,
                         alpha=0.55)
            ax_r.annotate(
                rf"peak $\alpha = {xs_c[i_peak]:.1f}$",
                xy=(xs_c[i_peak], ys_c[i_peak]),
                xytext=(6, -14), textcoords="offset points",
                color=COL_COMP, fontsize=8, ha="left",
            )

        ax.set_xticks(ALPHA_GRID)
        ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID], fontsize=9)
        ax.set_xlim(-0.05, 0.95)
        ax.set_xlabel(r"$\alpha$  "
                      r"(cross-modal $\leftrightarrow$ structural)",
                      fontsize=10)
        ax.set_ylabel("plan row entropy (nats)", fontsize=10, color="black")
        ax_r.set_ylabel(rf"composed bridge $R@10$  "
                        rf"(scope = {scope_composed.replace('_', ' ')})",
                        fontsize=10, color=COL_COMP)
        ax_r.tick_params(axis="y", labelcolor=COL_COMP)
        ax.set_title(pair_spec["label"], fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax_r.grid(False)

        if not leg_data_ok:
            ax.text(0.5, 0.5,
                    "no plan_row_entropy column yet\n"
                    "(re-run sbatch jobs/all.job to populate)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="grey", style="italic")

    if not any_data:
        print(f"[entropy] no data found at K={K}.")
        plt.close(fig)
        return

    legend_handles = [
        Line2D([], [], color=COL_LEG_A, lw=2.2, marker="o",
               label="Leg A: image $\\to$ visual-text  (mean row entropy)"),
        Line2D([], [], color=COL_LEG_B, lw=2.2, marker="s",
               label="Leg B: audio $\\to$ audio-text  (mean row entropy)"),
        Line2D([], [], color=COL_GEO, lw=1.8, ls="--", marker="x",
               label=r"$\sqrt{H(T_{iv})\,H(T_{ac})}$  (geometric mean)"),
        Line2D([], [], color=COL_COMP, lw=2.2, marker="D",
               label=r"Composed Transitive bridge $R@10$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2,
               fontsize=10, frameon=False)

    fig.suptitle(
        rf"FGW smoothing diagnostic: leg row entropy vs composed bridge "
        rf"$R@10$  ($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[entropy] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="K at which to read both leg sweeps and the "
                         "composed bridge. Default 300.")
    ap.add_argument("--scope-leg", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for leg entropy. Default heldout.")
    ap.add_argument("--scope-composed", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for composed bridge R@10. "
                         "Default heldout (shared with reference "
                         "recipes).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs from K "
                         "under results/exp_grid/plots/.")
    args = ap.parse_args()

    default_name = f"entropy_smoothing__K{args.K}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.scope_leg, args.scope_composed)


if __name__ == "__main__":
    main()
