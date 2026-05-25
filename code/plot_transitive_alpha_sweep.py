r"""Alpha sweep for the Transitive Transport Bridge at a fixed K,
overlaid against the other recipes as horizontal references or as
parallel sweeps where applicable.

For each metric (R@10, cat_precision_10, routes ratio, Pearson r),
one panel per encoder pair:
  X-axis: alpha in {0.0, 0.3, 0.5, 0.7, 0.9}.
  Y-axis: metric value at chosen scope.
  Transitive bridge: line with markers, varying with alpha at the
                     chosen K (default K = 300, the canonical
                     operating point).
  Caption-Distance FGW: parallel alpha-sweep curve (no K dependence
                       in that recipe).
  Pure-GW: single marker at alpha = 1.
  Random, Text-only: horizontal reference lines (no alpha dependence).

The point of the figure is to read off the alpha response of the
composed Transitive bridge against the in-formulation references,
holding K fixed. Defaults to one figure per pair, with rows =
encoder pair (canonical + text-free).

Usage:
  python code/plot_transitive_alpha_sweep.py
  python code/plot_transitive_alpha_sweep.py --K 200 --scope aggregate
  python code/plot_transitive_alpha_sweep.py --pair canonical
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
              palette="colorblind", font_scale=0.9)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]


# Encoder pairs and their suffix conventions.
PAIRS = {
    "canonical": {"suffix": "",
                  "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)"},
    "textfree":  {"suffix": "__dinov2-large__mert-330m",
                  "label":  "text-free (DINOv2-L $\\times$ MERT-330m)"},
}

# Metrics: (csv_column, display_label, axis_lo, axis_hi_or_None).
# Bounds None so the y-axis auto-fits the data range, including
# negative Pearson r at small held-out sizes.
METRICS = [
    ("R@10",            "$R@10$",                       None, None),
    ("cat_precision_10",   "Cat-prec@10 (class)",        None, None),
    ("__routes_ratio",  "Routes (correct / $K_{cl}$)",  None, None),
    ("pearson_r",       "Pearson $r$",                  None, None),
]


PALETTE = {
    "c-transitive": "#2a7fff",   # blue, the protagonist
    "random":       "#777777",   # grey
    "d":            "#dd8452",   # orange
    "unsup":        "#55a868",   # green
    "text":         "#c44e52",   # red
}


def _metric_from_row(row: pd.Series, col: str) -> float:
    if col == "__routes_ratio":
        total = float(row.get("routes_total", float("nan")))
        if not np.isfinite(total) or total == 0:
            return float("nan")
        return float(row.get("routes_correct", float("nan"))) / total
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _load_transitive_alpha_curve(suffix: str, K: int,
                                  scope: str) -> pd.DataFrame:
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & (df.K == K)]
    return df.sort_values("alpha")


def _load_d_alpha_curve(suffix: str, scope: str) -> pd.DataFrame:
    csv = RES / f"exp_d{suffix}" / "sweep.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[df.scope == scope]
    return df.sort_values("alpha")


def _load_single_value(name: str, suffix: str, scope: str,
                       metric: str) -> float:
    csv = RES / f"exp_{name}{suffix}" / "sweep.csv"
    if not csv.exists():
        return float("nan")
    df = pd.read_csv(csv)
    sub = df[df.scope == scope]
    if sub.empty:
        return float("nan")
    return _metric_from_row(sub.iloc[0], metric)


def render(out_path: Path, pair_keys: list[str],
           K: int, scope: str) -> None:
    n_rows = len(pair_keys)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    any_data = False
    for row_idx, pair_key in enumerate(pair_keys):
        suffix = PAIRS[pair_key]["suffix"]
        pair_label = PAIRS[pair_key]["label"]

        df_tr     = _load_transitive_alpha_curve(suffix, K, scope)
        df_tr_agg = _load_transitive_alpha_curve(suffix, K, "aggregate")
        df_d      = _load_d_alpha_curve(suffix, scope)
        if not df_tr.empty or not df_tr_agg.empty or not df_d.empty:
            any_data = True

        for col_idx, (col, label, vmin, vmax) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            # Transitive aggregate (in-sample) curve, drawn first so it
            # sits behind the held-out solid line. Includes anchor rows.
            if not df_tr_agg.empty:
                ys = [_metric_from_row(r, col) for _, r in df_tr_agg.iterrows()]
                xs = df_tr_agg["alpha"].values
                ax.plot(xs, ys, marker="o", markersize=4, lw=1.2,
                        ls="--", color=PALETTE["c-transitive"], alpha=0.4,
                        label="Transitive (in-sample)")

            # Transitive held-out curve.
            if not df_tr.empty:
                ys = [_metric_from_row(r, col) for _, r in df_tr.iterrows()]
                xs = df_tr["alpha"].values
                ax.plot(xs, ys, marker="o", lw=1.8,
                        color=PALETTE["c-transitive"],
                        label="Transitive (held-out)")

            # Caption-Distance FGW curve at the same scope.
            if not df_d.empty:
                ys = [_metric_from_row(r, col) for _, r in df_d.iterrows()]
                xs = df_d["alpha"].values
                ax.plot(xs, ys, marker="s", markersize=5, lw=1.4,
                        ls="--", color=PALETTE["d"], alpha=0.85,
                        label="Caption Distance FGW")

            # Random horizontal reference (no alpha dependence).
            v = _load_single_value("random", suffix, scope, col)
            if np.isfinite(v):
                ax.axhline(v, color=PALETTE["random"], linestyle=":",
                           lw=1.4, alpha=0.85, label="Random (baseline)")

            # Pure-GW single marker at alpha = 1.
            v = _load_single_value("unsup", suffix, scope, col)
            if np.isfinite(v):
                ax.plot(1.0, v, marker="*", markersize=13,
                        color=PALETTE["unsup"], markeredgecolor="black",
                        markeredgewidth=0.5, linestyle="None",
                        label=r"Pure-GW ($\alpha = 1$)")

            # Text-only horizontal reference (no alpha dependence).
            v = _load_single_value("text", suffix, scope, col)
            if np.isfinite(v):
                ax.axhline(v, color=PALETTE["text"],
                           linestyle=(0, (3, 1, 1, 1)),
                           lw=1.4, alpha=0.85,
                           label="Text-only (ceiling)")

            ax.set_xticks(ALPHA_GRID + [1.0])
            ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_GRID] + ["1.0"],
                               fontsize=7)
            ax.set_xlim(-0.05, 1.05)
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\alpha$  (cross-modal $\leftrightarrow$"
                              r" structural)", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{pair_label}\n\n{label}", fontsize=9)
            else:
                ax.set_ylabel(label, fontsize=8)
            if row_idx == 0:
                ax.set_title(label, fontsize=9)
            if vmin is not None and vmax is not None:
                ax.set_ylim(vmin, vmax)
            elif vmin is not None:
                ax.set_ylim(bottom=vmin)
            elif vmax is not None:
                ax.set_ylim(top=vmax)
            ax.grid(True, which="both", alpha=0.25)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[transitive-alpha] no transitive sweep data found at "
              f"K={K}, scope={scope}; nothing rendered.")
        plt.close(fig)
        return

    # Shared bottom legend from the first non-empty panel.
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
        if labels:
            break
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 6),
               fontsize=9, frameon=False)

    fig.suptitle(
        rf"Transitive Transport Bridge --- $\alpha$ sweep at $K = {K}$ "
        rf"(held-out solid, in-sample dashed; "
        rf"scope = {scope.replace('_', ' ')})",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[transitive-alpha] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300,
                    help="Fix K for the transitive alpha sweep. "
                         "Default 300 (canonical operating point).")
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for the sweep and reference recipes.")
    ap.add_argument("--pair", choices=["canonical", "textfree", "both"],
                    default="both")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs a name from "
                         "K and pair under results/exp_grid/plots/.")
    args = ap.parse_args()

    if args.pair == "both":
        keys = list(PAIRS.keys())
        default_name = f"transitive_alpha_sweep__K{args.K}.png"
    else:
        keys = [args.pair]
        default_name = (f"transitive_alpha_sweep__K{args.K}"
                        f"__{args.pair}.png")
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, keys, args.K, args.scope)


if __name__ == "__main__":
    main()
