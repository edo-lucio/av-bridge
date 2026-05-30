r"""K sweep for the Transitive Transport Bridge at a fixed alpha,
overlaid against the other recipes as horizontal reference lines.

For each metric (R@10, Cat-prec@10, instance Prec@10, Pearson r), one panel:
  X-axis: K (paired-anchor budget per leg; same K is used on both legs
           of the transitive composition).
  Y-axis: metric value at chosen scope.
  Transitive: line with markers, varying with K at the chosen alpha.
  Random, Caption-Distance FGW (at the same alpha), Pure-GW, Text-only:
  horizontal reference lines (no K dependence in this comparison).

The point of the figure is to identify the K at which the transitive
bridge surpasses each other recipe on each metric. Defaults to one
figure per pair, with rows = encoder pair (canonical + text-free).

Usage:
  python code/plot_transitive_K_sweep.py
  python code/plot_transitive_K_sweep.py --alpha 0.5 --scope aggregate
  python code/plot_transitive_K_sweep.py --pair canonical
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


PAIRS = {
    "canonical": {"suffix": "",
                  "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)"},
    "textfree":  {"suffix": "__dinov2-large__mert-330m",
                  "label":  "text-free (DINOv2-L $\\times$ MERT-330m)"},
}

# Both bounds are None so the y-axis auto-fits the data range.
METRICS = [
    ("R@10",            "$R@10$",  None, None),
    ("cat_precision_10","Cat@10",  None, None),
]

REFERENCE_RECIPES = [
    {"key":     "random",
     "label":   "Random (baseline)",
     "csv":     "exp_random{suffix}/sweep.csv",
     "select":  {"K": 0},
     "ls":      ":"},
    {"key":     "d",
     "label":   "Caption Distance FGW",
     "csv":     "exp_d{suffix}/sweep.csv",
     "select":  None,
     "ls":      "--",
     "needs_alpha": True},
    {"key":     "unsup",
     "label":   "Pure-GW",
     "csv":     "exp_unsup{suffix}/sweep.csv",
     "select":  {"K": 0},
     "ls":      "-."},
    {"key":     "text",
     "label":   "Text-only (ceiling)",
     "csv":     "exp_text{suffix}/sweep.csv",
     "select":  {"K": 0},
     "ls":      (0, (3, 1, 1, 1))},
]

PALETTE = {
    "c-transitive": "#2a7fff",
    "random":       "#777777",
    "d":            "#dd8452",
    "unsup":        "#55a868",
    "text":         "#c44e52",
}


def _metric_from_row(row: pd.Series, col: str) -> float:
    if col == "__routes_ratio":
        total = float(row.get("routes_total", float("nan")))
        if not np.isfinite(total) or total == 0:
            return float("nan")
        return float(row.get("routes_correct", float("nan"))) / total
    if col == "__precision_10":
        # Instance precision@10 under single-GT-per-query: number of
        # correct partners in top-10 (always 0 or 1) divided by k=10.
        # Equal to R@10 / 10 in expectation across queries.
        r10 = float(row.get("R@10", float("nan")))
        if not np.isfinite(r10):
            return float("nan")
        return r10 / 10.0
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _load_transitive_curve(suffix: str, alpha: float,
                            scope: str) -> pd.DataFrame:
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[(df.scope == scope) & np.isclose(df.alpha, alpha)]
    return df.sort_values("K")


def _load_reference_value(rec: dict, suffix: str, alpha: float,
                          scope: str, metric: str) -> float:
    csv = RES / rec["csv"].format(suffix=suffix)
    if not csv.exists():
        return float("nan")
    df = pd.read_csv(csv)
    sub = df[df.scope == scope]
    if rec.get("needs_alpha"):
        sub = sub[np.isclose(sub.alpha, alpha)]
    elif rec["select"] is not None:
        for k, v in rec["select"].items():
            sub = sub[sub[k] == v]
    if sub.empty:
        return float("nan")
    return _metric_from_row(sub.iloc[0], metric)


def render(out_path: Path, pair_keys: list[str],
           alpha: float, scope: str) -> None:
    n_rows = len(pair_keys)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    any_data = False
    for row_idx, pair_key in enumerate(pair_keys):
        suffix = PAIRS[pair_key]["suffix"]
        pair_label = PAIRS[pair_key]["label"]

        df_tr     = _load_transitive_curve(suffix, alpha, scope)
        df_tr_agg = _load_transitive_curve(suffix, alpha, "aggregate")
        if not df_tr.empty or not df_tr_agg.empty:
            any_data = True

        for col_idx, (col, label, vmin, vmax) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            # Aggregate (in-sample) curve, drawn first so it sits behind
            # the held-out line. Includes anchor rows for Transitive's
            # two legs, so it is the optimistic in-sample reading; the
            # held-out line is the generalisation reading.
            if not df_tr_agg.empty:
                ys_agg = [_metric_from_row(r, col) for _, r in df_tr_agg.iterrows()]
                xs_agg = df_tr_agg["K"].values
                ax.plot(xs_agg, ys_agg, marker="o", markersize=4, lw=1.2,
                        ls="--", color=PALETTE["c-transitive"], alpha=0.4,
                        label="Transitive (in-sample)")

            if not df_tr.empty:
                ys = [_metric_from_row(r, col) for _, r in df_tr.iterrows()]
                xs = df_tr["K"].values
                ax.plot(xs, ys, marker="o", lw=1.8,
                        color=PALETTE["c-transitive"],
                        label="Transitive (held-out)")

            for rec in REFERENCE_RECIPES:
                v = _load_reference_value(rec, suffix, alpha, scope, col)
                if not np.isfinite(v):
                    continue
                ax.axhline(v, color=PALETTE[rec["key"]], linestyle=rec["ls"],
                           lw=1.4, alpha=0.85, label=rec["label"])

            ax.set_xscale("log")
            ax.set_xlabel("$K$ (paired-anchor budget per leg)", fontsize=8)
            ax.set_xticks([10, 20, 50, 100, 200, 300])
            ax.set_xticklabels([str(k) for k in [10, 20, 50, 100, 200, 300]],
                               fontsize=7)
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
        print(f"[transitive-K] no transitive sweep data found at "
              f"alpha={alpha}, scope={scope}; nothing rendered.")
        plt.close(fig)
        return

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
               ncol=min(len(labels), 5),
               fontsize=9, frameon=False)

    fig.suptitle(
        rf"Transitive Transport Bridge K sweep at $\alpha = {alpha}$ "
        rf"(held-out solid, in-sample dashed; "
        rf"scope = {scope.replace('_', ' ')})",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[transitive-K] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Fix alpha for both legs of the transitive sweep "
                         "and for the Caption-Distance FGW reference line. "
                         "Default 0.7.")
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"],
                    help="Scope for the sweep and reference recipes.")
    ap.add_argument("--pair", choices=["canonical", "textfree", "both"],
                    default="both")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path. Default constructs a name from "
                         "alpha and pair under results/exp_grid/plots/.")
    args = ap.parse_args()

    if args.pair == "both":
        keys = list(PAIRS.keys())
        default_name = f"transitive_K_sweep__alpha{args.alpha:.1f}.png"
    else:
        keys = [args.pair]
        default_name = (f"transitive_K_sweep__alpha{args.alpha:.1f}"
                        f"__{args.pair}.png")
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, keys, args.alpha, args.scope)


if __name__ == "__main__":
    main()
