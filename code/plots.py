"""Phase 9 — tables and plots.

For each experiment, reads results/exp_*/sweep*.csv and produces:
  - grid_aggregate.csv and grid_heldout.csv (K × alpha grid of R@10 / routes)
  - sweep.png (R@k panel + structural panel at alpha=0.5)

Also writes comparison.csv and comparison_structure.csv at the top-level
results/ folder.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global thesis-style theme. Affects every matplotlib axes too: line plots
# and the dense pairwise-distance scatter inherit the grid, palette, and
# tick styling without per-call refactoring.
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="colorblind",
    font_scale=0.95,
    rc={
        "figure.dpi":      140,
        "savefig.dpi":     140,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon":  True,
        "legend.framealpha": 0.85,
    },
)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_sweep(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def grid_for(df: pd.DataFrame, scope: str, K_cl: int) -> pd.DataFrame:
    sub = df[df["scope"] == scope]
    if sub.empty:
        return pd.DataFrame()
    grid = sub.pivot_table(
        index="K", columns="alpha",
        values=["R@10", "routes_correct"], aggfunc="mean",
    )
    rows = []
    for K in grid.index:
        row = {"K": K}
        for alpha in sorted(set(sub["alpha"])):
            r10 = grid.loc[K, ("R@10", alpha)]
            rc = grid.loc[K, ("routes_correct", alpha)]
            if pd.isna(r10) or pd.isna(rc):
                row[f"alpha={alpha:.1f}"] = f"nan / 0/{K_cl}"
            else:
                row[f"alpha={alpha:.1f}"] = f"{r10:.3f} / {int(rc)}/{K_cl}"
        rows.append(row)
    return pd.DataFrame(rows)


def find_elbow(K_vals: list[int], r10: list[float]) -> int:
    """First K such that marginal gain on R@10 falls below 0.02, else max-R@10 K."""
    if not K_vals:
        return -1
    pairs = sorted(zip(K_vals, r10), key=lambda x: x[0])
    for i in range(1, len(pairs)):
        if pairs[i][1] - pairs[i - 1][1] < 0.02:
            return pairs[i - 1][0]
    return max(pairs, key=lambda x: x[1])[0]


def plot_sweep(df: pd.DataFrame, K_cl: int, out: Path, alpha_focus: float = 0.5) -> None:
    sub = df[(df["scope"] == "aggregate") & (df["alpha"] == alpha_focus)].sort_values("K")
    if sub.empty:
        # fall back to whatever alpha is present
        sub = df[df["scope"] == "aggregate"].sort_values(["alpha", "K"])
        if sub.empty:
            print(f"[plot] no data for {out}")
            return
        alpha_focus = sub["alpha"].iloc[0]
        sub = sub[sub["alpha"] == alpha_focus]

    K = sub["K"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k_metric, marker in zip(["R@1", "R@5", "R@10", "R@20"], ["o", "s", "^", "D"]):
        axes[0].plot(K, sub[k_metric], marker=marker, label=k_metric)
    elbow = find_elbow(K, sub["R@10"].tolist())
    if elbow > 0:
        axes[0].axvline(elbow, linestyle="--", alpha=0.5, color="grey")
        axes[0].text(elbow, 0.02, f" elbow K={elbow}", color="grey")
    axes[0].set_xlabel("K (number of paired anchors)")
    axes[0].set_ylabel(f"Recall@k  (alpha={alpha_focus})")
    axes[0].set_title("Retrieval")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(K, sub["nmi"], marker="o", label="cluster NMI")
    route_acc = sub["routes_correct"] / sub["routes_total"]
    axes[1].plot(K, route_acc, marker="s", label="route acc")
    axes[1].plot(K, sub["pearson_r"], marker="^", label="Pearson r")
    if elbow > 0:
        axes[1].axvline(elbow, linestyle="--", alpha=0.5, color="grey")
    axes[1].set_xlabel("K (number of paired anchors)")
    axes[1].set_ylabel(f"structural metric (alpha={alpha_focus})")
    axes[1].set_title("Structural")
    axes[1].legend()

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


KCL_MAP = {
    "exp_a": 10, "exp_b": 20, "exp_c": 15,
    "exp_d": 15, "exp_unsup": 15, "exp_text": 15,
}


def plot_alpha_sweep(df: pd.DataFrame, K_cl: int, out: Path,
                     scopes: tuple[str, ...] = ("aggregate", "heldout"),
                     K_focus: int | None = 300,
                     title: str | None = None) -> None:
    """Two-panel figure: evaluation metrics vs α (FGW blend).

    Left panel  : R@1, R@5, R@10, R@20 vs α.
    Right panel : cluster NMI, route accuracy, Pearson r, kNN overlap vs α.

    Multiple scopes are overlaid in the same panels using line style:
    aggregate -> dashed, held-out (or heldout) -> solid.
    Colour encodes metric, marker also encodes metric for distinctness.

    `K_focus` filters rows by K when supplied; pass None to plot every row
    in `df` (used for Experiment D, which has no K dimension).
    """
    base = df if K_focus is None else df[df["K"] == K_focus]
    if base.empty:
        print(f"[plot] alpha-sweep skip {out.name}: no rows "
              f"(K_focus={K_focus})")
        return

    style_for = {
        "aggregate":      dict(linestyle="--", alpha=0.6),
        "heldout":        dict(linestyle="-",  alpha=1.0),
        "heldout": dict(linestyle="-",  alpha=1.0),
    }
    ret_metrics = [("R@1", "C0", "o"), ("R@5", "C1", "s"),
                   ("R@10", "C2", "^"), ("R@20", "C3", "D")]
    str_metrics = [("nmi",         "C0", "o", "cluster NMI"),
                   ("__route",     "C1", "s", "route acc"),
                   ("pearson_r",   "C2", "^", "Pearson r"),
                   ("knn_overlap", "C3", "D", "kNN overlap")]
    if "ami" in df.columns:
        str_metrics.append(("ami", "C4", "v", "cluster AMI"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    drew_any = False
    for scope in scopes:
        sub = base[base["scope"] == scope].sort_values("alpha")
        if sub.empty:
            continue
        drew_any = True
        alpha_vals = sub["alpha"].tolist()
        sty = style_for.get(scope, dict(linestyle=":", alpha=0.8))

        for m, col, mk in ret_metrics:
            axes[0].plot(alpha_vals, sub[m], marker=mk, color=col,
                         linestyle=sty["linestyle"], alpha=sty["alpha"],
                         label=f"{m} ({scope})")
        route_acc = sub["routes_correct"] / sub["routes_total"]
        for m, col, mk, lbl in str_metrics:
            y = route_acc if m == "__route" else sub[m]
            axes[1].plot(alpha_vals, y, marker=mk, color=col,
                         linestyle=sty["linestyle"], alpha=sty["alpha"],
                         label=f"{lbl} ({scope})")

    if not drew_any:
        plt.close(fig)
        print(f"[plot] alpha-sweep skip {out.name}: no rows in any scope")
        return

    k_suffix = f"K={K_focus}" if K_focus is not None else "no K axis"
    axes[0].set_xlabel("α (FGW blend; 0 = pure Sinkhorn, 1 = pure GW)")
    axes[0].set_ylabel(f"Recall@k  ({k_suffix})")
    axes[0].set_title("Retrieval vs α")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("α (FGW blend; 0 = pure Sinkhorn, 1 = pure GW)")
    axes[1].set_ylabel(f"structural metric  ({k_suffix})")
    axes[1].set_title("Structural vs α")
    axes[1].legend(fontsize=7, ncol=2)

    for ax in axes:
        sns.despine(ax=ax)
    fig.suptitle(title or f"{out.parent.name} — α sweep ({k_suffix})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def emit_experiment_artifacts(exp_dir: Path, sweep_name: str) -> None:
    name = exp_dir.name
    K_cl = KCL_MAP.get(name, 10)
    df = load_sweep(exp_dir / sweep_name)
    if df.empty:
        print(f"[skip] {exp_dir} {sweep_name}")
        return

    grid_a = grid_for(df, "aggregate", K_cl)
    grid_h = grid_for(df, "heldout", K_cl)
    if not grid_a.empty:
        grid_a.to_csv(exp_dir / "grid_aggregate.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_aggregate.csv'}")
    if not grid_h.empty:
        grid_h.to_csv(exp_dir / "grid_heldout.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_heldout.csv'}")

    plot_sweep(df, K_cl, exp_dir / "sweep.png", alpha_focus=0.5)
    # Orthogonal cut: α-sweep at the canonical operating point, with
    # aggregate (dashed) and held-out (solid) overlaid so the
    # memorisation gap is visible at every α.
    plot_alpha_sweep(df, K_cl=K_cl,
                     out=exp_dir / "alpha_sweep.png",
                     scopes=("aggregate", "heldout"),
                     K_focus=300,
                     title=f"{name} — α sweep at K=300 "
                           "(aggregate dashed, held-out solid)")


def emit_exp_d_artifacts(exp_dir: Path) -> None:
    """Experiment D has no K sweep, only an alpha sweep + a same-rows row.

    Emits a single grid_aggregate.csv with one row per alpha (R@10 and
    routes_correct), plus grid_heldout.csv with the one same-rows-view
    row tagged 'heldout'. No sweep.png (no K axis to plot).
    """
    name = exp_dir.name
    K_cl = KCL_MAP.get(name, 15)
    df = load_sweep(exp_dir / "sweep.csv")
    if df.empty:
        print(f"[skip] {exp_dir} sweep.csv")
        return

    agg = df[df["scope"] == "aggregate"].sort_values("alpha")
    if not agg.empty:
        rows = []
        for _, r in agg.iterrows():
            rows.append({
                "alpha": float(r["alpha"]),
                "cell": f"{r['R@10']:.3f} / {int(r['routes_correct'])}/{K_cl}",
            })
        pd.DataFrame(rows).to_csv(exp_dir / "grid_aggregate.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_aggregate.csv'}")

    cmp = df[df["scope"] == "heldout"]
    if not cmp.empty:
        r = cmp.iloc[0]
        pd.DataFrame([{
            "alpha": float(r["alpha"]),
            "cell": f"{r['R@10']:.3f} / {int(r['routes_correct'])}/{K_cl}",
        }]).to_csv(exp_dir / "grid_heldout.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_heldout.csv'} (heldout)")

    # α-sweep figure: only for FGW experiments where α actually varies.
    # Experiment D sweeps α in {0, 0.3, 0.5, 0.7, 0.9}; same-rows view
    # has a single α=0.7 point that gets drawn as a marker on top.
    # Pure-GW (α≡1) and Text-only (α=NaN) are not FGW — skip.
    n_alpha = df[df["scope"] == "aggregate"]["alpha"].nunique()
    if n_alpha >= 2:
        plot_alpha_sweep(df, K_cl=K_cl,
                         out=exp_dir / "alpha_sweep.png",
                         scopes=("aggregate", "heldout"),
                         K_focus=None,
                         title=f"{name} — α sweep "
                               "(aggregate dashed; same-rows solid marker at α=0.7)")


def emit_comparison() -> None:
    """Two comparison tables (retrieval + structure), method × experiment grid."""
    rows_ret: list[dict] = []
    rows_str: list[dict] = []

    def pick(df: pd.DataFrame, scope: str, K: int, alpha: float):
        sub = df[(df["scope"] == scope) & (df["K"] == K) & (df["alpha"] == alpha)]
        if sub.empty:
            return None
        return sub.iloc[0]

    a = load_sweep(RES / "exp_a" / "sweep.csv")
    b = load_sweep(RES / "exp_b" / "sweep.csv")
    # C-direct is no longer in the suite; emit_comparison leaves the
    # corresponding cells empty (rendered as a dash) rather than crashing.
    c_dir = pd.DataFrame()
    c_tr = load_sweep(RES / "exp_c" / "sweep_transitive.csv")
    d = load_sweep(RES / "exp_d" / "sweep.csv")
    unsup = load_sweep(RES / "exp_unsup" / "sweep.csv")
    text = load_sweep(RES / "exp_text" / "sweep.csv")

    def elbow_K(df: pd.DataFrame, alpha: float = 0.5) -> int:
        sub = df[(df["scope"] == "aggregate") & (df["alpha"] == alpha)].sort_values("K")
        if sub.empty:
            return -1
        return find_elbow(sub["K"].tolist(), sub["R@10"].tolist())

    def cell_ret(df: pd.DataFrame, scope: str, K: int, alpha: float, K_cl: int):
        r = pick(df, scope, K, alpha)
        if r is None:
            return "—"
        return f"{r['R@10']:.3f} / {int(r['routes_correct'])}/{K_cl}"

    def cell_str(df: pd.DataFrame, scope: str, K: int, alpha: float):
        r = pick(df, scope, K, alpha)
        if r is None:
            return "—"
        ami = r["ami"] if "ami" in r.index and pd.notna(r["ami"]) else float("nan")
        ami_str = f" ami={ami:.3f}" if not pd.isna(ami) else ""
        return (f"NMI={r['nmi']:.3f}{ami_str} "
                f"knn={r['knn_overlap']:.3f} r={r['pearson_r']:.3f}")

    K_target = 300  # reusable plan K
    alpha = 0.5

    sources = [
        ("exp_a", a, 10),
        ("exp_b", b, 20),
        ("exp_c_direct", c_dir, 15),
    ]

    # Per-method rows: "elbow" and "K=300"
    for method, K_pick in [("elbow@a=0.5", None), ("K=300@a=0.5", K_target)]:
        ret_row = {"method": method}
        str_row = {"method": method}
        for name, df, K_cl in sources:
            K_use = elbow_K(df, alpha) if K_pick is None else K_target
            ret_row[name + "_agg"] = cell_ret(df, "aggregate", K_use, alpha, K_cl)
            ret_row[name + "_hel"] = cell_ret(df, "heldout", K_use, alpha, K_cl)
            str_row[name + "_agg"] = cell_str(df, "aggregate", K_use, alpha)
            str_row[name + "_hel"] = cell_str(df, "heldout", K_use, alpha)
        rows_ret.append(ret_row)
        rows_str.append(str_row)

    # Add C-transitive row
    if not c_tr.empty:
        agg = c_tr[c_tr["scope"] == "aggregate"]
        hel = c_tr[c_tr["scope"] == "heldout"]
        if not agg.empty:
            r = agg.iloc[0]
            ret_row = {"method": "C-transitive"}
            ret_row["exp_a_agg"] = "—"
            ret_row["exp_a_hel"] = "—"
            ret_row["exp_b_agg"] = "—"
            ret_row["exp_b_hel"] = "—"
            ret_row["exp_c_direct_agg"] = f"{r['R@10']:.3f} / {int(r['routes_correct'])}/15 (transitive)"
            ret_row["exp_c_direct_hel"] = (
                f"{hel.iloc[0]['R@10']:.3f} / {int(hel.iloc[0]['routes_correct'])}/15"
                if not hel.empty else "—"
            )
            rows_ret.append(ret_row)
            str_row = {"method": "C-transitive"}
            str_row["exp_a_agg"] = "—"
            str_row["exp_a_hel"] = "—"
            str_row["exp_b_agg"] = "—"
            str_row["exp_b_hel"] = "—"
            str_row["exp_c_direct_agg"] = (
                f"NMI={r['nmi']:.3f}"
                + (f" ami={r['ami']:.3f}" if 'ami' in r.index and pd.notna(r.get('ami', float('nan'))) else "")
                + f" knn={r['knn_overlap']:.3f} r={r['pearson_r']:.3f}"
            )
            if not hel.empty:
                rh = hel.iloc[0]
                str_row["exp_c_direct_hel"] = (
                    f"NMI={rh['nmi']:.3f}"
                    + (f" ami={rh['ami']:.3f}"
                       if 'ami' in rh.index and pd.notna(rh.get('ami', float('nan')))
                       else "")
                    + f" knn={rh['knn_overlap']:.3f} r={rh['pearson_r']:.3f}"
                )
            else:
                str_row["exp_c_direct_hel"] = "—"
            rows_str.append(str_row)

    # Each "extra" image-audio variant (D, Unsup, Text) is a single-cell
    # experiment with two scopes: 'aggregate' and 'heldout'.
    # We pluck both, format retrieval and structural strings.
    def _extra_cells(df: pd.DataFrame, alpha_filter):
        """Return {ret_agg, ret_hel, str_agg, str_hel} for a single-cell exp.

        `alpha_filter` is a predicate over rows of `df` selecting the
        canonical α for that experiment (e.g. lambda r: r.alpha == 0.7).
        Pass `None` to ignore α entirely (text-only baseline has α = NaN).
        """
        def pluck(scope: str, fmt):
            if df.empty:
                return "—"
            sub = df[df["scope"] == scope]
            if alpha_filter is not None:
                sub = sub[sub["alpha"].apply(alpha_filter)]
            if sub.empty:
                return "—"
            return fmt(sub.iloc[0])

        ret = lambda r: f"{r['R@10']:.3f} / {int(r['routes_correct'])}/15"
        ret_same = lambda r: f"{r['R@10']:.3f} / {int(r['routes_correct'])}/15 (same-rows)"
        def _ami_part(r):
            if 'ami' in r.index and pd.notna(r.get('ami', float('nan'))):
                return f" ami={r['ami']:.3f}"
            return ""
        st = lambda r: (f"NMI={r['nmi']:.3f}{_ami_part(r)} "
                        f"knn={r['knn_overlap']:.3f} r={r['pearson_r']:.3f}")
        return {
            "ret_agg": pluck("aggregate", ret),
            "ret_hel": pluck("heldout", ret_same),
            "str_agg": pluck("aggregate", st),
            "str_hel": pluck("heldout", st),
        }

    d_cells = _extra_cells(d, lambda a: np.isclose(a, 0.7))
    unsup_cells = _extra_cells(unsup, lambda a: np.isclose(a, 1.0))
    text_cells = _extra_cells(text, None)  # text baseline has alpha=NaN

    # Inject the extra-experiment cells into every existing row, so the
    # comparison row at "K=300@α=0.5" can be read horizontally across the
    # full set of image-audio variants. The extra experiments are
    # operating-point-free; the same single number appears in every row.
    for row in rows_ret:
        row["exp_d_agg"]     = d_cells["ret_agg"]
        row["exp_d_hel"]     = d_cells["ret_hel"]
        row["exp_unsup_agg"] = unsup_cells["ret_agg"]
        row["exp_unsup_hel"] = unsup_cells["ret_hel"]
        row["exp_text_agg"]  = text_cells["ret_agg"]
        row["exp_text_hel"]  = text_cells["ret_hel"]
    for row in rows_str:
        row["exp_d_agg"]     = d_cells["str_agg"]
        row["exp_d_hel"]     = d_cells["str_hel"]
        row["exp_unsup_agg"] = unsup_cells["str_agg"]
        row["exp_unsup_hel"] = unsup_cells["str_hel"]
        row["exp_text_agg"]  = text_cells["str_agg"]
        row["exp_text_hel"]  = text_cells["str_hel"]

    # Dedicated rows for the three image-audio variants that don't fit the
    # ridge sweep narrative — D, Unsup, Text — so each can be read on its
    # own line with the supervised columns blanked out.
    def _empty_supervised():
        return {
            "exp_a_agg": "—", "exp_a_hel": "—",
            "exp_b_agg": "—", "exp_b_hel": "—",
            "exp_c_direct_agg": "—", "exp_c_direct_hel": "—",
        }

    if not d.empty:
        rows_ret.append({"method": "D-caption-cost", **_empty_supervised(),
                         "exp_d_agg": d_cells["ret_agg"], "exp_d_hel": d_cells["ret_hel"],
                         "exp_unsup_agg": "—", "exp_unsup_hel": "—",
                         "exp_text_agg": "—",  "exp_text_hel": "—"})
        rows_str.append({"method": "D-caption-cost", **_empty_supervised(),
                         "exp_d_agg": d_cells["str_agg"], "exp_d_hel": d_cells["str_hel"],
                         "exp_unsup_agg": "—", "exp_unsup_hel": "—",
                         "exp_text_agg": "—",  "exp_text_hel": "—"})
    if not unsup.empty:
        rows_ret.append({"method": "Unsup-pure-GW", **_empty_supervised(),
                         "exp_d_agg": "—", "exp_d_hel": "—",
                         "exp_unsup_agg": unsup_cells["ret_agg"], "exp_unsup_hel": unsup_cells["ret_hel"],
                         "exp_text_agg": "—",  "exp_text_hel": "—"})
        rows_str.append({"method": "Unsup-pure-GW", **_empty_supervised(),
                         "exp_d_agg": "—", "exp_d_hel": "—",
                         "exp_unsup_agg": unsup_cells["str_agg"], "exp_unsup_hel": unsup_cells["str_hel"],
                         "exp_text_agg": "—",  "exp_text_hel": "—"})
    if not text.empty:
        rows_ret.append({"method": "Text-only", **_empty_supervised(),
                         "exp_d_agg": "—", "exp_d_hel": "—",
                         "exp_unsup_agg": "—", "exp_unsup_hel": "—",
                         "exp_text_agg": text_cells["ret_agg"], "exp_text_hel": text_cells["ret_hel"]})
        rows_str.append({"method": "Text-only", **_empty_supervised(),
                         "exp_d_agg": "—", "exp_d_hel": "—",
                         "exp_unsup_agg": "—", "exp_unsup_hel": "—",
                         "exp_text_agg": text_cells["str_agg"], "exp_text_hel": text_cells["str_hel"]})

    cols = ["method",
            "exp_a_agg", "exp_a_hel",
            "exp_b_agg", "exp_b_hel",
            "exp_c_direct_agg", "exp_c_direct_hel",
            "exp_d_agg", "exp_d_hel",
            "exp_unsup_agg", "exp_unsup_hel",
            "exp_text_agg", "exp_text_hel"]
    with (RES / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_ret:
            w.writerow({c: r.get(c, "—") for c in cols})
    with (RES / "comparison_structure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_str:
            w.writerow({c: r.get(c, "—") for c in cols})
    print(f"[comparison] wrote {RES / 'comparison.csv'}")
    print(f"[comparison] wrote {RES / 'comparison_structure.csv'}")


# ---------------------------------------------------------------------------
# Encoder-ablation grid (Phase 8g): heatmaps and bar charts.
# ---------------------------------------------------------------------------
def _heatmap(pivot: pd.DataFrame, title: str, out: Path,
             cmap: str = "viridis", vmin=None, vmax=None,
             fmt: str = ".3f", cbar_label: str | None = None) -> None:
    """Annotated heatmap of `pivot` (rows × cols), rendered with seaborn."""
    if pivot.empty:
        print(f"[heatmap] skip {out.name}: empty pivot")
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pivot.columns) + 3),
                                    max(4, 0.6 * len(pivot.index) + 2)))
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        annot=True, fmt=fmt, annot_kws={"fontsize": 8},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": cbar_label or "value", "shrink": 0.85},
        square=False,
    )
    ax.set_xlabel(pivot.columns.name or "")
    ax.set_ylabel(pivot.index.name or "")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] wrote {out}")


def _bar(pivot: pd.DataFrame, title: str, out: Path,
         ylabel: str = "value") -> None:
    """Grouped bar chart: rows index = encoder, cols = scope (aggregate/heldout).

    Reshapes the pivot to long form and renders via `sns.barplot` so colour,
    grouping, and legend styling come from the global seaborn theme.
    """
    if pivot.empty:
        print(f"[bar] skip {out.name}: empty pivot")
        return
    idx_name = pivot.index.name or "encoder"
    col_name = pivot.columns.name or "scope"
    long = (
        pivot.reset_index()
             .melt(id_vars=idx_name, var_name=col_name, value_name="value")
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(pivot.index) + 2), 4.5))
    sns.barplot(
        data=long, x=idx_name, y="value", hue=col_name,
        ax=ax, palette="colorblind", edgecolor="white",
    )
    # Annotate bar values for quick reading.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title=col_name, loc="best", frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[bar] wrote {out}")


def emit_grid_plots(grid_dir: Path) -> None:
    """Heatmaps + bar charts from results/exp_grid/sweep.csv."""
    csv_path = grid_dir / "sweep.csv"
    df = load_sweep(csv_path)
    if df.empty:
        print(f"[grid] skip: {csv_path} missing or empty")
        return

    plots_dir = grid_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cross-modal heatmaps (vision × audio) ---------------------------
    def heat(exp_name: str, scope: str, metric: str,
             out_name: str, title: str,
             cmap: str = "viridis", vmin=None, vmax=None,
             fmt: str = ".3f", cbar_label: str | None = None) -> pd.DataFrame:
        sub = df[(df["experiment"] == exp_name) & (df["scope"] == scope)]
        if sub.empty:
            print(f"[grid] no rows for {exp_name}/{scope}")
            return pd.DataFrame()
        pv = sub.pivot(index="image_encoder", columns="audio_encoder",
                       values=metric).sort_index().sort_index(axis=1)
        _heatmap(pv, title, plots_dir / out_name,
                 cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt,
                 cbar_label=cbar_label or metric)
        return pv

    has_ami = "ami" in df.columns

    # C-direct removed from the suite -- no heatmaps for it.

    # C-transitive (identity bridge): heatmaps mirror D.
    heat("c-transitive", "heldout", "R@10",
         "ctrans_R10_heldout.png",
         "Caption-index C-transitive -- R@10 (held-out)",
         cmap="viridis", vmin=0.0, vmax=0.6)
    if has_ami:
        heat("c-transitive", "heldout", "ami",
             "ctrans_AMI_heldout.png",
             "Caption-index C-transitive -- AMI (held-out)",
             cmap="magma", vmin=-0.05, vmax=0.6)
    heat("c-transitive", "aggregate", "R@10",
         "ctrans_R10_aggregate.png",
         "Caption-index C-transitive -- R@10 (aggregate)",
         cmap="viridis", vmin=0.0, vmax=1.0)

    # Random baseline: useful for visual reference of the chance floor.
    heat("random", "heldout", "R@10",
         "random_R10_heldout.png",
         "Random baseline -- R@10 (held-out, chance floor)",
         cmap="viridis", vmin=0.0, vmax=0.05)

    # D -- caption-cost FGW
    heat("d", "aggregate", "R@10",
         "d_R10_aggregate.png",
         "Experiment D — R@10 (aggregate, α=0.7)",
         cmap="viridis", vmin=0.0, vmax=0.6)
    heat("d", "heldout", "R@10",
         "d_R10_sameRows.png",
         "Experiment D — R@10 (same-rows view)",
         cmap="viridis", vmin=0.0, vmax=0.6)
    heat("d", "heldout", "nmi",
         "d_NMI_sameRows.png",
         "Experiment D — cluster NMI (same-rows, uncorrected)",
         cmap="magma", vmin=0.0, vmax=1.0)
    if has_ami:
        heat("d", "heldout", "ami",
             "d_AMI_sameRows.png",
             "Experiment D — cluster AMI (same-rows, chance-corrected)",
             cmap="magma", vmin=-0.05, vmax=0.6)

    # Unsup — pure GW
    heat("unsup", "aggregate", "R@10",
         "unsup_R10_aggregate.png",
         "Experiment Unsup — R@10 (aggregate, pure GW)",
         cmap="viridis", vmin=0.0, vmax=0.15)
    heat("unsup", "heldout", "nmi",
         "unsup_NMI_sameRows.png",
         "Experiment Unsup — cluster NMI (same-rows, uncorrected)",
         cmap="magma", vmin=0.0, vmax=1.0)
    if has_ami:
        heat("unsup", "heldout", "ami",
             "unsup_AMI_sameRows.png",
             "Experiment Unsup — cluster AMI (same-rows, chance-corrected)",
             cmap="magma", vmin=-0.05, vmax=0.6)

    # Text-only baseline
    heat("text", "aggregate", "R@10",
         "text_R10_aggregate.png",
         "Text-only baseline — R@10 (aggregate)",
         cmap="viridis", vmin=0.0, vmax=0.6)
    if has_ami:
        heat("text", "heldout", "ami",
             "text_AMI_sameRows.png",
             "Text-only baseline — cluster AMI (same-rows, chance-corrected)",
             cmap="magma", vmin=-0.05, vmax=0.6)

    # Delta R@10: C-transitive (held-out_like_c) - Random (held-out_like_c)
    # The "above-chance retrieval signal" carried by the caption-index
    # composition over the no-information floor.
    sub_c = df[(df["experiment"] == "c-transitive") & (df["scope"] == "heldout")]
    sub_r = df[(df["experiment"] == "random") & (df["scope"] == "heldout")]
    if not sub_c.empty and not sub_r.empty:
        pv_c = sub_c.pivot(index="image_encoder",
                           columns="audio_encoder", values="R@10")
        pv_r = sub_r.pivot(index="image_encoder",
                           columns="audio_encoder", values="R@10")
        pv_diff = (pv_c - pv_r).sort_index().sort_index(axis=1)
        _heatmap(pv_diff,
                 "Delta R@10 = C-transitive (held-out) minus Random (held-out)",
                 plots_dir / "delta_ctrans_minus_random_R10.png",
                 cmap="RdBu_r", vmin=-0.2, vmax=0.2, fmt="+.3f",
                 cbar_label="Delta R@10")

    # ΔR@10: D (same-rows) − Text-only (same-rows)  (how much FGW adds over raw captions)
    sub_t = df[(df["experiment"] == "text") & (df["scope"] == "heldout")]
    if not sub_d.empty and not sub_t.empty:
        pv_d_sr = sub_d.pivot(index="image_encoder",
                              columns="audio_encoder", values="R@10")
        pv_t_sr = sub_t.pivot(index="image_encoder",
                              columns="audio_encoder", values="R@10")
        pv_diff = (pv_d_sr - pv_t_sr).sort_index().sort_index(axis=1)
        _heatmap(pv_diff,
                 "ΔR@10 = D − Text-only (same-rows) — paid for by FGW + intra-modal C1,C2",
                 plots_dir / "delta_d_minus_text_R10.png",
                 cmap="RdBu_r", vmin=-0.1, vmax=0.1, fmt="+.3f",
                 cbar_label="ΔR@10")

    # ---- Unimodal bar charts ----------------------------------------------
    a = df[df["experiment"] == "a"]
    if not a.empty:
        pv = a.pivot(index="image_encoder", columns="scope", values="R@10")
        pv = pv.reindex(columns=["aggregate", "heldout"])
        _bar(pv, "Experiment A — image → visual-caption R@10 (K=300, α=0.5)",
             plots_dir / "a_R10_by_image_encoder.png", ylabel="R@10")
        pv_nmi = a.pivot(index="image_encoder", columns="scope", values="nmi")
        pv_nmi = pv_nmi.reindex(columns=["aggregate", "heldout"])
        _bar(pv_nmi, "Experiment A — cluster NMI by image encoder (uncorrected)",
             plots_dir / "a_NMI_by_image_encoder.png", ylabel="NMI")
        if has_ami:
            pv_ami = a.pivot(index="image_encoder", columns="scope", values="ami")
            pv_ami = pv_ami.reindex(columns=["aggregate", "heldout"])
            _bar(pv_ami,
                 "Experiment A — cluster AMI by image encoder (chance-corrected)",
                 plots_dir / "a_AMI_by_image_encoder.png", ylabel="AMI")

    b = df[df["experiment"] == "b"]
    if not b.empty:
        pv = b.pivot(index="audio_encoder", columns="scope", values="R@10")
        pv = pv.reindex(columns=["aggregate", "heldout"])
        _bar(pv, "Experiment B — audio → audio-caption R@10 (K=300, α=0.5)",
             plots_dir / "b_R10_by_audio_encoder.png", ylabel="R@10")
        pv_nmi = b.pivot(index="audio_encoder", columns="scope", values="nmi")
        pv_nmi = pv_nmi.reindex(columns=["aggregate", "heldout"])
        _bar(pv_nmi, "Experiment B — cluster NMI by audio encoder (uncorrected)",
             plots_dir / "b_NMI_by_audio_encoder.png", ylabel="NMI")
        if has_ami:
            pv_ami = b.pivot(index="audio_encoder", columns="scope", values="ami")
            pv_ami = pv_ami.reindex(columns=["aggregate", "heldout"])
            _bar(pv_ami,
                 "Experiment B — cluster AMI by audio encoder (chance-corrected)",
                 plots_dir / "b_AMI_by_audio_encoder.png", ylabel="AMI")

    # ---- Long-format tidy tables for downstream reuse ---------------------
    df.to_csv(grid_dir / "tidy.csv", index=False)
    print(f"[grid] wrote {grid_dir / 'tidy.csv'}")


# ---------------------------------------------------------------------------
# Cluster confusion matrices (cross-modal class-level alignment visual).
# ---------------------------------------------------------------------------
EMB = ROOT / "embeddings"

# Cross-modal image -> audio plans for the side-by-side confusion-matrix
# figure. Each entry: display_label -> path to a saved (n, n) plan that
# routes the canonical CLIP-L/14 rows (source) to CLAP-unfused rows (target).
DEFAULT_CONFUSION_PLANS: list[tuple[str, Path]] = [
    ("D — caption-cost FGW",       RES / "exp_d"     / "T_caption.npy"),
    ("C-transitive — text bridge", RES / "exp_c"     / "T_transitive.npy"),
    ("Pure-GW",                    RES / "exp_unsup" / "T_gw.npy"),
    ("Text-only — caption cosine", RES / "exp_text"  / "T_text.npy"),
]


def _hungarian_column_perm(C: np.ndarray) -> np.ndarray:
    """Permutation of target columns that maximises the trace of C.

    Returns an index array `col_ind` such that `C[:, col_ind]` is the
    column-permuted confusion matrix with maximum diagonal mass. Falls
    back to the identity if scipy is missing or the matrix is degenerate.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return np.arange(C.shape[1])
    row_ind, col_ind = linear_sum_assignment(-C)
    perm = np.arange(C.shape[1])
    perm[row_ind] = col_ind
    return perm


def _draw_confusion(
    ax,
    C: np.ndarray,
    title: str,
    annotate: bool = True,
    vmax: float | None = None,
):
    """Annotated heatmap of a single confusion matrix, rendered with seaborn.

    Returns the `QuadMesh` so the caller can attach a single shared
    colourbar across multiple subplots.
    """
    K = C.shape[0]
    do_annot = annotate and K <= 20
    sns.heatmap(
        C, ax=ax, cmap="magma",
        vmin=0.0, vmax=vmax if vmax is not None else max(0.2, float(C.max())),
        annot=do_annot, fmt=".2f", annot_kws={"fontsize": 6},
        linewidths=0.3, linecolor="white", cbar=False, square=True,
    )
    ax.set_xlabel("target K-means cluster (Hungarian-permuted)")
    ax.set_ylabel("source K-means cluster")
    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax.collections[0] if ax.collections else None


def emit_cluster_confusions(
    image_encoder: str = "clip-large",
    audio_encoder: str = "clap-unfused",
    K_cl: int = 15,
    mode: str = "soft",
    plans: list[tuple[str, Path]] | None = None,
    out_dir: Path | None = None,
) -> None:
    """Side-by-side confusion-matrix figure for the four cross-modal plans.

    For each saved plan T (n × n), computes the source-cluster ×
    target-cluster confusion matrix induced by T (`metrics.cluster_confusion`),
    Hungarian-permutes the target columns so the diagonal is dominant,
    and renders all plans in a single figure for direct visual comparison
    of cluster-level alignment quality.

    Also writes per-plan CSVs with the (unpermuted) confusion matrix and
    the column permutation, so the figure can be reconstructed downstream.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import cluster_confusion, cluster_agreement

    plans = plans if plans is not None else DEFAULT_CONFUSION_PLANS
    out_dir = out_dir if out_dir is not None else (RES / "exp_grid" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = EMB / f"vision_{image_encoder}.npy"
    Y_path = EMB / f"audio_{audio_encoder}.npy"
    if not X_path.exists() or not Y_path.exists():
        print(f"[confusion] skip: missing embeddings ({X_path.name} / "
              f"{Y_path.name}). Run on HPC with embeddings/ populated.")
        return
    X = np.load(X_path)
    Y = np.load(Y_path)
    print(f"[confusion] X={X.shape} (vision_{image_encoder})  "
          f"Y={Y.shape} (audio_{audio_encoder})  K_cl={K_cl}  mode={mode}")

    # Load all plans up-front and skip silently on missing files.
    loaded: list[tuple[str, np.ndarray, Path]] = []
    for label, plan_path in plans:
        if not plan_path.exists():
            print(f"[confusion] skip {label}: plan not found at {plan_path}")
            continue
        T = np.load(plan_path)
        if T.shape != (X.shape[0], Y.shape[0]):
            print(f"[confusion] skip {label}: shape {T.shape} != "
                  f"({X.shape[0]}, {Y.shape[0]})")
            continue
        loaded.append((label, T, plan_path))
    if not loaded:
        print("[confusion] no plans loaded; nothing to render.")
        return

    # First pass: compute all (permuted) confusion matrices so a global
    # vmax can be used for fair colour-scale comparison across subplots.
    # Also persists per-plan CSVs (raw confusion + Hungarian permutation)
    # next to the figure so the matrices are inspectable without re-running.
    computed: list[tuple[str, np.ndarray, dict, Path]] = []
    global_vmax = 0.0
    for label, T, plan_path in loaded:
        C, _, _ = cluster_confusion(T, X, Y, K_cl, mode=mode)
        perm = _hungarian_column_perm(C)
        C_perm = C[:, perm]
        agree = cluster_agreement(T, X, Y, K_cl)
        computed.append((label, C_perm, agree, plan_path))
        global_vmax = max(global_vmax, float(C_perm.max()))

        df_C = pd.DataFrame(
            C,
            index=[f"src_{s}" for s in range(K_cl)],
            columns=[f"tgt_{t}" for t in range(K_cl)],
        )
        df_C.to_csv(
            out_dir / f"confusion_{plan_path.parent.name}_{plan_path.stem}.csv"
        )
        pd.DataFrame({"target_column_index": perm}).to_csv(
            out_dir / f"confusion_{plan_path.parent.name}_"
                      f"{plan_path.stem}_perm.csv",
            index_label="diag_row",
        )
    global_vmax = max(0.2, global_vmax)

    n_cells = len(computed)
    cols = min(n_cells, 4)
    rows = int(np.ceil(n_cells / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.4 * cols, 4.6 * rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    last_im = None
    for k, (label, C_perm, agree, plan_path) in enumerate(computed):
        ax = axes_flat[k]
        sub_title = (
            f"{label}\n"
            f"AMI={agree['ami']:.3f}  ARI={agree['ari']:.3f}  "
            f"V={agree['v_measure']:.3f}\n"
            f"hom={agree['homogeneity']:.3f}  comp={agree['completeness']:.3f}"
        )
        last_im = _draw_confusion(ax, C_perm, sub_title, vmax=global_vmax)

    # Blank out any unused subplots.
    for k in range(n_cells, len(axes_flat)):
        axes_flat[k].axis("off")

    if last_im is not None:
        fig.colorbar(
            last_im, ax=axes.tolist() if rows > 1 else axes_flat.tolist(),
            shrink=0.85, label="row-normalised mass",
        )
    fig.suptitle(
        f"Cross-modal cluster confusion  "
        f"({image_encoder} → {audio_encoder},  K={K_cl},  mode={mode},  "
        f"Hungarian-permuted target columns)",
        fontsize=12, y=1.02,
    )
    out_path = out_dir / f"cross_modal_cluster_confusions__K{K_cl}_{mode}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[confusion] wrote {out_path}")


def emit_pearson_scatter(
    image_encoder: str = "clip-large",
    audio_encoder: str = "clap-unfused",
    plans: list[tuple[str, Path]] | None = None,
    out_dir: Path | None = None,
    sample_pairs: int = 40000,
    seed: int = 0,
) -> None:
    """Per-method pairwise-distance scatter for the cross-modal plans.

    For each saved plan T, compute the partner mapping
    ``j*(i) = argmax T[i, :]`` and plot, with one point per unordered
    source pair (i, i'):

      x-axis = ‖X_i − X_{i'}‖           (Euclidean dist. in source space)
      y-axis = ‖Y_{j*(i)} − Y_{j*(i')}‖ (Euclidean dist. in partner-target space)

    A tight band along a straight line through the origin is the visual
    signature of a plan that preserves global pairwise distances —
    i.e., a high Pearson r. A smeared cloud is the signature of a plan
    that scrambles pairwise geometry even if local categorical
    correspondence (NMI / AMI) is high. The Pearson r value annotated
    on each subplot matches the scalar reported in the structural
    metric column.

    Renders one subplot per plan in a single figure, with a y = x
    reference line, a least-squares fit line, and density-based alpha
    so that overplotting in the n(n-1)/2 ≈ 80k pair regime stays
    legible.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import pearson_pairwise  # noqa: F401  (kept for parity)
    from sklearn.metrics import pairwise_distances
    from scipy.stats import pearsonr

    plans = plans if plans is not None else DEFAULT_CONFUSION_PLANS
    out_dir = out_dir if out_dir is not None else (RES / "exp_grid" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = EMB / f"vision_{image_encoder}.npy"
    Y_path = EMB / f"audio_{audio_encoder}.npy"
    if not X_path.exists() or not Y_path.exists():
        print(f"[scatter] skip: missing embeddings ({X_path.name} / "
              f"{Y_path.name}). Run on HPC with embeddings/ populated.")
        return
    X = np.load(X_path)
    Y = np.load(Y_path)
    n = X.shape[0]
    print(f"[scatter] X={X.shape} (vision_{image_encoder})  "
          f"Y={Y.shape} (audio_{audio_encoder})")

    # Source-side pairwise distance is plan-independent: compute once.
    Ds = pairwise_distances(X)
    iu = np.triu_indices(n, k=1)
    a_full = Ds[iu]

    # Optional subsample for plotting (full n(n-1)/2 may overplot).
    rng = np.random.default_rng(seed)
    n_pairs = a_full.size
    if sample_pairs and sample_pairs < n_pairs:
        sub_idx = rng.choice(n_pairs, size=sample_pairs, replace=False)
    else:
        sub_idx = np.arange(n_pairs)

    loaded: list[tuple[str, np.ndarray, Path]] = []
    for label, plan_path in plans:
        if not plan_path.exists():
            print(f"[scatter] skip {label}: plan not found at {plan_path}")
            continue
        T = np.load(plan_path)
        if T.shape != (n, n):
            print(f"[scatter] skip {label}: shape {T.shape} != ({n}, {n})")
            continue
        loaded.append((label, T, plan_path))
    if not loaded:
        print("[scatter] no plans loaded; nothing to render.")
        return

    n_cells = len(loaded)
    cols = min(n_cells, 4)
    rows = int(np.ceil(n_cells / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.2 * cols, 4.2 * rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for k, (label, T, plan_path) in enumerate(loaded):
        ax = axes_flat[k]
        partners = T.argmax(axis=1)
        Dt = pairwise_distances(Y[partners])
        b_full = Dt[iu]
        # Full-set Pearson — matches the scalar reported elsewhere.
        if np.std(a_full) < 1e-12 or np.std(b_full) < 1e-12:
            r_full = float("nan")
        else:
            r_full = float(pearsonr(a_full, b_full)[0])

        a = a_full[sub_idx]
        b = b_full[sub_idx]

        sns.regplot(
            x=a, y=b, ax=ax, ci=None,
            scatter_kws={"s": 4, "alpha": 0.06, "edgecolor": "none",
                         "color": sns.color_palette("colorblind")[0]},
            line_kws={"color": sns.color_palette("colorblind")[3], "lw": 1.6,
                      "label": "OLS fit"},
        )
        lo = 0.0
        hi = max(float(a.max()), float(b.max()))
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.9,
                linestyle="--", label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\|X_i - X_{i'}\|$  (source)")
        ax.set_ylabel(r"$\|Y_{j^*(i)} - Y_{j^*(i')}\|$  (partner-target)")
        ax.set_title(f"{label}\nPearson r = {r_full:.3f}  "
                     f"(full {n_pairs} pairs)", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        sns.despine(ax=ax)

        # Persist the scalar so it can be cross-checked against the CSV.
        pd.DataFrame([{"method": label,
                       "plan": str(plan_path.relative_to(ROOT)),
                       "pearson_r_full": r_full,
                       "n_pairs": n_pairs,
                       "n_subsample": int(sub_idx.size)}]).to_csv(
            out_dir / f"scatter_r_{plan_path.parent.name}_{plan_path.stem}.csv",
            index=False,
        )

    for k in range(n_cells, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle(
        f"Cross-modal pairwise-distance scatter  "
        f"({image_encoder} → {audio_encoder}, subsample={sub_idx.size} / "
        f"{n_pairs} pairs)",
        fontsize=12, y=1.02,
    )
    out_path = out_dir / "cross_modal_pearson_scatter.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[scatter] wrote {out_path}")


# ---------------------------------------------------------------------------
# Cross-encoder comparison plots for the within-modality legs A and B.
#
# Auto-discovers every `results/exp_a*/sweep.csv` (one per image encoder)
# and `results/exp_b*/sweep.csv` (one per audio encoder), then overlays
# one curve per encoder on shared K and alpha axes. Works with whatever
# is on disk: with a single CSV the plots are still meaningful (single
# curve); with multiple CSVs each encoder is a separate hue.
# ---------------------------------------------------------------------------
COMPARISON_DIR = RES / "comparison_plots"

# Default operating point for the comparison plots. The K-sweep is shown
# at this alpha and held-out scope; the alpha-sweep is shown at this K.
COMPARISON_ALPHA = 0.5
COMPARISON_K = 300
COMPARISON_SCOPE = "heldout"

# Metrics to overlay; (column, ylabel, marker).
COMPARISON_METRICS = [
    ("R@10",       r"$R@10$",                  "o"),
    ("nmi",        "NMI (uncorrected)",        "s"),
    ("pearson_r",  r"Pearson $r$",             "^"),
]


def _discover_encoder_sweeps(letter: str) -> list[tuple[str, Path]]:
    """List (encoder_name, sweep.csv) for every `exp_{letter}*` dir on disk.

    The canonical run lives in `exp_{letter}/sweep.csv` and is reported
    under the default encoder name; non-canonical runs live in
    `exp_{letter}__<encoder>/sweep.csv`.
    """
    default_name = {"a": "clip-large", "b": "clap-unfused"}[letter]
    out: list[tuple[str, Path]] = []
    canonical = RES / f"exp_{letter}" / "sweep.csv"
    if canonical.exists():
        out.append((default_name, canonical))
    for d in sorted(RES.glob(f"exp_{letter}__*")):
        sw = d / "sweep.csv"
        if sw.exists():
            enc = d.name.split("__", 1)[1]
            out.append((enc, sw))
    return out


def _plot_overlay(
    series: list[tuple[str, pd.DataFrame]],
    x_col: str,
    out: Path,
    x_label: str,
    title: str,
    fixed_other_col: str,
    fixed_other_val,
    scope: str,
) -> None:
    """One figure with 3 panels (R@10 / NMI / Pearson r) vs x_col, one curve
    per encoder (the rows of `series`)."""
    if not series:
        print(f"[cmp] skip {out.name}: no encoder sweeps found")
        return
    palette = sns.color_palette("colorblind", n_colors=max(3, len(series)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=False)

    drew_any = False
    for i, (enc, df) in enumerate(series):
        sub = df[(df.scope == scope) & (df[fixed_other_col] == fixed_other_val)]
        sub = sub.sort_values(x_col)
        if sub.empty:
            continue
        drew_any = True
        for k, (col, ylabel, marker) in enumerate(COMPARISON_METRICS):
            if col not in sub.columns:
                continue
            axes[k].plot(sub[x_col], sub[col], marker=marker, color=palette[i],
                         label=enc, lw=1.4)

    if not drew_any:
        plt.close(fig)
        print(f"[cmp] skip {out.name}: no rows match scope={scope}, "
              f"{fixed_other_col}={fixed_other_val}")
        return

    for k, (col, ylabel, marker) in enumerate(COMPARISON_METRICS):
        axes[k].set_xlabel(x_label)
        axes[k].set_ylabel(ylabel)
        axes[k].grid(alpha=0.3)
        sns.despine(ax=axes[k])
    axes[0].legend(fontsize=8, loc="best", title="encoder")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[cmp] wrote {out}")


def _plot_canonical_bars(
    series: list[tuple[str, pd.DataFrame]],
    out: Path,
    title: str,
    K: int,
    alpha: float,
    scope: str,
) -> None:
    """One bar chart per metric (3 panels), grouped by encoder at a fixed
    (K, alpha, scope) cell."""
    if not series:
        return
    rows = []
    for enc, df in series:
        sub = df[(df.scope == scope) & (df.K == K)
                 & np.isclose(df.alpha.astype(float), alpha)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({
            "encoder": enc,
            "R@10": float(r["R@10"]),
            "NMI":  float(r["nmi"]),
            "Pearson r": float(r["pearson_r"]),
        })
    if not rows:
        print(f"[cmp] skip {out.name}: no cells at K={K}, alpha={alpha}, "
              f"scope={scope}")
        return
    df_long = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(df_long)))
    for i, metric in enumerate(["R@10", "NMI", "Pearson r"]):
        sns.barplot(data=df_long, x="encoder", y=metric, ax=axes[i],
                    palette=palette, edgecolor="white")
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt="%.3f", padding=2, fontsize=8)
        axes[i].set_xlabel("")
        axes[i].set_title(metric)
        plt.setp(axes[i].get_xticklabels(), rotation=20, ha="right")
        sns.despine(ax=axes[i])

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[cmp] wrote {out}")


def emit_unimodal_comparison() -> None:
    """Emit comparison plots for Experiments A and B across encoders."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    for letter, name, x_axis_label in [
        ("a", "Experiment A — image $\\rightarrow$ visual-caption text",
         r"$K$ (number of paired anchors)"),
        ("b", "Experiment B — audio $\\rightarrow$ audio-caption text",
         r"$K$ (number of paired anchors)"),
    ]:
        paths = _discover_encoder_sweeps(letter)
        if not paths:
            print(f"[cmp] skip {letter}: no sweep CSVs found")
            continue
        # Load each CSV once so downstream helpers operate on DataFrames.
        series = [(enc, pd.read_csv(p)) for enc, p in paths]
        encoders = [s[0] for s in series]
        print(f"[cmp] {letter}: {len(series)} encoder(s) -> {encoders}")

        # K-sweep at fixed alpha, on held-out (the honest measure).
        _plot_overlay(
            series, x_col="K",
            out=COMPARISON_DIR / f"{letter}__K_sweep_by_encoder.png",
            x_label=x_axis_label,
            title=f"{name}: K-sweep across encoders "
                  f"($\\alpha = {COMPARISON_ALPHA}$, scope = held-out)",
            fixed_other_col="alpha",
            fixed_other_val=COMPARISON_ALPHA,
            scope=COMPARISON_SCOPE,
        )

        # Alpha-sweep at fixed K, on held-out.
        _plot_overlay(
            series, x_col="alpha",
            out=COMPARISON_DIR / f"{letter}__alpha_sweep_by_encoder.png",
            x_label=r"$\alpha$ (FGW blend; 0 = Sinkhorn, 1 = pure GW)",
            title=f"{name}: $\\alpha$-sweep across encoders "
                  f"($K = {COMPARISON_K}$, scope = held-out)",
            fixed_other_col="K",
            fixed_other_val=COMPARISON_K,
            scope=COMPARISON_SCOPE,
        )

        # Canonical-operating-point bar chart, one panel per metric.
        _plot_canonical_bars(
            series,
            out=COMPARISON_DIR / f"{letter}__canonical_bars.png",
            title=f"{name}: canonical operating point "
                  f"($K = {COMPARISON_K}$, $\\alpha = {COMPARISON_ALPHA}$, "
                  f"scope = held-out)",
            K=COMPARISON_K,
            alpha=COMPARISON_ALPHA,
            scope=COMPARISON_SCOPE,
        )


# ---------------------------------------------------------------------------
# Caption-agreement plots (external-semantic metric, see code/metrics.py).
# ---------------------------------------------------------------------------
CAPTION_DIR = RES / "exp_grid" / "plots"

# Recipe key in the grid CSV  ->  human-readable label used in plots.
RECIPE_LABELS = {
    "random":       "Random (baseline)",
    "c-transitive": "Transitive Transport Bridge",
    "d":            "Caption Distance FGW",
    "unsup":        "GW (intra-modal geometry alone)",
    "text":         "Raw caption cosine (ceiling)",
}

# Per-recipe scope to read from the grid CSV (matches the conventions used
# elsewhere in the chapter).
RECIPE_SCOPE = {
    "random":       "heldout",
    "c-transitive": "heldout",
    "d":            "heldout",
    "unsup":        "heldout",
    "text":         "heldout",
}

CAP_COLUMNS = ["cap_cos_argmax", "cap_cos_planmass",
               "cap_cos_chance", "cap_cos_identity", "cap_cos_lift"]


def _has_caption_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in CAP_COLUMNS)


def emit_caption_agreement(
    grid_csv: Path = RES / "exp_grid" / "sweep.csv",
    out_dir: Path = CAPTION_DIR,
    canon_image: str = "clip-large",
    canon_audio: str = "clap-unfused",
) -> None:
    """Three caption-agreement figures from the encoder-grid sweep CSV.

    Reads ``results/exp_grid/sweep.csv`` (must include the five
    ``cap_cos_*`` columns produced by the updated ``code/metrics.py``)
    and emits:

      caption_agreement_bars.png
        Per-recipe bar chart at the canonical encoder pair, plotting
        argmax / plan-mass / chance / identity cosine side by side, with
        the lift annotated.

      caption_lift_heatmap_{recipe}.png  (one per recipe)
        $6 \\times 5$ heatmap of ``cap_cos_lift`` across the encoder
        grid; cells with positive lift are direct evidence that the
        recipe retrieves semantically related targets above chance.

      caption_agreement_refpairs.png
        Per-recipe x reference-pair bars: same scalar comparison but
        with the canonical (text-aligned) and the DINOv2-large x
        MERT-330m (text-free) pairs side by side. Useful for the Pure-GW
        narrative because the text-free pair is the cleanest evidence
        of recipe-induced semantic agreement.
    """
    if not grid_csv.exists():
        print(f"[caption] skip: {grid_csv} not present")
        return
    df = pd.read_csv(grid_csv)
    if not _has_caption_cols(df):
        print(f"[caption] skip: {grid_csv} does not have the cap_cos_* "
              f"columns. Re-run experiments with the updated metrics.py.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Canonical-pair per-recipe bar chart ----------------------
    rows = []
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        sub = df[(df.experiment == exp) & (df.scope == scope)
                 & (df.image_encoder == canon_image)
                 & (df.audio_encoder == canon_audio)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({"method": label, "argmax": r["cap_cos_argmax"],
                     "plan_mass": r["cap_cos_planmass"],
                     "chance": r["cap_cos_chance"],
                     "identity": r["cap_cos_identity"],
                     "lift": r["cap_cos_lift"]})
    if not rows:
        print("[caption] skip: no canonical-pair rows for any recipe")
    else:
        df_canon = pd.DataFrame(rows)
        long = df_canon.melt(
            id_vars="method",
            value_vars=["chance", "argmax", "plan_mass", "identity"],
            var_name="quantity", value_name="cosine",
        )
        # Order the categories for legend readability.
        long["quantity"] = pd.Categorical(
            long["quantity"], ["chance", "argmax", "plan_mass", "identity"]
        )
        fig, ax = plt.subplots(figsize=(11, 5.0))
        sns.barplot(data=long, x="method", y="cosine", hue="quantity",
                    ax=ax, palette="colorblind", edgecolor="white")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
        ax.set_ylabel("caption cosine")
        ax.set_xlabel("")
        ax.set_title(
            f"Caption-agreement scalars per recipe "
            f"(canonical pair: {canon_image} $\\times$ {canon_audio}, "
            f"held-out / same-rows scope)"
        )
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.legend(title="quantity", loc="best")
        sns.despine(ax=ax)
        fig.tight_layout()
        out = out_dir / "caption_agreement_bars.png"
        fig.savefig(out, bbox_inches="tight", dpi=140)
        plt.close(fig)
        print(f"[caption] wrote {out}")

    # ---- 2. Lift heatmap per recipe ---------------------------------
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        sub = df[(df.experiment == exp) & (df.scope == scope)].dropna(
            subset=["cap_cos_lift"])
        if sub.empty:
            continue
        pv = sub.pivot(index="image_encoder", columns="audio_encoder",
                       values="cap_cos_lift").sort_index().sort_index(axis=1)
        # Symmetric colour scale around zero so the chance baseline is
        # visually flat; positive cells are coloured warm, negative cool.
        vmax = max(0.05, float(np.nanmax(np.abs(pv.values))))
        _heatmap(
            pv,
            f"{label}: caption-cosine lift over chance "
            f"({scope.replace('_', ' ')})",
            out_dir / f"caption_lift_heatmap_{exp}.png",
            cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
            fmt="+.3f", cbar_label="cap_cos_lift",
        )

    # ---- 3. Reference-pair comparison (canonical vs text-free) ------
    free_image, free_audio = "dinov2-large", "mert-330m"
    rows = []
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        for pair_label, img_enc, aud_enc in [
            ("CLIP $\\times$ CLAP",      canon_image, canon_audio),
            ("DINOv2 $\\times$ MERT",    free_image,  free_audio),
        ]:
            sub = df[(df.experiment == exp) & (df.scope == scope)
                     & (df.image_encoder == img_enc)
                     & (df.audio_encoder == aud_enc)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({"method": label, "pair": pair_label,
                         "lift": r["cap_cos_lift"]})
    if rows:
        df_pairs = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.barplot(data=df_pairs, x="method", y="lift", hue="pair",
                    ax=ax, palette="colorblind", edgecolor="white")
        for container in ax.containers:
            ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=8)
        ax.axhline(0.0, color="grey", linestyle="--", lw=1.0,
                   label="chance (lift = 0)")
        ax.set_ylabel("cap_cos_lift  (= argmax cosine $-$ chance)")
        ax.set_xlabel("")
        ax.set_title(
            r"Caption-agreement lift per recipe at the two reference pairs"
        )
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        sns.despine(ax=ax)
        ax.legend(title="encoder pair", loc="best")
        fig.tight_layout()
        out = out_dir / "caption_agreement_refpairs.png"
        fig.savefig(out, bbox_inches="tight", dpi=140)
        plt.close(fig)
        print(f"[caption] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",
                    choices=["a", "b", "c", "d", "unsup", "text",
                             "grid", "confusion", "scatter",
                             "comparison", "caption", "tradeoff", "all"],
                    default="all")
    ap.add_argument("--comparison", action="store_true")
    ap.add_argument("--confusion-K", type=int, default=15,
                    help="Number of K-means clusters per side for the "
                         "cross-modal confusion-matrix figure.")
    ap.add_argument("--confusion-mode", choices=["soft", "hard"],
                    default="soft",
                    help="'soft': use full plan mass per (src cluster, tgt "
                         "cluster) cell. 'hard': use argmax-partner counts.")
    ap.add_argument("--image-encoder", default="clip-large",
                    help="Vision encoder for the cross-modal confusion-matrix.")
    ap.add_argument("--audio-encoder", default="clap-unfused",
                    help="Audio encoder for the cross-modal confusion-matrix.")
    ap.add_argument("--scatter-subsample", type=int, default=40000,
                    help="Number of pairs to subsample for the cross-modal "
                         "pairwise-distance scatter (Pearson r figure). 0 or "
                         "a value >= n(n-1)/2 plots all pairs.")
    args = ap.parse_args()

    if args.exp in ("a", "all"):
        emit_experiment_artifacts(RES / "exp_a", "sweep.csv")
    if args.exp in ("b", "all"):
        emit_experiment_artifacts(RES / "exp_b", "sweep.csv")
    if args.exp in ("c", "all"):
        emit_experiment_artifacts(RES / "exp_c", "sweep_direct.csv")
    if args.exp in ("d", "all"):
        emit_exp_d_artifacts(RES / "exp_d")
    if args.exp in ("unsup", "all"):
        # Same single-cell shape as Exp D — reuse the same emitter.
        emit_exp_d_artifacts(RES / "exp_unsup")
    if args.exp in ("text", "all"):
        emit_exp_d_artifacts(RES / "exp_text")
    if args.exp in ("grid", "all"):
        emit_grid_plots(RES / "exp_grid")
    if args.exp in ("confusion", "all"):
        emit_cluster_confusions(
            image_encoder=args.image_encoder,
            audio_encoder=args.audio_encoder,
            K_cl=args.confusion_K,
            mode=args.confusion_mode,
        )
    if args.exp in ("scatter", "all"):
        emit_pearson_scatter(
            image_encoder=args.image_encoder,
            audio_encoder=args.audio_encoder,
            sample_pairs=args.scatter_subsample,
        )
    if args.exp in ("comparison", "all"):
        emit_unimodal_comparison()
    if args.exp in ("caption", "all"):
        emit_caption_agreement()
    if args.exp in ("tradeoff", "all"):
        import subprocess, sys as _sys
        subprocess.run([_sys.executable,
                        str(Path(__file__).with_name("analyse_tradeoff.py"))],
                       check=False)

    if args.comparison or args.exp == "all":
        emit_comparison()


if __name__ == "__main__":
    main()
