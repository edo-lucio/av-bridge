r"""Trade-off plots between structural and semantic alignment.

Three complementary figures:

  tradeoff_pareto_scatter.png
    Four-panel scatter, one point per (recipe, encoder-pair) cell from
    ``results/exp_grid/sweep.csv``. Each panel pairs a structural axis
    (Pearson $r$ or AMI) against a semantic axis (caption-cosine lift)
    or against the exemplar-identity axis $R@10$. The Pareto front
    per panel is overlaid; recipes are colour-coded.

  tradeoff/tradeoff__<structural>__vs__<alignment>.png
    One scatter per (structural X axis, alignment Y axis) combination,
    saved to its own file under ``results/exp_grid/plots/tradeoff/``.
    Each scatter has one point per (recipe, encoder pair); a per-recipe
    regression line and a global + per-recipe Pearson r annotation make
    the orthogonality claim quantitative rather than visual. Markers
    distinguish text-aligned encoder pairs (CLIP $\times$ CLAP) from
    text-free ones (DINOv2 / ViT-MAE $\times$ MERT). Structural X axes
    are AMI and Pearson $r$; alignment Y axes span the strict-to-coarse
    spectrum: $R@10$, category-recall@10, routes correct / $K_{cl}$,
    and caption-cosine lift. Splitting one panel per file makes each
    relationship easier to read than a single multi-panel figure.

  tradeoff_alpha_curves.png
    The FGW $\\alpha$ knob is itself a trade-off lever ($\\alpha = 0$
    Sinkhorn -> semantic-leaning, $\\alpha = 1$ pure GW -> structural).
    For every recipe that admits an $\\alpha$ sweep at a fixed K we
    trace its trajectory in the (structural, semantic) plane as
    $\\alpha$ moves from $0$ to $0.9$. The result is one curve per
    recipe / encoder pair with $\\alpha$ labels on the markers.

All figures consume CSVs already on disk; no experiments needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

RECIPE_LABELS = {
    "random":       "Random (baseline)",
    "c-transitive": "Transitive Transport Bridge",
    "d":            "Caption Distance FGW",
    "unsup":        "GW (intra-modal geometry alone)",
    "text":         "Raw caption cosine (ceiling)",
}

RECIPE_SCOPE = {
    "random":       "heldout",
    "c-transitive": "heldout",
    "d":            "heldout",
    "unsup":        "heldout",
    "text":         "heldout",
}


# ----------------------------------------------------------------------------
# Pareto front utility
# ----------------------------------------------------------------------------
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Indices of Pareto-optimal points (maximising both x and y).

    points: (N, 2) array; returns int array of indices, sorted by x.
    """
    n = points.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        # Any other point that dominates i?
        for j in range(n):
            if i == j:
                continue
            if (points[j, 0] >= points[i, 0] and
                    points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or
                     points[j, 1] > points[i, 1])):
                is_dominated[i] = True
                break
    front = np.where(~is_dominated)[0]
    return front[np.argsort(points[front, 0])]


# ----------------------------------------------------------------------------
# A. Pareto scatter
# ----------------------------------------------------------------------------
PANELS = [
    {"x": "pearson_r", "x_label": "structural: Pearson $r$ (pairwise distances)",
     "y": "cap_cos_lift", "y_label": "semantic: caption-cosine lift over chance",
     "title": "Geometric structure vs external semantic agreement"},
    {"x": "ami", "x_label": "structural: AMI (chance-corrected cluster MI)",
     "y": "cap_cos_lift", "y_label": "semantic: caption-cosine lift over chance",
     "title": "Categorical structure vs external semantic agreement"},
    {"x": "ami", "x_label": "structural: AMI",
     "y": "R@10", "y_label": "exemplar identity: $R@10$",
     "title": "Categorical structure vs exemplar identity"},
    {"x": "pearson_r", "x_label": "structural: Pearson $r$",
     "y": "R@10", "y_label": "exemplar identity: $R@10$",
     "title": "Geometric structure vs exemplar identity (the GW failure mode)"},
]


def emit_pareto_scatter(
    grid_csv: Path = RES / "exp_grid" / "sweep.csv",
    out: Path = PLOT_DIR / "tradeoff_pareto_scatter.png",
) -> None:
    if not grid_csv.exists():
        print(f"[tradeoff] skip: {grid_csv} not present")
        return
    df = pd.read_csv(grid_csv)

    # Filter to one row per (recipe, encoder pair) at the recipe's scope.
    rows = []
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        sub = df[(df.experiment == exp) & (df.scope == scope)]
        for _, r in sub.iterrows():
            rows.append({
                "recipe": label,
                "image_encoder": r["image_encoder"],
                "audio_encoder": r["audio_encoder"],
                "R@10": r.get("R@10", float("nan")),
                "ami": r.get("ami", float("nan")),
                "pearson_r": r.get("pearson_r", float("nan")),
                "cap_cos_lift": r.get("cap_cos_lift", float("nan")),
            })
    if not rows:
        print("[tradeoff] skip: no cross-modal rows in grid CSV")
        return
    df_pts = pd.DataFrame(rows)

    # Drop panels whose semantic column is all NaN (caption-agreement not
    # yet populated). Quietly skip those rather than crashing.
    panels = [p for p in PANELS
              if not df_pts[p["x"]].isna().all()
              and not df_pts[p["y"]].isna().all()]
    if not panels:
        print("[tradeoff] skip: no panels have both axes populated. "
              "Re-run experiments with the updated metrics.py.")
        return
    n_panels = len(panels)
    fig, axes = plt.subplots(
        nrows=2 if n_panels > 2 else 1, ncols=2 if n_panels > 1 else 1,
        figsize=(12, 10 if n_panels > 2 else 5),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    palette = dict(zip(RECIPE_LABELS.values(),
                       sns.color_palette("colorblind", n_colors=len(RECIPE_LABELS))))

    for ax, p in zip(axes_flat, panels):
        sub = df_pts.dropna(subset=[p["x"], p["y"]])
        if sub.empty:
            ax.set_axis_off()
            continue
        sns.scatterplot(data=sub, x=p["x"], y=p["y"], hue="recipe",
                        ax=ax, palette=palette, s=55, edgecolor="white",
                        linewidth=0.6, alpha=0.85)

        # Pareto front (maximise both axes).
        pts = sub[[p["x"], p["y"]]].values.astype(float)
        front_idx = pareto_front(pts)
        if front_idx.size >= 2:
            ax.plot(pts[front_idx, 0], pts[front_idx, 1],
                    color="grey", linestyle="--", lw=1.2, alpha=0.7,
                    label="Pareto front")
            ax.scatter(pts[front_idx, 0], pts[front_idx, 1],
                       facecolor="none", edgecolor="black", s=120,
                       linewidth=1.0, zorder=5)

        # Global + per-recipe Pearson r between the two axes.
        global_r = _pearson_safe(sub[p["x"]].values, sub[p["y"]].values)
        annot_lines = [f"global $r$ = {global_r:+.2f}"]
        for recipe in palette.keys():
            rec_sub = sub[sub.recipe == recipe]
            r_val = _pearson_safe(rec_sub[p["x"]].values,
                                  rec_sub[p["y"]].values)
            short = recipe.split("(")[0].split("FGW")[0].strip().rstrip(":")
            if not short:
                short = recipe
            short = short[:22]
            annot_lines.append(f"  {short}: {r_val:+.2f}"
                               if np.isfinite(r_val)
                               else f"  {short}:   n/a")
        ax.text(0.02, 0.98, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="lightgrey", alpha=0.85))

        ax.set_xlabel(p["x_label"])
        ax.set_ylabel(p["y_label"])
        ax.set_title(p["title"], fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        sns.despine(ax=ax)

    for ax in axes_flat[len(panels):]:
        ax.set_axis_off()

    fig.suptitle(
        r"Structural vs semantic trade-off across recipes and encoder pairs",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] wrote {out}")


# ----------------------------------------------------------------------------
# A.b Structure-vs-semantics scatter (focused orthogonality figure)
# ----------------------------------------------------------------------------
# Encoder families used for marker style. "Text-aligned" = trained with
# a text contrastive objective on the same side (CLIP for images, CLAP
# for audio). "Text-free" = self-supervised, no text contact (DINOv2,
# ViT-MAE for images; MERT for audio).
_TEXT_ALIGNED_IMG = {"clip-base", "clip-large"}
_TEXT_ALIGNED_AUD = {"clap-fused", "clap-larger", "clap-unfused"}


def _encoder_family(image_enc: str, audio_enc: str) -> str:
    img_aligned = image_enc in _TEXT_ALIGNED_IMG
    aud_aligned = audio_enc in _TEXT_ALIGNED_AUD
    if img_aligned and aud_aligned:
        return "text-aligned both sides"
    if (not img_aligned) and (not aud_aligned):
        return "text-free both sides"
    return "mixed"


_FAMILY_MARKERS = {
    "text-aligned both sides": "o",
    "mixed":                   "D",
    "text-free both sides":    "s",
}


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN guards; returns float('nan') if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x, y = x[mask], y[mask]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# Structural X axes (properties the plan preserves).
STRUCT_AXES = [
    {"col": "ami",         "label": "structural: AMI",          "slug": "ami"},
    {"col": "pearson_r",   "label": "structural: Pearson $r$",  "slug": "pearson_r"},
]

# Y axes spanning the strict-to-coarse alignment-quality spectrum.
# ``__routes_ratio`` is computed inline as routes_correct / routes_total.
Y_AXES = [
    {"col": "R@10",            "label": "identity: $R@10$",
     "slug": "R10"},
    {"col": "cat_precision_10",   "label": "class retrieval: cat-prec@10",
     "slug": "cat_precision_10"},
    {"col": "__routes_ratio",  "label": "routing: routes correct / $K_{cl}$",
     "slug": "routes_ratio"},
    {"col": "cap_cos_lift",    "label": "semantic: caption-cosine lift",
     "slug": "cap_cos_lift"},
]


def _add_routes_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Append a __routes_ratio column to df, defensive against missing/zero."""
    if "__routes_ratio" in df.columns:
        return df
    if "routes_total" not in df.columns or "routes_correct" not in df.columns:
        df = df.copy()
        df["__routes_ratio"] = float("nan")
        return df
    total = df["routes_total"].astype(float)
    correct = df["routes_correct"].astype(float)
    ratio = np.where((total > 0) & np.isfinite(total),
                     correct / np.where(total == 0, 1.0, total),
                     float("nan"))
    df = df.copy()
    df["__routes_ratio"] = ratio
    return df


def _emit_single_tradeoff_panel(
    df_pts: pd.DataFrame,
    palette: dict,
    x_col: str, x_label: str,
    y_col: str, y_label: str,
    out: Path,
) -> None:
    """Render a single (structural X, alignment Y) scatter to its own file."""
    sub = df_pts.dropna(subset=[x_col, y_col])
    if sub.empty:
        print(f"[struct-vs-Y] skip {out.name}: no rows with both axes populated.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0))

    # Per-recipe scatter, per-family marker.
    for recipe, color in palette.items():
        for family, marker in _FAMILY_MARKERS.items():
            cell = sub[(sub.recipe == recipe) & (sub.family == family)]
            if cell.empty:
                continue
            ax.scatter(cell[x_col], cell[y_col],
                       color=color, marker=marker, s=72,
                       edgecolor="white", linewidth=0.7, alpha=0.9)

    # Per-recipe regression line (no CI, line-only).
    for recipe, color in palette.items():
        rec_sub = sub[sub.recipe == recipe].dropna(subset=[x_col, y_col])
        if len(rec_sub) >= 3:
            sns.regplot(data=rec_sub, x=x_col, y=y_col, ax=ax,
                        ci=None, scatter=False,
                        line_kws={"color": color, "lw": 1.2,
                                  "alpha": 0.55})

    # Global + per-recipe Pearson r annotation.
    global_r = _pearson_safe(sub[x_col].values, sub[y_col].values)
    annot_lines = [f"global $r$ = {global_r:+.2f}"]
    for recipe in palette.keys():
        rec_sub = sub[sub.recipe == recipe]
        r_val = _pearson_safe(rec_sub[x_col].values,
                              rec_sub[y_col].values)
        short = recipe.split("(")[0].split("FGW")[0].strip().rstrip(":")
        if not short:
            short = recipe
        short = short[:24]
        annot_lines.append(f"  {short}: {r_val:+.2f}"
                           if np.isfinite(r_val)
                           else f"  {short}:   n/a")
    ax.text(0.02, 0.98, "\n".join(annot_lines),
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{x_label.split(': ')[-1]}  vs  {y_label.split(': ')[-1]}",
                 fontsize=12)
    sns.despine(ax=ax)

    # Two-block legend below the axis: recipe (colour) + encoder family (marker).
    from matplotlib.lines import Line2D
    recipe_handles = [
        Line2D([], [], color=color, marker="o", linestyle="None",
               markersize=7, label=label)
        for label, color in palette.items()
    ]
    family_handles = [
        Line2D([], [], color="grey", marker=marker, linestyle="None",
               markersize=7, label=family)
        for family, marker in _FAMILY_MARKERS.items()
    ]
    fig.legend(handles=recipe_handles, title="recipe",
               loc="lower left", bbox_to_anchor=(0.02, -0.02),
               ncol=2, fontsize=8, title_fontsize=9, frameon=False)
    fig.legend(handles=family_handles, title="encoder pair family",
               loc="lower right", bbox_to_anchor=(0.98, -0.02),
               ncol=1, fontsize=8, title_fontsize=9, frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[struct-vs-Y] wrote {out}")


def emit_structure_vs_semantics(
    grid_csv: Path = RES / "exp_grid" / "sweep.csv",
    out_dir:  Path = PLOT_DIR / "tradeoff",
) -> None:
    """Are structural and alignment-quality metrics correlated across recipes?

    Emits one scatter per (structural X axis, alignment Y axis) combination
    to its own file under ``out_dir``. Each scatter has one point per
    (recipe, encoder pair); per-recipe regression lines and a global +
    per-recipe Pearson r annotation make the orthogonality claim
    quantitative rather than visual.

    Y axes span the strict-to-coarse spectrum of alignment quality:
    identity ($R@10$), class retrieval (cat-prec@10), cluster routing
    (routes correct / $K_{cl}$), and external semantic agreement
    (caption-cosine lift). Splitting one panel per file makes each
    relationship easier to read than the previous 2x2 grid."""
    if not grid_csv.exists():
        print(f"[struct-vs-Y] skip: {grid_csv} not present")
        return
    df = pd.read_csv(grid_csv)
    df = _add_routes_ratio(df)

    rows = []
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        sub = df[(df.experiment == exp) & (df.scope == scope)]
        for _, r in sub.iterrows():
            rows.append({
                "recipe": label,
                "image_encoder": r["image_encoder"],
                "audio_encoder": r["audio_encoder"],
                "family": _encoder_family(r["image_encoder"],
                                          r["audio_encoder"]),
                "R@10":            r.get("R@10",            float("nan")),
                "cat_precision_10":   r.get("cat_precision_10",   float("nan")),
                "__routes_ratio":  r.get("__routes_ratio",  float("nan")),
                "ami":             r.get("ami",             float("nan")),
                "pearson_r":       r.get("pearson_r",       float("nan")),
                "cap_cos_lift":    r.get("cap_cos_lift",    float("nan")),
            })
    if not rows:
        print("[struct-vs-Y] skip: no cross-modal rows in grid CSV")
        return
    df_pts = pd.DataFrame(rows)

    palette = dict(zip(RECIPE_LABELS.values(),
                       sns.color_palette("colorblind",
                                         n_colors=len(RECIPE_LABELS))))

    out_dir.mkdir(parents=True, exist_ok=True)
    for x_spec in STRUCT_AXES:
        for y_spec in Y_AXES:
            x_col, y_col = x_spec["col"], y_spec["col"]
            if df_pts[x_col].isna().all() or df_pts[y_col].isna().all():
                continue
            out = out_dir / f"tradeoff__{x_spec['slug']}__vs__{y_spec['slug']}.png"
            _emit_single_tradeoff_panel(
                df_pts, palette,
                x_col=x_col, x_label=x_spec["label"],
                y_col=y_col, y_label=y_spec["label"],
                out=out,
            )


# ----------------------------------------------------------------------------
# B. Alpha-curve trajectories per recipe (FGW knob as trade-off lever)
# ----------------------------------------------------------------------------
# Per-recipe sweep CSV and the K row at which to trace alpha. C-direct
# admits a full K-sweep; D / A / B have only one effective K.
ALPHA_SOURCES = [
    {"label": "Experiment A  (image $\\to$ visual-text)",
     "csv":   RES / "exp_a" / "sweep.csv",
     "K":     300, "scope": "heldout"},
    {"label": "Experiment B  (audio $\\to$ audio-text)",
     "csv":   RES / "exp_b" / "sweep.csv",
     "K":     300, "scope": "heldout"},
    {"label": "C-direct  (image $\\to$ audio, supervised)",
     "csv":   RES / "exp_c" / "sweep_direct.csv",
     "K":     300, "scope": "heldout"},
    {"label": "D  (caption-cost FGW)",
     "csv":   RES / "exp_d" / "sweep.csv",
     "K":     0,   "scope": "aggregate"},
]


def emit_alpha_curves(
    out: Path = PLOT_DIR / "tradeoff_alpha_curves.png",
) -> None:
    palette = sns.color_palette("colorblind", n_colors=len(ALPHA_SOURCES))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    drew_any = False

    for ax, (x_col, y_col, x_label, y_label, title) in zip(
        axes,
        [
            ("pearson_r", "R@10",
             "structural: Pearson $r$",
             "exemplar identity: $R@10$",
             r"$\alpha$-trajectory: structural ($r$) vs identity ($R@10$)"),
            ("ami", "cap_cos_lift",
             "structural: AMI",
             "semantic: caption-cosine lift",
             r"$\alpha$-trajectory: structural (AMI) vs semantic (caption lift)"),
        ],
    ):
        for src, color in zip(ALPHA_SOURCES, palette):
            if not src["csv"].exists():
                continue
            df = pd.read_csv(src["csv"])
            if x_col not in df.columns or y_col not in df.columns:
                continue
            sub = df[(df.scope == src["scope"]) & (df.K == src["K"])]
            sub = sub.dropna(subset=[x_col, y_col]).sort_values("alpha")
            if sub.empty:
                continue
            drew_any = True
            xs = sub[x_col].values.astype(float)
            ys = sub[y_col].values.astype(float)
            ax.plot(xs, ys, color=color, lw=1.6, alpha=0.85,
                    label=src["label"])
            ax.scatter(xs, ys, color=color, s=45, edgecolor="white",
                       linewidth=0.6, zorder=5)
            # Label each marker with its alpha value.
            for x, y, a in zip(xs, ys, sub["alpha"].values):
                ax.annotate(f"$\\alpha={a:.1f}$",
                            xy=(x, y), xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=6.5, color=color, alpha=0.9)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        sns.despine(ax=ax)

    if not drew_any:
        plt.close(fig)
        print("[tradeoff] skip alpha-curves: no rows in any source CSV")
        return

    fig.suptitle(
        r"FGW $\alpha$ as a trade-off lever within each recipe",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] wrote {out}")


# ----------------------------------------------------------------------------
# C. Recipe-level alignment vs pre-alignment encoder similarity (CKA)
# ----------------------------------------------------------------------------
# CKA is computed once per encoder pair, before any recipe is applied;
# it is a property of the two embedding spaces themselves. Plotting each
# recipe's alignment metrics against CKA answers the question
# "does the recipe's alignment quality scale with how similar the two
# encoder spaces already were?"
def emit_cka_vs_recipe(
    grid_csv: Path = RES / "exp_grid" / "sweep.csv",
    cka_csv:  Path = RES / "exp_grid" / "geometric_similarity.csv",
    out:      Path = PLOT_DIR / "tradeoff_cka_vs_recipe.png",
) -> None:
    """Two-row panel: each recipe's structural / semantic / identity
    metric plotted against the pre-alignment CKA of the same encoder
    pair. Top row: structural and identity axes (Pearson r, AMI,
    R@10). Bottom row: semantic axis (cap_cos_lift), populated once
    the caption-agreement re-runs have landed."""
    if not grid_csv.exists():
        print(f"[cka-vs-recipe] skip: {grid_csv} not present")
        return
    if not cka_csv.exists():
        print(f"[cka-vs-recipe] skip: {cka_csv} not present "
              "(run analyse_geometry.py on HPC first)")
        return

    df_grid = pd.read_csv(grid_csv)
    df_cka  = pd.read_csv(cka_csv)

    # Merge per-cell metrics with the CKA / identity-Pearson values.
    merged_rows = []
    for exp, label in RECIPE_LABELS.items():
        scope = RECIPE_SCOPE[exp]
        sub = df_grid[(df_grid.experiment == exp) & (df_grid.scope == scope)]
        for _, r in sub.iterrows():
            cka_row = df_cka[(df_cka.image_encoder == r["image_encoder"])
                             & (df_cka.audio_encoder == r["audio_encoder"])]
            if cka_row.empty:
                continue
            cka = float(cka_row.iloc[0]["cka"])
            rid = float(cka_row.iloc[0]["pearson_r_identity"])
            merged_rows.append({
                "recipe": label,
                "image_encoder": r["image_encoder"],
                "audio_encoder": r["audio_encoder"],
                "cka": cka,
                "id_pearson": rid,
                "pearson_r":   r.get("pearson_r", float("nan")),
                "ami":         r.get("ami", float("nan")),
                "R@10":        r.get("R@10", float("nan")),
                "cap_cos_lift": r.get("cap_cos_lift", float("nan")),
            })
    if not merged_rows:
        print("[cka-vs-recipe] skip: no rows match between CSVs")
        return
    df_m = pd.DataFrame(merged_rows)

    palette = dict(zip(RECIPE_LABELS.values(),
                       sns.color_palette("colorblind", n_colors=len(RECIPE_LABELS))))

    # Top row: structural & identity axes.
    # Bottom row: semantic axis (if populated).
    has_cap = not df_m["cap_cos_lift"].isna().all()
    n_rows = 2 if has_cap else 1
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(15, 5.0 * n_rows),
                             squeeze=False)

    def _scatter_with_trend(ax, df, x_col, y_col, ylabel, title):
        sub = df.dropna(subset=[x_col, y_col])
        if sub.empty:
            ax.set_axis_off()
            return
        # legend=False -> per-panel legend suppressed; we use one figure-
        # level legend at the bottom instead.
        sns.scatterplot(data=sub, x=x_col, y=y_col, hue="recipe",
                        ax=ax, palette=palette, s=55,
                        edgecolor="white", linewidth=0.6, alpha=0.85,
                        legend=False)
        # Per-recipe regression line. ci=None keeps the panel uncluttered.
        for recipe, color in palette.items():
            rec_sub = sub[sub.recipe == recipe]
            if len(rec_sub) >= 3:
                sns.regplot(data=rec_sub, x=x_col, y=y_col,
                            ax=ax, ci=None, scatter=False,
                            line_kws={"color": color, "lw": 1.2,
                                      "alpha": 0.55})
        # Global + per-recipe Pearson r annotation.
        global_r = _pearson_safe(sub[x_col].values, sub[y_col].values)
        annot_lines = [f"global $r$ = {global_r:+.2f}"]
        for recipe in palette.keys():
            rec_sub = sub[sub.recipe == recipe]
            r_val = _pearson_safe(rec_sub[x_col].values,
                                  rec_sub[y_col].values)
            short = recipe.split("(")[0].split("FGW")[0].strip().rstrip(":")
            if not short:
                short = recipe
            short = short[:22]
            annot_lines.append(f"  {short}: {r_val:+.2f}"
                               if np.isfinite(r_val)
                               else f"  {short}:   n/a")
        ax.text(0.02, 0.98, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="lightgrey", alpha=0.85))
        ax.set_xlabel(r"linear CKA (encoder-space similarity, pre-alignment)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        sns.despine(ax=ax)

    # Top row: structural (Pearson r), structural (AMI), identity (R@10).
    _scatter_with_trend(axes[0, 0], df_m, "cka", "pearson_r",
                        ylabel=r"structural: Pearson $r$",
                        title=r"Recipe structural ($r$) vs encoder CKA")
    _scatter_with_trend(axes[0, 1], df_m, "cka", "ami",
                        ylabel=r"structural: AMI",
                        title=r"Recipe AMI vs encoder CKA")
    _scatter_with_trend(axes[0, 2], df_m, "cka", "R@10",
                        ylabel=r"identity: $R@10$",
                        title=r"Recipe identity ($R@10$) vs encoder CKA")

    # Bottom row: semantic (cap_cos_lift) — three views of the same axis
    # so the figure is informative even before the rerun by leaving
    # other slots blank.
    if has_cap:
        _scatter_with_trend(axes[1, 0], df_m, "cka", "cap_cos_lift",
                            ylabel=r"semantic: caption-cosine lift",
                            title=r"Recipe semantic (caption lift) vs encoder CKA")
        # Combined view: structural Pearson r vs semantic, coloured by CKA.
        ax = axes[1, 1]
        sub_c = df_m.dropna(subset=["pearson_r", "cap_cos_lift"])
        if not sub_c.empty:
            sc = ax.scatter(sub_c["pearson_r"], sub_c["cap_cos_lift"],
                            c=sub_c["cka"], cmap="viridis",
                            s=55, edgecolor="white", linewidth=0.6)
            fig.colorbar(sc, ax=ax, label="encoder CKA", shrink=0.85)
            ax.set_xlabel(r"structural: Pearson $r$")
            ax.set_ylabel(r"semantic: caption-cosine lift")
            ax.set_title(r"Structural vs semantic, coloured by CKA",
                         fontsize=10)
            sns.despine(ax=ax)
        else:
            ax.set_axis_off()

        # Lift-over-CKA: cap_cos_lift / cka, per-encoder-pair, per-recipe.
        # Read as "how much semantic signal does the recipe extract per
        # unit of pre-alignment encoder similarity?"
        ax = axes[1, 2]
        sub_l = df_m.dropna(subset=["cka", "cap_cos_lift"])
        sub_l = sub_l[sub_l.cka > 0.05]  # avoid divide-by-near-zero
        if not sub_l.empty:
            sub_l = sub_l.assign(
                ratio=lambda d: d["cap_cos_lift"] / d["cka"])
            order = (sub_l.groupby("recipe")["ratio"].median()
                     .sort_values(ascending=False).index.tolist())
            sns.boxplot(data=sub_l, x="recipe", y="ratio", order=order,
                        hue="recipe", ax=ax, palette=palette,
                        fliersize=2, legend=False)
            ax.axhline(0.0, color="grey", linestyle="--", lw=1.0)
            ax.set_ylabel(r"caption lift / CKA  "
                          r"(semantic signal per unit CKA)")
            ax.set_xlabel("")
            ax.set_title(r"Semantic-lift efficiency per recipe",
                         fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
            sns.despine(ax=ax)
        else:
            ax.set_axis_off()

    # Single figure-level recipe legend at the bottom, shared across panels.
    from matplotlib.lines import Line2D
    recipe_handles = [
        Line2D([], [], color=color, marker="o", linestyle="None",
               markersize=8, label=label)
        for label, color in palette.items()
    ]
    fig.legend(handles=recipe_handles, title="recipe",
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(recipe_handles), 6),
               fontsize=9, title_fontsize=10, frameon=False)

    fig.suptitle(
        r"Recipe alignment vs pre-alignment encoder similarity (CKA)",
        fontsize=13, y=1.005,
    )
    # Leave room at the bottom for the shared legend.
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] wrote {out}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    emit_pareto_scatter()
    emit_structure_vs_semantics()
    emit_alpha_curves()
    emit_cka_vs_recipe()


if __name__ == "__main__":
    main()
