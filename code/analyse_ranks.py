r"""Empirical CDF of the ground-truth partner's rank under each plan.

For each saved cross-modal transport plan $T \in \mathbb{R}^{n \times n}$
at a fixed encoder pair, and for each source row $i$ with ground-truth
partner $j^* = i$ (identity correspondence under the clip-identity
lemma), compute the rank of $j^*$ in $\arg\sort(-T[i, :])$.

The empirical CDF of these ranks tells us whether the recipe places
the true partner *near the top* even when $R@10$ is modest --- the
semantic-plausibility test that $R@k$ at any single $k$ cannot
answer.

This script overlays the rank-CDF for two reference encoder pairs:
  - Canonical (CLIP-L/14 + CLAP-HTSAT-unfused): text-aligned default.
  - Text-free (DINOv2-large + MERT-330m): no encoder--text alignment.

Recipes are colour-coded; the pair is encoded by linestyle (solid =
canonical, dashed = text-free).

Outputs:
  results/exp_grid/plots/rank_distribution.png
  results/exp_grid/rank_distributions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"


# Reference pairs and the suffix used in directory names.
PAIRS = [
    {"label": "text-grounded (CLIP-L/14 + CLAP-HTSAT-unfused)",
     "short": "canonical",
     "suffix": "",
     "linestyle": "-"},
    {"label": "text-free (DINOv2-large + MERT-330m)",
     "short": "text-free",
     "suffix": "__dinov2-large__mert-330m",
     "linestyle": "--"},
]


# Per-recipe plan filename (within whichever exp_*<suffix> directory).
RECIPES = [
    {"label": "Random (baseline)",            "exp": "exp_random", "file": "T_random.npy"},
    {"label": "Transitive Transport Bridge",  "exp": "exp_c",      "file": "T_transitive.npy"},
    {"label": "Caption Distance FGW",         "exp": "exp_d",      "file": "T_caption.npy"},
    {"label": "GW (intra-modal geometry)",    "exp": "exp_unsup",  "file": "T_gw.npy"},
    {"label": "Raw caption cosine (ceiling)", "exp": "exp_text",   "file": "T_text.npy"},
]


def ranks_for(plan: np.ndarray) -> np.ndarray:
    """Rank (1-indexed) of GT = diagonal entry within each row's sort."""
    diag = np.diag(plan)
    ranks = (plan > diag[:, None]).sum(axis=1) + 1
    return ranks


def main(transitive_alpha: float | None = None, K: int = 300,
         out_suffix: str = "") -> None:
    # When transitive_alpha is set, the Transitive recipe loads the saved
    # per-(K, alpha) composed plan instead of the canonical T_transitive.npy
    # (which is fixed at REUSABLE_ALPHA). Outputs get out_suffix so the
    # canonical artefacts are never overwritten.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    # (recipe_label, pair_label, linestyle, ranks_all, ranks_heldout, n)
    plot_data: list[tuple[str, str, str, np.ndarray, np.ndarray, int]] = []
    n_global: int | None = None

    for pair in PAIRS:
        for rec in RECIPES:
            dir_ = RES / f"{rec['exp']}{pair['suffix']}"
            # Override the Transitive plan with a specific (K, alpha) cell
            # from plans/ when requested; all other recipes are unaffected.
            if transitive_alpha is not None and rec["exp"] == "exp_c":
                path = (dir_ / "plans" /
                        f"T__K{K}__a{transitive_alpha:.2f}.npy")
            else:
                path = dir_ / rec["file"]
            if not path.exists():
                print(f"[skip] {pair['short']} / {rec['label']}: "
                      f"missing {path}")
                continue
            T = np.load(path)
            if T.ndim != 2 or T.shape[0] != T.shape[1]:
                print(f"[skip] {path}: shape {T.shape} not square")
                continue
            n = T.shape[0]
            if n_global is None:
                n_global = n

            r = ranks_for(T)

            # Held-out subset = rows NOT in the union of both legs'
            # anchor sets (S_compare). Saved by each experiment as
            # heldout_compare_idx.npy.
            idx_path = dir_ / "heldout_compare_idx.npy"
            if idx_path.exists():
                s_compare = np.load(idx_path).astype(int)
                heldout_rows = np.setdiff1d(np.arange(n), s_compare)
                r_heldout = r[heldout_rows]
            else:
                print(f"  [warn] no heldout_compare_idx in {dir_}; "
                      f"using full rank array as fallback")
                r_heldout = r

            plot_data.append((rec["label"], pair["label"],
                              pair["linestyle"], r, r_heldout, n))
            records.append(pd.DataFrame({
                "recipe": rec["label"], "pair": pair["short"],
                "scope": "aggregate", "rank": r,
            }))
            records.append(pd.DataFrame({
                "recipe": rec["label"], "pair": pair["short"],
                "scope": "heldout", "rank": r_heldout,
            }))
            print(f"[loaded] {pair['short']:10s}  {rec['label']:30s}  "
                  f"n={n}  med(all)={np.median(r):.1f}  "
                  f"med(heldout)={np.median(r_heldout):.1f}  "
                  f"P@10(heldout)={(r_heldout <= 10).mean():.3f}  "
                  f"|heldout|={len(r_heldout)}")

    if not plot_data:
        print("[err] no plans loaded; nothing to render")
        return

    long = pd.concat(records, ignore_index=True)
    csv_out = RES / "exp_grid" / f"rank_distributions{out_suffix}.csv"
    long.to_csv(csv_out, index=False)
    print(f"[wrote] {csv_out}")

    alpha_note = (rf"  (Transitive at $\alpha={transitive_alpha:.1f}$)"
                  if transitive_alpha is not None else "")

    # Per-recipe colour so the same recipe's two curves share a hue.
    recipe_labels = list(dict.fromkeys(r["label"] for r in RECIPES))
    palette = dict(zip(recipe_labels,
                       sns.color_palette("colorblind",
                                         n_colors=len(recipe_labels))))

    def _render(out_name: str, scope: str, title_suffix: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ks = np.arange(1, n_global + 1)
        chance = ks / n_global
        ax.plot(ks, chance, color="grey", linestyle=":", lw=1.2,
                label="random (chance)")

        for recipe_label, pair_label, ls, r_all, r_heldout, n in plot_data:
            r = r_all if scope == "aggregate" else r_heldout
            if r.size == 0:
                continue
            ecdf = np.array([(r <= k).mean() for k in ks[:n]])
            ax.plot(ks[:n], ecdf, color=palette[recipe_label],
                    linestyle=ls, lw=1.6, alpha=0.95)

        ax.set_xscale("log")
        ax.set_xlabel(r"rank $k$ of GT partner (log scale)")
        ax.set_ylabel(r"$\Pr[\,\mathrm{rank}(j^\star) \leq k\,]$")
        ax.set_title(
            "Empirical CDF of the GT partner's rank, "
            "two reference encoder pairs" + title_suffix
        )
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax)

        # Two-part legend: one block for recipe (hue), one for pair
        # (linestyle).
        recipe_handles = [
            mlines.Line2D([], [], color=palette[r], lw=1.6, label=r)
            for r in recipe_labels
        ]
        pair_handles = [
            mlines.Line2D([], [], color="black",
                          linestyle=p["linestyle"], lw=1.6,
                          label=p["label"])
            for p in PAIRS
        ]
        chance_handle = mlines.Line2D([], [], color="grey", linestyle=":",
                                      lw=1.2, label="random (chance)")
        leg1 = ax.legend(handles=recipe_handles, title="recipe",
                         fontsize=8, title_fontsize=9,
                         loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax.legend(handles=pair_handles + [chance_handle],
                  title="encoder pair", fontsize=8, title_fontsize=9,
                  loc="upper left", bbox_to_anchor=(0.0, 1.0))
        ax.add_artist(leg1)

        fig.tight_layout()
        out = PLOT_DIR / out_name
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {out}")

    _render(f"rank_distribution{out_suffix}.png", "aggregate",
            "  (aggregate, all queries)" + alpha_note)
    _render(f"rank_distribution_heldout{out_suffix}.png", "heldout",
            "  (held-out, cross-recipe)" + alpha_note)

    # Side-by-side comparison: aggregate (smooth) vs heldout (noisy).
    def _render_compare(out_name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                                 sharey=True, squeeze=False)
        ks = np.arange(1, n_global + 1)
        chance = ks / n_global

        for ax_idx, (scope, title) in enumerate([
            ("aggregate", "Aggregate  (all queries)"),
            ("heldout",   "Held-out  (cross-recipe subset)"),
        ]):
            ax = axes[0, ax_idx]
            ax.plot(ks, chance, color="grey", linestyle=":", lw=1.2,
                    label="random (chance)")
            n_eval = None
            for recipe_label, pair_label, ls, r_all, r_heldout, n in plot_data:
                r = r_all if scope == "aggregate" else r_heldout
                if r.size == 0:
                    continue
                if n_eval is None:
                    n_eval = r.size
                ecdf = np.array([(r <= k).mean() for k in ks[:n]])
                ax.plot(ks[:n], ecdf, color=palette[recipe_label],
                        linestyle=ls, lw=1.6, alpha=0.95)

            ax.set_xscale("log")
            ax.set_xlabel(r"rank $k$ of GT partner (log scale)")
            if ax_idx == 0:
                ax.set_ylabel(r"$\Pr[\,\mathrm{rank}(j^\star) \leq k\,]$")
            n_suffix = f"  ($n_{{canon}} = {n_eval}$)" if n_eval else ""
            ax.set_title(title + n_suffix, fontsize=11)
            ax.set_ylim(0, 1.02)
            ax.grid(True, which="both", alpha=0.3)
            sns.despine(ax=ax)

        # Shared bottom legend: recipe (hue) + pair (linestyle).
        recipe_handles = [
            mlines.Line2D([], [], color=palette[r], lw=1.6, label=r)
            for r in recipe_labels
        ]
        pair_handles = [
            mlines.Line2D([], [], color="black",
                          linestyle=p["linestyle"], lw=1.6,
                          label=p["label"])
            for p in PAIRS
        ]
        chance_handle = mlines.Line2D([], [], color="grey", linestyle=":",
                                      lw=1.2, label="random (chance)")
        leg1 = fig.legend(handles=recipe_handles, title="recipe",
                          loc="lower left", bbox_to_anchor=(0.02, -0.05),
                          ncol=3, fontsize=8.5, title_fontsize=9.5,
                          frameon=False)
        fig.legend(handles=pair_handles + [chance_handle],
                   title="encoder pair / chance",
                   loc="lower right", bbox_to_anchor=(0.98, -0.05),
                   ncol=3, fontsize=8.5, title_fontsize=9.5,
                   frameon=False)

        fig.suptitle(
            "Rank-CDF: aggregate vs held-out side by side",
            fontsize=13, y=1.005,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        out = PLOT_DIR / out_name
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {out}")

    _render_compare(f"rank_distribution_compare{out_suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitive-alpha", type=float, default=None,
                    help="If set, load the Transitive plan from "
                         "exp_c*/plans/T__K{K}__a{alpha}.npy instead of "
                         "the canonical T_transitive.npy. Outputs are "
                         "suffixed so canonical artefacts are preserved.")
    ap.add_argument("--K", type=int, default=300,
                    help="Anchor count for the per-alpha Transitive plan.")
    args = ap.parse_args()
    suffix = (f"__a{args.transitive_alpha:.2f}"
              if args.transitive_alpha is not None else "")
    main(transitive_alpha=args.transitive_alpha, K=args.K,
         out_suffix=suffix)
