r"""Mass-dispersion panel: how the coupling spreads as alpha increases.

For the canonical bridge plans we already have on disk, this draws a
``regime x alpha`` grid of transport-plan heatmaps so the reader can *see*
where the mass goes as the FGW structure weight alpha rises.

Each panel is the held-out block of the plan: rows are the held-out image
queries (the complement of the K=300 ``compare`` set scored by
``evaluate_heldout``), columns are those same clips' audio as the candidate
pool, ordered so the *correct* image->audio pairing lies on the diagonal.
Rows are renormalised to sum to 1, so each row is that query's mass
distribution over the probe candidates; a bright diagonal = correct mass, a
diffuse row = mass dispersed onto wrong (off-diagonal) pairings.

Two summaries are annotated per panel:
  * ``diag`` -- mean diagonal mass (fraction landing on the correct pairing);
  * ``supp`` -- mean effective support exp(row-entropy), i.e. the effective
    number of candidates each query spreads over.

The headline trend (both regimes): as alpha rises the mass spreads (``supp``
grows) but *off* the diagonal (``diag`` falls) -- more structure, less
correct retrieval -- the coupling-level view of "the most geometry-preserving
alpha is not the best-retrieving one".

Inputs (all on disk; no embeddings needed):
  transitive: results/exp_c{suffix}/plans/T__K{K}__a{alpha}.npy
  caption:    results/exp_d{suffix}/plans/T__a{alpha}.npy
  + results/exp_{c,d}{suffix}/heldout_compare_idx.npy

Output:
  results/exp_grid/plots/mass_dispersion_panel__{recipe}.png

Usage:
  python code/plot_mass_dispersion_panel.py                      # transitive, K=300
  python code/plot_mass_dispersion_panel.py --recipe caption
  python code/plot_mass_dispersion_panel.py --alphas 0.3 0.5 0.7 0.9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="white", context="notebook", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

REGIMES = [
    (r"Text-grounded" "\n" r"(CLIP-L $\times$ CLAP-unfused)", ""),
    (r"Text-free" "\n" r"(DINOv2-L $\times$ MERT-330m)", "__dinov2-large__mert-330m"),
]

CMAP = "magma"


def _plan_path(recipe: str, suffix: str, K: int, alpha: float) -> Path:
    if recipe == "transitive":
        return RES / f"exp_c{suffix}" / "plans" / f"T__K{K}__a{alpha:.2f}.npy"
    return RES / f"exp_d{suffix}" / "plans" / f"T__a{alpha:.2f}.npy"


def _heldout_idx_path(recipe: str, suffix: str) -> Path:
    exp = "exp_c" if recipe == "transitive" else "exp_d"
    return RES / f"{exp}{suffix}" / "heldout_compare_idx.npy"


def _held_block(plan: Path, idx: Path) -> np.ndarray | None:
    """Row-normalised held-out x held-out block with the correct pairing on
    the diagonal, or None if an input is missing."""
    if not plan.exists() or not idx.exists():
        return None
    T = np.load(plan)
    S = np.load(idx)
    held = np.setdiff1d(np.arange(T.shape[0]), S)   # rows scored by evaluate_heldout
    blk = T[np.ix_(held, held)]                      # GT of held[k] is column k
    return blk / np.clip(blk.sum(axis=1, keepdims=True), 1e-12, None)


def _summaries(blk: np.ndarray) -> tuple[float, float]:
    """(mean diagonal mass, mean effective support exp(row-entropy))."""
    diag = float(np.diag(blk).mean())
    ent = -(blk * np.log(np.clip(blk, 1e-12, None))).sum(axis=1)
    return diag, float(np.exp(ent).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["transitive", "caption"],
                    default="transitive")
    ap.add_argument("--K", type=int, default=300,
                    help="anchor budget for the transitive recipe (ignored "
                         "for caption).")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    alphas = args.alphas
    nrow, ncol = len(REGIMES), len(alphas)

    # Shared colour ceiling so panels are comparable; 99th pct keeps the
    # bright diagonal from washing out spread.
    blocks: dict[tuple[int, int], np.ndarray] = {}
    vmax = 0.0
    for r, (_, suffix) in enumerate(REGIMES):
        for c, a in enumerate(alphas):
            blk = _held_block(_plan_path(args.recipe, suffix, args.K, a),
                              _heldout_idx_path(args.recipe, suffix))
            if blk is not None:
                blocks[(r, c)] = blk
                vmax = max(vmax, float(np.quantile(blk, 0.99)))
    if not blocks:
        raise SystemExit("[err] no plans found on disk for recipe="
                         f"{args.recipe!r}; check results/exp_* dirs.")
    vmax = vmax or 1.0

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow),
                             squeeze=False, layout="constrained")
    im = None
    for r, (row_label, _) in enumerate(REGIMES):
        for c, a in enumerate(alphas):
            ax = axes[r][c]
            blk = blocks.get((r, c))
            if blk is None:
                ax.text(0.5, 0.5, "missing plan", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = ax.imshow(blk, cmap=CMAP, vmin=0.0, vmax=vmax,
                               aspect="auto", interpolation="nearest")
                n = blk.shape[0]
                ax.plot([0, n - 1], [0, n - 1], ls="--", lw=0.8,
                        color="white", alpha=0.45)
                diag, supp = _summaries(blk)
                ax.text(0.03, 0.97, f"diag {diag:.2f}\nsupp {supp:.1f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                  ec="none", alpha=0.45))
                ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(rf"$\alpha = {a:g}$", fontsize=12)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9,
                            location="right")
        cbar.set_label("row-normalised plan mass", fontsize=10)

    recipe_name = {"transitive": "Transitive Transport Bridge",
                   "caption": "Caption-Distance FGW"}[args.recipe]
    fig.suptitle(
        f"Where the mass goes as $\\alpha$ rises — {recipe_name} (held-out probe)\n"
        "rows = image queries, cols = candidate audio, diagonal = correct pairing; "
        "brighter diagonal = correct mass, more diffuse rows = dispersion",
        fontsize=12,
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"mass_dispersion_panel__{args.recipe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
