r"""Row-sorted cumulative mass profile of the composed Transitive plan
across an alpha sweep (default 0.0 / 0.3 / 0.5 / 0.7 / 0.9), with a
category-precision companion.

Top row (mass): for each query row of a transport plan, sort its column
mass descending and accumulate it. A sharp plan reaches cumulative = 1.0
within a handful of columns (steep cliff); a smooth plan climbs gradually
(slow ramp). Averaging across rows gives one curve per (alpha, regime).
As alpha grows the curve flattens -- mass spreads over more columns.

Bottom row (category precision@k): of a row's top-k columns (by mass),
the fraction that fall in the SAME target-side k-means cluster as the
row's ground-truth partner, averaged over rows, as a function of k. This
asks whether the mass that alpha spreads lands on semantically related
(same-cluster) audio or on random audio: if category precision@k stays
high as alpha widens the plan, the spreading recovers structure at the
cluster level even as it gives up the exact instance; if it collapses,
the spreading is wasteful dilution. Together the two rows connect the
sharp-vs-wide mechanism (mass) to its coarse-retrieval payoff (category).

Layout: 2 rows x (one column per encoder regime). Three alpha curves per
panel; a chance line on the category panels (Simpson index of cluster
sizes).

Inputs:
  results/exp_c{suffix}/plans/T__K{K}__a{alpha:.2f}.npy   (composed plans)
  embeddings/audio_{aud}.npy                              (target clustering)

The category row needs the audio embeddings; if they are absent the script
still renders the mass row and annotates the category row as unavailable.

Usage:
  python code/plot_row_decay_profile.py
  python code/plot_row_decay_profile.py --K 300 --alphas 0.0 0.5 0.9 --kcl 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EMB = ROOT / "embeddings"
PLOT_DIR = RES / "exp_grid" / "plots"
SEED = 42

PAIRS = {
    "text-grounded": {
        "label":  "text-grounded (CLIP-L $\\times$ CLAP-unfused)",
        "suffix": "",
        "aud":    "clap-unfused",
    },
    "text-free": {
        "label":  "text-free (DINOv2-L $\\times$ MERT-330m)",
        "suffix": "__dinov2-large__mert-330m",
        "aud":    "mert-330m",
    },
}


def _mean_sorted_cumulative(T: np.ndarray) -> np.ndarray:
    """Mean over rows of sort-descending cumulative mass profile."""
    row_sums = T.sum(axis=1, keepdims=True)
    P = T / np.maximum(row_sums, 1e-12)
    sorted_desc = -np.sort(-P, axis=1)
    return np.cumsum(sorted_desc, axis=1).mean(axis=0)


def _category_precision_curve(T: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Mean over rows of category-precision@k for all k.

    For row i, walk its columns in descending-mass order and accumulate
    how many of the top-k share the target-cluster of i's GT partner
    (identity correspondence: the partner of row i is target i, cluster
    tgt[i]). Returns precision@k averaged over rows, length n.
    """
    order = np.argsort(-T, axis=1)
    same = (tgt[order] == tgt[:, None]).astype(float)
    ks = np.arange(1, T.shape[1] + 1)
    cumprec = np.cumsum(same, axis=1) / ks[None, :]
    return cumprec.mean(axis=0)


def _chance_precision(tgt: np.ndarray) -> float:
    """Simpson index: P(two random items share a cluster) = sum_c (n_c/n)^2."""
    sizes = np.bincount(tgt) / tgt.size
    return float((sizes * sizes).sum())


def render(out_path: Path, K: int, alphas: list[float], kcl: int) -> None:
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.4 * n_cols, 9.0),
                             squeeze=False)

    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(alphas)))
    colour_for = dict(zip(alphas, cmap))

    any_mass = False
    for col_idx, (pair_key, pair_spec) in enumerate(PAIRS.items()):
        ax_mass = axes[0, col_idx]
        ax_cat = axes[1, col_idx]

        tgt = None
        y_path = EMB / f"audio_{pair_spec['aud']}.npy"
        if y_path.exists():
            Y = np.load(y_path)
            tgt = KMeans(min(kcl, Y.shape[0]), random_state=SEED,
                         n_init=10).fit_predict(Y)

        n_eff = None
        for alpha in alphas:
            path = (RES / f"exp_c{pair_spec['suffix']}" /
                    "plans" / f"T__K{K}__a{alpha:.2f}.npy")
            if not path.exists():
                print(f"  [skip] {path}")
                continue
            T = np.load(path)
            n_eff = T.shape[1]
            any_mass = True
            ks = np.arange(1, n_eff + 1)
            ax_mass.plot(ks, _mean_sorted_cumulative(T), lw=2.0,
                         color=colour_for[alpha], label=f"$\\alpha = {alpha:.1f}$")
            if tgt is not None and tgt.shape[0] == n_eff:
                ax_cat.plot(ks, _category_precision_curve(T, tgt), lw=2.0,
                            color=colour_for[alpha])

        if n_eff is not None:
            ax_mass.plot(np.arange(1, n_eff + 1),
                         np.arange(1, n_eff + 1) / n_eff,
                         color="grey", ls=":", lw=1.2, label="uniform (random)")
        ax_mass.set_xscale("log")
        ax_mass.set_xlabel(r"rank $k$ within the row (log scale)", fontsize=10)
        ax_mass.set_ylabel(r"cumulative mass on top-$k$ columns", fontsize=10)
        ax_mass.set_title(pair_spec["label"], fontsize=11)
        ax_mass.set_ylim(0, 1.02)
        ax_mass.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax_mass)

        ax_cat.set_xscale("log")
        ax_cat.set_xlabel(r"top-$k$ retrieved (log scale)", fontsize=10)
        ax_cat.set_ylabel(rf"category precision@$k$  ($K_{{cl}}={kcl}$)",
                          fontsize=10)
        ax_cat.set_ylim(0, 1.02)
        ax_cat.grid(True, which="both", alpha=0.3)
        sns.despine(ax=ax_cat)
        if tgt is not None and n_eff is not None:
            ax_cat.axhline(_chance_precision(tgt), color="grey", ls=":",
                           lw=1.2, label="chance (Simpson index)")
            ax_cat.set_title("same-cluster recovery of the spread mass",
                             fontsize=10)
        else:
            ax_cat.text(0.5, 0.5, f"audio_{pair_spec['aud']}.npy not found\n"
                        "(category panel needs the audio embeddings)",
                        transform=ax_cat.transAxes, ha="center", va="center",
                        fontsize=9, color="grey", style="italic")

    if not any_mass:
        print(f"[row-decay] no composed plans at K={K}; regenerate first.")
        plt.close(fig)
        return

    handles = [
        Line2D([], [], color=colour_for[a], lw=2.2, label=f"$\\alpha = {a:.1f}$")
        for a in alphas
    ] + [
        Line2D([], [], color="grey", ls=":", lw=1.2,
               label="uniform / chance reference"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.03),
               ncol=len(handles), fontsize=10, frameon=False)
    fig.suptitle(
        rf"Sharp vs wide and its category payoff: mass spread (top) vs "
        rf"category precision@$k$ (bottom), composed plan ($K = {K}$)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[row-decay] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--kcl", type=int, default=15,
                    help="number of target-side k-means clusters")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    kcl_tag = "" if args.kcl == 15 else f"__kcl{args.kcl}"
    default_name = f"row_decay_profile__K{args.K}{kcl_tag}.png"
    out = Path(args.out) if args.out else PLOT_DIR / default_name
    render(out, args.K, args.alphas, args.kcl)


if __name__ == "__main__":
    main()
