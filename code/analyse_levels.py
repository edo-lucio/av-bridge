r"""Multi-granularity abstraction analysis for the cross-modal recipes.

Answers ``even when instance retrieval fails, do the recipes recover
semantically meaningful information at coarser abstraction levels?''

Two complementary measures, each computed per saved cross-modal plan
at the canonical encoder pair:

  1. AMI vs cluster granularity $K_{\mathrm{cl}}$. We sweep
     $K_{\mathrm{cl}} \\in \\{5, 10, 15, 20, 30, 50, 75, 100\\}$, fit a
     $k$-means partition of both $X$ and $Y$ at each level, decode the
     plan's argmax partner mapping, and report chance-corrected AMI
     between the source-cluster labels and the partner-mapped
     target-cluster labels. A curve that stays above zero across $K$
     says ``the method preserves categorical structure at every
     granularity''.

  2. Category-recall$@K$: the average fraction of a query's top-$K$
     retrievals that share its $k$-means cluster (at fixed
     $K_{\\mathrm{cl}} = 15$, the chapter's default). This is the
     ``soft recall'' counterpart to instance $R@K$ --- the partner
     does not need to be the exact GT, only to share the same
     semantic cluster.

Outputs:
  results/exp_grid/plots/levels_ami_vs_kcl.png
  results/exp_grid/plots/levels_category_precision_at_k.png
  results/exp_grid/levels_ami.csv
  results/exp_grid/levels_category_precision.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

# Encoder pairs the analysis can run on. ``suffix`` matches the
# results/exp_*<suffix>/ directory naming emitted by run_experiments.py
# (empty for the canonical text-grounded pair); ``label`` is for titles;
# ``tag`` is appended to output filenames so the two pairs do not
# overwrite each other (empty for text-grounded -> backward-compatible
# filenames that downstream scripts already read).
PAIRS: dict[str, dict[str, str]] = {
    "text-grounded": {"img": "clip-large",   "aud": "clap-unfused",
                      "suffix": "",                            "tag": "",
                      "label": r"CLIP-L $\times$ CLAP-unfused"},
    "text-free":     {"img": "dinov2-large", "aud": "mert-330m",
                      "suffix": "__dinov2-large__mert-330m",   "tag": "__text-free",
                      "label": r"DINOv2-L $\times$ MERT-330m"},
}

K_CL_GRID = [5, 10, 15, 20, 30, 50, 75, 100]
RECALL_K_GRID = [1, 3, 5, 10, 20, 50, 100]


def build_plans(suffix: str) -> list[tuple[str, Path]]:
    """Saved cross-modal plans for the encoder pair identified by ``suffix``."""
    return [
        ("Random (baseline)",             RES / f"exp_random{suffix}"     / "T_random.npy"),
        ("Transitive Transport Bridge",   RES / f"exp_c{suffix}"          / "T_transitive.npy"),
        ("Caption Distance FGW",          RES / f"exp_d{suffix}"          / "T_caption.npy"),
        ("Pure-GW (intra-modal geom.)",   RES / f"exp_unsup{suffix}"      / "T_gw.npy"),
        ("Raw caption cosine (ceiling)",  RES / f"exp_text{suffix}"       / "T_text.npy"),
        ("C-MCR (learned text bridge)",   RES / f"exp_cmcr{suffix}"       / "T_cmcr.npy"),
        ("Procrustes (rigid supervised)", RES / f"exp_procrustes{suffix}" / "T_procrustes.npy"),
        ("Direct ridge (supervised)",     RES / f"exp_direct{suffix}"     / "T_direct.npy"),
    ]

SEED = 42


def chance_ami_baseline(n: int, K: int, n_draws: int = 500,
                       rng_seed: int = 0) -> float:
    """Expected AMI of two uniform random K-partitions of n points."""
    rng = np.random.default_rng(rng_seed)
    vals = []
    for _ in range(n_draws):
        a = rng.integers(0, K, size=n)
        b = rng.integers(0, K, size=n)
        vals.append(adjusted_mutual_info_score(a, b))
    return float(np.mean(vals))


def ami_at_granularity(T: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      K_cl: int) -> float:
    src = KMeans(K_cl, random_state=SEED, n_init=10).fit_predict(X)
    tgt = KMeans(K_cl, random_state=SEED, n_init=10).fit_predict(Y)
    partners = T.argmax(axis=1)
    return float(adjusted_mutual_info_score(src, tgt[partners]))


def category_precision_at_k(T: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          K_cl: int, k_values: list[int],
                          query_idx: np.ndarray | None = None) -> dict[int, float]:
    """Mean fraction of each query's top-K retrievals that share the
    \\emph{target-side} cluster of the query's ground-truth partner.

    Since the K-means partitions of X and Y are independently labelled
    (cluster index 5 on the source side has no relationship to cluster
    index 5 on the target side), the meaningful comparison is between
    the target-clusters of the retrievals and the target-cluster of
    the GT partner (under the identity correspondence
    \\mathrm{gt}(i) = i, this is \\texttt{tgt[i]}).

    ``query_idx`` restricts the *query rows* that are scored (the columns,
    i.e. the candidate pool, stay the full target set). Pass the held-out
    row set so anchor-supervised recipes (Transitive, Caption-FGW) are not
    inflated by their in-sample anchor rows.
    """
    tgt = KMeans(K_cl, random_state=SEED, n_init=10).fit_predict(Y)
    ranks_desc = np.argsort(-T, axis=1)
    rows = query_idx if query_idx is not None else np.arange(Y.shape[0])
    out: dict[int, float] = {}
    for k in k_values:
        topk = ranks_desc[rows, :k]
        # Target-cluster of each retrieval vs target-cluster of GT partner.
        hits = (tgt[topk] == tgt[rows, None]).mean(axis=1)
        out[k] = float(hits.mean())
    return out


def main(img_enc: str, aud_enc: str, suffix: str, tag: str,
         pair_label: str, cat_kcls: list[int] | None = None) -> None:
    cat_kcls = cat_kcls or [15]
    x_path = EMB / f"vision_{img_enc}.npy"
    y_path = EMB / f"audio_{aud_enc}.npy"
    if not (x_path.exists() and y_path.exists()):
        print(f"[err] missing embeddings: {x_path.name} or {y_path.name}")
        print("       Run on HPC where embeddings/ is populated.")
        return
    X = np.load(x_path)
    Y = np.load(y_path)
    n = X.shape[0]
    print(f"[loaded] {pair_label}: X={X.shape}  Y={Y.shape}")

    loaded: list[tuple[str, np.ndarray]] = []
    for label, p in build_plans(suffix):
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        T = np.load(p)
        if T.shape != (n, n):
            print(f"[skip] {label}: shape {T.shape} != ({n}, {n})")
            continue
        loaded.append((label, T))
    if not loaded:
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    chance_curve = {K: chance_ami_baseline(n, K) for K in K_CL_GRID}
    for label, T in loaded:
        for K in K_CL_GRID:
            ami = ami_at_granularity(T, X, Y, K)
            rows.append({"recipe": label, "K_cl": K, "ami": ami,
                         "ami_chance": chance_curve[K]})
    df_ami = pd.DataFrame(rows)
    ami_csv = RES / "exp_grid" / f"levels_ami{tag}.csv"
    df_ami.to_csv(ami_csv, index=False)
    print(f"[wrote] {ami_csv}")

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(loaded)))
    for (label, _), color in zip(loaded, palette):
        sub = df_ami[df_ami.recipe == label].sort_values("K_cl")
        ax.plot(sub.K_cl, sub.ami, marker="o", lw=1.6, color=color, label=label)
    Ks = K_CL_GRID
    chance_vals = [chance_curve[K] for K in Ks]
    ax.plot(Ks, chance_vals, color="grey", linestyle="--", lw=1.2,
            label="chance baseline")
    ax.set_xlabel(r"$K_{\mathrm{cl}}$ (number of $k$-means clusters per side)")
    ax.set_ylabel("AMI (chance-corrected)")
    ax.set_title(f"Categorical alignment vs cluster granularity ({pair_label})")
    ax.legend(fontsize=9, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOT_DIR / f"levels_ami_vs_kcl{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")

    # Score only the held-out rows so the anchor-supervised recipes
    # (Transitive, Caption-FGW) are read out-of-sample, not inflated by
    # their in-sample anchor rows. The held-out partition is the shared
    # complement of the K=REUSABLE_K anchor union, saved by every
    # cross-modal experiment; we read it from the Transitive bridge dir.
    query_idx = None
    held_path = RES / f"exp_c{suffix}" / "heldout_compare_idx.npy"
    if held_path.exists():
        s_compare = np.load(held_path).astype(int)
        query_idx = np.setdiff1d(np.arange(n), s_compare)
        print(f"[heldout] category-precision scored on {query_idx.size} "
              f"held-out rows (of {n})")
    else:
        print("[warn] no exp_c/heldout_compare_idx.npy; "
              "category-precision falls back to all rows (aggregate)")
    # One category-precision table + plot per target-cluster granularity.
    # Coarser clusters (small K_cl) are where the structural (high-alpha)
    # recipes have the best shot at category precision; kcl=15 keeps the
    # canonical (untagged) filenames for backward compatibility.
    for kcl in cat_kcls:
        rows = []
        # Chance baseline: probability that a uniformly random retrieval lands
        # in the same target-cluster as a uniformly random query's GT partner.
        # = sum_c (n_c / n)^2 where n_c are the target-cluster sizes.
        tgt = KMeans(min(kcl, n), random_state=SEED, n_init=10).fit_predict(Y)
        tgt_sizes = np.bincount(tgt) / n
        expected_chance = float((tgt_sizes * tgt_sizes).sum())
        for label, T in loaded:
            ks_dict = category_precision_at_k(T, X, Y, kcl, RECALL_K_GRID,
                                              query_idx=query_idx)
            for k, v in ks_dict.items():
                rows.append({"recipe": label, "k": k, "category_precision": v,
                             "category_precision_chance": expected_chance})
        df_cr = pd.DataFrame(rows)
        kcl_tag = "" if kcl == 15 else f"__kcl{kcl}"
        cr_csv = RES / "exp_grid" / f"levels_category_precision{tag}{kcl_tag}.csv"
        df_cr.to_csv(cr_csv, index=False)
        print(f"[wrote] {cr_csv}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for (label, _), color in zip(loaded, palette):
            sub = df_cr[df_cr.recipe == label].sort_values("k")
            ax.plot(sub.k, sub.category_precision, marker="o", lw=1.6,
                    color=color, label=label)
        ax.axhline(expected_chance, color="grey", linestyle="--", lw=1.2,
                   label=f"chance ({expected_chance:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ (top-$k$ retrievals)")
        ax.set_ylabel(rf"category precision$@k$  ($K_{{\mathrm{{cl}}}} = {kcl}$)")
        scope_note = ("held-out queries" if query_idx is not None
                      else "all queries")
        ax.set_title(f"Categorical precision vs retrieval depth "
                     rf"({pair_label}; {scope_note}; $K_{{\mathrm{{cl}}}}={kcl}$)")
        ax.legend(fontsize=9, loc="best")
        sns.despine(ax=ax)
        fig.tight_layout()
        out = PLOT_DIR / f"levels_category_precision_at_k{tag}{kcl_tag}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", choices=list(PAIRS), default="text-grounded",
                    help="Encoder pair to analyse. 'text-grounded' (default) "
                         "writes the canonical filenames; 'text-free' writes a "
                         "'__text-free' suffixed set.")
    ap.add_argument("--img-enc", default=None,
                    help="Override the image encoder for --pair.")
    ap.add_argument("--aud-enc", default=None,
                    help="Override the audio encoder for --pair.")
    ap.add_argument("--suffix", default=None,
                    help="Override the results/exp_*<suffix>/ directory suffix.")
    ap.add_argument("--cat-kcl", type=int, nargs="+", default=[15],
                    help="Target-cluster counts for the category-precision@k "
                         "table/plot. kcl=15 keeps the canonical (untagged) "
                         "filenames; others get a __kcl{N} suffix.")
    args = ap.parse_args()

    cfg = PAIRS[args.pair]
    img_enc = args.img_enc or cfg["img"]
    aud_enc = args.aud_enc or cfg["aud"]
    suffix = args.suffix if args.suffix is not None else cfg["suffix"]
    main(img_enc=img_enc, aud_enc=aud_enc, suffix=suffix,
         tag=cfg["tag"], pair_label=cfg["label"], cat_kcls=args.cat_kcl)
