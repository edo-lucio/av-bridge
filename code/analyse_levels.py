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

# Canonical encoder pair used throughout the chapter.
IMG_ENC = "clip-large"
AUD_ENC = "clap-unfused"

K_CL_GRID = [5, 10, 15, 20, 30, 50, 75, 100]
RECALL_K_GRID = [1, 3, 5, 10, 20, 50, 100]

PLANS: list[tuple[str, Path]] = [
    ("Random (baseline)",             RES / "exp_random" / "T_random.npy"),
    ("Transitive Transport Bridge",   RES / "exp_c"      / "T_transitive.npy"),
    ("Caption Distance FGW",          RES / "exp_d"      / "T_caption.npy"),
    ("Pure-GW (intra-modal geom.)",   RES / "exp_unsup"  / "T_gw.npy"),
    ("Raw caption cosine (ceiling)",  RES / "exp_text"   / "T_text.npy"),
]

SEED = 42


# ----------------------------------------------------------------------------
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
                          K_cl: int, k_values: list[int]) -> dict[int, float]:
    """Mean fraction of each query's top-K retrievals that share the
    \\emph{target-side} cluster of the query's ground-truth partner.

    Since the K-means partitions of X and Y are independently labelled
    (cluster index 5 on the source side has no relationship to cluster
    index 5 on the target side), the meaningful comparison is between
    the target-clusters of the retrievals and the target-cluster of
    the GT partner (under the identity correspondence
    \\mathrm{gt}(i) = i, this is \\texttt{tgt[i]}).
    """
    n = Y.shape[0]
    tgt = KMeans(K_cl, random_state=SEED, n_init=10).fit_predict(Y)
    ranks_desc = np.argsort(-T, axis=1)
    out: dict[int, float] = {}
    for k in k_values:
        topk = ranks_desc[:, :k]  # (n, k)
        # Target-cluster of each retrieval vs target-cluster of GT partner.
        hits = (tgt[topk] == tgt[:, None]).mean(axis=1)
        out[k] = float(hits.mean())
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    x_path = EMB / f"vision_{IMG_ENC}.npy"
    y_path = EMB / f"audio_{AUD_ENC}.npy"
    if not (x_path.exists() and y_path.exists()):
        print(f"[err] missing embeddings: {x_path.name} or {y_path.name}")
        print("       Run on HPC where embeddings/ is populated.")
        return
    X = np.load(x_path)
    Y = np.load(y_path)
    n = X.shape[0]
    print(f"[loaded] X={X.shape}  Y={Y.shape}")

    loaded: list[tuple[str, np.ndarray]] = []
    for label, p in PLANS:
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

    # --- (1) AMI vs K_cl ---
    rows = []
    chance_curve = {K: chance_ami_baseline(n, K) for K in K_CL_GRID}
    for label, T in loaded:
        for K in K_CL_GRID:
            ami = ami_at_granularity(T, X, Y, K)
            rows.append({"recipe": label, "K_cl": K, "ami": ami,
                         "ami_chance": chance_curve[K]})
    df_ami = pd.DataFrame(rows)
    df_ami.to_csv(RES / "exp_grid" / "levels_ami.csv", index=False)
    print(f"[wrote] {RES / 'exp_grid' / 'levels_ami.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(loaded)))
    for (label, _), color in zip(loaded, palette):
        sub = df_ami[df_ami.recipe == label].sort_values("K_cl")
        ax.plot(sub.K_cl, sub.ami, marker="o", lw=1.6, color=color, label=label)
    # chance baseline (the expected AMI of two random K-partitions; very close to 0)
    Ks = K_CL_GRID
    chance_vals = [chance_curve[K] for K in Ks]
    ax.plot(Ks, chance_vals, color="grey", linestyle="--", lw=1.2,
            label="chance baseline")
    ax.set_xlabel(r"$K_{\mathrm{cl}}$ (number of $k$-means clusters per side)")
    ax.set_ylabel("AMI (chance-corrected)")
    ax.set_title("Categorical alignment vs cluster granularity")
    ax.legend(fontsize=9, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOT_DIR / "levels_ami_vs_kcl.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")

    # --- (2) Category-recall @ K ---
    rows = []
    # Chance baseline: probability that a uniformly random retrieval lands in
    # the same target-cluster as a uniformly random query's GT partner.
    # = sum_c (n_c / n)^2 where n_c are the target-cluster sizes.
    tgt_kcl15 = KMeans(15, random_state=SEED, n_init=10).fit_predict(Y)
    tgt_sizes = np.bincount(tgt_kcl15) / n
    expected_chance = float((tgt_sizes * tgt_sizes).sum())  # exp. fraction match
    for label, T in loaded:
        ks_dict = category_precision_at_k(T, X, Y, 15, RECALL_K_GRID)
        for k, v in ks_dict.items():
            rows.append({"recipe": label, "k": k, "category_precision": v,
                         "category_precision_chance": expected_chance})
    df_cr = pd.DataFrame(rows)
    df_cr.to_csv(RES / "exp_grid" / "levels_category_precision.csv", index=False)
    print(f"[wrote] {RES / 'exp_grid' / 'levels_category_precision.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, _), color in zip(loaded, palette):
        sub = df_cr[df_cr.recipe == label].sort_values("k")
        ax.plot(sub.k, sub.category_precision, marker="o", lw=1.6, color=color,
                label=label)
    ax.axhline(expected_chance, color="grey", linestyle="--", lw=1.2,
               label=f"chance ({expected_chance:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ (top-$k$ retrievals)")
    ax.set_ylabel(r"category precision$@k$  ($K_{\mathrm{cl}} = 15$)")
    ax.set_title("Categorical precision vs retrieval depth")
    ax.legend(fontsize=9, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOT_DIR / "levels_category_precision_at_k.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
