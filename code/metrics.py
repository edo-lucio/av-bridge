"""Phase 5 — evaluation metrics for transport plans.

Every metric takes:
  - T   : transport plan, shape (n, m)
  - gt  : ground-truth pairing, shape (n,) with gt[i] = target row id
  - X_src, Y_tgt : source/target embeddings used for structural metrics
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    pairwise_distances,
    v_measure_score,
)


def recall_at_k(T: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Top-k retrieval recall against gt."""
    n = T.shape[0]
    k = min(k, T.shape[1])
    topk = np.argpartition(-T, k - 1, axis=1)[:, :k]
    hits = 0
    for i in range(n):
        if gt[i] in topk[i]:
            hits += 1
    return float(hits / n) if n > 0 else 0.0


def cluster_routing(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> tuple[int, int]:
    """Cluster-to-cluster routing accuracy.

    Returns (correct, K_cl).
    """
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    correct = 0
    for s in range(K_cl):
        idx = np.where(src_lab == s)[0]
        if len(idx) == 0:
            continue
        mass = np.zeros(K_cl)
        for j in range(T.shape[1]):
            mass[tgt_lab[j]] += T[idx, j].sum()
        predicted = int(mass.argmax())
        gt_cluster = int(
            np.bincount(tgt_lab[gt[idx]], minlength=K_cl).argmax()
        )
        if predicted == gt_cluster:
            correct += 1
    return correct, K_cl


def knn_overlap(T: np.ndarray, X_src: np.ndarray, Y_tgt: np.ndarray, k: int = 5) -> float:
    """Mean overlap between source-knn (mapped through plan) and target-knn of partner."""
    if len(X_src) < 2 or len(Y_tgt) < 2:
        return float("nan")
    partners = T.argmax(axis=1)
    Ds = pairwise_distances(X_src)
    Dt = pairwise_distances(Y_tgt)
    k = min(k, len(X_src) - 1, len(Y_tgt) - 1)
    overlaps = []
    for i in range(len(X_src)):
        src_nn = np.argsort(Ds[i])[1:k + 1]
        tgt_nn = np.argsort(Dt[partners[i]])[1:k + 1]
        mapped = partners[src_nn]
        overlaps.append(len(set(mapped) & set(tgt_nn)) / k)
    return float(np.mean(overlaps))


def caption_agreement(
    T: np.ndarray,
    Z_src: np.ndarray,
    Z_tgt: np.ndarray,
    seed: int = 42,
    n_chance_draws: int = 200,
) -> dict:
    """Caption-similarity score of the plan's argmax retrievals.

    The captions $z^{\\,\\text{src}}_i$ and $z^{\\,\\text{tgt}}_j$ are
    encoded by a third encoder that the recipe's transport plan does
    not see (in our setup: MiniLM-L6-v2 vs CLIP/CLAP). Cosine
    similarity between caption pools is therefore an *external*
    semantic-agreement signal independent of the modality embeddings
    the plan was fitted on.

    Returns a dict with five keys:
      cap_cos_argmax    : mean cosine between the source-caption of i
                          and the target-caption of the plan's argmax
                          partner of i.
      cap_cos_planmass  : plan-mass-weighted cosine; reads the full
                          plan distribution per row rather than just
                          the argmax. For uniform-marginal plans the
                          two metrics coincide in expectation under a
                          permutation upper bound.
      cap_cos_chance    : Monte-Carlo estimate of the mean cosine
                          under a random permutation of the partner
                          mapping. The honest chance baseline.
      cap_cos_identity  : mean cosine under the identity permutation
                          (the empirical upper bound on this dataset
                          --- visual and audio captions of the same
                          clip are written by independent annotators
                          and do not perfectly agree).
      cap_cos_lift      : argmax cosine minus chance cosine; positive
                          values are direct evidence of semantic
                          agreement above chance.

    Returns NaN-only dict if Z_src or Z_tgt is missing.
    """
    NAN_OUT = {
        "cap_cos_argmax":   float("nan"),
        "cap_cos_planmass": float("nan"),
        "cap_cos_chance":   float("nan"),
        "cap_cos_identity": float("nan"),
        "cap_cos_lift":     float("nan"),
    }
    if Z_src is None or Z_tgt is None:
        return NAN_OUT

    h, m = T.shape  # h source rows scored, m total target candidates
    if Z_src.shape[0] != h or Z_tgt.shape[0] != m:
        return NAN_OUT

    # Normalise once: dot product == cosine. Skip if already unit-norm.
    def _l2norm(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return M / norms

    Zs = _l2norm(Z_src.astype(np.float64))
    Zt = _l2norm(Z_tgt.astype(np.float64))

    # Full (h x m) cosine matrix between source and target captions.
    # At h, m <= 400 this is trivial in time and memory.
    C = Zs @ Zt.T

    # 1. Argmax-partner cosine.
    partners = T.argmax(axis=1)
    argmax_cos = C[np.arange(h), partners]

    # 2. Plan-mass-weighted cosine. Row-normalise the plan first so the
    # mass per source row sums to 1, otherwise rows with low marginal
    # mass contribute less than they should.
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    T_rowstoch = T / row_sums
    planmass_cos = (T_rowstoch * C).sum(axis=1)

    # 3. Identity-permutation reference (the dataset upper bound). On the
    # full-set call h == m == n and this is C[i, i]; on the held-out call
    # we cannot read C[i, i] because the held-out source rows are
    # re-indexed 0..h-1 while their GT partners sit at their original
    # indices in Y. The held-out path therefore supplies Z_src already
    # restricted to the held rows; we approximate the identity by taking
    # the per-row max of C, which is a meaningful upper-bound proxy.
    identity_cos = (C[np.arange(h), np.arange(h)]
                    if h == m else C.max(axis=1))

    # 4. Random-permutation chance: sample partners uniformly at random
    # from the m target candidates, recompute the mean argmax-style cosine,
    # average over draws.
    rng = np.random.default_rng(seed)
    chance_vals = np.empty(n_chance_draws, dtype=np.float64)
    idx = np.arange(h)
    for k in range(n_chance_draws):
        rand_partners = rng.integers(0, m, size=h)
        chance_vals[k] = C[idx, rand_partners].mean()
    chance_cos = float(chance_vals.mean())

    return {
        "cap_cos_argmax":   float(argmax_cos.mean()),
        "cap_cos_planmass": float(planmass_cos.mean()),
        "cap_cos_chance":   chance_cos,
        "cap_cos_identity": float(identity_cos.mean()),
        "cap_cos_lift":     float(argmax_cos.mean() - chance_cos),
    }


def pearson_pairwise(T: np.ndarray, X_src: np.ndarray, Y_tgt: np.ndarray) -> float:
    """Pearson correlation of pairwise distances in source space vs partner-target space."""
    if len(X_src) < 3:
        return float("nan")
    partners = T.argmax(axis=1)
    Ds = pairwise_distances(X_src)
    Dt = pairwise_distances(Y_tgt[partners])
    iu = np.triu_indices(len(X_src), k=1)
    a = Ds[iu]
    b = Dt[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


_AGREEMENT_NANS: dict = {
    "nmi": float("nan"),
    "ami": float("nan"),
    "ari": float("nan"),
    "v_measure": float("nan"),
    "homogeneity": float("nan"),
    "completeness": float("nan"),
}

_CAPAGREE_NANS: dict = {
    "cap_cos_argmax":   float("nan"),
    "cap_cos_planmass": float("nan"),
    "cap_cos_chance":   float("nan"),
    "cap_cos_identity": float("nan"),
    "cap_cos_lift":     float("nan"),
}


def cluster_agreement(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> dict:
    """Cluster-agreement metrics between source clusters and partner-target
    clusters under the plan's hard argmax assignment.

    Returns a dict with six metrics:

    - ``nmi`` — normalised mutual information (arithmetic-mean normalisation).
      Not chance-corrected; expected value of independent labellings grows
      with K_cl / n.
    - ``ami`` — chance-corrected mutual information. ≈ 0 under independent
      labellings; positive values have a "lift over chance" reading.
    - ``ari`` — adjusted Rand index. Chance-corrected pairwise agreement:
      asks, of all source pairs (i, i'), whether they end up in the same
      target cluster iff they were in the same source cluster.
    - ``v_measure`` — symmetric harmonic mean of homogeneity and
      completeness; coincides with NMI under arithmetic-mean normalisation.
    - ``homogeneity`` — each predicted cluster contains members of a single
      source class.
    - ``completeness`` — all members of a source class end up in the same
      predicted cluster. Fragmentation vs impurity decomposes NMI into
      these two complementary axes.
    """
    if len(X_src) < K_cl or len(Y_tgt) < K_cl:
        return dict(_AGREEMENT_NANS)
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    partners = T.argmax(axis=1)
    mapped = tgt_lab[partners]
    return {
        "nmi":          float(normalized_mutual_info_score(src_lab, mapped)),
        "ami":          float(adjusted_mutual_info_score(src_lab, mapped)),
        "ari":          float(adjusted_rand_score(src_lab, mapped)),
        "v_measure":    float(v_measure_score(src_lab, mapped)),
        "homogeneity":  float(homogeneity_score(src_lab, mapped)),
        "completeness": float(completeness_score(src_lab, mapped)),
    }


def cluster_nmi_ami(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> tuple[float, float]:
    """Back-compat wrapper returning (NMI, AMI) only."""
    a = cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)
    return a["nmi"], a["ami"]


def cluster_nmi(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> float:
    """Back-compat wrapper returning NMI only."""
    return cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)["nmi"]


def cluster_confusion(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
    mode: str = "soft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-cluster × target-cluster confusion matrix induced by the plan.

    - ``mode="soft"`` (default): row s, col t of the K × K matrix is the
      total mass T[i, j] aggregated over source rows i in source-cluster s
      and target columns j in target-cluster t. Uses the full plan, not
      argmax — informative for diffuse OT plans.
    - ``mode="hard"``: counts of source rows whose argmax partner falls
      into each target cluster. Equivalent to the hard-partner confusion
      used inside ``cluster_routing``.

    The returned matrix is row-normalised (each source-cluster row sums to
    one when non-empty); a row of zeros indicates an empty source cluster.
    Also returns the source and target K-means label vectors so callers
    can apply a Hungarian column permutation for block-diagonal display.
    """
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    C = np.zeros((K_cl, K_cl), dtype=np.float64)
    if mode == "soft":
        for s in range(K_cl):
            src_idx = np.where(src_lab == s)[0]
            if src_idx.size == 0:
                continue
            row_mass = T[src_idx].sum(axis=0)
            for t in range(K_cl):
                tgt_idx = np.where(tgt_lab == t)[0]
                if tgt_idx.size == 0:
                    continue
                C[s, t] = row_mass[tgt_idx].sum()
    elif mode == "hard":
        partners = T.argmax(axis=1)
        for s in range(K_cl):
            src_idx = np.where(src_lab == s)[0]
            for i in src_idx:
                C[s, tgt_lab[partners[i]]] += 1.0
    else:
        raise ValueError(f"unknown confusion mode: {mode!r}")
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    C = C / row_sums
    return C, src_lab, tgt_lab


def category_precision_at_k(
    T: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    K_cl: int,
    k: int,
    seed: int = 42,
) -> float:
    """Fraction of each query's top-$k$ retrievals that share the
    target-side K-means cluster of the query's GT partner. This is a
    per-query precision averaged over queries (proportion of correct-
    cluster items within the top-$k$ window), not a standard recall;
    the name was historically ``category_recall_at_k`` and has been
    renamed for clarity.

    A coarse-retrieval softening of $R@k$: ``R@k`` asks "is the exact
    target row in the top-$k$?", category-precision@k asks "are the
    top-$k$ in the *right neighbourhood* (same target cluster as the
    GT)?".
    """
    if Y_tgt.shape[0] < 2 or T.shape[1] < 1:
        return float("nan")
    Kc = int(min(K_cl, Y_tgt.shape[0]))
    if Kc < 2:
        return float("nan")
    tgt = KMeans(Kc, random_state=seed, n_init=10).fit_predict(Y_tgt)
    k_use = int(min(k, T.shape[1]))
    if k_use < 1:
        return float("nan")
    topk = np.argsort(-T, axis=1)[:, :k_use]
    gt_cluster = tgt[np.asarray(gt, dtype=int)]
    hits = (tgt[topk] == gt_cluster[:, None]).mean(axis=1)
    return float(hits.mean())


def evaluate(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    K_cl: int,
    seed: int = 42,
    Z_src_cap: np.ndarray | None = None,
    Z_tgt_cap: np.ndarray | None = None,
) -> dict:
    """Full metric suite. Used by run_experiments.py for each (K, alpha) cell.

    Optional ``Z_src_cap`` and ``Z_tgt_cap`` are the per-row caption
    embeddings of each side, used to compute the external-semantic
    caption-agreement scores (\\S\\ref{sec:res-classvis}). When omitted
    the caption-agreement columns of the return dict are NaN, preserving
    backwards compatibility with callers that pre-date this change.
    """
    r1 = recall_at_k(T, gt, 1)
    r5 = recall_at_k(T, gt, 5)
    r10 = recall_at_k(T, gt, 10)
    r20 = recall_at_k(T, gt, 20)
    r_correct, r_total = cluster_routing(T, X_src, Y_tgt, gt, K_cl, seed=seed)
    kno = knn_overlap(T, X_src, Y_tgt, k=5)
    pr = pearson_pairwise(T, X_src, Y_tgt)
    agree = cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)
    cap = caption_agreement(T, Z_src_cap, Z_tgt_cap, seed=seed)
    cat10 = category_precision_at_k(T, Y_tgt, gt, K_cl, k=10, seed=seed)
    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr,
        **agree,
        **cap,
        "cat_precision_10": cat10,
    }


def evaluate_heldout(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    S: np.ndarray,
    K_cl: int,
    seed: int = 42,
    Z_src_cap: np.ndarray | None = None,
    Z_tgt_cap: np.ndarray | None = None,
) -> dict:
    """Same suite but restricted to rows whose index is NOT in S.

    For recall_at_k this is a row mask: we keep the full plan (so the
    columns the held-out rows can match against are still the whole target
    set, which is the honest setup) but only score rows not in S.
    For structural metrics we recompute on the masked subset. Caption
    embeddings, when supplied, are likewise restricted to the held-out
    rows on the source side; the target side keeps all $n$ candidates.
    """
    n = T.shape[0]
    in_S = np.zeros(n, dtype=bool)
    in_S[np.asarray(S, dtype=int)] = True
    mask = ~in_S
    held = np.where(mask)[0]
    if held.size == 0:
        return {
            "R@1": float("nan"), "R@5": float("nan"),
            "R@10": float("nan"), "R@20": float("nan"),
            "routes_correct": 0, "routes_total": K_cl,
            "knn_overlap": float("nan"),
            "pearson_r": float("nan"),
            **_AGREEMENT_NANS,
            **_CAPAGREE_NANS,
            "cat_precision_10": float("nan"),
        }

    # Row-masked recall.
    T_h = T[held]
    gt_h = gt[held]
    r1 = recall_at_k(T_h, gt_h, 1)
    r5 = recall_at_k(T_h, gt_h, 5)
    r10 = recall_at_k(T_h, gt_h, 10)
    r20 = recall_at_k(T_h, gt_h, 20)

    # Structural metrics on the held-out subset.
    X_h = X_src[held]
    # gt[held] indexes into Y_tgt; we evaluate Y_tgt at the partner side.
    # For cluster_routing we need a remapping: cluster labels for X_h and Y_tgt's full set,
    # then aggregate plan mass restricted to held rows.
    Kc = min(K_cl, len(X_h), len(Y_tgt))
    if Kc < 2:
        r_correct, r_total = 0, K_cl
        kno = float("nan")
        pr = float("nan")
        agree = dict(_AGREEMENT_NANS)
    else:
        src_lab = KMeans(Kc, random_state=seed, n_init=10).fit_predict(X_h)
        tgt_lab = KMeans(Kc, random_state=seed, n_init=10).fit_predict(Y_tgt)
        partners_h = T_h.argmax(axis=1)

        correct = 0
        for s in range(Kc):
            idx = np.where(src_lab == s)[0]
            if len(idx) == 0:
                continue
            mass = np.zeros(Kc)
            for j in range(T.shape[1]):
                mass[tgt_lab[j]] += T_h[idx, j].sum()
            predicted = int(mass.argmax())
            gt_cluster = int(
                np.bincount(tgt_lab[gt_h[idx]], minlength=Kc).argmax()
            )
            if predicted == gt_cluster:
                correct += 1
        r_correct, r_total = correct, Kc

        # knn_overlap on held-out
        Ds = pairwise_distances(X_h)
        Dt = pairwise_distances(Y_tgt)
        k = min(5, len(X_h) - 1, len(Y_tgt) - 1)
        overlaps = []
        for i in range(len(X_h)):
            src_nn = np.argsort(Ds[i])[1:k + 1]
            tgt_nn = np.argsort(Dt[partners_h[i]])[1:k + 1]
            mapped = partners_h[src_nn]
            overlaps.append(len(set(mapped) & set(tgt_nn)) / k)
        kno = float(np.mean(overlaps)) if overlaps else float("nan")

        # pearson on held-out, partner side
        Dt_p = pairwise_distances(Y_tgt[partners_h])
        iu = np.triu_indices(len(X_h), k=1)
        a = Ds[iu]
        b = Dt_p[iu]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            pr = float("nan")
        else:
            pr = float(pearsonr(a, b)[0])

        mapped_h = tgt_lab[partners_h]
        agree = {
            "nmi":          float(normalized_mutual_info_score(src_lab, mapped_h)),
            "ami":          float(adjusted_mutual_info_score(src_lab, mapped_h)),
            "ari":          float(adjusted_rand_score(src_lab, mapped_h)),
            "v_measure":    float(v_measure_score(src_lab, mapped_h)),
            "homogeneity":  float(homogeneity_score(src_lab, mapped_h)),
            "completeness": float(completeness_score(src_lab, mapped_h)),
        }

    # Caption agreement on the held-out subset of source rows.
    if Z_src_cap is not None and Z_tgt_cap is not None:
        cap = caption_agreement(T_h, Z_src_cap[held], Z_tgt_cap, seed=seed)
    else:
        cap = dict(_CAPAGREE_NANS)

    # Coarse-retrieval analog of R@10 on the held-out rows. The target
    # K-means is computed on the *full* target pool (the candidate
    # set the held-out queries can retrieve into is all of Y_tgt).
    cat10 = category_precision_at_k(T_h, Y_tgt, gt_h, K_cl, k=10, seed=seed)

    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr,
        **agree,
        **cap,
        "cat_precision_10": cat10,
    }
