"""Phase 4 — core algorithms.

Implements the semi-supervised optimal-transport recipe (ridge cross-space
projection + entropic Sinkhorn / Fused Gromov-Wasserstein) plus the text
bridge used in Experiment C-transitive.

The functions below assume embedding matrices are already L2-normalised.
"""
from __future__ import annotations

import numpy as np
import ot


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, memory-efficient.

    For L2-normalised rows this equals 2 - 2 * A @ B.T, but we use the
    general identity so non-normalised inputs (e.g. ridge-projected ones)
    also work.
    """
    A2 = np.einsum("ij,ij->i", A, A)[:, None]
    B2 = np.einsum("ij,ij->i", B, B)[None, :]
    D = A2 + B2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    return D


def ridge_project(
    X: np.ndarray,
    Y: np.ndarray,
    S_src: np.ndarray,
    S_tgt: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Closed-form ridge map X[S_src] -> Y[S_tgt]: returns W of shape (d_src, d_tgt)."""
    X_S = X[S_src]
    Y_S = Y[S_tgt]
    d = X.shape[1]
    W = np.linalg.solve(X_S.T @ X_S + lam * np.eye(d), X_S.T @ Y_S)
    return W


def sinkhorn(M: np.ndarray, eps: float = 0.005, num_iter: int = 2000) -> np.ndarray:
    """Entropic OT plan with uniform marginals."""
    n, m = M.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    # numItermax keeps high-precision convergence at small eps.
    T = ot.sinkhorn(a, b, M, reg=eps, numItermax=num_iter, stopThr=1e-9)
    return np.asarray(T)


def fgw(
    M: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    alpha: float = 0.7,
    eps: float = 0.005,
    num_iter: int = 2000,
) -> np.ndarray:
    """Entropic Fused Gromov-Wasserstein with uniform marginals.

    alpha = 0  -> pure Sinkhorn on M
    alpha = 1  -> pure Gromov-Wasserstein on (C1, C2)
    """
    n, m = M.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M, C1=C1, C2=C2, p=a, q=b,
        loss_fun="square_loss",
        alpha=alpha, epsilon=eps,
        max_iter=num_iter, tol=1e-9, log=False, verbose=False,
    )
    return np.asarray(T)


def recipe(
    X: np.ndarray,
    Y: np.ndarray,
    S_src: np.ndarray,
    S_tgt: np.ndarray,
    alpha: float = 0.7,
    eps: float = 0.005,
    lam: float = 1.0,
) -> np.ndarray:
    """Full semi-supervised cross-modal recipe.

    1. Closed-form ridge cross-space projection from X[S_src] to Y[S_tgt].
    2. Project all source rows into target space, renormalise.
    3. Build normalised cross-modal cost matrix M.
    4. FGW (alpha > 0) or Sinkhorn (alpha == 0).
    """
    W = ridge_project(X, Y, S_src, S_tgt, lam=lam)

    X_proj = X @ W
    norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
    # Guard against zero rows (very unlikely after ridge but be safe).
    norms = np.where(norms > 1e-12, norms, 1.0)
    X_proj = X_proj / norms

    M = _pairwise_sqdist(X_proj, Y)
    M_max = float(M.max())
    if M_max > 0:
        M = M / M_max

    if alpha > 0:
        C1 = _pairwise_sqdist(X, X)
        c1_max = float(C1.max())
        if c1_max > 0:
            C1 = C1 / c1_max
        C2 = _pairwise_sqdist(Y, Y)
        c2_max = float(C2.max())
        if c2_max > 0:
            C2 = C2 / c2_max
        return fgw(M, C1, C2, alpha=alpha, eps=eps)
    return sinkhorn(M, eps=eps)


def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
    S_src: np.ndarray,
    S_tgt: np.ndarray,
) -> np.ndarray:
    """Closed-form (semi-)orthogonal Procrustes alignment from X to Y.

    Given paired rows (X[S_src], Y[S_tgt]) returns W with shape
    (d_src, d_tgt) that minimises ||X[S_src] W - Y[S_tgt]||_F^2
    subject to W being semi-orthogonal (W^T W = I when d_src >= d_tgt,
    W W^T = I when d_src < d_tgt). Solved via the SVD of the
    cross-covariance matrix; see Schoenemann (1966).

    The semi-orthogonal generalisation lets us align embedding spaces
    of different dimensionality (e.g. CLIP-large at 768 vs CLAP-unfused
    at 512) without padding or PCA preprocessing.
    """
    X_S = X[S_src]
    Y_S = Y[S_tgt]
    M = X_S.T @ Y_S
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return W


def procrustes_recipe(
    X: np.ndarray,
    Y: np.ndarray,
    S_src: np.ndarray,
    S_tgt: np.ndarray,
) -> np.ndarray:
    """Procrustes baseline: rigid (orthogonal) supervised alignment.

    1. Fit a semi-orthogonal map W: source-space -> target-space from
       the paired anchors (S_src, S_tgt).
    2. Project all source rows X -> X W and renormalise rows to unit
       length so the output sits on the target-space hypersphere
       alongside Y.
    3. Return the (n, m) similarity matrix (X W) Y^T, in the same
       shape contract as ``text_only_retrieval``: the metric suite
       treats it as a transport plan via row-wise argmax / argsort.

    Compared to ``recipe`` (ridge + Sinkhorn/FGW), Procrustes is the
    strict isometry restriction: any structural distortion of X under
    the alignment is forbidden. It is the natural "supervised
    structural" baseline -- the same constraint Pure-GW imposes
    (preserve pairwise distances) but with identity-paired supervision
    instead of unsupervised geometric matching.
    """
    W = procrustes_align(X, Y, S_src, S_tgt)
    X_proj = X @ W
    norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    X_proj = X_proj / norms
    return X_proj @ Y.T


def ridge_recipe(
    X: np.ndarray,
    Y: np.ndarray,
    S_src: np.ndarray,
    S_tgt: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Direct ridge baseline: flexible (unconstrained linear) supervised
    image -> audio alignment.

    1. Fit a closed-form ridge map W: source-space -> target-space from
       the paired anchors (S_src, S_tgt) -- the same ``ridge_project`` the
       FGW recipe uses, but applied *directly* between the two modalities.
    2. Project all source rows X -> X W, renormalise to the unit
       hypersphere alongside Y.
    3. Return the (n, m) cosine-similarity matrix (X W) Y^T, consumed by
       the metric suite via row-wise argmax / argsort.

    This is the flexible-linear counterpart to ``procrustes_recipe``:
    Procrustes restricts W to a rotation (preserves geometry, forbids
    rescaling); ridge allows arbitrary linear distortion. Comparing the
    two isolates whether retrieval needs a distortion-allowing map. As the
    simplest method that uses the true image-audio pairs, it is the
    supervised reference ceiling the text-mediated methods are measured
    against.
    """
    W = ridge_project(X, Y, S_src, S_tgt, lam=lam)
    X_proj = X @ W
    norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    X_proj = X_proj / norms
    return X_proj @ Y.T


def random_baseline(n: int, m: int, seed: int = 0) -> np.ndarray:
    """Uniform random row-stochastic plan -- the "no information" floor.

    Each row is a Dirichlet-like random distribution over targets, so the
    argmax of each row is essentially uniformly random over the m
    target rows. Used as a baseline against which every other recipe's
    above-chance behaviour can be measured.

    Expected metrics under this plan:
      R@k         ~ k / m
      AMI / ARI   ~ 0 (cluster partition agreement at chance)
      Pearson r   ~ 0 (no pairwise-distance preservation)
      cap_cos_lift ~ 0 (no caption-similarity signal)
    """
    rng = np.random.default_rng(seed)
    T = rng.uniform(size=(n, m))
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    return T / row_sums


def text_only_retrieval(ZV: np.ndarray, ZA: np.ndarray) -> np.ndarray:
    """Naive text-caption retrieval baseline (no OT, no FGW).

    Returns the n x m cosine-similarity matrix between the two caption
    pools, to be consumed by the standard metric suite as if it were a
    transport plan: argmax / argsort over rows yields top-k retrieval
    by raw caption similarity, with image and audio embeddings entirely
    bypassed. Inputs are assumed L2-normalised so dot products equal
    cosine similarities.
    """
    return ZV @ ZA.T


def pure_gw(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 0.005,
    num_iter: int = 2000,
) -> np.ndarray:
    """Pure entropic Gromov-Wasserstein with uniform marginals.

    No feature term, no ridge, no anchors: the plan is determined entirely
    by the alignment of the two intra-modal distance geometries C1, C2.
    Inputs are assumed L2-normalised row-wise.
    """
    n, m = X.shape[0], Y.shape[0]
    p = np.full(n, 1.0 / n)
    q = np.full(m, 1.0 / m)
    C1 = _pairwise_sqdist(X, X)
    c1_max = float(C1.max())
    if c1_max > 0:
        C1 = C1 / c1_max
    C2 = _pairwise_sqdist(Y, Y)
    c2_max = float(C2.max())
    if c2_max > 0:
        C2 = C2 / c2_max
    T = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q,
        loss_fun="square_loss",
        epsilon=eps,
        max_iter=num_iter, tol=1e-9, log=False, verbose=False,
    )
    return np.asarray(T)


def caption_cost_recipe(
    X: np.ndarray,
    Y: np.ndarray,
    ZV: np.ndarray,
    ZA: np.ndarray,
    alpha: float = 0.7,
    eps: float = 0.005,
) -> np.ndarray:
    """Caption-cost FGW recipe (Experiment D).

    No ridge, no anchors, no image-audio supervision: the cross-modal
    feature term M is built directly from caption distances.

    M[i, j]  = || ZV[i] - ZA[j] ||^2  / max
    C1[i, j] = || X[i]  - X[j]  ||^2  / max
    C2[i, j] = || Y[i]  - Y[j]  ||^2  / max
    T        = FGW(M, C1, C2, alpha)  (or Sinkhorn(M) when alpha == 0)

    All input matrices are assumed L2-normalised row-wise.
    """
    M = _pairwise_sqdist(ZV, ZA)
    m_max = float(M.max())
    if m_max > 0:
        M = M / m_max

    if alpha > 0:
        C1 = _pairwise_sqdist(X, X)
        c1_max = float(C1.max())
        if c1_max > 0:
            C1 = C1 / c1_max
        C2 = _pairwise_sqdist(Y, Y)
        c2_max = float(C2.max())
        if c2_max > 0:
            C2 = C2 / c2_max
        return fgw(M, C1, C2, alpha=alpha, eps=eps)
    return sinkhorn(M, eps=eps)


def build_bridge(
    ZV: np.ndarray,
    ZA: np.ndarray,
    top_k: int = 20,
    tau: float = 0.1,
) -> np.ndarray:
    """Sparse softmax text bridge.

    Each row of B is a top-k softmax (temperature tau) distribution over
    audio-caption rows for a given visual-caption row.

    Assumes both inputs are L2-normalised so ZV @ ZA.T is cosine.
    """
    sims = ZV @ ZA.T
    B = np.zeros_like(sims)
    k = min(top_k, sims.shape[1])
    for j in range(sims.shape[0]):
        topk = np.argpartition(-sims[j], k - 1)[:k]
        logits = sims[j, topk] / tau
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        B[j, topk] = w
    return B


def transitive_plan(T_iv: np.ndarray, B: np.ndarray, T_ac: np.ndarray) -> np.ndarray:
    """Compose image->visual-text, text bridge, audio->audio-text plans.

    Returns a row-normalised image -> audio plan of shape (n, n).
    Kept as a utility; the main C-transitive recipe now uses the
    identity bridge via ``transitive_plan_identity`` below.
    """
    raw = T_iv @ B @ T_ac.T
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    return raw / row_sums


def transitive_plan_identity(T_iv: np.ndarray, T_ac: np.ndarray) -> np.ndarray:
    """Compose image->visual-text and audio->audio-text plans through the
    ground-truth caption-row pairing (identity bridge).

    AVCaps provides paired captions per clip: row k of Z_V and row k of
    Z_A describe the same underlying clip. The composition therefore
    routes purely on the shared caption-row index k:

        T = row-normalise( T_iv @ T_ac.T )

    Equivalent to ``transitive_plan(T_iv, I_n, T_ac)`` but skips the
    identity-matrix multiply.
    """
    raw = T_iv @ T_ac.T
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    return raw / row_sums
