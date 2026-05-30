"""Phase 3 + 6 + 7 + 8 — anchor sampling and the experiments.

Usage:
  python code/run_experiments.py --exp a
  python code/run_experiments.py --exp b
  python code/run_experiments.py --exp c-direct
  python code/run_experiments.py --exp c-transitive
  python code/run_experiments.py --exp d

Encoders default to clip-large + clap-unfused (matching the spec's CLIP/CLAP
spirit). Override with --image-encoder / --audio-encoder.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from algorithms import (
    caption_cost_recipe,
    procrustes_recipe,
    pure_gw,
    random_baseline,
    recipe,
    ridge_recipe,
    text_only_retrieval,
    transitive_plan_identity,
)
from metrics import evaluate, evaluate_heldout

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
RES = ROOT / "results"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

K_GRID = [10, 20, 50, 100, 160, 200, 300]
ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

KCL = {
    "a": 10, "b": 20,
    "c-transitive": 15,
    "d": 15, "unsup": 15, "text": 15,
    "random": 15, "procrustes": 15, "direct": 15, "cmcr": 15,
}

# Rescaled from spec's 350 for n=400.
REUSABLE_K = 300
REUSABLE_ALPHA = 0.7

CSV_COLS = [
    "K", "alpha", "scope",
    "R@1", "R@5", "R@10", "R@20",
    "routes_correct", "routes_total",
    "knn_overlap", "pearson_r",
    "nmi", "ami", "ari", "v_measure", "homogeneity", "completeness",
    "cap_cos_argmax", "cap_cos_planmass",
    "cap_cos_chance", "cap_cos_identity", "cap_cos_lift",
    "cat_precision_10",
    "plan_row_entropy",
]


def plan_row_entropy(T: np.ndarray) -> float:
    """Mean row entropy of a coupling, in nats.

    Diagnostic for plan sharpness: argmax-style sharp plans have low
    row entropy; uniform plans have entropy log(n_cols). Used to
    visualise the smoothing effect of FGW's structural term as alpha
    increases.
    """
    if T.size == 0:
        return float("nan")
    row_sums = T.sum(axis=1, keepdims=True)
    P = T / np.maximum(row_sums, 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(axis=1)
    return float(H.mean())


def kmeans_stratified_indices(
    X: np.ndarray,
    n: int,
    n_clusters: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Pick n indices by walking KMeans clusters round-robin, taking the
    cluster centre's nearest unused member each turn.
    """
    n = int(min(n, X.shape[0]))
    if n == X.shape[0]:
        return np.arange(n)
    n_clusters = int(min(n_clusters, n))
    n_clusters = max(n_clusters, 1)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    chosen: list[int] = []
    taken: set[int] = set()
    for round_idx in range(n * 4):
        if len(chosen) == n:
            break
        c = round_idx % n_clusters
        members = [i for i in np.where(km.labels_ == c)[0] if i not in taken]
        if not members:
            continue
        dists = np.linalg.norm(
            X[members] - km.cluster_centers_[c], axis=1
        )
        pick = members[int(np.argmin(dists))]
        chosen.append(pick)
        taken.add(pick)
    return np.array(chosen, dtype=int)


def load_anchors(X_image: np.ndarray, n: int = 400) -> np.ndarray:
    path = EMB / "anchor_indices.npy"
    if path.exists():
        return np.load(path)
    idx = kmeans_stratified_indices(X_image, n=n, n_clusters=min(10, n), seed=SEED)
    np.save(path, idx)
    return idx


def load_embedding(name: str) -> np.ndarray:
    """Loads a saved L2-normalised embedding by file stem."""
    path = EMB / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding: {path}")
    return np.load(path).astype(np.float32)


def select_for_anchors(M: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return M[anchors]


def fmt_cell(metrics: dict, K_cl: int) -> str:
    r10 = metrics.get("R@10", float("nan"))
    rc = metrics.get("routes_correct", 0)
    return f"{r10:.3f} / {rc}/{K_cl}"


def run_sweep(
    X: np.ndarray,
    Y: np.ndarray,
    K_cl: int,
    csv_path: Path,
    save_plan_at: tuple[int, float] | None = None,
    plan_path: Path | None = None,
    K_grid: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    Z_src_cap: np.ndarray | None = None,
    Z_tgt_cap: np.ndarray | None = None,
) -> None:
    """Run the (K, alpha) sweep, evaluate, append rows to csv_path.

    ``Z_src_cap`` / ``Z_tgt_cap`` are optional per-row caption embeddings
    used to compute the external caption-agreement metric. When omitted
    those columns of the output CSV land as NaN; the rest of the metric
    suite is unaffected.
    """
    n = X.shape[0]
    gt = np.arange(n)
    K_grid = K_grid or K_GRID
    alpha_grid = alpha_grid or ALPHA_GRID

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()

        for K in K_grid:
            if K > n:
                continue
            S = kmeans_stratified_indices(
                X, n=K, n_clusters=min(10, K), seed=SEED
            )
            for alpha in alpha_grid:
                print(f"  K={K:4d}  alpha={alpha:.2f}")
                T = recipe(X, Y, S, S, alpha=alpha, eps=0.005, lam=1.0)
                ent = plan_row_entropy(T)

                agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                               Z_src_cap=Z_src_cap, Z_tgt_cap=Z_tgt_cap)
                w.writerow({"K": K, "alpha": alpha, "scope": "aggregate",
                            **agg, "plan_row_entropy": ent})

                hel = evaluate_heldout(T, X, Y, gt, S, K_cl, seed=SEED,
                                       Z_src_cap=Z_src_cap, Z_tgt_cap=Z_tgt_cap)
                w.writerow({"K": K, "alpha": alpha, "scope": "heldout",
                            **hel, "plan_row_entropy": ent})

                if (save_plan_at is not None
                        and (K, alpha) == save_plan_at
                        and plan_path is not None):
                    np.save(plan_path, T)
                    print(f"    saved plan -> {plan_path.name}")
                f.flush()


DEFAULT_IMAGE = "clip-large"
DEFAULT_AUDIO = "clap-unfused"


def _suffix_for(default: str, picked: str) -> str:
    """Empty when running at the default encoder; otherwise '__<encoder>'."""
    return "" if picked == default else f"__{picked}"


def exp_a(image_name: str, K_grid: list[int] | None = None) -> None:
    print(f"[Exp A] image={image_name}  target=visual captions")
    X_image = load_embedding(f"vision_{image_name}")
    ZV = load_embedding("ZV_text")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = ZV[anchors]
    out_dir = RES / f"exp_a{_suffix_for(DEFAULT_IMAGE, image_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_sweep(
        X, Y,
        K_cl=KCL["a"],
        csv_path=out_dir / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=out_dir / "T_iv.npy",
        K_grid=K_grid,
    )


def exp_b(audio_name: str, K_grid: list[int] | None = None) -> None:
    print(f"[Exp B] audio={audio_name}  target=audio captions")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZA = load_embedding("ZA_text")
    # Anchor ordering is shared across image-audio experiments by anchoring
    # to the canonical vision embedding, so B's row order matches A's.
    X_image = load_embedding(f"vision_clip-large")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = Y_audio[anchors]
    Y = ZA[anchors]
    out_dir = RES / f"exp_b{_suffix_for(DEFAULT_AUDIO, audio_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_sweep(
        X, Y,
        K_cl=KCL["b"],
        csv_path=out_dir / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=out_dir / "T_ac.npy",
        K_grid=K_grid,
    )


def exp_c_transitive(image_name: str, audio_name: str,
                     K_grid: list[int] | None = None,
                     alpha_grid: list[float] | None = None,
                     out_suffix: str = "") -> None:
    """Identity-bridge C-transitive sweep across (K, alpha) for both legs.

    The two within-modality plans are refit at every (K, alpha) cell
    rather than being loaded from disk -- this lets us sweep them
    independently of Experiments A and B's saved canonical plan.
    The cross-modal bridge is the ground-truth caption-row identity:
        T = row-norm(T_iv @ T_ac.T)
    (See ``algorithms.transitive_plan_identity``.)

    Both legs share the same (K, alpha) per cell. Each leg uses its
    own modality-stratified k-means anchor partition. The plan at the
    canonical reusable point (K=REUSABLE_K, alpha=REUSABLE_ALPHA) is
    saved to ``T_transitive.npy`` for downstream consumers.
    """
    print(f"[Exp C-transitive (identity bridge)] "
          f"image={image_name}  audio={audio_name}")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]
    n = X.shape[0]
    gt = np.arange(n)

    K_cl = KCL["c-transitive"]
    K_grid = K_grid or K_GRID
    alpha_grid = alpha_grid or ALPHA_GRID

    # Same-rows held-out partition matches D / unsup / text / random
    # at the REUSABLE_K budget, so the held-out rows are comparable to
    # every other cross-modal recipe in the chapter regardless of the
    # current sweep cell's K.
    S_a_compare = kmeans_stratified_indices(
        X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b_compare = kmeans_stratified_indices(
        Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a_compare, S_b_compare]))

    out_dir = RES / f"exp_c{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_transitive.csv"

    # Persist the heldout-row subset (rows whose i is in S_compare are the
    # "anchor union" rows; the remaining 100 are the heldout slice).
    # Downstream metric scripts can recover the heldout split without
    # access to the image / audio embeddings.
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for K in K_grid:
            if K > n:
                continue
            # The legs are independent supervised problems, so each picks
            # its own stratification on its own source manifold.
            S_iv = kmeans_stratified_indices(
                X, n=K, n_clusters=min(10, K), seed=SEED)
            S_ac = kmeans_stratified_indices(
                Y, n=K, n_clusters=min(10, K), seed=SEED)
            for alpha in alpha_grid:
                print(f"  K={K:4d}  alpha={alpha:.2f}  fitting legs...")
                T_iv = recipe(X, ZV_a, S_iv, S_iv,
                              alpha=alpha, eps=0.005, lam=1.0)
                T_ac = recipe(Y, ZA_a, S_ac, S_ac,
                              alpha=alpha, eps=0.005, lam=1.0)
                T = transitive_plan_identity(T_iv, T_ac)
                ent = plan_row_entropy(T)

                agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                               Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
                hel = evaluate_heldout(T, X, Y, gt, S_compare, K_cl,
                                       seed=SEED,
                                       Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
                w.writerow({"K": K, "alpha": alpha,
                            "scope": "aggregate", **agg,
                            "plan_row_entropy": ent})
                w.writerow({"K": K, "alpha": alpha,
                            "scope": "heldout", **hel,
                            "plan_row_entropy": ent})
                f.flush()

                # Persist every (K, alpha) plan under plans/ so the
                # core-set metric (and any other per-cell analysis) can
                # be run after the fact without re-fitting.
                np.save(
                    plans_dir / f"T__K{K}__a{alpha:.2f}.npy",
                    T,
                )
                # Also persist the two leg plans per (K, alpha) for
                # downstream leg-vs-composition diagnostics (e.g.
                # top-K leg agreement, leg row-entropy comparisons).
                # Cheap on disk; eliminates the need for HPC re-fitting
                # when adding new analyses that compare the leg plans
                # at different alpha values.
                np.save(
                    plans_dir / f"T_iv__K{K}__a{alpha:.2f}.npy",
                    T_iv,
                )
                np.save(
                    plans_dir / f"T_ac__K{K}__a{alpha:.2f}.npy",
                    T_ac,
                )

                # Persist the plan at the canonical reusable point so
                # downstream tools (qualitative renders, ranks, etc.)
                # find the same filename they did before the swap.
                if K == REUSABLE_K and abs(alpha - REUSABLE_ALPHA) < 1e-9:
                    np.save(out_dir / "T_transitive.npy", T)
                    print(f"    saved plan -> T_transitive.npy")

    print(f"  wrote {csv_path}")
    print(f"  wrote per-cell plans -> {plans_dir}/")
    print(f"  wrote heldout subset -> {out_dir / 'heldout_compare_idx.npy'}")


def exp_d_caption(image_name: str, audio_name: str,
                  alpha_grid: list[float] | None = None,
                  out_suffix: str = "") -> None:
    """Experiment D — caption-cost FGW for image -> audio.

    No anchors, no ridge: M is built directly from caption distances
    (ZV vs ZA) and FGW combines it with the intra-modal structural
    costs of the image and audio embeddings.
    """
    print(f"[Exp D] image={image_name}  audio={audio_name}  "
          f"M = caption sqdist (ZV vs ZA)")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV = ZV[anchors]
    ZA = ZA[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["d"]
    alpha_grid = alpha_grid or ALPHA_GRID

    out_dir = RES / f"exp_d{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # "Same-rows view" partition for comparability with C-transitive's held-out:
    # rows lying outside the union of A's and B's K=REUSABLE_K anchor sets.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    np.save(out_dir / "heldout_compare_idx.npy", S_compare)

    saved_plan = None
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for alpha in alpha_grid:
            print(f"  alpha={alpha:.2f}")
            T = caption_cost_recipe(X, Y, ZV, ZA, alpha=alpha, eps=0.005)
            ent = plan_row_entropy(T)
            agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                           Z_src_cap=ZV, Z_tgt_cap=ZA)
            # K=0 sentinel: no anchors / no ridge in this experiment.
            w.writerow({"K": 0, "alpha": alpha, "scope": "aggregate", **agg,
                        "plan_row_entropy": ent})

            # Same-rows "held-out-like" evaluation at EVERY alpha so
            # downstream plotters (transitive K-sweep, full-vs-heldout)
            # can pick the right alpha. For D specifically the
            # aggregate/heldout distinction is sampling noise because D
            # has no anchors, but we still need the row to exist at
            # every alpha for the plot's scope-consistent comparison.
            cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                                   Z_src_cap=ZV, Z_tgt_cap=ZA)
            w.writerow({"K": 0, "alpha": alpha, "scope": "heldout", **cmp,
                        "plan_row_entropy": ent})
            if abs(alpha - REUSABLE_ALPHA) < 1e-12:
                saved_plan = T

            np.save(plans_dir / f"T__a{alpha:.2f}.npy", T)

            f.flush()

    if saved_plan is not None:
        np.save(out_dir / "T_caption.npy", saved_plan)
        print(f"  wrote {csv_path}")
        print(f"  wrote {out_dir / 'T_caption.npy'}")
        print(f"  wrote per-alpha plans -> {plans_dir}/")
        print(f"  wrote heldout subset -> {out_dir / 'heldout_compare_idx.npy'}")


def exp_unsupervised_gw(image_name: str, audio_name: str,
                        out_suffix: str = "") -> None:
    """Fully unsupervised image -> audio alignment with pure entropic GW.

    No anchors, no ridge, no captions: the plan is determined entirely by
    the alignment of the image and audio intra-modal distance geometries.
    This is the lower-bound reference (Regime 2 of the methodology) for
    every other image-audio variant.
    """
    print(f"[Exp unsup-GW] image={image_name}  audio={audio_name}  "
          f"M = none, pure entropic GW(C1, C2)")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    # Caption embeddings are loaded only to enable the caption-agreement
    # evaluation downstream; they are *not* fed into pure_gw(), so the
    # recipe itself remains text-blind.
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["unsup"]

    print("  solving entropic GW...")
    T = pure_gw(X, Y, eps=0.005)
    print(f"  plan shape={T.shape}  sum={T.sum():.6f}")

    out_dir = RES / f"exp_unsup{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # Same-rows view: 100 rows outside the union of A's and B's K=300
    # anchor sets, for direct comparison with C-direct held-out,
    # C-transitive held-out, and Experiment D same-rows.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                   Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
    cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        ent = plan_row_entropy(T)
        w.writerow({"K": 0, "alpha": 1.0, "scope": "aggregate", **agg,
                    "plan_row_entropy": ent})
        w.writerow({"K": 0, "alpha": 1.0, "scope": "heldout", **cmp,
                    "plan_row_entropy": ent})

    np.save(out_dir / "T_gw.npy", T)
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_gw.npy'}")
    print(f"  wrote {out_dir / 'heldout_compare_idx.npy'}")


def exp_text_only(image_name: str, audio_name: str,
                  out_suffix: str = "") -> None:
    """Pure text-caption retrieval baseline for image -> audio.

    No OT, no FGW, no image or audio embeddings at all. The "plan" is
    the raw cosine-similarity matrix between the visual-caption and
    audio-caption pools, fed to the same metric suite as the other
    experiments so the numbers are directly comparable.

    The image_name and audio_name arguments are accepted only so that
    the structural metrics (kNN overlap, Pearson r, cluster NMI) can
    be computed on the same X, Y as the other image-audio experiments;
    they do NOT enter the construction of the plan itself.
    """
    print(f"[Exp text-only] retrieval = cosine(ZV, ZA) directly  (no OT)")

    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")
    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    ZV = ZV[anchors]
    ZA = ZA[anchors]
    X = X_image[anchors]
    Y = Y_audio[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["text"]

    T = text_only_retrieval(ZV, ZA)
    print(f"  similarity matrix shape={T.shape}  range=[{T.min():.3f},{T.max():.3f}]")

    out_dir = RES / f"exp_text{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # Same-rows view: rows outside the union of A's and B's K=300 anchor
    # sets, matching every other image-audio variant for direct comparison.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                   Z_src_cap=ZV, Z_tgt_cap=ZA)
    cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                           Z_src_cap=ZV, Z_tgt_cap=ZA)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        ent = plan_row_entropy(T)
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "aggregate",
                    **agg, "plan_row_entropy": ent})
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "heldout",
                    **cmp, "plan_row_entropy": ent})

    np.save(out_dir / "T_text.npy", T)
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_text.npy'}")
    print(f"  wrote {out_dir / 'heldout_compare_idx.npy'}")


def exp_random(image_name: str, audio_name: str,
               out_suffix: str = "") -> None:
    """Random baseline: a uniform row-stochastic plan, no information.

    Every other recipe should beat this on every metric; whatever
    quantity it does not beat random on is, by definition, a quantity
    in which the recipe carries no above-chance signal."""
    print(f"[Exp Random] image={image_name}  audio={audio_name}  "
          f"uniform random row-stochastic plan")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["random"]

    T = random_baseline(X.shape[0], Y.shape[0], seed=SEED)
    print(f"  plan shape={T.shape}  row sums in "
          f"[{T.sum(axis=1).min():.6f}, {T.sum(axis=1).max():.6f}]")

    out_dir = RES / f"exp_random{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    S_a = kmeans_stratified_indices(X, n=REUSABLE_K,
                                    n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K,
                                    n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                   Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
    hel = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        ent = plan_row_entropy(T)
        w.writerow({"K": 0, "alpha": float("nan"),
                    "scope": "aggregate", **agg,
                    "plan_row_entropy": ent})
        w.writerow({"K": 0, "alpha": float("nan"),
                    "scope": "heldout", **hel,
                    "plan_row_entropy": ent})

    np.save(out_dir / "T_random.npy", T)
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_random.npy'}")
    print(f"  wrote {out_dir / 'heldout_compare_idx.npy'}")


def exp_cmcr(image_name: str, audio_name: str,
             out_suffix: str = "") -> None:
    """C-MCR baseline (Wang et al., NeurIPS 2023): learned text bridge
    connecting CLIP and CLAP for image -> audio retrieval without any
    image-audio pairs. End-to-end pretrained, using C-MCR's own encoders.

    The plan is the cosine-similarity matrix between C-MCR's shared-space
    image and audio embeddings (produced by ``code/encode_cmcr.py`` in a
    separate env). Like exp_text, the canonical X / Y / captions are passed
    to ``evaluate`` only so the structural and category metrics are computed
    on the same rows as every other recipe; they do not enter the plan.

    Text-grounded only: C-MCR connects two text-contrastive encoders, so it
    has no text-free analogue.
    """
    print(f"[Exp C-MCR] learned text bridge (CLIP x CLAP), end-to-end pretrained")
    try:
        Vc = load_embedding("cmcr_vision")
        Ac = load_embedding("cmcr_audio")
    except FileNotFoundError:
        print("  ! skip: embeddings/cmcr_{vision,audio}.npy not found. "
              "Run code/encode_cmcr.py in the C-MCR env first.")
        return

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    Vc = Vc[anchors]
    Ac = Ac[anchors]
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]
    n = X.shape[0]
    gt = np.arange(n)
    K_cl = KCL["cmcr"]

    # Defensive re-normalise, then cosine plan image -> audio in C-MCR space.
    Vc = Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), 1e-12)
    Ac = Ac / np.maximum(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-12)
    T = Vc @ Ac.T
    print(f"  plan shape={T.shape}  range=[{T.min():.3f},{T.max():.3f}]")

    out_dir = RES / f"exp_cmcr{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED, Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
    cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        ent = plan_row_entropy(T)
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "aggregate",
                    **agg, "plan_row_entropy": ent})
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "heldout",
                    **cmp, "plan_row_entropy": ent})

    np.save(out_dir / "T_cmcr.npy", T)
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_cmcr.npy'}")


def exp_direct(image_name: str, audio_name: str,
               K_grid: list[int] | None = None,
               out_suffix: str = "") -> None:
    """Direct ridge comparison: flexible (unconstrained linear) supervised
    image -> audio alignment fit on the true clip-identity pairs.

    The simplest method that uses the image-audio supervision the bridge
    forgoes: a ridge map X -> Y, projected and retrieved by cosine. It is
    the supervised reference ceiling, and the flexible-linear counterpart
    to Procrustes (which restricts the map to a rotation). K-swept, no
    alpha; canonical plan saved to ``T_direct.npy``.
    """
    print(f"[Exp Direct (ridge supervised image->audio)] "
          f"image={image_name}  audio={audio_name}")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]
    n = X.shape[0]
    gt = np.arange(n)

    K_cl = KCL["direct"]
    K_grid = K_grid or K_GRID

    S_a_compare = kmeans_stratified_indices(
        X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b_compare = kmeans_stratified_indices(
        Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a_compare, S_b_compare]))

    out_dir = RES / f"exp_direct{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for K in K_grid:
            if K > n:
                continue
            S = kmeans_stratified_indices(
                X, n=K, n_clusters=min(10, K), seed=SEED)
            print(f"  K={K:4d}  ridge fit + cosine retrieval...")
            T = ridge_recipe(X, Y, S, S, lam=1.0)
            ent = plan_row_entropy(T)
            agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                                   Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
            w.writerow({"K": K, "alpha": float("nan"), "scope": "aggregate",
                        **agg, "plan_row_entropy": ent})
            w.writerow({"K": K, "alpha": float("nan"), "scope": "heldout",
                        **hel, "plan_row_entropy": ent})
            f.flush()
            if K == REUSABLE_K:
                np.save(out_dir / "T_direct.npy", T)
                print("    saved plan -> T_direct.npy")

    print(f"  wrote {csv_path}")
    print(f"  wrote heldout subset -> {out_dir / 'heldout_compare_idx.npy'}")


def exp_procrustes(image_name: str, audio_name: str,
                   K_grid: list[int] | None = None,
                   out_suffix: str = "") -> None:
    """Procrustes comparison: rigid (semi-orthogonal) supervised image ->
    audio alignment fit on the true clip-identity pairs.

    A single rotation (+ reflection) maps the image hypersphere onto the
    audio one, with no distortion allowed -- the classical test of whether
    two representation spaces coincide up to a rotation. Supervised (uses
    paired anchors), K-swept, no alpha; written with the alpha=NaN sentinel
    like the other non-FGW references (Text-only, Random). The canonical
    plan is saved to ``T_procrustes.npy`` for the comparison plots.
    """
    print(f"[Exp Procrustes (rigid supervised image->audio)] "
          f"image={image_name}  audio={audio_name}")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]
    n = X.shape[0]
    gt = np.arange(n)

    K_cl = KCL["procrustes"]
    K_grid = K_grid or K_GRID

    S_a_compare = kmeans_stratified_indices(
        X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b_compare = kmeans_stratified_indices(
        Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a_compare, S_b_compare]))

    out_dir = RES / f"exp_procrustes{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    np.save(out_dir / "heldout_compare_idx.npy", S_compare)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for K in K_grid:
            if K > n:
                continue
            S = kmeans_stratified_indices(
                X, n=K, n_clusters=min(10, K), seed=SEED)
            print(f"  K={K:4d}  rigid Procrustes fit...")
            T = procrustes_recipe(X, Y, S, S)
            ent = plan_row_entropy(T)
            agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                                   Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
            w.writerow({"K": K, "alpha": float("nan"), "scope": "aggregate",
                        **agg, "plan_row_entropy": ent})
            w.writerow({"K": K, "alpha": float("nan"), "scope": "heldout",
                        **hel, "plan_row_entropy": ent})
            f.flush()
            if K == REUSABLE_K:
                np.save(out_dir / "T_procrustes.npy", T)
                print("    saved plan -> T_procrustes.npy")

    print(f"  wrote {csv_path}")
    print(f"  wrote heldout subset -> {out_dir / 'heldout_compare_idx.npy'}")


def _discover_encoders() -> tuple[list[str], list[str]]:
    """List all (image, audio) encoder names that have a saved embedding."""
    vision = sorted(p.stem.replace("vision_", "") for p in EMB.glob("vision_*.npy"))
    audio = sorted(p.stem.replace("audio_", "") for p in EMB.glob("audio_*.npy"))
    return vision, audio


GRID_CSV_COLS = [
    "experiment", "image_encoder", "audio_encoder",
    "K", "alpha", "scope",
    "R@1", "R@5", "R@10", "R@20",
    "routes_correct", "routes_total",
    "knn_overlap", "pearson_r",
    "nmi", "ami", "ari", "v_measure", "homogeneity", "completeness",
    "cap_cos_argmax", "cap_cos_planmass",
    "cap_cos_chance", "cap_cos_identity", "cap_cos_lift",
    "cat_precision_10",
]


def exp_encoder_grid(
    image_only: list[str] | None = None,
    audio_only: list[str] | None = None,
) -> None:
    """Canonical-operating-point ablation across the full encoder registry.

    For each combination of available image and audio encoders, run
        - C-direct  at (K = REUSABLE_K, alpha = 0.5),
        - D         at alpha = 0.7,
        - Unsup     (pure entropic GW),
        - Text      (raw caption cosine).
    For each image encoder run A at the canonical (K, alpha); same for B.
    Append every row to results/exp_grid/sweep.csv in long form.
    """
    vision_all, audio_all = _discover_encoders()
    vision = image_only if image_only else vision_all
    audio = audio_only if audio_only else audio_all
    print(f"[grid] vision={vision}")
    print(f"[grid] audio ={audio}")

    ZV_full = load_embedding("ZV_text")
    ZA_full = load_embedding("ZA_text")
    K_target = REUSABLE_K
    alpha_iv = 0.5
    alpha_d  = 0.7
    eps = 0.005

    out_dir = RES / "exp_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.csv"

    def write_rows(rows: list[dict]) -> None:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GRID_CSV_COLS)
            w.writeheader()
            w.writerows(rows)

    rows: list[dict] = []

    # Cache within-modality plans for C-transitive composition below.
    t_iv_by_image: dict[str, np.ndarray] = {}
    t_ac_by_audio: dict[str, np.ndarray] = {}

    for img in vision:
        try:
            X_full = load_embedding(f"vision_{img}")
        except FileNotFoundError as e:
            print(f"  ! skip A on {img}: {e}")
            continue
        anchors = load_anchors(X_full, n=X_full.shape[0])
        X = X_full[anchors]
        Y = ZV_full[anchors]
        S = kmeans_stratified_indices(X, n=K_target,
                                      n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])
        print(f"  [A] image={img}")
        T = recipe(X, Y, S, S, alpha=alpha_iv, eps=eps, lam=1.0)
        t_iv_by_image[img] = T
        for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["a"], seed=SEED)),
                         ("heldout",   evaluate_heldout(T, X, Y, gt, S, KCL["a"], seed=SEED))]:
            rows.append({"experiment": "a", "image_encoder": img, "audio_encoder": "—",
                         "K": K_target, "alpha": alpha_iv, "scope": scope, **m})

    for aud in audio:
        try:
            Y_full = load_embedding(f"audio_{aud}")
        except FileNotFoundError as e:
            print(f"  ! skip B on {aud}: {e}")
            continue
        # Same anchor ordering as A so B's rows line up with the other
        # experiments; fall back to identity if no vision encoder is on disk.
        anchor_ref = vision[0] if vision else None
        if anchor_ref is not None:
            anchors = load_anchors(load_embedding(f"vision_{anchor_ref}"),
                                   n=Y_full.shape[0])
        else:
            anchors = np.arange(Y_full.shape[0])
        X = Y_full[anchors]
        Y = ZA_full[anchors]
        S = kmeans_stratified_indices(X, n=K_target,
                                      n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])
        print(f"  [B] audio={aud}")
        T = recipe(X, Y, S, S, alpha=alpha_iv, eps=eps, lam=1.0)
        t_ac_by_audio[aud] = T
        for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["b"], seed=SEED)),
                         ("heldout",   evaluate_heldout(T, X, Y, gt, S, KCL["b"], seed=SEED))]:
            rows.append({"experiment": "b", "image_encoder": "—", "audio_encoder": aud,
                         "K": K_target, "alpha": alpha_iv, "scope": scope, **m})

    for img in vision:
        try:
            X_full = load_embedding(f"vision_{img}")
        except FileNotFoundError:
            continue
        anchors = load_anchors(X_full, n=X_full.shape[0])
        X = X_full[anchors]
        ZV = ZV_full[anchors]
        ZA = ZA_full[anchors]
        S_a = kmeans_stratified_indices(X, n=K_target,
                                        n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])

        for aud in audio:
            try:
                Y_full = load_embedding(f"audio_{aud}")
            except FileNotFoundError as e:
                print(f"  ! skip cross on (image={img}, audio={aud}): {e}")
                continue
            Y = Y_full[anchors]
            S_b = kmeans_stratified_indices(Y, n=K_target,
                                            n_clusters=min(10, K_target), seed=SEED)
            S_compare = np.unique(np.concatenate([S_a, S_b]))
            print(f"  [grid] image={img}  audio={aud}")

            cap_kw = {"Z_src_cap": ZV, "Z_tgt_cap": ZA}

            T_iv = t_iv_by_image.get(img)
            T_ac = t_ac_by_audio.get(aud)
            if T_iv is not None and T_ac is not None:
                T = transitive_plan_identity(T_iv, T_ac)
                for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["c-transitive"], seed=SEED, **cap_kw)),
                                 ("heldout",   evaluate_heldout(T, X, Y, gt, S_compare,
                                                                KCL["c-transitive"], seed=SEED, **cap_kw))]:
                    # Force the held-out scope label so this row aligns with
                    # d / unsup / text held-out partitioning (rows outside
                    # S_a U S_b) rather than the within-modality scope name.
                    scope_out = "heldout" if scope == "heldout" else scope
                    rows.append({"experiment": "c-transitive",
                                 "image_encoder": img, "audio_encoder": aud,
                                 "K": K_target, "alpha": alpha_iv,
                                 "scope": scope_out, **m})

            T = caption_cost_recipe(X, Y, ZV, ZA, alpha=alpha_d, eps=eps)
            agg = evaluate(T, X, Y, gt, KCL["d"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["d"], seed=SEED, **cap_kw)
            rows.append({"experiment": "d", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": alpha_d, "scope": "aggregate", **agg})
            rows.append({"experiment": "d", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": alpha_d, "scope": "heldout", **hel})

            T = pure_gw(X, Y, eps=eps)
            agg = evaluate(T, X, Y, gt, KCL["unsup"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["unsup"], seed=SEED, **cap_kw)
            rows.append({"experiment": "unsup", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": 1.0, "scope": "aggregate", **agg})
            rows.append({"experiment": "unsup", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": 1.0, "scope": "heldout", **hel})

            # Text-only: the plan itself is encoder-independent, but
            # structural metrics depend on Y, so we evaluate per cell.
            T = text_only_retrieval(ZV, ZA)
            agg = evaluate(T, X, Y, gt, KCL["text"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["text"], seed=SEED, **cap_kw)
            rows.append({"experiment": "text", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "aggregate", **agg})
            rows.append({"experiment": "text", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "heldout", **hel})

            T = random_baseline(X.shape[0], Y.shape[0], seed=SEED)
            agg = evaluate(T, X, Y, gt, KCL["random"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["random"], seed=SEED, **cap_kw)
            rows.append({"experiment": "random", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "aggregate", **agg})
            rows.append({"experiment": "random", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "heldout", **hel})

            T = procrustes_recipe(X, Y, S_a, S_a)
            agg = evaluate(T, X, Y, gt, KCL["procrustes"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["procrustes"], seed=SEED, **cap_kw)
            rows.append({"experiment": "procrustes", "image_encoder": img, "audio_encoder": aud,
                         "K": REUSABLE_K, "alpha": float("nan"), "scope": "aggregate", **agg})
            rows.append({"experiment": "procrustes", "image_encoder": img, "audio_encoder": aud,
                         "K": REUSABLE_K, "alpha": float("nan"), "scope": "heldout", **hel})

            T = ridge_recipe(X, Y, S_a, S_a, lam=1.0)
            agg = evaluate(T, X, Y, gt, KCL["direct"], seed=SEED, **cap_kw)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["direct"], seed=SEED, **cap_kw)
            rows.append({"experiment": "direct", "image_encoder": img, "audio_encoder": aud,
                         "K": REUSABLE_K, "alpha": float("nan"), "scope": "aggregate", **agg})
            rows.append({"experiment": "direct", "image_encoder": img, "audio_encoder": aud,
                         "K": REUSABLE_K, "alpha": float("nan"), "scope": "heldout", **hel})

            # Persist after every cell so a crash doesn't lose hours of work.
            write_rows(rows)

    write_rows(rows)
    print(f"  wrote {out_path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        choices=["a", "b", "c-transitive",
                 "d", "unsup", "text", "random", "procrustes", "direct",
                 "cmcr", "grid", "all"],
        required=True,
    )
    ap.add_argument("--grid-images", type=str, default=None,
                    help="Comma-separated subset of image encoders for --exp grid")
    ap.add_argument("--grid-audios", type=str, default=None,
                    help="Comma-separated subset of audio encoders for --exp grid")
    ap.add_argument("--image-encoder", default="clip-large")
    ap.add_argument("--audio-encoder", default="clap-unfused")
    ap.add_argument("--K-grid", type=str, default=None,
                    help="Override K sweep, e.g. '50,100,200,300,400'")
    ap.add_argument("--alpha-grid", type=str, default=None,
                    help="Override alpha sweep, e.g. '0.0,0.3,0.5,0.7,0.9'")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Append this suffix to per-experiment output "
                         "directories (e.g. '__dinov2-large__mert-330m'). "
                         "Leaves the canonical-pair artefacts intact when "
                         "running additional encoder pairs.")
    args = ap.parse_args()

    K_grid = (
        [int(x) for x in args.K_grid.split(",")]
        if args.K_grid else None
    )
    alpha_grid = (
        [float(x) for x in args.alpha_grid.split(",")]
        if args.alpha_grid else None
    )

    if args.exp in ("a", "all"):
        exp_a(args.image_encoder, K_grid=K_grid)
    if args.exp in ("b", "all"):
        exp_b(args.audio_encoder, K_grid=K_grid)
    if args.exp in ("c-transitive", "all"):
        exp_c_transitive(
            args.image_encoder, args.audio_encoder,
            K_grid=K_grid, alpha_grid=alpha_grid,
            out_suffix=args.out_suffix,
        )
    if args.exp in ("d", "all"):
        exp_d_caption(args.image_encoder, args.audio_encoder,
                      out_suffix=args.out_suffix)
    if args.exp in ("unsup", "all"):
        exp_unsupervised_gw(args.image_encoder, args.audio_encoder,
                            out_suffix=args.out_suffix)
    if args.exp in ("text", "all"):
        exp_text_only(args.image_encoder, args.audio_encoder,
                      out_suffix=args.out_suffix)
    if args.exp in ("random", "all"):
        exp_random(args.image_encoder, args.audio_encoder,
                   out_suffix=args.out_suffix)
    if args.exp in ("procrustes", "all"):
        exp_procrustes(args.image_encoder, args.audio_encoder,
                       K_grid=K_grid, out_suffix=args.out_suffix)
    if args.exp in ("direct", "all"):
        exp_direct(args.image_encoder, args.audio_encoder,
                   K_grid=K_grid, out_suffix=args.out_suffix)
    if args.exp in ("cmcr", "all"):
        # Skips gracefully if embeddings/cmcr_*.npy are absent (i.e. until
        # code/encode_cmcr.py has been run in the C-MCR env).
        exp_cmcr(args.image_encoder, args.audio_encoder,
                 out_suffix=args.out_suffix)
    if args.exp == "grid":
        image_only = (args.grid_images.split(",")
                      if args.grid_images else None)
        audio_only = (args.grid_audios.split(",")
                      if args.grid_audios else None)
        exp_encoder_grid(image_only=image_only, audio_only=audio_only)


if __name__ == "__main__":
    main()
