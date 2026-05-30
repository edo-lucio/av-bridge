r"""C-transitive ablation: identity bridge instead of caption-cosine softmax.

The standard C-transitive composition is

    T = row-normalise( T_iv @ B @ T_ac.T )

where B = topk-softmax(Z_V Z_A^T / tau) is a learned bridge over caption
embeddings. The motivation was to match two independent caption pools
without ground-truth pairing.

In AVCaps we *do* have ground-truth pairing: row k of Z_V and row k of
Z_A describe the same clip. Even when the two captions describe
different aspects (visual vs sonic) of the same observation and
therefore sit far apart in MiniLM-space, they still refer to the same
clip k. The cosine bridge can be replaced by the identity bridge:

    T' = row-normalise( T_iv @ T_ac.T )

which routes purely on the shared caption-row index, independent of any
embedding-space similarity between Z_V[k] and Z_A[k].

This script is a one-off ablation: it does not touch
``exp_c_transitive``, ``exp_encoder_grid``, ``transitive_plan``, or any
of the existing artefacts. Output lands in
``results/exp_c_idbridge[__<img>__<aud>]/`` alongside (not replacing)
the cosine-bridge results in ``results/exp_c/``.

Usage:
  python code/ablation_transitive_idbridge.py
  python code/ablation_transitive_idbridge.py \
      --image-encoder dinov2-large --audio-encoder mert-330m
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from metrics import evaluate, evaluate_heldout  # noqa: E402
from run_experiments import (  # noqa: E402
    CSV_COLS,
    DEFAULT_AUDIO,
    DEFAULT_IMAGE,
    KCL,
    REUSABLE_ALPHA,
    REUSABLE_K,
    SEED,
    _suffix_for,
    kmeans_stratified_indices,
    load_anchors,
    load_embedding,
)

ROOT = CODE_DIR.parent
RES = ROOT / "results"


def identity_bridge_transitive(T_iv: np.ndarray,
                                T_ac: np.ndarray) -> np.ndarray:
    """Compose with no caption bridge: T' = row-norm(T_iv @ T_ac.T).

    Equivalent to ``transitive_plan(T_iv, I_n, T_ac)`` but does not
    construct the identity explicitly. Row-normalised so the output
    is a valid row-stochastic plan, matching the contract of
    ``algorithms.transitive_plan``.
    """
    raw = T_iv @ T_ac.T
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    return raw / row_sums


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-encoder", default=DEFAULT_IMAGE)
    ap.add_argument("--audio-encoder", default=DEFAULT_AUDIO)
    args = ap.parse_args()

    img_enc, aud_enc = args.image_encoder, args.audio_encoder
    print(f"[Ablation C-transitive (identity bridge)] "
          f"image={img_enc}  audio={aud_enc}")

    a_dir = RES / f"exp_a{_suffix_for(DEFAULT_IMAGE, img_enc)}"
    b_dir = RES / f"exp_b{_suffix_for(DEFAULT_AUDIO, aud_enc)}"
    t_iv_path = a_dir / "T_iv.npy"
    t_ac_path = b_dir / "T_ac.npy"
    if not t_iv_path.exists():
        raise FileNotFoundError(
            f"missing within-image plan: {t_iv_path}  "
            f"(run Exp A at image={img_enc} first)"
        )
    if not t_ac_path.exists():
        raise FileNotFoundError(
            f"missing within-audio plan: {t_ac_path}  "
            f"(run Exp B at audio={aud_enc} first)"
        )
    T_iv = np.load(t_iv_path)
    T_ac = np.load(t_ac_path)
    print(f"  loaded T_iv shape={T_iv.shape}  from {t_iv_path}")
    print(f"  loaded T_ac shape={T_ac.shape}  from {t_ac_path}")

    X_image = load_embedding(f"vision_{img_enc}")
    Y_audio = load_embedding(f"audio_{aud_enc}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["c-transitive"]

    print("  composing T = row-norm(T_iv @ T_ac.T)  (identity bridge)")
    T = identity_bridge_transitive(T_iv, T_ac)
    print(f"  composed plan shape={T.shape}  sum={T.sum():.6f}")

    suffix = ""
    if img_enc != DEFAULT_IMAGE or aud_enc != DEFAULT_AUDIO:
        suffix = f"__{img_enc}__{aud_enc}"
    out_dir = RES / f"exp_c_idbridge{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # Same-rows partition matching D / unsup / text / cosine-bridge
    # C-transitive: rows outside the union of A's and B's K=300 anchor
    # sets, for direct comparison with the other heldout rows.
    S_a = kmeans_stratified_indices(
        X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(
        Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED,
                   Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)
    hel = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED,
                           Z_src_cap=ZV_a, Z_tgt_cap=ZA_a)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        # K = REUSABLE_K and alpha = REUSABLE_ALPHA so the row keys
        # match the cosine-bridge sweep_transitive.csv for direct
        # head-to-head comparison; bridge_top_k / bridge_tau are not
        # applicable here, so they're absent from the CSV row.
        w.writerow({"K": REUSABLE_K, "alpha": REUSABLE_ALPHA,
                    "scope": "aggregate", **agg})
        w.writerow({"K": REUSABLE_K, "alpha": REUSABLE_ALPHA,
                    "scope": "heldout", **hel})

    np.save(out_dir / "T_transitive_idbridge.npy", T)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_transitive_idbridge.npy'}")

    cosine_csv = (RES / f"exp_c{suffix}" / "sweep_transitive.csv")
    if cosine_csv.exists():
        import pandas as pd
        cos = pd.read_csv(cosine_csv)
        print()
        print("  --- side-by-side vs cosine-bridge ---")
        for scope, label in [("aggregate", "aggregate"),
                             ("heldout", "heldout")]:
            cos_row = cos[cos.scope == scope]
            if cos_row.empty:
                cos_row = cos[cos.scope == label]
            if cos_row.empty:
                continue
            cr = cos_row.iloc[0]
            new = agg if "agg" in scope else hel
            print(f"  {scope:18s}  "
                  f"R@10  cos={cr['R@10']:.3f}  id={new['R@10']:.3f}    "
                  f"AMI  cos={cr['ami']:.3f}  id={new['ami']:.3f}    "
                  f"r    cos={cr['pearson_r']:.3f}  id={new['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
