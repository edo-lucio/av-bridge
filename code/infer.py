"""Inference pipeline for image -> audio retrieval via a frozen transport plan.

Implements the four-step query procedure of methodology.tex § 5:

  1. Embed the query (external image file, or look up an existing AVCaps clip).
  2. Soft-assign the query to source anchors by top-k cosine + softmax.
  3. Push through the plan: w_tgt = w_src^T @ T.
  4. Smooth (optional) and rank target anchors by w_final.

Defaults to results/exp_d/T_caption.npy (Experiment D, the best unsupervised
image-audio recipe in the repo).

Usage:
    python code/infer.py --query path/to/image.jpg
    python code/infer.py --query-clip 10001787725
    python code/infer.py --eval-same-rows
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"
RES = ROOT / "results"
MANIFEST = DATA / "manifest.csv"
AUDIO = DATA / "audio"
DEFAULT_PLAN = RES / "exp_d" / "T_caption.npy"

# Reusable-K from run_experiments.py — needed to reconstruct the same-rows
# partition for eval mode.
REUSABLE_K = 300


# ---------- helpers --------------------------------------------------------
def load_manifest_clip_ids() -> list[str]:
    """Return clip_ids in the same order the embeddings are saved in."""
    rows: list[str] = []
    with MANIFEST.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r["clip_id"])
    return rows


def load_plan(path: Path) -> np.ndarray:
    """Load a transport plan and print basic diagnostics."""
    T = np.load(path)
    n_rows, n_cols = T.shape
    row_sums = T.sum(axis=1)
    print(f"[plan] {path.name}  shape={T.shape}  total_sum={T.sum():.4f}")
    print(f"        row-sum range=[{row_sums.min():.4f}, {row_sums.max():.4f}]  "
          f"sparsity={(T <= 0).mean():.3f}")
    return T


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def four_step_push(v_query: np.ndarray, X_anchors: np.ndarray, T: np.ndarray,
                   query_top_k: int, query_temp: float, smooth: float) -> np.ndarray:
    """Methodology §5: soft-assign + push + smooth. Returns w_final of shape (m,).

    `v_query` must be L2-normalised and live in the same space as the rows of
    `X_anchors`. `T` is row-indexed by the same anchors.
    """
    n = X_anchors.shape[0]
    m = T.shape[1]
    if v_query.shape[0] != X_anchors.shape[1]:
        raise ValueError(
            f"query dimension {v_query.shape[0]} doesn't match anchor "
            f"dimension {X_anchors.shape[1]}; did you use the right "
            f"--image-encoder for this plan?"
        )
    sims = X_anchors @ v_query
    k = min(max(1, query_top_k), n)
    if k == 1:
        top_idx = np.array([int(sims.argmax())], dtype=np.int64)
        weights = np.ones(1, dtype=np.float64)
    else:
        top_idx = np.argpartition(-sims, k - 1)[:k]
        logits = sims[top_idx] / max(query_temp, 1e-12)
        weights = softmax(logits.astype(np.float64))
    w_src = np.zeros(n, dtype=np.float64)
    w_src[top_idx] = weights
    w_tgt = w_src @ T
    if smooth < 1.0:
        return smooth * w_tgt + (1.0 - smooth) / m
    return w_tgt


def encode_external_image(path: Path, encoder_name: str) -> np.ndarray:
    """Encode a single external image with the chosen vision encoder.

    Reuses VISION_ENCODERS / VISION_DISPATCH from code/encode.py: we feed
    the dispatch function a one-item manifest where `frame_path` is the
    absolute path of the query image.
    """
    sys.path.insert(0, str(ROOT / "code"))
    from encode import VISION_ENCODERS, VISION_DISPATCH, l2_normalise
    if encoder_name not in VISION_ENCODERS:
        raise KeyError(
            f"unknown image encoder: {encoder_name!r}; "
            f"options: {list(VISION_ENCODERS)}"
        )
    cfg = VISION_ENCODERS[encoder_name]
    fn = VISION_DISPATCH[cfg["type"]]
    fake_manifest = [{"frame_path": str(Path(path).resolve())}]
    feats = fn(cfg["hf_id"], fake_manifest)
    return l2_normalise(feats)[0]


def print_topk(indices: np.ndarray, scores: np.ndarray,
               clip_ids: list[str]) -> None:
    print(f"{'rank':>4}  {'clip_id':>14}  {'score':>10}  audio_path")
    for rank, (i, s) in enumerate(zip(indices, scores), start=1):
        cid = clip_ids[int(i)]
        ap = AUDIO / f"{cid}.wav"
        print(f"{rank:>4}  {cid:>14}  {s:>10.6f}  {ap}")


def write_csv(out_path: Path, header: list[str], rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {out_path}")


# ---------- single-query mode ---------------------------------------------
def mode_single(args: argparse.Namespace) -> None:
    clip_ids = load_manifest_clip_ids()
    n = len(clip_ids)
    plan = load_plan(Path(args.plan))
    if plan.shape != (n, n):
        print(f"[warn] plan shape {plan.shape} != ({n}, {n}); proceeding, "
              f"but row/column indices may not align with the manifest.")

    X = np.load(EMB / f"vision_{args.image_encoder}.npy")
    if X.shape[0] != n:
        raise RuntimeError(
            f"vision_{args.image_encoder}.npy has {X.shape[0]} rows, "
            f"manifest has {n}"
        )

    if args.query:
        v = encode_external_image(Path(args.query), args.image_encoder)
        print(f"[query] external image: {args.query}")
    else:
        if args.query_clip not in clip_ids:
            raise KeyError(
                f"clip_id {args.query_clip!r} not in manifest "
                f"(first few: {clip_ids[:3]} ...)"
            )
        i = clip_ids.index(args.query_clip)
        v = X[i]
        print(f"[query] clip_id={args.query_clip} (row {i})")

    w_final = four_step_push(v, X, plan,
                             args.query_top_k, args.query_temp, args.smooth)
    order = np.argsort(-w_final)[: args.top_k]
    scores = w_final[order]
    print_topk(order, scores, clip_ids)

    if args.out:
        rows = [
            {
                "rank": r + 1,
                "clip_id": clip_ids[int(i)],
                "score": float(s),
                "audio_path": str(AUDIO / f"{clip_ids[int(i)]}.wav"),
            }
            for r, (i, s) in enumerate(zip(order, scores))
        ]
        write_csv(Path(args.out),
                  ["rank", "clip_id", "score", "audio_path"], rows)


# ---------- eval-same-rows mode -------------------------------------------
def _kmeans_partition(X: np.ndarray, K: int) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "code"))
    from run_experiments import kmeans_stratified_indices
    return kmeans_stratified_indices(X, n=K,
                                     n_clusters=min(10, K), seed=42)


def mode_eval_same_rows(args: argparse.Namespace) -> None:
    clip_ids = load_manifest_clip_ids()
    n = len(clip_ids)
    plan = load_plan(Path(args.plan))
    X = np.load(EMB / f"vision_{args.image_encoder}.npy")
    Y = np.load(EMB / f"audio_{args.audio_encoder}.npy")

    # Reconstruct the same-rows partition used by Experiment D's
    # heldout scope and Experiment C-transitive's held-out.
    S_a = _kmeans_partition(X, REUSABLE_K)
    S_b = _kmeans_partition(Y, REUSABLE_K)
    S_union = np.unique(np.concatenate([S_a, S_b]))
    in_S = np.zeros(n, dtype=bool)
    in_S[S_union] = True
    held = np.where(~in_S)[0]
    print(f"[eval] same-rows partition: {len(held)} clips of {n}")

    gt = np.arange(n)
    hits = {k: 0 for k in (1, 5, 10, 20)}
    per_clip = []
    for i in held:
        w_final = four_step_push(X[i], X, plan,
                                 args.query_top_k, args.query_temp,
                                 args.smooth)
        ranked = np.argsort(-w_final)
        gt_pos = int(np.where(ranked == gt[i])[0][0])
        for k in hits:
            if gt_pos < k:
                hits[k] += 1
        per_clip.append({"clip_id": clip_ids[i],
                         "gt_rank": gt_pos,
                         "top1_clip": clip_ids[int(ranked[0])]})

    print(f"\n[eval] same-rows R@k  (held={len(held)},  "
          f"query_top_k={args.query_top_k},  query_temp={args.query_temp})")
    for k in (1, 5, 10, 20):
        print(f"  R@{k:<3} = {hits[k] / max(1, len(held)):.4f}")

    if args.out:
        write_csv(Path(args.out),
                  ["clip_id", "gt_rank", "top1_clip"], per_clip)


# ---------- CLI -----------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Image -> audio inference via a frozen transport plan."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", type=str,
                      help="External image path (encoded at runtime).")
    mode.add_argument("--query-clip", type=str,
                      help="Existing AVCaps clip_id (embedding looked up).")
    mode.add_argument("--eval-same-rows", action="store_true",
                      help="Reproduce R@k on the 100-row same-rows partition.")
    ap.add_argument("--plan", type=str, default=str(DEFAULT_PLAN),
                    help="Path to transport plan .npy (default: D's plan).")
    ap.add_argument("--image-encoder", type=str, default="clip-large",
                    help="Vision encoder for the source side / external queries.")
    ap.add_argument("--audio-encoder", type=str, default="clap-unfused",
                    help="Audio encoder; only used to rebuild the eval partition.")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of audio clips to return per query.")
    ap.add_argument("--query-top-k", type=int, default=20,
                    help="Number of source anchors for soft-assign.")
    ap.add_argument("--query-temp", type=float, default=0.05,
                    help="Softmax temperature for the soft-assign step.")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Smoothing weight t in step 4 (1.0 = no smoothing).")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional CSV output path for batch / eval logs.")
    args = ap.parse_args()

    if args.eval_same_rows:
        mode_eval_same_rows(args)
    else:
        mode_single(args)


if __name__ == "__main__":
    main()
