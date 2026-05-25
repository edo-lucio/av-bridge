"""Generate transport plans for the best-NMI encoder pair per experiment.

The encoder-ablation grid (Phase 8g) only records *metrics* per cell — it
does not persist the transport plans themselves. To produce qualitative
figures for the encoder pairs that score best on NMI (the headline
class-level-alignment metric), we have to re-fit the plans for those
specific pairs.

This script:

  1. Reads `results/exp_grid/sweep.csv`.
  2. For each experiment in {c-direct, d, unsup, text}, finds the
     (image_encoder, audio_encoder) cell with the highest NMI at the
     experiment's evaluation scope (held-out for C-direct, same-rows
     for D / Unsup / Text).
  3. Re-fits the plan for that pair using the canonical operating point
     of the experiment and saves it under `results/qualitative_best/`.
  4. Builds a K-means-stratified sample of N query clips
     (default 12) using the D-best image encoder, for visual diversity.
  5. Writes `results/qualitative_best/manifest.json` linking each plan
     to its encoders and the clip sample.

Usage (on HPC, where `embeddings/` is populated):

    python code/render_best_plans.py
    python code/render_best_plans.py --n-clips 16 --metric ami

Output layout::

    results/qualitative_best/
      manifest.json
      T_d__clip-base__mert-95m.npy
      T_unsup__clip-base__mert-330m.npy
      T_text__clip-large__mert-95m.npy
      T_cdirect__clip-large__clap-larger.npy
      clips.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _path_relative_to_root(p: Path) -> str:
    """Return p as a string relative to ROOT when possible, absolute
    otherwise. Avoids the `is not in the subpath of` ValueError that
    pathlib.relative_to raises when paths don't share an ancestor."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
RES = ROOT / "results"
EMB = ROOT / "embeddings"

sys.path.insert(0, str(ROOT / "code"))
from algorithms import caption_cost_recipe, pure_gw, recipe, text_only_retrieval
from run_experiments import (
    KCL,
    REUSABLE_K,
    SEED,
    kmeans_stratified_indices,
    load_anchors,
    load_embedding,
)


# Per-experiment scope used to rank cells. C-direct's grid has a true
# held-out partition; the others have a same-rows (`heldout`)
# view that puts D / Unsup / Text on the same 100-row reference.
EXP_SCOPE = {
    "c-direct": "heldout",
    "d":        "heldout",
    "unsup":    "heldout",
    "text":     "heldout",
}

# Canonical operating point of each experiment.
EXP_ALPHA = {"c-direct": 0.5, "d": 0.7, "unsup": 1.0, "text": float("nan")}


# ------------------------- best-cell selection ----------------------------
def pick_pair_for(df: pd.DataFrame, exp: str, scope: str,
                  metric: str, rank: str = "best",
                  ) -> tuple[str, str, float] | None:
    """Return (image_encoder, audio_encoder, metric_value) for the
    highest-metric (rank='best') or lowest-metric (rank='worst') cell of
    the given experiment / scope, or None if the grid does not include
    that experiment."""
    sub = df[(df["experiment"] == exp) & (df["scope"] == scope)].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return None
    ascending = (rank == "worst")
    row = sub.sort_values(metric, ascending=ascending).iloc[0]
    return str(row["image_encoder"]), str(row["audio_encoder"]), float(row[metric])


# Back-compat alias for code that imports `best_pair_for` directly.
def best_pair_for(df: pd.DataFrame, exp: str, scope: str, metric: str):
    return pick_pair_for(df, exp, scope, metric, rank="best")


# ------------------------- plan builders ----------------------------------
def _load_pair(img: str, aud: str) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Load X, Y, Z_vis, Z_aud, anchored to the canonical n = 400 row order
    (full identity at n = 400)."""
    X_full = load_embedding(f"vision_{img}")
    Y_full = load_embedding(f"audio_{aud}")
    ZV_full = load_embedding("ZV_text")
    ZA_full = load_embedding("ZA_text")
    anchors = load_anchors(X_full, n=X_full.shape[0])
    return X_full[anchors], Y_full[anchors], ZV_full[anchors], ZA_full[anchors]


def build_plan(exp: str, img: str, aud: str) -> np.ndarray:
    """Re-fit the transport plan for one experiment at the best-NMI pair."""
    X, Y, ZV, ZA = _load_pair(img, aud)
    if exp == "c-direct":
        S = kmeans_stratified_indices(
            X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED,
        )
        return recipe(X, Y, S, S, alpha=EXP_ALPHA["c-direct"],
                      eps=0.005, lam=1.0)
    if exp == "d":
        return caption_cost_recipe(X, Y, ZV, ZA,
                                   alpha=EXP_ALPHA["d"], eps=0.005)
    if exp == "unsup":
        return pure_gw(X, Y, eps=0.005)
    if exp == "text":
        return text_only_retrieval(ZV, ZA)
    raise ValueError(f"unknown experiment: {exp!r}")


# ------------------------- stratified clip sample -------------------------
def stratified_clip_sample(image_encoder: str, n: int = 12,
                           seed: int = SEED) -> list[str]:
    """Pick `n` clips that diversify the visual content space.

    Uses the same `kmeans_stratified_indices` helper that drives the
    held-out anchor set, on the image encoder of the D-best pair. Returns
    a list of clip_ids (strings) in increasing order.
    """
    X = load_embedding(f"vision_{image_encoder}")
    idx = kmeans_stratified_indices(X, n=n,
                                    n_clusters=min(10, n), seed=seed)
    idx = sorted(int(i) for i in idx)

    # Cross-reference against the manifest to get clip_ids in row order.
    import csv as _csv
    rows: list[str] = []
    with (ROOT / "data" / "manifest.csv").open() as f:
        for r in _csv.DictReader(f):
            rows.append(r["clip_id"])
    return [rows[i] for i in idx]


# ------------------------- driver -----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-csv", type=str,
                    default=str(RES / "exp_grid" / "sweep.csv"),
                    help="Long-form grid CSV produced by run_experiments.")
    ap.add_argument("--metric", type=str, default="nmi",
                    choices=["nmi", "ami", "ari", "v_measure",
                             "homogeneity", "completeness", "R@10"],
                    help="Column used to rank cells per experiment.")
    ap.add_argument("--n-clips", type=int, default=12,
                    help="Number of query clips to render figures for.")
    ap.add_argument("--rank", type=str, default="best",
                    choices=["best", "worst", "both"],
                    help="'best' = highest-metric pair per experiment, "
                         "'worst' = lowest, 'both' = render plans for both "
                         "ends so the qualitative comparison spans the "
                         "encoder-quality spectrum.")
    ap.add_argument("--out-dir", type=str,
                    default=str(RES / "qualitative_best"),
                    help="Directory for the generated plans and manifest.")
    args = ap.parse_args()

    grid_csv = Path(args.grid_csv)
    if not grid_csv.exists():
        sys.exit(f"[err] grid CSV not found: {grid_csv}. "
                 "Run `python code/run_experiments.py --exp grid` first.")
    df = pd.read_csv(grid_csv)
    if args.metric not in df.columns:
        sys.exit(
            f"[err] metric {args.metric!r} not present in {grid_csv}. "
            "Re-run the grid after adding it to the metric suite."
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = ["best", "worst"] if args.rank == "both" else [args.rank]

    plans_meta: list[dict] = []
    for rank in ranks:
        for exp in ("c-direct", "d", "unsup", "text"):
            scope = EXP_SCOPE[exp]
            pick = pick_pair_for(df, exp, scope, args.metric, rank=rank)
            if pick is None:
                print(f"[skip] {exp}/{rank}: no rows in {grid_csv} "
                      f"for scope={scope}")
                continue
            img, aud, val = pick
            print(f"[{rank:5s}] {exp:8s} scope={scope:14s} "
                  f"by {args.metric}={val:.4f}  -> ({img}, {aud})")

            T = build_plan(exp, img, aud)
            plan_name = (
                f"T_{exp.replace('-', '_')}_{rank}__{img}__{aud}.npy"
            )
            plan_path = out_dir / plan_name
            np.save(plan_path, T)
            print(f"          wrote {plan_path}  shape={T.shape}  "
                  f"sum={T.sum():.3f}")

            plans_meta.append({
                "experiment": exp,
                "rank": rank,
                "image_encoder": img,
                "audio_encoder": aud,
                "metric_used": args.metric,
                "metric_value": val,
                "scope": scope,
                "plan_path": _path_relative_to_root(plan_path),
            })

    # Stratified clip sample driven by the D-best image encoder (independent
    # of --rank: we want the *content diversity* of the query set, not
    # encoder-quality alignment). If --rank=worst was the only request, fall
    # back to any D entry; if none exist, default to clip-large.
    d_pick = next(
        (m for m in plans_meta
         if m["experiment"] == "d" and m["rank"] == "best"),
        None,
    )
    if d_pick is None:
        d_pick = next(
            (m for m in plans_meta if m["experiment"] == "d"), None,
        )
    ref_img = d_pick["image_encoder"] if d_pick is not None else "clip-large"
    clips = stratified_clip_sample(ref_img, n=args.n_clips)
    clips_path = out_dir / "clips.txt"
    clips_path.write_text(",".join(clips))
    print(f"[clips] wrote {clips_path}  ({len(clips)} clips, "
          f"stratified by vision_{ref_img})")

    manifest = {
        "metric_used": args.metric,
        "rank_requested": args.rank,
        "ref_image_encoder_for_clip_sample": ref_img,
        "clips": clips,
        "plans": plans_meta,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
