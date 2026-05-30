"""Cluster-stratified sample of AVCaps query clips for qualitative figures.

Picks N clips so that each one comes from a different k-means cluster of
the audio embeddings (CLAP-HTSAT-unfused by default). This ensures the
qualitative figure covers different acoustic content categories rather
than accidentally over-representing one (e.g. "car noises").

Output:
  data/qual_query_clips.txt   (one clip_id per line, ordered by cluster)

Usage:
    python code/sample_query_clips.py
    python code/sample_query_clips.py --n 5 --audio-encoder clap-larger
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="Number of query clips to sample.")
    ap.add_argument("--audio-encoder", default="clap-unfused",
                    help="Audio encoder used to build the stratification "
                         "clusters. The choice does not affect any "
                         "downstream retrieval; it only controls which "
                         "audio neighbourhoods are represented in the "
                         "qualitative panel.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(DATA / "qual_query_clips.txt"))
    args = ap.parse_args()

    Y_path = EMB / f"audio_{args.audio_encoder}.npy"
    if not Y_path.exists():
        raise SystemExit(f"[err] embeddings missing: {Y_path}")
    Y = np.load(Y_path)
    n_total = Y.shape[0]
    print(f"[sample] Y from {Y_path.name}, shape {Y.shape}")

    km = KMeans(n_clusters=args.n, random_state=args.seed, n_init=10)
    labels = km.fit_predict(Y)
    centroids = km.cluster_centers_
    chosen_rows: list[int] = []
    for c in range(args.n):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        d = np.linalg.norm(Y[members] - centroids[c], axis=1)
        chosen_rows.append(int(members[d.argmin()]))

    manifest_csv = DATA / "manifest.csv"
    clip_ids: list[str] = []
    with manifest_csv.open() as f:
        for r in csv.DictReader(f):
            clip_ids.append(r["clip_id"])
    if len(clip_ids) != n_total:
        raise SystemExit(
            f"[err] manifest has {len(clip_ids)} rows; "
            f"embeddings have {n_total}"
        )

    picked = [clip_ids[i] for i in chosen_rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picked) + "\n")
    print(f"[sample] wrote {out}")
    print(f"         clips: {picked}")


if __name__ == "__main__":
    main()
