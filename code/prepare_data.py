"""Phase 1 — extract middle frame + 48 kHz mono audio, build manifest.csv.

For each video in data/captions/{val,test}.json, locate the matching .mp4
under val_videos/ or test_videos/, use ffmpeg to extract:

  - middle frame (single JPEG) -> data/frames/{clip_id}.jpg
  - mono 48 kHz WAV            -> data/audio/{clip_id}.wav

Then write data/manifest.csv (clip_id, split, frame_path, audio_path,
visual_captions, audio_captions). Clips missing any of those four pieces
are dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"
AUDIO = DATA / "audio"
MANIFEST = DATA / "manifest.csv"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


RAW_VIDEO_DIRS = {
    "val": ROOT / "val_videos" / "val_videos",
    "test": ROOT / "test_videos" / "test_videos",
}


def probe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def extract_frame(video: Path, dst: Path, t: float) -> bool:
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def extract_audio(video: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ac", "1", "-ar", "48000",
        "-vn", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N clips per split (debug aid)")
    args = ap.parse_args()

    rows: list[dict] = []
    dropped: list[tuple[str, str, str]] = []  # (split, clip_id, reason)

    for split, vdir in RAW_VIDEO_DIRS.items():
        caps_path = DATA / "captions" / f"{split}.json"
        if not caps_path.exists():
            print(f"[warn] no captions for split={split} at {caps_path}; skipping")
            continue
        caps = json.loads(caps_path.read_text())

        clip_ids = sorted(caps.keys())
        if args.limit is not None:
            clip_ids = clip_ids[: args.limit]

        for clip_id in clip_ids:
            entry = caps[clip_id]
            visual_caps = [c.strip() for c in entry.get("visual_captions", []) if c and c.strip()]
            audio_caps = [c.strip() for c in entry.get("audio_captions", []) if c and c.strip()]
            if not visual_caps or not audio_caps:
                dropped.append((split, clip_id, "empty caption pool"))
                continue

            frame_path = FRAMES / f"{clip_id}.jpg"
            audio_path = AUDIO / f"{clip_id}.wav"
            video = vdir / f"{clip_id}.mp4"

            # If pre-extracted frame + audio already exist on disk, accept the row
            # without requiring the raw .mp4 (useful when only data/ is synced to HPC).
            already_have = (
                frame_path.exists() and frame_path.stat().st_size > 0
                and audio_path.exists() and audio_path.stat().st_size > 0
            )

            if not already_have:
                if not video.exists():
                    dropped.append((split, clip_id, "no video and no pre-extracted data"))
                    continue
                dur = probe_duration(video)
                if dur is None or dur <= 0:
                    dropped.append((split, clip_id, "ffprobe failed"))
                    continue
                if not extract_frame(video, frame_path, dur / 2.0):
                    dropped.append((split, clip_id, "frame extract failed"))
                    continue
                if not extract_audio(video, audio_path):
                    dropped.append((split, clip_id, "audio extract failed"))
                    continue

            rows.append({
                "clip_id": clip_id,
                "split": split,
                "frame_path": str(frame_path.relative_to(ROOT)),
                "audio_path": str(audio_path.relative_to(ROOT)),
                "visual_captions": json.dumps(visual_caps, ensure_ascii=False),
                "audio_captions": json.dumps(audio_caps, ensure_ascii=False),
            })

    rows.sort(key=lambda r: (r["split"], r["clip_id"]))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id", "split", "frame_path", "audio_path",
                "visual_captions", "audio_captions",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"manifest rows: {len(rows)}  ->  {MANIFEST}")
    by_split: dict[str, int] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    for k, v in sorted(by_split.items()):
        print(f"  {k}: {v}")
    if dropped:
        print(f"dropped: {len(dropped)}")
        for s, c, why in dropped[:10]:
            print(f"  [{s}] {c}  ({why})")
        if len(dropped) > 10:
            print(f"  ... +{len(dropped) - 10} more")


if __name__ == "__main__":
    main()
