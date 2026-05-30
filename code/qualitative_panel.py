"""Method-vs-baseline comparison plate for cross-modal retrieval.

One figure, one direction (image to audio OR audio to image), with:
  - One row per method (including the random-permutation chance baseline).
  - One column per query clip (typically 5, cluster-stratified).
  - Each cell shows the method's top-1 retrieval for that query.

This is the headline qualitative figure: the reader scans across each
row and immediately sees which methods consistently retrieve
semantically related content vs which behave like random.

Usage:
    python code/qualitative_panel.py \\
        --direction image_to_audio \\
        --query-file data/qual_query_clips.txt \\
        --out results/qualitative_best/panel_i2a_top1.png
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"
RES = ROOT / "results"
MANIFEST = DATA / "manifest.csv"


IMG_ENC = "clip-large"
AUD_ENC = "clap-unfused"


METHODS = [
    {"label": "FGW with caption cost",
     "short": "fgw-cap",
     "plan":  RES / "exp_d" / "T_caption.npy"},
    {"label": "Text-bridged plan composition",
     "short": "text-bridge",
     "plan":  RES / "exp_c" / "T_transitive.npy"},
    {"label": "Ridge-supervised FGW",
     "short": "ridge-sup",
     "plan":  RES / "exp_c" / "T_cdirect.npy"},
    {"label": "GW (intra-modal geometry alone)",
     "short": "gw",
     "plan":  RES / "exp_unsup" / "T_gw.npy"},
    {"label": "Raw caption cosine",
     "short": "cap-cos",
     "plan":  RES / "exp_text" / "T_text.npy"},
    {"label": "Random permutation (chance baseline)",
     "short": "random",
     "plan":  None},
]


def load_manifest() -> list[dict]:
    rows: list[dict] = []
    with MANIFEST.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "clip_id": r["clip_id"],
                "frame_path": ROOT / r["frame_path"],
                "audio_path": ROOT / r["audio_path"],
                "visual_captions": json.loads(r["visual_captions"]),
                "audio_captions": json.loads(r["audio_captions"]),
            })
    return rows


def synth_random_plan(n: int, seed: int) -> np.ndarray:
    """Random one-to-one permutation as a sparse plan, mass 1/n on the
    chosen target per row."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    plan = np.zeros((n, n), dtype=np.float64)
    plan[np.arange(n), perm] = 1.0 / n
    return plan


def _truncate(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _render_image(ax, path: Path, title: str = "",
                  border: str | None = None) -> None:
    try:
        img = Image.open(path).convert("RGB")
        ax.imshow(np.asarray(img))
    except Exception:
        ax.text(0.5, 0.5, "image\nunavailable",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.5)
    if border:
        for spine in ax.spines.values():
            spine.set_color(border); spine.set_linewidth(2.0)


def _render_spectrogram(ax, path: Path, title: str = "",
                        border: str | None = None,
                        sr_load: int = 22050, duration: float = 10.0) -> None:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=sr_load, duration=duration, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr // 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
    except Exception:
        ax.text(0.5, 0.5, "audio\nunavailable",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.5, color="white",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="black", alpha=0.6,
                               edgecolor="none"))
    if border:
        for spine in ax.spines.values():
            spine.set_color(border); spine.set_linewidth(2.0)


def _render_text(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.0, 0.5, textwrap.fill(_truncate(text, 90), width=40),
            ha="left", va="center", fontsize=7, transform=ax.transAxes)


def retrieve_top1(query_idx: int, plan: np.ndarray,
                  direction: str) -> tuple[int, float]:
    """Return (target_idx, score) for the top-1 retrieval, self-suppressed.

    direction:
        'image_to_audio' : use row plan[query_idx]
        'audio_to_image' : use column plan[:, query_idx] (== row of plan.T)
    """
    if direction == "image_to_audio":
        row = plan[query_idx].copy()
    elif direction == "audio_to_image":
        row = plan[:, query_idx].copy()
    else:
        raise ValueError(direction)
    row[query_idx] = -np.inf
    j = int(np.argmax(row))
    return j, float(plan[query_idx, j] if direction == "image_to_audio"
                    else plan[j, query_idx])


def build_plate(query_ids: list[str], direction: str, out: Path,
                random_seed: int = 0) -> None:
    manifest = load_manifest()
    n = len(manifest)
    ids = [m["clip_id"] for m in manifest]
    qrows = [ids.index(c) for c in query_ids]

    loaded: list[dict] = []
    for m in METHODS:
        if m["short"] == "random":
            T = synth_random_plan(n, random_seed)
        else:
            if not m["plan"].exists():
                print(f"[skip] {m['label']}: {m['plan']} not present")
                continue
            T = np.load(m["plan"])
            if T.shape != (n, n):
                print(f"[skip] {m['label']}: shape {T.shape} != ({n}, {n})")
                continue
        loaded.append({**m, "T": T})

    if not loaded:
        raise SystemExit("[err] no methods to render")

    n_rows = len(loaded)
    n_cols = 1 + len(qrows)
    fig = plt.figure(
        figsize=(2.5 * n_cols + 0.2, 1.9 * n_rows + 0.6),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[0.45] + [1.0] * n_rows,
        width_ratios=[0.45] + [1.0] * len(qrows),
        hspace=0.25, wspace=0.10,
        left=0.02, right=0.99, top=0.95, bottom=0.03,
    )

    hax = fig.add_subplot(gs[0, 0]); hax.axis("off")
    hax.text(0.5, 0.5, "method", ha="center", va="center",
             fontsize=9, fontweight="bold")
    for col, q_idx in enumerate(qrows, start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"query #{col}\nclip {manifest[q_idx]['clip_id']}",
                ha="center", va="center", fontsize=8, fontweight="bold")

    for r_idx, m in enumerate(loaded, start=1):
        rax = fig.add_subplot(gs[r_idx, 0])
        rax.axis("off")
        label_wrapped = textwrap.fill(m["label"], width=18)
        rax.text(1.0, 0.5, label_wrapped, ha="right", va="center",
                 fontsize=8.5, fontweight="bold")
        for col, q_idx in enumerate(qrows, start=1):
            ax = fig.add_subplot(gs[r_idx, col])
            tgt, score = retrieve_top1(q_idx, m["T"], direction)
            item = manifest[tgt]
            border = "tab:green" if tgt == q_idx else None
            sub_title = f"top-1: clip {item['clip_id']}  s={score:.3f}"
            if direction == "image_to_audio":
                _render_spectrogram(ax, item["audio_path"],
                                    title=sub_title, border=border)
            else:
                _render_image(ax, item["frame_path"],
                              title=sub_title, border=border)

    direction_str = ("Image $\\to$ Audio" if direction == "image_to_audio"
                     else "Audio $\\to$ Image")
    fig.text(0.5, 0.005,
             f"{direction_str}; encoder pair = ({IMG_ENC}, {AUD_ENC}); "
             f"top-1 retrievals shown, self-match suppressed; "
             f"green border = exact ground-truth partner.",
             ha="center", va="bottom", fontsize=7.5, color="dimgray")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[panel] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["image_to_audio", "audio_to_image"],
                    default="image_to_audio")
    ap.add_argument("--query-file", default=str(DATA / "qual_query_clips.txt"),
                    help="Newline-separated list of clip_ids to use as "
                         "queries. See code/sample_query_clips.py.")
    ap.add_argument("--out", default=None,
                    help="Output PNG; default builds a name from the "
                         "direction.")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    q_path = Path(args.query_file)
    if not q_path.exists():
        raise SystemExit(
            f"[err] {q_path} not found. Run "
            f"`python code/sample_query_clips.py` first."
        )
    queries = [l.strip() for l in q_path.read_text().splitlines() if l.strip()]
    if not queries:
        raise SystemExit(f"[err] {q_path} is empty")

    if args.out is None:
        out = (RES / "qualitative_best"
               / f"panel_{args.direction}_top1.png")
    else:
        out = Path(args.out)

    build_plate(queries, args.direction, out,
                random_seed=args.random_seed)


if __name__ == "__main__":
    main()
