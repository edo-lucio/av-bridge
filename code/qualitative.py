"""Qualitative renderer for image <-> audio retrieval.

Given an AVCaps clip_id, renders a single PNG with two stacked panels:

  - Image -> Audio: query frame and one of its visual captions on the left,
    top-K retrieved audio clips (mel spectrogram + audio caption) on the right.
  - Audio -> Image: query audio (mel spectrogram + audio caption) on the left,
    top-K retrieved image clips (frame + visual caption) on the right.

Retrieval uses the same four-step push that code/infer.py uses; the
audio->image direction simply transposes the plan (FGW / Sinkhorn produce
plans that are approximately doubly stochastic under uniform marginals).

Usage:
    python code/qualitative.py --clip-id 10001787725
    python code/qualitative.py --clip-id 10001787725 \\
        --plan results/exp_c/T_transitive.npy --top-k 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
DEFAULT_PLAN = RES / "exp_d" / "T_caption.npy"


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


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def four_step_push(v_query: np.ndarray, X_anchors: np.ndarray, T: np.ndarray,
                   query_top_k: int, query_temp: float) -> np.ndarray:
    n = X_anchors.shape[0]
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
    return w_src @ T


def retrieve(query_idx: int, anchors: np.ndarray, plan: np.ndarray,
             top_k: int, query_top_k: int, query_temp: float,
             suppress_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    w = four_step_push(anchors[query_idx], anchors, plan,
                       query_top_k=query_top_k, query_temp=query_temp)
    w = w.copy()
    if suppress_self:
        w[query_idx] = -np.inf
    order = np.argsort(-w)[:top_k]
    return order, w[order]


def _truncate(s: str, n: int = 140) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _overlay_label(ax, label: str, dark_bg: bool = False) -> None:
    if not label:
        return
    color = "white" if dark_bg else "black"
    facecolor = "black" if dark_bg else "white"
    ax.text(
        0.02, 0.97, label,
        ha="left", va="top", fontsize=8.5, transform=ax.transAxes,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor=facecolor, edgecolor="none", alpha=0.75),
    )


def _render_image(ax, path: Path, title: str = "", border: str | None = None) -> None:
    try:
        img = Image.open(path).convert("RGB")
        ax.imshow(np.asarray(img))
    except Exception as exc:
        ax.text(0.5, 0.5, f"<image\nunavailable>\n{exc.__class__.__name__}",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    _overlay_label(ax, title, dark_bg=False)
    if border:
        for spine in ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(2.5)


def _render_spectrogram(ax, path: Path, title: str = "",
                        border: str | None = None,
                        sr_load: int = 22050, duration: float = 10.0) -> None:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=sr_load, duration=duration, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr // 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
    except Exception as exc:
        ax.text(0.5, 0.5, f"<audio\nunavailable>\n{exc.__class__.__name__}",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    _overlay_label(ax, title, dark_bg=True)
    if border:
        for spine in ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(2.5)


def _render_captions(ax, captions: list[str], n_show: int = 2) -> None:
    """Caption text in its own axes; up to n_show lines, wrapped."""
    ax.axis("off")
    body = []
    for c in captions[:n_show]:
        body.append(textwrap.fill(_truncate(c, 140), width=42))
    ax.text(0.0, 1.0, "\n\n".join(body),
            ha="left", va="top", fontsize=8,
            transform=ax.transAxes, wrap=True)


def _render_dual_captions(ax, vcaps: list[str], acaps: list[str],
                          n_show: int = 1) -> None:
    """Both modality captions stacked with labels."""
    ax.axis("off")
    parts = []
    for c in vcaps[:n_show]:
        parts.append(("visual", textwrap.fill(_truncate(c, 140), width=40)))
    for c in acaps[:n_show]:
        parts.append(("audio",  textwrap.fill(_truncate(c, 140), width=40)))
    y = 1.0
    for tag, body in parts:
        ax.text(0.0, y, tag, ha="left", va="top",
                fontsize=7.5, fontweight="bold",
                color="tab:blue" if tag == "visual" else "tab:red",
                transform=ax.transAxes)
        ax.text(0.16, y, body, ha="left", va="top",
                fontsize=7.5, transform=ax.transAxes)
        y -= 0.50


def make_figure(query_clip: str, manifest: list[dict],
                X_image: np.ndarray, Y_audio: np.ndarray, plan: np.ndarray,
                top_k: int, query_top_k: int, query_temp: float,
                title: str, out: Path,
                method_label: str | None = None,
                suppress_self: bool = False) -> None:
    ids = [m["clip_id"] for m in manifest]
    if query_clip not in ids:
        raise KeyError(f"clip_id {query_clip!r} not in manifest")
    q_idx = ids.index(query_clip)
    q = manifest[q_idx]

    i2a_idx, i2a_scores = retrieve(
        q_idx, X_image, plan, top_k, query_top_k, query_temp,
        suppress_self=suppress_self,
    )

    a2i_idx, a2i_scores = retrieve(
        q_idx, Y_audio, plan.T, top_k, query_top_k, query_temp,
        suppress_self=suppress_self,
    )

    n_cols = 1 + top_k
    fig = plt.figure(figsize=(3.4 * n_cols, 10.4))
    gs = fig.add_gridspec(
        nrows=7, ncols=n_cols,
        height_ratios=[0.35, 3.0, 1.9, 0.5, 0.35, 3.0, 1.9],
        hspace=0.18, wspace=0.18,
        left=0.04, right=0.99, top=0.97, bottom=0.05,
    )

    def _header(row: int, txt: str) -> None:
        hax = fig.add_subplot(gs[row, :])
        hax.axis("off")
        hax.text(0.5, 0.5, txt, ha="center", va="center",
                 fontsize=13, fontweight="bold", transform=hax.transAxes)

    method_str = f"  ·  {method_label}" if method_label else ""
    _header(0, f"Image → Audio   (query clip {query_clip}){method_str}")

    ax = fig.add_subplot(gs[1, 0])
    _render_image(ax, q["frame_path"], title="query (image)", border="tab:blue")
    cap_ax = fig.add_subplot(gs[2, 0])
    _render_dual_captions(cap_ax, q["visual_captions"], q["audio_captions"])

    for k, (tgt, score) in enumerate(zip(i2a_idx, i2a_scores), start=1):
        item = manifest[int(tgt)]
        is_gt = (int(tgt) == q_idx)
        tag = "  [GT]" if is_gt else ""
        ax = fig.add_subplot(gs[1, k])
        _render_spectrogram(ax, item["audio_path"],
                            title=f"top-{k}  clip {item['clip_id']}{tag}\nscore={score:.4f}",
                            border="tab:green" if is_gt else None)
        cap_ax = fig.add_subplot(gs[2, k])
        _render_dual_captions(cap_ax, item["visual_captions"], item["audio_captions"])

    _header(4, f"Audio → Image   (query clip {query_clip}){method_str}")

    ax = fig.add_subplot(gs[5, 0])
    _render_spectrogram(ax, q["audio_path"],
                        title="query (audio)", border="tab:blue")
    cap_ax = fig.add_subplot(gs[6, 0])
    _render_dual_captions(cap_ax, q["visual_captions"], q["audio_captions"])

    for k, (tgt, score) in enumerate(zip(a2i_idx, a2i_scores), start=1):
        item = manifest[int(tgt)]
        is_gt = (int(tgt) == q_idx)
        tag = "  [GT]" if is_gt else ""
        ax = fig.add_subplot(gs[5, k])
        _render_image(ax, item["frame_path"],
                      title=f"top-{k}  clip {item['clip_id']}{tag}\nscore={score:.4f}",
                      border="tab:green" if is_gt else None)
        cap_ax = fig.add_subplot(gs[6, k])
        _render_dual_captions(cap_ax, item["visual_captions"], item["audio_captions"])

    fig.text(0.5, 0.012, title, ha="center", va="bottom",
             fontsize=8, color="dimgray")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[qual] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Qualitative figure: image <-> audio retrieval with captions."
    )
    ap.add_argument("--clip-id", required=True,
                    help="AVCaps clip_id used as both query positions.")
    ap.add_argument("--plan", type=str, default=str(DEFAULT_PLAN),
                    help="Path to the transport plan .npy.")
    ap.add_argument("--image-encoder", default="clip-large")
    ap.add_argument("--audio-encoder", default="clap-unfused")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of retrieved items per direction.")
    ap.add_argument("--query-top-k", type=int, default=1,
                    help="Soft-assign breadth (1 = strict argmax-of-row).")
    ap.add_argument("--query-temp", type=float, default=1e-6,
                    help="Soft-assign temperature (~0 = strict argmax).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default: results/qualitative/<clip>.png).")
    ap.add_argument("--method-label", type=str, default=None,
                    help="Human-readable name shown in the figure header "
                         "(e.g. 'FGW with caption cost'). If omitted, the "
                         "plan filename stem is used.")
    ap.add_argument("--random-plan", type=str, default=None,
                    choices=["permutation", "sinkhorn", "uniform"],
                    help="If set, synthesise a chance-level plan in memory "
                         "instead of loading --plan. 'permutation' = random "
                         "one-to-one matching (each row → one random "
                         "target, mass 1). 'sinkhorn' = random cost matrix "
                         "passed through Sinkhorn at the same ε used by "
                         "the real solvers (ε = 5e-3). 'uniform' = "
                         "1/n^2 everywhere (degenerate floor).")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="Seed for --random-plan. Use the same seed across "
                         "query clips so a single random permutation is "
                         "compared like a real saved plan.")
    ap.add_argument("--suppress-self", action="store_true",
                    help="Drop the self-match before picking top-k. Default "
                         "off so the figure can show whether the plan "
                         "actually identifies the partner clip (GT highlight).")
    args = ap.parse_args()

    manifest = load_manifest()
    X = np.load(EMB / f"vision_{args.image_encoder}.npy")
    Y = np.load(EMB / f"audio_{args.audio_encoder}.npy")

    if args.random_plan is not None:
        n = X.shape[0]
        rng = np.random.default_rng(args.random_seed)
        if args.random_plan == "permutation":
            perm = rng.permutation(n)
            plan = np.zeros((n, n), dtype=np.float64)
            plan[np.arange(n), perm] = 1.0 / n
        elif args.random_plan == "uniform":
            plan = np.full((n, n), 1.0 / (n * n), dtype=np.float64)
        else:
            from algorithms import sinkhorn
            M = rng.standard_normal((n, n)).astype(np.float64)
            M = M - M.min()
            M = M / max(M.max(), 1e-12)
            plan = sinkhorn(M, eps=0.005)
        plan_label = f"random-{args.random_plan}-seed{args.random_seed}"
        print(f"[plan] random:{args.random_plan} seed={args.random_seed}  "
              f"shape={plan.shape}  total={plan.sum():.4f}")
    else:
        plan = np.load(args.plan)
        plan_label = Path(args.plan).stem
        print(f"[plan] {Path(args.plan).name}  shape={plan.shape}  "
              f"total={plan.sum():.4f}")

    out = Path(args.out) if args.out else (
        RES / "qualitative" / f"{args.clip_id}__{plan_label}.png"
    )
    display_label = args.method_label if args.method_label else plan_label
    title = (f"method = {display_label}   "
             f"encoders = ({args.image_encoder}, {args.audio_encoder})   "
             f"top_k = {args.top_k}   q_top_k = {args.query_top_k}   "
             f"q_temp = {args.query_temp}")
    make_figure(args.clip_id, manifest, X, Y, plan,
                top_k=args.top_k,
                query_top_k=args.query_top_k,
                query_temp=args.query_temp,
                title=title, out=out,
                method_label=args.method_label,
                suppress_self=args.suppress_self)


if __name__ == "__main__":
    main()
