r"""Visualise the recovered transport that aligns the image space to the audio
space, as a three-panel figure.

A solved coupling ``P`` (rows = image clips, cols = audio clips; row-stochastic
after normalisation) is the object this thesis recovers. Because the two
embedding spaces have different dimensions they cannot be scattered together
raw; this script instead *uses* the coupling to push one space onto the other
and shows three complementary views:

  (a) Transport routing  -- the image cloud and the audio cloud side by side
      (each projected to 2D on its own), with an edge from every image clip to
      the audio clip that receives most of its mass (its top-1 target). Edges
      are green when that target is the clip's true partner, red otherwise.

  (b) Barycentric push-forward -- each image clip is mapped into the AUDIO
      space by its transport barycentre Y_hat = rownorm(P) @ Y, then the audio
      points and the transported image points are projected jointly. A thin
      line joins each transported point to its TRUE audio partner; short green
      lines mean the transport landed the clip next to the right sound.

  (c) Coupling matrix -- rownorm(P) shown in clip order, so the diagonal is the
      ground-truth pairing. Mass concentrated on the diagonal = good alignment.

Rows are clip-for-clip aligned across embeddings and the plan (the n=400
manifest order), so index i is the same clip in X, Y and P, and P's diagonal is
the true correspondence. Projections use the cosine metric to match retrieval.

Inputs (no experiments run):
  embeddings/vision_{img}.npy, embeddings/audio_{aud}.npy
  transitive: results/exp_c{suffix}/plans/T__K{K}__a{alpha:.2f}.npy
  caption:    results/exp_d{suffix}/plans/T__a{alpha:.2f}.npy

Output:
  results/exp_grid/plots/transport_map__{recipe}__{img}__{aud}__a{alpha}.png

Usage:
  python code/plot_transport_maps.py --recipe transitive --pair text-grounded
  python code/plot_transport_maps.py --recipe caption --pair text-free --alpha 0.9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sns.set_theme(style="white", context="notebook", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

OK_C, BAD_C = "#2ca25f", "#de2d26"

PAIRS = {
    "text-grounded": {"image": "clip-large",   "audio": "clap-unfused",
                      "suffix": "",
                      "label": r"text-grounded (CLIP-L $\times$ CLAP-unfused)"},
    "text-free":     {"image": "dinov2-large", "audio": "mert-330m",
                      "suffix": "__dinov2-large__mert-330m",
                      "label": r"text-free (DINOv2-L $\times$ MERT-330m)"},
}

RECIPES = {
    "transitive": {"exp": "exp_c", "use_K": True,
                   "label": "transitive bridge"},
    "caption":    {"exp": "exp_d", "use_K": False,
                   "label": "caption-cost FGW"},
}


def load_embedding(name: str) -> np.ndarray:
    path = EMB / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding: {path}")
    X = np.load(path).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def load_plan(recipe: dict, suffix: str, alpha: float, K: int) -> Path | None:
    plans = RES / f"{recipe['exp']}{suffix}" / "plans"
    name = (f"T__K{int(K)}__a{alpha:.2f}.npy" if recipe["use_K"]
            else f"T__a{alpha:.2f}.npy")
    fp = plans / name
    return fp if fp.exists() else None


def row_normalise(P: np.ndarray) -> np.ndarray:
    rs = P.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return P / rs


def project(M: np.ndarray, method: str, seed: int) -> np.ndarray:
    n = M.shape[0]
    if method == "umap":
        import umap
        nn = min(15, max(2, n - 1))
        return umap.UMAP(n_components=2, metric="cosine", n_neighbors=nn,
                         min_dist=0.12, random_state=seed).fit_transform(M)
    perp = min(30.0, max(5.0, (n - 1) / 3.0))
    return TSNE(n_components=2, metric="cosine", init="pca",
                learning_rate="auto", perplexity=perp,
                random_state=seed).fit_transform(M)


def _unit_box(Z: np.ndarray) -> np.ndarray:
    """Rescale a 2D layout into the unit square for side-by-side placement."""
    lo, hi = Z.min(0), Z.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return (Z - lo) / span


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.85")


def render(out_path: Path, img_enc: str, aud_enc: str, suffix: str,
           pair_label: str, recipe: dict, recipe_key: str, alpha: float,
           K: int, method: str, seed: int) -> None:
    X = load_embedding(f"vision_{img_enc}")
    Y = load_embedding(f"audio_{aud_enc}")
    plan_fp = load_plan(recipe, suffix, alpha, K)
    if plan_fp is None:
        print(f"[transport] no plan for recipe={recipe_key} "
              f"alpha={alpha} K={K}; skipping {out_path.name}")
        return
    P = np.load(plan_fp).astype(np.float64)
    n = X.shape[0]
    if P.shape != (n, Y.shape[0]):
        print(f"[transport] shape mismatch P={P.shape} vs "
              f"(X={n}, Y={Y.shape[0]}); skipping {out_path.name}")
        return

    Pn = row_normalise(P)
    tgt = Pn.argmax(axis=1)
    truth = np.arange(n)
    correct = tgt == truth
    acc = float(correct.mean())
    edge_alpha = float(np.clip(40.0 / n, 0.05, 0.3))

    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.8), layout="constrained")

    ax = axes[0]
    Zx = _unit_box(project(X, method, seed))
    Zy = _unit_box(project(Y, method, seed))
    gap = 1.7
    Zy_shift = Zy.copy()
    Zy_shift[:, 0] += gap
    for i in range(n):
        j = tgt[i]
        ax.plot([Zx[i, 0], Zy_shift[j, 0]], [Zx[i, 1], Zy_shift[j, 1]],
                "-", color=(OK_C if correct[i] else BAD_C),
                lw=0.6, alpha=edge_alpha, zorder=1)
    ax.scatter(Zx[:, 0], Zx[:, 1], s=16, color="0.30", zorder=3,
               edgecolor="white", linewidth=0.2)
    ax.scatter(Zy_shift[:, 0], Zy_shift[:, 1], s=16, color="0.30", zorder=3,
               edgecolor="white", linewidth=0.2)
    ax.text(0.5, -0.04, "image space", ha="center", va="top",
            transform=ax.transAxes, fontsize=10)
    ax.text(0.5 + gap / (1 + gap), -0.04, "audio space", ha="center", va="top",
            transform=ax.transAxes, fontsize=10)
    ax.set_title("(a) top-1 transport routing", fontsize=11)
    ax.margins(0.06)
    _clean(ax)

    ax = axes[1]
    Yhat = Pn @ Y
    Zj = project(np.vstack([Y, Yhat]), method, seed)
    Zy2, Zh = Zj[:n], Zj[n:]
    for i in range(n):
        ax.plot([Zh[i, 0], Zy2[i, 0]], [Zh[i, 1], Zy2[i, 1]], "-",
                color=(OK_C if correct[i] else BAD_C),
                lw=0.6, alpha=edge_alpha, zorder=1)
    ax.scatter(Zy2[:, 0], Zy2[:, 1], s=18, color="0.78", zorder=2,
               edgecolor="white", linewidth=0.2, label="audio clip (true)")
    ax.scatter(Zh[correct, 0], Zh[correct, 1], s=20, color=OK_C, zorder=3,
               edgecolor="white", linewidth=0.2, label="image clip, transported (correct)")
    ax.scatter(Zh[~correct, 0], Zh[~correct, 1], s=20, color=BAD_C, zorder=3,
               edgecolor="white", linewidth=0.2, label="image clip, transported (wrong)")
    ax.set_title(f"(b) image space transported onto audio space\n"
                 f"top-1 accuracy = {acc:.2f}", fontsize=11)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.85, markerscale=1.2)
    ax.margins(0.06)
    _clean(ax)

    ax = axes[2]
    pos = Pn[Pn > 0]
    vmin = float(np.quantile(pos, 0.50)) if pos.size else 1e-4
    vmin = max(vmin, 1e-5)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    Pm = np.ma.masked_where(Pn <= 0, Pn)
    im = ax.imshow(Pm, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=Pn.max()),
                   aspect="auto", interpolation="nearest", origin="upper")
    ax.plot([0, n - 1], [0, n - 1], color="white", lw=0.7, alpha=0.5)
    ax.set_xlabel("audio clip index", fontsize=10)
    ax.set_ylabel("image clip index", fontsize=10)
    ax.set_title("(c) transport plan  (diagonal = true pair)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                 label="transported mass (log)")

    k_note = rf", $K={K}$" if recipe["use_K"] else ""
    fig.suptitle(
        rf"Optimal-transport alignment of the image and audio spaces — "
        rf"{pair_label}" "\n"
        rf"{recipe['label']} ($\alpha={alpha:g}${k_note}) · {n} clips · "
        rf"green = lands on the true partner",
        fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[transport] wrote {out_path}  (acc={acc:.3f}, n={n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES.keys()), default="transitive")
    ap.add_argument("--pair", choices=list(PAIRS.keys()), default="text-grounded")
    ap.add_argument("--image-encoder", default=None)
    ap.add_argument("--audio-encoder", default=None)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--method", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    spec = PAIRS[args.pair]
    img_enc = args.image_encoder or spec["image"]
    aud_enc = args.audio_encoder or spec["audio"]
    suffix = spec["suffix"]
    recipe = RECIPES[args.recipe]
    pair_label = (spec["label"] if (args.image_encoder is None
                                    and args.audio_encoder is None)
                  else rf"{img_enc} $\times$ {aud_enc}")

    method = args.method
    if method == "umap":
        try:
            import umap  # noqa: F401
        except ImportError:
            print("[transport] umap-learn missing; falling back to t-SNE.")
            method = "tsne"

    a_tag = f"{args.alpha:.2f}".replace(".", "p")
    k_tag = f"__K{args.K}" if recipe["use_K"] else ""
    out = (Path(args.out) if args.out else
           PLOT_DIR / f"transport_map__{args.recipe}__{img_enc}__{aud_enc}"
                      f"{k_tag}__a{a_tag}.png")
    render(out, img_enc, aud_enc, suffix, pair_label, recipe, args.recipe,
           args.alpha, args.K, method, args.seed)


if __name__ == "__main__":
    main()
