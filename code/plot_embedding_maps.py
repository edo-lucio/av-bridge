r"""UMAP and t-SNE maps of the image and audio embedding spaces.

The two encoders this thesis aligns live in separate, differently-shaped
latent spaces, so they cannot be scattered in a single shared plot without
first aligning them (which is the whole problem). Instead this script lays the
two spaces side by side and projects each one to 2D with both UMAP and t-SNE,
giving a 2x2 grid:

        |  Image space        |  Audio space
  ------+---------------------+----------------------
  UMAP  |  vision_{img} -> 2D |  audio_{aud} -> 2D
  t-SNE |  vision_{img} -> 2D |  audio_{aud} -> 2D

Points are coloured by a k-means clustering so structure is visible. By
default the clustering is computed once on the IMAGE space and the SAME labels
are reused for the audio panels (rows are clip-for-clip aligned): if the two
spaces organise the world similarly, the image-derived colour groups should
still look locally coherent in the audio projection. Use ``--cluster-on each``
to colour every panel by its own clustering instead (colours are then not
comparable across panels).

Embeddings are read from embeddings/{vision,audio}_{encoder}.npy (already
L2-normalised, row-aligned to data/manifest.csv); projections use the cosine
metric to match the retrieval geometry.

Output:
  results/exp_grid/plots/embedding_maps__{img}__{aud}.png

Usage:
  python code/plot_embedding_maps.py                      # text-grounded pair
  python code/plot_embedding_maps.py --pair text-free
  python code/plot_embedding_maps.py --image-encoder clip-large \
      --audio-encoder clap-unfused --n-clusters 12 --methods umap,tsne
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sns.set_theme(style="white", context="notebook", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
PLOT_DIR = ROOT / "results" / "exp_grid" / "plots"

PAIRS = {
    "text-grounded": {"image": "clip-large",   "audio": "clap-unfused",
                      "label": r"text-grounded (CLIP-L $\times$ CLAP-unfused)"},
    "text-free":     {"image": "dinov2-large", "audio": "mert-330m",
                      "label": r"text-free (DINOv2-L $\times$ MERT-330m)"},
}

METHOD_LABELS = {"umap": "UMAP", "tsne": "t-SNE"}


def _tt(name: str) -> str:
    """Encoder name as plain text. Kept out of mathtext ($...$) on purpose:
    matplotlib's mathtext renders hyphens as minus signs and underscores as
    subscripts, which mangles names like ``clip-large`` / ``mert-330m``."""
    return name


def load_embedding(name: str) -> np.ndarray:
    path = EMB / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding: {path}")
    X = np.load(path).astype(np.float32)
    # Guard against any non-finite rows so the projectors don't choke.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def kmeans_labels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = max(2, min(k, X.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def project(X: np.ndarray, method: str, seed: int,
            perplexity: float | None, n_neighbors: int | None) -> np.ndarray:
    n = X.shape[0]
    if method == "tsne":
        perp = perplexity if perplexity is not None else min(30.0, max(5.0, (n - 1) / 3.0))
        perp = min(perp, max(2.0, n - 1.0))
        return TSNE(n_components=2, metric="cosine", init="pca",
                    learning_rate="auto", perplexity=perp,
                    random_state=seed).fit_transform(X)
    if method == "umap":
        import umap  # local import: optional dependency
        nn = n_neighbors if n_neighbors is not None else min(15, max(2, n - 1))
        nn = min(nn, max(2, n - 1))
        return umap.UMAP(n_components=2, metric="cosine", n_neighbors=nn,
                         min_dist=0.1, random_state=seed).fit_transform(X)
    raise ValueError(f"unknown method {method!r}")


def _style_panel(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("0.85")
    ax.set_facecolor("white")


def render(out_path: Path, img_enc: str, aud_enc: str, pair_label: str,
           methods: list[str], n_clusters: int, cluster_on: str,
           seed: int, perplexity: float | None, n_neighbors: int | None) -> None:
    X_img = load_embedding(f"vision_{img_enc}")
    X_aud = load_embedding(f"audio_{aud_enc}")
    if X_img.shape[0] != X_aud.shape[0]:
        n = min(X_img.shape[0], X_aud.shape[0])
        print(f"[emb-maps] row mismatch (img={X_img.shape[0]}, "
              f"aud={X_aud.shape[0]}); truncating to {n}")
        X_img, X_aud = X_img[:n], X_aud[:n]
    n = X_img.shape[0]

    if cluster_on == "each":
        lab_img = kmeans_labels(X_img, n_clusters, seed)
        lab_aud = kmeans_labels(X_aud, n_clusters, seed)
        colour_note = (rf"colour = per-space $k$-means ($k={n_clusters}$); "
                       r"colours are not comparable across the two columns")
    else:
        base = X_img if cluster_on == "image" else X_aud
        shared = kmeans_labels(base, n_clusters, seed)
        lab_img = lab_aud = shared
        colour_note = (rf"colour = shared $k$-means ($k={n_clusters}$) computed on "
                       rf"the {cluster_on} space and reused for both columns")

    k_eff = int(max(lab_img.max(), lab_aud.max())) + 1
    palette = sns.color_palette("husl", k_eff)

    cols = [("Image space", img_enc, X_img, lab_img),
            ("Audio space", aud_enc, X_aud, lab_aud)]
    nrows, ncols = len(methods), len(cols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.4 * ncols, 5.0 * nrows),
                             squeeze=False, layout="constrained")

    for r, method in enumerate(methods):
        for c, (space_name, enc, X, lab) in enumerate(cols):
            ax = axes[r][c]
            try:
                Z = project(X, method, seed, perplexity, n_neighbors)
            except Exception as e:  # noqa: BLE001
                ax.text(0.5, 0.5, f"{METHOD_LABELS[method]} failed:\n"
                        f"{e.__class__.__name__}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="firebrick")
                _style_panel(ax)
                continue
            ax.scatter(Z[:, 0], Z[:, 1],
                       c=[palette[i] for i in lab], s=22, alpha=0.82,
                       linewidth=0.3, edgecolor="white", zorder=3)
            _style_panel(ax)
            if r == 0:
                ax.set_title(rf"{space_name}" "\n" rf"{_tt(enc)}  "
                             rf"($d={X.shape[1]}$)", fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=13,
                              fontweight="bold", labelpad=10)

    fig.suptitle(
        rf"Latent geometry of the image and audio spaces — {pair_label}"
        "\n" rf"{n} clips · cosine metric · {colour_note}",
        fontsize=13.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[emb-maps] wrote {out_path}  ({nrows}x{ncols} panels, n={n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS.keys()), default="text-grounded",
                    help="canonical encoder pair (overridden by explicit encoders)")
    ap.add_argument("--image-encoder", default=None,
                    help="vision encoder stem, e.g. clip-large (default: from --pair)")
    ap.add_argument("--audio-encoder", default=None,
                    help="audio encoder stem, e.g. clap-unfused (default: from --pair)")
    ap.add_argument("--methods", default="umap,tsne",
                    help="comma list from {umap,tsne}")
    ap.add_argument("--n-clusters", type=int, default=12,
                    help="k for the k-means colouring")
    ap.add_argument("--cluster-on", choices=["image", "audio", "each"],
                    default="image",
                    help="space whose k-means labels colour the points")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=None,
                    help="t-SNE perplexity (default: adaptive to n)")
    ap.add_argument("--n-neighbors", type=int, default=None,
                    help="UMAP n_neighbors (default: adaptive to n)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    spec = PAIRS[args.pair]
    img_enc = args.image_encoder or spec["image"]
    aud_enc = args.audio_encoder or spec["audio"]
    pair_label = (spec["label"] if (args.image_encoder is None
                                    and args.audio_encoder is None)
                  else rf"{_tt(img_enc)} $\times$ {_tt(aud_enc)}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "umap" in methods:
        try:
            import umap  # noqa: F401
        except ImportError:
            print("[emb-maps] umap-learn not installed; dropping UMAP "
                  "(pip install umap-learn). Proceeding with t-SNE only.")
            methods = [m for m in methods if m != "umap"]
    if not methods:
        print("[emb-maps] no usable methods; nothing to plot.")
        return

    out = (Path(args.out) if args.out else
           PLOT_DIR / f"embedding_maps__{img_enc}__{aud_enc}.png")
    render(out, img_enc, aud_enc, pair_label, methods,
           args.n_clusters, args.cluster_on, args.seed,
           args.perplexity, args.n_neighbors)


if __name__ == "__main__":
    main()
