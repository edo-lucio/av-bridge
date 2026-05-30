r"""t-SNE cluster panel: why a purely structural (pure-GW) alignment fails.

Pure Gromov-Wasserstein aligns two spaces by matching their *intra-modal*
relational structure (it only ever sees within-space distances, never a
cross-space cost). That can only recover the correct image<->audio pairing if
the two geometries actually correspond -- if a neighbourhood in image space
maps to the analogous neighbourhood in audio space. This figure shows that
they do not.

Layout: a 2x2 grid sharing two t-SNE layouts (image left, audio right).
  Row 1 colours both panels by k-means computed on the IMAGE space.
  Row 2 colours both panels by k-means computed on the AUDIO space.
On the diagonal (image coloured by image clusters, audio by audio clusters)
each space looks cleanly clustered -- the structure is real. Off the diagonal
(image clusters shown in audio space, and vice versa) the colours scatter:
the clusters of one modality do not survive in the other.

Two numbers quantify the mismatch (both computed from embeddings only):
  * AMI(image k-means, audio k-means) -- chance-corrected agreement of the two
    independent partitions; ~0 means the cluster structures are unrelated.
  * mean k-NN overlap -- for each clip, the fraction of its k nearest image
    neighbours that are also among its k nearest audio neighbours; low means
    local geometry does not correspond.

Because pure-GW matches exactly this relational structure (up to isometry /
permutation symmetries), a low AMI and a low k-NN overlap mean it has no
consistent correspondence to lock onto, so its instance retrieval sits near
chance even though it can still recover coarse blobs.

Reads embeddings/{vision,audio}_{enc}.npy only (no plans, no experiments).

Output:
  results/exp_grid/plots/tsne_cluster_panel__{img}__{aud}.png

Usage:
  python code/plot_tsne_cluster_panel.py --pair text-grounded
  python code/plot_tsne_cluster_panel.py --pair text-free --n-clusters 10
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
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

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


def load_embedding(name: str) -> np.ndarray:
    path = EMB / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding: {path}")
    X = np.load(path).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def kmeans_labels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = max(2, min(k, X.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def tsne_2d(X: np.ndarray, seed: int) -> np.ndarray:
    n = X.shape[0]
    perp = min(30.0, max(5.0, (n - 1) / 3.0))
    return TSNE(n_components=2, metric="cosine", init="pca",
                learning_rate="auto", perplexity=perp,
                random_state=seed).fit_transform(X)


def knn_overlap(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """Mean fraction of each clip's k nearest image-neighbours that are also
    among its k nearest audio-neighbours (self excluded)."""
    k = min(k, X.shape[0] - 1)
    ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X) \
        .kneighbors(X, return_distance=False)[:, 1:]
    iy = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Y) \
        .kneighbors(Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iy)]))


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.85")


def render(out_path: Path, img_enc: str, aud_enc: str, pair_label: str,
           n_clusters: int, k_nn: int, seed: int) -> None:
    X = load_embedding(f"vision_{img_enc}")
    Y = load_embedding(f"audio_{aud_enc}")
    if X.shape[0] != Y.shape[0]:
        n = min(X.shape[0], Y.shape[0])
        print(f"[tsne-panel] row mismatch; truncating to {n}")
        X, Y = X[:n], Y[:n]
    n = X.shape[0]

    lab_img = kmeans_labels(X, n_clusters, seed)
    lab_aud = kmeans_labels(Y, n_clusters, seed)
    ami = adjusted_mutual_info_score(lab_img, lab_aud)
    ov = knn_overlap(X, Y, k_nn)

    Zx, Zy = tsne_2d(X, seed), tsne_2d(Y, seed)
    k_eff = int(max(lab_img.max(), lab_aud.max())) + 1
    palette = sns.color_palette("husl", k_eff)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 11.0),
                             layout="constrained")
    rows = [("coloured by\nimage $k$-means", lab_img),
            ("coloured by\naudio $k$-means", lab_aud)]
    cols = [("Image space (t-SNE)", Zx), ("Audio space (t-SNE)", Zy)]

    for r, (row_label, lab) in enumerate(rows):
        for c, (col_title, Z) in enumerate(cols):
            ax = axes[r][c]
            ax.scatter(Z[:, 0], Z[:, 1], c=[palette[i] for i in lab],
                       s=22, alpha=0.82, linewidth=0.3, edgecolor="white")
            _clean(ax)
            diagonal = (r == 0 and c == 0) or (r == 1 and c == 1)
            if diagonal:
                ax.set_title("own clustering — coherent", fontsize=9,
                             color="0.25", loc="right", style="italic")
            if r == 0:
                ax.set_title(col_title, fontsize=12, pad=8)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11.5, fontweight="bold",
                              labelpad=10)

    fig.suptitle(
        rf"Image vs audio geometry — {pair_label}" "\n"
        rf"$k$-means $k={n_clusters}$ · "
        rf"AMI(image, audio partitions) $= {ami:.2f}$ (0 = unrelated) · "
        rf"mean {k_nn}-NN overlap $= {ov:.2f}$" "\n"
        r"each space clusters cleanly under its own labels (diagonal) but those "
        r"clusters scatter in the other space (off-diagonal):" "\n"
        r"the intra-modal geometries don't correspond, so pure-GW ($\alpha=1$) "
        r"has no consistent structure to match $\Rightarrow$ instance retrieval near chance",
        fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne-panel] wrote {out_path}  (AMI={ami:.3f}, "
          f"{k_nn}-NN overlap={ov:.3f}, n={n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS.keys()),
                    default="text-grounded")
    ap.add_argument("--image-encoder", default=None)
    ap.add_argument("--audio-encoder", default=None)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--k-nn", type=int, default=10,
                    help="neighbourhood size for the k-NN overlap statistic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    spec = PAIRS[args.pair]
    img_enc = args.image_encoder or spec["image"]
    aud_enc = args.audio_encoder or spec["audio"]
    pair_label = (spec["label"] if (args.image_encoder is None
                                    and args.audio_encoder is None)
                  else rf"{img_enc} $\times$ {aud_enc}")

    out = (Path(args.out) if args.out else
           PLOT_DIR / f"tsne_cluster_panel__{img_enc}__{aud_enc}.png")
    render(out, img_enc, aud_enc, pair_label, args.n_clusters, args.k_nn,
           args.seed)


if __name__ == "__main__":
    main()
