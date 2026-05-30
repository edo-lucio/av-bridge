r"""Transitive bridge: held-out retrieval across the K x alpha grid,
coloured by the structural weight alpha.

The Transitive Transport bridge has two knobs: the cluster/anchor count K
and the FGW structural weight alpha on its legs. This view scatters every
(K, alpha) cell -- x = K, y = retrieval (held-out) -- and colours each
point by alpha, so the intended trend reads off directly: warmer (higher
alpha) points sit higher, i.e. leaning on the Gromov-Wasserstein structural
term improves held-out retrieval, and it does so more as K grows.

Points of equal alpha are joined by a faint line of the same colour to make
the per-alpha trajectory across K legible. A small horizontal dodge per
alpha keeps coincident cells from hiding each other.

Reads only the transitive sweep CSVs (no embeddings):
  text-grounded  results/exp_c/sweep_transitive.csv
  text-free      results/exp_c__dinov2-large__mert-330m/sweep_transitive.csv

Output:
  results/exp_grid/plots/transitive_k_alpha_scatter__{metric}__{scope}.png

Usage:
  python code/plot_transitive_k_alpha_scatter.py
  python code/plot_transitive_k_alpha_scatter.py --metric cat_precision_10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

REGIMES = {
    "text-grounded": {"suffix": "",
                      "label": r"text-grounded (CLIP-L $\times$ CLAP-unfused)"},
    "text-free":     {"suffix": "__dinov2-large__mert-330m",
                      "label": r"text-free (DINOv2-L $\times$ MERT-330m)"},
}

METRIC_LABELS = {
    "R@10": r"$R@10$",
    "cat_precision_10": r"Cat-prec@10",
    "R@5": r"$R@5$",
    "R@1": r"$R@1$",
}

CMAP = plt.get_cmap("viridis")


def _load(suffix: str, scope: str) -> pd.DataFrame | None:
    csv = RES / f"exp_c{suffix}" / "sweep_transitive.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df.scope == scope].copy()
    return df if not df.empty else None


def render(out_path: Path, metric: str, scope: str) -> None:
    data = {k: _load(v["suffix"], scope) for k, v in REGIMES.items()}
    data = {k: v for k, v in data.items() if v is not None and metric in v}
    if not data:
        print(f"[k-alpha] no transitive data for metric={metric}, "
              f"scope={scope}")
        return

    alphas = sorted({float(a) for df in data.values()
                     for a in df.alpha.unique()})
    norm = Normalize(vmin=min(alphas), vmax=max(alphas))
    # Small symmetric horizontal dodge (in log space) so coincident
    # (K, *) cells of different alpha don't fully overlap.
    n_a = max(len(alphas), 1)
    dodge = {a: 1.0 + 0.045 * (i - (n_a - 1) / 2)
             for i, a in enumerate(alphas)}

    fig, axes = plt.subplots(1, len(data), figsize=(6.6 * len(data), 5.2),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, (reg, df) in zip(axes[0], data.items()):
        for a in alphas:
            d = df[np.isclose(df.alpha, a)].sort_values("K")
            if d.empty:
                continue
            x = d.K.to_numpy(float) * dodge[a]
            y = d[metric].to_numpy(float)
            color = CMAP(norm(a))
            ax.plot(x, y, "-", color=color, lw=1.3, alpha=0.55, zorder=2)
            ax.scatter(x, y, s=85, color=color, edgecolor="black",
                       linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(sorted(df.K.unique()))
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"cluster / anchor count $K$  (log scale)", fontsize=10)
        ax.set_title(REGIMES[reg]["label"], fontsize=11)
        ax.margins(y=0.12)
        sns.despine(ax=ax)
    axes[0][0].set_ylabel(f"{METRIC_LABELS.get(metric, metric)}  ({scope})",
                          fontsize=10)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.046, pad=0.02,
                        ticks=alphas)
    cbar.set_label(r"structural weight $\alpha$  (GW influence $\to$)",
                   fontsize=10)

    fig.suptitle(
        f"Transitive bridge — {METRIC_LABELS.get(metric, metric)} across "
        f"$K\\times\\alpha$ ({scope})\n"
        r"warmer = higher $\alpha$; leaning on the GW term lifts held-out "
        r"retrieval, more so as $K$ grows",
        fontsize=12.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[k-alpha] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="R@10",
                    choices=list(METRIC_LABELS.keys()))
    ap.add_argument("--scope", default="heldout",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    safe = args.metric.replace("@", "").replace("_", "")
    out = (Path(args.out) if args.out else
           PLOT_DIR / f"transitive_k_alpha_scatter__{safe}__{args.scope}.png")
    render(out, args.metric, args.scope)


if __name__ == "__main__":
    main()
