r"""Visualisation for the C-transitive identity-bridge ablation.

Compares the cosine-bridge (original) and identity-bridge (ablation)
versions of C-transitive on the same metrics, side by side, for both
reference encoder pairs (canonical and text-free).

Inputs:
  results/exp_c[__<img>__<aud>]/sweep_transitive.csv       -- cosine bridge
  results/exp_c_idbridge[__<img>__<aud>]/sweep.csv         -- identity bridge

Output:
  results/exp_grid/plots/ablation_idbridge_vs_cosine.png

The figure is a 2 x 2 panel grid: rows = scope (aggregate / held-out),
columns = encoder pair (canonical / text-free). Each panel plots grouped
bars: x-axis = metric (R@10, AMI, Pearson r), grouped by bridge type
(cosine vs identity). Missing cells render as blank bars so partial
data does not break the figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_OUT = RES / "exp_grid" / "plots" / "ablation_idbridge_vs_cosine.png"

PRETTY = {
    "clip-base":   "CLIP-B/32",
    "clip-large":  "CLIP-L/14",
    "dinov2-small": "DINOv2-S",
    "dinov2-base":  "DINOv2-B",
    "dinov2-large": "DINOv2-L",
    "vit-mae-base": "ViT-MAE-B",
    "clap-fused":   "CLAP-fused",
    "clap-unfused": "CLAP-unfused",
    "clap-larger":  "CLAP-larger",
    "mert-95m":     "MERT-95m",
    "mert-330m":    "MERT-330m",
}


def _discover_pairs() -> list[dict]:
    """Auto-discover pairs from both cosine and identity directories.

    A pair is included if EITHER an identity-bridge sweep file or a
    cosine-bridge sweep file is on disk for it. Missing sides are
    rendered as 'n/a' bars but the panel still appears.
    """
    suffixes: set[str] = set()
    for p in RES.glob("exp_c_idbridge*/sweep.csv"):
        suffixes.add(p.parent.name.removeprefix("exp_c_idbridge"))
    # Cosine-bridge directories are exp_c[__<img>__<aud>]/sweep_transitive.csv
    # but the suffix must NOT belong to a different recipe (e.g. exp_c_idbridge).
    for p in RES.glob("exp_c*/sweep_transitive.csv"):
        tail = p.parent.name.removeprefix("exp_c")
        # Exclude exp_c_idbridge* paths.
        if tail.startswith("_idbridge"):
            continue
        suffixes.add(tail)
    pairs = []
    for suffix in sorted(suffixes):
        if suffix == "":
            label_img, label_aud = "CLIP-L/14", "CLAP-unfused"
        else:
            # "__<img>__<aud>" -> ("<img>", "<aud>")
            parts = suffix.removeprefix("__").split("__")
            if len(parts) != 2:
                continue
            img_raw, aud_raw = parts
            label_img = PRETTY.get(img_raw, img_raw)
            label_aud = PRETTY.get(aud_raw, aud_raw)
        pairs.append({
            "label":  f"{label_img}\n$\\times$ {label_aud}",
            "suffix": suffix,
        })
    return pairs


PAIRS = _discover_pairs()

# (scope-key-in-cosine-csv, scope-key-in-identity-csv, display-label)
SCOPES = [
    ("aggregate", "aggregate",      "aggregate"),
    ("heldout",   "heldout", "held-out"),
]

METRICS = [
    ("R@10",        "$R@10$"),
    ("ami",         "AMI"),
    ("pearson_r",   "Pearson $r$"),
]

BRIDGE_PALETTE = {
    "cosine bridge (original)":  "#4c72b0",
    "identity bridge (ablation)": "#dd8452",
}


def _load_row(csv_path: Path, scope_key: str) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[df.scope == scope_key]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def main() -> None:
    fig, axes = plt.subplots(len(SCOPES), len(PAIRS),
                             figsize=(12, 8), squeeze=False)
    x = np.arange(len(METRICS))
    width = 0.35

    any_data = False
    for row_idx, (scope_cos, scope_id, scope_label) in enumerate(SCOPES):
        for col_idx, pair in enumerate(PAIRS):
            ax = axes[row_idx, col_idx]
            cosine_csv = RES / f"exp_c{pair['suffix']}" / "sweep_transitive.csv"
            id_csv     = RES / f"exp_c_idbridge{pair['suffix']}" / "sweep.csv"

            cos_row = _load_row(cosine_csv, scope_cos)
            id_row  = _load_row(id_csv,     scope_id)

            cos_vals = [cos_row[m] if cos_row else np.nan
                        for m, _ in METRICS]
            id_vals  = [id_row[m]  if id_row  else np.nan
                        for m, _ in METRICS]

            if any(not np.isnan(v) for v in cos_vals + id_vals):
                any_data = True

            b1 = ax.bar(
                x - width / 2, [0 if np.isnan(v) else v for v in cos_vals],
                width, label="cosine bridge (original)",
                color=BRIDGE_PALETTE["cosine bridge (original)"],
                edgecolor="white",
            )
            b2 = ax.bar(
                x + width / 2, [0 if np.isnan(v) else v for v in id_vals],
                width, label="identity bridge (ablation)",
                color=BRIDGE_PALETTE["identity bridge (ablation)"],
                edgecolor="white",
            )

            # Number labels above each bar.
            for bars, vals in [(b1, cos_vals), (b2, id_vals)]:
                for rect, v in zip(bars, vals):
                    if np.isnan(v):
                        ax.text(rect.get_x() + rect.get_width() / 2,
                                0.005, "n/a",
                                ha="center", va="bottom",
                                fontsize=7, color="grey")
                    else:
                        ax.text(rect.get_x() + rect.get_width() / 2,
                                v + 0.005, f"{v:.3f}",
                                ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([h for _, h in METRICS])
            ax.set_ylabel("value")
            ax.set_title(f"{pair['label']}  --  {scope_label}", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", alpha=0.3)
            sns.despine(ax=ax)

    if not any_data:
        print(f"[ablation-plot] no data found in any of the four cells; "
              f"run code/ablation_transitive_idbridge.py first.")
        plt.close(fig)
        return

    # Single shared legend at the bottom (one entry per bridge type).
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=BRIDGE_PALETTE[k], edgecolor="white")
        for k in BRIDGE_PALETTE
    ]
    labels = list(BRIDGE_PALETTE.keys())
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=10, frameon=False)

    fig.suptitle(
        r"C-transitive ablation: caption-cosine bridge vs identity bridge",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation-plot] wrote {PLOT_OUT}")


if __name__ == "__main__":
    main()
