r"""Standalone LaTeX document of all C-direct (Ridge-supervised FGW) results.

Combines:
  - K x alpha sweep at the canonical pair (CLIP-L x CLAP-unfused),
    aggregate and held-out scopes.
  - Cross-encoder summary at (K=300, alpha=0.5) from the grid sweep,
    held-out scope.

Output: prints a complete article-class LaTeX document to stdout.
Run:  python code/gen_c_direct_doc.py > /tmp/c_direct.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from gen_c_direct_tables import PAIRS, render_table  # noqa: E402

RES = ROOT / "results"
GRID_CSV = RES / "exp_grid" / "sweep.csv"


def render_encoder_grid_table(scope: str = "heldout") -> str:
    if not GRID_CSV.exists():
        return f"% Skipped grid table: {GRID_CSV} not found.\n"
    df = pd.read_csv(GRID_CSV)
    df = df[(df.experiment == "c-direct") & (df.scope == scope)
            & (df.K == 300) & (df.alpha == 0.5)]
    if df.empty:
        return f"% No C-direct rows in grid at K=300, alpha=0.5, scope={scope}.\n"

    df = df.sort_values("R@10", ascending=False)

    best = {m: df[m].max() for m in ["R@1", "R@10", "R@20", "ami",
                                     "pearson_r", "knn_overlap"]}

    def fmt(v: float, col: str) -> str:
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if abs(v - best[col]) < 1e-12 else s

    body_lines = []
    for _, r in df.iterrows():
        cells = [
            r["image_encoder"], r["audio_encoder"],
            fmt(r["R@1"], "R@1"),
            fmt(r["R@10"], "R@10"),
            fmt(r["R@20"], "R@20"),
            fmt(r["ami"], "ami"),
            fmt(r["pearson_r"], "pearson_r"),
            fmt(r["knn_overlap"], "knn_overlap"),
            f"{int(r['routes_correct'])}/{int(r['routes_total'])}",
        ]
        body_lines.append(" & ".join(cells) + " \\\\")

    header = ("image enc.\\ & audio enc.\\ & $R@1$ & $R@10$ & $R@20$ "
              "& AMI & $r$ & kNN & route \\\\")
    return (
        "\\begin{table}[!ht]\n"
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{ll cccccc c}\n"
        "\\toprule\n"
        + header + "\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{Ridge-supervised FGW (C-direct) at $K=300$, "
        f"$\\alpha=0.5$, {scope} scope, across every available "
        "image$\\times$audio encoder pair (sorted by $R@10$ descending). "
        "Each row reports top-$k$ retrieval, AMI, identity-Pearson $r$, "
        "kNN overlap, and semantic-route accuracy. Best per metric column "
        "in bold.}\n"
        f"\\label{{tab:c-direct-grid-{scope}}}\n"
        "\\end{table}\n"
    )


def main() -> None:
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{array}
\usepackage{hyperref}

\title{C-direct (Ridge-supervised FGW): image $\to$ audio results}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Overview}
The Ridge-supervised FGW recipe (\emph{C-direct}) fits a closed-form
ridge map from $K$ paired image--audio anchor rows, projects all
source rows through it to build the cross-modal cost matrix
$M[i,j] = \lVert (XW)_i - Y_j \rVert^2$, and then runs entropic FGW with
intra-modal costs $C_1, C_2$ from pairwise sqdists in each space.
The blend $\alpha$ runs from $\alpha=0$ (pure Sinkhorn on $M$ alone)
to $\alpha=1$ (pure Gromov--Wasserstein on $(C_1, C_2)$).

Tables \ref{tab:c-direct-canonical-aggregate} and
\ref{tab:c-direct-canonical-heldout} sweep $K \times \alpha$ at the
canonical encoder pair (CLIP-L/14 $\times$ CLAP-HTSAT-unfused).
Aggregate scope evaluates on all rows (including the anchors used to
fit the ridge map); the held-out scope evaluates on rows outside the
anchor partition and is the meaningful generalisation column. At
$K=400$ aggregate-scope is fully supervised, so no held-out row
remains.

Table \ref{tab:c-direct-grid-heldout} summarises every available
image$\times$audio encoder pair at the canonical operating point
$(K=300, \alpha=0.5)$ at held-out scope. Within each row of the
$K \times \alpha$ tables, the best value per metric across the
$\alpha$ sweep is bolded. In the encoder-grid table the bolding is
column-wise (across pairs).
"""

    parts = [preamble]
    parts.append("\n\\section*{Canonical pair: $K \\times \\alpha$ sweep "
                 "(aggregate)}\n")
    parts.append(render_table(PAIRS["canonical"], "aggregate"))
    parts.append("\n\\section*{Canonical pair: $K \\times \\alpha$ sweep "
                 "(held-out)}\n")
    parts.append(render_table(PAIRS["canonical"], "heldout"))
    parts.append("\n\\section*{Encoder-pair summary at $(K=300, "
                 "\\alpha=0.5)$, held-out}\n")
    parts.append(render_encoder_grid_table("heldout"))
    parts.append("\n\\end{document}\n")
    sys.stdout.write("".join(parts))


if __name__ == "__main__":
    main()
