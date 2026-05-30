r"""LaTeX cross-pair comparison table for Pure Gromov-Wasserstein.

Pure-GW has no \alpha knob and no image-audio anchors -- the plan is
fully determined by the two intra-modal distance geometries
(C_1, C_2). The table therefore has one row per encoder pair and the
metric columns R@10, NMI, Pearson r at a single chosen scope.

Usage:
  python code/gen_unsup_tables.py                       # aggregate scope (default)
  python code/gen_unsup_tables.py --scope heldout
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

GRID_CSV = RES / "exp_grid" / "sweep.csv"

PAIRS = [
    {"img_enc": "clip-large",   "aud_enc": "clap-unfused",
     "img":     "CLIP-L/14",    "aud":     "CLAP-HTSAT-unfused"},
    {"img_enc": "clip-large",   "aud_enc": "mert-330m",
     "img":     "CLIP-L/14",    "aud":     "MERT-330m"},
    {"img_enc": "dinov2-large", "aud_enc": "clap-unfused",
     "img":     "DINOv2-large", "aud":     "CLAP-HTSAT-unfused"},
    {"img_enc": "dinov2-large", "aud_enc": "mert-330m",
     "img":     "DINOv2-large", "aud":     "MERT-330m"},
]

METRICS = [
    ("R@1",         "$R@1$"),
    ("R@10",        "$R@10$"),
    ("R@20",        "$R@20$"),
    ("nmi",         "NMI"),
    ("ami",         "AMI"),
    ("pearson_r",   "$r$"),
    ("knn_overlap", "kNN"),
]

ROUTE_HEADER = "route"


def _fmt(v: float, is_max: bool) -> str:
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_max else s


def _fmt_route(correct: int, total: int, is_max: bool) -> str:
    s = f"{correct}/{total}"
    return f"\\textbf{{{s}}}" if is_max else s


def render_table(pairs: list[dict], scope: str,
                 grid_csv: Path = GRID_CSV) -> str:
    if not grid_csv.exists():
        return f"% Skipped: {grid_csv} not found.\n"
    df = pd.read_csv(grid_csv)
    df = df[(df.experiment == "unsup") & (df.scope == scope)]

    rows: list[tuple[str, dict[str, float], tuple[int, int]]] = []
    for p in pairs:
        sub = df[(df.image_encoder == p["img_enc"])
                 & (df.audio_encoder == p["aud_enc"])]
        if sub.empty:
            print(f"% Skipped {p['img']} x {p['aud']}: no row in grid "
                  f"CSV for unsup at scope={scope}.")
            continue
        r = sub.iloc[0]
        pair_name = f"{p['img']} $\\times$ {p['aud']}"
        route = (int(r["routes_correct"]), int(r["routes_total"]))
        rows.append((pair_name,
                     {m: float(r[m]) for m, _ in METRICS},
                     route))

    if not rows:
        return "% No pairs had data; nothing to render.\n"

    best = {m: max(d[m] for _, d, _ in rows) for m, _ in METRICS}
    best_route_ratio = max(
        rc / rt if rt > 0 else 0.0 for _, _, (rc, rt) in rows
    )

    n_metric_cols = len(METRICS) + 1  # +1 for route
    col_spec = "l " + "c" * n_metric_cols
    metric_header = " & ".join(_h for _, _h in METRICS) + " & " + ROUTE_HEADER

    body_lines = []
    for pair_name, d, (rc, rt) in rows:
        cells = [_fmt(d[m], abs(d[m] - best[m]) < 1e-12) for m, _ in METRICS]
        ratio = rc / rt if rt > 0 else 0.0
        cells.append(_fmt_route(rc, rt, abs(ratio - best_route_ratio) < 1e-12))
        body_lines.append(f"{pair_name} & " + " & ".join(cells) + " \\\\")

    return (
        "\\begin{table}[!ht]\n"
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        "Pair & " + metric_header + " \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\caption{Pure Gromov--Wasserstein --- image-to-audio: "
        f"{scope.replace('_', ' ')}-scope results across the four "
        "regime-crossed encoder pairs (text-aligned vs text-free on "
        "each side). Columns: top-$k$ retrieval at "
        "$k\\in\\{1,10,20\\}$; structural agreement via NMI / AMI / "
        "identity-Pearson $r$ (pairwise distances); kNN overlap; "
        "semantic-route accuracy (correct out of total query "
        "categories). $K_{\\mathrm{cl}}=15$. Pure-GW has no $\\alpha$ "
        "knob and no image--audio anchors: the plan is determined "
        "entirely by the alignment of the two intra-modal distance "
        "geometries $C_1, C_2$ (equivalent to the FGW $\\alpha = 1$ "
        "endpoint). Best per column in bold.}\n"
        f"\\label{{tab:unsup-twopair-{scope}}}\n"
        "\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"])
    args = ap.parse_args()
    print(f"% --- Pure-GW two-pair table (scope={args.scope}) ---")
    print(render_table(PAIRS, args.scope))


if __name__ == "__main__":
    main()
