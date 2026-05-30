r"""LaTeX K x alpha sweep tables for Experiments A (within-image) and
B (within-audio).

Same shape and bolding convention as ``gen_c_direct_tables.py``:
K-rows, alpha-column-groups, three metrics per group (R@10 / AMI /
Pearson r). Best per metric within each K row in bold.

Usage:
  python code/gen_ab_tables.py --exp b --encoder clap-unfused
  python code/gen_ab_tables.py --exp a --encoder clip-large --scope heldout
  python code/gen_ab_tables.py --exp b --encoder clap-fused --scope aggregate
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PRETTY = {
    "clip-base":   "CLIP-B/32",
    "clip-large":  "CLIP-L/14",
    "dinov2-small": "DINOv2-small",
    "dinov2-base":  "DINOv2-base",
    "dinov2-large": "DINOv2-large",
    "vit-mae-base": "ViT-MAE-base",
    "clap-fused":   "CLAP-HTSAT-fused",
    "clap-unfused": "CLAP-HTSAT-unfused",
    "clap-larger":  "CLAP-larger",
    "mert-95m":     "MERT-95m",
    "mert-330m":    "MERT-330m",
}

ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9]
METRICS = [("R@10", "$R@10$"), ("ami", "AMI"), ("pearson_r", "$r$")]


def _fmt(v: float, is_max: bool) -> str:
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_max else s


def _exp_meta(exp: str, encoder: str) -> tuple[str, Path, str, str]:
    """Resolve (display label, csv path, default-encoder name, side)."""
    if exp == "a":
        default = "clip-large"
        suffix = "" if encoder == default else f"__{encoder}"
        csv_path = RES / f"exp_a{suffix}" / "sweep.csv"
        side = "image $\\to$ visual-caption (Experiment A)"
        return PRETTY.get(encoder, encoder), csv_path, default, side
    if exp == "b":
        default = "clap-unfused"
        suffix = "" if encoder == default else f"__{encoder}"
        csv_path = RES / f"exp_b{suffix}" / "sweep.csv"
        side = "audio $\\to$ audio-caption (Experiment B)"
        return PRETTY.get(encoder, encoder), csv_path, default, side
    raise ValueError(f"unknown exp: {exp!r}")


def render_table(exp: str, encoder: str, scope: str) -> str:
    pretty_enc, csv_path, default_enc, side_label = _exp_meta(exp, encoder)
    if not csv_path.exists():
        return (f"% Skipped: {csv_path} not found. "
                f"Run Experiment {exp.upper()} for this encoder first.\n")
    df = pd.read_csv(csv_path)
    df = df[df.scope == scope].copy()
    if df.empty:
        return f"% Skipped: {csv_path} has no rows for scope={scope}.\n"

    ks = sorted(df.K.unique())
    body_lines = []
    for K in ks:
        sub = df[df.K == K].set_index("alpha")
        rows_alpha = [a for a in ALPHAS if a in sub.index]
        if not rows_alpha:
            continue
        # Skip K rows that are entirely NaN (e.g. K=400 heldout when
        # the whole sample is anchors, leaving nothing to evaluate on).
        all_vals = [float(sub.loc[a, m])
                    for a in rows_alpha for m, _ in METRICS]
        if all(math.isnan(v) for v in all_vals):
            continue
        best = {}
        for m, _ in METRICS:
            vals = [float(sub.loc[a, m]) for a in rows_alpha]
            vals = [v for v in vals if not math.isnan(v)]
            best[m] = max(vals) if vals else float("nan")
        cells = []
        for a in ALPHAS:
            if a in sub.index:
                for m, _ in METRICS:
                    v = float(sub.loc[a, m])
                    if math.isnan(v):
                        cells.append("--")
                    else:
                        is_max = (not math.isnan(best[m])
                                  and abs(v - best[m]) < 1e-12)
                        cells.append(_fmt(v, is_max))
            else:
                cells.extend(["--"] * 3)
        body_lines.append(f"{K} & " + " & ".join(cells) + " \\\\")

    if not body_lines:
        return "% No usable rows.\n"

    multicols = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{$\\alpha = {a}$}}"
        for a in ALPHAS
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}"
        for i in range(len(ALPHAS))
    )
    metric_header = " & ".join(_h for _, _h in METRICS)
    metric_row = "$K$ & " + " & ".join(metric_header for _ in ALPHAS) + " \\\\"

    return (
        "\\begin{table}[!ht]\n"
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l ccc ccc ccc ccc ccc}\n"
        "\\toprule\n"
        "      & " + multicols + " \\\\\n"
        "      " + cmidrules + "\n"
        + metric_row + "\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\caption{Ridge-supervised FGW within-modality: "
        f"{side_label}, encoder $= \\text{{{pretty_enc}}}$, "
        f"{scope}-scope. Rows: anchor budget $K$ (paired source$\\to$caption "
        "rows used to fit the ridge map). Column groups: FGW blend "
        "$\\alpha$ ($\\alpha=0$: pure Sinkhorn on $M$; $\\alpha=1$: pure "
        "Gromov--Wasserstein on $(C_1, C_2)$). Each cell reports $R@10$ "
        "/ AMI / Pearson $r$; $K_{\\mathrm{cl}}$ matches the chapter "
        "convention. Best per metric (within each $K$ row) in bold.}\n"
        f"\\label{{tab:exp-{exp}-{encoder}-{scope}}}\n"
        "\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["a", "b"])
    ap.add_argument("--encoder", required=True,
                    help="Image encoder name for --exp a, audio encoder "
                         "name for --exp b. E.g. clip-large, clap-unfused.")
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"])
    args = ap.parse_args()
    print(f"% --- Experiment {args.exp.upper()} | encoder={args.encoder} "
          f"| scope={args.scope} ---")
    print(render_table(args.exp, args.encoder, args.scope))


if __name__ == "__main__":
    main()
