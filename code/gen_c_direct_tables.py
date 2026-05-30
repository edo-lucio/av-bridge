r"""LaTeX K x alpha sweep table for Ridge-supervised FGW (Experiment C-direct).

C-direct varies both the anchor budget K (paired image-audio supervision)
and the FGW blend alpha, so the natural table is K-rows by alpha-column
groups, with three metrics per group (R@10 / AMI / Pearson r).

Usage:
  python code/gen_c_direct_tables.py                       # aggregate scope
  python code/gen_c_direct_tables.py --scope heldout
  python code/gen_c_direct_tables.py --pair textfree       # DINOv2 x MERT (if data exists)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PAIRS = {
    "canonical": {
        "csv":   RES / "exp_c" / "sweep_direct.csv",
        "img":   "CLIP-L/14",
        "aud":   "CLAP-HTSAT-unfused",
        "tag":   "canonical",
    },
    "textfree": {
        "csv":   RES / "exp_c__dinov2-large__mert-330m" / "sweep_direct.csv",
        "img":   "DINOv2-large",
        "aud":   "MERT-330m",
        "tag":   "textfree",
    },
}

ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9]
METRICS = [("R@10", "$R@10$"), ("ami", "AMI"), ("pearson_r", "$r$")]


def _fmt(v: float, is_max: bool) -> str:
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_max else s


def render_table(pair: dict, scope: str) -> str:
    csv_path = pair["csv"]
    if not csv_path.exists():
        return (f"% Skipped: {csv_path} not found. "
                f"Run C-direct for this pair first.\n")
    df = pd.read_csv(csv_path)
    df = df[df.scope == scope].copy()
    if df.empty:
        return (f"% Skipped: {csv_path} has no rows for scope={scope}.\n")

    ks_present = sorted(df.K.unique())
    if not ks_present:
        return "% No K values present.\n"

    body_lines = []
    for K in ks_present:
        sub = df[df.K == K].set_index("alpha")
        rows_alpha = [a for a in ALPHAS if a in sub.index]
        if not rows_alpha:
            continue
        all_vals = [float(sub.loc[a, m])
                    for a in rows_alpha for m, _ in METRICS]
        import math
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
        "\\caption{Ridge-supervised FGW (C-direct) --- image-to-audio: "
        f"{scope}-scope results at $({pair['img']}, {pair['aud']})$. "
        "Rows: anchor budget $K$ (number of paired image--audio rows "
        "used to fit the ridge map). Column groups: FGW blend "
        "$\\alpha$ ($\\alpha=0$: pure Sinkhorn on the ridge-induced "
        "$M$; $\\alpha=1$: pure Gromov--Wasserstein). Each cell "
        "reports $R@10$ / AMI / Pearson $r$; $K_{\\mathrm{cl}}=15$. "
        "Best per metric (within each $K$ row) in bold.}\n"
        f"\\label{{tab:c-direct-{pair['tag']}-{scope}}}\n"
        "\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS) + ["all"],
                    default="canonical")
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"])
    args = ap.parse_args()
    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    for k in keys:
        print(f"% --- {k} pair ({PAIRS[k]['img']} x {PAIRS[k]['aud']}, "
              f"scope={args.scope}) ---")
        print(render_table(PAIRS[k], args.scope))


if __name__ == "__main__":
    main()
