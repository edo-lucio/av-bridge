r"""LaTeX α-sweep tables for Caption-Cost FGW (Experiment D).

Reads one or more per-pair sweep CSVs (default: canonical and text-free
reference pairs) and prints a paper-ready table per pair, bolding the
best value in each metric column. Each table has a single row
(``K=0``) because D has no direct image-audio anchors -- only ``alpha``
varies.

Usage:
  python code/gen_d_alpha_tables.py                       # both pairs
  python code/gen_d_alpha_tables.py --pair canonical      # one pair only
  python code/gen_d_alpha_tables.py --scope heldout
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PAIRS = {
    "canonical": {
        "csv":     RES / "exp_d" / "sweep.csv",
        "img":     "CLIP-L/14",
        "aud":     "CLAP-HTSAT-unfused",
        "tag":     "canonical",
        "label":   "tab:d-alpha-canonical",
    },
    "textfree": {
        "csv":     RES / "exp_d__dinov2-large__mert-330m" / "sweep.csv",
        "img":     "DINOv2-large",
        "aud":     "MERT-330m",
        "tag":     "textfree",
        "label":   "tab:d-alpha-textfree",
    },
}


ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9]
METRICS = [("R@10", "$R@10$"), ("ami", "AMI"), ("pearson_r", "$r$")]


def _fmt(v: float, is_max: bool) -> str:
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_max else s


def render_table(csv_path: Path, img: str, aud: str, tag: str,
                 label: str, scope: str) -> str:
    if not csv_path.exists():
        return (f"% Skipped: {csv_path} not found. "
                f"Run the experiment first.\n")
    df = pd.read_csv(csv_path)
    df = df[(df.K == 0) & (df.scope == scope)].copy()
    df = df.set_index("alpha")
    rows_present = [a for a in ALPHAS if a in df.index]
    if not rows_present:
        return (f"% Skipped: {csv_path} has no rows for K=0, scope={scope}.\n")

    best = {m: df.loc[rows_present, m].max() for m, _ in METRICS}

    cells = []
    for a in ALPHAS:
        if a in df.index:
            for m, _ in METRICS:
                v = float(df.loc[a, m])
                cells.append(_fmt(v, abs(v - best[m]) < 1e-12))
        else:
            cells.extend(["--"] * 3)

    multicols = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{$\\alpha = {a}$}}"
        for a in ALPHAS
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}"
        for i in range(len(ALPHAS))
    )
    metric_header = " & ".join(_h for _, _h in METRICS)
    metric_row = "$K$ & " + " & ".join(
        metric_header for _ in ALPHAS
    ) + " \\\\"
    body_row = "0 & " + " & ".join(cells) + " \\\\"

    return (
        "\\begin{table}[!ht]\n"
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l ccc ccc ccc ccc ccc}\n"
        "\\toprule\n"
        "       & " + multicols + " \\\\\n"
        "       " + cmidrules + "\n"
        + metric_row + "\n"
        "\\midrule\n"
        + body_row + "\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        f"\\caption{{Caption-Cost FGW --- image-to-audio: "
        f"{scope.replace('_', ' ')}-scope results at $({img}, {aud})$. "
        "Each cell reports $R@10$ / AMI / Pearson $r$; $K_{\\mathrm{cl}}=15$. "
        "$K=0$ because Caption-Cost FGW uses no direct image--audio "
        "anchors; only $\\alpha$ varies; $\\alpha$ blends the "
        "cross-modal caption cost $M$ against the intra-modal "
        "structural costs $C_1, C_2$ ($\\alpha=0$: pure Sinkhorn on "
        "$M$ alone; $\\alpha=1$: pure Gromov--Wasserstein on "
        "$(C_1, C_2)$). Best per metric in bold.}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def render_merged_table(pairs: list[dict], scope: str) -> str:
    """Single table, one row per encoder pair. No K column.

    Bolding logic: per row, best of each metric across the alpha sweep
    (same convention as the per-pair table). The first column carries
    the encoder-pair distinction ("CLIP $\\times$ CLAP" etc.).
    """
    rows: list[tuple[str, dict[float, dict[str, float]]]] = []
    for p in pairs:
        if not p["csv"].exists():
            print(f"% Skipped {p['tag']}: {p['csv']} not found.")
            continue
        df = pd.read_csv(p["csv"])
        df = df[(df.K == 0) & (df.scope == scope)].copy()
        if df.empty:
            print(f"% Skipped {p['tag']}: no rows for K=0, scope={scope}.")
            continue
        df = df.set_index("alpha")
        series = {
            a: {m: float(df.loc[a, m]) for m, _ in METRICS}
            for a in ALPHAS if a in df.index
        }
        pair_name = f"{p['img']} $\\times$ {p['aud']}"
        rows.append((pair_name, series))

    if not rows:
        return "% No pairs had data; nothing to render.\n"

    multicols = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{$\\alpha = {a}$}}"
        for a in ALPHAS
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}"
        for i in range(len(ALPHAS))
    )
    metric_header = " & ".join(_h for _, _h in METRICS)
    metric_row = "Pair & " + " & ".join(metric_header for _ in ALPHAS) + " \\\\"

    body_lines = []
    for pair_name, series in rows:
        best = {
            m: max(series[a][m] for a in ALPHAS if a in series)
            for m, _ in METRICS
        }
        cells = []
        for a in ALPHAS:
            if a in series:
                for m, _ in METRICS:
                    v = series[a][m]
                    cells.append(_fmt(v, abs(v - best[m]) < 1e-12))
            else:
                cells.extend(["--"] * 3)
        body_lines.append(f"{pair_name} & " + " & ".join(cells) + " \\\\")

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
        "\\caption{Caption-Cost FGW --- image-to-audio: "
        f"{scope.replace('_', ' ')}-scope results at the two reference "
        "encoder pairs. Each cell reports $R@10$ / AMI / Pearson $r$; "
        "$K_{\\mathrm{cl}}=15$. Caption-Cost FGW uses no direct "
        "image--audio anchors, so only $\\alpha$ varies; "
        "$\\alpha$ blends the cross-modal caption cost $M$ against "
        "the intra-modal structural costs $C_1, C_2$ "
        "($\\alpha=0$: pure Sinkhorn on $M$ alone; "
        "$\\alpha=1$: pure Gromov--Wasserstein on $(C_1, C_2)$). "
        "Best per metric (within each row) in bold.}\n"
        "\\label{tab:d-alpha-merged}\n"
        "\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS) + ["all"], default="all")
    ap.add_argument("--scope", default="aggregate",
                    choices=["aggregate", "heldout"])
    ap.add_argument("--merged", action="store_true",
                    help="Emit a single merged table with one row per pair "
                         "(no K column).")
    args = ap.parse_args()

    if args.merged:
        keys = list(PAIRS) if args.pair == "all" else [args.pair]
        print(f"% --- merged table ({', '.join(keys)}; scope={args.scope}) ---")
        print(render_merged_table([PAIRS[k] for k in keys], args.scope))
        return

    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    for k in keys:
        p = PAIRS[k]
        print(f"% --- {k} pair ({p['img']} x {p['aud']}, scope={args.scope}) ---")
        print(render_table(p["csv"], p["img"], p["aud"], p["tag"],
                           p["label"], args.scope))


if __name__ == "__main__":
    main()
