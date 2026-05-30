"""Convert every result CSV under `results/` into a LaTeX table.

Output is a single `results/tables.tex` file containing one LaTeX table per
CSV, each wrapped in a `\\begin{table}...\\end{table}` environment with a
descriptive `\\caption{...}` and a `\\label{tab:...}` suitable for
cross-referencing from the Overleaf report.

Run:
    python code/csv_to_tex.py
    python code/csv_to_tex.py --out report/tables.tex

The generated file:

  * uses the `booktabs` package (`\\toprule`, `\\midrule`, `\\bottomrule`);
  * uses `graphicx` for `\\resizebox` on the wide cross-method comparison
    tables;
  * uses `longtable` for the per-experiment full sweep CSVs that exceed one
    page;
  * is `\\input{...}`-ready: it contains only table environments and
    comments, no document preamble.

Tables included:

  1.  Cross-experiment retrieval and structural comparison
      (`results/comparison.csv`, `results/comparison_structure.csv`).
  2.  Per-experiment K × α retrieval grids (`grid_aggregate.csv`,
      `grid_heldout.csv`) for Experiments A, B, C-direct, D, Unsup, Text.
  3.  C-transitive single-cell summary (`exp_c/sweep_transitive.csv`).
  4.  Per-experiment full sweeps (`sweep.csv`) — every (K, α, scope) cell
      with all retrieval, routing, and cluster-agreement metrics. Long;
      typeset via `longtable`.
  5.  Cross-modal Pearson-r scatter summary (concatenation of the four
      `exp_grid/plots/scatter_r_*.csv` files).

Confusion-matrix CSVs (15 × 15 numeric matrices, one per method) are
intentionally skipped — they are already reported as the
`results/exp_grid/plots/cross_modal_cluster_confusions__K15_soft.png`
figure, which is the right surface for a 15 × 15 matrix in print.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _clean_header(name: str) -> str:
    """Replace underscores and a few CSV quirks with LaTeX-friendly tokens."""
    return (
        name.replace("_", r"\_")
            .replace("alpha=", r"$\alpha$=")
            .replace("R@", r"R$@$")
    )


def _render_tabular(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    """Render a DataFrame to a `tabular` body using `booktabs` rules."""
    df = df.copy()
    # Clean cell-level NaN/"nan" strings to em-dash so the table reads as
    # the comparison CSVs do (e.g., "—" for an absent scope).
    df = df.fillna(r"\textemdash")
    df = df.map(
        lambda v: r"\textemdash" if (isinstance(v, str)
                                     and v.strip().lower() == "nan")
        else v
    )
    df.columns = [_clean_header(str(c)) for c in df.columns]
    return df.to_latex(
        index=False, escape=False,
        column_format="l" * df.shape[1],
        float_format=float_fmt,
    )


def render_table(df: pd.DataFrame, caption: str, label: str,
                 small: bool = True, resize: bool = False,
                 longtable: bool = False) -> str:
    """Wrap a DataFrame in `\\begin{table}...\\end{table}` with caption / label.

    - `small`     → adds `\\small` for verbose tables.
    - `resize`    → wraps the tabular in `\\resizebox{\\textwidth}{!}{...}`.
    - `longtable` → uses a `longtable` environment instead, for sweeps that
                    do not fit on a single page.
    """
    body = _render_tabular(df)
    if longtable:
        cols = "l" * df.shape[1]
        df_for_lt = df.copy()
        df_for_lt = df_for_lt.fillna(r"\textemdash")
        df_for_lt.columns = [_clean_header(str(c)) for c in df_for_lt.columns]
        rows = []
        for _, row in df_for_lt.iterrows():
            cells = [
                f"{v:.3f}" if isinstance(v, float) else str(v)
                for v in row.values
            ]
            rows.append(" & ".join(cells) + r" \\")
        body_rows = "\n".join(rows)
        header = " & ".join(df_for_lt.columns) + r" \\"
        return (
            f"\\begin{{longtable}}{{{cols}}}\n"
            f"\\caption{{{caption}}}\\label{{tab:{label}}}\\\\\n"
            f"\\toprule\n{header}\n\\midrule\n\\endfirsthead\n"
            f"\\multicolumn{{{df.shape[1]}}}{{l}}"
            f"{{\\small\\itshape continued from previous page}}\\\\\n"
            f"\\toprule\n{header}\n\\midrule\n\\endhead\n"
            f"\\midrule\n\\multicolumn{{{df.shape[1]}}}{{r}}"
            f"{{\\small\\itshape continued on next page}}\\\\\n\\endfoot\n"
            f"\\bottomrule\n\\endlastfoot\n"
            f"{body_rows}\n"
            f"\\end{{longtable}}\n"
        )

    size = r"\small" if small else ""
    open_env = "\\begin{table}[!ht]\n\\centering\n" + (size + "\n" if size else "")
    inner = body.strip()
    if resize:
        inner = "\\resizebox{\\textwidth}{!}{%\n" + inner + "\n}"
    close_env = (
        f"\n\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\end{{table}}\n"
    )
    return open_env + inner + close_env


def emit_comparison(out: list[str]) -> None:
    """Cross-method comparison tables (retrieval + structure)."""
    out.append("% =====================================================================\n"
               "% 1. Cross-experiment comparison tables\n"
               "% =====================================================================\n")
    for fname, label, cap in [
        ("comparison.csv",
         "comparison-retrieval",
         "Cross-experiment retrieval comparison on the AVCaps subset "
         "($n = 400$, canonical encoder pair CLIP-L/14 + CLAP-HTSAT-unfused). "
         "Each cell is reported as "
         "$R@10 \\,/\\, \\text{routes\\_correct}/K_{\\text{cl}}$. "
         "Column suffix \\texttt{\\_agg} = aggregate scope (all $n$ rows); "
         "\\texttt{\\_hel} = held-out scope (for ridge-supervised methods) "
         "or same-rows scope (for the no-supervision methods, on the same "
         "100-row reference partition outside A's and B's $K = 300$ anchor "
         "sets). The \\texttt{elbow} row reports the elbow $K$ from each "
         "experiment's $R@10$ sweep; \\texttt{K=300@$\\alpha$=0.5} reports "
         "the canonical operating point. The remaining rows are the "
         "no-supervision baselines (D, Unsup, Text-only) and "
         "C-transitive."),
        ("comparison_structure.csv",
         "comparison-structure",
         "Cross-experiment structural comparison (mirrors "
         "Table~\\ref{tab:comparison-retrieval}). Each cell reports "
         "NMI (uncorrected), AMI (chance-corrected), kNN overlap, and "
         "Pearson $r$ on pairwise distances. Higher is better for all four."),
    ]:
        df = pd.read_csv(RES / fname)
        out.append(render_table(df, cap, label, small=True, resize=True))


def _grid_caption(exp_label: str, scope: str) -> str:
    scope_phrase = ("aggregate scope (averaged over all $n = 400$ rows)"
                    if scope == "aggregate"
                    else "held-out scope (rows outside the $K$-anchor set; "
                         "the $K = 400$ row has no held-out partition)")
    return (
        f"Experiment {exp_label}: $K \\times \\alpha$ grid of $R@10 / "
        f"\\text{{routes\\_correct}}/K_{{\\text{{cl}}}}$ at {scope_phrase}. "
        f"Rows are supervision strength $K$, columns are the FGW blend "
        f"$\\alpha$ (0 = pure Sinkhorn, 1 = pure GW)."
    )


def _single_grid_caption(exp_label: str, exp_kind: str, scope: str) -> str:
    if exp_kind == "d":
        return (
            f"Experiment {exp_label}: $\\alpha$ ablation at "
            f"{'aggregate' if scope == 'aggregate' else 'same-rows'} scope. "
            f"$K = 0$ because the recipe uses no anchors; "
            f"the same-rows row is reported only at the canonical "
            f"$\\alpha = 0.7$."
        )
    if exp_kind == "unsup":
        return (
            f"Experiment {exp_label} (pure entropic GW): single-cell "
            f"summary at "
            f"{'aggregate' if scope == 'aggregate' else 'same-rows'} scope. "
            f"$\\alpha = 1$ (pure GW); no $K$ axis."
        )
    if exp_kind == "text":
        return (
            f"Text-only baseline: single-cell summary at "
            f"{'aggregate' if scope == 'aggregate' else 'same-rows'} scope. "
            f"$T = Z^{{\\text{{vis}}}} (Z^{{\\text{{aud}}}})^{{\\top}}$ is "
            f"the raw caption-cosine matrix; no $K$ axis and "
            f"$\\alpha$ undefined."
        )
    return ""


def emit_grid_tables(out: list[str]) -> None:
    """Per-experiment K × α grids (or single-cell α tables) for the
    aggregate and held-out scopes."""
    sections = [
        ("Experiment~A — image $\\to$ visual-caption text", "exp_a", "a", "sweep"),
        ("Experiment~B — audio $\\to$ audio-caption text", "exp_b", "b", "sweep"),
        ("Experiment~C-direct — image $\\to$ audio", "exp_c", "c", "sweep"),
        ("Experiment~D — caption-cost FGW", "exp_d", "d", "single"),
        ("Experiment~Unsup — pure entropic GW", "exp_unsup", "unsup", "single"),
        ("Text-only baseline", "exp_text", "text", "single"),
    ]
    out.append("% =====================================================================\n"
               "% 2. Per-experiment retrieval grids\n"
               "% =====================================================================\n")
    for title, dirname, key, kind in sections:
        out.append(f"% --- {title} ---\n")
        for scope_file, scope_name, scope_label in [
            ("grid_aggregate.csv", "aggregate", "agg"),
            ("grid_heldout.csv",   "heldout",   "hel"),
        ]:
            path = RES / dirname / scope_file
            if not path.exists():
                continue
            df = pd.read_csv(path)
            cap = (_grid_caption(title, scope_name) if kind == "sweep"
                   else _single_grid_caption(title, key, scope_name))
            label = f"{key}-{scope_label}"
            out.append(render_table(df, cap, label,
                                    small=True, resize=False))


def emit_transitive(out: list[str]) -> None:
    """Single-cell C-transitive summary."""
    path = RES / "exp_c" / "sweep_transitive.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    cap = (
        "Experiment~C-transitive — image $\\to$ audio through the text "
        "bridge. Single canonical operating point: per-leg $K = 300$, "
        "$\\alpha = 0.7$; bridge top-$k = 20$, softmax temperature "
        "$\\tau = 0.1$. The composed plan uses no image$\\leftrightarrow$"
        "audio labels. Reported at both aggregate and held-out scope."
    )
    out.append("% =====================================================================\n"
               "% 3. C-transitive single-cell summary\n"
               "% =====================================================================\n")
    out.append(render_table(df, cap, "c-transitive",
                            small=True, resize=True))


def emit_full_sweeps(out: list[str]) -> None:
    """Per-experiment full sweep CSVs (every K × α × scope cell)."""
    sweeps = [
        ("Experiment~A", "exp_a", "sweep.csv", "a"),
        ("Experiment~B", "exp_b", "sweep.csv", "b"),
        ("Experiment~C-direct", "exp_c", "sweep_direct.csv", "c"),
        ("Experiment~D", "exp_d", "sweep.csv", "d"),
        ("Experiment~Unsup", "exp_unsup", "sweep.csv", "unsup"),
        ("Text-only baseline", "exp_text", "sweep.csv", "text"),
    ]
    out.append("% =====================================================================\n"
               "% 4. Per-experiment full sweep tables (all metrics, every cell)\n"
               "% =====================================================================\n")
    for title, dirname, fname, key in sweeps:
        path = RES / dirname / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n_rows = len(df)
        cap = (
            f"{title}: full per-cell metric suite. Each row is one "
            f"$(K, \\alpha, \\text{{scope}})$ cell with retrieval "
            f"($R@1, R@5, R@10, R@20$), cluster routing, structural "
            f"fidelity (kNN overlap, Pearson $r$), and the cluster-agreement "
            f"family (NMI, AMI, ARI, V-measure, homogeneity, completeness). "
            f"Total: {n_rows} rows."
        )
        # The full sweep for A/B/C has many rows; D/Unsup/Text have few.
        use_long = n_rows > 25
        out.append(render_table(df, cap, f"sweep-{key}",
                                small=True, resize=not use_long,
                                longtable=use_long))


def _pick_row(df: pd.DataFrame, **filt) -> pd.Series | None:
    """Return the first row matching every column=value in `filt`, or None."""
    mask = pd.Series(True, index=df.index)
    for col, val in filt.items():
        if col not in df.columns:
            return None
        if isinstance(val, float):
            import numpy as _np
            mask &= _np.isclose(df[col].astype(float), val)
        else:
            mask &= df[col] == val
    sub = df[mask]
    return sub.iloc[0] if not sub.empty else None


def emit_d_vs_gw(out: list[str]) -> None:
    """Head-to-head D (caption-cost FGW) vs Pure-GW across the three
    evaluation families, on the canonical encoder pair."""
    d_path = RES / "exp_d" / "sweep.csv"
    u_path = RES / "exp_unsup" / "sweep.csv"
    if not d_path.exists() or not u_path.exists():
        return
    d = pd.read_csv(d_path)
    u = pd.read_csv(u_path)

    d_agg = _pick_row(d, alpha=0.7, scope="aggregate")
    d_hel = _pick_row(d, alpha=0.7, scope="heldout")
    u_agg = _pick_row(u, alpha=1.0, scope="aggregate")
    u_hel = _pick_row(u, alpha=1.0, scope="heldout")
    if any(r is None for r in (d_agg, d_hel, u_agg, u_hel)):
        return

    def fmt(v):
        return "\\textemdash" if (v is None or pd.isna(v)) else f"{float(v):.3f}"

    rows = [
        ("Retrieval",           "$R@10$ (same-rows)",
            fmt(d_hel["R@10"]), fmt(u_hel["R@10"])),
        ("Retrieval",           "$R@10$ (aggregate)",
            fmt(d_agg["R@10"]), fmt(u_agg["R@10"])),
        ("Cluster alignment",   "NMI (same-rows, uncorrected)",
            fmt(d_hel["nmi"]),  fmt(u_hel["nmi"])),
        ("Cluster alignment",   "AMI (same-rows, chance-corrected)",
            fmt(d_hel["ami"]),  fmt(u_hel["ami"])),
        ("Cluster alignment",   "AMI (aggregate)",
            fmt(d_agg["ami"]),  fmt(u_agg["ami"])),
        ("Structural fidelity", "Pearson $r$ (same-rows)",
            fmt(d_hel["pearson_r"]), fmt(u_hel["pearson_r"])),
        ("Structural fidelity", "Pearson $r$ (aggregate)",
            fmt(d_agg["pearson_r"]), fmt(u_agg["pearson_r"])),
        ("Structural fidelity", "$k$-NN overlap (same-rows)",
            fmt(d_hel["knn_overlap"]), fmt(u_hel["knn_overlap"])),
    ]
    df = pd.DataFrame(rows, columns=[
        "Family", "Metric", "D (caption-cost FGW)", "Pure-GW",
    ])
    cap = (
        "Head-to-head comparison of Experiment~D (caption-cost FGW, "
        "$\\alpha = 0.7$) against the Pure-GW reference ($\\alpha = 1$), "
        "on the canonical CLIP-L/14 $\\times$ CLAP-HTSAT-unfused pair. "
        "Same-rows scope is the 100-row reference partition outside A's "
        "and B's $K = 300$ anchor sets; aggregate scope averages over all "
        "$n = 400$ rows. D wins retrieval by roughly $4\\times$, Pure-GW "
        "wins global geometry (Pearson $r$, $k$-NN overlap), and the two "
        "methods are essentially indistinguishable on chance-corrected "
        "cluster alignment (AMI same-rows). See "
        "Section~\\ref{sec:res-d}/\\ref{sec:res-gw} for the mechanism."
    )
    out.append("% =====================================================================\n"
               "% 6. Synthesised head-to-head: D vs Pure-GW\n"
               "% =====================================================================\n")
    out.append(render_table(df, cap, "d-vs-gw",
                            small=True, resize=False))


def emit_ami_interpretation(out: list[str]) -> None:
    """One-line AMI reading per method, sourced from the per-experiment
    sweep CSVs at each method's canonical operating point and same-rows
    (or held-out) scope."""
    sources = [
        # (label, csv_path, filter dict, reading)
        ("C-transitive (text bridge)",
         RES / "exp_c" / "sweep_transitive.csv",
         {"scope": "heldout"},
         "Strong: the text-bridge composition preserves the most "
         "category-level structure of any method in the study."),
        ("D — caption-cost FGW",
         RES / "exp_d" / "sweep.csv",
         {"alpha": 0.7, "scope": "heldout"},
         "Moderate--strong: captions transport categorical info from one "
         "modality to the other."),
        ("Pure-GW",
         RES / "exp_unsup" / "sweep.csv",
         {"alpha": 1.0, "scope": "heldout"},
         "Moderate--strong: intra-modal geometry alone aligns categories "
         "--- the ``shared relational structure'' claim survives chance "
         "correction."),
        ("Text-only (caption cosine)",
         RES / "exp_text" / "sweep.csv",
         {"scope": "heldout"},
         "Moderate: raw caption cosine produces real categorical "
         "alignment but less than FGW or text-bridging."),
        ("C-direct (supervised ridge)",
         RES / "exp_c" / "sweep_direct.csv",
         {"K": 300, "alpha": 0.5, "scope": "heldout"},
         "Weak: supervised ridge optimises exemplar identity, not "
         "categorical structure --- AMI sits below every no-supervision "
         "method in the comparison."),
    ]

    rows = []
    for label, path, filt, reading in sources:
        if not path.exists():
            continue
        df_in = pd.read_csv(path)
        r = _pick_row(df_in, **filt)
        if r is None or "ami" not in df_in.columns:
            continue
        rows.append({
            "Method":  label,
            "AMI":     f"{float(r['ami']):.3f}",
            "Reading": reading,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    cap = (
        "Adjusted Mutual Information (AMI) reading per method on the "
        "100-row same-rows / held-out partition, canonical encoder pair. "
        "AMI is the chance-corrected version of NMI: $\\mathrm{AMI} = 0$ "
        "corresponds to random labellings, $\\mathrm{AMI} = 1$ to "
        "identical partitions. The cross-method ordering on AMI inverts "
        "the unsupervised-vs-supervised intuition: C-transitive is the "
        "strongest at category level despite chance retrieval, while "
        "C-direct (supervised) sits at the bottom because ridge optimises "
        "exemplar identity, not cluster structure."
    )
    out.append("% =====================================================================\n"
               "% 7. AMI per-method reading\n"
               "% =====================================================================\n")
    out.append(render_table(df, cap, "ami-reading",
                            small=True, resize=True))


def emit_scatter_r(out: list[str]) -> None:
    """Pearson-r scatter scalars across the four cross-modal plans."""
    rows = []
    for sub_dir, label in [
        ("exp_d/T_caption",     "D (caption-cost FGW)"),
        ("exp_c/T_transitive",  "C-transitive (text bridge)"),
        ("exp_unsup/T_gw",      "Pure-GW"),
        ("exp_text/T_text",     "Text-only (caption cosine)"),
    ]:
        path = RES / "exp_grid" / "plots" / f"scatter_r_{sub_dir.replace('/', '_')}.csv"
        if not path.exists():
            continue
        rec = pd.read_csv(path).iloc[0]
        rows.append({
            "Method":          label,
            "Plan file":       rec["plan"],
            "Pearson $r$":     f"{rec['pearson_r_full']:.4f}",
            "$n_{\\text{pairs}}$": int(rec["n_pairs"]),
            "Subsample (plot)":   int(rec["n_subsample"]),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    cap = (
        "Cross-modal pairwise-distance Pearson $r$ per plan. For each "
        "saved plan $T$, the partner mapping "
        "$\\hat\\jmath(i) = \\arg\\max_j T_{ij}$ is decoded and $r$ is "
        "computed between the full set of source-space pairwise distances "
        "$\\| X_i - X_{i'} \\|$ and the partner-target-space pairwise "
        "distances $\\| Y_{\\hat\\jmath(i)} - Y_{\\hat\\jmath(i')} \\|$. "
        "High $r$ = pairs that are close in CLIP are mapped to pairs "
        "that are close in CLAP; low $r$ = pairwise geometry scrambled "
        "under the plan, even when categorical correspondence (NMI / "
        "AMI) is high. Scatter figures: "
        "\\texttt{results/exp\\_grid/plots/cross\\_modal\\_pearson\\_scatter.png}."
    )
    out.append("% =====================================================================\n"
               "% 5. Cross-modal Pearson-r scatter summary\n"
               "% =====================================================================\n")
    out.append(render_table(df, cap, "pearson-r-scatter",
                            small=True, resize=False))


HEADER = r"""% =====================================================================
% tables.tex
% Auto-generated from the CSVs under results/.  Re-run with:
%     python code/csv_to_tex.py
% This file is `\input{...}`-ready: it contains only \begin{table}...\end{table}
% environments. The host document must load:
%
%     \usepackage{booktabs}     % \toprule / \midrule / \bottomrule
%     \usepackage{graphicx}     % \resizebox on the wide comparison tables
%     \usepackage{longtable}    % multi-page sweep tables
%
% Tables are organised in seven sections:
%   1. cross-experiment comparison (retrieval + structure)
%   2. per-experiment K x alpha retrieval grids
%   3. C-transitive single-cell summary
%   4. per-experiment full sweep tables (every metric, every cell)
%   5. cross-modal Pearson-r scatter summary
%   6. head-to-head D vs Pure-GW synthesis
%   7. AMI per-method reading
%
% Confusion-matrix CSVs are intentionally not typeset as tables (15 x 15 of
% them); they appear as the figure
%   results/exp_grid/plots/cross_modal_cluster_confusions__K15_soft.png
% =====================================================================
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(RES / "tables.tex"),
                    help="Output .tex path (default: results/tables.tex).")
    args = ap.parse_args()

    chunks: list[str] = [HEADER]
    emit_comparison(chunks)
    emit_grid_tables(chunks)
    emit_transitive(chunks)
    emit_full_sweeps(chunks)
    emit_scatter_r(chunks)
    emit_d_vs_gw(chunks)
    emit_ami_interpretation(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks))
    print(f"[tables] wrote {out_path}  ({out_path.stat().st_size:,} bytes)")
    n_tables = sum(c.count("\\begin{table}") + c.count("\\begin{longtable}")
                   for c in chunks)
    print(f"[tables] {n_tables} table environments emitted")


if __name__ == "__main__":
    main()
