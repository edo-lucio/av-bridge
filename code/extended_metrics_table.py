"""Builds the extended metrics table.

For every (method, K, alpha, scope) cell in the sweep CSVs we already have
R@10, routes_correct / routes_total, and NMI. This script adds a new
column: the adaptive per-row mu+sigma core-set hit rate paired with the
mean per-row core size.

The script reads:
  - results/exp_c/sweep_transitive.csv  (FGW transitive)
  - results/exp_d/sweep.csv             (FGW direct, M = caption cos)
  - results/exp_unsup/sweep.csv         (pure GW, no supervision)
  - results/exp_text/sweep.csv          (text-only retrieval baseline)
  - results/exp_random/sweep.csv        (random row-stochastic baseline)

Per-cell transport plans (one per (K, alpha)) are expected under
  results/exp_<x>/plans/T__K{K}__a{alpha:.2f}.npy   (exp_c)
  results/exp_<x>/plans/T__a{alpha:.2f}.npy         (exp_d)

A heldout row-index subset is expected at
  results/exp_<x>/heldout_compare_idx.npy

If a per-cell plan is missing, the core-set column for that cell is
marked "—". The script tracks every missing cell and writes a clear
"missing artifacts" log to results/core_set_analysis/MISSING.md so
that someone with the data + embeddings can re-run only the cells
that need to be filled.

For the text and random baselines we replace the core-set column with
"R@10 / 10" — these are not row-stochastic distributions so the
mu+sigma threshold has no meaningful interpretation on them.

Outputs (under results/core_set_analysis/):
  - extended_table.csv     (machine-readable, full grid)
  - extended_table.json    (same data, JSON form)
  - extended_table.md      (human-readable table)
  - MISSING.md             (list of cells still requiring a re-run)

Usage:
  python code/extended_metrics_table.py
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core_set_metric import core_set_hit_rate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT_DIR = RES / "core_set_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentSpec:
    """One method block in the extended table."""

    label: str                        # e.g. "FGW transitive"
    csv_path: Path                    # sweep csv with R@10/Routes/NMI rows
    plan_fmt: str | None              # "{K}__{a:.2f}" / "{a:.2f}" / None for non-row-stochastic
    plan_dir: Path | None
    heldout_idx_path: Path | None
    metric_kind: str = "core_set"     # "core_set" or "r10_over_10"


SPECS: list[ExperimentSpec] = [
    ExperimentSpec(
        label="GW (unsup, alpha=1.0)",
        csv_path=RES / "exp_unsup" / "sweep.csv",
        plan_fmt=None,  # single plan, see resolve_plan_path
        plan_dir=RES / "exp_unsup",
        heldout_idx_path=RES / "exp_unsup" / "heldout_compare_idx.npy",
    ),
    ExperimentSpec(
        label="FGW direct (M = caption cos)",
        csv_path=RES / "exp_d" / "sweep.csv",
        plan_fmt="a",
        plan_dir=RES / "exp_d" / "plans",
        heldout_idx_path=RES / "exp_d" / "heldout_compare_idx.npy",
    ),
    ExperimentSpec(
        label="FGW transitive (identity bridge)",
        csv_path=RES / "exp_c" / "sweep_transitive.csv",
        plan_fmt="K_a",
        plan_dir=RES / "exp_c" / "plans",
        heldout_idx_path=RES / "exp_c" / "heldout_compare_idx.npy",
    ),
    ExperimentSpec(
        label="Random baseline",
        csv_path=RES / "exp_random" / "sweep.csv",
        plan_fmt=None,
        plan_dir=RES / "exp_random",
        heldout_idx_path=RES / "exp_random" / "heldout_compare_idx.npy",
        metric_kind="core_set",  # random IS row-stochastic, metric applies
    ),
    ExperimentSpec(
        label="Text baseline (cosine sim, NOT row-stochastic)",
        csv_path=RES / "exp_text" / "sweep.csv",
        plan_fmt=None,
        plan_dir=RES / "exp_text",
        heldout_idx_path=RES / "exp_text" / "heldout_compare_idx.npy",
        metric_kind="r10_over_10",
    ),
]


REUSABLE_K = 300
REUSABLE_ALPHA = 0.7


def resolve_plan_path(spec: ExperimentSpec, K: str, alpha: str) -> Path | None:
    """Return the expected plan path for a (K, alpha) cell, or None if N/A.

    Falls back to canonical filenames (T_transitive.npy / T_caption.npy)
    for the canonical (K=300, alpha=0.7) cells so cells already present
    on disk in the un-patched clone are filled too.
    """
    if spec.plan_fmt is None:
        if spec.label.startswith("GW"):
            return spec.plan_dir / "T_gw.npy"
        if spec.label.startswith("Random"):
            return spec.plan_dir / "T_random.npy"
        if spec.label.startswith("Text"):
            return spec.plan_dir / "T_text.npy"
        return None

    if spec.plan_fmt == "a":
        try:
            a = float(alpha)
        except ValueError:
            return None
        per_alpha = spec.plan_dir / f"T__a{a:.2f}.npy"
        if per_alpha.exists():
            return per_alpha
        # Canonical fallback: T_caption.npy lives one level up from plans/
        if abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_caption.npy"
            if canonical.exists():
                return canonical
        return per_alpha  # report missing with the patched path

    if spec.plan_fmt == "K_a":
        try:
            a = float(alpha)
            k = int(K)
        except ValueError:
            return None
        per_cell = spec.plan_dir / f"T__K{k}__a{a:.2f}.npy"
        if per_cell.exists():
            return per_cell
        if k == REUSABLE_K and abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_transitive.npy"
            if canonical.exists():
                return canonical
        return per_cell  # report missing with the patched path
    return None


def fmt_core(hit: float | None, size: float | None) -> str:
    if hit is None or size is None or math.isnan(hit) or math.isnan(size):
        return "—"
    return f"{hit:.3f} / {size:.1f}"


def compute_for_row(spec: ExperimentSpec, csv_row: dict) -> dict:
    """Compute (R@10 / 10) or core-set hit/|core| for one CSV row."""
    out = {
        "method": spec.label,
        "K": csv_row.get("K"),
        "alpha": csv_row.get("alpha"),
        "scope": csv_row.get("scope"),
        "r_at_10": _maybe_float(csv_row.get("R@10")),
        "routes_correct": _maybe_int(csv_row.get("routes_correct")),
        "routes_total": _maybe_int(csv_row.get("routes_total")),
        "nmi": _maybe_float(csv_row.get("nmi")),
        "core_hit": None,
        "core_mean_size": None,
        "metric_kind": spec.metric_kind,
        "plan_path": None,
        "missing_reason": None,
    }

    if spec.metric_kind == "r10_over_10":
        # Baselines: just expose R@10 and use 10 as the implicit denominator.
        return out

    # core_set: need the matching plan and (for heldout) the index file.
    plan_path = resolve_plan_path(spec, str(csv_row.get("K")), str(csv_row.get("alpha")))
    if plan_path is None or not plan_path.exists():
        if plan_path:
            try:
                rel = plan_path.relative_to(ROOT)
            except ValueError:
                rel = plan_path
            out["plan_path"] = str(rel)
            out["missing_reason"] = f"plan file not found: {rel}"
        else:
            out["plan_path"] = None
            out["missing_reason"] = "plan_fmt undefined"
        return out

    T = np.load(plan_path)
    try:
        rel = plan_path.relative_to(ROOT)
    except ValueError:
        rel = plan_path
    out["plan_path"] = str(rel)
    # quick row-stochastic check
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, row_sums.mean(), atol=1e-3):
        out["missing_reason"] = "plan not row-stochastic; core-set metric not applicable"
        return out

    n = T.shape[0]
    gt = np.arange(n)
    scope = csv_row.get("scope")

    if scope == "aggregate":
        hit, size = core_set_hit_rate(T, gt)
    elif scope == "heldout":
        if spec.heldout_idx_path is None or not spec.heldout_idx_path.exists():
            try:
                rel = spec.heldout_idx_path.relative_to(ROOT)
            except (ValueError, AttributeError):
                rel = spec.heldout_idx_path
            out["missing_reason"] = f"heldout subset index not found: {rel}"
            return out
        S_compare = np.load(spec.heldout_idx_path)
        heldout = np.setdiff1d(np.arange(n), S_compare)
        if heldout.size == 0:
            out["missing_reason"] = "heldout subset is empty"
            return out
        hit, size = core_set_hit_rate(T, gt, row_subset=heldout)
    else:
        out["missing_reason"] = f"unknown scope: {scope}"
        return out

    out["core_hit"] = hit
    out["core_mean_size"] = size
    return out


def _maybe_float(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _maybe_int(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def render_markdown(rows: list[dict]) -> str:
    header = ("| Method | K | alpha | scope | R@10 | Routes | NMI | core-hit / |core| |")
    sep = "|---|---:|---:|---|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        if r["metric_kind"] == "r10_over_10":
            r10v = r["r_at_10"]
            core_col = f"{r10v:.3f} / 10 (R@10)" if r10v is not None else "—"
        else:
            core_col = fmt_core(r["core_hit"], r["core_mean_size"])
        r10 = f"{r['r_at_10']:.3f}" if r["r_at_10"] is not None else "—"
        nmi = f"{r['nmi']:.3f}" if r["nmi"] is not None else "—"
        routes = (
            f"{r['routes_correct']}/{r['routes_total']}"
            if r["routes_correct"] is not None and r["routes_total"] is not None
            else "—"
        )
        K = r["K"] if r["K"] not in (None, "") else "—"
        a = r["alpha"] if r["alpha"] not in (None, "") else "—"
        lines.append(
            f"| {r['method']} | {K} | {a} | {r['scope']} | {r10} | {routes} | {nmi} | {core_col} |"
        )
    return "\n".join(lines) + "\n"


def render_missing(rows: list[dict]) -> str:
    missing = [r for r in rows if r["missing_reason"]]
    if not missing:
        return "# Missing artifacts\n\nNone. All core-set cells filled.\n"
    by_method: dict[str, list[dict]] = {}
    for r in missing:
        by_method.setdefault(r["method"], []).append(r)

    lines: list[str] = []
    lines.append("# Missing artifacts for the core-set metric")
    lines.append("")
    lines.append(
        "Every cell listed below could not be filled because the matching "
        "transport plan or heldout-subset file is absent from the local "
        "clone. To fill these cells, run the corresponding experiment "
        "with the patched `code/run_experiments.py` on a machine that "
        "has the av-bridge image / audio / caption embeddings."
    )
    lines.append("")
    lines.append("The patch additions (already applied in this branch):")
    lines.append("")
    lines.append("- `exp_c_transitive` now saves every (K, alpha) plan to "
                 "`results/exp_c/plans/T__K{K}__a{alpha:.2f}.npy`")
    lines.append("- `exp_d_caption` now saves every alpha plan to "
                 "`results/exp_d/plans/T__a{alpha:.2f}.npy`")
    lines.append("- all five image-audio experiments now save the heldout "
                 "row-index subset to `results/exp_<x>/heldout_compare_idx.npy`")
    lines.append("")
    lines.append("After re-running, re-run this script to regenerate the "
                 "table:")
    lines.append("")
    lines.append("```bash")
    lines.append("python code/run_experiments.py --experiment c")
    lines.append("python code/run_experiments.py --experiment d")
    lines.append("python code/run_experiments.py --experiment unsup")
    lines.append("python code/run_experiments.py --experiment text")
    lines.append("python code/run_experiments.py --experiment random")
    lines.append("python code/extended_metrics_table.py")
    lines.append("```")
    lines.append("")
    lines.append("## Missing cells, by method")
    lines.append("")
    for method, ms in by_method.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| K | alpha | scope | reason |")
        lines.append("|---|---:|---|---|")
        for r in ms:
            lines.append(
                f"| {r['K']} | {r['alpha']} | {r['scope']} | "
                f"{r['missing_reason']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    all_rows: list[dict] = []
    for spec in SPECS:
        if not spec.csv_path.exists():
            print(f"  [skip] {spec.label}: no sweep CSV at {spec.csv_path}")
            continue
        with spec.csv_path.open() as f:
            for csv_row in csv.DictReader(f):
                all_rows.append(compute_for_row(spec, csv_row))

    # CSV
    csv_out = OUT_DIR / "extended_table.csv"
    fieldnames = [
        "method", "K", "alpha", "scope", "r_at_10",
        "routes_correct", "routes_total", "nmi",
        "core_hit", "core_mean_size", "metric_kind",
        "plan_path", "missing_reason",
    ]
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # JSON
    json_out = OUT_DIR / "extended_table.json"
    json_out.write_text(json.dumps(all_rows, indent=2, default=str))

    # Markdown
    md_out = OUT_DIR / "extended_table.md"
    md_out.write_text(render_markdown(all_rows))

    # Missing log
    miss_out = OUT_DIR / "MISSING.md"
    miss_out.write_text(render_missing(all_rows))

    total = len(all_rows)
    filled = sum(
        1
        for r in all_rows
        if (r["metric_kind"] == "r10_over_10" and r["r_at_10"] is not None)
        or (r["metric_kind"] == "core_set" and r["core_hit"] is not None)
    )
    print(f"wrote {csv_out}")
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    print(f"wrote {miss_out}")
    print(f"{filled}/{total} cells filled  ({total - filled} missing)")


if __name__ == "__main__":
    main()
