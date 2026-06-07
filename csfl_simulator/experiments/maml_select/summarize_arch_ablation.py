"""Summarize the selector-architecture (meta-policy capacity) ablation.

Reads result.json files produced by the ``arch_ablation`` profile and writes a
compact CSV summary plus a publication-style LaTeX table reporting final accuracy,
fairness, and per-round selection overhead as a function of the policy MLP width.
It does not run any training.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RUNS = REPO_ROOT / "runs" / "maml_select_arch_ablation"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "maml_select" / "arch_ablation"

EXPERIMENT_ID = "architecture_ablation"


def _params_for_width(hidden_dim: int) -> int:
    """Trainable parameter count of the 6-h-h-1 ReLU MLP (matches manuscript: 4,673 at h=64)."""
    h = int(hidden_dim)
    return h * h + 9 * h + 1


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_hidden(label: str, params: Dict[str, Any]) -> float:
    if "hidden_dim" in params:
        return _as_float(params["hidden_dim"])
    match = re.search(r"hidden=([0-9]+)", label)
    return _as_float(match.group(1)) if match else float("nan")


def load_rows(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for result_path in sorted(runs_root.rglob("result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("experiment_id") != EXPERIMENT_ID:
            continue
        metrics = payload.get("simulation", {}).get("metrics", [])
        if not metrics:
            continue
        final = dict(metrics[-1])
        label = str(payload.get("method_label", payload.get("method_key", "")))
        params = dict(payload.get("method_params", {}))
        overhead = [
            _as_float(row.get("selection_overhead_seconds", row.get("selection_seconds")))
            for row in metrics
        ]
        overhead = [value for value in overhead if math.isfinite(value)]
        rows.append(
            {
                "hidden_dim": _parse_hidden(label, params),
                "seed": int(payload.get("seed", -1)),
                "final_accuracy": _as_float(final.get("accuracy")),
                "final_f1": _as_float(final.get("f1")),
                "fairness_jain": _as_float(final.get("fairness_jain")),
                "mean_selection_overhead_ms": 1000.0 * float(np.mean(overhead)) if overhead else float("nan"),
                "result_path": str(result_path),
            }
        )
    return rows


def _fmt_mean_std(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    mean *= scale
    std = (std * scale) if math.isfinite(std) else float("nan")
    if not math.isfinite(mean):
        return "--"
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def write_table(summary: "pd.DataFrame", tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Selector-architecture (meta-policy capacity) ablation for MAML-Select on Fashion-MNIST.",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Sensitivity of MAML-Select to the meta-policy MLP width on Fashion-MNIST "
        "(200 diagnostic rounds, mean$\\pm$std over seeds 42/123/2026). The selector is a "
        "$6$-$h$-$h$-$1$ MLP; $h=64$ ($4{,}673$ parameters) is the configuration used in the main paper.}",
        "\\label{tab:arch_ablation}",
        "\\small",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Width $h$ & Params & Acc. (\\%) & Jain & Overhead (ms) \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        h = int(row["hidden_dim"])
        marker = "\\textbf{%d}" % h if h == 64 else f"{h}"
        lines.append(
            f"{marker} & {_params_for_width(h):,} & "
            f"{_fmt_mean_std(row['accuracy_mean'], row['accuracy_std'], 100.0)} & "
            f"{row['jain_mean']:.2f} & "
            f"{_fmt_mean_std(row['overhead_mean'], row['overhead_std'], 1.0)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (tables_dir / "tab_arch_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.runs_root)
    if not rows:
        raise SystemExit(f"No architecture-ablation result.json files found under {args.runs_root}")

    import pandas as pd

    analysis_dir = args.output_dir / "analysis"
    tables_dir = args.output_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("hidden_dim", dropna=False, as_index=False)
        .agg(
            n=("seed", "nunique"),
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            jain_mean=("fairness_jain", "mean"),
            overhead_mean=("mean_selection_overhead_ms", "mean"),
            overhead_std=("mean_selection_overhead_ms", "std"),
        )
        .sort_values("hidden_dim")
    )
    summary["params"] = summary["hidden_dim"].map(lambda h: _params_for_width(int(h)))

    frame.to_csv(analysis_dir / "arch_ablation_raw.csv", index=False)
    summary.to_csv(analysis_dir / "arch_ablation_summary.csv", index=False)
    write_table(summary, tables_dir)

    print(f"Loaded {len(frame)} completed run(s) across {summary.shape[0]} widths.")
    print(summary.to_string(index=False))
    print(f"Analysis written to {analysis_dir}")
    print(f"Table written to {tables_dir / 'tab_arch_ablation.tex'}")


if __name__ == "__main__":
    main()
