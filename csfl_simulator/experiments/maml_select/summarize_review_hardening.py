"""Summarize the targeted review-hardening experiments.

This script consumes the result.json files produced by the review_hardening
profile and, when available, the existing Fashion-MNIST diagnostic runs from
the main MAML-Select campaign. It writes compact CSV/LaTeX tables plus
publication-style PNG/EPS diagnostic plots. It does not run any training.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RUNS = REPO_ROOT / "runs" / "maml_select_review_hardening"
DEFAULT_FASHION_RUNS = REPO_ROOT / "runs" / "maml_select"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "maml_select" / "review_hardening"

EXPERIMENTS = {
    "lambda_sensitivity",
    "feature_ablation",
    "cifar10_lambda_sensitivity",
    "inner_step_ablation",
    # lambda=0 no-penalty anchors (resubmission); fold into the lambda_sensitivity family.
    "cifar10_lambda_anchor",
    "fashion_lambda_anchor",
}
DATASET_LABELS = {
    "fashion_main": "Fashion-MNIST",
    "cifar10_main": "CIFAR-10",
    "cifar10_review_100": "CIFAR-10",
}


def _family(experiment_id: str) -> str:
    if experiment_id in {
        "lambda_sensitivity",
        "cifar10_lambda_sensitivity",
        "cifar10_lambda_anchor",
        "fashion_lambda_anchor",
    }:
        return "lambda_sensitivity"
    if experiment_id == "feature_ablation":
        return "feature_ablation"
    if experiment_id == "inner_step_ablation":
        return "inner_step_ablation"
    return experiment_id


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_lambda(label: str, params: Dict[str, Any]) -> float:
    if "lambda_latency" in params:
        return _as_float(params["lambda_latency"])
    match = re.search(r"lambda=([0-9.]+)", label)
    return _as_float(match.group(1)) if match else float("nan")


def _parse_inner_steps(label: str, params: Dict[str, Any]) -> float:
    if "inner_steps" in params:
        return _as_float(params["inner_steps"])
    match = re.search(r"inner steps=([0-9]+)", label)
    return _as_float(match.group(1)) if match else float("nan")


def load_rows(runs_root: Path, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for result_path in sorted(runs_root.rglob("result.json")):
        payload = _load_json(result_path)
        experiment_id = payload.get("experiment_id")
        if experiment_id not in EXPERIMENTS:
            continue
        metrics = payload.get("simulation", {}).get("metrics", [])
        if not metrics:
            continue
        final = dict(metrics[-1])
        label = str(payload.get("method_label", payload.get("method_key", "")))
        params = dict(payload.get("method_params", {}))
        selection_values = [
            _as_float(row.get("selection_overhead_seconds", row.get("selection_seconds")))
            for row in metrics
        ]
        selection_values = [value for value in selection_values if math.isfinite(value)]
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": _family(str(experiment_id)),
                "scenario_name": payload.get("scenario_name", ""),
                "dataset": DATASET_LABELS.get(payload.get("scenario_name", ""), payload.get("scenario_name", "")),
                "source": source,
                "method_key": payload.get("method_key", ""),
                "variant": label,
                "lambda": _parse_lambda(label, params),
                "inner_steps": _parse_inner_steps(label, params),
                "seed": int(payload.get("seed", -1)),
                "final_accuracy": _as_float(final.get("accuracy")),
                "final_precision": _as_float(final.get("precision")),
                "final_recall": _as_float(final.get("recall")),
                "final_f1": _as_float(final.get("f1")),
                "cum_training_tflops": _as_float(final.get("cum_training_tflops")),
                "cum_modelled_energy_wh": _as_float(final.get("cum_modelled_energy_wh")),
                "cum_modelled_carbon_g": _as_float(final.get("cum_modelled_carbon_g")),
                "fairness_jain": _as_float(final.get("fairness_jain")),
                "coverage_pct": 100.0 * _as_float(final.get("participation_coverage_ratio")),
                "mean_selection_overhead_ms": 1000.0 * float(np.mean(selection_values)) if selection_values else float("nan"),
                "result_path": str(result_path),
            }
        )
    return rows


def summarize(frame: "pd.DataFrame") -> "pd.DataFrame":
    group_cols = ["family", "experiment_id", "scenario_name", "dataset", "variant", "lambda", "inner_steps"]
    return (
        frame.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n=("seed", "nunique"),
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            f1_mean=("final_f1", "mean"),
            tflops_mean=("cum_training_tflops", "mean"),
            tflops_std=("cum_training_tflops", "std"),
            energy_mean=("cum_modelled_energy_wh", "mean"),
            carbon_mean=("cum_modelled_carbon_g", "mean"),
            jain_mean=("fairness_jain", "mean"),
            coverage_mean=("coverage_pct", "mean"),
            selection_overhead_ms_mean=("mean_selection_overhead_ms", "mean"),
            selection_overhead_ms_std=("mean_selection_overhead_ms", "std"),
        )
        .sort_values(["family", "scenario_name", "lambda", "inner_steps", "variant"])
    )


def _fmt_mean_std(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    mean *= scale
    std *= scale
    if not math.isfinite(mean):
        return "--"
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def write_tex_tables(summary: "pd.DataFrame", tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    lambda_rows = summary[summary["family"] == "lambda_sensitivity"].copy()
    lambda_rows = lambda_rows.sort_values(["scenario_name", "lambda"])
    lines = [
        "% Lambda sensitivity for MAML-Select on Fashion-MNIST and CIFAR-10.",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Sensitivity of MAML-Select to the cost trade-off $\\lambda$. Fashion-MNIST uses the existing diagnostic runs, while CIFAR-10 uses 100 diagnostic rounds.}",
        "\\label{tab:lambda_sensitivity_two_dataset}",
        "\\small",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Dataset & $\\lambda$ & Acc. (\\%) & TFLOPs & Energy (Wh) & Jain \\\\",
        "\\midrule",
    ]
    previous_dataset = None
    for _, row in lambda_rows.iterrows():
        dataset = str(row["dataset"])
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append("\\addlinespace[1pt]")
        previous_dataset = dataset
        lines.append(
            f"{dataset} & {row['lambda']:.1f} & "
            f"{_fmt_mean_std(row['accuracy_mean'], row['accuracy_std'], 100.0)} & "
            f"{row['tflops_mean']:.0f} & {row['energy_mean']:.0f} & {row['jain_mean']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (tables_dir / "tab_lambda_sensitivity_two_dataset.tex").write_text("\n".join(lines), encoding="utf-8")

    feature_rows = summary[summary["family"] == "feature_ablation"].copy()
    if not feature_rows.empty:
        order = [
            "all features", "without loss", "without gradient norm", "without latency",
            "without battery", "without frequency", "without staleness",
        ]
        import pandas as pd
        feature_rows["variant"] = pd.Categorical(feature_rows["variant"], order, ordered=True)
        feature_rows = feature_rows.sort_values(["scenario_name", "variant"])
        lines = [
            "% Fashion-MNIST state-feature ablation for MAML-Select.",
            "\\begin{table}[!t]",
            "\\centering",
            "\\caption{State-feature ablation of MAML-Select on Fashion-MNIST.}",
            "\\label{tab:fashion_feature_ablation}",
            "\\small",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Variant & Acc. (\\%) & TFLOPs & Energy (Wh) & Jain \\\\",
            "\\midrule",
        ]
        for _, row in feature_rows.iterrows():
            variant = str(row["variant"])
            label = "Full state vector" if variant == "all features" else variant.replace("without ", "w/o ")
            if variant == "all features":
                label = f"\\textbf{{{label}}}"
            lines.append(
                f"{label} & "
                f"{_fmt_mean_std(row['accuracy_mean'], row['accuracy_std'], 100.0)} & "
                f"{row['tflops_mean']:.0f} & {row['energy_mean']:.0f} & {row['jain_mean']:.2f} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        (tables_dir / "tab_fashion_feature_ablation.tex").write_text("\n".join(lines), encoding="utf-8")

    inner_rows = summary[summary["family"] == "inner_step_ablation"].copy()
    inner_rows = inner_rows.sort_values(["scenario_name", "inner_steps"])
    if not inner_rows.empty:
        lines = [
            "% Inner-step ablation for MAML-Select.",
            "\\begin{table}[!t]",
            "\\centering",
            "\\caption{CIFAR-10 inner-loop step ablation for MAML-Select over 100 diagnostic rounds.}",
            "\\label{tab:inner_step_ablation}",
            "\\small",
            "\\begin{tabular}{@{}llrrrr@{}}",
            "\\toprule",
            "Dataset & Steps & Acc. (\\%) & TFLOPs & Jain & Overhead (ms) \\\\",
            "\\midrule",
        ]
        for _, row in inner_rows.iterrows():
            lines.append(
                f"{row['dataset']} & {int(row['inner_steps'])} & "
                f"{_fmt_mean_std(row['accuracy_mean'], row['accuracy_std'], 100.0)} & "
                f"{row['tflops_mean']:.0f} & {row['jain_mean']:.2f} & "
                f"{_fmt_mean_std(row['selection_overhead_ms_mean'], row['selection_overhead_ms_std'], 1.0)} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        (tables_dir / "tab_inner_step_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def save_plots(summary: "pd.DataFrame", plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#111111",
    })

    lambda_rows = summary[summary["family"] == "lambda_sensitivity"].copy()
    lambda_rows = lambda_rows.sort_values(["scenario_name", "lambda"])
    if not lambda_rows.empty:
        scenarios = list(lambda_rows["scenario_name"].drop_duplicates())
        fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 3.2), squeeze=False)
        for ax1, scenario in zip(axes[0], scenarios):
            part = lambda_rows[lambda_rows["scenario_name"] == scenario].sort_values("lambda")
            x = part["lambda"].to_numpy(dtype=float)
            ax1.errorbar(
                x,
                100.0 * part["accuracy_mean"].to_numpy(dtype=float),
                yerr=100.0 * part["accuracy_std"].fillna(0.0).to_numpy(dtype=float),
                marker="o",
                color="#111111",
                linewidth=1.8,
                capsize=3,
                label="Accuracy",
            )
            ax1.set_xscale("log")
            ax1.set_xlabel(r"Cost trade-off $\lambda$")
            ax1.set_ylabel("Final accuracy (%)")
            ax1.set_title(DATASET_LABELS.get(str(scenario), str(scenario)))
            ax1.grid(True, alpha=0.25)
            ax2 = ax1.twinx()
            base_tflops = float(part["tflops_mean"].iloc[0])
            base_energy = float(part["energy_mean"].iloc[0])
            base_jain = float(part["jain_mean"].iloc[0])
            ax2.plot(x, part["tflops_mean"] / base_tflops, marker="s", color="#E69F00", label="Compute")
            ax2.plot(x, part["energy_mean"] / base_energy, marker="^", color="#009E73", label="Energy")
            ax2.plot(x, part["jain_mean"] / base_jain, marker="D", color="#0072B2", label="Fairness")
            ax2.set_ylabel(r"Relative to $\lambda=0.1$")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        for ext in ("png", "eps"):
            fig.savefig(plots_dir / f"lambda_sensitivity_two_dataset.{ext}", dpi=400, bbox_inches="tight")
        plt.close(fig)

    inner_rows = summary[summary["family"] == "inner_step_ablation"].copy()
    if not inner_rows.empty:
        part = inner_rows.sort_values("inner_steps")
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        x = np.arange(len(part))
        ax.bar(
            x,
            100.0 * part["accuracy_mean"].to_numpy(dtype=float),
            yerr=100.0 * part["accuracy_std"].fillna(0.0).to_numpy(dtype=float),
            color="#4E79A7",
            edgecolor="#111111",
            capsize=3,
            width=0.62,
            label="Accuracy",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in part["inner_steps"]])
        ax.set_xlabel("Inner-loop steps")
        ax.set_ylabel("Final accuracy (%)")
        ax.set_title("CIFAR-10 Inner-Step Ablation")
        ax.grid(axis="y", alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(
            x,
            part["selection_overhead_ms_mean"].to_numpy(dtype=float),
            color="#CC3311",
            marker="o",
            linewidth=1.6,
            label="Overhead",
        )
        ax2.set_ylabel("Selection overhead (ms)")
        fig.tight_layout()
        for ext in ("png", "eps"):
            fig.savefig(plots_dir / f"inner_step_ablation.{ext}", dpi=400, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--fashion-runs-root",
        type=Path,
        default=DEFAULT_FASHION_RUNS,
        help="Optional root containing existing Fashion-MNIST lambda/feature-ablation runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extra-runs-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Additional run roots to fold in (e.g. the lambda=0 anchor runs). May be repeated.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(load_rows(args.fashion_runs_root, source="fashion_existing"))
    rows.extend(load_rows(args.runs_root, source="review_hardening"))
    for extra in args.extra_runs_root:
        rows.extend(load_rows(extra, source="lambda_anchor"))
    if not rows:
        raise SystemExit(
            "No diagnostic result.json files found under "
            f"{args.runs_root} or {args.fashion_runs_root}"
        )

    import pandas as pd

    out = args.output_dir
    analysis_dir = out / "analysis"
    plots_dir = out / "plots"
    tables_dir = out / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(analysis_dir / "review_hardening_raw.csv", index=False)
    summary.to_csv(analysis_dir / "review_hardening_summary.csv", index=False)
    summary[summary["family"] == "lambda_sensitivity"].to_csv(
        analysis_dir / "lambda_sensitivity_two_dataset_summary.csv", index=False
    )
    summary[summary["family"] == "feature_ablation"].to_csv(
        analysis_dir / "fashion_feature_ablation_summary.csv", index=False
    )
    summary[summary["family"] == "inner_step_ablation"].to_csv(
        analysis_dir / "inner_step_ablation_summary.csv", index=False
    )
    write_tex_tables(summary, tables_dir)
    if not args.no_plots:
        save_plots(summary, plots_dir)

    print(f"Loaded {len(frame)} completed run(s).")
    print(f"Analysis written to {analysis_dir}")
    print(f"Tables written to {tables_dir}")
    if not args.no_plots:
        print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    main()
