"""Build reviewer-facing MAML-Select plots and tables.

This script merges the Mac CIFAR-100 results with the Windows Fashion-MNIST
and CIFAR-10 runs, then writes publication-style figures in both EPS and PNG.
It intentionally uses the available TiFL/FedCor seeds as-is; no missing seeds
are imputed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover - optional dependency guard
    ttest_rel = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_WINDOWS_RESULTS = Path("/Users/advaitpathak/Desktop/maml_select_runs_windows")
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "maml_select" / "review_pack"

METHOD_SHORT_TO_KEY = {
    "fedavg": "baseline.fedavg",
    "fedcs": "system_aware.fedcs",
    "oort": "system_aware.oort",
    "tifl": "system_aware.tifl",
    "fedcor": "ml.fedcor",
    "criticalfl": "research.criticalfl",
    "fedgcs": "research.fedgcs",
    "maml_select": "research.maml_select",
}

METHOD_LABELS = {
    "baseline.fedavg": "FedAvg",
    "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort",
    "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor",
    "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS",
    "research.maml_select": "MAML-Select",
}

METHOD_ORDER = [
    "baseline.fedavg",
    "system_aware.fedcs",
    "system_aware.oort",
    "system_aware.tifl",
    "ml.fedcor",
    "research.criticalfl",
    "research.fedgcs",
    "research.maml_select",
]

DATASET_LABELS = {
    "fashion_main": "Fashion-MNIST",
    "cifar10_main": "CIFAR-10",
    "cifar100_main": "CIFAR-100",
}

DATASET_ORDER = ["fashion_main", "cifar10_main", "cifar100_main"]

COLORS = {
    "baseline.fedavg": "#4E79A7",
    "system_aware.fedcs": "#F28E2B",
    "system_aware.oort": "#59A14F",
    "system_aware.tifl": "#E15759",
    "ml.fedcor": "#B07AA1",
    "research.criticalfl": "#9C755F",
    "research.fedgcs": "#76B7B2",
    "research.maml_select": "#111111",
}

HATCHES = {
    "baseline.fedavg": "",
    "system_aware.fedcs": "//",
    "system_aware.oort": "\\\\",
    "system_aware.tifl": "..",
    "ml.fedcor": "xx",
    "research.criticalfl": "--",
    "research.fedgcs": "++",
    "research.maml_select": "",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _parse_run_dir(run_dir: Path) -> Dict[str, Any] | None:
    name = run_dir.name
    match = re.match(
        r"^(?P<experiment>main_benchmarks|cifar100_benchmarks)_"
        r"(?P<scenario>fashion_main|cifar10_main|cifar100_main)_"
        r"(?P<method>.+)_s(?P<seed>\d+)$",
        name,
    )
    if not match:
        return None
    method_short = match.group("method")
    return {
        "experiment_id": match.group("experiment"),
        "scenario_name": match.group("scenario"),
        "method_key": METHOD_SHORT_TO_KEY.get(method_short, method_short),
        "method_short": method_short,
        "seed": int(match.group("seed")),
    }


def _final_from_metrics(metrics: Sequence[Dict[str, Any]], max_round: int | None) -> Dict[str, Any]:
    filtered = []
    for row in metrics:
        round_idx = int(row.get("round", -1))
        if round_idx < 0:
            continue
        if max_round is not None and round_idx > max_round:
            continue
        filtered.append(row)
    if not filtered:
        return dict(metrics[-1]) if metrics else {}
    evaluated = [row for row in filtered if bool(row.get("evaluated", False))]
    return dict(evaluated[-1] if evaluated else filtered[-1])


def _load_main_run(run_dir: Path, source_label: str) -> Dict[str, Any] | None:
    parsed = _parse_run_dir(run_dir)
    if parsed is None:
        return None
    scenario = parsed["scenario_name"]
    max_round = 150 if scenario == "cifar100_main" else 199
    result_path = run_dir / "result.json"
    jsonl_path = run_dir / "round_metrics.jsonl"
    payload: Dict[str, Any] | None = None
    metrics: List[Dict[str, Any]] = []
    has_result_json = result_path.exists()
    if has_result_json:
        payload = _load_json(result_path)
        parsed.update(
            {
                "experiment_id": payload.get("experiment_id", parsed["experiment_id"]),
                "scenario_name": payload.get("scenario_name", scenario),
                "method_key": payload.get("method_key", parsed["method_key"]),
                "seed": int(payload.get("seed", parsed["seed"])),
            }
        )
        metrics = list(payload.get("simulation", {}).get("metrics", []))
    elif jsonl_path.exists():
        metrics = _read_jsonl(jsonl_path)
    else:
        return None
    if not metrics:
        return None
    observed_rounds = [int(row.get("round", -1)) for row in metrics if int(row.get("round", -1)) >= 0]
    if not observed_rounds or max(observed_rounds) < max_round:
        return None
    final = _final_from_metrics(metrics, max_round)
    return {
        **parsed,
        "dataset": DATASET_LABELS.get(parsed["scenario_name"], parsed["scenario_name"]),
        "method_label": METHOD_LABELS.get(parsed["method_key"], parsed["method_key"]),
        "source": source_label,
        "has_result_json": bool(has_result_json),
        "run_dir": str(run_dir),
        "round": int(final.get("round", -1)),
        "final_accuracy": _as_float(final.get("accuracy")),
        "final_f1": _as_float(final.get("f1")),
        "loss": _as_float(final.get("loss")),
        "cum_time": _as_float(final.get("cum_time")),
        "cum_training_tflops": _as_float(final.get("cum_training_tflops")),
        "cum_modelled_energy_wh": _as_float(final.get("cum_modelled_energy_wh")),
        "cum_modelled_carbon_g": _as_float(final.get("cum_modelled_carbon_g")),
        "cum_comm_mb": _as_float(final.get("cum_comm_mb")),
        "fairness_gini": _as_float(final.get("fairness_gini")),
        "fairness_jain": _as_float(final.get("fairness_jain")),
        "utilization_entropy": _as_float(final.get("utilization_entropy")),
        "participation_coverage_ratio": _as_float(final.get("participation_coverage_ratio")),
        "tier_0_selection_rate": _as_float(final.get("tier_0_selection_rate")),
        "tier_1_selection_rate": _as_float(final.get("tier_1_selection_rate")),
        "tier_2_selection_rate": _as_float(final.get("tier_2_selection_rate")),
        "mean_cohort_size": _as_float(final.get("mean_cohort_size")),
        "metrics": metrics,
    }


def load_main_results(roots: Sequence[tuple[Path, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for root, source in roots:
        if not root.exists():
            continue
        candidates = {path.parent for path in root.rglob("result.json")}
        candidates.update({path.parent for path in root.rglob("round_metrics.jsonl")})
        for run_dir in sorted(candidates):
            if run_dir in seen_dirs:
                continue
            seen_dirs.add(run_dir)
            loaded = _load_main_run(run_dir, source)
            if loaded is None:
                continue
            metrics = loaded.pop("metrics")
            runs.append(loaded)
            max_round = 150 if loaded["scenario_name"] == "cifar100_main" else 199
            for row in metrics:
                round_idx = int(row.get("round", -1))
                if round_idx < 0 or round_idx > max_round:
                    continue
                round_rows.append(
                    {
                        "scenario_name": loaded["scenario_name"],
                        "dataset": loaded["dataset"],
                        "method_key": loaded["method_key"],
                        "method_label": loaded["method_label"],
                        "seed": loaded["seed"],
                        "source": loaded["source"],
                        "round": round_idx,
                        "accuracy": _as_float(row.get("accuracy")),
                        "f1": _as_float(row.get("f1")),
                        "cum_training_tflops": _as_float(row.get("cum_training_tflops")),
                        "cum_modelled_energy_wh": _as_float(row.get("cum_modelled_energy_wh")),
                        "cum_modelled_carbon_g": _as_float(row.get("cum_modelled_carbon_g")),
                        "cum_comm_mb": _as_float(row.get("cum_comm_mb")),
                        "fairness_jain": _as_float(row.get("fairness_jain")),
                        "fairness_gini": _as_float(row.get("fairness_gini")),
                        "participation_coverage_ratio": _as_float(row.get("participation_coverage_ratio")),
                    }
                )
    run_frame = pd.DataFrame(runs)
    round_frame = pd.DataFrame(round_rows)
    if not run_frame.empty:
        run_frame["method_key"] = pd.Categorical(run_frame["method_key"], METHOD_ORDER, ordered=True)
        run_frame["scenario_name"] = pd.Categorical(run_frame["scenario_name"], DATASET_ORDER, ordered=True)
        run_frame = run_frame.sort_values(["scenario_name", "method_key", "seed"]).reset_index(drop=True)
    return run_frame, round_frame


def _mean_std(values: pd.Series, scale: float = 1.0, suffix: str = "") -> str:
    values = values.astype(float).dropna() * scale
    if len(values) == 0:
        return "--"
    if len(values) == 1:
        return f"{values.iloc[0]:.2f}{suffix}"
    return f"{values.mean():.2f}±{values.std(ddof=1):.2f}{suffix}"


def write_tables(
    runs: pd.DataFrame,
    lambda_frame: pd.DataFrame,
    ablation_frame: pd.DataFrame,
    scaling_frame: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables_dir / "review_all_run_finals.csv", index=False)

    metrics = {
        "final_accuracy": 100.0,
        "final_f1": 100.0,
        "cum_training_tflops": 1.0,
        "cum_modelled_energy_wh": 1.0,
        "cum_modelled_carbon_g": 1.0,
        "cum_comm_mb": 1.0,
        "fairness_jain": 1.0,
        "participation_coverage_ratio": 100.0,
    }
    summary_rows = []
    for (scenario, method), group in runs.groupby(["scenario_name", "method_key"], observed=True):
        row = {
            "dataset": DATASET_LABELS.get(str(scenario), str(scenario)),
            "method": METHOD_LABELS.get(str(method), str(method)),
            "method_key": str(method),
            "available_seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
            "n": int(group["seed"].nunique()),
        }
        for metric, scale in metrics.items():
            row[metric] = _mean_std(group[metric], scale=scale)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "review_main_benchmark_summary.csv", index=False)
    summary.to_latex(tables_dir / "review_main_benchmark_summary.tex", index=False, escape=False)

    relative_rows = []
    for scenario, group in runs.groupby("scenario_name", observed=True):
        fedavg = group[group["method_key"].astype(str) == "baseline.fedavg"]
        ours = group[group["method_key"].astype(str) == "research.maml_select"]
        if fedavg.empty or ours.empty:
            continue
        fedavg_means = fedavg.groupby("seed").first()
        ours_means = ours.groupby("seed").first()
        shared = sorted(set(fedavg_means.index) & set(ours_means.index))
        if not shared:
            continue
        for seed in shared:
            f = fedavg_means.loc[seed]
            o = ours_means.loc[seed]
            relative_rows.append(
                {
                    "dataset": DATASET_LABELS.get(str(scenario), str(scenario)),
                    "seed": int(seed),
                    "accuracy_pp_vs_fedavg": 100.0 * (o["final_accuracy"] - f["final_accuracy"]),
                    "tflops_reduction_pct_vs_fedavg": 100.0
                    * (f["cum_training_tflops"] - o["cum_training_tflops"])
                    / max(float(f["cum_training_tflops"]), 1e-12),
                    "energy_reduction_pct_vs_fedavg": 100.0
                    * (f["cum_modelled_energy_wh"] - o["cum_modelled_energy_wh"])
                    / max(float(f["cum_modelled_energy_wh"]), 1e-12),
                    "carbon_reduction_pct_vs_fedavg": 100.0
                    * (f["cum_modelled_carbon_g"] - o["cum_modelled_carbon_g"])
                    / max(float(f["cum_modelled_carbon_g"]), 1e-12),
                    "comm_reduction_pct_vs_fedavg": 100.0
                    * (f["cum_comm_mb"] - o["cum_comm_mb"])
                    / max(float(f["cum_comm_mb"]), 1e-12),
                }
            )
    relative = pd.DataFrame(relative_rows)
    if not relative.empty:
        relative.to_csv(tables_dir / "review_maml_vs_fedavg_by_seed.csv", index=False)
        relative_summary = relative.groupby("dataset").agg(["mean", "std", "count"])
        relative_summary.to_csv(tables_dir / "review_maml_vs_fedavg_summary.csv")
        relative_summary.to_latex(tables_dir / "review_maml_vs_fedavg_summary.tex", float_format="%.3f")

    tests = paired_tests(runs)
    tests.to_csv(tables_dir / "review_paired_tests.csv", index=False)

    lambda_frame.to_csv(tables_dir / "review_lambda_sensitivity_summary.csv", index=False)
    lambda_frame.to_latex(tables_dir / "review_lambda_sensitivity_summary.tex", index=False, float_format="%.4f")

    ablation_frame.to_csv(tables_dir / "review_feature_ablation_summary.csv", index=False)
    ablation_frame.to_latex(tables_dir / "review_feature_ablation_summary.tex", index=False, float_format="%.4f")

    scaling_frame.to_csv(tables_dir / "review_scaling_overhead.csv", index=False)
    scaling_frame.to_latex(tables_dir / "review_scaling_overhead.tex", index=False, float_format="%.6f")


def paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if runs.empty or ttest_rel is None:
        return pd.DataFrame(rows)
    metrics = [
        "final_accuracy",
        "cum_training_tflops",
        "cum_modelled_energy_wh",
        "cum_modelled_carbon_g",
        "cum_comm_mb",
        "fairness_jain",
    ]
    for scenario, group in runs.groupby("scenario_name", observed=True):
        refs = ["baseline.fedavg", "research.maml_select"]
        for ref in refs:
            ref_group = group[group["method_key"].astype(str) == ref].set_index("seed")
            if ref_group.empty:
                continue
            for method, method_group in group.groupby("method_key", observed=True):
                method = str(method)
                if method == ref:
                    continue
                other = method_group.set_index("seed")
                shared = sorted(set(ref_group.index) & set(other.index))
                if len(shared) < 2:
                    continue
                for metric in metrics:
                    left = ref_group.loc[shared, metric].astype(float)
                    right = other.loc[shared, metric].astype(float)
                    valid = np.isfinite(left) & np.isfinite(right)
                    left = left[valid]
                    right = right[valid]
                    if len(left) < 2:
                        continue
                    stat, pval = ttest_rel(left, right)
                    rows.append(
                        {
                            "dataset": DATASET_LABELS.get(str(scenario), str(scenario)),
                            "reference": METHOD_LABELS.get(ref, ref),
                            "comparison": METHOD_LABELS.get(method, method),
                            "metric": metric,
                            "paired_seeds": ",".join(str(int(seed)) for seed in shared),
                            "n": len(left),
                            "reference_minus_comparison": float((left - right).mean()),
                            "paired_t": float(stat),
                            "p_value": float(pval),
                        }
                    )
    return pd.DataFrame(rows)


def load_variant_summary(runs_root: Path, experiment_prefix: str) -> pd.DataFrame:
    rows = []
    for result_path in sorted(runs_root.glob(f"{experiment_prefix}_*/result.json")):
        payload = _load_json(result_path)
        metrics = payload.get("simulation", {}).get("metrics", [])
        if not metrics:
            continue
        final = dict(metrics[-1])
        label = payload.get("method_label", payload.get("method_key", result_path.parent.name))
        rows.append(
            {
                "variant": label,
                "method_key": payload.get("method_key", ""),
                "seed": int(payload.get("seed", -1)),
                "final_accuracy": _as_float(final.get("accuracy")),
                "cum_training_tflops": _as_float(final.get("cum_training_tflops")),
                "cum_modelled_energy_wh": _as_float(final.get("cum_modelled_energy_wh")),
                "cum_modelled_carbon_g": _as_float(final.get("cum_modelled_carbon_g")),
                "cum_comm_mb": _as_float(final.get("cum_comm_mb")),
                "fairness_jain": _as_float(final.get("fairness_jain")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["variant", "method_key"], as_index=False)
        .agg(
            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_std=("final_accuracy", "std"),
            cum_training_tflops_mean=("cum_training_tflops", "mean"),
            cum_training_tflops_std=("cum_training_tflops", "std"),
            cum_modelled_energy_wh_mean=("cum_modelled_energy_wh", "mean"),
            cum_modelled_carbon_g_mean=("cum_modelled_carbon_g", "mean"),
            cum_comm_mb_mean=("cum_comm_mb", "mean"),
            fairness_jain_mean=("fairness_jain", "mean"),
            n=("seed", "nunique"),
        )
        .sort_values(["variant"])
    )


def save_figure(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {"format": "pdf"}),
        ("eps", {"format": "eps"}),
    ):
        fig.savefig(plots_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def method_handles(methods: Sequence[str]) -> list[Any]:
    handles = []
    for method in methods:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=COLORS.get(method, "#333333"),
                marker="o",
                linestyle="-",
                label=METHOD_LABELS.get(method, method),
                linewidth=2.0,
            )
        )
    return handles


def plot_convergence(rounds: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.35), sharey=False)
    for ax, scenario in zip(axes, DATASET_ORDER):
        subset = rounds[rounds["scenario_name"] == scenario].copy()
        if subset.empty:
            ax.set_axis_off()
            continue
        for method in METHOD_ORDER:
            group = subset[subset["method_key"].astype(str) == method]
            if group.empty:
                continue
            summary = (
                group.groupby("round", as_index=False)
                .agg(accuracy=("accuracy", "mean"), std=("accuracy", "std"), n=("seed", "nunique"))
                .sort_values("round")
            )
            ax.plot(
                summary["round"],
                100.0 * summary["accuracy"],
                color=COLORS.get(method, "#333333"),
                label=METHOD_LABELS.get(method, method),
                linewidth=2.4 if method == "research.maml_select" else 1.5,
                zorder=4 if method == "research.maml_select" else 2,
            )
        ax.set_title(DATASET_LABELS[scenario])
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Accuracy (%)")
        ax.margins(x=0.01)
    fig.legend(
        handles=method_handles(METHOD_ORDER),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("Convergence Under Non-IID Client Selection", y=1.18, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, plots_dir, "review_fig_convergence_all_datasets")


def plot_efficiency_grid(runs: pd.DataFrame, plots_dir: Path) -> None:
    agg = (
        runs.groupby(["scenario_name", "method_key"], observed=True)
        .agg(
            accuracy=("final_accuracy", "mean"),
            tflops=("cum_training_tflops", "mean"),
            energy=("cum_modelled_energy_wh", "mean"),
            carbon=("cum_modelled_carbon_g", "mean"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )
    metrics = [
        ("accuracy", "Accuracy (%)", 100.0),
        ("tflops", "TFLOPs", 1.0),
        ("energy", "Energy (Wh)", 1.0),
        ("carbon", "Carbon (gCO$_2$e)", 1.0),
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), len(metrics), figsize=(13.2, 7.7), sharex=False)
    for row_idx, scenario in enumerate(DATASET_ORDER):
        data = agg[agg["scenario_name"] == scenario]
        for col_idx, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values = []
            labels = []
            colors = []
            hatches = []
            for method in METHOD_ORDER:
                row = data[data["method_key"].astype(str) == method]
                if row.empty:
                    continue
                values.append(float(row[metric].iloc[0]) * scale)
                labels.append(METHOD_LABELS[method])
                colors.append(COLORS[method])
                hatches.append(HATCHES.get(method, ""))
            xpos = np.arange(len(values))
            bars = ax.bar(xpos, values, color=colors, edgecolor="#222222", linewidth=0.55)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[scenario]}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.set_xticks(xpos)
            ax.set_xticklabels([label.replace("MAML-Select", "MAML") for label in labels], rotation=55, ha="right")
            if metric != "accuracy":
                ax.set_yscale("log")
            if row_idx == 0:
                ax.set_title(ylabel)
    fig.suptitle("Accuracy and Resource Cost Summary", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_efficiency_resource_grid")


def plot_accuracy_energy_scatter(runs: pd.DataFrame, plots_dir: Path) -> None:
    agg = (
        runs.groupby(["scenario_name", "method_key"], observed=True)
        .agg(
            accuracy=("final_accuracy", "mean"),
            energy=("cum_modelled_energy_wh", "mean"),
            carbon=("cum_modelled_carbon_g", "mean"),
            tflops=("cum_training_tflops", "mean"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    for ax, scenario in zip(axes, DATASET_ORDER):
        data = agg[agg["scenario_name"] == scenario]
        for method in METHOD_ORDER:
            row = data[data["method_key"].astype(str) == method]
            if row.empty:
                continue
            x = float(row["energy"].iloc[0])
            y = 100.0 * float(row["accuracy"].iloc[0])
            size = max(30.0, min(480.0, float(row["tflops"].iloc[0]) * 0.085))
            ax.scatter(
                x,
                y,
                s=size,
                color=COLORS[method],
                edgecolor="#222222",
                linewidth=0.55,
                zorder=4 if method == "research.maml_select" else 3,
            )
            label = "MAML" if method == "research.maml_select" else METHOD_LABELS[method]
            ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        ax.set_title(DATASET_LABELS[scenario])
        ax.set_xlabel("Modelled energy (Wh)")
        ax.set_ylabel("Final accuracy (%)")
        ax.set_xscale("log")
    fig.suptitle("Accuracy-Energy Trade-off (Bubble Area Proportional to TFLOPs)", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_accuracy_energy_tradeoff")


def plot_lambda(lambda_frame: pd.DataFrame, plots_dir: Path) -> None:
    if lambda_frame.empty:
        return
    frame = lambda_frame.copy()
    frame["lambda"] = frame["variant"].str.extract(r"lambda=([0-9.]+)").astype(float)
    frame = frame.sort_values("lambda")
    fig, ax1 = plt.subplots(figsize=(4.7, 3.35))
    ax2 = ax1.twinx()
    acc_handle = ax1.errorbar(
        frame["lambda"],
        100.0 * frame["final_accuracy_mean"],
        yerr=100.0 * frame["final_accuracy_std"].fillna(0.0),
        color="#111111",
        marker="o",
        capsize=3,
        label="Accuracy",
    )
    tflops_handle = ax2.plot(
        frame["lambda"],
        frame["cum_training_tflops_mean"],
        color="#D55E00",
        marker="s",
        linestyle="--",
        label="TFLOPs",
    )[0]
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Cost trade-off $\lambda$")
    ax1.set_ylabel("Final accuracy (%)", color="#111111")
    ax2.set_ylabel("Total TFLOPs", color="#D55E00")
    ax1.set_title("Lambda Sensitivity on Fashion-MNIST")
    ax1.legend(
        [acc_handle.lines[0], tflops_handle],
        ["Accuracy", "TFLOPs"],
        frameon=False,
        loc="best",
    )
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_lambda_sensitivity")


def plot_ablation(ablation_frame: pd.DataFrame, plots_dir: Path) -> None:
    if ablation_frame.empty:
        return
    frame = ablation_frame.copy()
    order = [
        "all features",
        "without loss",
        "without gradient norm",
        "without latency",
        "without battery",
        "without frequency",
        "without staleness",
    ]
    frame["variant"] = pd.Categorical(frame["variant"], order, ordered=True)
    frame = frame.sort_values("variant")
    base = frame[frame["variant"].astype(str) == "all features"]["final_accuracy_mean"]
    base_value = float(base.iloc[0]) if not base.empty else float(frame["final_accuracy_mean"].max())
    colors = ["#111111" if str(v) == "all features" else "#4E79A7" for v in frame["variant"]]
    fig, ax = plt.subplots(figsize=(7.3, 3.35))
    x = np.arange(len(frame))
    ax.bar(
        x,
        100.0 * frame["final_accuracy_mean"],
        yerr=100.0 * frame["final_accuracy_std"].fillna(0.0),
        capsize=3,
        color=colors,
        edgecolor="#222222",
        linewidth=0.55,
    )
    ax.axhline(100.0 * base_value, color="#D55E00", linestyle="--", linewidth=1.2, label="Full state vector")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("without ", "w/o ") for v in frame["variant"]], rotation=28, ha="right")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title("State-Feature Ablation on Fashion-MNIST")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_feature_ablation")


def plot_fairness(runs: pd.DataFrame, plots_dir: Path) -> None:
    agg = (
        runs.groupby(["scenario_name", "method_key"], observed=True)
        .agg(
            tier0=("tier_0_selection_rate", "mean"),
            tier1=("tier_1_selection_rate", "mean"),
            tier2=("tier_2_selection_rate", "mean"),
            jain=("fairness_jain", "mean"),
            coverage=("participation_coverage_ratio", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), sharey=True)
    tier_colors = ["#BFD7EA", "#7FB3D5", "#21618C"]
    for ax, scenario in zip(axes, DATASET_ORDER):
        data = agg[agg["scenario_name"] == scenario]
        labels, t0, t1, t2, jain = [], [], [], [], []
        for method in METHOD_ORDER:
            row = data[data["method_key"].astype(str) == method]
            if row.empty:
                continue
            labels.append(METHOD_LABELS[method].replace("MAML-Select", "MAML"))
            t0.append(float(row["tier0"].iloc[0]))
            t1.append(float(row["tier1"].iloc[0]))
            t2.append(float(row["tier2"].iloc[0]))
            jain.append(float(row["jain"].iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x, t0, color=tier_colors[0], edgecolor="#222222", linewidth=0.35, label="Tier 1")
        ax.bar(x, t1, bottom=t0, color=tier_colors[1], edgecolor="#222222", linewidth=0.35, label="Tier 2")
        ax.bar(
            x,
            t2,
            bottom=np.array(t0) + np.array(t1),
            color=tier_colors[2],
            edgecolor="#222222",
            linewidth=0.35,
            label="Tier 3",
        )
        ax2 = ax.twinx()
        ax2.plot(x, jain, color="#D55E00", marker="o", linewidth=1.4, label="Jain")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
        if ax is axes[-1]:
            ax2.set_ylabel("Jain fairness")
        else:
            ax2.set_yticklabels([])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(DATASET_LABELS[scenario])
        ax.set_ylabel("Tier selection share")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("Hardware-Tier Coverage and Participation Fairness", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_fairness_tier_coverage")


def plot_scaling(scaling: pd.DataFrame, plots_dir: Path) -> None:
    if scaling.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method in METHOD_ORDER:
        group = scaling[scaling["method"] == method].sort_values("N")
        if group.empty:
            continue
        ax.plot(
            group["N"],
            1000.0 * group["mean_selection_seconds"],
            color=COLORS[method],
            marker="o",
            linewidth=2.4 if method == "research.maml_select" else 1.5,
            label=METHOD_LABELS[method],
            zorder=4 if method in {"research.maml_select", "ml.fedcor"} else 2,
        )
    ax.set_xlabel("Client pool size N")
    ax.set_ylabel("Mean selection overhead (ms)")
    ax.set_yscale("log")
    ax.set_title("Client-Selection Overhead Scaling")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    save_figure(fig, plots_dir, "review_fig_scaling_overhead")


def write_readme(output_dir: Path, runs: pd.DataFrame, lambda_frame: pd.DataFrame, ablation_frame: pd.DataFrame) -> None:
    notes = [
        "# MAML-Select Review Artifact Pack",
        "",
        "This directory contains reviewer-facing tables and figures generated from:",
        "",
        "- `runs/maml_select_cifar100` for CIFAR-100.",
        "- `/Users/advaitpathak/Desktop/maml_select_runs_windows` for Fashion-MNIST and CIFAR-10.",
        "- `runs/maml_select` for lambda sensitivity, feature ablation, and scaling overhead.",
        "",
        "TiFL and FedCor are averaged over the seeds that are actually present; no missing seeds are imputed.",
        "Incomplete round-only traces are excluded from the main benchmark summaries.",
        "",
        f"Main benchmark runs loaded: {len(runs)}",
        f"Lambda variants loaded: {len(lambda_frame)}",
        f"Feature-ablation variants loaded: {len(ablation_frame)}",
        "",
        "Every plot is saved as both `.eps` and `.png`.",
    ]
    (output_dir / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-results", type=Path, default=DEFAULT_WINDOWS_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mac-cifar100-results", type=Path, default=REPO_ROOT / "runs" / "maml_select_cifar100")
    parser.add_argument("--mac-review-results", type=Path, default=REPO_ROOT / "runs" / "maml_select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs, rounds = load_main_results(
        [
            (args.windows_results, "windows"),
            (args.mac_cifar100_results, "mac-cifar100"),
        ]
    )
    if runs.empty:
        raise SystemExit("No main benchmark runs found.")
    runs["method_key"] = runs["method_key"].astype(str)
    rounds["method_key"] = rounds["method_key"].astype(str)

    lambda_summary = load_variant_summary(args.mac_review_results, "lambda_sensitivity")
    ablation_summary = load_variant_summary(args.mac_review_results, "feature_ablation")
    scaling_path = args.mac_review_results / "scaling" / "scaling_results.csv"
    scaling = pd.read_csv(scaling_path) if scaling_path.exists() else pd.DataFrame()

    write_tables(runs, lambda_summary, ablation_summary, scaling, tables_dir)
    plot_convergence(rounds, plots_dir)
    plot_efficiency_grid(runs, plots_dir)
    plot_accuracy_energy_scatter(runs, plots_dir)
    plot_lambda(lambda_summary, plots_dir)
    plot_ablation(ablation_summary, plots_dir)
    plot_fairness(runs, plots_dir)
    plot_scaling(scaling, plots_dir)
    write_readme(output_dir, runs, lambda_summary, ablation_summary)

    print(f"Wrote review artifact pack to {output_dir}")
    print(f"  plots:  {plots_dir}")
    print(f"  tables: {tables_dir}")


if __name__ == "__main__":
    main()
