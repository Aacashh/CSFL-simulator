"""Aggregate MAML-Select experiment results and export manuscript-ready EPS plots."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from csfl_simulator.core.utils import ART_ROOT


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DEFAULT_RESULTS = ART_ROOT / "maml_select_letter"
HERE = Path(__file__).resolve().parent
METHOD_NAMES = {
    "baseline.fedavg": "FedAvg",
    "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort",
    "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor (approx.)",
    "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS",
    "research.maml_select": "MAML-Select",
}
COLORS = {
    "baseline.fedavg": "#4C78A8",
    "system_aware.fedcs": "#F58518",
    "system_aware.oort": "#54A24B",
    "system_aware.tifl": "#E45756",
    "ml.fedcor": "#B279A2",
    "research.criticalfl": "#9D755D",
    "research.fedgcs": "#FF9DA7",
    "research.maml_select": "#1B9E77",
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _float(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _final_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["simulation"]["metrics"][-1]


def _flatten(payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
    final = _final_metrics(payload)
    hardware = payload.get("hardware_energy", {})
    hardware_status = hardware.get("status", "unknown")
    measured_energy = _float(hardware.get("measured_energy_kwh")) if hardware_status == "measured" else float("nan")
    measured_carbon = (
        _float(hardware.get("estimated_emissions_g_declared_intensity"))
        if hardware_status == "measured"
        else float("nan")
    )
    tracked_energy = (
        _float(hardware.get("measured_energy_kwh"))
        if hardware_status in {"measured", "tracked_unverified"}
        else float("nan")
    )
    tracked_carbon = (
        _float(hardware.get("estimated_emissions_g_declared_intensity"))
        if hardware_status in {"measured", "tracked_unverified"}
        else float("nan")
    )
    config = payload["simulation"]["config"]
    return {
        "source": str(path),
        "experiment_id": payload["experiment_id"],
        "scenario_name": payload["scenario_name"],
        "method_key": payload["method_key"],
        "method_label": payload.get("method_label", payload["method_key"]),
        "seed": int(payload["seed"]),
        "dataset": config["dataset"],
        "model": config["model"],
        "total_clients": int(config["total_clients"]),
        "clients_per_round": int(config["clients_per_round"]),
        "rounds": int(config["rounds"]),
        "dirichlet_alpha": _float(config["dirichlet_alpha"]),
        "final_accuracy": _float(final.get("accuracy")),
        "final_f1": _float(final.get("f1")),
        "cum_time": _float(final.get("cum_time")),
        "cum_training_tflops": _float(final.get("cum_training_tflops")),
        "cum_comm_mb": _float(final.get("cum_comm_mb")),
        "cum_modelled_energy_wh": _float(final.get("cum_modelled_energy_wh")),
        "cum_modelled_carbon_g": _float(final.get("cum_modelled_carbon_g")),
        "fairness_gini": _float(final.get("fairness_gini")),
        "fairness_jain": _float(final.get("fairness_jain")),
        "utilization_entropy": _float(final.get("utilization_entropy")),
        "participation_coverage_ratio": _float(final.get("participation_coverage_ratio")),
        "tier_0_selection_rate": _float(final.get("tier_0_selection_rate")),
        "tier_1_selection_rate": _float(final.get("tier_1_selection_rate")),
        "tier_2_selection_rate": _float(final.get("tier_2_selection_rate")),
        "label_coverage_ratio": _float(final.get("label_coverage_ratio")),
        "mean_cohort_size": _float(final.get("mean_cohort_size")),
        "time_to_80pct_final": _float(final.get("time_to_80pct_final")),
        "report_accuracy_target": _float(final.get("report_accuracy_target")),
        "target_reached": bool(final.get("target_reached", False)),
        "rounds_to_target": _float(final.get("rounds_to_target")),
        "time_to_target": _float(final.get("time_to_target")),
        "training_tflops_to_target": _float(final.get("training_tflops_to_target")),
        "modelled_energy_wh_to_target": _float(final.get("modelled_energy_wh_to_target")),
        "comm_mb_to_target": _float(final.get("comm_mb_to_target")),
        "rounds_completed": int(payload["simulation"].get("rounds_completed", config["rounds"])),
        "hardware_status": hardware_status,
        "measured_energy_kwh": measured_energy,
        "estimated_emissions_g_declared_intensity": measured_carbon,
        "tracked_energy_kwh": tracked_energy,
        "tracked_carbon_g_declared_intensity": tracked_carbon,
    }


def load_payloads(results_dir: Path) -> List[Dict[str, Any]]:
    payloads = []
    for path in sorted(Path(results_dir).rglob("result.json")):
        payload = _load(path)
        payload["_result_path"] = path
        payloads.append(payload)
    return payloads


def aggregate(payloads: Sequence[Dict[str, Any]], external_csv: Path | None) -> pd.DataFrame:
    rows = [_flatten(payload, payload["_result_path"]) for payload in payloads]
    frame = pd.DataFrame(rows)
    if external_csv and external_csv.exists():
        external = pd.read_csv(external_csv)
        external["source"] = str(external_csv)
        frame = pd.concat([frame, external], ignore_index=True, sort=False)
    return frame


def _name(key: str) -> str:
    return METHOD_NAMES.get(key, key.replace("research.maml_select.", "MAML-Select: "))


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.eps", format="eps", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _line_summary(payloads: Sequence[Dict[str, Any]], scenario_name: str) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        if payload["experiment_id"] != "main_benchmarks" or payload["scenario_name"] != scenario_name:
            continue
        for metric in payload["simulation"]["metrics"]:
            if not bool(metric.get("evaluated", True)):
                continue
            rows.append(
                {
                    "method_key": payload["method_key"],
                    "seed": payload["seed"],
                    "round": int(metric["round"]),
                    "accuracy": _float(metric["accuracy"]),
                    "cum_time": _float(metric.get("cum_time", 0.0)),
                    "cum_training_tflops": _float(metric.get("cum_training_tflops", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def round_metrics_table(payloads: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for metric in payload["simulation"]["metrics"]:
            row = {
                "experiment_id": payload["experiment_id"],
                "scenario_name": payload["scenario_name"],
                "method_key": payload["method_key"],
                "method_label": payload.get("method_label", payload["method_key"]),
                "seed": int(payload["seed"]),
            }
            row.update(metric)
            rows.append(row)
    return pd.DataFrame(rows)


def roundwise_main_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "accuracy",
        "f1",
        "cum_time",
        "cum_training_tflops",
        "cum_modelled_energy_wh",
        "cum_modelled_carbon_g",
        "cum_comm_mb",
        "fairness_jain",
        "utilization_entropy",
        "participation_coverage_ratio",
    ]
    subset = frame[
        (frame["experiment_id"] == "main_benchmarks")
        & (frame["round"] >= 0)
        & frame["evaluated"].astype(bool)
    ]
    return subset.groupby(["scenario_name", "method_key", "round"])[columns].agg(["mean", "std", "count"])


def plot_convergence(payloads: Sequence[Dict[str, Any]], output_dir: Path, scenario_name: str) -> None:
    frame = _line_summary(payloads, scenario_name)
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for method_key, group in frame.groupby("method_key"):
        summary = group.groupby("round").agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            tflops=("cum_training_tflops", "mean"),
        )
        color = COLORS.get(method_key)
        label = _name(method_key)
        axes[0].errorbar(
            summary.index,
            summary["accuracy"],
            yerr=summary["accuracy_std"].fillna(0.0),
            label=label,
            color=color,
            linewidth=1.5,
            capsize=1.5,
            elinewidth=0.65,
            errorevery=2,
        )
        axes[1].plot(summary["tflops"], summary["accuracy"], label=label, color=color, linewidth=1.5)
    axes[0].set_xlabel("FL round")
    axes[0].set_ylabel("Test accuracy")
    axes[1].set_xlabel("Cumulative training TFLOPs")
    axes[1].set_ylabel("Test accuracy")
    axes[0].grid(color="#DDDDDD", linewidth=0.5)
    axes[1].grid(color="#DDDDDD", linewidth=0.5)
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    _save(fig, output_dir, f"convergence_{scenario_name}")


def plot_figure2_efficiency(payloads: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    scenarios = [("fashion_main", "Fashion-MNIST"), ("cifar10_main", "CIFAR-10")]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    found = False
    for row, (scenario_name, dataset_label) in enumerate(scenarios):
        frame = _line_summary(payloads, scenario_name)
        if frame.empty:
            continue
        found = True
        for method_key, group in frame.groupby("method_key"):
            summary = group.groupby("round").agg(
                accuracy=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                cum_time=("cum_time", "mean"),
                tflops=("cum_training_tflops", "mean"),
            )
            color = COLORS.get(method_key)
            label = _name(method_key)
            lower = summary["accuracy"] - summary["accuracy_std"].fillna(0.0)
            upper = summary["accuracy"] + summary["accuracy_std"].fillna(0.0)
            for column, x_values in enumerate((summary["cum_time"], summary["tflops"])):
                axis = axes[row, column]
                axis.plot(x_values, summary["accuracy"], label=label, color=color)
                axis.fill_between(x_values, lower, upper, color=color, alpha=0.10, linewidth=0.0)
        axes[row, 0].set_ylabel(f"{dataset_label}\ntest accuracy")
        axes[row, 0].set_xlabel("Accumulated modeled latency")
        axes[row, 1].set_xlabel("Cumulative training TFLOPs")
        for axis in axes[row]:
            axis.grid(color="#DDDDDD", linewidth=0.5)
    if not found:
        plt.close(fig)
        return
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.subplots_adjust(top=0.88)
    _save(fig, output_dir, "fig2_efficiency_comparison")


def _bar_values(frame: pd.DataFrame, experiment_id: str, scenario_name: str, metric: str) -> pd.DataFrame:
    subset = frame[(frame["experiment_id"] == experiment_id) & (frame["scenario_name"] == scenario_name)].copy()
    subset = subset[np.isfinite(pd.to_numeric(subset[metric], errors="coerce"))]
    return subset.groupby("method_key")[metric].agg(["mean", "std"]).sort_index()


def plot_hardware_energy(frame: pd.DataFrame, output_dir: Path, scenario_name: str) -> None:
    subset = frame[
        (frame["experiment_id"] == "main_benchmarks")
        & (frame["scenario_name"] == scenario_name)
        & (frame["hardware_status"] == "measured")
    ].copy()
    if subset.empty:
        return
    summary = subset.groupby("method_key").agg(
        energy=("measured_energy_kwh", "mean"),
        energy_std=("measured_energy_kwh", "std"),
        carbon=("estimated_emissions_g_declared_intensity", "mean"),
        carbon_std=("estimated_emissions_g_declared_intensity", "std"),
    )
    keys = list(summary.index)
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    colors = [COLORS.get(key, "#777777") for key in keys]
    axes[0].bar(x, summary["energy"], yerr=summary["energy_std"].fillna(0.0), color=colors, capsize=2)
    axes[1].bar(x, summary["carbon"], yerr=summary["carbon_std"].fillna(0.0), color=colors, capsize=2)
    for axis in axes:
        axis.set_xticks(x, [_name(key) for key in keys], rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].set_ylabel("Measured compute energy (kWh)")
    axes[1].set_ylabel("Estimated emissions (gCO2eq)")
    _save(fig, output_dir, f"hardware_energy_carbon_{scenario_name}")


def plot_tracked_energy_estimate(frame: pd.DataFrame, output_dir: Path, scenario_name: str) -> None:
    subset = frame[
        (frame["experiment_id"] == "main_benchmarks")
        & (frame["scenario_name"] == scenario_name)
        & frame["hardware_status"].isin(["measured", "tracked_unverified"])
    ].copy()
    if subset.empty:
        return
    summary = subset.groupby("method_key").agg(
        energy=("tracked_energy_kwh", "mean"),
        energy_std=("tracked_energy_kwh", "std"),
        carbon=("tracked_carbon_g_declared_intensity", "mean"),
        carbon_std=("tracked_carbon_g_declared_intensity", "std"),
    )
    keys = list(summary.index)
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    colors = [COLORS.get(key, "#777777") for key in keys]
    axes[0].bar(x, summary["energy"], yerr=summary["energy_std"].fillna(0.0), color=colors, capsize=2)
    axes[1].bar(x, summary["carbon"], yerr=summary["carbon_std"].fillna(0.0), color=colors, capsize=2)
    for axis in axes:
        axis.set_xticks(x, [_name(key) for key in keys], rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].set_ylabel("CodeCarbon tracked estimate (kWh)")
    axes[1].set_ylabel("Estimated emissions (gCO2eq)")
    _save(fig, output_dir, f"tracked_energy_estimate_{scenario_name}")


def plot_energy_to_target(frame: pd.DataFrame, output_dir: Path, scenario_name: str) -> None:
    subset = frame[
        (frame["experiment_id"] == "hardware_energy_to_target")
        & (frame["scenario_name"] == scenario_name)
    ].copy()
    if subset.empty:
        return
    summary = subset.groupby("method_key").agg(
        modelled_energy=("cum_modelled_energy_wh", "mean"),
        modelled_energy_std=("cum_modelled_energy_wh", "std"),
        measured_energy=("measured_energy_kwh", "mean"),
        measured_energy_std=("measured_energy_kwh", "std"),
        target_reach_rate=("target_reached", "mean"),
    )
    keys = list(summary.index)
    x = np.arange(len(keys))
    colors = [COLORS.get(key, "#777777") for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    axes[0].bar(
        x,
        summary["modelled_energy"],
        yerr=summary["modelled_energy_std"].fillna(0.0),
        color=colors,
        capsize=2,
    )
    axes[1].bar(
        x,
        summary["measured_energy"],
        yerr=summary["measured_energy_std"].fillna(0.0),
        color=colors,
        capsize=2,
    )
    for axis in axes:
        axis.set_xticks(x, [_name(key) for key in keys], rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].set_ylabel("Modeled client energy (Wh)")
    axes[1].set_ylabel("Measured compute energy (kWh)")
    axes[0].set_title("Until target or round cap")
    axes[1].set_title("Until target or round cap")
    for index, reach_rate in enumerate(summary["target_reach_rate"]):
        axes[0].text(index, 0.0, f"{100.0 * reach_rate:.0f}% reached", rotation=90, va="bottom", ha="center", fontsize=6)
    _save(fig, output_dir, f"energy_to_target_{scenario_name}")


def plot_fairness(frame: pd.DataFrame, output_dir: Path, scenario_name: str) -> None:
    subset = frame[(frame["experiment_id"] == "main_benchmarks") & (frame["scenario_name"] == scenario_name)]
    if subset.empty:
        return
    summary = subset.groupby("method_key")[["fairness_jain", "utilization_entropy"]].mean()
    keys = list(summary.index)
    x = np.arange(len(keys))
    width = 0.36
    fig, axis = plt.subplots(figsize=(4.8, 2.75))
    axis.bar(x - width / 2, summary["fairness_jain"], width, label="Jain index", color="#4C78A8")
    axis.bar(x + width / 2, summary["utilization_entropy"], width, label="Utilization entropy", color="#F58518")
    axis.set_xticks(x, [_name(key) for key in keys], rotation=40, ha="right", fontsize=7)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Participation fairness (higher is better)")
    axis.legend(frameon=False, fontsize=7)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    _save(fig, output_dir, f"fairness_{scenario_name}")


def plot_variant_bars(frame: pd.DataFrame, output_dir: Path, experiment_id: str, stem: str) -> None:
    subset = frame[frame["experiment_id"] == experiment_id].copy()
    if subset.empty:
        return
    summary = subset.groupby(["method_key", "method_label"]).agg(
        accuracy=("final_accuracy", "mean"),
        accuracy_std=("final_accuracy", "std"),
        energy=("cum_modelled_energy_wh", "mean"),
    )
    labels = [_name(label) for _, label in summary.index]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    axes[0].bar(x, summary["accuracy"], yerr=summary["accuracy_std"].fillna(0.0), color="#4C78A8", capsize=2)
    axes[1].bar(x, summary["energy"], color="#1B9E77")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=40, ha="right", fontsize=7)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].set_ylabel("Final test accuracy")
    axes[1].set_ylabel("Modeled client energy (Wh)")
    _save(fig, output_dir, stem)


def plot_scaling(frame: pd.DataFrame, output_dir: Path) -> None:
    subset = frame[frame["experiment_id"] == "scaling"].copy()
    if subset.empty:
        return
    summary = subset.groupby(["method_key", "total_clients"]).agg(
        energy=("cum_modelled_energy_wh", "mean"),
        accuracy=("final_accuracy", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for method_key in summary.index.get_level_values(0).unique():
        group = summary.loc[method_key]
        axes[0].plot(group.index, group["accuracy"], marker="o", label=_name(method_key), color=COLORS.get(method_key))
        axes[1].plot(group.index, group["energy"], marker="o", label=_name(method_key), color=COLORS.get(method_key))
    axes[0].set_ylabel("Final test accuracy")
    axes[1].set_ylabel("Modeled client energy (Wh)")
    for axis in axes:
        axis.set_xlabel("Number of clients")
        axis.grid(color="#DDDDDD", linewidth=0.5)
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, output_dir, "client_count_scaling")


def significance_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = frame[frame["experiment_id"] == "main_benchmarks"]
    metrics = [
        "final_accuracy",
        "cum_time",
        "cum_training_tflops",
        "cum_modelled_energy_wh",
        "fairness_gini",
        "measured_energy_kwh",
    ]
    for scenario_name, scenario in subset.groupby("scenario_name"):
        maml = scenario[scenario["method_key"] == "research.maml_select"].set_index("seed")
        for method_key, baseline in scenario.groupby("method_key"):
            if method_key == "research.maml_select":
                continue
            baseline = baseline.set_index("seed")
            shared = sorted(set(maml.index).intersection(baseline.index))
            for metric in metrics:
                left = pd.to_numeric(maml.loc[shared, metric], errors="coerce")
                right = pd.to_numeric(baseline.loc[shared, metric], errors="coerce")
                valid = np.isfinite(left) & np.isfinite(right)
                left, right = left[valid], right[valid]
                test = ttest_rel(left, right) if len(left) >= 2 else None
                difference = left - right
                difference_std = float(difference.std(ddof=1)) if len(difference) >= 2 else float("nan")
                difference_mean = float(difference.mean()) if len(difference) else float("nan")
                margin = (
                    float(student_t.ppf(0.975, len(difference) - 1) * difference_std / math.sqrt(len(difference)))
                    if len(difference) >= 2 and math.isfinite(difference_std)
                    else float("nan")
                )
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "baseline": method_key,
                        "metric": metric,
                        "paired_seeds": int(len(left)),
                        "maml_minus_baseline": difference_mean,
                        "difference_95pct_ci_low": difference_mean - margin,
                        "difference_95pct_ci_high": difference_mean + margin,
                        "paired_cohens_dz": difference_mean / difference_std if difference_std > 0.0 else float("nan"),
                        "paired_t_statistic": float(test.statistic) if test else float("nan"),
                        "paired_p_value": float(test.pvalue) if test else float("nan"),
                    }
                )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["holm_adjusted_p_value"] = float("nan")
    for _, indices in tests.groupby(["scenario_name", "metric"]).groups.items():
        valid = tests.loc[indices, "paired_p_value"].dropna().sort_values()
        adjusted_so_far = 0.0
        count = len(valid)
        for rank, (index, p_value) in enumerate(valid.items()):
            adjusted_so_far = max(adjusted_so_far, min(1.0, (count - rank) * float(p_value)))
            tests.loc[index, "holm_adjusted_p_value"] = adjusted_so_far
    return tests


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "final_accuracy",
        "cum_time",
        "cum_training_tflops",
        "cum_modelled_energy_wh",
        "measured_energy_kwh",
        "estimated_emissions_g_declared_intensity",
        "tracked_energy_kwh",
        "tracked_carbon_g_declared_intensity",
        "fairness_jain",
        "utilization_entropy",
        "participation_coverage_ratio",
        "tier_0_selection_rate",
        "tier_1_selection_rate",
        "tier_2_selection_rate",
        "target_reached",
        "rounds_to_target",
        "time_to_target",
        "training_tflops_to_target",
        "modelled_energy_wh_to_target",
        "comm_mb_to_target",
    ]
    main = frame[frame["experiment_id"] == "main_benchmarks"]
    return main.groupby(["scenario_name", "method_key"])[columns].agg(["mean", "std", "count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--external-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.results_dir)
    if not payloads:
        raise SystemExit(f"No result.json files found under {args.results_dir}")
    frame = aggregate(payloads, args.external_csv)
    frame.to_csv(output_dir / "runs.csv", index=False)
    round_metrics = round_metrics_table(payloads)
    round_metrics.to_csv(output_dir / "round_metrics.csv", index=False)
    round_summary = roundwise_main_summary(round_metrics)
    round_summary.to_csv(output_dir / "roundwise_main_summary.csv")
    (output_dir / "roundwise_main_summary.tex").write_text(round_summary.to_latex(float_format="%.4f"))
    summary = summary_table(frame)
    summary.to_csv(output_dir / "main_summary.csv")
    (output_dir / "main_summary.tex").write_text(summary.to_latex(float_format="%.4f"))
    energy = frame[frame["experiment_id"] == "hardware_energy_to_target"]
    if not energy.empty:
        energy_summary = energy.groupby(["scenario_name", "method_key"])[
            [
                "target_reached",
                "rounds_completed",
                "cum_modelled_energy_wh",
                "measured_energy_kwh",
                "estimated_emissions_g_declared_intensity",
            ]
        ].agg(["mean", "std", "count"])
        energy_summary.to_csv(output_dir / "energy_to_target_summary.csv")
        (output_dir / "energy_to_target_summary.tex").write_text(energy_summary.to_latex(float_format="%.4f"))
    tests = significance_tests(frame)
    tests.to_csv(output_dir / "paired_significance_tests.csv", index=False)

    plot_convergence(payloads, output_dir, "fashion_main")
    plot_convergence(payloads, output_dir, "cifar10_main")
    plot_figure2_efficiency(payloads, output_dir)
    plot_hardware_energy(frame, output_dir, "fashion_main")
    plot_hardware_energy(frame, output_dir, "cifar10_main")
    plot_tracked_energy_estimate(frame, output_dir, "fashion_main")
    plot_tracked_energy_estimate(frame, output_dir, "cifar10_main")
    plot_energy_to_target(frame, output_dir, "fashion_energy_target")
    plot_energy_to_target(frame, output_dir, "cifar10_energy_target")
    plot_fairness(frame, output_dir, "fashion_main")
    plot_fairness(frame, output_dir, "cifar10_main")
    plot_variant_bars(frame, output_dir, "local_fashion_pilot", "local_fashion_pilot")
    plot_variant_bars(frame, output_dir, "local_lambda_pilot", "local_lambda_pilot")
    plot_variant_bars(frame, output_dir, "lambda_sensitivity", "lambda_sensitivity")
    plot_variant_bars(frame, output_dir, "feature_ablation", "feature_ablation")
    plot_scaling(frame, output_dir)
    print(f"Wrote aggregate tables and EPS/PDF/PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
