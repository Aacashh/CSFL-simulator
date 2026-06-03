"""Generate CSV summaries and EPS plots for the FSDD audio experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_NAMES = {
    "baseline.fedavg": "FedAvg",
    "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort",
    "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor",
    "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS",
    "research.maml_select": "MAML-Select",
}

METHOD_COLORS = {
    "baseline.fedavg": "#4C78A8",
    "system_aware.fedcs": "#F58518",
    "system_aware.oort": "#54A24B",
    "system_aware.tifl": "#E45756",
    "ml.fedcor": "#B279A2",
    "research.criticalfl": "#9D755D",
    "research.fedgcs": "#FF9DA7",
    "research.maml_select": "#1B9E77",
}


def _float(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _payloads(results_dir: Path) -> List[Dict[str, Any]]:
    payloads = []
    for path in sorted(results_dir.rglob("result.json")):
        with path.open() as handle:
            payload = json.load(handle)
        if payload.get("experiment_id") == "audio_fsdd_100r":
            payload["_path"] = str(path)
            payloads.append(payload)
    return payloads


def _final_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    final = payload["simulation"]["metrics"][-1]
    cfg = payload["simulation"]["config"]
    return {
        "method_key": payload["method_key"],
        "method": METHOD_NAMES.get(payload["method_key"], payload["method_key"]),
        "seed": int(payload["seed"]),
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "rounds": int(cfg["rounds"]),
        "final_accuracy": _float(final.get("accuracy")),
        "final_f1": _float(final.get("f1")),
        "cum_time": _float(final.get("cum_time")),
        "cum_training_tflops": _float(final.get("cum_training_tflops")),
        "cum_modelled_energy_wh": _float(final.get("cum_modelled_energy_wh")),
        "cum_modelled_carbon_g": _float(final.get("cum_modelled_carbon_g")),
        "cum_comm_mb": _float(final.get("cum_comm_mb")),
        "fairness_gini": _float(final.get("fairness_gini")),
        "fairness_jain": _float(final.get("fairness_jain")),
        "participation_coverage_ratio": _float(final.get("participation_coverage_ratio")),
        "utilization_entropy": _float(final.get("utilization_entropy")),
        "label_coverage_ratio": _float(final.get("label_coverage_ratio")),
    }


def _round_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for metric in payload["simulation"]["metrics"]:
        if not bool(metric.get("evaluated", True)):
            continue
        row = {
            "method_key": payload["method_key"],
            "method": METHOD_NAMES.get(payload["method_key"], payload["method_key"]),
            "seed": int(payload["seed"]),
            "round": int(metric.get("round", -1)),
            "accuracy": _float(metric.get("accuracy")),
            "f1": _float(metric.get("f1")),
            "cum_time": _float(metric.get("cum_time")),
            "cum_training_tflops": _float(metric.get("cum_training_tflops")),
            "cum_modelled_energy_wh": _float(metric.get("cum_modelled_energy_wh")),
            "cum_modelled_carbon_g": _float(metric.get("cum_modelled_carbon_g")),
            "cum_comm_mb": _float(metric.get("cum_comm_mb")),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_eps(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = plots_dir / f"{stem}.eps"
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy(round_rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    methods = sorted({row["method_key"] for row in round_rows})
    for method in methods:
        rows = sorted([row for row in round_rows if row["method_key"] == method], key=lambda r: r["round"])
        ax.plot(
            [row["round"] for row in rows],
            [row["accuracy"] for row in rows],
            label=METHOD_NAMES.get(method, method),
            color=METHOD_COLORS.get(method, "#777777"),
            linewidth=2.0,
        )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("FSDD Audio: Accuracy vs. Rounds")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save_eps(fig, plots_dir, "audio_fsdd_accuracy_rounds")


def plot_efficiency(round_rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    methods = sorted({row["method_key"] for row in round_rows})
    panels = [
        ("cum_time", "Cumulative Modeled Time (s)"),
        ("cum_training_tflops", "Cumulative Training TFLOPs"),
        ("cum_modelled_energy_wh", "Cumulative Modeled Energy (Wh)"),
    ]
    for ax, (x_key, x_label) in zip(axes, panels):
        for method in methods:
            rows = sorted([row for row in round_rows if row["method_key"] == method], key=lambda r: r["round"])
            ax.plot(
                [row[x_key] for row in rows],
                [row["accuracy"] for row in rows],
                label=METHOD_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, "#777777"),
                linewidth=1.8,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Test Accuracy")
        ax.grid(color="#DDDDDD", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.78)
    _save_eps(fig, plots_dir, "audio_fsdd_efficiency")


def plot_final_bars(final_rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    rows = sorted(final_rows, key=lambda row: row["final_accuracy"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    labels = [row["method"] for row in rows]
    values = [row["final_accuracy"] for row in rows]
    colors = [METHOD_COLORS.get(row["method_key"], "#777777") for row in rows]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("FSDD Audio: Final Accuracy After 100 Rounds")
    ax.set_ylim(0.0, max(1.0, max(values) * 1.10 if values else 1.0))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.tick_params(axis="x", labelrotation=35)
    _save_eps(fig, plots_dir, "audio_fsdd_final_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = _payloads(args.results_dir)
    if not payloads:
        raise SystemExit(f"No audio_fsdd_100r result.json files found under {args.results_dir}")
    final_rows = [_final_row(payload) for payload in payloads]
    round_rows = [row for payload in payloads for row in _round_rows(payload)]
    _write_csv(args.analysis_dir / "audio_fsdd_summary.csv", final_rows)
    _write_csv(args.analysis_dir / "audio_fsdd_round_metrics.csv", round_rows)
    plot_accuracy(round_rows, args.plots_dir)
    plot_efficiency(round_rows, args.plots_dir)
    plot_final_bars(final_rows, args.plots_dir)
    print(f"Wrote audio FSDD summaries to {args.analysis_dir}")


if __name__ == "__main__":
    main()
