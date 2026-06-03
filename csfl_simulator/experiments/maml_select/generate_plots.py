"""Generate publication-quality EPS plots for the MAML-Select manuscript.

ALL plots are saved strictly in .eps format for publication quality.

Generates:
    1. Figure 2: 4-panel efficiency analysis (Acc vs Rounds, vs SimTime, vs TFLOPS, vs Comm)
    2. Lambda Sensitivity: dual-axis line plot
    3. Feature Ablation: bar chart
    4. Fairness & Coverage: stacked bar chart by hardware tier
    5. Scaling Overhead: line plot showing O(N) vs O(N³)

Usage:
    python -m csfl_simulator.experiments.maml_select.generate_plots
    python -m csfl_simulator.experiments.maml_select.generate_plots --results-dir <path>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RESULTS = REPO_ROOT / "runs" / "maml_select"
DEFAULT_PLOTS = REPO_ROOT / "artifacts" / "maml_select" / "plots"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# All 8 methods — display names, colors, markers
METHOD_META = {
    "baseline.fedavg":       {"name": "FedAvg",     "color": "#4C78A8", "marker": "o"},
    "system_aware.fedcs":    {"name": "FedCS",      "color": "#F58518", "marker": "s"},
    "system_aware.oort":     {"name": "Oort",       "color": "#54A24B", "marker": "^"},
    "system_aware.tifl":     {"name": "TiFL",       "color": "#E45756", "marker": "D"},
    "ml.fedcor":             {"name": "FedCor (approx.)", "color": "#B279A2", "marker": "v"},
    "research.criticalfl":   {"name": "CriticalFL", "color": "#9D755D", "marker": "P"},
    "research.fedgcs":       {"name": "FedGCS-style (approx.)", "color": "#FF9DA7", "marker": "X"},
    "research.maml_select":  {"name": "MAML-Select","color": "#1B9E77", "marker": "*"},
    "research.maml_select_v2": {"name": "MAML-Select v2", "color": "#0072B2", "marker": "*"},
}
MAIN_EXPERIMENT_IDS = {"main_benchmarks", "cifar100_benchmarks"}


def _name(key: str) -> str:
    return METHOD_META.get(key, {}).get("name", key)


def _color(key: str) -> str:
    return METHOD_META.get(key, {}).get("color", "#777777")


def _marker(key: str) -> str:
    return METHOD_META.get(key, {}).get("marker", "o")


def _save_eps(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save figure strictly in EPS format only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = output_dir / f"{stem}.eps"
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _float(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_main_payloads(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result.json files from the results directory."""
    payloads = []
    for path in sorted(Path(results_dir).rglob("result.json")):
        try:
            with path.open() as f:
                payload = json.load(f)
            payload["_path"] = str(path)
            payloads.append(payload)
        except Exception:
            continue
    return payloads


def _extract_round_series(
    payloads: List[Dict],
    experiment_id: str,
    scenario_name: str,
) -> Dict[str, Dict[str, List]]:
    """Extract per-method round-level series grouped by seed."""
    series: Dict[str, Dict[str, List]] = {}  # method -> {metric_name -> [values per round]}

    for payload in payloads:
        if payload.get("experiment_id") != experiment_id:
            continue
        if payload.get("scenario_name") != scenario_name:
            continue

        method = payload["method_key"]
        if method not in series:
            series[method] = {"rounds": [], "accuracies": [], "cum_times": [],
                              "cum_tflops": [], "cum_comms": [], "seeds": []}

        seed_data = {"round": [], "accuracy": [], "cum_time": [],
                     "cum_training_tflops": [], "cum_comm_mb": []}
        for m in payload["simulation"]["metrics"]:
            if not bool(m.get("evaluated", True)):
                continue
            seed_data["round"].append(int(m.get("round", -1)))
            seed_data["accuracy"].append(_float(m.get("accuracy", 0)))
            seed_data["cum_time"].append(_float(m.get("cum_time", 0)))
            seed_data["cum_training_tflops"].append(_float(m.get("cum_training_tflops", 0)))
            seed_data["cum_comm_mb"].append(_float(m.get("cum_comm_mb", 0)))

        series[method]["seeds"].append(seed_data)
    return series


def _mean_std_series(seeds_data: List[Dict]) -> Dict[str, np.ndarray]:
    """Compute mean/std across seeds at each round index."""
    if not seeds_data:
        return {}

    # Align by round index
    min_len = min(len(sd["round"]) for sd in seeds_data)
    result = {}
    for key in ["round", "accuracy", "cum_time", "cum_training_tflops", "cum_comm_mb"]:
        matrix = np.array([sd[key][:min_len] for sd in seeds_data])
        result[f"{key}_mean"] = matrix.mean(axis=0)
        result[f"{key}_std"] = matrix.std(axis=0) if matrix.shape[0] > 1 else np.zeros(min_len)
    return result


# ── Plot 1: Figure 2 — 4-Panel Efficiency ────────────────────────────────────

def plot_figure2(payloads: List[Dict], output_dir: Path) -> None:
    """Recreate the 4-panel efficiency analysis for both datasets."""
    scenarios = [
        ("main_benchmarks", "fashion_main", "Fashion-MNIST"),
        ("main_benchmarks", "cifar10_main", "CIFAR-10"),
        ("cifar100_benchmarks", "cifar100_main", "CIFAR-100"),
    ]

    for experiment_id, scenario_name, dataset_label in scenarios:
        series = _extract_round_series(payloads, experiment_id, scenario_name)
        if not series:
            # Try cifar10_reconciled too
            series = _extract_round_series(payloads, "cifar10_reconciled", scenario_name)
        if not series:
            print(f"  [skip] No data for {scenario_name}")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for method_key, data in series.items():
            ms = _mean_std_series(data["seeds"])
            if not ms:
                continue

            name = _name(method_key)
            color = _color(method_key)
            marker = _marker(method_key)
            markevery = max(1, len(ms["round_mean"]) // 10)

            # Panel 1: Accuracy vs Rounds
            axes[0, 0].plot(ms["round_mean"], ms["accuracy_mean"],
                           label=name, color=color, marker=marker, markevery=markevery)
            axes[0, 0].fill_between(ms["round_mean"],
                                    ms["accuracy_mean"] - ms["accuracy_std"],
                                    ms["accuracy_mean"] + ms["accuracy_std"],
                                    color=color, alpha=0.1)

            # Panel 2: Accuracy vs Simulation Time
            axes[0, 1].plot(ms["cum_time_mean"], ms["accuracy_mean"],
                           label=name, color=color, marker=marker, markevery=markevery)
            axes[0, 1].fill_between(ms["cum_time_mean"],
                                    ms["accuracy_mean"] - ms["accuracy_std"],
                                    ms["accuracy_mean"] + ms["accuracy_std"],
                                    color=color, alpha=0.1)

            # Panel 3: Accuracy vs TFLOPS
            axes[1, 0].plot(ms["cum_training_tflops_mean"], ms["accuracy_mean"],
                           label=name, color=color, marker=marker, markevery=markevery)
            axes[1, 0].fill_between(ms["cum_training_tflops_mean"],
                                    ms["accuracy_mean"] - ms["accuracy_std"],
                                    ms["accuracy_mean"] + ms["accuracy_std"],
                                    color=color, alpha=0.1)

            # Panel 4: Accuracy vs Communication
            axes[1, 1].plot(ms["cum_comm_mb_mean"], ms["accuracy_mean"],
                           label=name, color=color, marker=marker, markevery=markevery)
            axes[1, 1].fill_between(ms["cum_comm_mb_mean"],
                                    ms["accuracy_mean"] - ms["accuracy_std"],
                                    ms["accuracy_mean"] + ms["accuracy_std"],
                                    color=color, alpha=0.1)

        axes[0, 0].set_xlabel("Communication Round")
        axes[0, 0].set_ylabel("Test Accuracy")
        axes[0, 0].set_title(f"{dataset_label} — Accuracy vs. Rounds")

        axes[0, 1].set_xlabel("Cumulative Simulation Time (s)")
        axes[0, 1].set_ylabel("Test Accuracy")
        axes[0, 1].set_title(f"{dataset_label} — Accuracy vs. Time")

        axes[1, 0].set_xlabel("Cumulative Training TFLOPs")
        axes[1, 0].set_ylabel("Test Accuracy")
        axes[1, 0].set_title(f"{dataset_label} — Accuracy vs. TFLOPs")

        axes[1, 1].set_xlabel("Cumulative Communication (MB)")
        axes[1, 1].set_ylabel("Test Accuracy")
        axes[1, 1].set_title(f"{dataset_label} — Accuracy vs. Communication")

        for ax in axes.flat:
            ax.grid(color="#DDDDDD", linewidth=0.5)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 1.02))
        fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.30)

        _save_eps(fig, output_dir, f"fig2_efficiency_{scenario_name}")


# ── Plot 2: Lambda Sensitivity (Dual-Axis) ───────────────────────────────────

def plot_lambda_sensitivity(
    payloads: List[Dict],
    output_dir: Path,
    sensitivity_dir: Optional[Path] = None,
) -> None:
    """Dual-axis plot: lambda vs accuracy (left) and lambda vs TFLOPS (right)."""
    data_dir = sensitivity_dir or (output_dir.parent / "sensitivity")
    csv_path = data_dir / "sensitivity_summary.csv"
    if not HAS_PANDAS:
        print("  [skip] pandas required for sensitivity plot")
        return

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for payload in payloads:
            if payload.get("experiment_id") != "lambda_sensitivity":
                continue
            final = payload["simulation"]["metrics"][-1]
            rows.append(
                {
                    "lambda": float(payload["method_params"]["lambda_latency"]),
                    "final_accuracy": _float(final.get("accuracy")),
                    "cum_training_tflops": _float(final.get("cum_training_tflops")),
                }
            )
        if not rows:
            print(f"  [skip] No sensitivity data at {csv_path}")
            return
        df = pd.DataFrame(rows).groupby("lambda").agg(
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            tflops_mean=("cum_training_tflops", "mean"),
            tflops_std=("cum_training_tflops", "std"),
        ).reset_index()
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    lambdas = df["lambda"].values
    acc_mean = df["accuracy_mean"].values
    acc_std = df.get("accuracy_std", pd.Series([0]*len(df))).values
    tflops_mean = df["tflops_mean"].values
    tflops_std = df.get("tflops_std", pd.Series([0]*len(df))).values

    # Accuracy on left axis
    line1 = ax1.errorbar(lambdas, acc_mean, yerr=acc_std,
                         color="#1B9E77", marker="o", linewidth=2.5,
                         capsize=4, label="Final Accuracy", zorder=5)
    ax1.set_xlabel(r"Cost-Function Trade-off ($\lambda$)", fontsize=14)
    ax1.set_ylabel("Final Test Accuracy", color="#1B9E77", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="#1B9E77")

    # TFLOPS on right axis
    line2 = ax2.errorbar(lambdas, tflops_mean, yerr=tflops_std,
                         color="#E45756", marker="s", linewidth=2.5,
                         capsize=4, linestyle="--", label="Total TFLOPs", zorder=5)
    ax2.set_ylabel("Total Training TFLOPs", color="#E45756", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#E45756")

    ax1.set_xscale("log")
    ax1.grid(color="#DDDDDD", linewidth=0.5)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=False, fontsize=11)

    fig.suptitle(r"MAML-Select Sensitivity to $\lambda$", fontsize=15, y=0.98)
    _save_eps(fig, output_dir, "lambda_sensitivity")


# ── Plot 3: Feature Ablation Bar Chart ────────────────────────────────────────

def plot_feature_ablation(
    payloads: List[Dict],
    output_dir: Path,
    ablation_dir: Optional[Path] = None,
) -> None:
    """Bar chart comparing accuracy when specific features are removed."""
    data_dir = ablation_dir or (output_dir.parent / "ablation")
    csv_path = data_dir / "ablation_summary.csv"
    if not HAS_PANDAS:
        print("  [skip] pandas required for ablation plot")
        return

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for payload in payloads:
            if payload.get("experiment_id") != "feature_ablation":
                continue
            final = payload["simulation"]["metrics"][-1]
            rows.append(
                {
                    "variant": payload.get("method_label", payload["method_key"]),
                    "final_accuracy": _float(final.get("accuracy")),
                    "cum_training_tflops": _float(final.get("cum_training_tflops")),
                }
            )
        if not rows:
            print(f"  [skip] No ablation data at {csv_path}")
            return
        df = pd.DataFrame(rows).groupby("variant").agg(
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            tflops_mean=("cum_training_tflops", "mean"),
        ).reset_index()
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    variants = df["variant"].values
    acc_mean = df["accuracy_mean"].values
    acc_std = df.get("accuracy_std", pd.Series([0]*len(df))).values

    x = np.arange(len(variants))
    colors = ["#1B9E77" if "All" in v else "#4C78A8" for v in variants]

    bars = ax.bar(x, acc_mean, yerr=acc_std, color=colors, capsize=4,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Final Test Accuracy", fontsize=14)
    ax.set_title("Feature Ablation Study", fontsize=15)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, acc_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    _save_eps(fig, output_dir, "feature_ablation")


# ── Plot 4: Fairness & Coverage (Stacked Bar) ────────────────────────────────

def plot_fairness_coverage(payloads: List[Dict], output_dir: Path) -> None:
    """Stacked bar chart of tier selection frequencies over 200 rounds."""
    method_tiers: Dict[str, Dict[int, float]] = {}

    for payload in payloads:
        if payload.get("experiment_id") not in MAIN_EXPERIMENT_IDS:
            continue
        if payload.get("scenario_name") not in ("fashion_main", "cifar10_main", "cifar100_main"):
            continue

        method = payload["method_key"]
        final = payload["simulation"]["metrics"][-1]

        if method not in method_tiers:
            method_tiers[method] = {0: [], 1: [], 2: []}

        for tier in (0, 1, 2):
            rate = _float(final.get(f"tier_{tier}_selection_rate", 0))
            method_tiers[method][tier].append(rate)

    if not method_tiers:
        print("  [skip] No fairness data available")
        return

    # Average across seeds/scenarios
    methods = sorted(method_tiers.keys(), key=lambda k: list(METHOD_META.keys()).index(k)
                     if k in METHOD_META else 99)
    tier0 = [np.mean(method_tiers[m][0]) if method_tiers[m][0] else 0 for m in methods]
    tier1 = [np.mean(method_tiers[m][1]) if method_tiers[m][1] else 0 for m in methods]
    tier2 = [np.mean(method_tiers[m][2]) if method_tiers[m][2] else 0 for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.65

    ax.bar(x, tier0, width, label="Tier 1 (Low-end)", color="#F58518")
    ax.bar(x, tier1, width, bottom=tier0, label="Tier 2 (Mid-range)", color="#4C78A8")
    ax.bar(x, tier2, width, bottom=np.array(tier0) + np.array(tier1),
           label="Tier 3 (High-end)", color="#1B9E77")

    ax.set_xticks(x)
    ax.set_xticklabels([_name(m) for m in methods], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Selection Rate (fraction)", fontsize=14)
    ax.set_title("Client Selection by Hardware Tier", fontsize=15)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_ylim(0, 1.1)

    _save_eps(fig, output_dir, "fairness_coverage_tiers")


# ── Plot 5: Scaling Overhead ─────────────────────────────────────────────────

def plot_scaling_overhead(output_dir: Path, scaling_dir: Optional[Path] = None) -> None:
    """Line plot of selection overhead as N scales, highlighting O(N) vs O(N³)."""
    data_dir = scaling_dir or (output_dir.parent / "scaling")
    csv_path = data_dir / "scaling_results.csv"
    if not csv_path.exists():
        print(f"  [skip] No scaling data at {csv_path}")
        return

    if not HAS_PANDAS:
        print("  [skip] pandas required for scaling plot")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    highlight_methods = {
        "research.maml_select": {"lw": 3.0, "ls": "-"},
        "research.maml_select_v2": {"lw": 3.0, "ls": "--"},
        "ml.fedcor": {"lw": 3.0, "ls": "-"},
    }

    for method in df["method"].unique():
        mdf = df[df["method"] == method].sort_values("N")
        meta = METHOD_META.get(method, {})
        name = meta.get("name", method)
        color = meta.get("color", "#777777")
        marker = meta.get("marker", "o")
        style = highlight_methods.get(method, {"lw": 1.5, "ls": "--"})

        ax.plot(mdf["N"], mdf["mean_selection_seconds"] * 1000,
                label=name, color=color, marker=marker,
                linewidth=style["lw"], linestyle=style["ls"])

    ax.set_xlabel("Number of Clients ($N$)", fontsize=14)
    ax.set_ylabel("Mean Selection Time (ms)", fontsize=14)
    ax.set_title("Client Selection Overhead Scaling", fontsize=15)
    ax.set_yscale("log")
    ax.grid(color="#DDDDDD", linewidth=0.5, which="both")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    # Add complexity annotations
    ax.annotate(r"$\mathcal{O}(N \cdot |\phi|)$", xy=(0.75, 0.15),
                xycoords="axes fraction", fontsize=13, color="#1B9E77",
                fontweight="bold")
    ax.annotate(r"$\mathcal{O}(N^3)$", xy=(0.75, 0.75),
                xycoords="axes fraction", fontsize=13, color="#B279A2",
                fontweight="bold")

    _save_eps(fig, output_dir, "scaling_overhead")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Root directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS,
                        help="Output directory for EPS publication plots")
    parser.add_argument("--sensitivity-dir", type=Path, default=None,
                        help="Directory containing sensitivity_summary.csv")
    parser.add_argument("--ablation-dir", type=Path, default=None,
                        help="Directory containing ablation_summary.csv")
    parser.add_argument("--scaling-dir", type=Path, default=None,
                        help="Directory containing scaling_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    payloads = load_main_payloads(args.results_dir)
    print(f"  Found {len(payloads)} result files.")

    print("\nGenerating plots (EPS format only)...")

    print("\n[1/5] Figure 2: Efficiency Analysis")
    plot_figure2(payloads, output_dir)

    print("\n[2/5] Lambda Sensitivity")
    plot_lambda_sensitivity(payloads, output_dir, args.sensitivity_dir)

    print("\n[3/5] Feature Ablation")
    plot_feature_ablation(payloads, output_dir, args.ablation_dir)

    print("\n[4/5] Fairness & Coverage")
    plot_fairness_coverage(payloads, output_dir)

    print("\n[5/5] Scaling Overhead")
    plot_scaling_overhead(output_dir, args.scaling_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
