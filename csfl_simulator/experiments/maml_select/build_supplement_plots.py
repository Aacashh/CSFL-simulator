"""Build the *enhanced* MAML-Select supplementary-material plots and tables.

This is an additive, exploratory companion to ``build_review_visuals.py`` and
``build_final_pack.py``. It imports their loaders/styling verbatim (no logic is
copied) and produces four new diagnostic figures plus two statistics tables for
the enhanced supplement (``Supplementary_Material_v2.tex``):

Figures (written to ``Paper Corrections/MAML__Letter/supplementary_assets_v2/``):
  * ``supp_resource_trajectories``   - cumulative TFLOPs / energy / carbon vs round.
  * ``supp_fairness_trajectories``   - Jain index and participation coverage vs round.
  * ``supp_reproducibility_seeds``   - per-seed final accuracy (dots + mean).
  * ``supp_pareto_fronts``           - accuracy vs cumulative energy and TFLOPs.

Tables (written to ``artifacts/maml_select/review_pack/tables_supp/``):
  * ``supp_full_benchmark``          - all methods x datasets, mean +/- s.d. on every metric.
  * ``supp_stats_maml_vs_fedavg``    - signed delta, 95% CI, Cohen's dz, paired p.

Every number is computed from the run logs; nothing is imputed. Methods/datasets
that lack complete runs (e.g. CriticalFL on Fashion-MNIST and CIFAR-10, which
crashed) are simply absent and are rendered as blank cells, never filled in.

Run from the repo root:

    python -m csfl_simulator.experiments.maml_select.build_supplement_plots
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover - optional dependency guard
    student_t = None
    ttest_rel = None

from csfl_simulator.experiments.maml_select.build_review_visuals import (
    COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_WINDOWS_RESULTS,
    METHOD_LABELS,
    METHOD_ORDER,
    REPO_ROOT,
    _mean_std,
    configure_style,
    load_main_results,
    method_handles,
    save_figure,
)

PAPER_DIR = REPO_ROOT / "csfl_simulator" / "Paper Corrections" / "MAML__Letter"
DEFAULT_FIG_DIR = PAPER_DIR / "supplementary_assets_v2"
DEFAULT_TABLE_DIR = REPO_ROOT / "artifacts" / "maml_select" / "review_pack" / "tables_supp"

MAML_KEY = "research.maml_select"
FEDAVG_KEY = "baseline.fedavg"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _seed_mean_curve(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and std of ``metric`` over seeds, indexed by round."""
    return (
        group.groupby("round")
        .agg(mean=(metric, "mean"), std=(metric, "std"), n=("seed", "nunique"))
        .sort_index()
    )


def _plot_metric_vs_round(
    ax: plt.Axes, rounds: pd.DataFrame, scenario: str, metric: str, scale: float = 1.0
) -> bool:
    """Plot one metric-vs-round panel for a single dataset. Returns False if empty."""
    subset = rounds[rounds["scenario_name"].astype(str) == scenario]
    if subset.empty:
        ax.set_axis_off()
        return False
    plotted = False
    for method in METHOD_ORDER:
        group = subset[subset["method_key"].astype(str) == method]
        if group.empty:
            continue
        curve = _seed_mean_curve(group, metric)
        if curve.empty or curve["mean"].dropna().empty:
            continue
        is_maml = method == MAML_KEY
        ax.plot(
            curve.index,
            curve["mean"] * scale,
            color=COLORS.get(method, "#333333"),
            linewidth=2.5 if is_maml else 1.4,
            zorder=6 if is_maml else 2,
            label=METHOD_LABELS.get(method, method),
        )
        if is_maml:
            std = curve["std"].fillna(0.0)
            ax.fill_between(
                curve.index,
                (curve["mean"] - std) * scale,
                (curve["mean"] + std) * scale,
                color=COLORS.get(method, "#333333"),
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
        plotted = True
    ax.margins(x=0.01)
    return plotted


# --------------------------------------------------------------------------- #
# Figure 1: cumulative resource trajectories (datasets x {TFLOPs, energy, carbon})
# --------------------------------------------------------------------------- #
def fig_resource_trajectories(rounds: pd.DataFrame, fig_dir: Path) -> None:
    metrics = [
        ("cum_training_tflops", "Cumulative compute (TFLOPs)"),
        ("cum_modelled_energy_wh", "Cumulative energy (Wh)"),
        ("cum_modelled_carbon_g", "Cumulative carbon (g CO$_2$)"),
    ]
    nrows, ncols = len(DATASET_ORDER), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.6, 8.6))
    for r, scenario in enumerate(DATASET_ORDER):
        for c, (metric, col_title) in enumerate(metrics):
            ax = axes[r, c]
            _plot_metric_vs_round(ax, rounds, scenario, metric)
            if r == 0:
                ax.set_title(col_title)
            if r == nrows - 1:
                ax.set_xlabel("Communication round")
            if c == 0:
                ax.set_ylabel(f"{DATASET_LABELS[scenario]}", fontweight="bold")
    fig.legend(
        handles=method_handles(METHOD_ORDER),
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.suptitle(
        "Cumulative Resource Use Over Training (mean over seeds; band = $\\pm$1 s.d. for MAML-Select)",
        y=1.065,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, fig_dir, "supp_resource_trajectories")


# --------------------------------------------------------------------------- #
# Figure 2: fairness / participation trajectories ({Jain, coverage} x datasets)
# --------------------------------------------------------------------------- #
def fig_fairness_trajectories(rounds: pd.DataFrame, fig_dir: Path) -> None:
    rows = [
        ("fairness_jain", "Jain fairness index", 1.0),
        ("participation_coverage_ratio", "Participation coverage (%)", 100.0),
    ]
    fig, axes = plt.subplots(len(rows), len(DATASET_ORDER), figsize=(11.6, 6.2))
    for r, (metric, row_title, scale) in enumerate(rows):
        for c, scenario in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            _plot_metric_vs_round(ax, rounds, scenario, metric, scale=scale)
            if r == 0:
                ax.set_title(DATASET_LABELS[scenario])
            if r == len(rows) - 1:
                ax.set_xlabel("Communication round")
            if c == 0:
                ax.set_ylabel(row_title)
    fig.legend(
        handles=method_handles(METHOD_ORDER),
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(
        "Fairness and Participation Over Training (mean over seeds)",
        y=1.10,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, fig_dir, "supp_fairness_trajectories")


# --------------------------------------------------------------------------- #
# Figure 3: per-seed reproducibility (final accuracy dots + mean)
# --------------------------------------------------------------------------- #
def fig_reproducibility_seeds(runs: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(12.2, 3.7))
    rng = np.random.default_rng(42)  # deterministic horizontal jitter
    for ax, scenario in zip(axes, DATASET_ORDER):
        subset = runs[runs["scenario_name"].astype(str) == scenario]
        present = [m for m in METHOD_ORDER if not subset[subset["method_key"].astype(str) == m].empty]
        for x, method in enumerate(present):
            group = subset[subset["method_key"].astype(str) == method]
            acc = group["final_accuracy"].astype(float).values * 100.0
            jitter = rng.uniform(-0.12, 0.12, size=len(acc))
            ax.scatter(
                np.full(len(acc), x) + jitter,
                acc,
                s=34,
                color=COLORS.get(method, "#333333"),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                alpha=0.95,
            )
            ax.hlines(
                acc.mean(),
                x - 0.28,
                x + 0.28,
                color=COLORS.get(method, "#333333"),
                linewidth=2.4,
                zorder=4,
            )
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in present], rotation=40, ha="right")
        ax.set_title(DATASET_LABELS[scenario])
        if scenario == DATASET_ORDER[0]:
            ax.set_ylabel("Final accuracy (%)")
    fig.suptitle(
        "Per-Seed Final Accuracy (dots = seeds 42/123/2026; bar = mean)",
        y=1.04,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "supp_reproducibility_seeds")


# --------------------------------------------------------------------------- #
# Figure 4: accuracy-vs-cost Pareto fronts (accuracy vs energy and vs TFLOPs)
# --------------------------------------------------------------------------- #
def _agg_runs(runs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    subset = runs[runs["scenario_name"].astype(str) == scenario]
    agg = (
        subset.groupby("method_key", observed=True)
        .agg(
            acc_mean=("final_accuracy", "mean"),
            acc_std=("final_accuracy", "std"),
            tflops_mean=("cum_training_tflops", "mean"),
            tflops_std=("cum_training_tflops", "std"),
            energy_mean=("cum_modelled_energy_wh", "mean"),
            energy_std=("cum_modelled_energy_wh", "std"),
        )
        .reset_index()
    )
    agg["method_key"] = agg["method_key"].astype(str)
    return agg


def fig_pareto_fronts(runs: pd.DataFrame, fig_dir: Path) -> None:
    rows = [
        ("energy_mean", "energy_std", "Cumulative energy (Wh)"),
        ("tflops_mean", "tflops_std", "Cumulative compute (TFLOPs)"),
    ]
    fig, axes = plt.subplots(len(rows), len(DATASET_ORDER), figsize=(12.2, 7.4))
    for r, (xcol, xerr, xlabel) in enumerate(rows):
        for c, scenario in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            agg = _agg_runs(runs, scenario)
            for _, row in agg.iterrows():
                method = row["method_key"]
                is_maml = method == MAML_KEY
                ax.errorbar(
                    row[xcol],
                    row["acc_mean"] * 100.0,
                    xerr=0.0 if not np.isfinite(row[xerr]) else row[xerr],
                    yerr=0.0 if not np.isfinite(row["acc_std"]) else row["acc_std"] * 100.0,
                    fmt="*" if is_maml else "o",
                    markersize=16 if is_maml else 8,
                    color=COLORS.get(method, "#333333"),
                    markeredgecolor="black",
                    markeredgewidth=0.9 if is_maml else 0.4,
                    ecolor=COLORS.get(method, "#333333"),
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=6 if is_maml else 3,
                )
            if r == 0:
                ax.set_title(DATASET_LABELS[scenario])
            if r == len(rows) - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel("Final accuracy (%)")
            # Lower cost is left, higher accuracy is up => "better" points top-left.
            ax.annotate(
                "better",
                xy=(0.03, 0.96),
                xytext=(0.27, 0.80),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=7.5,
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0),
            )
    fig.legend(
        handles=method_handles(METHOD_ORDER),
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "Accuracy vs. Resource Cost (top-left = cheaper and more accurate = better)",
        y=1.075,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, fig_dir, "supp_pareto_fronts")


# --------------------------------------------------------------------------- #
# Table A: full benchmark, mean +/- s.d. on every metric
# --------------------------------------------------------------------------- #
FULL_METRICS = [
    ("final_accuracy", 100.0, "Acc. (\\%)"),
    ("final_f1", 100.0, "F1 (\\%)"),
    ("cum_training_tflops", 1.0, "TFLOPs"),
    ("cum_modelled_energy_wh", 1.0, "Energy (Wh)"),
    ("cum_modelled_carbon_g", 1.0, "Carbon (g)"),
    ("cum_comm_mb", 1.0, "Comm. (MB)"),
    ("fairness_jain", 1.0, "Jain"),
    ("participation_coverage_ratio", 100.0, "Coverage (\\%)"),
]


def build_full_benchmark(runs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in DATASET_ORDER:
        subset = runs[runs["scenario_name"].astype(str) == scenario]
        for method in METHOD_ORDER:
            group = subset[subset["method_key"].astype(str) == method]
            row = {
                "dataset": DATASET_LABELS[scenario],
                "method": METHOD_LABELS.get(method, method),
                "method_key": method,
                "n": int(group["seed"].nunique()) if not group.empty else 0,
                "available": not group.empty,
            }
            for col, scale, _ in FULL_METRICS:
                row[col] = _mean_std(group[col], scale=scale) if not group.empty else "--"
            rows.append(row)
    return pd.DataFrame(rows)


def write_full_benchmark_tex(frame: pd.DataFrame, path: Path) -> None:
    col_spec = "ll" + "r" * (len(FULL_METRICS) + 1)
    headers = ["Dataset", "Method"] + [hdr for _, _, hdr in FULL_METRICS] + ["$n$"]
    lines = [
        "% Auto-generated by build_supplement_plots.py - do not hand-edit.",
        "\\begin{tabular}{@{}" + col_spec + "@{}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    last_dataset = None
    for _, row in frame.iterrows():
        dataset_cell = row["dataset"] if row["dataset"] != last_dataset else ""
        if dataset_cell and last_dataset is not None:
            lines.append("\\midrule")
        last_dataset = row["dataset"]
        cells = [dataset_cell, row["method"]]
        for col, _, _ in FULL_METRICS:
            cells.append(str(row[col]).replace("±", "$\\pm$"))
        cells.append("--" if not row["available"] else str(row["n"]))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Table B: MAML-Select vs FedAvg, signed delta + 95% CI + Cohen's dz + p
# --------------------------------------------------------------------------- #
STAT_METRICS = [
    ("final_accuracy", 100.0, "$\\Delta$Acc. (pp)"),
    ("cum_training_tflops", 1.0, "$\\Delta$TFLOPs"),
    ("cum_modelled_energy_wh", 1.0, "$\\Delta$Energy (Wh)"),
    ("cum_modelled_carbon_g", 1.0, "$\\Delta$Carbon (g)"),
    ("cum_comm_mb", 1.0, "$\\Delta$Comm. (MB)"),
    ("fairness_jain", 1.0, "$\\Delta$Jain"),
]


def build_stats_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in DATASET_ORDER:
        group = runs[runs["scenario_name"].astype(str) == scenario]
        fed = group[group["method_key"].astype(str) == FEDAVG_KEY].set_index("seed")
        maml = group[group["method_key"].astype(str) == MAML_KEY].set_index("seed")
        shared = sorted(set(fed.index) & set(maml.index))
        if len(shared) < 2:
            continue
        for col, scale, label in STAT_METRICS:
            a = maml.loc[shared, col].astype(float).to_numpy() * scale  # MAML-Select
            b = fed.loc[shared, col].astype(float).to_numpy() * scale   # FedAvg
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            n = len(a)
            if n < 2:
                continue
            diff = a - b  # MAML - FedAvg (negative => MAML lower)
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = mean / sd if sd > 0 else float("nan")
            if student_t is not None and sd > 0:
                tcrit = float(student_t.ppf(0.975, n - 1))
                half = tcrit * sd / math.sqrt(n)
            else:
                half = float("nan")
            if ttest_rel is not None:
                _, pval = ttest_rel(a, b)
                pval = float(pval)
            else:
                pval = float("nan")
            rows.append(
                {
                    "dataset": DATASET_LABELS[scenario],
                    "metric": label,
                    "n": n,
                    "delta_mean": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "cohen_dz": dz,
                    "p_value": pval,
                }
            )
    return pd.DataFrame(rows)


def write_stats_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Auto-generated by build_supplement_plots.py - do not hand-edit.",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Dataset & Metric & $\\Delta$ (mean) & 95\\% CI & Cohen's $d_z$ & $p$ \\\\",
        "\\midrule",
    ]
    last_dataset = None
    for _, row in frame.iterrows():
        dataset_cell = row["dataset"] if row["dataset"] != last_dataset else ""
        if dataset_cell and last_dataset is not None:
            lines.append("\\midrule")
        last_dataset = row["dataset"]
        ci = (
            "--"
            if not (np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]))
            else f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        )
        dz = "--" if not np.isfinite(row["cohen_dz"]) else f"{row['cohen_dz']:.2f}"
        p = "--" if not np.isfinite(row["p_value"]) else f"{row['p_value']:.3f}"
        lines.append(
            " & ".join(
                [dataset_cell, row["metric"], f"{row['delta_mean']:.2f}", ci, dz, p]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-results", type=Path, default=DEFAULT_WINDOWS_RESULTS)
    parser.add_argument(
        "--mac-cifar100-results", type=Path, default=REPO_ROOT / "runs" / "maml_select_cifar100"
    )
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    runs, rounds = load_main_results(
        [
            (args.windows_results, "windows"),
            (args.mac_cifar100_results, "mac-cifar100"),
        ]
    )
    if runs.empty:
        raise SystemExit("No main benchmark runs found.")
    runs["method_key"] = runs["method_key"].astype(str)
    runs["scenario_name"] = runs["scenario_name"].astype(str)
    rounds["method_key"] = rounds["method_key"].astype(str)
    rounds["scenario_name"] = rounds["scenario_name"].astype(str)

    print(f"Loaded {len(runs)} complete runs across "
          f"{runs['scenario_name'].nunique()} datasets and "
          f"{runs['method_key'].nunique()} methods.")

    # Figures
    fig_resource_trajectories(rounds, args.fig_dir)
    fig_fairness_trajectories(rounds, args.fig_dir)
    fig_reproducibility_seeds(runs, args.fig_dir)
    fig_pareto_fronts(runs, args.fig_dir)
    print(f"Wrote 4 figures (png/pdf/eps) to {args.fig_dir}")

    # Tables
    full = build_full_benchmark(runs)
    full.to_csv(args.table_dir / "supp_full_benchmark.csv", index=False)
    write_full_benchmark_tex(full, args.table_dir / "supp_full_benchmark.tex")

    stats = build_stats_table(runs)
    stats.to_csv(args.table_dir / "supp_stats_maml_vs_fedavg.csv", index=False)
    write_stats_tex(stats, args.table_dir / "supp_stats_maml_vs_fedavg.tex")
    print(f"Wrote 2 tables (csv/tex) to {args.table_dir}")

    # Console echo for transcription / sanity-checking
    print("\n===== FULL BENCHMARK (mean +/- s.d.) =====")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(full.drop(columns=["method_key", "available"]).to_string(index=False))
    print("\n===== MAML-Select vs FedAvg (delta, 95% CI, dz, p) =====")
    if stats.empty:
        print("(no paired data)")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
