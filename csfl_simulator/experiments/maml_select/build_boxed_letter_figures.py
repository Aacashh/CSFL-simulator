"""Build manuscript figures with boxed axes, without overwriting originals.

The professor-requested style is an axis box around each individual graph
(visible top/right spines), not a LaTeX frame around the whole figure file.
This script writes *_boxed.{pdf,eps,png} files into the letter folder and leaves
the original figure files untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csfl_simulator.experiments.maml_select import build_final_pack as bfp
from csfl_simulator.experiments.maml_select.build_maml_scaling_figure import (
    DEFAULT_SCALING_CSV,
    POLICY_PARAMS,
    load_maml_scaling,
    selector_work,
)


DEFAULT_OUTPUT_DIR = bfp.REPO_ROOT / "csfl_simulator" / "Paper Corrections" / "MAML__Letter"
OURS = "research.maml_select"


def boxed_axes(ax) -> None:
    """Give every individual subplot a closed axis box."""
    ax.grid(True, **bfp.GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.9, labelsize=10, top=False, right=False)


def save_boxed(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {"format": "pdf"}),
        ("eps", {"format": "eps"}),
    ):
        fig.savefig(output_dir / f"{stem}_boxed.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def patch_final_pack_style() -> None:
    """Reuse build_final_pack plots, but force boxed axes and boxed filenames."""
    bfp._clean_axes = boxed_axes
    bfp.save_figure = save_boxed


def build_lambda_boxed(lambda_frame: pd.DataFrame, output_dir: Path) -> None:
    lam = lambda_frame.copy()
    lam["lambda"] = lam["variant"].str.extract(r"lambda=([0-9.]+)").astype(float)
    lam = lam.sort_values("lambda")
    lams = lam["lambda"].to_numpy()

    def rel(col: str) -> np.ndarray:
        values = lam[col].astype(float).to_numpy()
        return values / values[0]

    series = [
        ("Accuracy", rel("final_accuracy_mean"), "#111111", "o"),
        ("Compute", rel("cum_training_tflops_mean"), "#D55E00", "s"),
        ("Energy", rel("cum_modelled_energy_wh_mean"), "#009E73", "^"),
        ("Fairness", rel("fairness_jain_mean"), "#0072B2", "D"),
    ]
    fig, ax = plt.subplots(figsize=(3.48, 2.28))
    for label, values, color, marker in series:
        ax.plot(lams, values, color=color, marker=marker, linewidth=1.8, markersize=4.2, label=label)
    boxed_axes(ax)
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels(["%g" % v for v in lams])
    ax.minorticks_off()
    ax.set_ylim(0.52, 1.04)
    ax.set_xlabel(r"Cost trade-off $\lambda$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Normalized value ($\lambda=0.1$)", fontsize=11, fontweight="bold")
    ax.set_title(r"$\lambda$ Sensitivity", fontsize=12, fontweight="bold")
    leg = ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.94,
        handlelength=1.2,
        columnspacing=0.8,
        handletextpad=0.35,
        borderpad=0.35,
    )
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=0.35)
    save_boxed(fig, output_dir, "fig_lambda_sensitivity_multimetric")


def build_feature_ablation_boxed(ablation_frame: pd.DataFrame, output_dir: Path) -> None:
    order = [
        "without loss",
        "without gradient norm",
        "without latency",
        "without battery",
        "without frequency",
        "without staleness",
    ]
    abl = ablation_frame.set_index("variant")
    base = 100.0 * float(abl.loc["all features", "final_accuracy_mean"])
    labels, deltas, errs = [], [], []
    for variant in order:
        if variant not in abl.index:
            continue
        labels.append(variant.replace("without ", "w/o "))
        deltas.append(100.0 * float(abl.loc[variant, "final_accuracy_mean"]) - base)
        err = abl.loc[variant, "final_accuracy_std"]
        errs.append(100.0 * float(0.0 if pd.isna(err) else err))

    x = np.arange(len(labels))
    colors = [bfp.GAIN_COLOR if d >= 0 else bfp.DROP_COLOR for d in deltas]
    fig, ax = plt.subplots(figsize=(3.48, 1.98))
    ax.bar(x, deltas, yerr=errs, capsize=2.5, color=colors, edgecolor="#333333", linewidth=0.55, zorder=3)
    ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=2)
    boxed_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"$\Delta$ acc. vs full (pp)", fontsize=11, fontweight="bold")
    ax.set_title("State-feature ablation", fontsize=12, fontweight="bold")
    ax.set_ylim(-1.2, 0.8)
    fig.tight_layout(pad=0.35)
    save_boxed(fig, output_dir, "fig_feature_ablation_trimmed")


def build_scaling_boxed(scaling_csv: Path, output_dir: Path) -> None:
    frame = load_maml_scaling(scaling_csv)
    n = frame["N"].astype(float).to_numpy()
    k = frame["K"].astype(float).to_numpy()
    mean_ms = 1000.0 * frame["mean_selection_seconds"].astype(float).to_numpy()
    std_ms = 1000.0 * frame["std_selection_seconds"].astype(float).fillna(0.0).to_numpy()
    samples = frame["samples"].astype(float).replace(0, np.nan).to_numpy()
    ci95 = 1.96 * std_ms / np.sqrt(samples)
    work_rel = selector_work(n, k) / selector_work(n, k)[0]

    line_color = "#0072B2"
    theory_color = "#D55E00"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.48, 1.85), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax.fill_between(n, mean_ms - ci95, mean_ms + ci95, color="#CFE8FF", linewidth=0.0, zorder=1)
    ax.plot(n, mean_ms, "o-", color=line_color, markersize=4.8, linewidth=2.0, zorder=3)
    ax.errorbar(n, mean_ms, yerr=ci95, fmt="none", ecolor=line_color, elinewidth=0.9, capsize=2.5, zorder=4)
    ax2.plot(n, work_rel, "s--", color=theory_color, markersize=4.3, linewidth=2.0, zorder=3)
    for axis in (ax, ax2):
        boxed_axes(axis)
        axis.set_xticks(n)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_xlabel("Client pool size $N$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Client pool size $N$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overhead (ms/round)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Work ($\times$ vs. $N=20$)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(18, float(np.nanmax(mean_ms + ci95)) + 1.5))
    ax2.set_ylim(0.6, max(5.5, float(work_rel.max()) + 0.3))
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("(a) Measured time", fontsize=11.5, fontweight="bold")
    ax2.set_title("(b) Analytical work", fontsize=11.5, fontweight="bold")
    ax.text(0.95, 0.10, "< 13 ms", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.0, color=line_color, fontweight="bold")
    ax2.annotate(
        r"$C(N,K)=NP+N\log K+2KP$" + "\n" + rf"$P={POLICY_PARAMS:,},\;K=N/10$",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout(pad=0.35, w_pad=1.15)
    save_boxed(fig, output_dir, "fig_scaling_maml_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-results", type=Path, default=bfp.DEFAULT_WINDOWS_RESULTS)
    parser.add_argument("--criticalfl-results", type=Path, default=Path.home() / "Desktop" / "main_benchmarks_critical")
    parser.add_argument("--mac-cifar100-results", type=Path, default=bfp.REPO_ROOT / "runs" / "maml_select_cifar100")
    parser.add_argument("--mac-review-results", type=Path, default=bfp.REPO_ROOT / "runs" / "maml_select")
    parser.add_argument("--scaling-csv", type=Path, default=DEFAULT_SCALING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bfp.configure_style()
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    patch_final_pack_style()

    runs, _rounds = bfp.load_main_results(
        [
            (args.windows_results, "windows"),
            (args.criticalfl_results, "criticalfl-windows"),
            (args.mac_cifar100_results, "mac-cifar100"),
        ]
    )
    runs["method_key"] = runs["method_key"].astype(str)
    runs["scenario_name"] = runs["scenario_name"].astype(str)
    runs, _ = bfp.append_extra_runs(runs, pd.DataFrame())

    c100_rounds = bfp.load_cifar100_metric_rounds(args.mac_cifar100_results)
    lambda_summary = bfp.load_variant_summary(args.mac_review_results, "lambda_sensitivity")
    ablation_summary = bfp.load_variant_summary(args.mac_review_results, "feature_ablation")

    bfp.fig_convergence(c100_rounds, args.output_dir)
    bfp.fig_tradeoff(runs, args.output_dir)
    build_scaling_boxed(args.scaling_csv, args.output_dir)
    build_lambda_boxed(lambda_summary, args.output_dir)
    build_feature_ablation_boxed(ablation_summary, args.output_dir)
    print(f"Wrote boxed-axis figures to {args.output_dir}")


if __name__ == "__main__":
    main()
