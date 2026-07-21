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


# Compact Table-II source for the combined efficiency/scaling figure.  The
# figure is a visual summary of Table II, so drawing from these values prevents
# missing points when an external run directory is unavailable.
TABLE2_TRADEOFF = [
    # scenario, method_key, final accuracy (%), cumulative training TFLOPs
    ("fashion_main", "baseline.fedavg", 90.22, 140.0),
    ("fashion_main", "system_aware.fedcs", 84.11, 76.0),
    ("fashion_main", "system_aware.oort", 86.70, 149.0),
    ("fashion_main", "system_aware.tifl", 86.73, 147.0),
    ("fashion_main", "ml.fedcor", 89.35, 121.0),
    ("fashion_main", "research.criticalfl", 90.16, 304.0),
    ("fashion_main", "research.fedgcs", 90.65, 136.0),
    ("fashion_main", "research.maml_select", 90.11, 122.0),
    ("cifar10_main", "baseline.fedavg", 79.43, 8341.0),
    ("cifar10_main", "system_aware.fedcs", 50.45, 4081.0),
    ("cifar10_main", "system_aware.oort", 60.19, 8303.0),
    ("cifar10_main", "system_aware.tifl", 61.93, 8200.0),
    ("cifar10_main", "ml.fedcor", 67.20, 7748.0),
    ("cifar10_main", "research.criticalfl", 77.34, 14239.0),
    ("cifar10_main", "research.fedgcs", 77.88, 7656.0),
    ("cifar10_main", "research.maml_select", 75.63, 6549.0),
    ("cifar100_main", "baseline.fedavg", 59.66, 6331.0),
    ("cifar100_main", "system_aware.fedcs", 27.52, 5334.0),
    ("cifar100_main", "system_aware.oort", 29.41, 5955.0),
    ("cifar100_main", "system_aware.tifl", 32.12, 5986.0),
    ("cifar100_main", "ml.fedcor", 34.18, 6428.0),
    ("cifar100_main", "research.criticalfl", 48.84, 9469.0),
    ("cifar100_main", "research.fedgcs", 58.66, 6140.0),
    ("cifar100_main", "research.maml_select", 58.15, 6224.0),
]


def table2_tradeoff_frame() -> pd.DataFrame:
    """Return Table-II values in the run-summary schema expected by _draw_tradeoff."""
    return pd.DataFrame(
        {
            "scenario_name": scenario,
            "method_key": method,
            "final_accuracy": acc / 100.0,
            "cum_training_tflops": tflops,
        }
        for scenario, method, acc, tflops in TABLE2_TRADEOFF
    )


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


def _draw_scaling(ax, ax2, scaling_csv: Path, *, label_fs: float = 9.0, tick_fs: float = 8.0,
                  title_fs: float = 9.0, letters=("a", "b"), stacked: bool = False) -> None:
    """Draw the (measured-time, analytical-work) scaling pair onto two given axes."""
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
    ax.fill_between(n, mean_ms - ci95, mean_ms + ci95, color="#CFE8FF", linewidth=0.0, zorder=1)
    ax.plot(n, mean_ms, "o-", color=line_color, markersize=4.4, linewidth=1.8, zorder=3)
    ax.errorbar(n, mean_ms, yerr=ci95, fmt="none", ecolor=line_color, elinewidth=0.9, capsize=2.3, zorder=4)
    ax2.plot(n, work_rel, "s--", color=theory_color, markersize=4.0, linewidth=1.8, zorder=3)
    for axis in (ax, ax2):
        boxed_axes(axis)
        axis.set_xscale("log")
        axis.set_xticks(n)
        axis.set_xticklabels([str(int(v)) if int(v) in (20, 100, 1000) else "" for v in n], fontsize=tick_fs)
        axis.minorticks_off()
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        axis.tick_params(labelsize=tick_fs)
    ax.set_ylabel("Computational\noverhead (ms/round)", fontsize=label_fs, fontweight="normal")
    ax2.set_ylabel(r"Work ($\times$ vs. $N{=}20$)", fontsize=label_fs, fontweight="normal")
    ax.set_ylim(0, float(np.nanmax(mean_ms + ci95)) + 5)
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, float(work_rel.max()) * 1.5)
    ax2.set_yticks([1, 2, 5, 10, 20, 50])
    ax2.set_yticklabels(["1", "2", "5", "10", "20", "50"], fontsize=tick_fs)
    ax.set_title(f"({letters[0]}) Measured time", fontsize=title_fs, fontweight="normal")
    ax2.set_title(f"({letters[1]}) Analytical work", fontsize=title_fs, fontweight="normal")
    ax.text(0.96, 0.93, r"flat $\approx$20--26 ms", transform=ax.transAxes, ha="right", va="top",
            fontsize=tick_fs, color=line_color)
    if stacked:
        # Both panels keep their own x-tick numbers; only the bottom one is labelled.
        ax2.set_xlabel("Client pool size $N$ (log)", fontsize=label_fs, fontweight="normal")
    else:
        ax.set_xlabel("Client pool size $N$ (log)", fontsize=label_fs, fontweight="normal")
        ax2.set_xlabel("Client pool size $N$ (log)", fontsize=label_fs, fontweight="normal")


def build_scaling_boxed(scaling_csv: Path, output_dir: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.48, 1.95), gridspec_kw={"width_ratios": [1.1, 1.0]})
    _draw_scaling(ax, ax2, scaling_csv, label_fs=9.0, tick_fs=8.0, title_fs=9.5, letters=("a", "b"))
    fig.tight_layout(pad=0.35, w_pad=1.15)
    save_boxed(fig, output_dir, "fig_scaling_maml_only")


def build_combined_boxed(runs: pd.DataFrame, scaling_csv: Path, output_dir: Path) -> None:
    """Merged single-column figure.

    Panel (a) uses the full left-side height so the efficiency--accuracy plot
    does not look compressed. Its method and dataset legends are placed below
    the axis, outside the data region, to avoid covering points. Panels (b) and
    (c) retain the compact stacked selector-scaling view on the right.
    """
    fig = plt.figure(figsize=(3.5, 3.25))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.34, 1.0],
        wspace=0.52,
        left=0.105,
        right=0.975,
        top=0.945,
        bottom=0.365,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    gr = outer[0, 1].subgridspec(2, 1, hspace=0.66)
    ax_b = fig.add_subplot(gr[0])
    ax_c = fig.add_subplot(gr[1])

    mh, dh = bfp._draw_tradeoff(
        ax_a,
        table2_tradeoff_frame(),
        label_fs=6.1,
        tick_fs=5.5,
        legend_fs=5.0,
        draw_legends=False,
    )
    ax_a.yaxis.labelpad = 1.2
    ax_a.xaxis.labelpad = 1.0
    ax_a.set_title("(a) Efficiency--accuracy trade-off", fontsize=6.9, fontweight="normal", pad=2.0)

    _draw_scaling(ax_b, ax_c, scaling_csv, label_fs=6.2, tick_fs=5.6, title_fs=7.0,
                  letters=("b", "c"), stacked=True)

    # Legends are inside panel (a), placed in its bottom-right open region.
    # The method legend uses colour; the dataset legend uses marker shape.
    method_leg = ax_a.legend(
        handles=mh,
        loc="lower right",
        bbox_to_anchor=(0.998, 0.035),
        bbox_transform=ax_a.transAxes,
        ncol=2,
        fontsize=3.75,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.97,
        handletextpad=0.52,
        columnspacing=1.25,
        labelspacing=0.42,
        handlelength=1.15,
        borderpad=0.40,
        markerscale=0.72,
    )
    method_leg.get_frame().set_linewidth(0.4)
    ax_a.add_artist(method_leg)
    dataset_leg = ax_a.legend(
        handles=dh,
        loc="lower right",
        bbox_to_anchor=(0.998, 0.485),
        bbox_transform=ax_a.transAxes,
        ncol=1,
        fontsize=3.9,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.97,
        handletextpad=0.55,
        columnspacing=0.48,
        labelspacing=0.42,
        handlelength=1.05,
        borderpad=0.38,
        markerscale=0.70,
    )
    dataset_leg.get_frame().set_linewidth(0.4)
    save_boxed(fig, output_dir, "fig_efficiency_scaling")


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
    build_combined_boxed(runs, args.scaling_csv, args.output_dir)
    build_lambda_boxed(lambda_summary, args.output_dir)
    build_feature_ablation_boxed(ablation_summary, args.output_dir)
    print(f"Wrote boxed-axis figures to {args.output_dir}")


if __name__ == "__main__":
    main()
