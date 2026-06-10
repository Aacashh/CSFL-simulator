"""Build the second-round ablation figures in the manuscript's boxed-axis style.

Produces, into the overleaf package:
  - fig_lambda_two_dataset_boxed.{pdf,eps,png}  : lambda sweep on Fashion-MNIST and CIFAR-10
  - fig_inner_step_boxed.{pdf,eps,png}          : CIFAR-10 inner-loop step ablation
  - fig_arch_width_boxed.{pdf,eps,png}          : selector-width (MLP capacity) ablation

It consumes the summary CSVs written by summarize_review_hardening / summarize_arch_ablation.
No training is run. Matches the boxed look of build_boxed_letter_figures.py (visible spines,
single-column width, light grid, Okabe-Ito palette).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PKG = REPO_ROOT / "csfl_simulator" / "Paper Corrections" / "overleaf_maml_select_package" / "images"
RH = REPO_ROOT / "artifacts" / "maml_select" / "review_hardening" / "analysis"
ARCH = REPO_ROOT / "artifacts" / "maml_select" / "arch_ablation" / "analysis"

# Okabe-Ito colourblind-safe palette, matching the rest of the letter figures.
C_ACC = "#000000"
C_COMPUTE = "#E69F00"
C_ENERGY = "#009E73"
C_FAIR = "#0072B2"
C_OVH = "#CC3311"
GRID = "#E6E6E6"
SPINE = "#222222"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "bold",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.0,
        "axes.labelweight": "bold",
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def _boxed(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(width=0.7, length=2.8, color=SPINE)


def _save(fig: plt.Figure, stem: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        kw = {"dpi": 400} if ext == "png" else {}
        fig.savefig(out / f"{stem}_boxed.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {stem}_boxed.{{pdf,eps,png}}")


def lambda_two_dataset(out: Path) -> None:
    df = pd.read_csv(RH / "lambda_sensitivity_two_dataset_summary.csv")
    df = df[df["family"] == "lambda_sensitivity"].copy()
    order = ["Fashion-MNIST", "CIFAR-10"]
    datasets = [d for d in order if d in set(df["dataset"])]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(3.5, 4.7), squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        part = df[df["dataset"] == ds].sort_values("lambda")
        x = part["lambda"].to_numpy(float)
        acc = 100.0 * part["accuracy_mean"].to_numpy(float)
        err = 100.0 * part["accuracy_std"].fillna(0.0).to_numpy(float)
        ax.errorbar(x, acc, yerr=err, marker="o", color=C_ACC, lw=1.8, ms=4.2,
                    capsize=2.5, label="Accuracy", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Final accuracy (%)")
        ax.set_title(ds)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        _boxed(ax)
        ax2 = ax.twinx()
        base = part.iloc[0]
        ax2.plot(x, part["tflops_mean"] / base["tflops_mean"], marker="s", color=C_COMPUTE,
                 lw=1.5, ms=3.8, label="Compute")
        ax2.plot(x, part["energy_mean"] / base["energy_mean"], marker="^", color=C_ENERGY,
                 lw=1.5, ms=3.8, label="Energy")
        ax2.plot(x, part["jain_mean"] / base["jain_mean"], marker="D", color=C_FAIR,
                 lw=1.5, ms=3.6, label="Fairness")
        ax2.set_ylabel(r"Relative to $\lambda=0.1$")
        for side in ("top", "right", "bottom", "left"):
            ax2.spines[side].set_visible(True)
            ax2.spines[side].set_color(SPINE)
            ax2.spines[side].set_linewidth(0.8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        leg = ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True,
                        framealpha=0.95, edgecolor="#CCCCCC", ncol=1,
                        columnspacing=1.0, handletextpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    _save(fig, "fig_lambda_two_dataset", out)


def inner_step(out: Path) -> None:
    df = pd.read_csv(RH / "inner_step_ablation_summary.csv").sort_values("inner_steps")
    steps = df["inner_steps"].to_numpy(float).astype(int)
    acc = 100.0 * df["accuracy_mean"].to_numpy(float)
    err = 100.0 * df["accuracy_std"].fillna(0.0).to_numpy(float)
    ovh = df["selection_overhead_ms_mean"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.48, 2.18))
    xs = np.arange(len(steps))
    ax.bar(xs, acc, yerr=err, width=0.6, color="#4E79A7", edgecolor=SPINE,
           linewidth=0.55, capsize=2.8, zorder=3, label="Accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Inner-loop steps")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_ylim(acc.min() - 4.5, acc.max() + 4.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _boxed(ax)
    ax2 = ax.twinx()
    ax2.plot(xs, ovh, marker="o", color=C_OVH, lw=1.8, ms=4.4, zorder=4, label="Overhead")
    ax2.set_ylabel("Selection overhead (ms)")
    for side in ("top", "right", "bottom", "left"):
        ax2.spines[side].set_visible(True); ax2.spines[side].set_color(SPINE); ax2.spines[side].set_linewidth(0.8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True, framealpha=0.95,
                    edgecolor="#CCCCCC", ncol=2, columnspacing=1.0, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=0.35)
    _save(fig, "fig_inner_step", out)


def arch_width(out: Path) -> None:
    csv = ARCH / "arch_ablation_summary.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found yet (run summarize_arch_ablation first).")
        return
    df = pd.read_csv(csv).sort_values("hidden_dim")
    h = df["hidden_dim"].to_numpy(float).astype(int)
    acc = 100.0 * df["accuracy_mean"].to_numpy(float)
    err = 100.0 * df["accuracy_std"].fillna(0.0).to_numpy(float)
    ovh = df["overhead_mean"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.48, 2.18))
    xs = np.arange(len(h))
    ax.errorbar(xs, acc, yerr=err, marker="o", color=C_ACC, lw=1.8, ms=4.6,
                capsize=2.8, zorder=4, label="Accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in h])
    ax.set_xlabel(r"Selector hidden width $h$")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_ylim(acc.min() - 3 * max(err.max(), 0.5), acc.max() + 3 * max(err.max(), 0.5))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _boxed(ax)
    ax2 = ax.twinx()
    ax2.plot(xs, ovh, marker="s", color=C_OVH, lw=1.8, ms=4.0, zorder=3, label="Overhead")
    ax2.set_ylabel("Selection overhead (ms)")
    for side in ("top", "right", "bottom", "left"):
        ax2.spines[side].set_visible(True); ax2.spines[side].set_color(SPINE); ax2.spines[side].set_linewidth(0.8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="lower center", frameon=True, framealpha=0.95,
                    edgecolor="#CCCCCC", ncol=2, columnspacing=1.0, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=0.35)
    _save(fig, "fig_arch_width", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=PKG)
    args = ap.parse_args()
    _style()
    print(f"Writing boxed figures to {args.output_dir}")
    lambda_two_dataset(args.output_dir)
    inner_step(args.output_dir)
    arch_width(args.output_dir)


if __name__ == "__main__":
    main()
