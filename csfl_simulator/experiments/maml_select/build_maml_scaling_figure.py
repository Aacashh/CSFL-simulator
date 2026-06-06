"""Build the MAML-only selector-scaling figure for the revised letter.

The figure intentionally focuses on MAML-Select rather than comparing all
selectors: the main-paper claim is the selector complexity of the proposed
method.  Bars/points show measured wall-clock selection overhead, while the
dashed curve shows the normalized operation count from the manuscript
complexity expression.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
REPO_ROOT = HERE.parents[3]
DEFAULT_SCALING_CSV = REPO_ROOT / "runs" / "maml_select" / "scaling" / "scaling_results.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "Paper Corrections" / "MAML__Letter"

MAML_KEY = "research.maml_select"
POLICY_PARAMS = 4673


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selector_work(n_clients: np.ndarray, k_clients: np.ndarray) -> np.ndarray:
    """Operation-count proxy from O(NP + N log K + 2KP)."""
    return (
        n_clients * POLICY_PARAMS
        + n_clients * np.log2(np.maximum(k_clients, 1))
        + 2.0 * k_clients * POLICY_PARAMS
    )


def load_maml_scaling(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    maml = frame[frame["method"].astype(str) == MAML_KEY].copy()
    if maml.empty:
        raise ValueError(f"No rows for {MAML_KEY} in {path}")
    return maml.sort_values("N")


def build_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    configure_style()

    n = frame["N"].astype(float).to_numpy()
    k = frame["K"].astype(float).to_numpy()
    mean_ms = 1000.0 * frame["mean_selection_seconds"].astype(float).to_numpy()
    std_ms = 1000.0 * frame["std_selection_seconds"].astype(float).fillna(0.0).to_numpy()
    samples = frame["samples"].astype(float).replace(0, np.nan).to_numpy()
    ci95 = 1.96 * std_ms / np.sqrt(samples)

    work = selector_work(n, k)
    work_rel = work / work[0]

    line_color = "#0072B2"
    theory_color = "#D55E00"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.48, 1.85), gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax.fill_between(n, mean_ms - ci95, mean_ms + ci95, color="#CFE8FF", linewidth=0.0, zorder=1)
    ax.plot(
        n,
        mean_ms,
        "o-",
        color=line_color,
        markersize=4.8,
        linewidth=2.0,
        zorder=3,
    )
    ax.errorbar(n, mean_ms, yerr=ci95, fmt="none", ecolor=line_color, elinewidth=0.9, capsize=2.5, zorder=4)
    ax2.plot(
        n,
        work_rel,
        "s--",
        color=theory_color,
        markersize=4.3,
        linewidth=2.0,
        zorder=3,
    )

    ax.set_xlabel("Client pool size $N$")
    ax2.set_xlabel("Client pool size $N$")
    ax.set_ylabel("Overhead (ms/round)")
    ax2.set_ylabel(r"Work ($\times$ vs. $N=20$)")

    for a in (ax, ax2):
        a.set_xticks(n)
        a.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        a.set_axisbelow(True)
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax.set_ylim(0, max(18, float(np.nanmax(mean_ms + ci95)) + 1.5))
    ax2.set_ylim(0.6, max(5.5, float(work_rel.max()) + 0.3))
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("(a) Measured selector time", fontsize=8.6)
    ax2.set_title("(b) Analytical selector work", fontsize=8.6)

    ax.text(
        0.95,
        0.10,
        "< 13 ms",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=line_color,
        fontweight="bold",
    )
    ax2.annotate(
        r"$C(N,K)=NP+N\log K+2KP$" + "\n" + r"$P=4{,}673,\;K=N/10$",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=6.7,
        color="#333333",
    )

    fig.tight_layout(pad=0.35, w_pad=0.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(output_dir / f"fig_scaling_maml_only.{ext}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-csv", type=Path, default=DEFAULT_SCALING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_maml_scaling(args.scaling_csv)
    build_figure(frame, args.output_dir)
    print(f"Wrote fig_scaling_maml_only.* to {args.output_dir}")


if __name__ == "__main__":
    main()
