"""Rebuild the selector-scaling figure for the MAML-Select letter.

The shipped figure had three problems. Its panel titles and axis labels
collided, it plotted the June 10 campaign while the manuscript text quoted the
June 5 one, and it was drawn at a size that had to be rescaled into the column,
which is what blew the type up.

This version is drawn at the printed width so nothing is rescaled, and it reads
the June 10 campaign, which is the newer run and reaches N = 1000.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "csfl_simulator", "Paper Corrections", "scaling",
                    "scaling_results.json")
OUT = os.path.join(ROOT, "csfl_simulator", "Paper Corrections", "MAML_REVISION_2",
                   "images", "fig_scaling_maml_only_boxed.pdf")

MAML = "research.maml_select"
P = 4673

# Blue and orange, the canonical pair that survives every common CVD type.
MEASURED = "#1f5fa8"
ANALYTIC = "#c8601a"
INK = "#1a1a1a"
GRID = "#d4d4d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.4,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    with open(DATA, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = sorted((r for r in d["results"] if r["method"] == MAML),
                  key=lambda r: r["N"])
    n = np.array([r["N"] for r in rows], dtype=float)
    k = np.array([r["K"] for r in rows], dtype=float)
    ms = np.array([r["mean_selection_seconds"] for r in rows]) * 1000.0
    sd = np.array([r["std_selection_seconds"] for r in rows]) * 1000.0
    samples = np.array([r["samples"] for r in rows], dtype=float)
    return n, k, ms, sd, samples


def work(n, k):
    """The operation count of Eq. (8), O(NP + N log K + 2KP)."""
    return n * P + n * np.log2(np.maximum(k, 1.0)) + 2.0 * k * P


def main():
    n, k, ms, sd, samples = load()
    ci = 1.96 * sd / np.sqrt(samples)          # 95 percent interval of the mean
    w = work(n, k) / work(n[0], k[0])

    # Drawn at the printed width so the type is never rescaled.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.5))

    ax1.errorbar(n, ms, yerr=ci, marker="o", markersize=3.2, linewidth=1.1,
                 color=MEASURED, ecolor=MEASURED, elinewidth=0.8, capsize=1.8,
                 zorder=3)
    ax1.set_xscale("log")
    ax1.set_ylim(0, max(ms + ci) * 1.32)
    ax1.set_title("(a) Measured overhead", pad=3)
    ax1.set_ylabel("ms per round", labelpad=2)

    ax2.plot(n, w, marker="s", markersize=3.0, linewidth=1.1, color=ANALYTIC,
             zorder=3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("(b) Analytical work", pad=3)
    ax2.set_ylabel(r"$\times$ vs. $N=20$", labelpad=2)

    for ax in (ax1, ax2):
        ax.set_xlabel("client pool size $N$", labelpad=2)
        ax.grid(True, which="major", color=GRID, linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks([20, 100, 1000])
        ax.set_xticklabels(["20", "100", "1000"])
        ax.tick_params(length=2.2, pad=1.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.28, w_pad=1.1)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=400, bbox_inches="tight",
                pad_inches=0.012)
    plt.close(fig)

    print(f"wrote {OUT}")
    print(f"  N        {[int(x) for x in n]}")
    print(f"  measured {[round(x, 2) for x in ms]} ms")
    print(f"  range    {ms.min():.1f} to {ms.max():.1f} ms")
    print(f"  work     {[round(x, 1) for x in w]} x")


if __name__ == "__main__":
    main()
