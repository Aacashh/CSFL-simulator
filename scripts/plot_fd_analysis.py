#!/usr/bin/env python3
"""Generate analysis figures for FD experiment analysis document.

Creates publication-quality plots summarizing key findings across all 10 experiments.
Output: docs/figures/analysis/

Usage:
    python scripts/plot_fd_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Visual config (consistent with plot_fd_experiments.py)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline.fedavg":          "#1f77b4",
    "system_aware.fedcs":       "#d62728",
    "system_aware.tifl":        "#2ca02c",
    "ml.fedcor":                "#9467bd",
    "ml.maml_select":           "#8c564b",
    "ml.apex_v2":               "#e377c2",
    "system_aware.oort":        "#17becf",
    "heuristic.label_coverage": "#ff7f0e",
}

SHORT = {
    "baseline.fedavg":          "FedAvg",
    "system_aware.fedcs":       "FedCS",
    "system_aware.tifl":        "TiFL",
    "ml.fedcor":                "FedCor",
    "ml.maml_select":           "MAML",
    "ml.apex_v2":               "APEX v2",
    "system_aware.oort":        "Oort",
    "heuristic.label_coverage": "LabelCov",
}

METHOD_ORDER = [
    "baseline.fedavg", "system_aware.fedcs", "system_aware.tifl",
    "ml.fedcor", "ml.maml_select", "ml.apex_v2",
    "system_aware.oort", "heuristic.label_coverage",
]

MARKERS = {
    "baseline.fedavg": "o", "system_aware.fedcs": "s",
    "system_aware.tifl": "^", "ml.fedcor": "D",
    "ml.maml_select": "v", "ml.apex_v2": "*",
    "system_aware.oort": "P", "heuristic.label_coverage": "X",
}


def apply_style(plt):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
    })


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
RUNS_DIR = Path("artifacts/runs")


def resolve(name: str) -> Path:
    """Resolve run name to directory (latest prefix match)."""
    if (RUNS_DIR / name).is_dir():
        return RUNS_DIR / name
    matches = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(name)],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No run matching '{name}'")


def load(name: str) -> dict:
    d = resolve(name)
    cp = d / "compare_results.json"
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    rp = d / "results.json"
    if rp.exists():
        with open(rp) as f:
            return json.load(f)
    raise FileNotFoundError(f"No results in {d}")


def final_metric(data: dict, method: str, metric: str, default=0.0):
    """Extract last-round value of a metric for a method."""
    res = data["results"].get(method, {})
    metrics_list = res.get("metrics", [])
    if not metrics_list:
        return default
    last = metrics_list[-1]
    return float(last.get(metric, default) or default)


def series(data: dict, method: str, metric: str):
    """Extract (rounds, values) for a metric."""
    res = data["results"].get(method, {})
    metrics_list = res.get("metrics", [])
    xs, ys = [], []
    for row in metrics_list:
        r = row.get("round", -1)
        if isinstance(r, (int, float)) and r >= 0:
            xs.append(int(r))
            ys.append(float(row.get(metric, 0.0) or 0.0))
    return np.array(xs), np.array(ys)


def smooth(ys, w=10):
    if len(ys) < w:
        return ys
    kernel = np.ones(w) / w
    s = np.convolve(ys, kernel, mode="valid")
    return np.concatenate([ys[:len(ys) - len(s)], s])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_ranking_shift_bars(plt, out: Path):
    """Side-by-side FL vs FD accuracy bars showing the ranking inversion."""
    fl = load("exp01_fl_baseline")
    fd = load("exp02_fd_main")

    methods = METHOD_ORDER
    fl_acc = [final_metric(fl, m, "accuracy") for m in methods]
    fd_acc = [final_metric(fd, m, "accuracy") for m in methods]
    names = [SHORT[m] for m in methods]

    # Sort by FL accuracy descending
    fl_order = np.argsort(fl_acc)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.patch.set_facecolor("white")

    x = np.arange(len(methods))
    w = 0.6

    # FL panel
    ax1.set_facecolor("white")
    bars1 = ax1.barh(x, [fl_acc[i] for i in fl_order], w,
                     color=[COLORS[methods[i]] for i in fl_order],
                     edgecolor="white", linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels([names[i] for i in fl_order])
    ax1.set_xlabel("Final Accuracy")
    ax1.set_title("(a) FL Ranking (CIFAR-10)")
    ax1.invert_yaxis()
    for i, idx in enumerate(fl_order):
        ax1.text(fl_acc[idx] + 0.003, i, f"{fl_acc[idx]:.3f}", va="center", fontsize=7)

    # FD panel
    fd_order = np.argsort(fd_acc)[::-1]
    ax2.set_facecolor("white")
    bars2 = ax2.barh(x, [fd_acc[i] for i in fd_order], w,
                     color=[COLORS[methods[i]] for i in fd_order],
                     edgecolor="white", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels([names[i] for i in fd_order])
    ax2.set_xlabel("Final Accuracy")
    ax2.set_title("(b) FD Ranking (CIFAR-10 + STL-10)")
    ax2.invert_yaxis()
    for i, idx in enumerate(fd_order):
        ax2.text(fd_acc[idx] + 0.002, i, f"{fd_acc[idx]:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "fig01_ranking_shift.png")
    fig.savefig(out / "fig01_ranking_shift.eps")
    plt.close(fig)
    print(f"  -> fig01_ranking_shift")


def fig02_fd_convergence(plt, out: Path):
    """FD accuracy convergence curves for all 8 methods (Exp 2)."""
    fd = load("exp02_fd_main")
    methods = METHOD_ORDER

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys = series(fd, m, "accuracy")
        if len(ys) == 0:
            continue
        ys_s = smooth(ys, 10)
        mark_every = max(1, len(xs) // 8)
        ax.plot(xs[:len(ys_s)], ys_s, color=COLORS[m], marker=MARKERS[m],
                markevery=mark_every, markersize=4, markerfacecolor="white",
                markeredgewidth=0.8, markeredgecolor=COLORS[m],
                linewidth=1.2, label=SHORT[m])

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Client Accuracy")
    ax.set_title("FD Accuracy Convergence (CIFAR-10)")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5)
    plt.tight_layout()
    fig.savefig(out / "fig02_fd_convergence.png")
    fig.savefig(out / "fig02_fd_convergence.eps")
    plt.close(fig)
    print(f"  -> fig02_fd_convergence")


def fig03_noise_degradation(plt, out: Path):
    """Accuracy across noise levels + degradation bars."""
    errfree = load("exp04_errfree")
    dl0 = load("exp04_dl0db")
    dl20 = load("exp04_dl-20db")

    methods = METHOD_ORDER
    ef_acc = [final_metric(errfree, m, "accuracy") for m in methods]
    d0_acc = [final_metric(dl0, m, "accuracy") for m in methods]
    d20_acc = [final_metric(dl20, m, "accuracy") for m in methods]
    names = [SHORT[m] for m in methods]

    degradation = [(ef - d20) / ef * 100 for ef, d20 in zip(ef_acc, d20_acc)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.patch.set_facecolor("white")

    x = np.arange(len(methods))
    w = 0.25

    # Grouped bars: accuracy at each noise level
    ax1.set_facecolor("white")
    ax1.bar(x - w, ef_acc, w, label="Error-free", color="#2ca02c", alpha=0.8)
    ax1.bar(x,     d0_acc, w, label="DL 0 dB",    color="#ff7f0e", alpha=0.8)
    ax1.bar(x + w, d20_acc, w, label="DL -20 dB", color="#d62728", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Final Accuracy")
    ax1.set_title("(a) Accuracy by Noise Level")
    ax1.legend(fontsize=6.5)

    # Degradation bar
    order = np.argsort(degradation)
    ax2.set_facecolor("white")
    ax2.barh(np.arange(len(methods)),
             [degradation[i] for i in order],
             color=[COLORS[methods[i]] for i in order],
             edgecolor="white", linewidth=0.5)
    ax2.set_yticks(np.arange(len(methods)))
    ax2.set_yticklabels([names[i] for i in order])
    ax2.set_xlabel("Accuracy Degradation (%)")
    ax2.set_title("(b) Error-Free to -20 dB Degradation")
    for i, idx in enumerate(order):
        ax2.text(degradation[idx] + 0.5, i, f"{degradation[idx]:.1f}%",
                 va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "fig03_noise_degradation.png")
    fig.savefig(out / "fig03_noise_degradation.eps")
    plt.close(fig)
    print(f"  -> fig03_noise_degradation")


def fig04_k_sweep(plt, out: Path):
    """Final accuracy vs K/N ratio (one line per method)."""
    ks = [5, 10, 15, 30]
    runs = {5: load("exp05_k5"), 10: load("exp05_k10"),
            15: load("exp05_k15"), 30: load("exp05_k30")}

    methods = METHOD_ORDER
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        accs = [final_metric(runs[k], m, "accuracy") for k in ks]
        ax.plot(ks, accs, color=COLORS[m], marker=MARKERS[m],
                markersize=6, markerfacecolor="white",
                markeredgewidth=1.0, markeredgecolor=COLORS[m],
                linewidth=1.4, label=SHORT[m])

    # Reference: K=30 FedAvg baseline
    k30_fedavg = final_metric(runs[30], "baseline.fedavg", "accuracy")
    ax.axhline(y=k30_fedavg, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.7, label=f"FedAvg K=30 ({k30_fedavg:.3f})")

    ax.set_xlabel("Clients per Round (K)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy vs Participation Rate")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K={k}\n({k/30:.0%})" for k in ks], fontsize=7)
    ax.legend(loc="upper right", fontsize=5.5, ncol=2)
    plt.tight_layout()
    fig.savefig(out / "fig04_k_sweep.png")
    fig.savefig(out / "fig04_k_sweep.eps")
    plt.close(fig)
    print(f"  -> fig04_k_sweep")


def fig05_alpha_sweep(plt, out: Path):
    """Accuracy vs Dirichlet alpha."""
    alphas = [0.1, 0.5, 1.0, 10.0]
    runs = {0.1: load("exp06_a01"), 0.5: load("exp06_a05"),
            1.0: load("exp06_a1"), 10.0: load("exp06_a10")}

    methods = METHOD_ORDER
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        accs = [final_metric(runs[a], m, "accuracy") for a in alphas]
        ax.plot(alphas, accs, color=COLORS[m], marker=MARKERS[m],
                markersize=6, markerfacecolor="white",
                markeredgewidth=1.0, markeredgecolor=COLORS[m],
                linewidth=1.4, label=SHORT[m])

    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy vs Non-IID Level")
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend(loc="upper left", fontsize=5.5, ncol=2)
    plt.tight_layout()
    fig.savefig(out / "fig05_alpha_sweep.png")
    fig.savefig(out / "fig05_alpha_sweep.eps")
    plt.close(fig)
    print(f"  -> fig05_alpha_sweep")


def fig06_grouping_benefit(plt, out: Path):
    """Grouping benefit (delta accuracy) per method."""
    nogroup = load("exp07_nogroup")
    group = load("exp07_group")

    methods = METHOD_ORDER
    names = [SHORT[m] for m in methods]
    deltas = [final_metric(group, m, "accuracy") - final_metric(nogroup, m, "accuracy")
              for m in methods]

    order = np.argsort(deltas)[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors_bar = ["#2ca02c" if d >= 0 else "#d62728" for d in [deltas[i] for i in order]]
    ax.barh(np.arange(len(methods)),
            [deltas[i] for i in order],
            color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Change with FedTSKD-G")
    ax.set_title("Grouping Benefit per Method")
    for i, idx in enumerate(order):
        sign = "+" if deltas[idx] >= 0 else ""
        ax.text(deltas[idx] + (0.001 if deltas[idx] >= 0 else -0.001), i,
                f"{sign}{deltas[idx]:.3f}", va="center", fontsize=7,
                ha="left" if deltas[idx] >= 0 else "right")

    plt.tight_layout()
    fig.savefig(out / "fig06_grouping_benefit.png")
    fig.savefig(out / "fig06_grouping_benefit.eps")
    plt.close(fig)
    print(f"  -> fig06_grouping_benefit")


def fig07_hetero_degradation(plt, out: Path):
    """Model heterogeneity degradation per method."""
    homo = load("exp08_homo")
    hetero = load("exp08_hetero")

    methods = METHOD_ORDER
    names = [SHORT[m] for m in methods]
    homo_acc = [final_metric(homo, m, "accuracy") for m in methods]
    hetero_acc = [final_metric(hetero, m, "accuracy") for m in methods]
    degradation = [(ho - he) / ho * 100 for ho, he in zip(homo_acc, hetero_acc)]

    order = np.argsort(degradation)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(np.arange(len(methods)),
            [degradation[i] for i in order],
            color=[COLORS[methods[i]] for i in order],
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Accuracy Degradation (%)")
    ax.set_title("Heterogeneity Impact (Homo to Hetero)")
    for i, idx in enumerate(order):
        ax.text(degradation[idx] + 0.2, i, f"{degradation[idx]:.1f}%",
                va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "fig07_hetero_degradation.png")
    fig.savefig(out / "fig07_hetero_degradation.eps")
    plt.close(fig)
    print(f"  -> fig07_hetero_degradation")


def fig08_fairness_accuracy(plt, out: Path):
    """Fairness (Gini) vs Accuracy scatter with method labels."""
    fd = load("exp02_fd_main")
    methods = METHOD_ORDER

    accs = [final_metric(fd, m, "accuracy") for m in methods]
    ginis = [final_metric(fd, m, "fairness_gini") for m in methods]
    names = [SHORT[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, m in enumerate(methods):
        ax.scatter(ginis[i], accs[i], color=COLORS[m], marker=MARKERS[m],
                   s=80, zorder=5, edgecolors="black", linewidth=0.5)
        # Label offset to avoid overlap
        offset_x, offset_y = 0.02, 0.003
        if names[i] in ("Oort", "LabelCov"):
            offset_y = -0.006
        if names[i] == "FedCS":
            offset_y = 0.006
        ax.annotate(names[i], (ginis[i], accs[i]),
                    xytext=(ginis[i] + offset_x, accs[i] + offset_y),
                    fontsize=7, ha="left")

    # Pareto front
    pareto_methods = ["baseline.fedavg", "ml.apex_v2", "ml.maml_select"]
    pareto_gini = sorted([(final_metric(fd, m, "fairness_gini"),
                           final_metric(fd, m, "accuracy")) for m in pareto_methods])
    pg, pa = zip(*pareto_gini)
    ax.plot(pg, pa, "--", color="gray", linewidth=0.8, alpha=0.6, label="Pareto front")

    ax.set_xlabel("Fairness Gini (lower = fairer)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy-Fairness Tradeoff (FD)")
    ax.legend(fontsize=6.5)
    plt.tight_layout()
    fig.savefig(out / "fig08_fairness_accuracy.png")
    fig.savefig(out / "fig08_fairness_accuracy.eps")
    plt.close(fig)
    print(f"  -> fig08_fairness_accuracy")


def fig09_comm_comparison(plt, out: Path):
    """FL vs FD cumulative communication bar."""
    fl = load("exp01_fl_baseline")
    fd = load("exp02_fd_main")

    fl_comm = final_metric(fl, "baseline.fedavg", "cum_comm")
    fd_comm = final_metric(fd, "baseline.fedavg", "cum_comm")

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(["FL", "FD"], [fl_comm, fd_comm],
                  color=["#d62728", "#2ca02c"], width=0.5,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Cumulative Communication (MB)")
    ax.set_title("FL vs FD Communication")
    ax.set_yscale("log")

    ax.text(0, fl_comm * 1.3, f"{fl_comm:.0f} MB", ha="center", fontsize=8, fontweight="bold")
    ax.text(1, fd_comm * 1.8, f"{fd_comm:.1f} MB", ha="center", fontsize=8, fontweight="bold")
    ax.text(0.5, (fl_comm * fd_comm) ** 0.5, f"{fl_comm/fd_comm:.0f}x reduction",
            ha="center", fontsize=9, fontweight="bold", color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ca02c", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out / "fig09_comm_comparison.png")
    fig.savefig(out / "fig09_comm_comparison.eps")
    plt.close(fig)
    print(f"  -> fig09_comm_comparison")


def fig10_channel_selection_paradox(plt, out: Path):
    """Good channel count vs accuracy --- shows selecting good channels != good accuracy."""
    fd = load("exp02_fd_main")
    methods = METHOD_ORDER

    accs = [final_metric(fd, m, "accuracy") for m in methods]
    good_ch = [final_metric(fd, m, "num_good_channel") for m in methods]
    names = [SHORT[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, m in enumerate(methods):
        ax.scatter(good_ch[i], accs[i], color=COLORS[m], marker=MARKERS[m],
                   s=80, zorder=5, edgecolors="black", linewidth=0.5)
        offset_y = 0.003
        if names[i] in ("Oort", "LabelCov", "FedCor"):
            offset_y = -0.005
        ax.annotate(names[i], (good_ch[i], accs[i]),
                    xytext=(good_ch[i] + 0.15, accs[i] + offset_y),
                    fontsize=7, ha="left")

    ax.set_xlabel("Good-Channel Clients Selected (out of 10)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Channel Quality vs Accuracy Paradox")
    plt.tight_layout()
    fig.savefig(out / "fig10_channel_paradox.png")
    fig.savefig(out / "fig10_channel_paradox.eps")
    plt.close(fig)
    print(f"  -> fig10_channel_paradox")


def fig11_mnist_ranking_shift(plt, out: Path):
    """MNIST FL vs FD ranking shift bars."""
    fl = load("exp03_fl_mnist")
    fd = load("exp03_fd_mnist")

    methods = METHOD_ORDER
    fl_acc = [final_metric(fl, m, "accuracy") for m in methods]
    fd_acc = [final_metric(fd, m, "accuracy") for m in methods]
    names = [SHORT[m] for m in methods]

    fl_order = np.argsort(fl_acc)[::-1]
    fd_order = np.argsort(fd_acc)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.patch.set_facecolor("white")

    x = np.arange(len(methods))
    w = 0.6

    ax1.set_facecolor("white")
    ax1.barh(x, [fl_acc[i] for i in fl_order], w,
             color=[COLORS[methods[i]] for i in fl_order],
             edgecolor="white", linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels([names[i] for i in fl_order])
    ax1.set_xlabel("Final Accuracy")
    ax1.set_title("(a) FL Ranking (MNIST)")
    ax1.invert_yaxis()
    for i, idx in enumerate(fl_order):
        ax1.text(fl_acc[idx] + 0.001, i, f"{fl_acc[idx]:.3f}", va="center", fontsize=7)

    ax2.set_facecolor("white")
    ax2.barh(x, [fd_acc[i] for i in fd_order], w,
             color=[COLORS[methods[i]] for i in fd_order],
             edgecolor="white", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels([names[i] for i in fd_order])
    ax2.set_xlabel("Final Accuracy")
    ax2.set_title("(b) FD Ranking (MNIST + FMNIST)")
    ax2.invert_yaxis()
    for i, idx in enumerate(fd_order):
        ax2.text(fd_acc[idx] + 0.002, i, f"{fd_acc[idx]:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "fig11_mnist_ranking_shift.png")
    fig.savefig(out / "fig11_mnist_ranking_shift.eps")
    plt.close(fig)
    print(f"  -> fig11_mnist_ranking_shift")


def fig12_robustness_heatmap(plt, out: Path):
    """Heatmap of method robustness across dimensions."""
    # Collect robustness metrics (lower degradation = better, normalize 0-1)
    errfree = load("exp04_errfree")
    dl20 = load("exp04_dl-20db")
    homo = load("exp08_homo")
    hetero = load("exp08_hetero")
    fd = load("exp02_fd_main")

    methods = METHOD_ORDER
    names = [SHORT[m] for m in methods]

    # Noise robustness (inverted degradation)
    ef_acc = [final_metric(errfree, m, "accuracy") for m in methods]
    d20_acc = [final_metric(dl20, m, "accuracy") for m in methods]
    noise_deg = [(ef - d20) / ef for ef, d20 in zip(ef_acc, d20_acc)]
    noise_rob = [1.0 - d for d in noise_deg]

    # Hetero robustness
    ho_acc = [final_metric(homo, m, "accuracy") for m in methods]
    he_acc = [final_metric(hetero, m, "accuracy") for m in methods]
    het_deg = [(ho - he) / ho for ho, he in zip(ho_acc, he_acc)]
    het_rob = [1.0 - d for d in het_deg]

    # FD accuracy (normalized to 0-1 within methods)
    fd_acc = [final_metric(fd, m, "accuracy") for m in methods]
    fd_min, fd_max = min(fd_acc), max(fd_acc)
    fd_norm = [(a - fd_min) / (fd_max - fd_min) if fd_max > fd_min else 0.5
               for a in fd_acc]

    # Fairness (inverted gini)
    ginis = [final_metric(fd, m, "fairness_gini") for m in methods]
    fair_norm = [1.0 - g for g in ginis]

    # Client equity (inverted std, normalized)
    stds = [final_metric(fd, m, "client_accuracy_std") for m in methods]
    std_min, std_max = min(stds), max(stds)
    eq_norm = [1.0 - (s - std_min) / (std_max - std_min) if std_max > std_min else 0.5
               for s in stds]

    matrix = np.array([fd_norm, noise_rob, het_rob, fair_norm, eq_norm]).T
    dimensions = ["FD Accuracy", "Noise\nRobustness", "Hetero\nRobustness",
                   "Fairness", "Client\nEquity"]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_xticklabels(dimensions, fontsize=7)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Method Robustness Profile (higher = better)")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(dimensions)):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Score (0-1)")
    plt.tight_layout()
    fig.savefig(out / "fig12_robustness_heatmap.png")
    fig.savefig(out / "fig12_robustness_heatmap.eps")
    plt.close(fig)
    print(f"  -> fig12_robustness_heatmap")


def fig13_spearman_table(plt, out: Path):
    """Visual table of Spearman rank correlations."""
    # CIFAR-10
    fl_c = load("exp01_fl_baseline")
    fd_c = load("exp02_fd_main")
    # MNIST
    fl_m = load("exp03_fl_mnist")
    fd_m = load("exp03_fd_mnist")

    methods = METHOD_ORDER

    def rank_list(data, metric="accuracy"):
        vals = [final_metric(data, m, metric) for m in methods]
        order = np.argsort(vals)[::-1]
        ranks = np.zeros(len(methods))
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        return ranks

    fl_c_ranks = rank_list(fl_c)
    fd_c_ranks = rank_list(fd_c)
    fl_m_ranks = rank_list(fl_m)
    fd_m_ranks = rank_list(fd_m)

    def spearman(r1, r2):
        n = len(r1)
        d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
        return 1 - (6 * d2) / (n * (n**2 - 1))

    rho_cifar = spearman(fl_c_ranks, fd_c_ranks)
    rho_mnist = spearman(fl_m_ranks, fd_m_ranks)

    names = [SHORT[m] for m in methods]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cols = ["Method", "FL Rank\n(CIFAR)", "FD Rank\n(CIFAR)", "Shift",
            "FL Rank\n(MNIST)", "FD Rank\n(MNIST)", "Shift"]
    cell_data = []
    for i, m in enumerate(methods):
        shift_c = int(fl_c_ranks[i] - fd_c_ranks[i])
        shift_m = int(fl_m_ranks[i] - fd_m_ranks[i])
        cell_data.append([
            names[i],
            int(fl_c_ranks[i]), int(fd_c_ranks[i]),
            f"{'+' if shift_c > 0 else ''}{shift_c}",
            int(fl_m_ranks[i]), int(fd_m_ranks[i]),
            f"{'+' if shift_m > 0 else ''}{shift_m}",
        ])

    # Add rho row
    cell_data.append(["Spearman rho", "", "", f"{rho_cifar:.3f}", "", "", f"{rho_mnist:.3f}"])

    ax.axis("off")
    table = ax.table(cellText=cell_data, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#f0f0f0"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.4)

    # Highlight rho row
    for j in range(len(cols)):
        cell = table[len(cell_data), j]
        cell.set_facecolor("#ffffcc")
        cell.set_text_props(fontweight="bold")

    ax.set_title("FL-to-FD Rank Correlation", fontsize=10, pad=10)
    plt.tight_layout()
    fig.savefig(out / "fig13_spearman_table.png")
    fig.savefig(out / "fig13_spearman_table.eps")
    plt.close(fig)
    print(f"  -> fig13_spearman_table")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)

    out = Path("docs/figures/analysis")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating analysis figures...")

    fig01_ranking_shift_bars(plt, out)
    fig02_fd_convergence(plt, out)
    fig03_noise_degradation(plt, out)
    fig04_k_sweep(plt, out)
    fig05_alpha_sweep(plt, out)
    fig06_grouping_benefit(plt, out)
    fig07_hetero_degradation(plt, out)
    fig08_fairness_accuracy(plt, out)
    fig09_comm_comparison(plt, out)
    fig10_channel_selection_paradox(plt, out)
    fig11_mnist_ranking_shift(plt, out)
    fig12_robustness_heatmap(plt, out)
    fig13_spearman_table(plt, out)

    print(f"\nAll figures saved to {out}/")
    print("PNG for preview, EPS for paper inclusion.")


if __name__ == "__main__":
    main()
