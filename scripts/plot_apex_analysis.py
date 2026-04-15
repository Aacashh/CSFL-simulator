#!/usr/bin/env python3
"""Generate analysis figures for APEX v2 FL experiment analysis document.

Creates publication-quality plots summarizing key findings across all APEX experiments.
Output: docs/figures/apex_analysis/

Usage:
    python scripts/plot_apex_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Visual config (consistent with plot_fd_analysis.py)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline.fedavg":       "#1f77b4",
    "system_aware.fedcs":    "#d62728",
    "system_aware.oort":     "#17becf",
    "system_aware.tifl":     "#2ca02c",
    "system_aware.poc":      "#ff7f0e",
    "heuristic.mmr_diverse": "#9467bd",
    "ml.fedcor":             "#8c564b",
    "ml.apex_v2":            "#e377c2",
}

SHORT = {
    "baseline.fedavg":       "FedAvg",
    "system_aware.fedcs":    "FedCS",
    "system_aware.oort":     "Oort",
    "system_aware.tifl":     "TiFL",
    "system_aware.poc":      "PoC",
    "heuristic.mmr_diverse": "MMR-Div",
    "ml.fedcor":             "FedCor",
    "ml.apex_v2":            "APEX v2",
    # Ablation variants
    "ml.apex_v2_no_adaptive_recency": "No Recency",
    "ml.apex_v2_no_hysteresis":       "No Hysteresis",
    "ml.apex_v2_no_het_scaling":      "No Het-Scale",
    "ml.apex_v2_no_posterior_reg":    "No Post-Reg",
    "ml.apex_v2_no_adaptive_gamma":   "No Adapt-Gamma",
}

METHOD_ORDER = [
    "baseline.fedavg", "system_aware.fedcs", "system_aware.oort",
    "system_aware.tifl", "system_aware.poc", "heuristic.mmr_diverse",
    "ml.fedcor", "ml.apex_v2",
]

MARKERS = {
    "baseline.fedavg": "o", "system_aware.fedcs": "s",
    "system_aware.oort": "P", "system_aware.tifl": "^",
    "system_aware.poc": "D", "heuristic.mmr_diverse": "v",
    "ml.fedcor": "X", "ml.apex_v2": "*",
}

ABLATION_COLORS = {
    "ml.apex_v2":                     "#e377c2",
    "ml.apex_v2_no_adaptive_recency": "#1f77b4",
    "ml.apex_v2_no_hysteresis":       "#d62728",
    "ml.apex_v2_no_het_scaling":      "#2ca02c",
    "ml.apex_v2_no_posterior_reg":    "#ff7f0e",
    "ml.apex_v2_no_adaptive_gamma":   "#9467bd",
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
    res = data["results"].get(method, {})
    metrics_list = res.get("metrics", [])
    if not metrics_list:
        return default
    last = metrics_list[-1]
    return float(last.get(metric, default) or default)


def peak_metric(data: dict, method: str, metric: str, default=0.0):
    res = data["results"].get(method, {})
    metrics_list = res.get("metrics", [])
    if not metrics_list:
        return default
    return max(float(m.get(metric, default) or default) for m in metrics_list)


def series(data: dict, method: str, metric: str):
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


def load_seeds(prefix: str) -> list[dict]:
    """Load all seed variants for a run prefix."""
    results = []
    for seed in ["s42", "s123", "s456"]:
        name = f"{prefix}_{seed}"
        try:
            results.append(load(name))
        except FileNotFoundError:
            pass
    return results


def mean_final_across_seeds(seeds: list[dict], method: str, metric: str) -> tuple[float, float]:
    """Return (mean, std) of final metric across seed runs."""
    vals = []
    for data in seeds:
        v = final_metric(data, method, metric)
        if v != 0.0 or method in data.get("results", {}):
            vals.append(v)
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def mean_series_across_seeds(seeds: list[dict], method: str, metric: str):
    """Return (rounds, mean_values, std_values) averaged across seeds."""
    all_ys = []
    ref_xs = None
    for data in seeds:
        xs, ys = series(data, method, metric)
        if len(ys) > 0:
            all_ys.append(ys)
            if ref_xs is None:
                ref_xs = xs
    if not all_ys:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(y) for y in all_ys)
    all_ys = np.array([y[:min_len] for y in all_ys])
    return ref_xs[:min_len], np.mean(all_ys, axis=0), np.std(all_ys, axis=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_main_benchmark_bars(plt, out: Path):
    """Final accuracy bar chart for APEX 1: main benchmark (mean +/- std across 3 seeds)."""
    seeds = load_seeds("main_cifar10_a03")
    if not seeds:
        print("  [SKIP] fig01 -- no main_cifar10_a03 runs found")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]
    means, stds = [], []
    for m in methods:
        mu, sigma = mean_final_across_seeds(seeds, m, "accuracy")
        means.append(mu)
        stds.append(sigma)
    names = [SHORT.get(m, m.split(".")[-1]) for m in methods]

    # Sort by accuracy descending
    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(methods))
    bars = ax.barh(y,
                   [means[i] for i in order],
                   xerr=[stds[i] for i in order],
                   height=0.6,
                   color=[COLORS.get(methods[i], "#888888") for i in order],
                   edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Final Accuracy (mean +/- std, 3 seeds)")
    ax.set_title("CIFAR-10, Dirichlet $\\alpha$=0.3, N=50, K=10")
    ax.invert_yaxis()

    for i, idx in enumerate(order):
        ax.text(means[idx] + stds[idx] + 0.005, i,
                f"{means[idx]:.3f}$\\pm${stds[idx]:.3f}",
                va="center", fontsize=6.5)

    ax.set_xlim(right=max(means) + max(stds) + 0.08)
    plt.tight_layout()
    fig.savefig(out / "fig01_main_benchmark.png")
    fig.savefig(out / "fig01_main_benchmark.eps")
    plt.close(fig)
    print("  -> fig01_main_benchmark")


def fig02_convergence_curves(plt, out: Path):
    """Accuracy convergence curves with shaded std bands (APEX 1)."""
    seeds = load_seeds("main_cifar10_a03")
    if not seeds:
        print("  [SKIP] fig02 -- no data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "accuracy")
        if len(ys_mean) == 0:
            continue
        ys_s = smooth(ys_mean, 10)
        std_s = smooth(ys_std, 10)
        xs_s = xs[:len(ys_s)]

        mark_every = max(1, len(xs_s) // 8)
        ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                marker=MARKERS.get(m, "o"),
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS.get(m, "#888"),
                linewidth=1.2, label=SHORT.get(m, m))
        ax.fill_between(xs_s, ys_s - std_s, ys_s + std_s,
                        color=COLORS.get(m, "#888"), alpha=0.1)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Convergence: CIFAR-10, $\\alpha$=0.3 (3 seeds)")
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    plt.tight_layout()
    fig.savefig(out / "fig02_convergence.png")
    fig.savefig(out / "fig02_convergence.eps")
    plt.close(fig)
    print("  -> fig02_convergence")


def fig03_heterogeneity_sweep(plt, out: Path):
    """Accuracy across alpha levels: 0.1, 0.3, 0.6 (grouped bar)."""
    alphas_cfg = [
        (0.1, "het_a01"),
        (0.3, "main_cifar10_a03"),
        (0.6, "het_a06"),
    ]

    # Find common methods across all experiments
    all_seeds = {}
    for alpha, prefix in alphas_cfg:
        s = load_seeds(prefix)
        if s:
            all_seeds[alpha] = s

    if not all_seeds:
        print("  [SKIP] fig03 -- no data")
        return

    # Use methods from the alpha=0.1 set (smaller method set = APEX_CORE)
    ref_methods = list(all_seeds[0.1][0]["results"].keys()) if 0.1 in all_seeds else []
    if not ref_methods:
        ref_methods = list(list(all_seeds.values())[0][0]["results"].keys())

    # Only keep methods present in all experiments
    common_methods = []
    for m in METHOD_ORDER:
        if all(m in all_seeds[a][0]["results"] for a in all_seeds):
            common_methods.append(m)
    if not common_methods:
        common_methods = ref_methods

    names = [SHORT.get(m, m.split(".")[-1]) for m in common_methods]
    alphas = sorted(all_seeds.keys())

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(common_methods))
    n_alpha = len(alphas)
    w = 0.8 / n_alpha
    alpha_colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    for i, alpha in enumerate(alphas):
        means, stds = [], []
        for m in common_methods:
            mu, sigma = mean_final_across_seeds(all_seeds[alpha], m, "accuracy")
            means.append(mu)
            stds.append(sigma)
        offset = (i - (n_alpha - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=stds, label=f"$\\alpha$={alpha}",
               color=alpha_colors[i], alpha=0.8, capsize=2, error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Final Accuracy (mean +/- std)")
    ax.set_title("Heterogeneity Robustness: CIFAR-10, N=50")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "fig03_heterogeneity_sweep.png")
    fig.savefig(out / "fig03_heterogeneity_sweep.eps")
    plt.close(fig)
    print("  -> fig03_heterogeneity_sweep")


def fig04_scalability(plt, out: Path):
    """Accuracy vs number of clients (N=50, 100, 200)."""
    scale_cfgs = [
        (50, "main_cifar10_a03"),
        (100, "scale_n100"),
        (200, "scale_n200"),
    ]

    all_seeds = {}
    for n, prefix in scale_cfgs:
        s = load_seeds(prefix)
        if s:
            all_seeds[n] = s

    if len(all_seeds) < 2:
        print("  [SKIP] fig04 -- insufficient data")
        return

    # Common methods (APEX_SCALE set)
    common_methods = []
    for m in METHOD_ORDER:
        if all(m in all_seeds[n][0]["results"] for n in all_seeds):
            common_methods.append(m)

    ns = sorted(all_seeds.keys())

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in common_methods:
        means, stds = [], []
        for n in ns:
            mu, sigma = mean_final_across_seeds(all_seeds[n], m, "accuracy")
            means.append(mu)
            stds.append(sigma)
        ax.errorbar(ns, means, yerr=stds, color=COLORS.get(m, "#888"),
                    marker=MARKERS.get(m, "o"), markersize=7,
                    markerfacecolor="white", markeredgewidth=1.0,
                    markeredgecolor=COLORS.get(m, "#888"),
                    linewidth=1.4, capsize=4, label=SHORT.get(m, m))

    ax.set_xlabel("Total Clients (N)")
    ax.set_ylabel("Final Accuracy")
    k_note = "/".join(str(n // 10 if n <= 100 else n // 10) for n in ns)
    ax.set_title(f"Scalability: CIFAR-10, $\\alpha$=0.3")
    ax.set_xticks(ns)
    ax.legend(loc="upper right", fontsize=6.5)
    plt.tight_layout()
    fig.savefig(out / "fig04_scalability.png")
    fig.savefig(out / "fig04_scalability.eps")
    plt.close(fig)
    print("  -> fig04_scalability")


def fig05_convergence_loss(plt, out: Path):
    """Loss convergence curves with shaded std bands (APEX 1)."""
    seeds = load_seeds("main_cifar10_a03")
    if not seeds:
        print("  [SKIP] fig05 -- no data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "loss")
        if len(ys_mean) == 0:
            continue
        ys_s = smooth(ys_mean, 10)
        xs_s = xs[:len(ys_s)]
        mark_every = max(1, len(xs_s) // 8)
        ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                marker=MARKERS.get(m, "o"),
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS.get(m, "#888"),
                linewidth=1.2, label=SHORT.get(m, m))

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Loss")
    ax.set_title("Loss Convergence: CIFAR-10, $\\alpha$=0.3 (3 seeds)")
    ax.legend(loc="upper right", ncol=2, fontsize=5.5)
    plt.tight_layout()
    fig.savefig(out / "fig05_loss_convergence.png")
    fig.savefig(out / "fig05_loss_convergence.eps")
    plt.close(fig)
    print("  -> fig05_loss_convergence")


def fig06_fairness_accuracy_tradeoff(plt, out: Path):
    """Scatter: Gini vs Accuracy for main benchmark (seed 42)."""
    try:
        data = load("main_cifar10_a03_s42")
    except FileNotFoundError:
        print("  [SKIP] fig06 -- no data")
        return

    methods = [m for m in METHOD_ORDER if m in data.get("results", {})]
    accs = [final_metric(data, m, "accuracy") for m in methods]
    ginis = [final_metric(data, m, "fairness_gini") for m in methods]
    names = [SHORT.get(m, m.split(".")[-1]) for m in methods]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, m in enumerate(methods):
        ax.scatter(ginis[i], accs[i], color=COLORS.get(m, "#888"),
                   marker=MARKERS.get(m, "o"),
                   s=80, zorder=5, edgecolors="black", linewidth=0.5)
        offset_x, offset_y = 0.02, 0.003
        if names[i] in ("Oort", "FedCS", "TiFL", "FedCor"):
            offset_y = -0.006
        ax.annotate(names[i], (ginis[i], accs[i]),
                    xytext=(ginis[i] + offset_x, accs[i] + offset_y),
                    fontsize=7, ha="left")

    ax.set_xlabel("Fairness Gini (lower = fairer)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Accuracy-Fairness Tradeoff (CIFAR-10)")
    plt.tight_layout()
    fig.savefig(out / "fig06_fairness_accuracy.png")
    fig.savefig(out / "fig06_fairness_accuracy.eps")
    plt.close(fig)
    print("  -> fig06_fairness_accuracy")


def fig07_extreme_noniid_convergence(plt, out: Path):
    """Convergence under extreme non-IID (alpha=0.1) with std bands."""
    seeds = load_seeds("het_a01")
    if not seeds:
        print("  [SKIP] fig07 -- no data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "accuracy")
        if len(ys_mean) == 0:
            continue
        ys_s = smooth(ys_mean, 10)
        std_s = smooth(ys_std, 10)
        xs_s = xs[:len(ys_s)]
        mark_every = max(1, len(xs_s) // 8)
        ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                marker=MARKERS.get(m, "o"),
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS.get(m, "#888"),
                linewidth=1.2, label=SHORT.get(m, m))
        ax.fill_between(xs_s, ys_s - std_s, ys_s + std_s,
                        color=COLORS.get(m, "#888"), alpha=0.1)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Extreme Non-IID: CIFAR-10, $\\alpha$=0.1 (3 seeds)")
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    plt.tight_layout()
    fig.savefig(out / "fig07_extreme_noniid.png")
    fig.savefig(out / "fig07_extreme_noniid.eps")
    plt.close(fig)
    print("  -> fig07_extreme_noniid")


def fig08_apex_advantage_heatmap(plt, out: Path):
    """Heatmap: APEX v2 accuracy advantage over each baseline across settings."""
    experiments = {
        "$\\alpha$=0.1\nN=50": "het_a01",
        "$\\alpha$=0.3\nN=50": "main_cifar10_a03",
        "$\\alpha$=0.6\nN=50": "het_a06",
        "$\\alpha$=0.3\nN=100": "scale_n100",
        "$\\alpha$=0.3\nN=200": "scale_n200",
        "CIFAR-100\n$\\alpha$=0.3": "cifar100",
        "IID\nCIFAR-10": "iid",
    }

    baselines = ["baseline.fedavg", "system_aware.oort", "system_aware.poc", "heuristic.mmr_diverse"]
    baseline_names = [SHORT.get(m, m) for m in baselines]
    setting_names = list(experiments.keys())

    matrix = np.full((len(baselines), len(experiments)), np.nan)

    for j, (setting, prefix) in enumerate(experiments.items()):
        seeds = load_seeds(prefix)
        if not seeds:
            continue
        apex_mean, _ = mean_final_across_seeds(seeds, "ml.apex_v2", "accuracy")
        for i, base in enumerate(baselines):
            base_mean, _ = mean_final_across_seeds(seeds, base, "accuracy")
            if base_mean > 0:
                matrix[i, j] = (apex_mean - base_mean) * 100  # percentage points

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                   vmin=-3, vmax=15)
    ax.set_xticks(np.arange(len(setting_names)))
    ax.set_xticklabels(setting_names, fontsize=7)
    ax.set_yticks(np.arange(len(baselines)))
    ax.set_yticklabels(baseline_names, fontsize=8)
    ax.set_title("APEX v2 Accuracy Advantage (pp) over Baselines")

    for i in range(len(baselines)):
        for j in range(len(experiments)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color="gray")
                continue
            color = "white" if abs(val) > 4 else "black"
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy Advantage (pp)")
    plt.tight_layout()
    fig.savefig(out / "fig08_advantage_heatmap.png")
    fig.savefig(out / "fig08_advantage_heatmap.eps")
    plt.close(fig)
    print("  -> fig08_advantage_heatmap")


def fig09_convergence_speed(plt, out: Path):
    """Bar chart: rounds to reach 60% and 65% accuracy (main benchmark)."""
    seeds = load_seeds("main_cifar10_a03")
    if not seeds:
        print("  [SKIP] fig09 -- no data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]
    thresholds = [0.60, 0.65]
    threshold_labels = ["60%", "65%"]

    method_times = {m: {t: [] for t in thresholds} for m in methods}

    for data in seeds:
        for m in methods:
            metrics_list = data["results"].get(m, {}).get("metrics", [])
            for t in thresholds:
                found = False
                for idx, row in enumerate(metrics_list):
                    if float(row.get("accuracy", 0)) >= t:
                        method_times[m][t].append(idx + 1)
                        found = True
                        break
                if not found:
                    method_times[m][t].append(201)  # did not reach

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Sort by time to 60%
    sorted_methods = sorted(methods,
                            key=lambda m: np.mean(method_times[m][0.60]))
    names = [SHORT.get(m, m.split(".")[-1]) for m in sorted_methods]

    x = np.arange(len(sorted_methods))
    w = 0.35

    for i, t in enumerate(thresholds):
        means = [np.mean(method_times[m][t]) for m in sorted_methods]
        stds = [np.std(method_times[m][t]) for m in sorted_methods]
        colors_t = ["#2ca02c" if t == 0.60 else "#ff7f0e"] * len(sorted_methods)
        ax.barh(x + (i - 0.5) * w, means, w, xerr=stds,
                label=f"To {threshold_labels[i]}", color=colors_t[0],
                alpha=0.8, capsize=2, error_kw={"linewidth": 0.7})

    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel("Rounds to Reach Threshold")
    ax.set_title("Convergence Speed (CIFAR-10, $\\alpha$=0.3)")
    ax.axvline(x=200, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Max (200)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out / "fig09_convergence_speed.png")
    fig.savefig(out / "fig09_convergence_speed.eps")
    plt.close(fig)
    print("  -> fig09_convergence_speed")


def fig10a_ablation_bars(plt, out: Path):
    """Ablation: accuracy of full APEX v2 vs each component-removed variant."""
    seeds = load_seeds("ablation")
    if not seeds:
        print("  [SKIP] fig10a -- no ablation data")
        return

    ablation_order = [
        "ml.apex_v2",
        "ml.apex_v2_no_adaptive_recency",
        "ml.apex_v2_no_hysteresis",
        "ml.apex_v2_no_het_scaling",
        "ml.apex_v2_no_posterior_reg",
        "ml.apex_v2_no_adaptive_gamma",
    ]
    abl_short = {
        "ml.apex_v2":                     "APEX v2 (Full)",
        "ml.apex_v2_no_adaptive_recency": "w/o Adapt. Recency",
        "ml.apex_v2_no_hysteresis":       "w/o Phase Hysteresis",
        "ml.apex_v2_no_het_scaling":      "w/o Het-Scaling",
        "ml.apex_v2_no_posterior_reg":    "w/o Post. Regulariz.",
        "ml.apex_v2_no_adaptive_gamma":   "w/o Adapt. Gamma",
    }

    methods = [m for m in ablation_order if m in seeds[0].get("results", {})]
    means, stds = [], []
    for m in methods:
        mu, sigma = mean_final_across_seeds(seeds, m, "accuracy")
        means.append(mu)
        stds.append(sigma)
    names = [abl_short.get(m, m) for m in methods]

    # Sort by accuracy descending
    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(methods))
    colors = [ABLATION_COLORS.get(methods[i], "#888") for i in order]
    ax.barh(y, [means[i] for i in order], xerr=[stds[i] for i in order],
            height=0.6, color=colors, edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("Final Accuracy (mean +/- std)")
    ax.set_title("Ablation Study: CIFAR-10, $\\alpha$=0.3")
    ax.invert_yaxis()

    # Full APEX line
    full_idx = methods.index("ml.apex_v2") if "ml.apex_v2" in methods else None
    if full_idx is not None:
        ax.axvline(x=means[full_idx], color=ABLATION_COLORS["ml.apex_v2"],
                   linestyle="--", linewidth=0.8, alpha=0.7)

    for i, idx in enumerate(order):
        ax.text(means[idx] + stds[idx] + 0.003, i,
                f"{means[idx]:.3f}$\\pm${stds[idx]:.3f}",
                va="center", fontsize=6)

    ax.set_xlim(right=max(means) + max(stds) + 0.06)
    plt.tight_layout()
    fig.savefig(out / "fig10a_ablation.png")
    fig.savefig(out / "fig10a_ablation.eps")
    plt.close(fig)
    print("  -> fig10a_ablation")


def fig10b_ablation_delta(plt, out: Path):
    """Ablation: accuracy delta (pp) from removing each component."""
    seeds = load_seeds("ablation")
    if not seeds:
        print("  [SKIP] fig10b -- no ablation data")
        return

    ablation_variants = [
        "ml.apex_v2_no_adaptive_recency",
        "ml.apex_v2_no_hysteresis",
        "ml.apex_v2_no_het_scaling",
        "ml.apex_v2_no_posterior_reg",
        "ml.apex_v2_no_adaptive_gamma",
    ]
    component_names = {
        "ml.apex_v2_no_adaptive_recency": "Adaptive Recency",
        "ml.apex_v2_no_hysteresis":       "Phase Hysteresis",
        "ml.apex_v2_no_het_scaling":      "Het-Aware Scaling",
        "ml.apex_v2_no_posterior_reg":    "Posterior Regulariz.",
        "ml.apex_v2_no_adaptive_gamma":   "Adaptive Gamma",
    }

    full_mean, _ = mean_final_across_seeds(seeds, "ml.apex_v2", "accuracy")

    deltas = []
    names = []
    for v in ablation_variants:
        if v not in seeds[0].get("results", {}):
            continue
        v_mean, _ = mean_final_across_seeds(seeds, v, "accuracy")
        delta = (v_mean - full_mean) * 100  # positive = variant is BETTER
        deltas.append(delta)
        names.append(component_names.get(v, v))

    order = np.argsort(deltas)

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(deltas))
    colors = ["#d62728" if deltas[i] > 0 else "#2ca02c" for i in order]
    ax.barh(y, [deltas[i] for i in order], height=0.55,
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Change (pp) vs Full APEX v2")
    ax.set_title("Impact of Removing Each Component")

    for i, idx in enumerate(order):
        sign = "+" if deltas[idx] > 0 else ""
        ha = "left" if deltas[idx] >= 0 else "right"
        offset = 0.1 if deltas[idx] >= 0 else -0.1
        ax.text(deltas[idx] + offset, i,
                f"{sign}{deltas[idx]:.2f} pp", va="center", fontsize=7, ha=ha)

    plt.tight_layout()
    fig.savefig(out / "fig10b_ablation_delta.png")
    fig.savefig(out / "fig10b_ablation_delta.eps")
    plt.close(fig)
    print("  -> fig10b_ablation_delta")


def fig11_cifar100_benchmark(plt, out: Path):
    """CIFAR-100 accuracy bars with error bars."""
    seeds = load_seeds("cifar100")
    if not seeds:
        print("  [SKIP] fig11 -- no CIFAR-100 data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]
    means, stds = [], []
    for m in methods:
        mu, sigma = mean_final_across_seeds(seeds, m, "accuracy")
        means.append(mu)
        stds.append(sigma)
    names = [SHORT.get(m, m.split(".")[-1]) for m in methods]

    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(methods))
    ax.barh(y, [means[i] for i in order], xerr=[stds[i] for i in order],
            height=0.6,
            color=[COLORS.get(methods[i], "#888") for i in order],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Final Accuracy (mean +/- std)")
    ax.set_title("CIFAR-100, Dirichlet $\\alpha$=0.3, N=50, K=10")
    ax.invert_yaxis()

    for i, idx in enumerate(order):
        ax.text(means[idx] + stds[idx] + 0.003, i,
                f"{means[idx]:.3f}$\\pm${stds[idx]:.3f}",
                va="center", fontsize=6.5)

    ax.set_xlim(right=max(means) + max(stds) + 0.06)
    plt.tight_layout()
    fig.savefig(out / "fig11_cifar100_benchmark.png")
    fig.savefig(out / "fig11_cifar100_benchmark.eps")
    plt.close(fig)
    print("  -> fig11_cifar100_benchmark")


def fig12_cifar100_convergence(plt, out: Path):
    """CIFAR-100 convergence curves."""
    seeds = load_seeds("cifar100")
    if not seeds:
        print("  [SKIP] fig12 -- no CIFAR-100 data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "accuracy")
        if len(ys_mean) == 0:
            continue
        ys_s = smooth(ys_mean, 10)
        std_s = smooth(ys_std, 10)
        xs_s = xs[:len(ys_s)]
        mark_every = max(1, len(xs_s) // 8)
        ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                marker=MARKERS.get(m, "o"),
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS.get(m, "#888"),
                linewidth=1.2, label=SHORT.get(m, m))
        ax.fill_between(xs_s, ys_s - std_s, ys_s + std_s,
                        color=COLORS.get(m, "#888"), alpha=0.1)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("CIFAR-100 Convergence, $\\alpha$=0.3")
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    plt.tight_layout()
    fig.savefig(out / "fig12_cifar100_convergence.png")
    fig.savefig(out / "fig12_cifar100_convergence.eps")
    plt.close(fig)
    print("  -> fig12_cifar100_convergence")


def fig10_scale_convergence(plt, out: Path):
    """Side-by-side convergence: N=50 vs N=100 vs N=200."""
    scale_data = [
        ("main_cifar10_a03", "(a) N=50, K=10"),
        ("scale_n100", "(b) N=100, K=10"),
        ("scale_n200", "(c) N=200, K=20"),
    ]

    panels = []
    for prefix, title in scale_data:
        s = load_seeds(prefix)
        if s:
            panels.append((s, title))

    if len(panels) < 2:
        print("  [SKIP] fig10 -- insufficient data")
        return

    # Find common methods across all panels
    common = []
    for m in METHOD_ORDER:
        if all(m in p[0][0]["results"] for p in panels):
            common.append(m)

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 2.8), sharey=True)
    fig.patch.set_facecolor("white")
    if ncols == 1:
        axes = [axes]

    for ax, (seeds, title) in zip(axes, panels):
        ax.set_facecolor("white")
        for m in common:
            xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "accuracy")
            if len(ys_mean) == 0:
                continue
            ys_s = smooth(ys_mean, 10)
            xs_s = xs[:len(ys_s)]
            mark_every = max(1, len(xs_s) // 8)
            ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                    marker=MARKERS.get(m, "o"),
                    markevery=mark_every, markersize=4,
                    markerfacecolor="white", markeredgewidth=0.8,
                    markeredgecolor=COLORS.get(m, "#888"),
                    linewidth=1.2, label=SHORT.get(m, m))
        ax.set_xlabel("Communication Round")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=5.5)

    axes[0].set_ylabel("Test Accuracy")
    plt.tight_layout()
    fig.savefig(out / "fig10_scale_convergence.png")
    fig.savefig(out / "fig10_scale_convergence.eps")
    plt.close(fig)
    print("  -> fig10_scale_convergence")


def fig13_cross_dataset(plt, out: Path):
    """Cross-dataset generalization: APEX v2 rank and accuracy across all datasets."""
    datasets_cfg = [
        ("MNIST", "mnist"),
        ("FMNIST", "fmnist"),
        ("CIFAR-10", "main_cifar10_a03"),
        ("CIFAR-100", "cifar100"),
    ]

    # Methods present across most datasets
    target_methods = ["baseline.fedavg", "system_aware.oort", "system_aware.poc", "ml.apex_v2"]

    all_data = {}
    for dname, prefix in datasets_cfg:
        s = load_seeds(prefix)
        if s:
            all_data[dname] = s

    if not all_data:
        print("  [SKIP] fig13 -- no data")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dnames = list(all_data.keys())
    x = np.arange(len(dnames))
    n_methods = len(target_methods)
    w = 0.8 / n_methods

    for i, m in enumerate(target_methods):
        means, stds = [], []
        for dname in dnames:
            seeds = all_data[dname]
            mu, sigma = mean_final_across_seeds(seeds, m, "accuracy")
            means.append(mu)
            stds.append(sigma)
        offset = (i - (n_methods - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=stds, label=SHORT.get(m, m),
               color=COLORS.get(m, "#888"), alpha=0.85, capsize=2,
               error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(dnames, fontsize=8)
    ax.set_ylabel("Final Accuracy (mean +/- std)")
    ax.set_title("Cross-Dataset Generalization (Dir. $\\alpha$=0.3, N=50)")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(out / "fig13_cross_dataset.png")
    fig.savefig(out / "fig13_cross_dataset.eps")
    plt.close(fig)
    print("  -> fig13_cross_dataset")


def fig14_iid_sanity(plt, out: Path):
    """IID sanity check: accuracy bars."""
    seeds = load_seeds("iid")
    if not seeds:
        print("  [SKIP] fig14 -- no IID data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]
    means, stds = [], []
    for m in methods:
        mu, sigma = mean_final_across_seeds(seeds, m, "accuracy")
        means.append(mu)
        stds.append(sigma)
    names = [SHORT.get(m, m.split(".")[-1]) for m in methods]

    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(methods))
    ax.barh(y, [means[i] for i in order], xerr=[stds[i] for i in order],
            height=0.6,
            color=[COLORS.get(methods[i], "#888") for i in order],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Final Accuracy (mean +/- std)")
    ax.set_title("IID Sanity Check: CIFAR-10, N=50, K=10")
    ax.invert_yaxis()

    for i, idx in enumerate(order):
        ax.text(means[idx] + stds[idx] + 0.005, i,
                f"{means[idx]:.3f}$\\pm${stds[idx]:.3f}",
                va="center", fontsize=6.5)

    ax.set_xlim(right=max(means) + max(stds) + 0.08)
    plt.tight_layout()
    fig.savefig(out / "fig14_iid_sanity.png")
    fig.savefig(out / "fig14_iid_sanity.eps")
    plt.close(fig)
    print("  -> fig14_iid_sanity")


def fig15_fmnist_convergence(plt, out: Path):
    """Fashion-MNIST convergence curves."""
    seeds = load_seeds("fmnist")
    if not seeds:
        print("  [SKIP] fig15 -- no FMNIST data")
        return

    methods = [m for m in METHOD_ORDER if m in seeds[0].get("results", {})]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for m in methods:
        xs, ys_mean, ys_std = mean_series_across_seeds(seeds, m, "accuracy")
        if len(ys_mean) == 0:
            continue
        ys_s = smooth(ys_mean, 10)
        std_s = smooth(ys_std, 10)
        xs_s = xs[:len(ys_s)]
        mark_every = max(1, len(xs_s) // 8)
        ax.plot(xs_s, ys_s, color=COLORS.get(m, "#888"),
                marker=MARKERS.get(m, "o"),
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS.get(m, "#888"),
                linewidth=1.2, label=SHORT.get(m, m))
        ax.fill_between(xs_s, ys_s - std_s, ys_s + std_s,
                        color=COLORS.get(m, "#888"), alpha=0.1)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Fashion-MNIST Convergence, $\\alpha$=0.3 (3 seeds)")
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    plt.tight_layout()
    fig.savefig(out / "fig15_fmnist_convergence.png")
    fig.savefig(out / "fig15_fmnist_convergence.eps")
    plt.close(fig)
    print("  -> fig15_fmnist_convergence")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)

    out = Path("docs/figures/apex_analysis")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating APEX v2 analysis figures...")

    fig01_main_benchmark_bars(plt, out)
    fig02_convergence_curves(plt, out)
    fig03_heterogeneity_sweep(plt, out)
    fig04_scalability(plt, out)
    fig05_convergence_loss(plt, out)
    fig06_fairness_accuracy_tradeoff(plt, out)
    fig07_extreme_noniid_convergence(plt, out)
    fig08_apex_advantage_heatmap(plt, out)
    fig09_convergence_speed(plt, out)
    fig10a_ablation_bars(plt, out)
    fig10b_ablation_delta(plt, out)
    fig11_cifar100_benchmark(plt, out)
    fig12_cifar100_convergence(plt, out)
    fig10_scale_convergence(plt, out)
    fig13_cross_dataset(plt, out)
    fig14_iid_sanity(plt, out)
    fig15_fmnist_convergence(plt, out)

    print(f"\nAll figures saved to {out}/")
    print("PNG for preview, EPS for paper inclusion.")


if __name__ == "__main__":
    main()
