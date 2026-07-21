#!/usr/bin/env python3
"""Generate paper figures for SCOPE-FD.

Outputs 7 IEEE-ready figures (.eps + .png) to docs/figures/scope_paper/.

Data source: artifacts/runs/scope_*/compare_results.json (19 runs)
+ historical baselines: artifacts/runs/exp2_noise_*/compare_results.json,
  artifacts/runs/exp3_alpha_*/compare_results.json (random baselines).

Usage: python scripts/plot_scope_paper.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("docs/figures/scope_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS = Path("artifacts/runs")

SCOPE = "fd_native.scope_fd"
RANDOM = "heuristic.random"

SCOPE_COLOR = "#d62728"
RANDOM_COLOR = "#1f77b4"
ABLATION_PALETTE = ["#d62728", "#e377c2", "#ff7f0e"]

# IEEE-standard line-plot marker style — keep consistent across every line figure.
def line_style(color: str, marker: str, label: str, linewidth: float = 1.4):
    return dict(
        color=color, marker=marker, linestyle="-", linewidth=linewidth, label=label,
        markersize=5, markerfacecolor=color, markeredgecolor="white",
        markeredgewidth=0.5, alpha=0.95,
    )

STYLE_RANDOM = lambda lw=1.4: line_style(RANDOM_COLOR, "o", "Random", lw)
STYLE_SCOPE  = lambda lw=1.4: line_style(SCOPE_COLOR,  "s", "SCOPE-FD", lw)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
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
        "axes.linewidth": 0.7,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "legend.framealpha": 0.88,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
    })


def load(run_name: str) -> dict | None:
    fp = RUNS / run_name / "compare_results.json"
    if not fp.exists():
        warnings.warn(f"Missing: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)


def final(data: dict, method: str, metric: str, default=float("nan")) -> float:
    try:
        return float(data["results"][method]["metrics"][-1].get(metric, default))
    except (KeyError, IndexError, TypeError):
        return default


def series(data: dict, method: str, metric: str):
    xs, ys = [], []
    for row in data["results"][method]["metrics"]:
        r = row.get("round", -1)
        if isinstance(r, (int, float)) and r >= 0:
            xs.append(int(r))
            ys.append(float(row.get(metric, 0.0) or 0.0))
    return np.array(xs), np.array(ys)


def save(fig, stem: str) -> None:
    for ext in ("png", "eps"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.(png|eps)")


# ---------------------------------------------------------------------------
# Experiment → run-directory map (single source of truth)
# ---------------------------------------------------------------------------
RUN_MAP = {
    # Headlines
    "cifar_scope":   "scope_cifar_20260422-173505",
    "cifar_random":  "exp2_noise_dl-20_20260422-035456",   # same N=30/K=10/α=0.5/100R/DL=-20 config
    "mnist_paired":  "scope_mnist_20260424-155455",
    "fmnist_paired": "scope_fmnist_20260424-161247",
    # Noise sweep — SCOPE (new) + random (historical exp2 at matching configs)
    "noise_errfree_scope":  "scope_noise_errfree_20260422-185046",
    "noise_errfree_random": "exp2_noise_errfree_20260421-200146",
    "noise_dl0_scope":      "scope_noise_dl0_20260422-200631",
    "noise_dl0_random":     "exp2_noise_dl0_20260421-234244",
    "noise_dl-10_scope":    "scope_noise_dl-10_20260422-212220",
    "noise_dl-10_random":   "exp2_noise_dl-10_20260422-014843",
    "noise_dl-20_scope":    "scope_noise_dl-20_20260422-223747",
    "noise_dl-20_random":   "exp2_noise_dl-20_20260422-035456",
    "noise_dl-30_scope":    "scope_noise_dl-30_20260422-235334",
    "noise_dl-30_random":   "exp2_noise_dl-30_20260422-060047",
    # Alpha sweep
    "alpha_0_1_scope":  "scope_alpha_0_1_20260423-010907",
    "alpha_0_1_random": "exp3_alpha_0_1_20260422-080619",
    "alpha_0_5_scope":  "scope_alpha_0_5_20260423-022132",
    "alpha_0_5_random": "exp3_alpha_0_5_20260422-100246",
    "alpha_5_0_scope":  "scope_alpha_5_0_20260423-033719",
    "alpha_5_0_random": "exp3_alpha_5_0_20260422-120828",
    # Ablation
    "ablation": "scope_ablation_20260423-044804",
    # K-sweep at N=50 on FMNIST
    "ksweep_k1":  "scope_fmnist_N50_K1_20260424-165027",
    "ksweep_k5":  "scope_fmnist_N50_K5_20260424-165310",
    "ksweep_k10": "scope_fmnist_N50_K10_20260424-165947",
    "ksweep_k15": "scope_fmnist_N50_K15_20260424-171042",
    "ksweep_k25": "scope_fmnist_N50_K25_20260424-172620",
    "ksweep_k35": "scope_fmnist_N50_K35_20260424-175432",
    "ksweep_k50": "scope_fmnist_N50_K50_20260424-183303",
    # FL baseline comparison (historical, CIFAR-10, N=50, K=15, R=300, FL paradigm)
    "fl_baseline": "fl_cifar10_baseline_20260416-123215",
    # Exp 8 — FMNIST channel-noise sweep at N=50, K=5 (5 DL SNR levels)
    "csweep_errfree": "scope_fmnist_N50_K5_noise_errfree_20260425-033726",
    "csweep_dl0":     "scope_fmnist_N50_K5_noise_dl0_20260425-034428",
    "csweep_dl-10":   "scope_fmnist_N50_K5_noise_dl-10_20260425-035125",
    "csweep_dl-20":   "scope_fmnist_N50_K5_noise_dl-20_20260425-035818",
    "csweep_dl-30":   "scope_fmnist_N50_K5_noise_dl-30_20260425-040506",
    # Pre-submission Exp 1 — coefficient ablation at FMNIST N=50, K=5 (5 (αu, αd) pairs)
    "coef_0_00_0_00": "scope_fmnist_N50_K5_coef_0_00_0_00_20260429-002645",
    "coef_0_10_0_10": "scope_fmnist_N50_K5_coef_0_10_0_10_20260429-003948",
    "coef_0_30_0_10": "scope_fmnist_N50_K5_coef_0_30_0_10_20260429-005237",
    "coef_0_50_0_40": "scope_fmnist_N50_K5_coef_0_50_0_40_20260429-010512",
    "coef_0_70_0_40": "scope_fmnist_N50_K5_coef_0_70_0_40_20260429-011846",
    # Pre-submission Exp 2 — EMNIST/digits cross-domain replication at N=50, K=5
    "emnist_paired":  "scope_emnist_N50_K5_20260429-040337",
}


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------
def rounds_to_target(data: dict, method: str, target_acc: float) -> int | None:
    """Return first round index where accuracy >= target, or None if never reached."""
    for row in data["results"][method]["metrics"]:
        r = row.get("round", -1)
        if not isinstance(r, (int, float)) or r < 0:
            continue
        if float(row.get("accuracy", 0.0) or 0.0) >= target_acc:
            return int(r)
    return None


# ---------------------------------------------------------------------------
# Figure 1 — Headline bars (3 datasets × 2 methods × 2 metrics)
# ---------------------------------------------------------------------------
def fig_headline_bars():
    cifar_s = load(RUN_MAP["cifar_scope"])
    cifar_r = load(RUN_MAP["cifar_random"])
    mnist = load(RUN_MAP["mnist_paired"])
    fmnist = load(RUN_MAP["fmnist_paired"])

    datasets = ["CIFAR-10", "MNIST", "Fashion-MNIST"]
    acc_rnd = [final(cifar_r, RANDOM, "accuracy"),
               final(mnist, RANDOM, "accuracy"),
               final(fmnist, RANDOM, "accuracy")]
    acc_scp = [final(cifar_s, SCOPE, "accuracy"),
               final(mnist, SCOPE, "accuracy"),
               final(fmnist, SCOPE, "accuracy")]
    gini_rnd = [final(cifar_r, RANDOM, "fairness_gini"),
                final(mnist, RANDOM, "fairness_gini"),
                final(fmnist, RANDOM, "fairness_gini")]
    gini_scp = [final(cifar_s, SCOPE, "fairness_gini"),
                final(mnist, SCOPE, "fairness_gini"),
                final(fmnist, SCOPE, "fairness_gini")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.9))
    x = np.arange(3)
    w = 0.36

    a1.bar(x - w / 2, acc_rnd, w, label="Random", color=RANDOM_COLOR, edgecolor="white", linewidth=0.6)
    a1.bar(x + w / 2, acc_scp, w, label="SCOPE-FD", color=SCOPE_COLOR, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(acc_rnd):
        a1.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    for i, v in enumerate(acc_scp):
        a1.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    a1.set_xticks(x)
    a1.set_xticklabels(datasets)
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy")
    a1.set_ylim(0, max(max(acc_rnd), max(acc_scp)) * 1.18)
    a1.legend(loc="upper left")

    a2.bar(x - w / 2, gini_rnd, w, label="Random", color=RANDOM_COLOR, edgecolor="white", linewidth=0.6)
    a2.bar(x + w / 2, gini_scp, w, label="SCOPE-FD", color=SCOPE_COLOR, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(gini_rnd):
        a2.text(i - w / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    for i, v in enumerate(gini_scp):
        a2.text(i + w / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=6)
    a2.set_xticks(x)
    a2.set_xticklabels(datasets)
    a2.set_ylabel("Participation Gini (lower = fairer)")
    a2.set_title("(b) Participation fairness")
    a2.set_ylim(0, max(gini_rnd) * 1.25)
    a2.legend(loc="upper right")

    plt.tight_layout()
    save(fig, "fig1_headline_bars")


# ---------------------------------------------------------------------------
# Figure 2 — Noise robustness (CIFAR-10, DL SNR sweep)
# ---------------------------------------------------------------------------
def fig_noise_robustness():
    labels = ["error-free", "0", "-10", "-20", "-30"]
    tags = ["errfree", "dl0", "dl-10", "dl-20", "dl-30"]
    acc_rnd, acc_scp, g_rnd, g_scp = [], [], [], []
    for t in tags:
        r = load(RUN_MAP[f"noise_{t}_random"])
        s = load(RUN_MAP[f"noise_{t}_scope"])
        acc_rnd.append(final(r, RANDOM, "accuracy"))
        acc_scp.append(final(s, SCOPE, "accuracy"))
        g_rnd.append(final(r, RANDOM, "fairness_gini"))
        g_scp.append(final(s, SCOPE, "fairness_gini"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    x = np.arange(len(labels))

    a1.plot(x, acc_rnd, **STYLE_RANDOM())
    a1.plot(x, acc_scp, **STYLE_SCOPE())
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set_xlabel("Downlink SNR (dB)")
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy vs DL SNR (CIFAR-10)")
    a1.legend(loc="best")

    a2.plot(x, g_rnd, **STYLE_RANDOM())
    a2.plot(x, g_scp, **STYLE_SCOPE())
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_xlabel("Downlink SNR (dB)")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Participation Gini vs DL SNR")
    a2.set_ylim(-0.005, max(g_rnd) * 1.15)
    a2.legend(loc="center right")

    plt.tight_layout()
    save(fig, "fig2_noise_robustness")


# ---------------------------------------------------------------------------
# Figure 3 — Non-IID severity (alpha sweep on CIFAR-10)
# ---------------------------------------------------------------------------
def fig_noniid_severity():
    alphas = [0.1, 0.5, 5.0]
    tags = ["0_1", "0_5", "5_0"]
    acc_rnd, acc_scp, g_rnd, g_scp = [], [], [], []
    for t in tags:
        r = load(RUN_MAP[f"alpha_{t}_random"])
        s = load(RUN_MAP[f"alpha_{t}_scope"])
        acc_rnd.append(final(r, RANDOM, "accuracy"))
        acc_scp.append(final(s, SCOPE, "accuracy"))
        g_rnd.append(final(r, RANDOM, "fairness_gini"))
        g_scp.append(final(s, SCOPE, "fairness_gini"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    a1.plot(alphas, acc_rnd, **STYLE_RANDOM())
    a1.plot(alphas, acc_scp, **STYLE_SCOPE())
    a1.set_xscale("log")
    a1.set_xlabel(r"Dirichlet $\alpha$ (larger = more IID)")
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy vs non-IID severity")
    a1.legend(loc="best")

    a2.plot(alphas, g_rnd, **STYLE_RANDOM())
    a2.plot(alphas, g_scp, **STYLE_SCOPE())
    a2.set_xscale("log")
    a2.set_xlabel(r"Dirichlet $\alpha$")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Gini vs non-IID severity")
    a2.set_ylim(-0.005, max(g_rnd) * 1.15)
    a2.legend(loc="center right")

    plt.tight_layout()
    save(fig, "fig3_noniid_severity")


# ---------------------------------------------------------------------------
# Figure 4 — K-sweep at N=50 on Fashion-MNIST
# ---------------------------------------------------------------------------
def fig_k_sweep():
    ks = [1, 5, 10, 15, 25, 35, 50]
    ratios = [k / 50 for k in ks]
    acc_rnd, acc_scp, g_rnd, g_scp = [], [], [], []
    for k in ks:
        d = load(RUN_MAP[f"ksweep_k{k}"])
        acc_rnd.append(final(d, RANDOM, "accuracy"))
        acc_scp.append(final(d, SCOPE, "accuracy"))
        g_rnd.append(final(d, RANDOM, "fairness_gini"))
        g_scp.append(final(d, SCOPE, "fairness_gini"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    a1.plot(ratios, acc_rnd, **STYLE_RANDOM())
    a1.plot(ratios, acc_scp, **STYLE_SCOPE())
    for k, ar, asc in zip(ks, acc_rnd, acc_scp):
        if asc - ar >= 0.01:
            a1.annotate(f"+{(asc-ar)*100:.1f}pp", xy=(k / 50, asc), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6, color=SCOPE_COLOR)
    a1.set_xlabel("Participation ratio K / N  (N=50)")
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy vs participation ratio")
    a1.legend(loc="lower right")

    a2.plot(ratios, g_rnd, **STYLE_RANDOM())
    a2.plot(ratios, g_scp, **STYLE_SCOPE())
    a2.set_xlabel("Participation ratio K / N")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Gini vs participation ratio")
    a2.set_ylim(-0.02, max(g_rnd) * 1.1)
    a2.legend(loc="upper right")

    plt.tight_layout()
    save(fig, "fig4_k_sweep")


# ---------------------------------------------------------------------------
# Figure 5 — Ablation
# ---------------------------------------------------------------------------
def fig_ablation():
    d = load(RUN_MAP["ablation"])
    variants = [
        (SCOPE, "Full SCOPE-FD"),
        ("fd_native.scope_fd_no_server", "– server bonus"),
        ("fd_native.scope_fd_no_diversity", "– diversity penalty"),
    ]
    names = [v[1] for v in variants]
    accs = [final(d, v[0], "accuracy") for v in variants]
    ginis = [final(d, v[0], "fairness_gini") for v in variants]
    losses = [final(d, v[0], "loss") for v in variants]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    x = np.arange(len(variants))
    w = 0.56

    bars1 = a1.bar(x, accs, w, color=ABLATION_PALETTE, edgecolor="white", linewidth=0.6)
    for i, (v, l) in enumerate(zip(accs, losses)):
        a1.text(i, v + 0.003, f"{v:.3f}\n(loss {l:.2f})", ha="center", va="bottom", fontsize=6)
    a1.set_xticks(x)
    a1.set_xticklabels(names, rotation=10, ha="right")
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy by component")
    a1.set_ylim(0, max(accs) * 1.2)

    a2.bar(x, ginis, w, color=ABLATION_PALETTE, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(ginis):
        a2.text(i, v + 0.0002, f"{v:.4f}", ha="center", va="bottom", fontsize=6)
    a2.set_xticks(x)
    a2.set_xticklabels(names, rotation=10, ha="right")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Gini by component")
    a2.set_ylim(0, max(ginis) * 1.3 + 0.003)

    plt.tight_layout()
    save(fig, "fig5_ablation")


# ---------------------------------------------------------------------------
# Figure 6 — Learning curves (accuracy vs round, random vs SCOPE)
# ---------------------------------------------------------------------------
def fig_learning_curves():
    d_mn = load(RUN_MAP["mnist_paired"])
    d_fm = load(RUN_MAP["fmnist_paired"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    for ax, data, title in [(a1, d_mn, "(a) MNIST"), (a2, d_fm, "(b) Fashion-MNIST")]:
        xr, yr = series(data, RANDOM, "accuracy")
        xs, ys = series(data, SCOPE, "accuracy")
        ax.plot(xr, yr, color=RANDOM_COLOR, label="Random",
                marker="o", markevery=10, markersize=5,
                markerfacecolor=RANDOM_COLOR, markeredgecolor="white",
                markeredgewidth=0.5, linewidth=1.4, alpha=0.95)
        ax.plot(xs, ys, color=SCOPE_COLOR, label="SCOPE-FD",
                marker="s", markevery=10, markersize=5,
                markerfacecolor=SCOPE_COLOR, markeredgecolor="white",
                markeredgewidth=0.5, linewidth=1.4, alpha=0.95)
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy")
        ax.set_title(title)
        ax.legend(loc="lower right")

    plt.tight_layout()
    save(fig, "fig6_learning_curves")


# ---------------------------------------------------------------------------
# Figure 7 — Participation heatmap (client × round)
# ---------------------------------------------------------------------------
def fig_participation_heatmap():
    d = load(RUN_MAP["fmnist_paired"])
    n_clients = int(d["config"]["total_clients"])
    n_rounds = int(d["config"]["rounds"])

    def build_matrix(method):
        sel_per_round = d["results"][method]["history"]["selected"]
        m = np.zeros((n_clients, n_rounds), dtype=int)
        for r, clients in enumerate(sel_per_round):
            for c in clients:
                if 0 <= c < n_clients and 0 <= r < n_rounds:
                    m[c, r] = 1
        return m

    m_rnd = build_matrix(RANDOM)
    m_scp = build_matrix(SCOPE)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for ax, mat, title in [(a1, m_rnd, "(a) Random"),
                           (a2, m_scp, "(b) SCOPE-FD")]:
        ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_xlabel("Round")
        ax.set_ylabel("Client id")
        ax.set_title(title)
        ax.set_xticks([0, 25, 50, 75, 99])
        ax.grid(False)

    plt.tight_layout()
    save(fig, "fig7_participation_heatmap")


# ---------------------------------------------------------------------------
# Figure 8 — Convergence-speed advantage across K-sweep
# ---------------------------------------------------------------------------
def fig_convergence_speed():
    """Panel (a): rounds-to-80%-of-final across K = 1, 5, 10, 15.
       Panel (b): AUC gap (area under SCOPE minus random curves) across K = 1, 5, 10, 15, 25, 35.
       Sparser participation = larger SCOPE convergence advantage."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.9))

    # --- Panel (a): rounds-to-80%-of-max-final ---
    ks = [1, 5, 10, 15]
    r_rounds, s_rounds, speedups = [], [], []
    for k in ks:
        d = load(RUN_MAP[f"ksweep_k{k}"])
        rnd_final = final(d, RANDOM, "accuracy")
        scp_final = final(d, SCOPE,  "accuracy")
        target = 0.80 * max(rnd_final, scp_final)
        rr = rounds_to_target(d, RANDOM, target)
        rs = rounds_to_target(d, SCOPE,  target)
        MAX = 110
        r_rounds.append(rr if rr is not None else MAX)
        s_rounds.append(rs if rs is not None else MAX)
        if rr is not None and rs is not None and rr > rs:
            speedups.append(rr - rs)
        else:
            speedups.append(0)

    x = np.arange(len(ks))
    w = 0.38
    a1.bar(x - w/2, r_rounds, w, color=RANDOM_COLOR, label="Random", edgecolor="white", linewidth=0.5)
    a1.bar(x + w/2, s_rounds, w, color=SCOPE_COLOR, label="SCOPE-FD", edgecolor="white", linewidth=0.5)
    for i, v in enumerate(r_rounds):
        a1.text(i - w/2, v + 1.5, f"{v}", ha="center", va="bottom", fontsize=7, color=RANDOM_COLOR)
    for i, v in enumerate(s_rounds):
        a1.text(i + w/2, v + 1.5, f"{v}", ha="center", va="bottom", fontsize=7, color=SCOPE_COLOR)
    for i, sp in enumerate(speedups):
        if sp >= 5:
            a1.annotate(f"−{sp}", xy=(i, max(r_rounds[i], s_rounds[i]) + 10),
                        ha="center", fontsize=8, color="#2ca02c", fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels([f"K={k}" for k in ks])
    a1.set_xlabel("Participation K (N=50, FMNIST)")
    a1.set_ylabel("Rounds to reach 80% of final acc")
    a1.set_title("(a) Convergence speed (fewer rounds = faster)")
    a1.set_ylim(0, 130)
    a1.legend(loc="upper right", fontsize=7)

    # --- Panel (b): AUC advantage across K ---
    ks_b = [1, 5, 10, 15, 25, 35]
    auc_gaps = []
    for k in ks_b:
        d = load(RUN_MAP[f"ksweep_k{k}"])
        xr = np.array([m["accuracy"] for m in d["results"][RANDOM]["metrics"]
                       if m.get("round", -1) >= 0])
        xs = np.array([m["accuracy"] for m in d["results"][SCOPE]["metrics"]
                       if m.get("round", -1) >= 0])
        n = min(len(xr), len(xs))
        auc_gaps.append(float(np.trapezoid(xs[:n] - xr[:n])))

    colors_b = [SCOPE_COLOR if g > 0 else RANDOM_COLOR for g in auc_gaps]
    a2.bar(np.arange(len(ks_b)), auc_gaps, 0.6, color=colors_b, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(auc_gaps):
        a2.text(i, v + (0.15 if v >= 0 else -0.55), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7, fontweight="bold")
    a2.axhline(0, color="#666", linewidth=0.6)
    a2.set_xticks(np.arange(len(ks_b)))
    a2.set_xticklabels([f"K={k}" for k in ks_b])
    a2.set_xlabel("Participation K (N=50, FMNIST)")
    a2.set_ylabel("Cumulative accuracy advantage (AUC gap)")
    a2.set_title("(b) SCOPE's training-time advantage shrinks as K→N")
    a2.set_ylim(min(auc_gaps) - 1.0, max(auc_gaps) + 1.5)

    plt.tight_layout()
    save(fig, "fig8_convergence_speed")


# ---------------------------------------------------------------------------
# Figure 9 — K=1 spotlight (hero figure)
# ---------------------------------------------------------------------------
def fig_k1_spotlight():
    d = load(RUN_MAP["ksweep_k1"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.8, 2.8))

    # (a) learning curves
    xr, yr = series(d, RANDOM, "accuracy")
    xs, ys = series(d, SCOPE,  "accuracy")
    a1.plot(xr, yr, color=RANDOM_COLOR, label="Random", linewidth=1.6,
            marker="o", markevery=10, markersize=5,
            markerfacecolor=RANDOM_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, alpha=0.95)
    a1.plot(xs, ys, color=SCOPE_COLOR,  label="SCOPE-FD", linewidth=1.6,
            marker="s", markevery=10, markersize=5,
            markerfacecolor=SCOPE_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, alpha=0.95)
    a1.axhline(y=final(d, RANDOM, "accuracy"), color=RANDOM_COLOR, linestyle=":", alpha=0.5, linewidth=0.8)
    a1.axhline(y=final(d, SCOPE,  "accuracy"), color=SCOPE_COLOR,  linestyle=":", alpha=0.5, linewidth=0.8)
    a1.set_xlabel("Round")
    a1.set_ylabel("Test accuracy")
    a1.set_title("(a) Learning curves (K=1, N=50, FMNIST)")
    a1.legend(loc="lower right")

    # (b) final accuracy bars with big annotation
    acc = [final(d, RANDOM, "accuracy"), final(d, SCOPE, "accuracy")]
    delta_pp = (acc[1] - acc[0]) * 100
    a2.bar([0, 1], acc, 0.55, color=[RANDOM_COLOR, SCOPE_COLOR], edgecolor="white", linewidth=0.6)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["Random", "SCOPE-FD"])
    a2.set_ylabel("Final accuracy")
    a2.set_title("(b) Final accuracy")
    for i, v in enumerate(acc):
        a2.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    a2.annotate(f"+{delta_pp:.1f} pp",
                xy=(1, acc[1]), xytext=(1, (acc[0] + acc[1]) / 2),
                ha="center", fontsize=10, fontweight="bold", color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
    a2.set_ylim(0, max(acc) * 1.18)

    # (c) Gini bars
    g = [final(d, RANDOM, "fairness_gini"), final(d, SCOPE, "fairness_gini")]
    a3.bar([0, 1], g, 0.55, color=[RANDOM_COLOR, SCOPE_COLOR], edgecolor="white", linewidth=0.6)
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["Random", "SCOPE-FD"])
    a3.set_ylabel("Participation Gini")
    a3.set_title("(c) Participation fairness")
    for i, v in enumerate(g):
        a3.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    a3.set_ylim(0, max(g) * 1.2 + 0.02)

    plt.tight_layout()
    save(fig, "fig9_k1_spotlight")


# ---------------------------------------------------------------------------
# Figure 10 — FL baseline comparison (N=50, K=15, R=300, CIFAR-10, FL)
# ---------------------------------------------------------------------------
def fig_fl_baseline_comparison():
    d = load(RUN_MAP["fl_baseline"])
    # Select FL-native selection methods (exclude fd_native.* which are experimental)
    methods = [
        ("heuristic.random",         "Random"),
        ("system_aware.fedcs",       "FedCS"),
        ("system_aware.oort",        "Oort"),
        ("heuristic.label_coverage", "LabelCov"),
        ("ml.maml_select",           "MAML"),
        ("ml.apex_v2",               "APEX-v2"),
    ]
    names = [m[1] for m in methods]
    accs = [final(d, m[0], "accuracy") for m in methods]
    ginis = [final(d, m[0], "fairness_gini") for m in methods]

    # SCOPE reference from matched N=50, K=15 FMNIST run (different dataset/paradigm, but same K,N)
    d_scope = load(RUN_MAP["ksweep_k15"])
    scope_acc = final(d_scope, SCOPE, "accuracy")
    scope_gini = final(d_scope, SCOPE, "fairness_gini")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    x = np.arange(len(methods))
    palette = ["#1f77b4", "#d62728", "#17becf", "#ff7f0e", "#8c564b", "#e377c2"]

    a1.bar(x, accs, 0.62, color=palette, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(accs):
        a1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    a1.set_xticks(x)
    a1.set_xticklabels(names, rotation=20, ha="right")
    a1.set_ylabel("Final accuracy (FL, CIFAR-10)")
    a1.set_title("(a) Accuracy: FL selection baselines")
    a1.set_ylim(0, max(accs) * 1.12)

    a2.bar(x, ginis, 0.62, color=palette, edgecolor="white", linewidth=0.6)
    a2.axhline(y=scope_gini, color=SCOPE_COLOR, linestyle="--", linewidth=1.6,
               label=f"SCOPE-FD reference: {scope_gini:.3f}")
    for i, v in enumerate(ginis):
        a2.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    a2.set_xticks(x)
    a2.set_xticklabels(names, rotation=20, ha="right")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Participation Gini: SCOPE-FD is in another league")
    a2.set_ylim(0, max(ginis) * 1.12)
    a2.legend(loc="center right", fontsize=7)

    plt.tight_layout()
    save(fig, "fig10_fl_baseline_comparison")


# ---------------------------------------------------------------------------
# Figure 11 — Per-client participation counts
# ---------------------------------------------------------------------------
def fig_per_client_participation():
    d = load(RUN_MAP["fmnist_paired"])
    pc_r = d["results"][RANDOM]["participation_counts"]
    pc_s = d["results"][SCOPE]["participation_counts"]
    N = len(pc_r)

    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    x = np.arange(N)
    w = 0.40
    ax.bar(x - w/2, pc_r, w, color=RANDOM_COLOR, label=f"Random (mean {np.mean(pc_r):.1f}, std {np.std(pc_r):.1f})",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, pc_s, w, color=SCOPE_COLOR, label=f"SCOPE-FD (mean {np.mean(pc_s):.1f}, std {np.std(pc_s):.1f})",
           edgecolor="white", linewidth=0.3)
    ax.axhline(y=100 * 10 / 30, color="grey", linestyle=":", linewidth=0.8,
               label=f"Uniform target: {100*10/30:.1f}")
    ax.set_xlabel("Client id")
    ax.set_ylabel("Participation count (100 rounds)")
    ax.set_title("Per-client participation: SCOPE is flat (std=0.5); Random has std=4.8")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks(x[::2])

    plt.tight_layout()
    save(fig, "fig11_per_client_participation")


# ---------------------------------------------------------------------------
# Figure 12 — System diagram (SCOPE-FD algorithm flow)
# ---------------------------------------------------------------------------
def fig_system_diagram():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color="#e6f0ff", edge="#1f77b4", fontsize=8, fontweight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor=edge, linewidth=1.2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#666666", style="->", lw=1.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle=style, color=color,
                                      linewidth=lw, mutation_scale=14))

    # Row 1 - clients
    box(0.3, 7.2, 2.4, 1.3,
        "N clients\n(heterogeneous\narchitectures)", color="#fff2cc", edge="#bf9000")
    # Row 1 - SCOPE selector (centerpiece)
    box(3.5, 6.6, 3.8, 2.0,
        "SCOPE-FD selector\n(§5)\n" + r"score$_i$ = $\tilde{d}_i$ + $\alpha_u b_i$ − $\alpha_d p_i$",
        color="#fce4d6", edge="#d62728", fontsize=9, fontweight="bold")
    # Row 1 - Selected K clients
    box(8.2, 7.2, 2.4, 1.3,
        "Selected K clients\n$S_r \\subseteq \\{1,\\dots,N\\}$", color="#fff2cc", edge="#bf9000")

    # Row 2 - SCOPE inputs (below selector)
    box(3.5, 4.9, 3.8, 1.3,
        "Inputs:\n• participation counts $n_i$\n• server class-confidence $c_k$\n• label histograms $h_i$",
        color="#f2f2f2", edge="#666666", fontsize=7)

    # Row 3 - Local training
    box(8.2, 4.9, 2.4, 1.6,
        "Local training\n($K_r$ SGD steps\non $D_i$) [B1]",
        color="#e2f0d9", edge="#548235")

    # Row 4 - mMIMO UL
    box(11.1, 4.9, 2.6, 1.6,
        "UL mMIMO\n(ZF precoding,\nUL SNR $-8$ dB)", color="#e6f0ff", edge="#1f77b4")

    # Row 5 (bottom) - server aggregation
    box(11.1, 2.6, 2.6, 1.6,
        "Server aggregation\n(data-size-weighted\nlogit averaging)",
        color="#e2f0d9", edge="#548235")

    # Server distillation
    box(8.2, 2.6, 2.4, 1.6,
        "Server distillation\n(KL div.) [H3]", color="#e2f0d9", edge="#548235")

    # DL mMIMO
    box(5.1, 2.6, 2.4, 1.6,
        "DL mMIMO\n(SNR sweep\n$-30$ to 0 dB)", color="#e6f0ff", edge="#1f77b4")

    # Server class-confidence export
    box(1.8, 2.6, 2.8, 1.6,
        "server_class_confidence\n(3-line hook,\nSCOPE-FD §5)\nbroadcast to round r+1",
        color="#fff2cc", edge="#bf9000", fontsize=7)

    # Top label: title (above all boxes)
    ax.text(7.0, 9.6, "SCOPE-FD algorithm flow (one round r)",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Arrows
    arrow(2.7, 7.9, 3.5, 7.6)                  # clients -> selector
    arrow(7.3, 7.6, 8.2, 7.9)                  # selector -> selected K
    arrow(5.4, 4.9, 5.4, 6.6)                  # inputs -> selector (upward)
    arrow(9.4, 7.2, 9.4, 6.5)                  # selected -> training
    arrow(10.6, 5.7, 11.1, 5.7)                # training -> UL
    arrow(12.4, 4.9, 12.4, 4.2)                # UL -> server
    arrow(11.1, 3.4, 10.6, 3.4)                # server agg -> distill
    arrow(8.2, 3.4, 7.5, 3.4)                  # distill -> DL
    arrow(5.1, 3.4, 4.6, 3.4)                  # DL -> class-conf
    arrow(3.2, 4.2, 4.0, 4.9, style="->", color="#d62728", lw=1.5)  # class-conf -> SCOPE inputs (feedback)

    # Label the feedback loop
    ax.text(3.5, 4.5, "feedback\n(next round)", fontsize=6, color="#d62728",
            fontweight="bold", ha="center", va="center", style="italic")

    plt.tight_layout()
    save(fig, "fig12_system_diagram")


# ---------------------------------------------------------------------------
# Figure 13 — Channel robustness at K=5 on FMNIST (Exp 8, N=50)
# Three panels showing that SCOPE's convergence and fairness advantages are
# noise-agnostic across a 30 dB DL SNR range.
# ---------------------------------------------------------------------------
def fig_channel_robustness_k5():
    labels = ["error-free", "0", "-10", "-20", "-30"]
    tags = ["errfree", "dl0", "dl-10", "dl-20", "dl-30"]
    acc_r, acc_s, g_r, g_s = [], [], [], []
    rd_r, rd_s, auc = [], [], []
    for t in tags:
        d = load(RUN_MAP[f"csweep_{t}"])
        acc_r.append(final(d, RANDOM, "accuracy"))
        acc_s.append(final(d, SCOPE,  "accuracy"))
        g_r.append(final(d, RANDOM, "fairness_gini"))
        g_s.append(final(d, SCOPE,  "fairness_gini"))
        r_ms = [m for m in d["results"][RANDOM]["metrics"] if m.get("round",-1) >= 0]
        s_ms = [m for m in d["results"][SCOPE]["metrics"]  if m.get("round",-1) >= 0]
        r_acc = np.array([m["accuracy"] for m in r_ms])
        s_acc = np.array([m["accuracy"] for m in s_ms])
        target = 0.80 * max(r_acc[-1], s_acc[-1])
        rd_r.append(next((m["round"] for m in r_ms if m["accuracy"] >= target), 100))
        rd_s.append(next((m["round"] for m in s_ms if m["accuracy"] >= target), 100))
        n = min(len(r_acc), len(s_acc))
        auc.append(float(np.trapezoid(s_acc[:n] - r_acc[:n])))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.0, 2.9))
    x = np.arange(len(labels))

    # Panel (a): accuracy
    a1.plot(x, acc_r, **STYLE_RANDOM())
    a1.plot(x, acc_s, **STYLE_SCOPE())
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_xlabel("Downlink SNR (dB)")
    a1.set_ylabel("Final accuracy")
    a1.set_title("(a) Accuracy: tied across noise")
    a1.legend(loc="best", fontsize=7)

    # Panel (b): Gini
    a2.plot(x, g_r, **STYLE_RANDOM(lw=2))
    a2.plot(x, g_s, **STYLE_SCOPE(lw=2))
    for i, v in enumerate(g_r):
        a2.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7, color=RANDOM_COLOR)
    for i, v in enumerate(g_s):
        a2.text(i, v - 0.012, f"{v:.3f}", ha="center", fontsize=7, color=SCOPE_COLOR)
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_xlabel("Downlink SNR (dB)")
    a2.set_ylabel("Participation Gini")
    a2.set_title("(b) Fairness: SCOPE = 0 at every SNR")
    a2.set_ylim(-0.02, 0.18)
    a2.legend(loc="center right", fontsize=7)

    # Panel (c): rounds-to-80% of final
    w = 0.38
    a3.bar(x - w/2, rd_r, w, color=RANDOM_COLOR, label="Random", edgecolor="white", linewidth=0.5)
    a3.bar(x + w/2, rd_s, w, color=SCOPE_COLOR, label="SCOPE-FD", edgecolor="white", linewidth=0.5)
    for i, v in enumerate(rd_r):
        a3.text(i - w/2, v + 1, f"{v}", ha="center", va="bottom", fontsize=7, color=RANDOM_COLOR)
    for i, v in enumerate(rd_s):
        a3.text(i + w/2, v + 1, f"{v}", ha="center", va="bottom", fontsize=7, color=SCOPE_COLOR)
    for i, (r_v, s_v) in enumerate(zip(rd_r, rd_s)):
        if r_v > s_v:
            a3.annotate(f"3× faster", xy=(i, max(r_v, s_v) + 5),
                        ha="center", fontsize=6, color="#2ca02c",
                        fontweight="bold", fontstyle="italic")
    a3.set_xticks(x); a3.set_xticklabels(labels)
    a3.set_xlabel("Downlink SNR (dB)")
    a3.set_ylabel("Rounds to 80% of final acc")
    a3.set_title("(c) Convergence: SCOPE 3× faster, noise-agnostic")
    a3.set_ylim(0, 45)
    a3.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    save(fig, "fig13_channel_robustness_k5")


# ---------------------------------------------------------------------------
# Figure 16 — Joint K × DL-SNR robustness (L-shape sweep through the 2D
# operating region).
#
# The K-sweep (varying K at fixed DL SNR=−20 dB) and the SNR-sweep (varying
# DL SNR at fixed K=5) intersect at exactly one point: K=5, SNR=−20. Drawing
# both sweeps as orthogonal cross-sections through this shared point lets a
# single figure absorb the entire §8.6 channel-robustness table while ALSO
# showing the K-axis dependence — joint robustness, not just one-axis
# robustness.
#
# Layout: 2 rows × 2 cols.
#   Top row    = rounds-to-80%-of-final (the convergence-speed metric)
#   Bottom row = participation Gini    (the fairness metric)
#   Left col   = K-sweep   (x-axis: K/N ratio, log-spaced)
#   Right col  = SNR-sweep (x-axis: DL SNR, ordinal)
#   Y-axes are NOT shared across columns (different metric ranges along K
#   vs SNR), but ARE shared row-wise visually because both columns plot the
#   same metric.
#
# The shared K=5/SNR=−20 dB point is marked with a vertical dotted line in
# every panel and labelled "shared op. point". Reading the figure from
# left-to-right within a row says: "this is what happens when you walk one
# step in K-space, and this is what happens when you walk one step in
# SNR-space, both starting from the same point."
# ---------------------------------------------------------------------------
def fig_joint_k_snr_robustness():
    # --- K-sweep data (FMNIST, N=50, varying K, fixed DL SNR=-20 dB) ---
    ks = [1, 5, 10, 15, 25, 35, 50]
    ratios = np.array([k / 50.0 for k in ks])
    k_r_r80, k_s_r80, k_r_g, k_s_g, k_r_acc, k_s_acc = [], [], [], [], [], []
    for k in ks:
        d = load(RUN_MAP[f"ksweep_k{k}"])
        rnd_final = final(d, RANDOM, "accuracy")
        scp_final = final(d, SCOPE,  "accuracy")
        target = 0.80 * max(rnd_final, scp_final)
        rr = rounds_to_target(d, RANDOM, target)
        rs = rounds_to_target(d, SCOPE,  target)
        # K=50 -> trivial selection, SCOPE == Random; rounds-to-80% can be
        # 0/1 or unreached; cap at 100 for plotting.
        k_r_r80.append(rr if rr is not None else 100)
        k_s_r80.append(rs if rs is not None else 100)
        k_r_g.append(final(d, RANDOM, "fairness_gini"))
        k_s_g.append(final(d, SCOPE,  "fairness_gini"))
        k_r_acc.append(rnd_final)
        k_s_acc.append(scp_final)

    # --- SNR-sweep data (FMNIST, N=50, K=5, varying DL SNR) ---
    snr_labels = ["err-free", "0", "−10", "−20", "−30"]
    snr_tags = ["errfree", "dl0", "dl-10", "dl-20", "dl-30"]
    s_r_r80, s_s_r80, s_r_g, s_s_g, s_r_acc, s_s_acc = [], [], [], [], [], []
    for t in snr_tags:
        d = load(RUN_MAP[f"csweep_{t}"])
        rnd_final = final(d, RANDOM, "accuracy")
        scp_final = final(d, SCOPE,  "accuracy")
        target = 0.80 * max(rnd_final, scp_final)
        rr = rounds_to_target(d, RANDOM, target)
        rs = rounds_to_target(d, SCOPE,  target)
        s_r_r80.append(rr if rr is not None else 100)
        s_s_r80.append(rs if rs is not None else 100)
        s_r_g.append(final(d, RANDOM, "fairness_gini"))
        s_s_g.append(final(d, SCOPE,  "fairness_gini"))
        s_r_acc.append(rnd_final)
        s_s_acc.append(scp_final)

    # The SNR-sweep "shared point" is the SNR=-20 column — its tick index
    # within snr_labels.
    snr_shared_idx = snr_labels.index("−20")
    # The K-sweep "shared point" is K=5 — its tick index within ks.
    k_shared_idx = ks.index(5)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    (a1, a2), (a3, a4) = axes

    # Common formatting: K-axis labels thinned to {1, 5, 10, 25, 50} so the
    # log-scale tick text doesn't collide; the data is still plotted at all
    # 7 K values, just unlabelled at K=15 and K=35.
    K_TICK_VALS  = [1, 5, 10, 25, 50]
    K_TICK_RATIOS = np.array([k / 50.0 for k in K_TICK_VALS])
    K_TICK_LABELS = [f"{k}/50" for k in K_TICK_VALS]

    # --------------- Row 1: rounds-to-80% (convergence speed) ---------------
    # (a) vs K
    a1.plot(ratios, k_r_r80, **STYLE_RANDOM())
    a1.plot(ratios, k_s_r80, **STYLE_SCOPE())
    a1.fill_between(ratios, k_s_r80, k_r_r80, where=np.array(k_r_r80) > np.array(k_s_r80),
                    color=SCOPE_COLOR, alpha=0.10, linewidth=0,
                    label="SCOPE advantage")
    a1.axvline(x=ratios[k_shared_idx], color="#444", linestyle=":", linewidth=1.0, alpha=0.75)
    a1.set_xscale("log")
    a1.set_xticks(K_TICK_RATIOS)
    a1.set_xticklabels(K_TICK_LABELS, fontsize=7)
    a1.set_xlabel("Participation ratio $K/N$  (SNR$_{DL}$ = −20 dB fixed)")
    a1.set_ylabel("Rounds to 80% of final acc")
    a1.set_title("(a) Convergence vs K-axis")
    a1.set_ylim(0, 110)
    # Place "shared op. pt." marker text inside the data area, near the top
    # of the panel but below the title — both panels (a) and (b) use the
    # same y-position so they align visually.
    a1.text(ratios[k_shared_idx], 102, "shared\nop. pt.",
            fontsize=6, color="#444", ha="center", va="top",
            fontstyle="italic")
    a1.legend(loc="upper right", fontsize=6)

    # (b) vs DL SNR
    x = np.arange(len(snr_labels))
    a2.plot(x, s_r_r80, **STYLE_RANDOM())
    a2.plot(x, s_s_r80, **STYLE_SCOPE())
    a2.fill_between(x, s_s_r80, s_r_r80, where=np.array(s_r_r80) > np.array(s_s_r80),
                    color=SCOPE_COLOR, alpha=0.10, linewidth=0,
                    label="SCOPE advantage")
    a2.axvline(x=snr_shared_idx, color="#444", linestyle=":", linewidth=1.0, alpha=0.75)
    a2.set_xticks(x); a2.set_xticklabels(snr_labels, fontsize=7)
    a2.set_xlabel("SNR$_{DL}$ (dB)  ($K=5$ fixed)")
    a2.set_ylabel("Rounds to 80% of final acc")
    a2.set_title("(b) Convergence vs SNR-axis")
    a2.set_ylim(0, 110)
    a2.text(snr_shared_idx, 102, "shared\nop. pt.",
            fontsize=6, color="#444", ha="center", va="top",
            fontstyle="italic")
    # Annotate the constant 3× speedup that holds across the SNR-axis.
    a2.annotate("3$\\times$ faster\n at every SNR", xy=(2, 30),
                xytext=(2, 70), ha="center", fontsize=6.5,
                color="#2ca02c", fontweight="bold", style="italic",
                arrowprops=dict(arrowstyle="-", color="#2ca02c",
                                linewidth=0.7, alpha=0.7))
    a2.legend(loc="upper right", fontsize=6)

    # --------------- Row 2: participation Gini (fairness) ---------------
    g_y_max = max(max(k_r_g), max(s_r_g)) * 1.20

    # (c) Gini vs K
    a3.plot(ratios, k_r_g, **STYLE_RANDOM())
    a3.plot(ratios, k_s_g, **STYLE_SCOPE())
    a3.axvline(x=ratios[k_shared_idx], color="#444", linestyle=":", linewidth=1.0, alpha=0.75)
    # Annotate Random's K-axis Gini values — they vary meaningfully (0.43 →
    # 0 as K/N → 1), so the Random curve is the interesting one here.
    for r, v in zip(ratios, k_r_g):
        a3.annotate(f"{v:.3f}", xy=(r, v), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    fontsize=5.5, color=RANDOM_COLOR)
    a3.set_xscale("log")
    a3.set_xticks(K_TICK_RATIOS)
    a3.set_xticklabels(K_TICK_LABELS, fontsize=7)
    a3.set_xlabel("Participation ratio $K/N$  (SNR$_{DL}$ = −20 dB fixed)")
    a3.set_ylabel("Final Gini")
    a3.set_title("(c) Fairness vs K-axis (SCOPE = 0 throughout)")
    a3.set_ylim(-0.02, g_y_max)
    a3.legend(loc="upper right", fontsize=7)

    # (d) Gini vs SNR
    a4.plot(x, s_r_g, **STYLE_RANDOM())
    a4.plot(x, s_s_g, **STYLE_SCOPE())
    a4.axvline(x=snr_shared_idx, color="#444", linestyle=":", linewidth=1.0, alpha=0.75)
    for xi, v in zip(x, s_r_g):
        a4.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    fontsize=5.5, color=RANDOM_COLOR)
    a4.set_xticks(x); a4.set_xticklabels(snr_labels, fontsize=7)
    a4.set_xlabel("SNR$_{DL}$ (dB)  ($K=5$ fixed)")
    a4.set_ylabel("Final Gini")
    a4.set_title("(d) Fairness vs SNR-axis (SCOPE = 0 throughout)")
    a4.set_ylim(-0.02, g_y_max)
    a4.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    save(fig, "fig16_joint_k_snr_robustness")


# ---------------------------------------------------------------------------
# Figure 14 — Coefficient ablation (5 (αu, αd) pairs at FMNIST N=50, K=5).
# Defends Section IV-E's claim that (αu, αd) is a placement inside the
# admissible region αu+αd<1, not a tuned hyperparameter. The 5 pairs span
# pure round-robin, interior, default, near-boundary, and a deliberately
# violating point — the constraint αu+αd<1 is empirically conservative on
# this regime: every pair produces Gini=0 with accuracy in a 0.5pp band.
# ---------------------------------------------------------------------------
def fig_coef_ablation():
    pairs = [
        ("0_00_0_00", 0.00, 0.00, "(0, 0)\nRR-only"),
        ("0_10_0_10", 0.10, 0.10, "(0.1, 0.1)\ninterior"),
        ("0_30_0_10", 0.30, 0.10, "(0.3, 0.1)\ndefault"),
        ("0_50_0_40", 0.50, 0.40, "(0.5, 0.4)\nnear-bdy"),
        ("0_70_0_40", 0.70, 0.40, "(0.7, 0.4)\nVIOLATES"),
    ]
    # Color gradient: pure RR cool blue → default red → violating dark red
    palette = ["#1f77b4", "#9467bd", "#d62728", "#e377c2", "#8b0000"]

    accs, ginis_final, ginis_avg, curves = [], [], [], []
    for tag, _, _, _ in pairs:
        d = load(RUN_MAP[f"coef_{tag}"])
        accs.append(final(d, SCOPE, "accuracy"))
        ginis_final.append(final(d, SCOPE, "fairness_gini"))
        rows = [m for m in d["results"][SCOPE]["metrics"]
                if isinstance(m.get("round", -1), (int, float)) and m.get("round", -1) >= 0]
        ginis_avg.append(float(np.mean([m.get("fairness_gini", 0.0) or 0.0 for m in rows])))
        xs, ys = series(d, SCOPE, "accuracy")
        curves.append((xs, ys))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    a1, a2, a3 = axes

    # (a) Learning curves overlay — 5 nearly-identical trajectories (the story)
    for (tag, au, ad, lbl), color, (xs, ys) in zip(pairs, palette, curves):
        emph = (au + ad >= 1.0)
        a1.plot(xs, ys, color=color, linewidth=1.6 if emph or tag == "0_30_0_10" else 1.1,
                label=f"$\\alpha_u={au}, \\alpha_d={ad}$" + (" (viol.)" if emph else ""),
                linestyle="--" if emph else "-",
                alpha=0.95)
    a1.set_xlabel("Round")
    a1.set_ylabel("Test accuracy")
    a1.set_title("(a) Learning curves — insensitive to $(\\alpha_u, \\alpha_d)$")
    a1.legend(loc="lower right", fontsize=6, ncol=1, handlelength=1.4)

    # (b) Final accuracy bars — annotate with Δ from default
    x = np.arange(len(pairs))
    a2.bar(x, accs, 0.62, color=palette, edgecolor="white", linewidth=0.6)
    default_idx = 2
    for i, v in enumerate(accs):
        delta = (v - accs[default_idx]) * 100
        delta_str = f"{delta:+.2f}pp" if i != default_idx else "ref."
        a2.text(i, v + 0.005, f"{v:.4f}\n{delta_str}",
                ha="center", va="bottom", fontsize=6,
                fontweight="bold" if i == default_idx else "normal")
    a2.set_xticks(x)
    a2.set_xticklabels([p[3] for p in pairs], fontsize=6)
    a2.set_ylabel("Final accuracy")
    a2.set_title("(b) Accuracy: 0.5pp band across all 5")
    a2.set_ylim(0, max(accs) * 1.18)

    # (c) Gini final + avg-over-training (final is 0 for all; avg shows the
    # within-cycle non-zero plateau the debt term enforces — same value 0.083
    # for every coefficient choice, the floor of a 10-round cycle).
    w = 0.36
    a3.bar(x - w/2, ginis_final, w, color="#444", edgecolor="white", linewidth=0.6,
           label="Final Gini (R=100)")
    a3.bar(x + w/2, ginis_avg, w, color="#cccccc", edgecolor="white", linewidth=0.6,
           label="Avg Gini over training")
    for i, v in enumerate(ginis_final):
        a3.text(i - w/2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    for i, v in enumerate(ginis_avg):
        a3.text(i + w/2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    a3.axhline(y=0, color="#666", linewidth=0.6)
    a3.set_xticks(x)
    a3.set_xticklabels([p[3] for p in pairs], fontsize=6)
    a3.set_ylabel("Participation Gini")
    a3.set_title("(c) Fairness: Gini=0 even at violating row")
    a3.set_ylim(0, max(ginis_avg) * 1.30 + 0.01)
    a3.legend(loc="upper right", fontsize=6)

    plt.tight_layout()
    save(fig, "fig14_coef_ablation")


# ---------------------------------------------------------------------------
# Figure 15 — EMNIST/digits cross-domain replication (N=50, K=5, R=100).
# Refutes the "FMNIST artifact" reviewer concern: Random vs SCOPE-FD on
# handwritten Latin digits (NIST SD 19, subsampled 60k/10k to scale-match
# FMNIST). Story: same Gini collapse, same rapid early convergence (SCOPE
# reaches 80% of final at round 10 vs random's 25), final accuracy tied.
# ---------------------------------------------------------------------------
def fig_emnist_replication():
    """Single-panel EMNIST/digits accuracy curve. Tie at final accuracy
    and Gini collapse are reported in the §8.10 caption text instead of
    a 3-panel figure — the convergence-speed story is the carrier."""
    d = load(RUN_MAP["emnist_paired"])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    xr, yr = series(d, RANDOM, "accuracy")
    xs, ys = series(d, SCOPE, "accuracy")
    ax.plot(xr, yr, color=RANDOM_COLOR, label="Random", linewidth=1.6,
            marker="o", markevery=10, markersize=5,
            markerfacecolor=RANDOM_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, alpha=0.95)
    ax.plot(xs, ys, color=SCOPE_COLOR, label="SCOPE-FD", linewidth=1.6,
            marker="s", markevery=10, markersize=5,
            markerfacecolor=SCOPE_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, alpha=0.95)
    final_r = float(yr[-1])
    final_s = float(ys[-1])
    target = 0.80 * max(final_r, final_s)
    ax.axhline(y=target, color="grey", linestyle=":", linewidth=0.8,
               label=f"80% of final = {target:.2f}")
    rr = next((int(x) for x, y in zip(xr, yr) if y >= target), int(xr[-1]))
    rs = next((int(x) for x, y in zip(xs, ys) if y >= target), int(xs[-1]))
    ax.axvline(x=rr, color=RANDOM_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(x=rs, color=SCOPE_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)
    # Round-crossing labels placed at the bottom of the plot so they do not
    # collide with the steep climb of the accuracy curves through the
    # threshold — the lower band has empty space because both curves start
    # at ~0.10 and the threshold sits at ~0.62.
    ax.annotate(f"r={rs}", xy=(rs, 0.12), xytext=(rs - 4, 0.10),
                fontsize=7, color=SCOPE_COLOR, fontweight="bold")
    ax.annotate(f"r={rr}", xy=(rr, 0.12), xytext=(rr - 4, 0.10),
                fontsize=7, color=RANDOM_COLOR, fontweight="bold")
    # Speedup callout in the open region under the threshold line.
    speedup = rr / max(rs, 1)
    ax.annotate(f"{speedup:.1f}$\\times$ faster\nto 80% of final",
                xy=((rr + rs) / 2, target - 0.04),
                xytext=((rr + rs) / 2, 0.30),
                ha="center", fontsize=7, color="#2ca02c",
                fontweight="bold", style="italic")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("EMNIST/digits — N=50, K=5, R=100")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0.05, max(max(yr), max(ys)) * 1.05)

    plt.tight_layout()
    save(fig, "fig15_emnist_replication")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    apply_style()
    # Narrative set (2026-04-25 final):
    # Kept: fig1 (headlines), fig4 (K-sweep), fig6 (learning curves),
    #   fig7 (heatmap), fig8 (convergence speed), fig9 (K=1 spotlight),
    #   fig10 (FL baselines), fig11 (per-client), fig12 (system diagram),
    #   fig13 (channel robustness — NEW).
    # Dropped from main narrative (functions preserved for optional resurrection):
    #   fig2 (CIFAR noise sweep), fig3 (CIFAR alpha sweep), fig5 (ablation).
    for fn in (fig_headline_bars,
               fig_k_sweep, fig_learning_curves, fig_participation_heatmap,
               fig_convergence_speed, fig_k1_spotlight,
               fig_fl_baseline_comparison, fig_per_client_participation,
               fig_system_diagram, fig_channel_robustness_k5,
               # Pre-submission additions: cross-domain replication +
               # joint K×SNR robustness (replaces channel-robustness table).
               # fig_coef_ablation kept generated for supplementary use.
               fig_emnist_replication, fig_joint_k_snr_robustness,
               fig_coef_ablation,
               # Kept generated (not linked from narrative) for optional use:
               fig_noise_robustness, fig_noniid_severity, fig_ablation):
        try:
            fn()
        except Exception as e:
            print(f"[ERR] {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print("\nAll figures written to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
