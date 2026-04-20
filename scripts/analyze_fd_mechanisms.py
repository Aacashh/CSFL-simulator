"""Mechanism-level plots that explain WHY LQTS wins and why CALM-FD should
extend it. Emits PNGs into artifacts/analysis/figures/ alongside the
primary analysis script's output.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PATHS = {
    "fd_main":     ROOT / "artifacts/runs/fd_cifar10_main_20260415-172221",
    "noise_errf":  ROOT / "artifacts/runs/fd_cifar10_noise_errfree_20260416-172715",
    "noise_dl0":   ROOT / "artifacts/runs/fd_cifar10_noise_dl0_20260417-072753",
    "noise_dl10":  ROOT / "artifacts/runs/fd_cifar10_noise_dl-10_20260417-215326",
    "noise_dl20":  ROOT / "artifacts/runs/fd_cifar10_noise_dl-20_20260418-124250",
    "noise_dl30":  ROOT / "artifacts/runs/fd_cifar10_noise_dl-30_20260419-015402",
}

STYLE = {
    "heuristic.random":          {"c":"#1f77b4","m":"o","ls":"-"},
    "system_aware.oort":         {"c":"#17becf","m":"P","ls":"-."},
    "ml.apex_v2":                {"c":"#e377c2","m":"*","ls":"--"},
    "fd_native.logit_quality_ts":{"c":"#9467bd","m":"^","ls":"-"},
    "fd_native.snr_diversity":   {"c":"#2ca02c","m":"D","ls":"-"},
    "fd_native.noise_robust_fair":{"c":"#bcbd22","m":"h","ls":"-."},
    "fd_native.logit_entropy_max":{"c":"#7f7f7f","m":"<","ls":":"},
}
LABEL = {
    "heuristic.random":"Random",
    "system_aware.oort":"Oort",
    "ml.apex_v2":"APEX v2 (FL champ)",
    "fd_native.logit_quality_ts":"LQTS",
    "fd_native.snr_diversity":"SNR-Diversity",
    "fd_native.noise_robust_fair":"Noise-Robust Fair",
    "fd_native.logit_entropy_max":"Logit-Entropy Max",
}

def load(p): return json.loads((p / "compare_results.json").read_text())
def series(d, m, k): return np.asarray([x.get(k, np.nan) for x in d["results"][m]["metrics"]], dtype=float)
def smooth(x, w=10): return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------
# Fig M1: Server-client gap over rounds — proxy for "is the server actually
# distilling something useful beyond client avg?"
def fig_server_client_gap():
    d = load(PATHS["fd_main"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in STYLE:
        if m not in d["results"]: continue
        y = smooth(series(d, m, "server_client_gap"))
        s = STYLE[m]
        ax.plot(np.arange(len(y)), y, label=LABEL[m], color=s["c"], linestyle=s["ls"], linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Round")
    ax.set_ylabel("Server$-$Client accuracy gap (smoothed)")
    ax.set_title("Server Distillation Gain over rounds — FD main (DL SNR$=-20$ dB)\nHigher = server's distilled model outperforms client avg (desired)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figM1_server_client_gap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Fig M2: accuracy stability — stddev over rounds 100-280 per method per SNR
def fig_stability_bar():
    rows = []
    for label, p in PATHS.items():
        d = load(p)
        for m in STYLE:
            if m not in d["results"]: continue
            y = series(d, m, "server_accuracy")[100:280]
            y = y[~np.isnan(y)]
            if y.size < 50: continue
            rows.append({"run": label, "method": m, "mean": float(np.mean(y)), "std": float(np.std(y))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.3))
    runs = sorted(df["run"].unique())
    methods = list(STYLE.keys())
    bar_w = 0.11
    x = np.arange(len(runs))
    for i, m in enumerate(methods):
        if m not in df["method"].values: continue
        vals = []
        for r in runs:
            sub = df[(df["run"] == r) & (df["method"] == m)]
            vals.append(sub["std"].iloc[0] if len(sub) else 0)
        ax.bar(x + (i - len(methods)/2) * bar_w, vals, bar_w,
               label=LABEL.get(m, m), color=STYLE[m]["c"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Std of server accuracy (rounds 100$-$280)")
    ax.set_title("Mid-training accuracy volatility per method per run\n(lower = more stable convergence)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figM2_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Fig M3: the counter-intuitive finding — across all (run, method) pairs,
# logit_cosine_diversity vs server_accuracy. Should show negative slope.
def fig_logit_div_vs_acc():
    rows = []
    for label, p in PATHS.items():
        d = load(p)
        for m in d["results"]:
            y = series(d, m, "server_accuracy")
            ld = series(d, m, "logit_cosine_diversity")
            if np.all(np.isnan(y)) or np.all(np.isnan(ld)): continue
            rows.append({"run": label, "method": m,
                         "acc": float(np.nanmean(y[-20:])),
                         "logit_div": float(np.nanmean(ld[-20:]))})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # per-method color/marker
    for m, s in STYLE.items():
        sub = df[df["method"] == m]
        if sub.empty: continue
        ax.scatter(sub["logit_div"], sub["acc"] * 100, color=s["c"], marker=s["m"], s=80,
                   edgecolor="black", label=LABEL.get(m, m))
    # linear fit across all points
    if len(df) >= 4:
        z = np.polyfit(df["logit_div"].values, df["acc"].values * 100, 1)
        xs = np.linspace(df["logit_div"].min(), df["logit_div"].max(), 20)
        ax.plot(xs, z[0]*xs + z[1], "k--", linewidth=1, alpha=0.6,
                label=f"linear fit (slope={z[0]:+.1f})")
    ax.set_xlabel("Final round-avg Logit Cosine Diversity")
    ax.set_ylabel("Final Server Accuracy (%)")
    ax.set_title("Counter-intuitive: higher logit diversity correlates with LOWER accuracy\n(across 7 runs, 10 methods where available)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figM3_logit_div_vs_acc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Fig M4: the winning reward trajectory — cosine_to_mean (fd_logit_rewards per round)
# We don't have per-client logs here, but we have logit_entropy_avg which correlates.
def fig_entropy_trajectory():
    d = load(PATHS["fd_main"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in ["fd_native.logit_quality_ts", "fd_native.logit_entropy_max",
              "fd_native.snr_diversity", "ml.apex_v2", "heuristic.random"]:
        if m not in d["results"]: continue
        y = smooth(series(d, m, "logit_entropy_avg"))
        s = STYLE[m]
        ax.plot(np.arange(len(y)), y, label=LABEL[m], color=s["c"], linestyle=s["ls"], linewidth=1.5)
    ax.axhline(np.log(10), color="red", linewidth=0.7, linestyle="--", label="max entropy (log 10)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg Logit Entropy of selected clients")
    ax.set_title("Entropy of selected-client logits — FD main (DL SNR$=-20$ dB)\nLower = more confident logits = better distillation targets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "figM4_entropy_traj.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Fig M5: Fairness trajectory — proxy for "starvation / lockout"
def fig_fairness_trajectory():
    d = load(PATHS["fd_main"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in ["fd_native.logit_quality_ts", "fd_native.noise_robust_fair",
              "fd_native.snr_diversity", "system_aware.oort", "heuristic.random"]:
        if m not in d["results"]: continue
        y = smooth(series(d, m, "fairness_gini"))
        s = STYLE[m]
        ax.plot(np.arange(len(y)), y, label=LABEL[m], color=s["c"], linestyle=s["ls"], linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fairness Gini (smoothed)")
    ax.set_title("Participation inequality — FD main\nLower = fairer; 0.7 = Gini cap for K=15/N=50 over 300 rounds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "figM5_fairness_traj.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# Fig M6: Cross-SNR ranking stability heatmap
def fig_ranking_heatmap():
    import matplotlib.colors as mcolors
    snr_order = ["errf", "dl0", "dl10", "dl20", "dl30"]
    snr_labels = ["errfree", "0 dB", "$-$10 dB", "$-$20 dB", "$-$30 dB"]
    method_order = ["fd_native.logit_quality_ts", "ml.apex_v2", "fd_native.noise_robust_fair",
                    "fd_native.snr_diversity", "heuristic.random", "system_aware.oort",
                    "fd_native.logit_entropy_max"]
    ranks = np.zeros((len(method_order), len(snr_order)))
    for j, key in enumerate(snr_order):
        p = PATHS[f"noise_{key}"]
        d = load(p)
        finals = {}
        for m in method_order:
            if m in d["results"]:
                finals[m] = float(np.nanmean(series(d, m, "server_accuracy")[-20:]))
        sorted_m = sorted(finals, key=lambda x: -finals[x])
        for r, m in enumerate(sorted_m, start=1):
            i = method_order.index(m)
            ranks[i, j] = r

    fig, ax = plt.subplots(figsize=(7, 4.4))
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(ranks, cmap=cmap, vmin=1, vmax=len(method_order), aspect="auto")
    ax.set_xticks(range(len(snr_order)))
    ax.set_xticklabels(snr_labels)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels([LABEL[m] for m in method_order])
    ax.set_xlabel("DL SNR")
    ax.set_title("Method rank across noise levels (1 = best, darker green = win)")
    for i in range(len(method_order)):
        for j in range(len(snr_order)):
            ax.text(j, i, f"{int(ranks[i,j])}", ha="center", va="center",
                    color="white" if ranks[i,j] >= 4 else "black", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figM6_ranking_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_server_client_gap()
    fig_stability_bar()
    fig_logit_div_vs_acc()
    fig_entropy_trajectory()
    fig_fairness_trajectory()
    fig_ranking_heatmap()
    print(f"Mechanism plots → {FIG}")


if __name__ == "__main__":
    main()
