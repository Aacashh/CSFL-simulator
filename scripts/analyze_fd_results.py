"""Analyse recent FD experiment artefacts and emit paper-ready CSVs + PNG figures.

Inputs (read-only): artifacts/runs/fd_cifar10_main_20260415-172221, fl_cifar10_baseline_20260416-123215,
fd_cifar10_noise_{errfree,dl0,dl-10,dl-20,dl-30}_2026041*.

Outputs: artifacts/analysis/{master_table.csv, ranking_fl_vs_fd.csv, noise_sweep_summary.csv,
fd_main_method_table.csv, figures/*.png}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
OUT = ROOT / "artifacts" / "analysis"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SMOOTH_WINDOW = 10
FINAL_ACC_TAIL = 20  # average last N rounds for robust final accuracy

# Recent-only paths (timestamps from the Apr 14-19 shell run)
PATHS = {
    "fd_main": RUNS / "fd_cifar10_main_20260415-172221",
    "fl_baseline": RUNS / "fl_cifar10_baseline_20260416-123215",
    "noise_errfree": RUNS / "fd_cifar10_noise_errfree_20260416-172715",
    "noise_dl0": RUNS / "fd_cifar10_noise_dl0_20260417-072753",
    "noise_dl-10": RUNS / "fd_cifar10_noise_dl-10_20260417-215326",
    "noise_dl-20": RUNS / "fd_cifar10_noise_dl-20_20260418-124250",
    "noise_dl-30": RUNS / "fd_cifar10_noise_dl-30_20260419-015402",
}

NOISE_LEVELS = [
    ("errfree", "noise_errfree", float("inf")),
    ("0 dB", "noise_dl0", 0.0),
    ("-10 dB", "noise_dl-10", -10.0),
    ("-20 dB", "noise_dl-20", -20.0),
    ("-30 dB", "noise_dl-30", -30.0),
]

METHOD_DISPLAY = {
    "heuristic.random": "Random (FedAvg)",
    "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort",
    "heuristic.label_coverage": "LabelCov",
    "ml.maml_select": "MAML-Select",
    "ml.apex_v2": "APEX v2",
    "fd_native.snr_diversity": "SNR-Diversity",
    "fd_native.logit_quality_ts": "Logit-Quality TS",
    "fd_native.noise_robust_fair": "Noise-Robust Fair",
    "fd_native.logit_entropy_max": "Logit-Entropy Max",
}

# Visual map (extends FD_experiments.md table)
METHOD_STYLE = {
    "heuristic.random":          {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "system_aware.fedcs":        {"color": "#d62728", "marker": "s", "ls": "--"},
    "system_aware.oort":         {"color": "#17becf", "marker": "P", "ls": "-."},
    "heuristic.label_coverage":  {"color": "#ff7f0e", "marker": "X", "ls": ":"},
    "ml.maml_select":            {"color": "#8c564b", "marker": "v", "ls": "-"},
    "ml.apex_v2":                {"color": "#e377c2", "marker": "*", "ls": "--"},
    "fd_native.snr_diversity":   {"color": "#2ca02c", "marker": "D", "ls": "-"},
    "fd_native.logit_quality_ts":{"color": "#9467bd", "marker": "^", "ls": "--"},
    "fd_native.noise_robust_fair":{"color": "#bcbd22", "marker": "h", "ls": "-."},
    "fd_native.logit_entropy_max":{"color": "#7f7f7f", "marker": "<", "ls": ":"},
}


def load(path: Path) -> dict[str, Any]:
    return json.loads((path / "compare_results.json").read_text())


def smooth(arr: list[float], w: int = SMOOTH_WINDOW) -> np.ndarray:
    s = pd.Series(arr).rolling(window=w, min_periods=1).mean().to_numpy()
    return s


def per_round(data: dict, method: str, key: str) -> np.ndarray:
    rounds = data["results"][method]["metrics"]
    return np.asarray([m.get(key, np.nan) for m in rounds], dtype=float)


def final_value(data: dict, method: str, key: str, tail: int = FINAL_ACC_TAIL) -> float:
    arr = per_round(data, method, key)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr[-tail:]))


def rounds_to(data: dict, method: str, target: float, key: str = "server_accuracy") -> int | None:
    arr = per_round(data, method, key)
    smoothed = smooth(arr.tolist(), w=5)
    hits = np.where(smoothed >= target)[0]
    return int(hits[0]) if hits.size else None


def primary_acc_key(data: dict, method: str) -> str:
    """server_accuracy if non-trivial (FD), else accuracy (FL)."""
    arr = per_round(data, method, "server_accuracy")
    if np.all(np.isnan(arr)) or np.nanmax(arr) <= 0.11:
        return "accuracy"
    return "server_accuracy"


# ---------- Master table ----------------------------------------------------

def build_master_table() -> pd.DataFrame:
    rows = []
    for run_key, path in PATHS.items():
        data = load(path)
        cfg = data["config"]
        for m in data["results"]:
            acc_key = primary_acc_key(data, m)
            row = {
                "run": run_key,
                "paradigm": cfg.get("paradigm", "fl"),
                "dataset": cfg.get("dataset"),
                "alpha": cfg.get("dirichlet_alpha"),
                "N": cfg.get("total_clients"),
                "K": cfg.get("clients_per_round"),
                "rounds": cfg.get("rounds"),
                "channel_noise": cfg.get("channel_noise", False),
                "ul_snr_db": cfg.get("ul_snr_db"),
                "dl_snr_db": cfg.get("dl_snr_db"),
                "method": m,
                "method_display": METHOD_DISPLAY.get(m, m),
                "acc_key_used": acc_key,
                "final_acc_tail20": final_value(data, m, acc_key),
                "final_client_acc_tail20": final_value(data, m, "accuracy"),
                "final_client_acc_std": final_value(data, m, "client_accuracy_std") if cfg.get("paradigm") == "fd" else float("nan"),
                "final_kl_div": final_value(data, m, "kl_divergence_avg"),
                "final_distill_loss": final_value(data, m, "distillation_loss_avg"),
                "final_eff_noise_var": final_value(data, m, "effective_noise_var"),
                "final_fairness_gini": final_value(data, m, "fairness_gini"),
                "final_participation_gini": final_value(data, m, "participation_gini"),
                "final_logit_cosine_div": final_value(data, m, "logit_cosine_diversity"),
                "final_logit_entropy_avg": final_value(data, m, "logit_entropy_avg"),
                "final_label_cov_ratio": final_value(data, m, "label_coverage_ratio"),
                "final_chan_quality_sel": final_value(data, m, "channel_quality_selected_avg"),
                "final_server_client_gap": final_value(data, m, "server_client_gap"),
                "total_logit_comm_kb": float(np.nansum(per_round(data, m, "logit_comm_kb"))),
                "total_fl_equiv_comm_mb": float(np.nansum(per_round(data, m, "fl_equiv_comm_mb"))),
                "comm_reduction_ratio_final": final_value(data, m, "comm_reduction_ratio"),
                "rounds_to_30pct": rounds_to(data, m, 0.30, acc_key),
                "rounds_to_50pct": rounds_to(data, m, 0.50, acc_key),
                "rounds_to_60pct": rounds_to(data, m, 0.60, acc_key),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "master_table.csv", index=False)
    return df


# ---------- FD main per-method headline table ------------------------------

def fd_main_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["run"] == "fd_main"].copy()
    sub = sub.sort_values("final_acc_tail20", ascending=False)
    cols = [
        "method_display", "final_acc_tail20", "final_client_acc_tail20",
        "final_kl_div", "final_eff_noise_var", "final_fairness_gini",
        "final_participation_gini", "final_logit_cosine_div",
        "final_label_cov_ratio", "comm_reduction_ratio_final",
        "rounds_to_30pct",
    ]
    out = sub[cols].copy()
    out.columns = [
        "Method", "Server Acc (last 20)", "Client Acc Avg",
        "KL Divergence", "Eff. Noise Var", "Fairness Gini",
        "Participation Gini", "Logit Cosine Div",
        "Label Coverage", "Comm Reduction",
        "Rounds to 30%",
    ]
    out.to_csv(OUT / "fd_main_method_table.csv", index=False)
    return out


# ---------- Ranking inversion (FL vs FD) -----------------------------------

def ranking_fl_vs_fd(df: pd.DataFrame) -> pd.DataFrame:
    fd = df[df["run"] == "fd_main"].set_index("method")
    fl = df[df["run"] == "fl_baseline"].set_index("method")
    common = sorted(set(fd.index) & set(fl.index))
    fl_acc = fl.loc[common, "final_acc_tail20"]
    fd_acc = fd.loc[common, "final_acc_tail20"]
    fl_rank = fl_acc.rank(ascending=False, method="min").astype(int)
    fd_rank = fd_acc.rank(ascending=False, method="min").astype(int)
    delta_rank = fd_rank - fl_rank
    out = pd.DataFrame({
        "method_display": [METHOD_DISPLAY.get(m, m) for m in common],
        "fl_final_acc": fl_acc.values,
        "fd_final_acc": fd_acc.values,
        "fl_rank": fl_rank.values,
        "fd_rank": fd_rank.values,
        "delta_rank_fd_minus_fl": delta_rank.values,
    }).sort_values("fl_rank")

    rho, rho_p = spearmanr(fl_acc.values, fd_acc.values)
    tau, tau_p = kendalltau(fl_acc.values, fd_acc.values)
    out.attrs["spearman_rho"] = float(rho)
    out.attrs["spearman_p"] = float(rho_p)
    out.attrs["kendall_tau"] = float(tau)
    out.attrs["kendall_p"] = float(tau_p)

    out.to_csv(OUT / "ranking_fl_vs_fd.csv", index=False)
    (OUT / "ranking_fl_vs_fd_stats.json").write_text(json.dumps({
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "kendall_tau": float(tau), "kendall_p": float(tau_p),
        "n_methods": len(common),
    }, indent=2))
    return out


# ---------- Noise sweep summary --------------------------------------------

def noise_sweep_summary() -> pd.DataFrame:
    rows = []
    runs_loaded = {tag: load(PATHS[run_key]) for _, run_key, _ in NOISE_LEVELS for tag in [run_key]}
    methods = sorted(set(runs_loaded["noise_errfree"]["results"].keys()))
    for m in methods:
        row = {"method": m, "method_display": METHOD_DISPLAY.get(m, m)}
        accs = {}
        for label, run_key, snr in NOISE_LEVELS:
            data = runs_loaded[run_key]
            acc_key = primary_acc_key(data, m)
            v = final_value(data, m, acc_key)
            row[f"acc_{label}"] = v
            accs[label] = v
            row[f"kl_{label}"] = final_value(data, m, "kl_divergence_avg")
            row[f"noise_var_{label}"] = final_value(data, m, "effective_noise_var")
        ef = accs["errfree"]
        for label, _, _ in NOISE_LEVELS:
            row[f"delta_vs_errfree_{label}"] = accs[label] - ef
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("acc_-20 dB", ascending=False)
    df.to_csv(OUT / "noise_sweep_summary.csv", index=False)
    return df


# ---------- Figures --------------------------------------------------------

def _style(method: str) -> dict[str, Any]:
    return METHOD_STYLE.get(method, {"color": "k", "marker": "x", "ls": "-"})


def fig_main_convergence():
    data = load(PATHS["fd_main"])
    methods = list(data["results"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for m in methods:
        acc_key = primary_acc_key(data, m)
        y = smooth(per_round(data, m, acc_key).tolist())
        s = _style(m)
        ax.plot(np.arange(len(y)), y, label=METHOD_DISPLAY.get(m, m),
                color=s["color"], linestyle=s["ls"], linewidth=1.5,
                marker=s["marker"], markersize=4, markevery=30)
    ax.set_xlabel("Round")
    ax.set_ylabel("Server Accuracy (smoothed)")
    ax.set_title("FD Main — CIFAR-10, $\\alpha$=0.5, DL SNR=$-$20 dB")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    ax2 = axes[1]
    for m in methods:
        y = smooth(per_round(data, m, "kl_divergence_avg").tolist())
        s = _style(m)
        ax2.plot(np.arange(len(y)), y, label=METHOD_DISPLAY.get(m, m),
                 color=s["color"], linestyle=s["ls"], linewidth=1.3)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("KL Divergence (avg)")
    ax2.set_title("Distillation Quality")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig1_main_convergence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_main_bar(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    methods_order = table["Method"].tolist()
    accs = table["Server Acc (last 20)"].values * 100
    colors = []
    for d in methods_order:
        m_key = next(k for k, v in METHOD_DISPLAY.items() if v == d)
        colors.append(_style(m_key)["color"])
    bars = ax.bar(range(len(methods_order)), accs, color=colors, edgecolor="black")
    ax.set_xticks(range(len(methods_order)))
    ax.set_xticklabels(methods_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Server Accuracy (%, mean of last 20 rounds)")
    ax.set_title("FD Main — Final accuracy ranking (CIFAR-10, DL SNR=$-$20 dB)")
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_main_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ranking_inversion(rk: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    methods = rk["method_display"].tolist()
    fl = rk["fl_final_acc"].values * 100
    fd = rk["fd_final_acc"].values * 100
    x = np.arange(len(methods))
    w = 0.4
    ax.bar(x - w / 2, fl, w, label="FL (LightCNN)", color="#1f77b4", edgecolor="black")
    ax.bar(x + w / 2, fd, w, label="FD (heterogeneous, DL=$-$20 dB)", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Accuracy (%)")
    title = (f"FL vs FD ranking — Spearman $\\rho$={rk.attrs['spearman_rho']:+.2f}, "
             f"Kendall $\\tau$={rk.attrs['kendall_tau']:+.2f}")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ranking_inversion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_noise_sensitivity(noise_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    snr_labels = [lab for lab, _, _ in NOISE_LEVELS]
    snr_x = [40 if v == float("inf") else v for _, _, v in NOISE_LEVELS]  # plot 'errfree' at +40dB

    ax = axes[0]
    for _, row in noise_df.iterrows():
        m = row["method"]
        ys = [row[f"acc_{lab}"] * 100 for lab in snr_labels]
        s = _style(m)
        ax.plot(snr_x, ys, label=METHOD_DISPLAY.get(m, m),
                color=s["color"], marker=s["marker"], linestyle=s["ls"], linewidth=1.5)
    ax.set_xlabel("Downlink SNR (dB)  (errfree shown at +40)")
    ax.set_ylabel("Final Server Accuracy (%)")
    ax.set_title("FD accuracy vs DL SNR (CIFAR-10, $\\alpha$=0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax2 = axes[1]
    methods = noise_df["method"].tolist()
    deltas = [(noise_df.loc[noise_df["method"] == m, "acc_errfree"].iloc[0] -
               noise_df.loc[noise_df["method"] == m, "acc_-30 dB"].iloc[0]) * 100 for m in methods]
    colors = [_style(m)["color"] for m in methods]
    bars = ax2.bar(range(len(methods)), deltas, color=colors, edgecolor="black")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods],
                        rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Accuracy Drop: errfree $\\rightarrow$ $-$30 dB (pp)")
    ax2.set_title("Channel-noise robustness")
    for b, v in zip(bars, deltas):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:+.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig4_noise_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fairness_pareto(df: pd.DataFrame):
    sub = df[df["run"] == "fd_main"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for _, row in sub.iterrows():
        m = row["method"]
        s = _style(m)
        ax.scatter(row["final_fairness_gini"], row["final_acc_tail20"] * 100,
                   color=s["color"], marker=s["marker"], s=110, edgecolor="black",
                   label=METHOD_DISPLAY.get(m, m))
    ax.set_xlabel("Fairness Gini (lower = fairer)")
    ax.set_ylabel("Final Server Accuracy (%)")
    ax.set_title("Accuracy–Fairness Pareto (FD main)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_fairness_pareto.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_communication(df: pd.DataFrame):
    sub_fd = df[df["run"] == "fd_main"].iloc[0]
    sub_fl = df[df["run"] == "fl_baseline"].iloc[0]
    fd_mb = sub_fd["total_logit_comm_kb"] / 1024
    fl_equiv_mb = sub_fd["total_fl_equiv_comm_mb"]
    fl_actual_mb = sub_fl.get("total_logit_comm_kb", 0) / 1024
    if not np.isfinite(fl_actual_mb) or fl_actual_mb == 0:
        fl_actual_mb = fl_equiv_mb  # fall back to FD's estimate
    ratio = fd_mb / fl_actual_mb if fl_actual_mb else float("nan")

    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(["FL (weights)", "FD (logits)"], [fl_actual_mb, fd_mb],
                  color=["#1f77b4", "#d62728"], edgecolor="black")
    ax.set_ylabel("Cumulative communication (MB)")
    ax.set_title(f"Communication: FD vs FL\nFD overhead = {ratio*100:.3f}% of FL")
    for b, v in zip(bars, [fl_actual_mb, fd_mb]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,.0f} MB",
                ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_communication.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"fd_mb": fd_mb, "fl_mb_estimate": fl_actual_mb, "ratio": ratio}


def fig_noise_var_curves():
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, run_key, _ in NOISE_LEVELS:
        data = load(PATHS[run_key])
        # average effective_noise_var across all methods (it's a channel property)
        all_vars = []
        for m in data["results"]:
            all_vars.append(per_round(data, m, "effective_noise_var"))
        mean_var = np.nanmean(np.vstack(all_vars), axis=0)
        ax.plot(np.arange(len(mean_var)), smooth(mean_var.tolist()), label=f"DL SNR = {label}", linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Effective Noise Variance (avg over methods)")
    ax.set_title("Channel noise environment per SNR level")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig7_noise_var_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_method_signals(df: pd.DataFrame):
    """Why methods win/lose: scatter of signal vs final accuracy."""
    sub = df[df["run"] == "fd_main"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (xcol, xlab) in zip(axes, [
        ("final_logit_cosine_div", "Logit Cosine Diversity (avg)"),
        ("final_label_cov_ratio", "Label Coverage Ratio"),
        ("final_chan_quality_sel", "Channel Quality of Selected"),
    ]):
        for _, row in sub.iterrows():
            m = row["method"]
            s = _style(m)
            ax.scatter(row[xcol], row["final_acc_tail20"] * 100,
                       color=s["color"], marker=s["marker"], s=120, edgecolor="black",
                       label=METHOD_DISPLAY.get(m, m))
        ax.set_xlabel(xlab)
        ax.set_ylabel("Final Server Accuracy (%)")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Method-level signal correlations (FD main)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_method_signals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------- Driver ---------------------------------------------------------

def main():
    print("[1] Building master_table.csv ...")
    df = build_master_table()
    print(f"    rows: {len(df)}, runs: {df['run'].nunique()}, methods: {df['method'].nunique()}")

    print("[2] FD main per-method headline table ...")
    table = fd_main_table(df)
    print(table.to_string(index=False))

    print("[3] FL vs FD ranking inversion ...")
    rk = ranking_fl_vs_fd(df)
    print(rk.to_string(index=False))
    print(f"    Spearman rho = {rk.attrs['spearman_rho']:+.3f} (p={rk.attrs['spearman_p']:.3f}); "
          f"Kendall tau = {rk.attrs['kendall_tau']:+.3f} (p={rk.attrs['kendall_p']:.3f})")

    print("[4] Noise sweep summary ...")
    noise_df = noise_sweep_summary()
    print(noise_df[["method_display", "acc_errfree", "acc_-20 dB", "acc_-30 dB",
                    "delta_vs_errfree_-30 dB"]].to_string(index=False))

    print("[5] Figures ...")
    fig_main_convergence()
    fig_main_bar(table)
    fig_ranking_inversion(rk)
    fig_noise_sensitivity(noise_df)
    fig_fairness_pareto(df)
    comm = fig_communication(df)
    fig_noise_var_curves()
    fig_method_signals(df)
    (OUT / "communication_stats.json").write_text(json.dumps(comm, indent=2))
    print(f"    comm: FD={comm['fd_mb']:.1f} MB, FL={comm['fl_mb_estimate']:.1f} MB, ratio={comm['ratio']*100:.4f}%")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
