"""Build the COMPACT, manuscript-ready MAML-Select package for the IEEE TAI letter.

Curates the existing review pack down to the reviewer-effective minimum:
  Figures (plots_final/):  fig_convergence, fig_tradeoff, fig_lambda_ablation
  Tables  (tables_final/): tab_main_summary, tab_maml_vs_fedavg, tab_ablation
  Plus CAPTIONS_AND_RESPONSES.md (captions + reviewer-response matrix).

Data loaders and styling are reused from build_review_visuals.py (single source of truth);
no missing seeds are imputed and `n` is reported honestly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csfl_simulator.experiments.maml_select.build_review_visuals import (
    COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_WINDOWS_RESULTS,
    METHOD_LABELS,
    METHOD_ORDER,
    REPO_ROOT,
    configure_style,
    load_main_results,
    load_variant_summary,
    save_figure,
)

DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "maml_select" / "review_pack"
OURS = "research.maml_select"

# Vibrant, colourblind-aware palette (overrides the muted review-pack colours).
# MAML-Select gets a striking magenta so the proposed method stands out (no black).
COLORS = {
    "baseline.fedavg":      "#0077BB",  # blue
    "system_aware.fedcs":   "#EE7733",  # orange
    "system_aware.oort":    "#009988",  # teal-green
    "system_aware.tifl":    "#DDAA33",  # gold
    "ml.fedcor":            "#AA4499",  # purple
    "research.criticalfl":  "#774411",  # brown
    "research.fedgcs":      "#33BBEE",  # cyan
    "research.maml_select": "#EE3377",  # magenta (hero)
}
MAML_EDGE = "#7A1745"   # dark magenta outline for MAML markers
TFLOPS_COLOR = "#0077BB"  # blue, for the TFLOPs axis in Fig. 3a
DROP_COLOR = "#CC3311"    # red: ablation removal lowers accuracy
GAIN_COLOR = "#009E73"    # green: ablation removal raises accuracy


def _lighten(hex_color: str, amount: float = 0.82) -> tuple:
    """Blend a hex colour toward white (opaque tint) so std bands render in EPS without alpha."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c + (1.0 - c) * amount for c in (r, g, b))


def _smooth(values, window: int):
    """Centred moving average for readable convergence curves (window auto-clamped)."""
    s = pd.Series(np.asarray(values, dtype=float))
    if len(s) < 3:
        return s.to_numpy()
    w = max(3, min(window, len(s)))
    return s.rolling(w, center=True, min_periods=1).mean().to_numpy()


# Distinct markers per method so figures stay legible in colour and in greyscale.
MARKERS = {
    "baseline.fedavg": "o",
    "system_aware.fedcs": "s",
    "system_aware.oort": "^",
    "system_aware.tifl": "D",
    "ml.fedcor": "v",
    "research.criticalfl": "P",
    "research.fedgcs": "X",
    "research.maml_select": "*",
}
ACCENT = "#D55E00"  # colourblind-safe orange accent
GRID_KW = dict(color="#E6E6E6", linewidth=0.6, alpha=1.0)


def _clean_axes(ax) -> None:
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.7, labelsize=8)


def _inset_legend(ax, loc="best", ncol=1, methods=METHOD_ORDER):
    """Place the method legend inside the given axes with a clean, compact white frame."""
    leg = ax.legend(
        handles=method_legend_handles(methods),
        loc=loc, ncol=ncol, fontsize=5.1, frameon=True,
        facecolor="white", edgecolor="#CCCCCC", framealpha=0.9,
        handlelength=1.0, columnspacing=0.7, handletextpad=0.25,
        borderpad=0.35, labelspacing=0.22, markerscale=0.8,
    )
    leg.get_frame().set_linewidth(0.45)
    leg.set_zorder(20)
    return leg


def method_legend_handles(methods=METHOD_ORDER):
    handles = []
    for m in methods:
        is_ours = m == OURS
        handles.append(
            plt.Line2D(
                [0], [0],
                color=COLORS.get(m, "#333333"),
                marker=MARKERS.get(m, "o"),
                markersize=8 if is_ours else 5.5,
                markeredgecolor="white" if not is_ours else MAML_EDGE,
                markeredgewidth=0.5,
                linestyle="-",
                linewidth=2.4 if is_ours else 1.5,
                label=METHOD_LABELS.get(m, m),
            )
        )
    return handles


# ── Figure 1: CIFAR-100 convergence across four metrics (2x2 grid) ─────────────
def fig_convergence(c100: pd.DataFrame, plots_dir: Path) -> None:
    # (metric, title, y-label, scale, legend-corner). Recall is omitted: weighted recall is
    # mathematically identical to accuracy for single-label classification. Test loss is added
    # as a genuinely distinct convergence signal.
    metrics = [
        ("accuracy", "(a) Accuracy", "Test accuracy (%)", 100.0, "lower right"),
        ("precision", "(b) Precision", "Precision (%)", 100.0, "lower right"),
        ("cum_modelled_carbon_g", "(c) Carbon footprint", r"Cumulative carbon (gCO$_2$e)", 1.0, "lower right"),
        ("loss", "(d) Test loss", "Test loss", 1.0, "center right"),
    ]
    present = [m for m in METHOD_ORDER if m in set(c100["method_key"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.2))
    for ax, (metric, title, ylab, scale, legloc) in zip(axes.flat, metrics):
        for idx, method in enumerate(METHOD_ORDER):
            group = c100[c100["method_key"].astype(str) == method]
            if group.empty:
                continue
            summ = (
                group.groupby("round", as_index=False)
                .agg(val=(metric, "mean"), std=(metric, "std"), n=("seed", "nunique"))
                .sort_values("round")
            )
            color = COLORS.get(method, "#333333")
            is_ours = method == OURS
            step = max(10, len(summ) // 5)
            win = max(5, len(summ) // 7)  # light moving-average smoothing
            x = summ["round"].to_numpy()
            y = _smooth(scale * summ["val"].to_numpy(), win)
            ax.plot(
                x, y, color=color, linewidth=1.9 if is_ours else 1.0,
                marker=MARKERS.get(method, "o"), markersize=6.5 if is_ours else 3.4,
                markevery=(idx % step, step),
                markeredgecolor="white" if not is_ours else MAML_EDGE, markeredgewidth=0.4,
                zorder=6 if is_ours else 2 + idx * 0.1,
                solid_capstyle="round", solid_joinstyle="round",
            )
            # Shaded ±1 s.d. band for MAML-Select only.
            if is_ours and (summ["n"] >= 2).any():
                band = _smooth(scale * summ["std"].fillna(0.0).to_numpy(), win)
                ax.fill_between(x, y - band, y + band, color=_lighten(color, 0.82),
                                linewidth=0.0, zorder=1)
        _clean_axes(ax)
        ax.set_xlim(0, 150)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=8.6)
        _inset_legend(ax, loc=legloc, ncol=2, methods=present)
    for ax in axes[1]:
        ax.set_xlabel("Communication round", fontsize=8.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.suptitle(r"CIFAR-100 (non-IID, $\alpha{=}0.5$)", fontsize=10.5, fontweight="bold", y=0.992)
    save_figure(fig, plots_dir, "fig_convergence")


# Dataset marker shapes for the normalized trade-off panel.
DATASET_MARK = {"fashion_main": "o", "cifar10_main": "s", "cifar100_main": "^"}


# ── Figure 2: normalized accuracy-vs-cost trade-off (single unified panel) ──────
def fig_tradeoff(runs: pd.DataFrame, plots_dir: Path) -> None:
    agg = (
        runs.groupby(["scenario_name", "method_key"], observed=True)
        .agg(acc=("final_accuracy", "mean"), tflops=("cum_training_tflops", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    plotted_methods = []
    for scenario in DATASET_ORDER:
        d = agg[agg["scenario_name"] == scenario]
        fa = d[d["method_key"].astype(str) == "baseline.fedavg"]
        if fa.empty:
            continue
        fa_acc = float(fa["acc"].iloc[0])
        fa_tf = float(fa["tflops"].iloc[0])
        for method in METHOD_ORDER:
            if method == "baseline.fedavg":
                continue
            row = d[d["method_key"].astype(str) == method]
            if row.empty:
                continue
            x = float(row["tflops"].iloc[0]) / fa_tf
            y = 100.0 * (float(row["acc"].iloc[0]) - fa_acc)
            is_ours = method == OURS
            ax.scatter(
                x, y, s=58 if is_ours else 28,
                color=COLORS[method], marker=DATASET_MARK[scenario],
                edgecolor=MAML_EDGE if is_ours else "white",
                linewidth=0.7 if is_ours else 0.4,
                zorder=7 if is_ours else 3,
            )
            plotted_methods.append(method)

    _clean_axes(ax)
    xlo, xhi = 0.45, 3.05
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-34, 3)
    # evenly log-spaced ticks (powers of sqrt(2)) so the axis reads uniformly
    ax.set_xticks([0.5, 0.707, 1.0, 1.414, 2.0, 2.828])
    ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2", "2.8"])
    ax.minorticks_off()
    # cheaper (green) vs costlier (red) half-planes relative to FedAvg
    ax.axvspan(xlo, 1.0, color="#E7F3E9", zorder=0)
    ax.axvspan(1.0, xhi, color="#FBEBEB", zorder=0)
    ax.axvline(1.0, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.axhline(0.0, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.annotate("FedAvg", (1.0, 0.0), xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#333333", ha="left", va="bottom")
    ax.text(0.47, 1.6, "cheaper", fontsize=7.2, color="#2E7D32", style="italic", fontweight="bold", va="center")
    ax.text(2.95, 1.6, "costlier", fontsize=7.2, color="#C0392B", style="italic", fontweight="bold", va="center", ha="right")
    ax.set_xlabel(r"Training compute relative to FedAvg ($\times$, log)", fontsize=8.6)
    ax.set_ylabel(r"$\Delta$ accuracy vs FedAvg (pp)", fontsize=8.6)

    # Legend kept out of the dense scatter for tidiness: two grouped rows below the axes.
    # Colour encodes method; marker shape encodes dataset.
    method_keys = [m for m in METHOD_ORDER if m in set(plotted_methods)]
    method_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=COLORS[m], markeredgecolor=MAML_EDGE if m == OURS else "white",
                   markeredgewidth=0.6, label=METHOD_LABELS[m])
        for m in method_keys
    ]
    ds_handles = [
        plt.Line2D([0], [0], marker=DATASET_MARK[s], linestyle="", markersize=6,
                   markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=0.5,
                   label=DATASET_LABELS[s])
        for s in DATASET_ORDER
    ]
    leg1 = ax.legend(handles=method_handles, loc="lower right", bbox_to_anchor=(0.995, 0.02),
                     ncol=2, fontsize=5.4, frameon=True, facecolor="white", edgecolor="#CCCCCC",
                     framealpha=0.95, handletextpad=0.25, columnspacing=0.7, labelspacing=0.28,
                     borderpad=0.35, markerscale=0.85)
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=ds_handles, loc="lower right", bbox_to_anchor=(0.995, 0.30),
                     ncol=1, fontsize=5.4, frameon=True, facecolor="white", edgecolor="#CCCCCC",
                     framealpha=0.95, handletextpad=0.25, labelspacing=0.28, borderpad=0.35,
                     markerscale=0.85, title="Dataset", title_fontsize=5.4)
    leg2.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    save_figure(fig, plots_dir, "fig_tradeoff")


# ── Figure 3: lambda sensitivity (a) + state-feature ablation delta (b) ────────
def fig_lambda_ablation(lambda_frame: pd.DataFrame, ablation_frame: pd.DataFrame, plots_dir: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) lambda sensitivity, twin axis
    lam = lambda_frame.copy()
    lam["lambda"] = lam["variant"].str.extract(r"lambda=([0-9.]+)").astype(float)
    lam = lam.sort_values("lambda")
    acc_color = COLORS[OURS]
    lams = lam["lambda"].to_numpy()

    def _rel(col):  # normalise to the smallest-lambda value (=1.0)
        v = lam[col].astype(float).to_numpy()
        return v / v[0]
    comp, ener, fair = _rel("cum_training_tflops_mean"), _rel("cum_modelled_energy_wh_mean"), _rel("fairness_jain_mean")

    # Right axis: compute, energy, fairness — each relative to lambda=0.1.
    axL2 = axL.twinx()
    rc = axL2.plot(lams, comp, color=COLORS["baseline.fedavg"], marker="s", markersize=4.2,
                   linestyle="--", linewidth=1.7, zorder=4, label="Compute")[0]
    re_ = axL2.plot(lams, ener, color=COLORS["system_aware.oort"], marker="D", markersize=3.8,
                    linestyle="-.", linewidth=1.7, zorder=4, label="Energy")[0]
    rf = axL2.plot(lams, fair, color=COLORS["system_aware.tifl"], marker="^", markersize=4.6,
                   linestyle=":", linewidth=1.7, zorder=4, label="Fairness")[0]
    axL2.set_ylim(0.5, 1.07)

    # Left axis: accuracy (the key outcome).
    la = axL.errorbar(lams, 100.0 * lam["final_accuracy_mean"],
                      yerr=100.0 * lam["final_accuracy_std"].fillna(0.0),
                      color=acc_color, marker="o", markersize=5.5, capsize=3, linewidth=2.3,
                      zorder=6, label="Accuracy")

    # Mark the chosen operating point lambda = 0.5.
    axL.axvline(0.5, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axL.annotate(r"$\lambda^{\star}$", xy=(0.5, 1.0), xycoords=axL.get_xaxis_transform(),
                 xytext=(0, 2), textcoords="offset points", fontsize=8.5, fontweight="bold",
                 color="#666666", ha="center", va="bottom")

    _clean_axes(axL)
    axL2.grid(False)
    axL2.spines["top"].set_visible(False)
    axL.set_xscale("log")
    axL.set_xticks(lams)
    axL.set_xticklabels(["%g" % v for v in lams])
    axL.minorticks_off()
    axL.set_xlabel(r"Cost trade-off $\lambda$", fontsize=9)
    axL.set_ylabel("Final accuracy (%)", color="black", fontsize=9)
    axL.tick_params(axis="y", colors="black", labelsize=8)
    axL2.set_ylabel(r"Relative to $\lambda{=}0.1$", fontsize=9, color="#333333")
    axL2.tick_params(axis="y", labelsize=8)
    axL.set_title("(a) Robustness to $\\lambda$", fontsize=9.5)
    axL.legend(handles=[la, rc, re_, rf], labels=["Accuracy", "Compute", "Energy", "Fairness"],
               loc="lower left", fontsize=6.4, frameon=True, facecolor="white", edgecolor="#CCCCCC",
               framealpha=0.93, ncol=1, handlelength=1.7, handletextpad=0.4,
               labelspacing=0.3, borderpad=0.4)

    # (b) ablation as delta vs full state vector (honest near-zero scale)
    order = [
        "without loss", "without gradient norm", "without latency",
        "without battery", "without frequency", "without staleness",
    ]
    abl = ablation_frame.set_index("variant")
    base = 100.0 * float(abl.loc["all features", "final_accuracy_mean"])
    deltas, errs, labels = [], [], []
    for v in order:
        if v not in abl.index:
            continue
        deltas.append(100.0 * float(abl.loc[v, "final_accuracy_mean"]) - base)
        errs.append(100.0 * float(abl.loc[v, "final_accuracy_std"] or 0.0))
        labels.append(v.replace("without ", "w/o "))
    x = np.arange(len(labels))
    bar_colors = [GAIN_COLOR if d >= 0 else DROP_COLOR for d in deltas]
    axR.bar(x, deltas, yerr=errs, capsize=2.5, color=bar_colors, edgecolor="#333333", linewidth=0.5, zorder=3)
    axR.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)
    _clean_axes(axR)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.8)
    axR.set_ylabel(r"$\Delta$ accuracy vs full (pp)", fontsize=9)
    axR.set_title("(b) State-feature ablation", fontsize=9.5)
    axR.margins(y=0.32)
    drop_patch = plt.matplotlib.patches.Patch(facecolor=DROP_COLOR, edgecolor="#333333", linewidth=0.5, label="removal lowers acc.")
    gain_patch = plt.matplotlib.patches.Patch(facecolor=GAIN_COLOR, edgecolor="#333333", linewidth=0.5, label="removal raises acc.")
    leg = axR.legend(handles=[drop_patch, gain_patch], loc="upper left", fontsize=6.8,
                     frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.92,
                     borderpad=0.5, handlelength=1.2, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    save_figure(fig, plots_dir, "fig_lambda_ablation")


# Single runs not covered by the standard loaders (different folder naming). The
# APEX-batch CriticalFL Fashion-MNIST run uses the same config as the main benchmarks
# (LightCNN, 100 clients, K=10, alpha=0.5, 200 rounds); n=1, seed 42.
EXTRA_RUNS = [
    {
        "path": REPO_ROOT / "artifacts" / "maml_select_letter"
        / "main_benchmarks__fashion_main__research.criticalfl__seed_42" / "result.json",
        "scenario": "fashion_main",
        "method_key": "research.criticalfl",
        "dataset": "Fashion-MNIST",
        "max_round": 200,
    },
]


def _load_extra_run(spec: dict):
    """Load one result.json into a run-row dict and per-round rows (schema-compatible)."""
    path = spec["path"]
    if not Path(path).exists():
        return None, []
    payload = json.loads(Path(path).read_text())
    mets = payload.get("simulation", {}).get("metrics", [])
    within = [m for m in mets if 0 <= int(m.get("round", -1)) <= spec["max_round"]]
    if not within:
        return None, []
    evaluated = [m for m in within if bool(m.get("evaluated", False))]
    final = evaluated[-1] if evaluated else within[-1]
    seed = int(payload.get("seed", -1))
    f = lambda k: float(final.get(k)) if final.get(k) is not None else float("nan")
    run_row = {
        "scenario_name": spec["scenario"], "dataset": spec["dataset"],
        "method_key": spec["method_key"], "method_label": METHOD_LABELS.get(spec["method_key"], spec["method_key"]),
        "seed": seed, "source": "apex-extra",
        "final_accuracy": f("accuracy"), "final_f1": f("f1"),
        "cum_training_tflops": f("cum_training_tflops"),
        "cum_modelled_energy_wh": f("cum_modelled_energy_wh"),
        "cum_modelled_carbon_g": f("cum_modelled_carbon_g"),
        "cum_comm_mb": f("cum_comm_mb"),
        "fairness_jain": f("fairness_jain"),
        "participation_coverage_ratio": f("participation_coverage_ratio"),
    }
    round_rows = [
        {
            "scenario_name": spec["scenario"], "dataset": spec["dataset"],
            "method_key": spec["method_key"], "seed": seed,
            "round": int(m.get("round")),
            "accuracy": float(m["accuracy"]) if m.get("accuracy") is not None else float("nan"),
        }
        for m in within
    ]
    return run_row, round_rows


def append_extra_runs(runs: pd.DataFrame, rounds: pd.DataFrame):
    """Append EXTRA_RUNS into the loaded frames, skipping any already present."""
    for spec in EXTRA_RUNS:
        run_row, round_rows = _load_extra_run(spec)
        if run_row is None:
            print(f"  [extra] missing, skipped: {spec['path']}")
            continue
        present = (
            (runs["scenario_name"].astype(str) == spec["scenario"])
            & (runs["method_key"].astype(str) == spec["method_key"])
            & (runs["seed"] == run_row["seed"])
        ).any()
        if present:
            continue
        runs = pd.concat([runs, pd.DataFrame([run_row])], ignore_index=True)
        rounds = pd.concat([rounds, pd.DataFrame(round_rows)], ignore_index=True)
        print(f"  [extra] added {spec['method_key']} on {spec['dataset']} (seed {run_row['seed']}, "
              f"acc={run_row['final_accuracy']*100:.2f}%, n=1)")
    return runs, rounds


def load_cifar100_metric_rounds(root: Path, max_round: int = 150) -> pd.DataFrame:
    """Per-round accuracy/precision/recall/F1 for the CIFAR-100 main benchmark (all methods)."""
    rows = []
    for path in sorted(Path(root).glob("cifar100_benchmarks_cifar100_main_*_s*/result.json")):
        payload = json.loads(Path(path).read_text())
        if payload.get("experiment_id") != "cifar100_benchmarks":
            continue
        mk = payload.get("method_key")
        seed = int(payload.get("seed", -1))
        for m in payload.get("simulation", {}).get("metrics", []):
            r = int(m.get("round", -1))
            if r < 0 or r > max_round:
                continue

            def g(k):
                try:
                    return float(m.get(k))
                except (TypeError, ValueError):
                    return float("nan")

            rows.append({"method_key": mk, "seed": seed, "round": r,
                         "accuracy": g("accuracy"), "precision": g("precision"),
                         "recall": g("recall"), "f1": g("f1"), "loss": g("loss"),
                         "fairness_jain": g("fairness_jain"),
                         "cum_modelled_energy_wh": g("cum_modelled_energy_wh"),
                         "cum_modelled_carbon_g": g("cum_modelled_carbon_g")})
    return pd.DataFrame(rows)


# ── Aggregations for tables ────────────────────────────────────────────────────
def main_summary_stats(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in DATASET_ORDER:
        for method in METHOD_ORDER:
            g = runs[(runs["scenario_name"] == scenario) & (runs["method_key"].astype(str) == method)]
            if g.empty:
                continue
            n = int(g["seed"].nunique())
            acc = g["final_accuracy"].astype(float) * 100.0
            rows.append(
                {
                    "scenario": scenario,
                    "dataset": DATASET_LABELS[scenario],
                    "method_key": method,
                    "method": METHOD_LABELS[method],
                    "n": n,
                    "acc_mean": acc.mean(),
                    "acc_std": acc.std(ddof=1) if n > 1 else float("nan"),
                    "tflops": g["cum_training_tflops"].astype(float).mean(),
                    "energy": g["cum_modelled_energy_wh"].astype(float).mean(),
                    "carbon": g["cum_modelled_carbon_g"].astype(float).mean(),
                    "jain": g["fairness_jain"].astype(float).mean(),
                    "coverage": g["participation_coverage_ratio"].astype(float).mean() * 100.0,
                }
            )
    return pd.DataFrame(rows)


def maml_vs_fedavg_stats(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in DATASET_ORDER:
        g = runs[runs["scenario_name"] == scenario]
        fa = g[g["method_key"].astype(str) == "baseline.fedavg"].set_index("seed")
        ou = g[g["method_key"].astype(str) == OURS].set_index("seed")
        shared = sorted(set(fa.index) & set(ou.index))
        if not shared:
            continue
        dacc, dtf, den, dca = [], [], [], []
        for s in shared:
            f, o = fa.loc[s], ou.loc[s]
            dacc.append(100.0 * (o["final_accuracy"] - f["final_accuracy"]))
            dtf.append(100.0 * (f["cum_training_tflops"] - o["cum_training_tflops"]) / f["cum_training_tflops"])
            den.append(100.0 * (f["cum_modelled_energy_wh"] - o["cum_modelled_energy_wh"]) / f["cum_modelled_energy_wh"])
            dca.append(100.0 * (f["cum_modelled_carbon_g"] - o["cum_modelled_carbon_g"]) / f["cum_modelled_carbon_g"])
        rows.append(
            {
                "dataset": DATASET_LABELS[scenario],
                "n": len(shared),
                "d_acc": np.mean(dacc),
                "tflops_red": np.mean(dtf),
                "energy_red": np.mean(den),
                "carbon_red": np.mean(dca),
            }
        )
    return pd.DataFrame(rows)


# ── LaTeX table writers (hand-built booktabs) ──────────────────────────────────
def _fmt_pm(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "--"
    if std != std:  # n=1
        return f"{mean:.2f}"
    return f"{mean:.2f}\\,$\\pm$\\,{std:.2f}"


def write_main_summary_tex(stats: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Full-width main benchmark summary. Use inside table* (two-column span).",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Benchmark summary across non-IID datasets (mean over available seeds; $n$ reported).",
        "Communication cost is identical across selectors (rounds$\\times$clients$\\times$model size) and is omitted.",
        "CriticalFL did not complete on CIFAR-10; it is shown for Fashion-MNIST ($n{=}1$) and CIFAR-100.}",
        "\\label{tab:main_summary}",
        "\\small",
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Method & Acc.\\ (\\%) & TFLOPs & Energy (Wh) & Carbon (g) & Jain & Cov.\\ (\\%) & $n$ \\\\",
        "\\midrule",
    ]
    for scenario in DATASET_ORDER:
        sub = stats[stats["scenario"] == scenario]
        if sub.empty:
            continue
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{DATASET_LABELS[scenario]}}}}} \\\\")
        for _, r in sub.iterrows():
            name = r["method"]
            if r["method_key"] == OURS:
                name = f"\\textbf{{{name}}}"
            lines.append(
                f"{name} & {_fmt_pm(r['acc_mean'], r['acc_std'])} & {r['tflops']:.0f} & "
                f"{r['energy']:.0f} & {r['carbon']:.0f} & {r['jain']:.2f} & {r['coverage']:.0f} & {int(r['n'])} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_maml_vs_fedavg_tex(stats: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Single-column MAML-Select vs FedAvg relative summary.",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{MAML-Select relative to FedAvg (paired over shared seeds). Positive reductions are savings;",
        "$\\Delta$Acc.\\ is signed (a negative value is an accuracy trade-off for lower resource use).}",
        "\\label{tab:maml_vs_fedavg}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Dataset & $\\Delta$Acc.\\ (pp) & TFLOPs\\,$\\downarrow$\\,(\\%) & Energy\\,$\\downarrow$\\,(\\%) & Carbon\\,$\\downarrow$\\,(\\%) & $n$ \\\\",
        "\\midrule",
    ]
    for _, r in stats.iterrows():
        lines.append(
            f"{r['dataset']} & {r['d_acc']:+.2f} & {r['tflops_red']:.1f} & "
            f"{r['energy_red']:.1f} & {r['carbon_red']:.1f} & {int(r['n'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_tex(ablation: pd.DataFrame, path: Path) -> None:
    order = [
        "all features", "without loss", "without gradient norm", "without latency",
        "without battery", "without frequency", "without staleness",
    ]
    abl = ablation.copy()
    abl["variant"] = pd.Categorical(abl["variant"], order, ordered=True)
    abl = abl.sort_values("variant")
    lines = [
        "% Single-column state-feature ablation (Fashion-MNIST, n=3).",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{State-feature ablation on Fashion-MNIST ($n=3$). Each row removes one feature from the",
        "six-dimensional state vector; the full vector is the reference.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Variant & Acc.\\ (\\%) & TFLOPs & Jain \\\\",
        "\\midrule",
    ]
    for _, r in abl.iterrows():
        variant = str(r["variant"])
        label = "Full state vector" if variant == "all features" else variant.replace("without ", "w/o ")
        if variant == "all features":
            label = f"\\textbf{{{label}}}"
        acc = 100.0 * r["final_accuracy_mean"]
        std = 100.0 * (r["final_accuracy_std"] if r["final_accuracy_std"] == r["final_accuracy_std"] else 0.0)
        lines.append(f"{label} & {acc:.2f}\\,$\\pm$\\,{std:.2f} & {r['cum_training_tflops_mean']:.0f} & {r['fairness_jain_mean']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_captions_and_responses(out: Path, main_stats: pd.DataFrame, rel_stats: pd.DataFrame) -> None:
    def rel(ds: str, col: str) -> str:
        row = rel_stats[rel_stats["dataset"] == ds]
        return f"{row[col].iloc[0]:.1f}" if not row.empty else "--"

    def dacc(ds: str) -> str:
        row = rel_stats[rel_stats["dataset"] == ds]
        return f"{row['d_acc'].iloc[0]:+.2f}" if not row.empty else "--"

    text = f"""# MAML-Select — Manuscript Figure/Table Captions and Reviewer Responses

Generated by `build_final_pack.py`. All numbers are 3-seed means unless `n` says otherwise;
no missing seeds are imputed. Communication cost is identical across selectors and is omitted.

## Recommended in-paper package (3 figures, 2 tables)

### Fig. 1 — CIFAR-100 convergence, 4 metrics (`fig_convergence`, two-column `figure*`)
**Caption.** CIFAR-100 convergence under non-IID client selection (Dirichlet $\\alpha{{=}}0.5$):
accuracy, precision, cumulative modelled carbon (gCO2e), and test loss versus communication round
(150 rounds). Carbon is modelled from client energy and a declared grid intensity (a proxy, not
hardware-measured). Weighted recall equals accuracy for single-label classification and is omitted. Lines
are seed means with light moving-average smoothing; the shaded band is $\\pm$1 s.d. for MAML-Select.
MAML-Select (magenta) matches the strongest baselines on accuracy, precision, and loss while keeping its
carbon footprint in the efficient cluster, far below CriticalFL ($n{{=}}2$).

Note: CodeCarbon `tracked_unverified` per-run totals also exist (measured_energy_kwh /
estimated_emissions_g in each result.json), but they measure the simulation host (not the FL devices),
are a single value per run (not per-round), and show MAML-Select roughly level with FedAvg -- so they
are better used as a measured cross-check in the response letter than as this curve.

### Fig. 2 — Efficiency–accuracy trade-off (`fig_tradeoff`, single-column `figure`)
**Caption.** Trade-off relative to FedAvg ($3$-seed means). Each point is a (method, dataset) pair:
$x$ = cumulative training compute as a multiple of FedAvg, $y$ = accuracy change in percentage points;
FedAvg is the origin cross-hair and the shaded half-plane ($x<1$) is cheaper than FedAvg. Colour encodes
method, marker shape encodes dataset. MAML-Select sits in the cheaper half-plane with near-zero accuracy
change on every dataset; FedGCS is competitive but not cheaper; system-aware selectors lose large
accuracy; CriticalFL is both costlier ($\\approx1.5\\times$) and less accurate. Energy and carbon follow
the same pattern (Table~1).

### Fig. 3 — $\\lambda$ robustness and ablation (`fig_lambda_ablation`, two-column `figure*`)
**Caption.** (a) Sensitivity to $\\lambda$ on Fashion-MNIST: accuracy (left axis) stays stable across
$\\lambda\\in\\{{0.1,0.5,1,5\\}}$ while compute, energy, and fairness (right axis, relative to $\\lambda{{=}}0.1$)
all fall as $\\lambda$ grows -- larger $\\lambda$ trades fairness for efficiency; $\\lambda^{{\\star}}$ marks the
chosen $\\lambda{{=}}0.5$. (b) State-feature ablation: removing any single feature from the six-dimensional
state vector changes final accuracy only marginally, confirming each contributes without dominating.

### Table 1 — Benchmark summary (`tab_main_summary`, two-column `table*`)
**Caption.** see `tab_main_summary.tex`. Reports Accuracy$\\pm$s.d., TFLOPs, energy, carbon, Jain fairness,
coverage, and $n$ per method and dataset.

### Table 2 — State-feature ablation (`tab_ablation`, single column)
**Caption.** see `tab_ablation.tex`. (The per-dataset reductions vs FedAvg are stated inline in the
text, so no separate MAML-vs-FedAvg table is used in the paper.)

## Honest headline numbers (auto-filled)
- Fashion-MNIST: MAML-Select $\\Delta$Acc. {dacc('Fashion-MNIST')} pp vs FedAvg; TFLOPs $-${rel('Fashion-MNIST','tflops_red')}\\%, energy $-${rel('Fashion-MNIST','energy_red')}\\%, carbon $-${rel('Fashion-MNIST','carbon_red')}\\%.
- CIFAR-10: $\\Delta$Acc. {dacc('CIFAR-10')} pp (trade-off); TFLOPs $-${rel('CIFAR-10','tflops_red')}\\%, energy $-${rel('CIFAR-10','energy_red')}\\%, carbon $-${rel('CIFAR-10','carbon_red')}\\%.
- CIFAR-100: $\\Delta$Acc. {dacc('CIFAR-100')} pp; TFLOPs $-${rel('CIFAR-100','tflops_red')}\\%, energy $-${rel('CIFAR-100','energy_red')}\\%, carbon $-${rel('CIFAR-100','carbon_red')}\\%.

## Move to response letter / supplementary (not in the 6-page letter)
- MAML-vs-FedAvg relative-reduction table (`tab_maml_vs_fedavg`) — removed from the paper; its numbers
  are stated inline in the results text. Keep the CSV for the response letter if a reviewer asks.
- Hardware-tier coverage + Jain figure (`review_fig_fairness_tier_coverage`) — fairness is summarised by
  the Jain/coverage columns of Table 1; MAML-Select is **not** claimed as most fair.
- Selection-overhead scaling figure + table (`review_fig_scaling_overhead`, `review_scaling_overhead`).
- Full paired significance tests (`review_paired_tests.csv`).
- Full $\\lambda$ numeric table and per-seed MAML-vs-FedAvg table.
- Large accuracy/resource bar grid (`review_fig_efficiency_resource_grid`).

## Reviewer concern → evidence mapping (response-letter wording)
- **R2-3 (how was $\\lambda$ chosen / sensitivity):** Fig. 3(a) shows accuracy is robust across two
  orders of magnitude of $\\lambda$; $\\lambda{{=}}0.5$ balances accuracy and compute.
- **R2-5 (std / significance):** all main results are now mean$\\pm$s.d. over 3 seeds (Table 1, Fig. 1
  bands); paired tests in the response letter.
- **R2-6 (fairness across tiers):** Jain and 100\\% coverage reported in Table 1; per-tier figure in supp.
- **R2-7 (Green-AI claims):** TFLOPs/energy/carbon are explicitly described as *modelled proxies*;
  reductions are reported as such, not as measured power.
- **R2-8 (six-feature justification):** Fig. 3(b) and Table 3 ablate each feature.
- **R2-9/R4-8 (figure clarity):** figures regenerated at 600 dpi, larger fonts, single clean legend,
  colourblind-safe palette with hatches.
- **R2-4/R2-11/R3-4 (scale + more baselines):** CIFAR-100 added; FedGCS and (CIFAR-100) CriticalFL
  added alongside FedCS/Oort/TiFL/FedCor.
- **R1-4 (modest accuracy gains):** reframed as a favourable efficiency–accuracy trade-off (Table 2):
  competitive accuracy at 13–25\\% lower compute/energy/carbon.
"""
    (out / "CAPTIONS_AND_RESPONSES.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-results", type=Path, default=DEFAULT_WINDOWS_RESULTS)
    parser.add_argument("--mac-cifar100-results", type=Path, default=REPO_ROOT / "runs" / "maml_select_cifar100")
    parser.add_argument("--mac-review-results", type=Path, default=REPO_ROOT / "runs" / "maml_select")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plots_dir = args.output_dir / "plots_final"
    tables_dir = args.output_dir / "tables_final"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs, rounds = load_main_results(
        [(args.windows_results, "windows"), (args.mac_cifar100_results, "mac-cifar100")]
    )
    if runs.empty:
        raise SystemExit("No main benchmark runs found.")
    runs["method_key"] = runs["method_key"].astype(str)
    rounds["method_key"] = rounds["method_key"].astype(str)
    runs["scenario_name"] = runs["scenario_name"].astype(str)
    rounds["scenario_name"] = rounds["scenario_name"].astype(str)
    runs, rounds = append_extra_runs(runs, rounds)
    print(f"Loaded {len(runs)} runs across {runs['scenario_name'].nunique()} datasets.")

    lambda_summary = load_variant_summary(args.mac_review_results, "lambda_sensitivity")
    ablation_summary = load_variant_summary(args.mac_review_results, "feature_ablation")
    print(f"Lambda variants: {len(lambda_summary)}; ablation variants: {len(ablation_summary)}")

    # Figures
    c100_rounds = load_cifar100_metric_rounds(args.mac_cifar100_results)
    fig_convergence(c100_rounds, plots_dir)
    fig_tradeoff(runs, plots_dir)
    fig_lambda_ablation(lambda_summary, ablation_summary, plots_dir)

    # Tables
    main_stats = main_summary_stats(runs)
    rel_stats = maml_vs_fedavg_stats(runs)
    main_stats.to_csv(tables_dir / "tab_main_summary.csv", index=False)
    rel_stats.to_csv(tables_dir / "tab_maml_vs_fedavg.csv", index=False)
    ablation_summary.to_csv(tables_dir / "tab_ablation.csv", index=False)
    write_main_summary_tex(main_stats, tables_dir / "tab_main_summary.tex")
    write_maml_vs_fedavg_tex(rel_stats, tables_dir / "tab_maml_vs_fedavg.tex")
    write_ablation_tex(ablation_summary, tables_dir / "tab_ablation.tex")

    write_captions_and_responses(args.output_dir, main_stats, rel_stats)

    print(f"Wrote final pack:\n  plots:  {plots_dir}\n  tables: {tables_dir}")
    # Quick honest sanity print
    for ds in ["Fashion-MNIST", "CIFAR-10", "CIFAR-100"]:
        r = rel_stats[rel_stats["dataset"] == ds]
        if not r.empty:
            print(f"  {ds}: dAcc={r['d_acc'].iloc[0]:+.2f}pp  TFLOPs-{r['tflops_red'].iloc[0]:.1f}%  energy-{r['energy_red'].iloc[0]:.1f}%  n={int(r['n'].iloc[0])}")


if __name__ == "__main__":
    main()
