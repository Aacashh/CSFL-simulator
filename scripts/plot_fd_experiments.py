#!/usr/bin/env python3
"""IEEE-quality plotting for Federated Distillation client selection experiments.

Generates publication-ready figures with:
  - Serif fonts (Times New Roman / Computer Modern)
  - Proper markers, dash patterns, and academic colour palette
  - Legends in every plot and subplot
  - 3.5-inch single-column or 7.16-inch double-column widths
  - EPS / PDF / PNG output at 300 DPI

Usage:
    python scripts/plot_fd_experiments.py --run <run_name_or_dir> --metrics accuracy,loss \
        --format eps --out-dir paper/figures

    # Multi-panel comparison across several experiment runs
    python scripts/plot_fd_experiments.py \
        --runs "fd_main_cifar10,fd_noniid_alpha1,fd_noniid_alpha5" \
        --panel-metric accuracy --panel-labels "alpha=0.5,alpha=1,alpha=5" \
        --format eps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# IEEE visual config
# ---------------------------------------------------------------------------

# Academic colour palette (colour-blind safe, print-friendly)
COLORS = [
    "#1f77b4",  # muted blue    — FedAvg
    "#d62728",  # brick red     — FedCS
    "#2ca02c",  # cooked green  — TiFL
    "#9467bd",  # muted purple  — FedCor
    "#8c564b",  # chestnut      — MAML
    "#e377c2",  # pink          — APEX v2
    "#17becf",  # cyan          — Oort
    "#ff7f0e",  # safety orange — LabelCov
]

MARKERS = ["o", "s", "^", "D", "v", "*", "P", "X"]

DASHES = [
    (1, 0),           # solid
    (4, 2),           # dashed
    (1, 1),           # dotted
    (4, 2, 1, 2),     # dashdot
    (2, 1),           # short dash
    (4, 1, 1, 1, 1, 1),  # dashdotdot
    (6, 2),
    (3, 1, 1, 1),
]

# Short display names for method keys
SHORT_NAMES = {
    "baseline.fedavg":     "FedAvg",
    "system_aware.fedcs":  "FedCS",
    "system_aware.tifl":   "TiFL",
    "ml.fedcor":           "FedCor",
    "ml.maml_select":      "MAML",
    "ml.apex_v2":          "APEX v2",
    "system_aware.oort":   "Oort",
    "heuristic.label_coverage": "LabelCov",
    "heuristic.random":    "Random",
}

METRIC_LABELS = {
    "accuracy":             "Testing Accuracy",
    "loss":                 "Loss",
    "f1":                   "F1 Score",
    "kl_divergence_avg":    "KL Divergence",
    "logit_comm_kb":        "Logit Comm. (KB)",
    "comm_reduction_ratio": "Comm. Reduction Ratio",
    "effective_noise_var":  r"Effective Noise Var. ($\sigma_\omega^2$)",
    "dynamic_steps_kr":     r"Dynamic Steps $K_r$",
    "client_accuracy_avg":  "Avg. Client Accuracy",
    "client_accuracy_std":  "Client Acc. Std. Dev.",
    "fairness_gini":        "Fairness (Gini)",
    "cum_comm":             "Cumulative Comm. (MB)",
    "wall_clock":           "Wall Clock (s)",
    "composite":            "Composite Score",
    "distillation_loss_avg":"Distillation Loss",
}


def _apply_ieee_rc(plt, dpi: int = 300):
    """Apply IEEE-recommended rcParams."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size":          9,
        "axes.labelsize":     10,
        "axes.titlesize":     10,
        "legend.fontsize":    7.5,
        "legend.title_fontsize": 8,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "figure.dpi":         dpi,
        "savefig.dpi":        dpi,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        "lines.linewidth":    1.4,
        "lines.markersize":   4.5,
        "axes.grid":          True,
        "grid.alpha":         0.30,
        "grid.linewidth":     0.5,
        "axes.linewidth":     0.6,
        "text.usetex":        False,
        "mathtext.fontset":   "cm",
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.7",
        "legend.fancybox":    False,
    })


def _short(key: str) -> str:
    if key in SHORT_NAMES:
        return SHORT_NAMES[key]
    return key.split(".")[-1] if "." in key else key


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def resolve_run_dir(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    arts = Path("artifacts/runs")
    if (arts / name).is_dir():
        return arts / name
    # prefix match
    if arts.is_dir():
        matches = sorted(
            [d for d in arts.iterdir() if d.is_dir() and d.name.startswith(name)],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if matches:
            return matches[0]
    print(f"Error: Cannot resolve run '{name}'", file=sys.stderr)
    sys.exit(1)


def load_compare_results(run_dir: Path) -> dict:
    cp = run_dir / "compare_results.json"
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    # single run → wrap
    rp = run_dir / "results.json"
    mp = run_dir / "metrics.json"
    if rp.exists():
        with open(rp) as f:
            data = json.load(f)
        method = data.get("method", "run")
        return {"methods": [method], "results": {method: data}}
    if mp.exists():
        with open(mp) as f:
            data = json.load(f)
        return {"methods": ["run"], "results": {"run": data}}
    print(f"Error: No results in {run_dir}", file=sys.stderr)
    sys.exit(1)


def extract_series(result: dict, metric: str):
    """Return (rounds, values) for a metric from a single-method result."""
    metrics_list = result.get("metrics", [])
    xs, ys = [], []
    for row in metrics_list:
        r = row.get("round", -1)
        if isinstance(r, (int, float)) and r >= 0:
            xs.append(int(r))
            ys.append(float(row.get(metric, 0.0) or 0.0))
    return xs, ys


def moving_average(ys, window: int = 10):
    import numpy as np
    if len(ys) < window:
        return ys
    kernel = np.ones(window) / window
    smoothed = np.convolve(ys, kernel, mode="valid")
    # pad beginning
    pad = ys[: len(ys) - len(smoothed)]
    return list(pad) + list(smoothed)


# ---------------------------------------------------------------------------
# Individual metric plot
# ---------------------------------------------------------------------------

def plot_metric(
    data: dict,
    metric: str,
    *,
    plt,
    fmt: str = "eps",
    dpi: int = 300,
    out_dir: Path = Path("."),
    width: float = 3.5,
    height: float = 2.6,
    smooth: int = 10,
    show_raw: bool = True,
):
    methods = data.get("methods", list(data["results"].keys()))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, method in enumerate(methods):
        xs, ys = extract_series(data["results"][method], metric)
        if not ys:
            continue
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        dash = DASHES[i % len(DASHES)]

        # Raw trace (transparent)
        if show_raw and smooth > 1:
            ax.plot(xs, ys, color=color, alpha=0.20, linewidth=0.7)

        # Smoothed
        ys_s = moving_average(ys, smooth) if smooth > 1 else ys
        xs_s = xs[: len(ys_s)]

        # Thin markers every N points
        mark_every = max(1, len(xs_s) // 8)
        ax.plot(
            xs_s, ys_s,
            color=color,
            marker=marker,
            markevery=mark_every,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markeredgecolor=color,
            dashes=dash,
            linewidth=1.4,
            label=_short(method),
        )

    ax.set_xlabel("Global Communication Rounds")
    ax.set_ylabel(_label(metric))
    ax.legend(
        loc="best",
        fontsize=7.5,
        framealpha=0.85,
        edgecolor="0.7",
        fancybox=False,
        handlelength=2.2,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.35,
    )
    fig.tight_layout()

    filepath = out_dir / f"{metric}.{fmt}"
    fig.savefig(str(filepath), format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------

def plot_multi_panel(
    data: dict,
    metrics: list[str],
    *,
    plt,
    fmt: str = "eps",
    dpi: int = 300,
    out_dir: Path = Path("."),
    width: float = 7.16,
    smooth: int = 10,
):
    n = len(metrics)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    height_per = 2.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height_per * nrows))
    fig.patch.set_facecolor("white")

    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes.tolist()]
    else:
        axes = [row.tolist() if hasattr(row, "tolist") else [row] for row in axes]

    methods = data.get("methods", list(data["results"].keys()))

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.set_facecolor("white")

        for i, method in enumerate(methods):
            xs, ys = extract_series(data["results"][method], metric)
            if not ys:
                continue
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            dash = DASHES[i % len(DASHES)]

            ys_s = moving_average(ys, smooth) if smooth > 1 else ys
            xs_s = xs[: len(ys_s)]
            mark_every = max(1, len(xs_s) // 8)

            ax.plot(
                xs_s, ys_s,
                color=color, marker=marker,
                markevery=mark_every, markersize=4,
                markerfacecolor="white", markeredgewidth=0.9,
                markeredgecolor=color,
                dashes=dash, linewidth=1.3,
                label=_short(method),
            )

        ax.set_xlabel("Global Communication Rounds")
        ax.set_ylabel(_label(metric))
        # subplot label (a), (b), ...
        ax.text(
            -0.02, 1.06, f"({chr(97 + idx)})",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right",
        )
        ax.legend(
            loc="best", fontsize=6.5,
            framealpha=0.85, edgecolor="0.7",
            fancybox=False, handlelength=2.0,
            handletextpad=0.4, borderpad=0.3,
            labelspacing=0.3,
        )

    # hide unused panels
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    filepath = out_dir / f"multi_panel.{fmt}"
    fig.savefig(str(filepath), format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# Bar chart for final accuracies
# ---------------------------------------------------------------------------

def plot_final_bar(
    data: dict,
    metric: str = "accuracy",
    *,
    plt,
    fmt: str = "eps",
    dpi: int = 300,
    out_dir: Path = Path("."),
    width: float = 3.5,
    height: float = 2.2,
):
    import numpy as np

    methods = data.get("methods", list(data["results"].keys()))
    names = [_short(m) for m in methods]
    vals = []
    for method in methods:
        _, ys = extract_series(data["results"][method], metric)
        vals.append(ys[-1] if ys else 0.0)

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(names))
    bars = ax.bar(
        x, vals, width=0.55,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        edgecolor="black", linewidth=0.5,
    )

    # value labels
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(_label(metric))
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1.0)
    fig.tight_layout()

    filepath = out_dir / f"bar_{metric}.{fmt}"
    fig.savefig(str(filepath), format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IEEE-quality plotting for FD client selection experiments",
    )
    parser.add_argument("--run", default=None, help="Single run name/dir")
    parser.add_argument("--runs", default=None,
                        help="Comma-separated run names for multi-run panel")
    parser.add_argument("--metrics", default="accuracy,loss,f1",
                        help="Comma-separated metrics to plot")
    parser.add_argument("--format", default="eps", choices=["eps", "pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smooth", type=int, default=10,
                        help="Moving average window (paper uses 10)")
    parser.add_argument("--width", type=float, default=3.5,
                        help="Single plot width in inches (3.5=IEEE single col)")
    parser.add_argument("--height", type=float, default=2.6)
    parser.add_argument("--no-raw", action="store_true",
                        help="Hide raw (transparent) trace, show only smoothed")
    parser.add_argument("--bar", action="store_true",
                        help="Also generate final-accuracy bar chart")

    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_ieee_rc(plt, args.dpi)

    plot_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.run:
        run_dir = resolve_run_dir(args.run)
        data = load_compare_results(run_dir)
        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for metric in plot_metrics:
            fp = plot_metric(
                data, metric, plt=plt, fmt=args.format, dpi=args.dpi,
                out_dir=out_dir, width=args.width, height=args.height,
                smooth=args.smooth, show_raw=not args.no_raw,
            )
            saved.append(fp)
            print(f"  {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")

        if len(plot_metrics) >= 2:
            fp = plot_multi_panel(
                data, plot_metrics, plt=plt, fmt=args.format, dpi=args.dpi,
                out_dir=out_dir, width=7.16, smooth=args.smooth,
            )
            saved.append(fp)
            print(f"  {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")

        if args.bar:
            fp = plot_final_bar(
                data, "accuracy", plt=plt, fmt=args.format, dpi=args.dpi,
                out_dir=out_dir, width=args.width, height=2.2,
            )
            saved.append(fp)
            print(f"  {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")

        print(f"\nSaved {len(saved)} figure(s) to {out_dir}/")
        print(f"Format: {args.format.upper()} | DPI: {args.dpi}")

    elif args.runs:
        print("Multi-run panel mode not yet implemented — use --run for now.")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
