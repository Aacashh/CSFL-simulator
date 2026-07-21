"""Plot aggregated SCOPE-FD results with 95% confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pick_group(data, key):
    groups = data["groups"]
    if key:
        if key not in groups:
            raise KeyError(f"Unknown group key: {key}")
        return key, groups[key]
    if not groups:
        raise ValueError("No aggregate groups found")
    first = next(iter(groups))
    return first, groups[first]


def plot_curves(group, metric, output_base, formats):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for method, result in group["methods"].items():
        points = [point for point in result["curves"].get(metric, []) if point["mean"] is not None]
        if not points:
            continue
        x = np.asarray([point["round"] + 1 for point in points])
        mean = np.asarray([point["mean"] for point in points])
        low = np.asarray([point["ci95_low"] for point in points])
        high = np.asarray([point["ci95_high"] for point in points])
        ax.plot(x, mean, label=method)
        ax.fill_between(x, low, high, alpha=0.18)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_base.with_suffix("." + fmt), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_bars(group, metric, output_base, formats):
    methods, means, errors = [], [], []
    for method, result in group["methods"].items():
        summary = result["final"].get(metric, {})
        if summary.get("mean") is None:
            continue
        methods.append(method)
        means.append(summary["mean"])
        errors.append(summary["mean"] - summary["ci95_low"])
    fig, ax = plt.subplots(figsize=(max(4.8, len(methods) * 0.75), 3.2))
    ax.bar(range(len(methods)), means, yerr=errors, capsize=3)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_base.with_suffix("." + fmt), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(data, method, metric, output_base, formats):
    cells = []
    for group in data["groups"].values():
        config = group["config"]
        result = group["methods"].get(method)
        if result is None or config.get("scope_au") is None or config.get("scope_ad") is None:
            continue
        summary = result["final"].get(metric, {})
        if summary.get("mean") is not None:
            cells.append((float(config["scope_au"]), float(config["scope_ad"]), summary["mean"]))
    if not cells:
        raise ValueError("No coefficient-grid cells found for the requested method/metric")
    aus = sorted({cell[0] for cell in cells})
    ads = sorted({cell[1] for cell in cells})
    grid = np.full((len(aus), len(ads)), np.nan)
    for au, ad, value in cells:
        grid[aus.index(au), ads.index(ad)] = value
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    image = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(range(len(ads)), ads)
    ax.set_yticks(range(len(aus)), aus)
    ax.set_xlabel("alpha diversity")
    ax.set_ylabel("alpha uncertainty")
    fig.colorbar(image, ax=ax, label=metric.replace("_", " ").title())
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_base.with_suffix("." + fmt), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("--kind", choices=("curves", "bars", "heatmap"), default="curves")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--method", default="fd_native.scope_fd")
    parser.add_argument("--group")
    parser.add_argument("--output", type=Path, default=Path("scope_revision_plot"))
    parser.add_argument("--formats", default="png,eps")
    args = parser.parse_args()

    data = json.loads(args.aggregate_json.read_text(encoding="utf-8"))
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "heatmap":
        plot_heatmap(data, args.method, args.metric, args.output, formats)
    else:
        _, group = _pick_group(data, args.group)
        if args.kind == "curves":
            plot_curves(group, args.metric, args.output, formats)
        else:
            plot_bars(group, args.metric, args.output, formats)
    print(f"Wrote {', '.join(str(args.output.with_suffix('.' + fmt)) for fmt in formats)}")


if __name__ == "__main__":
    main()
