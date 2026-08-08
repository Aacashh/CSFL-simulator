#!/usr/bin/env python3
"""Build fig_r8_ratio_gap, the accuracy gap against the participation ratio.

Same typography, axes convention and palette as make_revision_figures.py, so
the figure sits beside the others without looking imported. The form is a
paired scatter rather than a fitted line, because the relationship is monotone
but not linear and a regression line would claim more than the data supports.
The rank correlation is printed instead.

    python3 make_ratio_figure.py [runs-root]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "runs_scope_revised" / "runs_scope_revised"
OUT = HERE / "figures_revised"
OUT.mkdir(exist_ok=True)

COL = 3.50                      # \columnwidth in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "legend.frameon": False,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
    "axes.linewidth": 0.5, "axes.labelpad": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.4,
    "savefig.bbox": "standard",
})

INK, MUTED = "#111111", "#6b6b6b"
BAND = "#f2f4f7"
SCOPE, DEBT, RANDOM = "fd_native.scope_fd", "fd_native.scope_fd_debt_only", "heuristic.random"
C_SCOPE, C_DEBT = "#0072B2", "#009E73"
LOSS_BAND = "#fbe9e7"


def finals(result):
    rows = [x for x in result.get("metrics", []) if int(x.get("round", -1)) >= 0]
    return rows[-1] if rows else None


def collect():
    """Matched Fashion-MNIST configurations, everything else held fixed."""
    cells = defaultdict(lambda: defaultdict(dict))
    for res in sorted(ROOT.rglob("compare_results.json")):
        payload = json.loads(res.read_text())
        cfg, results = payload.get("config", {}), payload.get("results", {})
        if RANDOM not in results:
            continue
        if (cfg.get("dropout_prob") or cfg.get("staleness_window")
                or cfg.get("energy_budget") or cfg.get("channel_noise")):
            continue
        if cfg.get("dataset") != "Fashion-MNIST" or cfg.get("dirichlet_alpha") != 0.5:
            continue
        if (cfg.get("public_dataset") != "same" or cfg.get("public_dataset_size") != 2000
                or cfg.get("public_label_noise")):
            continue
        for method in (SCOPE, DEBT, RANDOM):
            f = finals(results.get(method, {}))
            if f:
                cells[(cfg["total_clients"], cfg["clients_per_round"])][method][
                    int(cfg["seed"])] = f["accuracy"]
    return cells


def paired(c, method):
    seeds = sorted(set(c.get(method, {})) & set(c[RANDOM]))
    if not seeds:
        return None
    gaps = [(c[method][s] - c[RANDOM][s]) * 100 for s in seeds]
    n = len(gaps)
    # One standard deviation of the paired gap, matching the convention used by
    # the whiskers and bands of every other figure in the paper.
    return st.fmean(gaps), (st.stdev(gaps) if n > 1 else 0.0), n


def main():
    cells = collect()
    rows = []
    for (N, K), c in sorted(cells.items(), key=lambda kv: kv[0][1] / kv[0][0]):
        s, d = paired(c, SCOPE), paired(c, DEBT)
        if s is None:
            continue
        rows.append((K / N, N, K, s, d))
    if not rows:
        raise SystemExit("no matched configurations found")

    n = len(rows)
    fig, ax = plt.subplots(figsize=(COL, 3.45))

    xlo, xhi = -2.05, 3.42
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    ax.axvspan(xlo, 0, color=LOSS_BAND, lw=0, alpha=0.75, zorder=1)
    ax.axvline(0, color=INK, lw=0.6, zorder=3)

    for i, (x, N, K, s, d) in enumerate(rows):
        if d is not None:
            ax.plot([s[0], d[0]], [i, i], color="#b8b8b8", lw=0.9,
                    solid_capstyle="round", zorder=2)
        ax.errorbar(s[0], i, xerr=s[1], fmt="none", ecolor=C_SCOPE,
                    elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=4)
        if d is not None:
            ax.plot(d[0], i, marker="^", color=C_DEBT, ms=3.3, lw=0,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.plot(s[0], i, marker="o", color=C_SCOPE, ms=3.7, lw=0,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{N}/{K}" for _, N, K, _, _ in rows])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(xlo, xhi)
    ax.set_ylabel("Client pool / cohort  $N/K$")
    ax.set_xlabel("Final accuracy minus uniform random (pp)")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(INK)
    ax.tick_params(which="both", colors=INK)

    # Right-hand axis carries the ratio, since that is the ordering variable.
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f"{x:.3f}" for x, _, _, _, _ in rows], fontsize=6.3,
                        color=MUTED)
    ax2.set_ylabel("Participation ratio  $K/N$", fontsize=7.4, color=MUTED,
                   labelpad=3)
    ax2.tick_params(axis="y", which="both", length=0, colors=MUTED)
    ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for sp in ax2.spines.values():
        sp.set_linewidth(0.5); sp.set_color(INK)

    ax.text(xlo + 0.12, n - 0.85, "worse than\nrandom", fontsize=6.2,
            color="#a04a3c", ha="left", va="top", linespacing=1.25, zorder=7)

    try:
        from scipy.stats import spearmanr
        rho_s = spearmanr([r[0] for r in rows], [r[3][0] for r in rows])[0]
        rho_d = spearmanr([r[0] for r in rows if r[4]],
                          [r[4][0] for r in rows if r[4]])[0]
        ax.text(xlo + 0.12, 0.30,
                f"Spearman $\\rho$ against $K/N$\n"
                f"SCOPE-FD {rho_s:+.2f}\nDebt only {rho_d:+.2f}",
                fontsize=6.2, color=MUTED, ha="left", va="center",
                linespacing=1.35, zorder=7)
    except Exception:
        pass

    handles = [
        Line2D([], [], color=C_SCOPE, marker="o", lw=0, ms=3.7,
               markeredgecolor="white", markeredgewidth=0.6, label="SCOPE-FD"),
        Line2D([], [], color=C_DEBT, marker="^", lw=0, ms=3.3,
               markeredgecolor="white", markeredgewidth=0.6, label="Debt only"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2,
              bbox_to_anchor=(1.005, 1.005), borderaxespad=0.0)

    fig.subplots_adjust(left=0.175, right=0.845, top=0.985, bottom=0.098)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r8_ratio_gap.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)

    print(f"wrote {OUT/'fig_r8_ratio_gap.pdf'} and .png")
    print(f"{len(rows)} configurations")
    for x, N, K, s, d in rows:
        dd = f"{d[0]:+.2f}" if d else "  --"
        print(f"   N={N:>3} K={K:>2}  K/N={x:.3f}  n={s[2]}  "
              f"SCOPE {s[0]:+.2f} +-{s[1]:.2f}   debt {dd}")


if __name__ == "__main__":
    main()
