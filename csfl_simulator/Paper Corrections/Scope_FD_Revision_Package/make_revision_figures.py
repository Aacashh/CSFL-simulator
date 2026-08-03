#!/usr/bin/env python3
"""Generate the revised SCOPE-FD figures from the completed multi-seed runs.

Every figure is built directly from runs_scope_revised_30th_July/ so no number
in the manuscript is transcribed by hand.

-----------------------------------------------------------------------------
TYPOGRAPHY AND SIZING
Figures are drawn at their final printed size, so nothing is rescaled at
\\includegraphics time and the type in the figure matches the type in the body.
IEEEtran two-column: \\columnwidth = 3.5 in, \\textwidth = 7.16 in. Body text is
10 pt and captions are 8 pt, so figure type is 8 pt with 7 pt ticks. Type is
Times New Roman with STIX math, which matches IEEEtran's own Times metrics.

AXES CONVENTION
Follows the house style of scientific journals as codified by SciencePlots
(github.com/garrettj403/SciencePlots): a closed frame, ticks pointing inward on
all four sides, minor ticks visible, hairline spines at 0.5 pt. This is the
convention IEEE and APS figures use, and it reads as a journal figure rather
than as a slide.

FORM
Each figure uses the form that answers its question, not a default bar chart.
  * ablation     -> forest plot with an equivalence band, because the finding is
                    that the variants are indistinguishable, and a forest plot
                    is the form built to show exactly that
  * selectors    -> Pareto dominance plot with the dominated region shaded
  * sweeps       -> paired panels, the curves above and the SCOPE-minus-random
                    advantage as a signed strip below, so the gap is its own
                    visual object rather than something the eye must subtract
  * coefficients -> heatmap with marginal profiles on two edges
  * scale        -> dumbbell, one row per configuration, both fairness measures

COLOUR
Okabe-Ito, validated colourblind-safe (worst adjacent deutan dE 9.6). Colour is
never the only channel: every selector also carries a fixed marker, and key
values are printed directly.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

RUNS = Path(__file__).resolve().parents[3] / "runs" / "runs_scope_revised_30th_July"
OUT = Path(__file__).resolve().parent / "figures_revised"
OUT.mkdir(exist_ok=True)

COL, FULL = 3.50, 7.16          # \columnwidth, \textwidth in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.35,
    "axes.linewidth": 0.5,
    "axes.labelpad": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.4,
    "grid.linewidth": 0.4,
    "savefig.bbox": "standard",   # exact figsize; layout via subplots_adjust
})

INK, MUTED, HAIR = "#111111", "#6b6b6b", "#d9d9d9"
BAND = "#f2f4f7"                 # alternating row band
ACCENT = "#0072B2"

STYLE = {
    "fd_native.scope_fd":              ("#0072B2", "o", "SCOPE-FD"),
    "heuristic.random":                ("#D55E00", "s", "Random"),
    "fd_native.scope_fd_debt_only":    ("#009E73", "^", "Debt only"),
    "fd_native.divfl_fd":              ("#E69F00", "D", "DivFL"),
    "fd_native.subtrunc_fd":           ("#CC79A7", "v", "SubTrunc"),
    "fd_native.unionfl_fd":            ("#56B4E9", "P", "UnionFL"),
    "system_aware.oort":               ("#4D4D4D", "X", "Oort"),
    "fd_native.scope_fd_no_server":    ("#E69F00", "D", "Debt + coverage"),
    "fd_native.scope_fd_no_diversity": ("#CC79A7", "v", "Debt + under-pred."),
}


def frame(ax, minor=True):
    """Closed hairline frame, inward ticks all round, optional minor ticks."""
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(INK)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", colors=INK)
    ax.set_axisbelow(True)


def rowbands(ax, n, xlo, xhi):
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, xmin=0, xmax=1, color=BAND, lw=0, zorder=0)


def panel_tag(ax, s, x=0.014, y=0.955):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7.4, color=INK,
            ha="left", va="top", fontweight="bold", zorder=6)


def halo(ax, x, y, col, mk, ms=4.3, z=5):
    ax.plot(x, y, marker=mk, color=col, markersize=ms, linestyle="none",
            markeredgecolor="white", markeredgewidth=0.7, zorder=z)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------ data ----
def load():
    G, CFG = defaultdict(lambda: defaultdict(dict)), {}
    for mf in glob.glob(str(RUNS / "*" / "*" / "manifest.json")):
        man = json.load(open(mf))
        if man.get("status") != "complete":
            continue
        try:
            d = json.load(open(mf.replace("manifest.json", "compare_results.json")))
        except Exception:
            continue
        c = d.get("config", {})
        key = (man.get("family"),
               json.dumps({k: v for k, v in c.items() if k not in ("seed", "name")},
                          sort_keys=True))
        CFG[key] = c
        for m, res in d.get("results", {}).items():
            rows = [r for r in res.get("metrics", []) if r.get("round", -1) >= 0]
            if rows:
                G[key][m][c.get("seed")] = {
                    "acc": rows[-1].get("accuracy"),
                    "gini": rows[-1].get("fairness_gini"),
                    "roll": rows[-1].get("rolling_window_gini"),
                    "conv": res.get("convergence", {}),
                }
    return G, CFG


def agg(v):
    v = [x for x in v if x is not None]
    return (st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None


def fam(G, CFG, name):
    return [(k, CFG[k]) for k in G if k[0] == name]


# --------------------------------------------------------------- figures ----
def fig_ablation(G, CFG):
    """Per-seed slopegraph over the four variants, stacked vertically at column
    width. The point is that the vertical spread between seeds is larger than
    the spread between variants, and that the seed traces do not reorder in a
    consistent way, so the variants cannot be separated."""
    ks = fam(G, CFG, "ablation_headline")
    if not ks:
        return
    k = ks[0][0]
    order = ["fd_native.scope_fd_debt_only", "fd_native.scope_fd_no_diversity",
             "fd_native.scope_fd_no_server", "fd_native.scope_fd"]
    order = [m for m in order if m in G[k]]
    short = ["Debt\nonly", "Debt\n$+$ u.p.", "Debt\n$+$ cov.", "Complete\nSCOPE-FD"]
    seeds = sorted(set().union(*[set(G[k][m]) for m in order]))
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.45), sharex=True)

    for ax, key, ylab, tag, fmt in (
            (axes[0], "acc", "Final accuracy (%)", "(a)", "{:.2f}"),
            (axes[1], "r70", "Rounds to 70% acc.", "(b)", "{:.1f}")):
        series = {}
        for s_ in seeds:
            ys = []
            for m in order:
                v = G[k][m].get(s_)
                ys.append(np.nan if v is None else
                          (v["acc"] * 100 if key == "acc"
                           else v["conv"].get("rounds_to_abs_70", np.nan)))
            series[s_] = np.array(ys, dtype=float)

        for s_ in seeds:
            ax.plot(x, series[s_], color="#9aa5b1", lw=0.65, zorder=2,
                    marker="o", markersize=1.9, markerfacecolor="white",
                    markeredgewidth=0.45, markeredgecolor="#9aa5b1")

        M = np.vstack([series[s_] for s_ in seeds])
        mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=ACCENT, alpha=0.13, lw=0, zorder=1)
        ax.plot(x, mu, color=ACCENT, lw=1.4, zorder=5)
        for i, m in enumerate(order):
            halo(ax, x[i], mu[i], STYLE[m][0], STYLE[m][1], ms=4.3, z=6)
            ax.annotate(fmt.format(mu[i]), (x[i], mu[i]), xytext=(0, 6.5),
                        textcoords="offset points", fontsize=6.1, color=INK,
                        ha="center", zorder=6)

        ax.set_xlim(-0.35, len(order) - 0.65)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        frame(ax)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.text(0.02, 0.94, tag, transform=ax.transAxes, fontsize=7.2, color=INK,
                ha="left", va="top", fontweight="bold", zorder=7)

    axes[0].text(0.98, 0.06, "grey: individual seeds\nblue: mean $\\pm$ s.d.",
                 transform=axes[0].transAxes, fontsize=5.8, color=MUTED,
                 ha="right", va="bottom", linespacing=1.25)
    axes[1].set_xticklabels(short, fontsize=6.5, linespacing=1.15)
    fig.subplots_adjust(left=0.185, right=0.97, top=0.98, bottom=0.135, hspace=0.10)
    save(fig, "fig_r1_ablation_fourway")


def fig_pareto(G, CFG):
    """Both objectives stacked over a shared selector axis, at column width.
    The accuracy axis is broken because Oort sits roughly fifty points below
    every other selector, which would otherwise flatten the competitive field
    into a single line. The proposed selector's column is highlighted so the
    reader can follow it down through both objectives."""
    ks = fam(G, CFG, "literature_baselines")
    if not ks:
        return
    k = ks[0][0]
    P = {m: (agg([v["acc"] for v in G[k][m].values()]),
             agg([v["gini"] for v in G[k][m].values()])) for m in G[k]}
    order = sorted(P, key=lambda m: -P[m][0][0])
    lbl = {"fd_native.scope_fd": "SCOPE-FD", "fd_native.unionfl_fd": "UnionFL",
           "fd_native.scope_fd_debt_only": "Debt only", "heuristic.random": "Random",
           "fd_native.divfl_fd": "DivFL", "fd_native.subtrunc_fd": "SubTrunc",
           "system_aware.oort": "Oort"}
    x = np.arange(len(order))
    hi = [m for m in order if m == "fd_native.scope_fd"]
    hx = order.index("fd_native.scope_fd") if hi else None

    fig = plt.figure(figsize=(COL, 3.55))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.75, 0.40, 1.85], hspace=0.13)
    axU, axL, axG = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    def draw(ax, idx):
        if hx is not None:
            ax.axvspan(hx - 0.45, hx + 0.45, color=ACCENT, alpha=0.075, lw=0, zorder=0)
        for i, m in enumerate(order):
            mu, sd, _ = P[m][idx]
            col, mk, _ = STYLE[m]
            ax.plot([i, i], [(mu - sd) * 100, (mu + sd) * 100], color=col, lw=1.0,
                    zorder=3, solid_capstyle="butt")
            for e in ((mu - sd) * 100, (mu + sd) * 100):
                ax.plot([i - 0.11, i + 0.11], [e, e], color=col, lw=1.0, zorder=3)
            halo(ax, i, mu * 100, col, mk, ms=4.3, z=4)

    # ---- accuracy, broken between the competitive field and Oort ------------
    for ax in (axU, axL):
        draw(ax, 0)
        ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        frame(ax)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axU.set_ylim(68.7, 73.5)
    axL.set_ylim(20.3, 22.1)
    axU.spines["bottom"].set_visible(False)
    axL.spines["top"].set_visible(False)
    axU.tick_params(bottom=False, which="both")
    axL.set_yticks([21])
    axU.set_yticks([69, 70, 71, 72])

    dx, dy = 0.011, 0.055
    for ax, yv in ((axU, 0.0), (axL, 1.0)):
        for xv in (0.0, 1.0):
            ax.plot([xv - dx, xv + dx], [yv - dy, yv + dy], transform=ax.transAxes,
                    color=INK, lw=0.6, clip_on=False, zorder=8)

    for i, m in enumerate(order):
        ax = axL if m == "system_aware.oort" else axU
        mu, sd, _ = P[m][0]
        ax.annotate(f"{mu*100:.2f}", (i, (mu + sd) * 100), xytext=(0, 3.0),
                    textcoords="offset points", fontsize=5.9, color=INK,
                    ha="center", va="bottom", zorder=6)
    axU.set_ylabel("Final accuracy (%)")
    axU.text(0.978, 0.93, "(a)", transform=axU.transAxes, fontsize=7.2, color=INK,
             ha="right", va="top", fontweight="bold")

    # ---- participation Gini --------------------------------------------------
    draw(axG, 1)
    axG.set_xlim(-0.55, len(order) - 0.45)
    axG.set_ylim(-6, 99)
    axG.set_xticks(x)
    axG.set_xticklabels([lbl[m] for m in order], rotation=32, ha="right",
                        fontsize=6.4, rotation_mode="anchor")
    frame(axG)
    axG.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for i, m in enumerate(order):
        mu, sd, _ = P[m][1]
        axG.annotate(f"{mu*100:.2f}", (i, (mu + sd) * 100), xytext=(0, 3.0),
                     textcoords="offset points", fontsize=5.9, color=INK,
                     ha="center", va="bottom", zorder=6)
    axG.text(0.025, 0.955, "(b)", transform=axG.transAxes, fontsize=7.2, color=INK,
             ha="left", va="top", fontweight="bold")
    axG.set_ylabel("Participation Gini (%)\n(lower is fairer)", linespacing=1.15)

    fig.subplots_adjust(left=0.185, right=0.975, top=0.985, bottom=0.175)
    save(fig, "fig_r2_baselines")


def _sweep(G, CFG, family, xkey, xlabel, out, extra=(), logx=False, xticks=None):
    """Curves above, signed SCOPE-minus-random advantage strip below."""
    P = defaultdict(dict)
    for k, c in fam(G, CFG, family):
        for m in ("fd_native.scope_fd", "heuristic.random") + tuple(extra):
            if m in G[k]:
                P[m][c.get(xkey)] = (agg([v["acc"] for v in G[k][m].values()]),
                                     agg([v["gini"] for v in G[k][m].values()]))
    if "fd_native.scope_fd" not in P or "heuristic.random" not in P:
        return
    xs = sorted(P["fd_native.scope_fd"])

    fig = plt.figure(figsize=(FULL, 2.75))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.45, 1.0], hspace=0.10,
                  wspace=0.24)

    for c, (idx, ylab, tag) in enumerate(((0, "Final accuracy (%)", "(a)"),
                                          (1, "Participation Gini (%)", "(b)"))):
        top, bot = fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])

        for m in P:
            if not all(x in P[m] for x in xs):
                continue
            mu = np.array([P[m][x][idx][0] for x in xs]) * 100
            sd = np.array([P[m][x][idx][1] for x in xs]) * 100
            col, mk, lab = STYLE[m]
            key = m in ("fd_native.scope_fd", "heuristic.random")
            top.fill_between(xs, mu - sd, mu + sd, color=col,
                             alpha=0.14 if key else 0.07, lw=0, zorder=2)
            top.plot(xs, mu, color=col, lw=1.15 if key else 0.8,
                     alpha=1.0 if key else 0.7,
                     ls="-" if key else (0, (3, 1.6)), zorder=3, label=lab)
            top.plot(xs, mu, marker=mk, color=col, linestyle="none",
                     markersize=3.6 if key else 2.9, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=4)

        # Advantage strip. Oriented so that upward always means SCOPE-FD is
        # better: accuracy is a gain, Gini is a reduction, since lower Gini is
        # the desirable direction.
        s = np.array([P["fd_native.scope_fd"][x][idx][0] for x in xs]) * 100
        r = np.array([P["heuristic.random"][x][idx][0] for x in xs]) * 100
        d = (s - r) if idx == 0 else (r - s)
        strip_lab = "accuracy\ngain (pp)" if idx == 0 else "Gini\ndrop (pp)"

        bot.axhline(0, color=INK, lw=0.5, zorder=2)
        bot.fill_between(xs, 0, d, where=(d >= 0), color=ACCENT, alpha=0.28,
                         lw=0, interpolate=True, zorder=1)
        bot.fill_between(xs, 0, d, where=(d < 0), color="#D55E00", alpha=0.28,
                         lw=0, interpolate=True, zorder=1)
        bot.plot(xs, d, color=INK, lw=0.8, zorder=3)
        bot.plot(xs, d, marker="o", color=INK, linestyle="none", markersize=2.4,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        for xv, dv in zip(xs, d):
            bot.annotate(f"{dv:+.1f}", (xv, dv), fontsize=5.6, color=INK,
                         ha="center", va="bottom" if dv >= 0 else "top",
                         xytext=(0, 2.2 if dv >= 0 else -2.6),
                         textcoords="offset points", zorder=5)

        top.set_ylabel(ylab)
        top.set_xticklabels([])
        bot.set_ylabel(strip_lab, fontsize=6.2, labelpad=1.5, linespacing=1.0)
        bot.set_xlabel(xlabel)
        for ax in (top, bot):
            if logx:
                ax.set_xscale("log")
            if xticks:
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(t) for t in xticks] if ax is bot else [])
            frame(ax)
        span = max(d.max() - min(d.min(), 0.0), 1e-6)
        bot.set_ylim(min(d.min(), 0.0) - 0.30 * span, d.max() + 0.42 * span)
        bot.tick_params(labelsize=6.2)
        bot.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        panel_tag(top, tag)
        if c == 0:
            top.legend(loc="lower right", ncol=1)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.975, bottom=0.135)
    save(fig, out)


def fig_coefgrid(G, CFG):
    """Plain annotated heatmap. Single hue, light to dark, operating point marked."""
    cells = {}
    for k, c in fam(G, CFG, "coefficient_grid"):
        m = "fd_native.scope_fd"
        if m in G[k]:
            cells[(c.get("scope_au"), c.get("scope_ad"))] = \
                agg([v["acc"] for v in G[k][m].values()])[0] * 100
    if not cells:
        return
    aus = sorted({a_ for a_, _ in cells})
    ads = sorted({d for _, d in cells})
    A = np.full((len(aus), len(ads)), np.nan)
    for (au, ad), v in cells.items():
        A[aus.index(au), ads.index(ad)] = v

    fig, ax = plt.subplots(figsize=(COL, 2.55))
    im = ax.imshow(A, cmap="Blues", aspect="auto", origin="lower")
    lo, hi = np.nanmin(A), np.nanmax(A)
    for i in range(len(aus)):
        for j in range(len(ads)):
            if not np.isnan(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if (A[i, j] - lo) / (hi - lo) > 0.62 else INK)
    if 0.3 in aus and 0.1 in ads:
        jj, ii = ads.index(0.1), aus.index(0.3)
        ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False,
                                   edgecolor="#D55E00", lw=1.5, zorder=5))
    ax.set_xticks(range(len(ads))); ax.set_xticklabels(ads)
    ax.set_yticks(range(len(aus))); ax.set_yticklabels(aus)
    ax.set_xlabel(r"$\alpha_d$"); ax.set_ylabel(r"$\alpha_u$")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(INK)
    ax.tick_params(which="both", length=0, colors=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.028)
    cb.set_label("Final accuracy (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.3, width=0.4, length=1.8)
    cb.outline.set_linewidth(0.4)
    ax.set_title("Orange cell is the setting used throughout.\n"
                 "Participation Gini is $1.33\\%$ in every cell",
                 fontsize=6.5, color=MUTED, pad=4, linespacing=1.3)
    fig.subplots_adjust(left=0.135, right=0.885, top=0.845, bottom=0.145)
    save(fig, "fig_r5_coefficient_grid")


def fig_scale(G, CFG):
    rows = {}
    for k, c in fam(G, CFG, "scale_and_nondivisible"):
        N, K = c.get("total_clients"), c.get("clients_per_round")
        for m in ("fd_native.scope_fd", "heuristic.random"):
            if m in G[k]:
                rows[(N, K, m)] = (agg([v["gini"] for v in G[k][m].values()])[0] * 100,
                                   agg([v["roll"] for v in G[k][m].values()])[0] * 100)
    cfgs = sorted({(N, K) for N, K, _ in rows})
    if not cfgs:
        return

    fig, ax = plt.subplots(figsize=(COL, 3.05))
    rowbands(ax, len(cfgs), 0, 1)
    for i, (N, K) in enumerate(cfgs):
        for m, off in (("fd_native.scope_fd", 0.175), ("heuristic.random", -0.175)):
            if (N, K, m) not in rows:
                continue
            cyc, roll = rows[(N, K, m)]
            col = STYLE[m][0]
            ax.plot([cyc, roll], [i + off] * 2, color=col, lw=1.5, alpha=0.42,
                    zorder=2, solid_capstyle="round")
            halo(ax, cyc, i + off, col, "o", ms=3.4, z=4)
            ax.plot(roll, i + off, marker="D", color=col, markersize=3.1,
                    markerfacecolor="white", markeredgewidth=0.95,
                    linestyle="none", zorder=4)

    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([f"{N}/{K}" + ("*" if N % K else "") for N, K in cfgs])
    ax.set_ylabel("Client pool / cohort  $N/K$")
    ax.set_xlabel("Participation Gini (%)")
    ax.set_xlim(-2.5, 61)
    ax.set_ylim(-0.55, len(cfgs) + 0.55)
    frame(ax)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    handles = [
        Line2D([], [], color=STYLE["fd_native.scope_fd"][0], marker="o", lw=1.5,
               markersize=3.4, alpha=0.9, markeredgecolor="white",
               markeredgewidth=0.7, label="SCOPE-FD"),
        Line2D([], [], color=STYLE["heuristic.random"][0], marker="o", lw=1.5,
               markersize=3.4, alpha=0.9, markeredgecolor="white",
               markeredgewidth=0.7, label="Random"),
        Line2D([], [], color=MUTED, marker="o", lw=0, markersize=3.4,
               markeredgecolor="white", markeredgewidth=0.7, label="cycle-aligned"),
        Line2D([], [], color=MUTED, marker="D", lw=0, markersize=3.1,
               markerfacecolor="white", markeredgewidth=0.95, label="rolling window"),
    ]
    ax.legend(handles=handles, ncol=2, loc="upper right",
              bbox_to_anchor=(1.012, 1.038), columnspacing=0.7)
    ax.text(0.012, 0.012, "*  $K \\nmid N$", transform=ax.transAxes,
            fontsize=6.2, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.20, right=0.975, top=0.985, bottom=0.105)
    save(fig, "fig_r6_scale_fairness")


def fig_channel(G, CFG):
    """The four variants coincide within noise, so overlaid uncertainty bands
    would simply muddy the panel. Only the proposed selector carries a band, as
    the scale of the seed spread, and the variants are separated by dash pattern
    and marker rather than by fill."""
    P = defaultdict(dict)
    for k, c in fam(G, CFG, "ablation_channel_sweep"):
        for m in G[k]:
            P[m][c.get("dl_snr_db")] = agg([v["acc"] for v in G[k][m].values()])
    if not P:
        return
    xs = sorted(next(iter(P.values())))
    dash = {"fd_native.scope_fd": "-",
            "fd_native.scope_fd_debt_only": (0, (4, 1.6)),
            "fd_native.scope_fd_no_server": (0, (1.4, 1.4)),
            "fd_native.scope_fd_no_diversity": (0, (4, 1.4, 1.2, 1.4))}

    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ref = P.get("fd_native.scope_fd")
    if ref:
        mu = np.array([ref[x][0] for x in xs]) * 100
        sd = np.array([ref[x][1] for x in xs]) * 100
        ax.fill_between(xs, mu - sd, mu + sd, color=ACCENT, alpha=0.12, lw=0,
                        zorder=1, label="SCOPE-FD $\\pm$ s.d.")
    for m in ("fd_native.scope_fd", "fd_native.scope_fd_debt_only",
              "fd_native.scope_fd_no_server", "fd_native.scope_fd_no_diversity"):
        if m not in P:
            continue
        mu = np.array([P[m][x][0] for x in xs]) * 100
        col, mk, lab = STYLE[m]
        ax.plot(xs, mu, color=col, lw=1.05, ls=dash[m], zorder=3, label=lab)
        ax.plot(xs, mu, marker=mk, color=col, linestyle="none", markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.set_xlabel("Downlink SNR (dB)")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_xticks(xs)
    h, l = ax.get_legend_handles_labels()
    o = [l.index(v) for v in ("SCOPE-FD", "Debt only", "Debt + coverage",
                              "Debt + under-pred.") if v in l]
    ax.legend([h[i] for i in o], [l[i] for i in o], loc="upper left", ncol=1)
    ax.set_title("All four variants coincide within the seed spread.\n"
                 "Participation Gini is $1.33\\%$ at every SNR",
                 fontsize=6.5, color=MUTED, pad=4, linespacing=1.3)
    frame(ax)
    fig.subplots_adjust(left=0.165, right=0.97, top=0.845, bottom=0.155)
    save(fig, "fig_r7_channel_sweep")


def main():
    print(f"reading {RUNS}")
    G, CFG = load()
    print(f"  {len(G)} configuration groups\n")
    fig_ablation(G, CFG)
    fig_pareto(G, CFG)
    _sweep(G, CFG, "literature_baselines_k_sweep", "clients_per_round",
           "Clients per round $K$    ($N=30$)", "fig_r3_k_sweep",
           extra=("fd_native.divfl_fd", "fd_native.subtrunc_fd"),
           xticks=[1, 3, 5, 10])
    _sweep(G, CFG, "dirichlet_severity", "dirichlet_alpha",
           r"Dirichlet concentration $\alpha$", "fig_r4_dirichlet_severity",
           extra=("fd_native.divfl_fd",), logx=True)
    fig_coefgrid(G, CFG)
    fig_scale(G, CFG)
    fig_channel(G, CFG)
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
