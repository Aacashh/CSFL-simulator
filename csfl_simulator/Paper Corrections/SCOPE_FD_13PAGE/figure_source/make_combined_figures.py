"""Two combined floats for the 13-page SCOPE-FD build.

fig_sweeps  : four panels (a)-(d), Dirichlet sweep then cohort-size sweep.
fig_robust  : three panels (a)-(c) at identical box height, so no panel is
              padded with whitespace to match a taller neighbour.

Panel tags are drawn into the artwork, so the manuscript can refer to
Fig. N(a) and the reader sees the same label on the page.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import make_revision_figures as M
from make_revision_figures import (
    STYLE, INK, MUTED, ACCENT, BAND, FULL,
    agg, fam, frame, rowbands, halo, panel_tag, _spearman,
)

OUT = Path(__file__).resolve().parent / "figures_combined"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------- sweeps ----
def _sweep_pair(G, CFG, family, xkey, xlabel, extra, logx, xticks,
                gs, fig, col0, tags):
    """Draw one family as two columns of the shared grid, accuracy then Gini."""
    P = defaultdict(dict)
    for k, c in fam(G, CFG, family):
        for m in ("fd_native.scope_fd", "heuristic.random") + tuple(extra):
            if m in G[k]:
                P[m][c.get(xkey)] = (agg([v["acc"] for v in G[k][m].values()]),
                                     agg([v["gini"] for v in G[k][m].values()]))
    xs = sorted(P["fd_native.scope_fd"])

    for j, (idx, ylab) in enumerate(((0, "Final accuracy (%)"),
                                     (1, "Participation Gini (%)"))):
        c = col0 + j
        top, bot = fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])

        for m in P:
            if not all(x in P[m] for x in xs):
                continue
            mu = np.array([P[m][x][idx][0] for x in xs]) * 100
            sd = np.array([P[m][x][idx][1] for x in xs]) * 100
            colr, mk, lab = STYLE[m]
            key = m in ("fd_native.scope_fd", "heuristic.random")
            top.fill_between(xs, mu - sd, mu + sd, color=colr,
                             alpha=0.14 if key else 0.07, lw=0, zorder=2)
            top.plot(xs, mu, color=colr, lw=1.15 if key else 0.8,
                     alpha=1.0 if key else 0.7,
                     ls="-" if key else (0, (3, 1.6)), zorder=3, label=lab)
            top.plot(xs, mu, marker=mk, color=colr, linestyle="none",
                     markersize=3.4 if key else 2.7, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=4)

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
        bot.plot(xs, d, marker="o", color=INK, linestyle="none", markersize=2.2,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        for xv, dv in zip(xs, d):
            bot.annotate(f"{dv:+.1f}", (xv, dv), fontsize=5.0, color=INK,
                         ha="center", va="bottom" if dv >= 0 else "top",
                         xytext=(0, 1.9 if dv >= 0 else -2.3),
                         textcoords="offset points", zorder=5)

        top.set_ylabel(ylab, fontsize=7.0, labelpad=2.0)
        top.tick_params(axis="x", which="both", labelbottom=False)
        bot.set_ylabel(strip_lab, fontsize=5.6, labelpad=1.4, linespacing=1.0)
        bot.set_xlabel(xlabel, fontsize=7.0, labelpad=1.8)
        for ax in (top, bot):
            if logx:
                ax.set_xscale("log")
            if xticks:
                ax.set_xticks(xticks)
                if ax is bot:
                    ax.set_xticklabels([str(t) for t in xticks])
            frame(ax)
            ax.tick_params(labelsize=6.0)
            ax.margins(x=0.085)
        top.tick_params(axis="x", which="both", labelbottom=False)
        span = max(d.max() - min(d.min(), 0.0), 1e-6)
        bot.set_ylim(min(d.min(), 0.0) - 0.34 * span, d.max() + 0.62 * span)
        bot.tick_params(labelsize=5.6)
        bot.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        panel_tag(top, tags[j])
        top.legend(loc="lower right" if idx == 0 else "upper right",
                   ncol=1, fontsize=5.6, borderpad=0.28, labelspacing=0.26,
                   handlelength=1.5, handletextpad=0.42, borderaxespad=0.28)


def fig_sweeps(G, CFG):
    fig = plt.figure(figsize=(FULL, 1.95))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[2.45, 1.0],
                  hspace=0.12, wspace=0.30)
    _sweep_pair(G, CFG, "dirichlet_severity", "dirichlet_alpha",
                r"Dirichlet concentration $\alpha$", ("fd_native.divfl_fd",),
                True, None, gs, fig, 0, ("(a)", "(b)"))
    _sweep_pair(G, CFG, "literature_baselines_k_sweep", "clients_per_round",
                r"Clients per round $K$", ("fd_native.divfl_fd",
                                           "fd_native.subtrunc_fd"),
                False, [1, 3, 5, 10], gs, fig, 2, ("(c)", "(d)"))
    fig.subplots_adjust(left=0.052, right=0.995, top=0.975, bottom=0.175)
    save(fig, "fig_sweeps")


# ------------------------------------------------------------ robustness ----
def _panel_channel(G, CFG, ax):
    P = defaultdict(dict)
    for k, c in fam(G, CFG, "ablation_channel_sweep"):
        for m in G[k]:
            P[m][c.get("dl_snr_db")] = agg([v["acc"] for v in G[k][m].values()])
    xs = sorted(next(iter(P.values())))
    dash = {"fd_native.scope_fd": "-",
            "fd_native.scope_fd_debt_only": (0, (4, 1.6)),
            "fd_native.scope_fd_no_server": (0, (1.4, 1.4)),
            "fd_native.scope_fd_no_diversity": (0, (4, 1.4, 1.2, 1.4))}
    ref = P.get("fd_native.scope_fd")
    mu = np.array([ref[x][0] for x in xs]) * 100
    sd = np.array([ref[x][1] for x in xs]) * 100
    ax.fill_between(xs, mu - sd, mu + sd, color=ACCENT, alpha=0.12, lw=0, zorder=1)
    for m in ("fd_native.scope_fd", "fd_native.scope_fd_debt_only",
              "fd_native.scope_fd_no_server", "fd_native.scope_fd_no_diversity"):
        if m not in P:
            continue
        y = np.array([P[m][x][0] for x in xs]) * 100
        colr, mk, lab = STYLE[m]
        ax.plot(xs, y, color=colr, lw=1.05, ls=dash[m], zorder=3, label=lab)
        ax.plot(xs, y, marker=mk, color=colr, linestyle="none", markersize=3.0,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.set_xlabel("Downlink SNR (dB)", fontsize=7.0, labelpad=1.8)
    ax.set_ylabel("Final accuracy (%)", fontsize=7.0, labelpad=2.0)
    ax.set_xticks(xs)
    h, l = ax.get_legend_handles_labels()
    o = [l.index(v) for v in ("SCOPE-FD", "Debt only", "Debt + coverage",
                              "Debt + under-pred.") if v in l]
    ax.legend([h[i] for i in o], [l[i] for i in o], loc="upper left", ncol=1,
              fontsize=5.8, borderpad=0.3, labelspacing=0.28, handlelength=1.7,
              handletextpad=0.45, borderaxespad=0.3)
    frame(ax)
    ax.tick_params(labelsize=6.2)


def _panel_scale(G, CFG, ax):
    rows = {}
    for k, c in fam(G, CFG, "scale_and_nondivisible"):
        N, K = c.get("total_clients"), c.get("clients_per_round")
        for m in ("fd_native.scope_fd", "heuristic.random"):
            if m in G[k]:
                rows[(N, K, m)] = (agg([v["gini"] for v in G[k][m].values()])[0] * 100,
                                   agg([v["roll"] for v in G[k][m].values()])[0] * 100)
    cfgs = sorted({(N, K) for N, K, _ in rows})
    rowbands(ax, len(cfgs), 0, 1)
    for i, (N, K) in enumerate(cfgs):
        for m, off in (("fd_native.scope_fd", 0.175), ("heuristic.random", -0.175)):
            if (N, K, m) not in rows:
                continue
            cyc, roll = rows[(N, K, m)]
            colr = STYLE[m][0]
            ax.plot([cyc, roll], [i + off] * 2, color=colr, lw=1.4, alpha=0.42,
                    zorder=2, solid_capstyle="round")
            halo(ax, cyc, i + off, colr, "o", ms=3.1, z=4)
            ax.plot(roll, i + off, marker="D", color=colr, markersize=2.8,
                    markerfacecolor="white", markeredgewidth=0.9,
                    linestyle="none", zorder=4)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([f"{N}/{K}" + ("*" if N % K else "") for N, K in cfgs])
    ax.set_ylabel("Client pool / cohort  $N/K$", fontsize=7.0, labelpad=2.0)
    ax.set_xlabel("Participation Gini (%)", fontsize=7.0, labelpad=1.8)
    ax.set_xlim(-2.5, 61)
    ax.set_ylim(-0.6, len(cfgs) + 1.35)
    frame(ax)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(labelsize=6.2)
    handles = [
        Line2D([], [], color=STYLE["fd_native.scope_fd"][0], marker="o", lw=1.4,
               markersize=3.1, markeredgecolor="white", markeredgewidth=0.7,
               label="SCOPE-FD"),
        Line2D([], [], color=STYLE["heuristic.random"][0], marker="o", lw=1.4,
               markersize=3.1, markeredgecolor="white", markeredgewidth=0.7,
               label="Random"),
        Line2D([], [], color=MUTED, marker="o", lw=0, markersize=3.1,
               markeredgecolor="white", markeredgewidth=0.7, label="cycle-aligned"),
        Line2D([], [], color=MUTED, marker="D", lw=0, markersize=2.8,
               markerfacecolor="white", markeredgewidth=0.9, label="rolling window"),
    ]
    ax.legend(handles=handles, ncol=2, loc="upper right",
              bbox_to_anchor=(0.995, 0.975), borderaxespad=0.28,
              columnspacing=0.6, fontsize=5.6, borderpad=0.28,
              labelspacing=0.26, handlelength=1.4, handletextpad=0.4)
    ax.text(0.012, 0.012, "*  $K \\nmid N$", transform=ax.transAxes,
            fontsize=5.8, color=MUTED, ha="left", va="bottom")


def _panel_ratio(G, CFG, ax):
    # Aggregation mirrors scope_fd_data_verification/ratio.py exactly, namely
    # pool every eligible family by (N, K) keyed on seed, then take the paired
    # per-seed difference. Anything else disagrees with the numbers in the text.
    EXCL = ("dropout", "channel_energy", "public_dataset_sensitivity",
            "histogram_privacy", "bounded_staleness", "coefficient_grid",
            "ablation_channel_sweep")
    order = ["fd_native.scope_fd", "fd_native.scope_fd_debt_only"]
    RND = "heuristic.random"
    pool = defaultdict(lambda: defaultdict(dict))
    for k in G:
        c = CFG[k]
        if c.get("dataset") != "Fashion-MNIST" or c.get("dirichlet_alpha") != 0.5:
            continue
        if k[0] in EXCL:
            continue
        N, K = c.get("total_clients"), c.get("clients_per_round")
        for m in G[k]:
            for seed, v in G[k][m].items():
                if v.get("acc") is not None:
                    pool[(N, K)][m][seed] = v["acc"] * 100

    rows = {}
    for nk, m in pool.items():
        if RND not in m or "fd_native.scope_fd" not in m:
            continue
        r = {}
        for meth in order:
            if meth not in m:
                continue
            sh = sorted(set(m[meth]) & set(m[RND]))
            if not sh:
                continue
            g = [m[meth][x] - m[RND][x] for x in sh]
            r[meth] = (float(np.mean(g)), float(np.std(g, ddof=1)) if len(g) > 1 else 0.0)
        if r:
            rows[nk] = r

    cfgs = sorted(rows, key=lambda nk: -(nk[1] / nk[0]))
    n_rows = len(cfgs)
    ax.axvspan(-2.5, 0, color="#f7d6d0", alpha=0.55, lw=0, zorder=0)
    rowbands(ax, n_rows, 0, 1)
    ax.axvline(0, color=INK, lw=0.6, zorder=2)
    offs = {"fd_native.scope_fd": 0.14, "fd_native.scope_fd_debt_only": -0.14}
    ratios, deltas = {m: [] for m in order}, {m: [] for m in order}
    for i, (N, K) in enumerate(cfgs):
        y = n_rows - 1 - i
        for m in order:
            if m not in rows[(N, K)]:
                continue
            d, dsd = rows[(N, K)][m]
            colr, mk, _ = STYLE[m]
            ax.plot([d - dsd, d + dsd], [y + offs[m]] * 2, color=colr, lw=1.0, zorder=3)
            halo(ax, d, y + offs[m], colr, mk, ms=3.2, z=4)
            ratios[m].append(K / N)
            deltas[m].append(d)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{N}/{K}" for N, K in reversed(cfgs)])
    ax.set_ylim(-0.7, n_rows + 0.5)
    ax.set_xlim(-2.0, 3.6)
    ax.set_ylabel("Client pool / cohort  $N/K$", fontsize=7.0, labelpad=2.0)
    ax.set_xlabel("Final accuracy minus uniform random (pp)", fontsize=7.0, labelpad=1.8)
    frame(ax)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(labelsize=6.2)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n_rows))
    ax2.set_yticklabels([f"{K/N:.3f}" for N, K in reversed(cfgs)],
                        fontsize=5.4, color=MUTED)
    ax2.set_ylabel("Participation ratio $K/N$", fontsize=6.2, color=MUTED, labelpad=1.6)
    ax2.tick_params(length=0, colors=MUTED)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=1.0,
                      markersize=3.2, markeredgecolor="white", markeredgewidth=0.7,
                      label=STYLE[m][2]) for m in order]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
              borderaxespad=0.28, ncol=1, frameon=True, framealpha=0.85,
              edgecolor="none", facecolor="white", fontsize=5.6, borderpad=0.28,
              labelspacing=0.26, handlelength=1.4, handletextpad=0.4)



def fig_robust(G, CFG):
    """One row, three equal-height boxes, so nothing is padded to fit."""
    fig = plt.figure(figsize=(FULL, 2.28))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _panel_channel(G, CFG, axes[0])
    _panel_scale(G, CFG, axes[1])
    _panel_ratio(G, CFG, axes[2])
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        panel_tag(ax, tag)
    fig.subplots_adjust(left=0.062, right=0.925, top=0.985, bottom=0.145)
    save(fig, "fig_robust")


def main():
    print(f"reading {M.RUNS}")
    G, CFG = M.load()
    print(f"  {len(G)} configuration groups\n")
    fig_sweeps(G, CFG)
    fig_robust(G, CFG)
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
