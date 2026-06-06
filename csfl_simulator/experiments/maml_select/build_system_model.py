"""Render a clean MAML-Select system-model / workflow diagram.

This is an additive replacement for the hand-drawn ``MAML.drawio`` / ``MAML.pdf``
figure used as ``Fig.~\\ref{fig:model_diagram}`` in the main letter. It draws a
two-band schematic -- heterogeneous edge clients (top) and the central
MAML-Select server pipeline (bottom) -- as a vector figure with matplotlib, so
it matches the rest of the figure tooling and can be previewed without LaTeX.

Output (new files; the original MAML.pdf/MAML.drawio are left untouched):
  Paper Corrections/MAML__Letter/system_model_v2.pdf
  Paper Corrections/MAML__Letter/system_model_v2.png

Run from the repo root:
    python -m csfl_simulator.experiments.maml_select.build_system_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from csfl_simulator.experiments.maml_select.build_review_visuals import REPO_ROOT

PAPER_DIR = REPO_ROOT / "csfl_simulator" / "Paper Corrections" / "MAML__Letter"

# ----------------------------------------------------------------------------- #
# Palette (soft fills with darker borders; print- and colorblind-friendly)
# ----------------------------------------------------------------------------- #
EDGE_FILL, EDGE_EDGE = "#EAF4EE", "#4E9A6B"      # client band (green)
SERVER_FILL, SERVER_EDGE = "#EEF2F8", "#3B5B7F"  # server band (slate blue)
CLIENT_FILL, CLIENT_EDGE = "#FFFFFF", "#8A8F98"
STORE_FILL, STORE_EDGE = "#F1F3F6", "#5B6470"
P1_FILL, P1_EDGE = "#FDE7D2", "#DD8A37"          # amber
P2_FILL, P2_EDGE = "#E2ECFA", "#3E6DB5"          # blue
P3_FILL, P3_EDGE = "#ECE3F4", "#7E5BA6"          # purple
PIPE = "#2B2F36"                                  # pipeline arrows
BROADCAST = "#2E8B57"                             # server -> clients
TELEMETRY = "#5A6472"                             # clients -> server
LOOP = "#9AA1AC"                                  # meta feedback loop
INK = "#1C2026"


def _box(ax, x0, y0, x1, y1, fc, ec, lw=1.6, radius=1.7):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            mutation_aspect=1.0,
            zorder=2,
        )
    )


def _text(ax, x, y, s, size, weight="normal", color=INK, style="normal", ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight, color=color,
            fontstyle=style, zorder=5, linespacing=1.25)


def _badge(ax, x, y, n, color):
    ax.add_patch(plt.Circle((x, y), 1.45, facecolor=color, edgecolor="white", linewidth=1.2, zorder=6))
    _text(ax, x, y, str(n), 9.5, weight="bold", color="white")


def _arrow(ax, p0, p1, color, lw=2.0, style="-", cs="arc3,rad=0", ms=15, z=4):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
            linestyle=style, connectionstyle=cs, shrinkA=3, shrinkB=3,
            joinstyle="round", capstyle="round", zorder=z,
        )
    )


def build(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # ----- Bands -------------------------------------------------------------- #
    _box(ax, 2, 45.5, 98, 62.5, EDGE_FILL, EDGE_EDGE, lw=1.8, radius=2.2)
    _box(ax, 2, 2.5, 98, 41.5, SERVER_FILL, SERVER_EDGE, lw=1.8, radius=2.2)
    _text(ax, 5.2, 60.2, "Heterogeneous Edge Clients", 9.5, weight="bold",
          color=EDGE_EDGE, ha="left")
    _text(ax, 5.2, 39.2, "Central Server — MAML-Select", 9.5, weight="bold",
          color=SERVER_EDGE, ha="left")

    # ----- Client cards ------------------------------------------------------- #
    clients = [
        (8, 28, "Client 1", "Tier 1 · low-end"),
        (40, 60, "Client i", "Tier 2 · mid-range"),
        (72, 92, "Client N", "Tier 3 · high-end"),
    ]
    for x0, x1, name, tier in clients:
        _box(ax, x0, 47.5, x1, 57.5, CLIENT_FILL, CLIENT_EDGE, lw=1.4, radius=1.4)
        cx = (x0 + x1) / 2
        _text(ax, cx, 53.6, name, 9.5, weight="bold")
        _text(ax, cx, 50.4, tier, 7.6, color="#5A6472")
    _text(ax, 34, 52.5, "⋯", 15, color="#7A828C")
    _text(ax, 66, 52.5, "⋯", 15, color="#7A828C")

    # ----- Server pipeline nodes --------------------------------------------- #
    y0, y1 = 12.5, 30.5
    yc = (y0 + y1) / 2
    # Meta-policy store
    _box(ax, 4.5, y0, 20.5, y1, STORE_FILL, STORE_EDGE, lw=1.5, radius=1.6)
    _text(ax, 12.5, yc + 3.1, "Meta-Policy", 9.3, weight="bold")
    _text(ax, 12.5, yc - 0.2, r"$h_\phi$", 11)
    _text(ax, 12.5, yc - 3.4, "MLP 6-64-64-1", 7.4, color="#5A6472")
    # Phase 1
    _box(ax, 26.5, y0, 46.5, y1, P1_FILL, P1_EDGE, radius=1.6)
    _text(ax, 36.5, yc + 3.0, "Inner Look-Ahead", 9.0, weight="bold")
    _text(ax, 36.5, yc - 1.8, r"$\phi'_t=\phi_t-\beta\,\nabla\mathcal{L}_{\mathrm{sup}}$", 8.4)
    # Phase 2
    _box(ax, 52.5, y0, 72.5, y1, P2_FILL, P2_EDGE, radius=1.6)
    _text(ax, 62.5, yc + 3.0, "Score & Select Top-K", 9.0, weight="bold")
    _text(ax, 62.5, yc - 1.8, r"$\arg\mathrm{TopK}\,[-h_{\phi'_t}(s_{i,t})]$", 8.0)
    # Phase 3
    _box(ax, 78.5, y0, 95.5, y1, P3_FILL, P3_EDGE, radius=1.6)
    _text(ax, 87.0, yc + 3.2, "Aggregate &", 9.0, weight="bold")
    _text(ax, 87.0, yc + 0.4, "Outer Update", 9.0, weight="bold")
    _text(ax, 87.0, yc - 3.1, r"$\phi_{t+1}=\phi_t-\eta\,\nabla\mathcal{L}_{\mathrm{qry}}$", 7.8)

    # phase badges
    _badge(ax, 27.8, y1 - 1.0, 1, P1_EDGE)
    _badge(ax, 53.8, y1 - 1.0, 2, P2_EDGE)
    _badge(ax, 79.8, y1 - 1.0, 3, P3_EDGE)

    # ----- Pipeline arrows (left -> right) ----------------------------------- #
    _arrow(ax, (20.5, yc), (26.5, yc), PIPE, lw=2.2)
    _arrow(ax, (46.5, yc), (52.5, yc), PIPE, lw=2.2)
    _arrow(ax, (72.5, yc), (78.5, yc), PIPE, lw=2.2)

    # ----- Meta feedback loop (Phase 3 -> Meta-Policy) ----------------------- #
    _arrow(ax, (87.0, y0), (12.5, y0), LOOP, lw=2.0, style=(0, (5, 3)),
           cs="arc3,rad=0.18", ms=15, z=1)
    _text(ax, 50, 7.0, "policy carried to round $t{+}1$", 7.6, style="italic", color="#7A828C")

    # ----- Inter-band communication ------------------------------------------ #
    # clients -> server: local updates + state telemetry (down, left side)
    _arrow(ax, (21, 47.5), (33, 30.5), TELEMETRY, lw=2.0, cs="arc3,rad=0.16")
    _text(ax, 12.5, 43.0, "local updates\n+ state $s_{i,t}$", 7.8, color=TELEMETRY, weight="bold")
    # server -> clients: broadcast global model to the selected Top-K (up, right side)
    _arrow(ax, (66, 30.5), (78, 47.5), BROADCAST, lw=2.0, cs="arc3,rad=-0.16")
    _text(ax, 86.5, 43.2, "broadcast $\\theta_t$\nto Top-K", 7.8, color=BROADCAST, weight="bold")

    # ----- Legend for state features (compact) ------------------------------- #
    _text(
        ax, 50, 4.0,
        "state $s_{i,t}$: training loss · gradient norm · latency · battery · selection frequency · staleness",
        6.8, color="#6A727E", style="italic",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--stem", type=str, default="system_model_v2")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    build(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(args.out_dir / f"{args.stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote {args.stem}.pdf and {args.stem}.png to {args.out_dir}")


if __name__ == "__main__":
    main()
