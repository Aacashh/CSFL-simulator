"""Build the selector-convergence verification figure (Statements 1 and 2).

Consumes the per-round selector logs written by run_convergence.sh:
    runs/maml_select_convergence/selector_convergence_{fashion,cifar10,cifar100}.jsonl

Each line holds: round, l_sup_before, l_sup_after, l_sup_descent, l_query,
meta_grad_norm, dphi_norm.

Produces, into the overleaf package images/ directory:
    fig_convergence_verification_boxed.{pdf,eps,png}

Three panels:
  (a) Inner-step support descent  g_t(phi') - g_t(phi)  <= 0  (Statement 1).
  (b) Query objective vs round, normalized to round 1        (Statement 2).
  (c) Meta-update magnitude ||phi_{t+1} - phi_t|| vs round   (stabilization).

No training is run. If a dataset log is missing it is skipped with a message.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PKG = REPO_ROOT / "csfl_simulator" / "Paper Corrections" / "overleaf_maml_select_package" / "images"
DEFAULT_RUNS = REPO_ROOT / "csfl_simulator" / "runs" / "maml_select_convergence"

# dataset tag -> (display label, colour)
DATASETS = [
    ("fashion", "Fashion-MNIST", "#0072B2"),
    ("cifar10", "CIFAR-10", "#D55E00"),
    ("cifar100", "CIFAR-100", "#009E73"),
]
SPINE = "#222222"
GRID = "#E6E6E6"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "bold",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.0,
        "axes.labelweight": "bold",
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def _boxed(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(width=0.9, length=3, color=SPINE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)


def _load(runs_dir: Path, tag: str):
    path = runs_dir / f"selector_convergence_{tag}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["round"]))
    arr = {k: np.array([float(r.get(k, np.nan)) for r in rows]) for k in rows[0]}
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--output-dir", type=Path, default=PKG)
    args = ap.parse_args()
    _style()

    loaded = [(tag, lbl, col, _load(args.runs_dir, tag)) for tag, lbl, col in DATASETS]
    available = [x for x in loaded if x[3] is not None]
    if not available:
        print(f"[no data] No selector_convergence_*.jsonl found in {args.runs_dir}.")
        print("Writing an obvious PLACEHOLDER so the supplement still compiles.")
        print("Run run_convergence.sh on the CUDA machine, then re-run this script to replace it.")
        fig, ax = plt.subplots(figsize=(7.2, 2.2))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "PLACEHOLDER — selector-convergence figure\n"
                "Regenerate after run_convergence.sh completes on the CUDA machine.",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#B00020")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "eps", "png"):
            fig.savefig(args.output_dir / f"fig_convergence_verification_boxed.{ext}", bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) inner-step support descent: should be <= 0 every round (Statement 1)
    axa = axes[0]
    ok_fracs = []
    for tag, lbl, col, arr in available:
        d = arr["l_sup_descent"]
        d = d[~np.isnan(d)]
        if d.size == 0:
            continue
        axa.plot(np.arange(1, d.size + 1), d, color=col, lw=1.6, label=lbl)
        ok_fracs.append(float(np.mean(d <= 1e-9)))
    axa.axhline(0.0, color="#999999", lw=1.0, ls="--", zorder=1)
    axa.set_xlabel("Communication round")
    axa.set_ylabel(r"$g_t(\phi'_t)-g_t(\phi_t)$")
    axa.set_title("(a) Inner-step descent")
    _boxed(axa)
    if ok_fracs:
        lo, hi = 100 * min(ok_fracs), 100 * max(ok_fracs)
        frac_txt = f"{lo:.0f}%" if abs(hi - lo) < 0.5 else f"{lo:.0f}-{hi:.0f}%"
        axa.text(0.96, 0.06, f"$\\leq 0$ in {frac_txt} of rounds",
                 transform=axa.transAxes, ha="right", va="bottom", fontsize=7.6, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.5))

    # (b) query objective, normalized to round 1 (Statement 2 -> outer convergence)
    axb = axes[1]
    for tag, lbl, col, arr in available:
        q = arr["l_query"]
        q = q[~np.isnan(q)]
        if q.size == 0 or q[0] == 0:
            continue
        axb.plot(np.arange(1, q.size + 1), q / q[0], color=col, lw=1.8, label=lbl)
    axb.set_xlabel("Communication round")
    axb.set_ylabel("Query loss (rel. round 1)")
    axb.set_title("(b) Outer-loop convergence")
    _boxed(axb)

    # (c) meta-update magnitude ||phi_{t+1} - phi_t|| (stabilization)
    axc = axes[2]
    for tag, lbl, col, arr in available:
        dphi = arr["dphi_norm"]
        dphi = dphi[~np.isnan(dphi)]
        if dphi.size == 0:
            continue
        axc.plot(np.arange(1, dphi.size + 1), dphi, color=col, lw=1.8, label=lbl)
    axc.set_xlabel("Communication round")
    axc.set_ylabel(r"$|\phi_{t+1}-\phi_t|_2$")
    axc.set_title("(c) Policy stabilization")
    _boxed(axc)

    axb.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
               fontsize=8.0, handletextpad=0.4, labelspacing=0.3)

    fig.tight_layout(pad=0.5, w_pad=1.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        kw = {"dpi": 400} if ext == "png" else {}
        fig.savefig(args.output_dir / f"fig_convergence_verification_boxed.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote fig_convergence_verification_boxed.{{pdf,eps,png}} to {args.output_dir}")
    print("datasets included:", ", ".join(lbl for _, lbl, _, _ in available))


if __name__ == "__main__":
    main()
