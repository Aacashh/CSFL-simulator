"""Tier-level selection-rate analysis from existing main-benchmark runs.

Each main run's progress.json records tier_0/1/2_selection_rate (fraction of
selections falling in each device tier) and participation_coverage_ratio. This
script aggregates them per (dataset, method) over seeds and prints a LaTeX
tabular body for the supplement, answering the tier-level part of the fairness
comment. No re-run is needed; it reads logs already on disk.

Tiers: Tier 1 = slow (1.0x, 20% of clients), Tier 2 = medium (2.0x, 50%),
Tier 3 = fast (4.0x, 30%).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SCEN = {"fashion_main": "Fashion-MNIST", "cifar10_main": "CIFAR-10", "cifar100_main": "CIFAR-100"}
METHOD_ORDER = ["fedavg", "fedcs", "oort", "tifl", "fedcor", "criticalfl", "fedgcs", "maml_select"]
METHOD_LABEL = {
    "fedavg": "FedAvg", "fedcs": "FedCS", "oort": "Oort", "tifl": "TiFL",
    "fedcor": "FedCor", "criticalfl": "CriticalFL", "fedgcs": "FedGCS",
    "maml_select": "\\textbf{MAML-Select}",
}
DEFAULT_ROOTS = [
    Path.home() / "Desktop" / "main_benchmarks_critical",
    Path.home() / "Desktop" / "maml_select_runs_windows",
]


def _parse(label: str):
    s = label.replace("main_benchmarks_", "")
    for sc in SCEN:
        if s.startswith(sc + "_"):
            return sc, re.sub(r"_s\d+$", "", s[len(sc) + 1:])
    return None, None


def collect(roots):
    tier = defaultdict(lambda: defaultdict(list))
    cov = defaultdict(list)
    for root in roots:
        for p in glob.glob(str(Path(root) / "**" / "progress.json"), recursive=True):
            try:
                lm = json.load(open(p)).get("latest_metrics", {})
                label = json.load(open(p)).get("run_label", "")
            except (OSError, json.JSONDecodeError):
                continue
            sc, m = _parse(label)
            if not sc:
                continue
            for i, key in enumerate(("tier_0_selection_rate", "tier_1_selection_rate", "tier_2_selection_rate")):
                if key in lm:
                    tier[(sc, m)][i].append(float(lm[key]))
            if "participation_coverage_ratio" in lm:
                cov[(sc, m)].append(float(lm["participation_coverage_ratio"]))
    return tier, cov


def latex(tier, cov):
    lines = []
    for sc, sc_label in SCEN.items():
        keys = [(sc, m) for m in METHOD_ORDER if (sc, m) in tier]
        if not keys:
            continue
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{sc_label}}}}} \\\\")
        for (s, m) in keys:
            t = [np.mean(tier[(s, m)][i]) if tier[(s, m)].get(i) else float("nan") for i in range(3)]
            c = 100 * np.mean(cov[(s, m)]) if cov.get((s, m)) else float("nan")
            lines.append(f"{METHOD_LABEL[m]} & {t[0]:.2f} & {t[1]:.2f} & {t[2]:.2f} & {c:.0f} \\\\")
        lines.append("\\addlinespace[1pt]")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+", default=DEFAULT_ROOTS)
    args = ap.parse_args()
    tier, cov = collect(args.roots)
    if not tier:
        print("[no data] no progress.json with tier rates under:", *args.roots)
        return
    print("% --- tier-level selection-rate table body (auto-generated) ---")
    print("% Tier1=slow(1.0x,20%), Tier2=medium(2.0x,50%), Tier3=fast(4.0x,30%)")
    print(latex(tier, cov))


if __name__ == "__main__":
    main()
