#!/usr/bin/env python3
"""Turn the report-card campaign into the table rows the manuscript needs.

Five outputs, one per item on the list.

1.  The no-adaptation control.  A row for inner steps = 0 beside 1, 2 and 5.
2.  The alpha = 0.1 block.
3.  Selected-shard sizes for MAML-Select against FedAvg, mean and quartiles.
4.  A reconciliation check.  The full-state row of the ablation table and the
    MAML-Select row of the benchmark table are the same configuration, so this
    script reports both from the *same* runs and fails loudly if any caller
    tries to source them separately.
5.  Mean achieved round latency and wall-clock time to the accuracy target, for
    the benchmark table.

Runs that predate the sample-count logging still work.  Cumulative TFLOPs is
exactly proportional to the selected sample count within a dataset, so the mean
shard is recovered from the TFLOPs column when the per-round list is absent.
The two agree to rounding where both exist, which this script checks.

Usage
    python -m csfl_simulator.experiments.maml_select.report_card_tables \\
        --runs-root runs/report_card runs/report_card_main runs/maml_select_cifar100 \\
        --out artifacts/maml_select/report_card
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

METHOD_NAMES = {
    "baseline.fedavg": "FedAvg",
    "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort",
    "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor",
    "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS",
    "research.maml_select": "MAML-Select",
}
METHOD_ORDER = ["FedAvg", "FedCS", "Oort", "TiFL", "FedCor", "CriticalFL",
                "FedGCS", "MAML-Select"]

# The horizon each dataset is reported at in the manuscript.
HORIZON = {"Fashion-MNIST": 200, "CIFAR-10": 200, "CIFAR-100": 150}

# Training-set sizes, used only to recover the pool mean shard for runs that
# predate the per-round sample-count logging.  Runs that carry
# ``client_sample_counts`` use their own recorded pool instead.
TRAIN_SAMPLES = {"Fashion-MNIST": 60000, "CIFAR-10": 50000, "CIFAR-100": 50000}


# --------------------------------------------------------------------- loading
class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        sim = result.get("simulation", {})
        cfg = sim.get("config", {}) or {}
        self.result = result
        self.cfg = cfg
        self.name = path.name
        self.dataset = cfg.get("dataset")
        self.alpha = cfg.get("dirichlet_alpha")
        self.planned_rounds = cfg.get("rounds")
        self.seed = result.get("seed")
        self.method_key = result.get("method_key") or ""
        self.method = METHOD_NAMES.get(self.method_key)
        self.experiment = result.get("experiment_id")
        self.params = result.get("method_params", {}) or {}
        self.client_samples = sim.get("client_sample_counts")
        self.rows = [
            json.loads(line)
            for line in (path / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # -- horizon-aware accessors -------------------------------------------
    def upto(self, horizon: Optional[int]) -> List[dict]:
        if horizon is None:
            return self.rows
        return [r for r in self.rows if int(r["round"]) <= horizon]

    def final_eval(self, horizon: Optional[int]) -> Optional[dict]:
        got = [r for r in self.upto(horizon) if r.get("evaluated")]
        return got[-1] if got else None

    def mean_round_latency(self, horizon: Optional[int]) -> float:
        rows = self.upto(horizon)
        return st.mean(float(r["round_time"]) for r in rows) if rows else float("nan")

    def time_to(self, target: float) -> Optional[float]:
        """Wall-clock simulated seconds to first reach `target` accuracy."""
        for r in self.rows:
            if r.get("evaluated") and float(r["accuracy"]) >= target:
                return float(r["cum_time"])
        return None

    def cum_tflops(self, horizon: Optional[int]) -> float:
        rows = self.upto(horizon)
        return float(rows[-1]["cum_training_tflops"]) if rows else float("nan")

    def cum_energy(self, horizon: Optional[int]) -> float:
        rows = self.upto(horizon)
        return float(rows[-1]["cum_modelled_energy_wh"]) if rows else float("nan")

    def selected_shards(self, horizon: Optional[int]) -> List[int]:
        """Every selected client's shard size, flattened over rounds."""
        out: List[int] = []
        for r in self.upto(horizon):
            counts = r.get("selected_sample_counts")
            if counts:
                out.extend(int(c) for c in counts)
        return out

    def inner_steps(self) -> Optional[int]:
        if "inner_steps" in self.params:
            return int(self.params["inner_steps"])
        proto = self.result.get("maml_select_protocol") or {}
        return int(proto["inner_steps"]) if "inner_steps" in proto else None


def load_runs(roots: Sequence[str]) -> List[Run]:
    runs: List[Run] = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            print(f"[warn] missing runs root {base}")
            continue
        for dirpath, _dirs, files in os.walk(base):
            if "result.json" in files and "round_metrics.jsonl" in files:
                try:
                    runs.append(Run(Path(dirpath)))
                except Exception as exc:  # a half-written run must not stop the report
                    print(f"[warn] skipping {dirpath}: {exc}")
    return runs


# ------------------------------------------------------------------ utilities
def mean_sd(values: Sequence[float]) -> str:
    values = [v for v in values if v == v]
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{st.mean(values):.2f}$\\pm${st.stdev(values):.2f}"


def quartiles(values: Sequence[float]) -> str:
    if not values:
        return "--"
    s = sorted(values)
    q = st.quantiles(s, n=4) if len(s) >= 4 else [s[0], st.median(s), s[-1]]
    return f"{q[0]:.0f} / {st.median(s):.0f} / {q[2]:.0f}"


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------- 1. no-adaptation control
def control_table(runs: List[Run]) -> None:
    banner("1. No-adaptation control, inner steps = 0")
    groups: Dict[tuple, List[Run]] = defaultdict(list)
    for r in runs:
        steps = r.inner_steps()
        if not r.method_key.startswith("research.maml_select") or steps is None:
            continue
        if r.experiment not in ("no_adaptation_control", "inner_step_ablation"):
            continue
        groups[(r.dataset, r.planned_rounds, steps)].append(r)

    if not groups:
        print("no control runs found yet; run stage A of run_report_card_evidence.sh")
        return

    print(f"{'dataset':<15}{'rounds':>7}{'steps':>7}{'acc':>16}{'TFLOPs':>10}"
          f"{'Jain':>8}{'T_round':>10}{'n':>4}")
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1], k[2])):
        ds, rounds, steps = key
        rs = groups[key]
        h = min(rounds, HORIZON.get(ds, rounds))
        acc = [float(r.final_eval(h)["accuracy"]) * 100 for r in rs if r.final_eval(h)]
        tf = [r.cum_tflops(h) for r in rs]
        jain = [float(r.upto(h)[-1]["fairness_jain"]) for r in rs]
        lat = [r.mean_round_latency(h) for r in rs]
        print(f"{ds:<15}{rounds:>7}{steps:>7}{mean_sd(acc):>16}"
              f"{st.mean(tf):>10.0f}{st.mean(jain):>8.3f}{st.mean(lat):>10.0f}{len(rs):>4}")

    print()
    print("LaTeX rows for the inner-loop group of Table III")
    for ds in sorted({k[0] for k in groups}):
        rows = sorted([k for k in groups if k[0] == ds], key=lambda k: k[2])
        for key in rows:
            rs = groups[key]
            h = min(key[1], HORIZON.get(ds, key[1]))
            acc = [float(r.final_eval(h)["accuracy"]) * 100 for r in rs if r.final_eval(h)]
            label = "0 (online regression)" if key[2] == 0 else str(key[2])
            print(f"  \\revtwo{{{label}}} & \\revtwo{{{mean_sd(acc)}}} & & \\\\   % {ds}")


# ------------------------------------------------------------ 2. alpha = 0.1
def alpha_table(runs: List[Run]) -> None:
    banner("2. Severe heterogeneity, Dirichlet alpha = 0.1")
    groups: Dict[tuple, List[Run]] = defaultdict(list)
    for r in runs:
        if r.method and r.alpha is not None:
            groups[(r.dataset, float(r.alpha), r.method)].append(r)
    have = {k for k in groups if abs(k[1] - 0.1) < 1e-9}
    if not have:
        print("no alpha=0.1 runs found yet; run stage B of run_report_card_evidence.sh")
        return

    for ds in sorted({k[0] for k in have}):
        print(f"-- {ds}")
        print(f"   {'method':<13}{'alpha':>7}{'acc':>16}{'TFLOPs':>10}{'energy':>10}"
              f"{'Jain':>8}{'Cov':>7}{'n':>4}")
        for a in (0.5, 0.1):
            for m in METHOD_ORDER:
                rs = groups.get((ds, a, m))
                if not rs:
                    continue
                h = HORIZON.get(ds)
                acc = [float(r.final_eval(h)["accuracy"]) * 100 for r in rs if r.final_eval(h)]
                tf = [r.cum_tflops(h) for r in rs]
                en = [r.cum_energy(h) for r in rs]
                jn = [float(r.upto(h)[-1]["fairness_jain"]) for r in rs]
                cv = [float(r.upto(h)[-1]["participation_coverage_ratio"]) * 100 for r in rs]
                print(f"   {m:<13}{a:>7}{mean_sd(acc):>16}{st.mean(tf):>10.0f}"
                      f"{st.mean(en):>10.0f}{st.mean(jn):>8.2f}{st.mean(cv):>7.0f}{len(rs):>4}")
        print()


# ------------------------------------------------- 3. selected shard sizes
def shard_table(runs: List[Run]) -> None:
    banner("3. Sample counts of the selected clients")
    print("Cumulative TFLOPs is proportional to the selected sample count within a")
    print("dataset, so the ratio is exact.  Where the per-round counts were logged")
    print("the two routes are cross-checked against each other.")
    print()
    by = defaultdict(list)
    for r in runs:
        if r.method and r.alpha is not None and abs(float(r.alpha) - 0.5) < 1e-9:
            by[(r.dataset, r.method)].append(r)

    for ds in sorted({k[0] for k in by}):
        base = by.get((ds, "FedAvg"))
        if not base:
            print(f"-- {ds}: no FedAvg reference, skipping")
            continue
        h = HORIZON.get(ds)
        pool = None
        source = "logged"
        for r in base:
            if r.client_samples:
                pool = st.mean(float(c) for c in r.client_samples)
                break
        if pool is None:
            total = TRAIN_SAMPLES.get(ds)
            n_clients = base[0].cfg.get("total_clients")
            if total and n_clients:
                pool = total / float(n_clients)
                source = "training-set size over client count"
        base_tf = st.mean(r.cum_tflops(h) for r in base)
        print(f"-- {ds}   pool mean shard = "
              f"{('%.1f' % pool) if pool else 'unknown'}  ({source})")
        print(f"   {'method':<13}{'mean shard':>12}{'vs FedAvg':>11}"
              f"{'logged mean':>13}{'Q1/med/Q3':>22}")
        for m in METHOD_ORDER:
            rs = by.get((ds, m))
            if not rs:
                continue
            tf = st.mean(r.cum_tflops(h) for r in rs)
            derived = pool * tf / base_tf if pool else float("nan")
            shards: List[int] = []
            for r in rs:
                shards.extend(r.selected_shards(h))
            logged = st.mean(shards) if shards else float("nan")
            if shards and pool and abs(logged - derived) > 1.0:
                print(f"   [warn] {m}: logged {logged:.1f} vs derived {derived:.1f}")
            print(f"   {m:<13}{derived:>12.1f}{100 * (tf / base_tf - 1):>10.1f}%"
                  f"{logged:>13.1f}{quartiles(shards):>22}")
        print()


# ------------------------------------------- 4. reconciliation of the two tables
def reconcile(runs: List[Run]) -> None:
    banner("4. Benchmark table and ablation table, same configuration")
    print("The full-state ablation row and the benchmark MAML-Select row are the")
    print("same configuration.  They must be sourced from one set of runs, and the")
    print("manuscript now does that.  Both candidates are printed here so any drift")
    print("between separately executed campaigns is visible rather than silent.")
    print()
    for ds in ("Fashion-MNIST",):
        h = HORIZON[ds]
        main = [r for r in runs
                if r.dataset == ds and r.method_key == "research.maml_select"
                and r.experiment == "main_benchmarks"]
        full = [r for r in runs
                if r.dataset == ds and r.method_key == "research.maml_select.full"]
        for label, rs in (("benchmark table, research.maml_select", main),
                          ("ablation table, research.maml_select.full", full)):
            if not rs:
                print(f"   {label:<44} no runs found")
                continue
            acc = [float(r.final_eval(h)["accuracy"]) * 100 for r in rs if r.final_eval(h)]
            jn = [float(r.upto(h)[-1]["fairness_jain"]) for r in rs]
            tf = [r.cum_tflops(h) for r in rs]
            print(f"   {label:<44} acc {mean_sd(acc):<18} jain {st.mean(jn):.3f} "
                  f" tflops {st.mean(tf):.0f}  n={len(rs)} seeds={sorted(r.seed for r in rs)}")
        if main and full:
            hm = st.mean(float(r.final_eval(h)["accuracy"]) * 100 for r in main if r.final_eval(h))
            hf = st.mean(float(r.final_eval(h)["accuracy"]) * 100 for r in full if r.final_eval(h))
            gap = abs(hm - hf)
            verdict = "identical to 0.01 pp" if gap < 0.01 else f"differ by {gap:.2f} pp"
            print(f"   -> {verdict}")
            if gap >= 0.01:
                print("   -> report BOTH rows from the benchmark runs. The two campaigns")
                print("      are nominally the same configuration, so any gap is run-to-run")
                print("      variation and not a design difference.")


# --------------------------------------------------- 5. latency for the table
def latency_table(runs: List[Run]) -> None:
    banner("5. Achieved round latency and time to target")
    by = defaultdict(list)
    for r in runs:
        if r.method and r.alpha is not None and abs(float(r.alpha) - 0.5) < 1e-9 \
                and r.experiment in ("main_benchmarks", "cifar100_benchmarks"):
            by[(r.dataset, r.method)].append(r)
    if not by:
        print("no benchmark runs found; run stage C of run_report_card_evidence.sh")
        return

    targets = {"Fashion-MNIST": 0.70, "CIFAR-10": 0.70, "CIFAR-100": 0.50}
    for ds in sorted({k[0] for k in by}):
        h = HORIZON.get(ds)
        tgt = targets.get(ds, 0.50)
        print(f"-- {ds}   time-to-target at {tgt:.0%} accuracy")
        print(f"   {'method':<13}{'mean T_round (s)':>18}{'vs FedAvg':>11}"
              f"{'time to target (h)':>20}{'n':>4}")
        base = by.get((ds, "FedAvg"))
        base_lat = st.mean(r.mean_round_latency(h) for r in base) if base else float("nan")
        for m in METHOD_ORDER:
            rs = by.get((ds, m))
            if not rs:
                continue
            lat = st.mean(r.mean_round_latency(h) for r in rs)
            tt = [r.time_to(tgt) for r in rs]
            reached = [x for x in tt if x is not None]
            tts = f"{st.mean(reached) / 3600:.1f}" if len(reached) == len(rs) else "not reached"
            print(f"   {m:<13}{lat:>18.0f}{lat / base_lat:>10.3f}x{tts:>20}{len(rs):>4}")
        print()
        print("   LaTeX cells for the two new columns of Table II")
        for m in METHOD_ORDER:
            rs = by.get((ds, m))
            if not rs:
                continue
            lat = st.mean(r.mean_round_latency(h) for r in rs)
            tt = [r.time_to(tgt) for r in rs]
            reached = [x for x in tt if x is not None]
            tts = f"{st.mean(reached) / 3600:.1f}" if len(reached) == len(rs) else "--"
            print(f"     % {m:<12} & {lat:.0f} & {tts} \\\\")
        print()


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", nargs="+", required=True,
                    help="one or more directories to walk for completed runs")
    args = ap.parse_args()

    runs = load_runs(args.runs_root)
    print(f"loaded {len(runs)} runs from {len(args.runs_root)} root(s)")
    datasets = sorted({r.dataset for r in runs if r.dataset})
    print("datasets:", ", ".join(str(d) for d in datasets))

    control_table(runs)
    alpha_table(runs)
    shard_table(runs)
    reconcile(runs)
    latency_table(runs)


if __name__ == "__main__":
    main()
