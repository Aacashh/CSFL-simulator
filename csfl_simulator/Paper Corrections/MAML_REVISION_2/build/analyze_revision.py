#!/usr/bin/env python3
"""Recompute every number the MAML-Select revision needs, from the run logs.

Writes ``revision_numbers.json`` next to this file and prints the same content as
tables.  Nothing here reads ``progress.json``: the contradictory tier/coverage
numbers in the submitted supplementary came from snapshots of *incomplete* runs,
so this script uses ``round_metrics.jsonl`` and ``result.json`` only, and records
the round each figure was read at.

Usage:  python3 analyze_revision.py [repo_root]
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

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
METHOD_ORDER = [
    "FedAvg", "FedCS", "Oort", "TiFL", "FedCor", "CriticalFL", "FedGCS", "MAML-Select",
]


# --------------------------------------------------------------------------- io
class Run:
    def __init__(self, run_dir: str, result: dict, rounds: List[dict]) -> None:
        self.dir = run_dir
        self.result = result
        self.rounds = rounds
        cfg = result.get("simulation", {}).get("config", {})
        self.dataset = cfg.get("dataset")
        self.N = cfg.get("total_clients")
        self.K = cfg.get("clients_per_round")
        self.planned_rounds = cfg.get("rounds")
        self.seed = result.get("seed")
        self.experiment = result.get("experiment_id")
        self.method_key = result.get("method_key")
        self.method = METHOD_NAMES.get(self.method_key)

    @property
    def last_round(self) -> int:
        return int(self.rounds[-1]["round"]) if self.rounds else -1

    def at(self, horizon: int) -> Optional[dict]:
        rows = [r for r in self.rounds if int(r["round"]) <= horizon]
        return rows[-1] if rows else None

    def last_evaluated(self, horizon: Optional[int] = None) -> Optional[dict]:
        rows = [
            r for r in self.rounds
            if r.get("evaluated") and (horizon is None or int(r["round"]) <= horizon)
        ]
        return rows[-1] if rows else None


def load_runs(root: str) -> List[Run]:
    runs: List[Run] = []
    for base in ("runs", "csfl_simulator"):
        top = os.path.join(root, base)
        if not os.path.isdir(top):
            continue
        for dirpath, _dirnames, filenames in os.walk(top):
            if "result.json" not in filenames or "round_metrics.jsonl" not in filenames:
                continue
            try:
                result = json.load(open(os.path.join(dirpath, "result.json")))
            except (OSError, ValueError):
                continue
            rows = []
            with open(os.path.join(dirpath, "round_metrics.jsonl")) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
            if rows:
                runs.append(Run(dirpath, result, rows))
    return runs


# ------------------------------------------------------------------ statistics
def mean_sd(values: Iterable[float], places: int = 2) -> str:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{st.fmean(vals):.{places}f}+-{st.stdev(vals):.{places}f}"


def jain(counts: Iterable[int], population: int) -> float:
    counts = list(counts)
    s1 = sum(counts)
    s2 = sum(c * c for c in counts)
    return (s1 * s1) / (population * s2) if s2 else float("nan")


# ------------------------------------------------------- benchmark aggregation
def benchmark_table(runs: List[Run], experiment: str) -> dict:
    """Per-method summary at the horizon every run in the experiment reaches."""
    group = [r for r in runs if r.experiment == experiment and r.method]
    if not group:
        return {}
    horizon = min(
        max((int(x["round"]) for x in r.rounds if x.get("evaluated")), default=-1)
        for r in group
    )
    truncated = {
        r.method: r.last_round
        for r in group
        if r.planned_rounds and r.last_round < r.planned_rounds - 1
    }
    per_method: Dict[str, dict] = {}
    for method in METHOD_ORDER:
        members = [r for r in group if r.method == method]
        if not members:
            continue
        acc, energy, tflops, cov, jn, tiers, seeds = [], [], [], [], [], [[], [], []], []
        for run in members:
            ev = run.last_evaluated(horizon)
            at = run.at(horizon)
            if ev is None or at is None:
                continue
            acc.append(ev.get("accuracy", float("nan")) * 100)
            energy.append(at.get("cum_modelled_energy_wh", float("nan")))
            tflops.append(at.get("cum_training_tflops", float("nan")))
            cov.append(at.get("participation_coverage_ratio", float("nan")) * 100)
            jn.append(at.get("fairness_jain", float("nan")))
            for i in range(3):
                tiers[i].append(at.get(f"tier_{i}_selection_rate", float("nan")))
            seeds.append(run.seed)
        if not acc:
            continue
        per_method[method] = {
            "n_seeds": len(acc),
            "seeds": sorted(str(s) for s in seeds),
            "accuracy": mean_sd(acc),
            "energy_wh": mean_sd(energy, 0),
            "tflops": mean_sd(tflops, 0),
            "coverage_pct": mean_sd(cov, 0),
            "jain": mean_sd(jn, 3),
            "tier_rates": [round(st.fmean(t), 2) for t in tiers],
            # a mean +- sd is meaningless at n = 1; the paper must say so
            "sd_reportable": len(acc) >= 2,
        }
    return {
        "experiment": experiment,
        "dataset": group[0].dataset,
        "matched_horizon_round": horizon,
        "truncated_runs": truncated,
        "methods": per_method,
    }


# ------------------------------------------------- fairness without cold start
def fairness_without_cold_start(runs: List[Run]) -> dict:
    """Coverage and Jain recomputed with the forced cold-start rounds removed.

    The reviewer asks whether the coverage claim is manufactured by the forced
    visit schedule.  The test is to drop those rounds entirely and recompute over
    the policy-driven rounds alone.
    """
    per_dataset: Dict[str, List[dict]] = defaultdict(list)
    for run in runs:
        if run.method_key is None or not str(run.method_key).startswith("research.maml_select"):
            continue
        if not run.N or not run.K:
            continue
        warm = math.ceil(run.N / run.K)
        if len(run.rounds) <= warm + 5:
            continue
        all_seen, post = set(), defaultdict(int)
        for row in run.rounds:
            all_seen.update(row.get("selected_clients", []))
        for row in run.rounds[warm:]:
            for cid in row.get("selected_clients", []):
                post[cid] += 1
        counts = [post.get(i, 0) for i in range(run.N)]
        per_dataset[run.dataset].append({
            "run": os.path.basename(run.dir),
            "seed": run.seed,
            "N": run.N, "K": run.K,
            "cold_start_rounds": warm,
            "rounds": len(run.rounds),
            "coverage_all_pct": 100.0 * len(all_seen) / run.N,
            "coverage_post_pct": 100.0 * len(post) / run.N,
            "jain_all": run.rounds[-1].get("fairness_jain", float("nan")),
            "jain_post": jain(counts, run.N),
        })
    summary = {}
    for dataset, entries in per_dataset.items():
        summary[dataset] = {
            "n_runs": len(entries),
            "coverage_all_pct": round(st.fmean(e["coverage_all_pct"] for e in entries), 1),
            "coverage_post_pct": round(st.fmean(e["coverage_post_pct"] for e in entries), 1),
            "jain_all": round(st.fmean(e["jain_all"] for e in entries), 3),
            "jain_post": round(st.fmean(e["jain_post"] for e in entries), 3),
            "min_coverage_post_pct": round(min(e["coverage_post_pct"] for e in entries), 1),
            "runs": entries,
        }
    return summary


# ---------------------------------------------------------- selector descent
def selector_convergence(root: str) -> dict:
    """Lemma 1 verification and the observed drift of the query objective."""
    base = os.path.join(root, "csfl_simulator", "Paper Corrections", "maml_select_convergence")
    out = {}
    for name in ("fashion", "cifar10", "cifar100"):
        path = os.path.join(base, f"selector_convergence_{name}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        descent = [r["l_sup_descent"] for r in rows if r.get("l_sup_descent") == r.get("l_sup_descent")]
        q = [r["l_query"] for r in rows if r.get("l_query") is not None]
        dq = [abs(q[i] - q[i - 1]) for i in range(1, len(q))]
        out[name] = {
            "rounds": len(rows),
            "inner_descent_nonpositive": sum(1 for d in descent if d <= 0),
            "inner_descent_total": len(descent),
            "query_first": round(q[0], 4),
            "query_last": round(q[-1], 4),
            "query_min": round(min(q), 4),
            "query_max": round(max(q), 4),
            "query_monotone": all(q[i] <= q[i - 1] for i in range(1, len(q))),
            "mean_abs_round_change": round(st.fmean(dq), 4),
            "max_abs_round_change": round(max(dq), 4),
            "sum_abs_round_change": round(sum(dq), 2),
        }
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    runs = load_runs(root)
    print(f"loaded {len(runs)} runs with round logs from {root}\n")

    report = {"benchmarks": {}, "fairness_without_cold_start": {}, "selector_convergence": {}}

    for experiment in sorted({r.experiment for r in runs if r.experiment and "benchmark" in str(r.experiment)}):
        table = benchmark_table(runs, experiment)
        if not table:
            continue
        report["benchmarks"][experiment] = table
        print(f"=== {table['dataset']}  [{experiment}]  matched horizon = round {table['matched_horizon_round']}")
        if table["truncated_runs"]:
            print(f"    TRUNCATED RUNS: {table['truncated_runs']}")
        head = f"{'Method':<13}{'n':>2}{'Acc(%)':>16}{'Energy':>12}{'Cov':>7}{'Jain':>8}   tiers"
        print(head + "\n" + "-" * len(head))
        for method, row in table["methods"].items():
            flag = "" if row["sd_reportable"] else "  <-- single seed, no sd"
            print(f"{method:<13}{row['n_seeds']:>2}{row['accuracy']:>16}{row['energy_wh']:>12}"
                  f"{row['coverage_pct']:>7}{row['jain']:>8}   {row['tier_rates']}{flag}")
        print()

    fair = fairness_without_cold_start(runs)
    report["fairness_without_cold_start"] = fair
    if fair:
        print("=== Coverage and Jain with the forced cold-start rounds removed")
        head = f"{'Dataset':<16}{'n':>3}{'cov_all':>9}{'cov_post':>10}{'jain_all':>10}{'jain_post':>11}"
        print(head + "\n" + "-" * len(head))
        for dataset, row in sorted(fair.items()):
            print(f"{dataset:<16}{row['n_runs']:>3}{row['coverage_all_pct']:>8.1f}%"
                  f"{row['coverage_post_pct']:>9.1f}%{row['jain_all']:>10.3f}{row['jain_post']:>11.3f}")
        print()

    conv = selector_convergence(root)
    report["selector_convergence"] = conv
    if conv:
        print("=== Selector convergence logs")
        head = f"{'Dataset':<10}{'rounds':>7}{'inner descent<=0':>18}{'q monotone':>12}{'mean|dq|':>10}{'max|dq|':>9}"
        print(head + "\n" + "-" * len(head))
        for name, row in conv.items():
            print(f"{name:<10}{row['rounds']:>7}"
                  f"{row['inner_descent_nonpositive']:>10}/{row['inner_descent_total']:<7}"
                  f"{str(row['query_monotone']):>12}{row['mean_abs_round_change']:>10.4f}"
                  f"{row['max_abs_round_change']:>9.4f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision_numbers.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
