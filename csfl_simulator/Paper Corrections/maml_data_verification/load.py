"""Load every MAML-Select run and expose it as a flat table.

Source of truth for every number in the MAML-Select letter. Nothing here reads
the manuscript, so a disagreement between this and the paper is a paper bug.
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs")

# Display names, in the order the benchmark table prints them.
METHOD_ORDER = [
    "FedAvg", "FedCS", "Oort", "TiFL", "FedCor",
    "CriticalFL", "FedGCS", "MAML-Select",
]

METHOD_NAMES = {
    "baseline.fedavg": "FedAvg",
    "system_aware.fedcs": "FedCS",
    "ml.oort": "Oort",
    "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor",
    "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS",
    "research.maml_select": "MAML-Select",
    "ml.maml_select": "MAML-Select",
}


def _method_name(key):
    if key in METHOD_NAMES:
        return METHOD_NAMES[key]
    tail = key.split(".")[-1]
    for k, v in METHOD_NAMES.items():
        if k.split(".")[-1] == tail:
            return v
    return tail


def find_results(subdirs=None):
    """Every result.json under runs/, optionally restricted to subdirectories."""
    out = []
    roots = [os.path.join(RUNS, s) for s in subdirs] if subdirs else [RUNS]
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if "result.json" in filenames:
                out.append(os.path.join(dirpath, "result.json"))
    return sorted(out)


def load_run(path):
    """One run as a dict, or None if it carries no round log."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    sim = d.get("simulation") or {}
    metrics = sim.get("metrics") or []
    if not metrics:
        return None
    return {
        "path": path,
        "label": d.get("run_label", ""),
        "experiment": d.get("experiment_id", ""),
        "scenario": d.get("scenario_name", ""),
        "method_key": d.get("method_key", ""),
        "method": _method_name(d.get("method_key", "")),
        "seed": d.get("seed"),
        "rounds_completed": sim.get("rounds_completed"),
        "metrics": metrics,
        "participation_counts": sim.get("participation_counts") or [],
        "config": sim.get("config") or {},
    }


def load_all(subdirs=None):
    runs = [load_run(p) for p in find_results(subdirs)]
    return [r for r in runs if r]


def at_round(run, horizon):
    """The last evaluated metric entry at or before `horizon`.

    Returns None when the run never reached the horizon, so a truncated run is
    dropped rather than silently compared at a shorter horizon.
    """
    best = None
    for e in run["metrics"]:
        r = e.get("round")
        if r is None or r < 0 or r > horizon:
            continue
        if not e.get("evaluated"):
            continue
        if best is None or r > best.get("round", -1):
            best = e
    if best is None:
        return None
    if (run["rounds_completed"] or 0) < horizon:
        return None
    return best


def final(run):
    """The last evaluated entry of a run, whatever round that is."""
    ev = [e for e in run["metrics"] if e.get("evaluated") and (e.get("round") or -1) >= 0]
    return ev[-1] if ev else None


FIELDS = {
    "acc": ("accuracy", 100.0),
    "prec": ("precision", 100.0),
    "rec": ("recall", 100.0),
    "f1": ("f1", 100.0),
    "tflops": ("cum_training_tflops", 1.0),
    "energy": ("cum_modelled_energy_wh", 1.0),
    "carbon": ("cum_modelled_carbon_g", 1.0),
    "jain": ("fairness_jain", 1.0),
    "cov": ("participation_coverage_ratio", 100.0),
    "gini": ("fairness_gini", 1.0),
}


def read(entry, field):
    key, scale = FIELDS[field]
    v = entry.get(key)
    return None if v is None else v * scale


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5, len(xs)


if __name__ == "__main__":
    runs = load_all()
    print(f"{len(runs)} runs with round logs under {RUNS}")
    by_exp = {}
    for r in runs:
        by_exp.setdefault(r["experiment"], []).append(r)
    for exp in sorted(by_exp):
        rs = by_exp[exp]
        methods = sorted({x["method"] for x in rs})
        print(f"  {exp:34s} {len(rs):4d} runs  {len(methods):2d} methods")
