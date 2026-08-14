"""Recompute the CIFAR-100 benchmark block of Table I.

Prints the table twice, once over the runs that existed before MAML-Revision-2
and once over every run, so the effect of the three new s2026 seeds is visible
rather than assumed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

HORIZON = 150
COLS = ["acc", "prec", "rec", "f1", "tflops", "energy", "carbon", "jain", "cov"]
WIDTH = {"acc": 2, "prec": 2, "rec": 2, "f1": 2, "tflops": 0,
         "energy": 0, "carbon": 0, "jain": 2, "cov": 0}


def block(runs, horizon=HORIZON):
    rows = {}
    for r in runs:
        e = load.at_round(r, horizon)
        if e is None:
            continue
        rows.setdefault(r["method"], []).append((r["seed"], e))
    return rows


def render(title, rows):
    print(f"\n=== {title}  (horizon = round {horizon_of(rows)})")
    head = f"{'Method':<13}{'n':>3} " + "".join(f"{c:>16}" for c in COLS)
    print(head)
    print("-" * len(head))
    for m in load.METHOD_ORDER:
        if m not in rows:
            continue
        seeds_entries = rows[m]
        cells = []
        for c in COLS:
            mu, sd, n = load.mean_sd([load.read(e, c) for _s, e in seeds_entries])
            if mu is None:
                cells.append(f"{'--':>16}")
                continue
            d = WIDTH[c]
            cells.append(f"{mu:>{10}.{d}f}+-{sd:<4.{d}f}"[:16].rjust(16))
        print(f"{m:<13}{len(seeds_entries):>3} " + "".join(cells))
    print()
    for m in load.METHOD_ORDER:
        if m in rows:
            print(f"    {m:<13} seeds {sorted(s for s, _ in rows[m])}")


def horizon_of(_rows):
    return HORIZON


if __name__ == "__main__":
    runs = [r for r in load.load_all() if r["experiment"] == "cifar100_benchmarks"]

    new = [r for r in runs if "MAML-Revision-2" in r["path"]]
    old = [r for r in runs if "MAML-Revision-2" not in r["path"]]

    print(f"cifar100_benchmarks: {len(runs)} runs, {len(old)} pre-existing, {len(new)} new")
    print("new runs:")
    for r in sorted(new, key=lambda x: x["label"]):
        print(f"    {r['label']:<52} rounds_completed={r['rounds_completed']}")

    print("\nrounds_completed by run:")
    for r in sorted(runs, key=lambda x: (x["method"], x["seed"] or 0)):
        keep = load.at_round(r, HORIZON) is not None
        print(f"    {r['method']:<13} s{r['seed']:<5} rounds={r['rounds_completed']:<5} "
              f"{'used' if keep else 'DROPPED, short of horizon'}")

    render("BEFORE MAML-Revision-2", block(old))
    render("AFTER MAML-Revision-2, all runs", block(runs))
