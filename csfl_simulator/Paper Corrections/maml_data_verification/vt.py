"""Measure V_T exactly as Corollary 1 of the round-2 manuscript defines it.

    V_T = sum_t ( q_{t+1}(phi_{t+1}) - q_t(phi_{t+1}) ),  a common iterate.

The selector logs `drift_increment = l_query_at_base - l_sup_before`. Because
D_sup(t) = D_query(t-1) we have g_t = q_{t-1}, so

    drift_increment(t) = q_t(phi_t) - q_{t-1}(phi_t),

which is the corollary's summand after reindexing. **V_T is therefore the signed
sum, not the absolute one.** The absolute sum is a different quantity, the total
variation, and must not be reported as V_T.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

CONV = os.path.join(load.RUNS, "MAML-Revision-2", "convergence")
SETS = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")]


def rows(name):
    path = os.path.join(CONV, f"selector_convergence_{name}.jsonl")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line.replace("NaN", "null")))
    return out


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


if __name__ == "__main__":
    print("Definition check, drift_increment against l_query_at_base - l_sup_before")
    for name, _ in SETS:
        rs = rows(name)
        hit = tot = 0
        for r in rs:
            if ok(r.get("drift_increment")) and ok(r.get("l_query_at_base")) and ok(r.get("l_sup_before")):
                tot += 1
                if abs((r["l_query_at_base"] - r["l_sup_before"]) - r["drift_increment"]) < 1e-9:
                    hit += 1
        print(f"  {name:<10} {hit}/{tot} rounds match the documented formula")

    print("\nV_T as Corollary 1 defines it, the signed sum")
    print(f"{'dataset':<15}{'rounds':>8}{'V_T':>12}{'V_T/T':>12}"
          f"{'total variation':>18}{'mean incr':>12}")
    for name, label in SETS:
        d = [r["drift_increment"] for r in rows(name) if ok(r.get("drift_increment"))]
        n = len(d)
        signed = sum(d)
        absolute = sum(abs(x) for x in d)
        print(f"{label:<15}{n:>8}{signed:>12.4f}{signed / n:>12.5f}"
              f"{absolute:>18.4f}{signed / n:>12.5f}")

    print("\nQuery objective range and largest single-round jump")
    print(f"{'dataset':<15}{'first':>10}{'last':>10}{'min':>10}{'max':>10}{'max |jump|':>12}")
    for name, label in SETS:
        q = [r["l_query"] for r in rows(name) if ok(r.get("l_query"))]
        jumps = [abs(q[i + 1] - q[i]) for i in range(len(q) - 1)]
        print(f"{label:<15}{q[0]:>10.3f}{q[-1]:>10.3f}{min(q):>10.3f}"
              f"{max(q):>10.3f}{max(jumps):>12.3f}")

    print("\nSign of V_T decides which branch of the corollary's remark applies")
    for name, label in SETS:
        d = [r["drift_increment"] for r in rows(name) if ok(r.get("drift_increment"))]
        s = sum(d)
        branch = "V_T <= 0, the benign case" if s <= 0 else "V_T > 0, tracking regime"
        print(f"  {label:<15} V_T = {s:+8.4f}   {branch}")
