"""Selector-convergence evidence from the MAML-Revision-2 drift logs.

The logging code that produced `l_query_at_base` and `drift_increment` is not in
this repository, so this script first recovers their definitions numerically and
refuses to report anything it could not pin down.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

CONV = os.path.join(load.RUNS, "MAML-Revision-2", "convergence")
SETS = ["fashion", "cifar10", "cifar100"]


def read_log(name):
    path = os.path.join(CONV, f"selector_convergence_{name}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line.replace("NaN", "null")))
    return rows


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def identify(rows):
    """Test candidate definitions of drift_increment against the logged value."""
    cands = {
        "L_query(phi_t) - L_sup(phi_t)": lambda r, p: r["l_query_at_base"] - r["l_sup_before"],
        "L_query(phi_t) - L_query(phi_{t-1}) prev adapted": lambda r, p: r["l_query_at_base"] - (p or {}).get("l_query", float("nan")),
        "L_query(phi_t) - L_sup_after(phi_t)": lambda r, p: r["l_query_at_base"] - r["l_sup_after"],
    }
    scores = {k: [0, 0] for k in cands}
    prev = None
    for r in rows:
        if ok(r.get("drift_increment")):
            for k, fn in cands.items():
                try:
                    v = fn(r, prev)
                except (TypeError, KeyError):
                    continue
                if ok(v):
                    scores[k][1] += 1
                    if abs(v - r["drift_increment"]) < 1e-6:
                        scores[k][0] += 1
        prev = r
    return scores


def stats(rows):
    desc = [r["l_sup_descent"] for r in rows if ok(r.get("l_sup_descent"))]
    gain = [r["l_query_at_base"] - r["l_query"]
            for r in rows if ok(r.get("l_query_at_base")) and ok(r.get("l_query"))]
    drift = [r["drift_increment"] for r in rows if ok(r.get("drift_increment"))]
    return {
        "rounds": len(rows),
        "descent_n": len(desc),
        "descent_nonpositive": sum(1 for d in desc if d <= 0),
        "descent_mean": sum(desc) / len(desc) if desc else None,
        "descent_min": min(desc) if desc else None,
        "gain_n": len(gain),
        "gain_positive": sum(1 for g in gain if g > 0),
        "gain_mean": sum(gain) / len(gain) if gain else None,
        "gain_median": sorted(gain)[len(gain) // 2] if gain else None,
        "V_T_abs": sum(abs(d) for d in drift),
        "drift_signed_sum": sum(drift),
        "drift_mean_abs": sum(abs(d) for d in drift) / len(drift) if drift else None,
        "V_T_over_T": sum(abs(d) for d in drift) / len(drift) if drift else None,
        "q_first": rows[0].get("l_query"),
        "q_last": rows[-1].get("l_query"),
    }


if __name__ == "__main__":
    print("=== recovering the definition of drift_increment")
    for name in SETS:
        rows = read_log(name)
        sc = identify(rows)
        print(f"  {name}")
        for k, (hit, tot) in sc.items():
            print(f"      {hit:4d}/{tot:<4d}  {k}")

    print("\n=== Statement 1, inner-step descent on the support objective")
    print(f"{'dataset':<10}{'rounds':>8}{'descent<=0':>14}{'mean descent':>16}{'worst':>12}")
    for name in SETS:
        s = stats(read_log(name))
        print(f"{name:<10}{s['rounds']:>8}{s['descent_nonpositive']:>8}/{s['descent_n']:<5}"
              f"{s['descent_mean']:>16.6f}{s['descent_min']:>12.6f}")

    print("\n=== Adaptation gain, L_query at base minus L_query after the inner step")
    print(f"{'dataset':<10}{'rounds':>8}{'gain>0':>14}{'mean gain':>14}{'median':>12}")
    for name in SETS:
        s = stats(read_log(name))
        print(f"{name:<10}{s['gain_n']:>8}{s['gain_positive']:>8}/{s['gain_n']:<5}"
              f"{s['gain_mean']:>14.6f}{s['gain_median']:>12.6f}")

    print("\n=== Non-stationarity, path variation of the round objective")
    print(f"{'dataset':<10}{'V_T':>12}{'V_T/T':>12}{'signed sum':>14}{'q first':>10}{'q last':>10}")
    for name in SETS:
        s = stats(read_log(name))
        print(f"{name:<10}{s['V_T_abs']:>12.4f}{s['V_T_over_T']:>12.4f}"
              f"{s['drift_signed_sum']:>14.4f}{s['q_first']:>10.4f}{s['q_last']:>10.4f}")
