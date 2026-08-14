"""Check the SCOPE-FD report card against the campaign, before changing the paper.

Five questions.

1.  Section VI-K claims the guarantee "strengthens rather than weakens as the
    pool grows". Equation (17) is arithmetic, so this is checkable exactly.
2.  UnionFL ties SCOPE-FD in Table II on all three headline axes. The proposed
    differentiator is that the SCOPE-FD Gini is a closed form with no seed
    variance. Does UnionFL's fluctuate?
3.  Is per-client accuracy dispersion already logged, so that participation
    fairness can be connected to outcome fairness with no new runs?
4.  Table II reports rounds-to-60 with no dispersion. Did the seeds agree?
5.  Three headline accuracies are quoted for one configuration. What is in the
    data?
"""

import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ROOT = r"C:/Users/drash/OneDrive/Desktop/CSFL-simulator/runs/runs_scope_revised"

NAMES = {
    "heuristic.random": "Uniform random",
    "fd_native.divfl_fd": "DivFL",
    "fd_native.subtrunc_fd": "SubTrunc",
    "fd_native.unionfl_fd": "UnionFL",
    "fd_native.scope_fd_debt_only": "Debt-only",
    "fd_native.scope_fd": "SCOPE-FD",
    "system_aware.oort": "Oort",
}


def gini(counts):
    n = len(counts)
    tot = float(sum(counts))
    if tot == 0:
        return 0.0
    num = sum(abs(a - b) for a in counts for b in counts)
    return num / (2.0 * n * tot)


def gini_law(N, K, R):
    m = (K * R) % N
    return m * (N - m) / float(N * K * R)


def load():
    """Every finished arm in the campaign, flattened."""
    out = []
    for cr in glob.glob(os.path.join(ROOT, "*", "*", "compare_results.json")):
        family = os.path.basename(os.path.dirname(os.path.dirname(cr)))
        try:
            d = json.load(open(cr, encoding="utf-8"))
        except Exception:
            continue
        cfg = d.get("config", {})
        for key, arm in (d.get("results") or {}).items():
            if not isinstance(arm, dict) or "metrics" not in arm:
                continue
            rows = arm["metrics"]
            if not rows:
                continue
            acfg = arm.get("config", cfg) or cfg
            out.append({
                "family": family, "dir": os.path.dirname(cr),
                "method": NAMES.get(key, key), "key": key,
                "N": acfg.get("total_clients"), "K": acfg.get("clients_per_round"),
                "R": acfg.get("rounds"), "seed": acfg.get("seed"),
                "dataset": acfg.get("dataset"), "alpha": acfg.get("dirichlet_alpha"),
                "counts": arm.get("participation_counts") or [],
                "last": rows[-1], "rows": rows,
                "conv": arm.get("convergence") or {},
            })
    return out


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ question 1
def q1_pool_size():
    banner("1. Does the guarantee strengthen as the pool grows?")
    print("Equation (17) is G = m(N-m)/(NKR) with m = KR mod N. Pure arithmetic.")
    print()
    configs = [(30, 5), (47, 6), (53, 7), (50, 3), (100, 5), (200, 10), (200, 20), (200, 40)]
    horizons = [97, 98, 99, 100, 101]
    print(f"{'N':>5}{'K':>4} " + "".join(f"{'R=' + str(r):>10}" for r in horizons))
    for N, K in configs:
        cells = "".join(f"{100 * gini_law(N, K, r):>9.2f}%" for r in horizons)
        print(f"{N:>5}{K:>4} {cells}")
    print()
    print("At R=100 the large pools sit at zero because KR is an exact multiple")
    print("of N. Move one round either side and they are the worst rows in the")
    print("table. The paper's own Section V-A already says the decay is not")
    print("smooth, so Section VI-K contradicts it.")
    print()
    for N, K in [(30, 5), (200, 10)]:
        vals = [100 * gini_law(N, K, r) for r in range(25, 121)]
        print(f"   N={N:>3} K={K:<3} over R in [25,120]: "
              f"min {min(vals):.2f}%  max {max(vals):.2f}%  mean {st.mean(vals):.2f}%")


# ------------------------------------------------------------------ question 2
def q2_unionfl(rows):
    banner("2. Does UnionFL's Gini fluctuate where SCOPE-FD's does not?")
    per = defaultdict(list)
    for r in rows:
        if not r["counts"] or not r["N"]:
            continue
        per[(r["method"], r["N"], r["K"], r["R"])].append(
            (r["seed"], 100 * gini(r["counts"]), r["family"]))

    watch = ["SCOPE-FD", "Debt-only", "UnionFL", "Uniform random", "DivFL", "SubTrunc", "Oort"]
    print(f"{'method':<16}{'N':>5}{'K':>4}{'R':>5}{'seeds':>7}"
          f"{'mean Gini':>11}{'sd':>8}{'law (17)':>10}")
    for m in watch:
        keys = sorted([k for k in per if k[0] == m], key=lambda k: (k[1], k[2], k[3]))
        for k in keys:
            v = per[k]
            g = [x[1] for x in v]
            law = 100 * gini_law(k[1], k[2], k[3]) if all(k[1:]) else float("nan")
            sd = st.stdev(g) if len(g) > 1 else 0.0
            print(f"{m:<16}{k[1]:>5}{k[2]:>4}{k[3]:>5}{len(v):>7}"
                  f"{st.mean(g):>10.2f}%{sd:>8.3f}{law:>9.2f}%")

    print()
    print("Spread of each method's Gini across ALL its configurations:")
    for m in watch:
        allg = [x[1] for k in per if k[0] == m for x in per[k]]
        if not allg:
            continue
        print(f"   {m:<16} n={len(allg):>4}  min {min(allg):>6.2f}%  "
              f"max {max(allg):>6.2f}%  sd {st.stdev(allg) if len(allg) > 1 else 0:>6.2f}")


# ------------------------------------------------------------------ question 3
def q3_client_dispersion(rows):
    banner("3. Per-client accuracy dispersion, already logged?")
    keys = ("client_accuracy_avg", "client_accuracy_std", "avg_per_client_accuracy",
            "accuracy_std", "server_client_gap")
    sample = rows[0]["last"] if rows else {}
    print("fields present in the per-round record:")
    for k in keys:
        print(f"   {k:<26} {'yes' if k in sample else 'NO'}")
    print()
    head = [r for r in rows
            if r["family"] == "literature_baselines" and r["N"] == 30 and r["K"] == 5]
    per = defaultdict(list)
    for r in head:
        last = r["last"]
        per[r["method"]].append((
            100 * float(last.get("accuracy", float("nan"))),
            100 * float(last.get("client_accuracy_avg", float("nan"))),
            100 * float(last.get("client_accuracy_std", float("nan"))),
            100 * float(last.get("server_client_gap", float("nan"))),
        ))
    print("headline configuration, N=30 K=5 R=100, mean over seeds")
    print(f"{'method':<16}{'server acc':>12}{'client acc':>12}"
          f"{'client sd':>12}{'gap':>9}{'seeds':>7}")
    for m in ("Uniform random", "DivFL", "SubTrunc", "UnionFL", "Oort",
              "Debt-only", "SCOPE-FD"):
        v = per.get(m)
        if not v:
            continue
        print(f"{m:<16}{st.mean(x[0] for x in v):>11.2f}%{st.mean(x[1] for x in v):>11.2f}%"
              f"{st.mean(x[2] for x in v):>11.2f}%{st.mean(x[3] for x in v):>8.2f}{len(v):>7}")
    print()
    print("A lower client sd at equal server accuracy is outcome fairness, not")
    print("just procedural fairness. That is the report card's item M5 and it")
    print("needs no new run.")

    # sparse regime, where the effect should be strongest
    sparse = [r for r in rows if r["K"] in (1, 3) and r["N"] == 30]
    if sparse:
        ps = defaultdict(list)
        for r in sparse:
            ps[(r["method"], r["K"])].append((
                100 * float(r["last"].get("accuracy", float("nan"))),
                100 * float(r["last"].get("client_accuracy_std", float("nan")))))
        print()
        print("sparse regime")
        print(f"{'method':<16}{'K':>3}{'server acc':>12}{'client sd':>12}{'seeds':>7}")
        for k in sorted(ps, key=lambda x: (x[1], x[0])):
            v = ps[k]
            print(f"{k[0]:<16}{k[1]:>3}{st.mean(x[0] for x in v):>11.2f}%"
                  f"{st.mean(x[1] for x in v):>11.2f}%{len(v):>7}")


# ------------------------------------------------------------------ question 4
def q4_rounds_dispersion(rows):
    banner("4. Rounds to 60 percent, did the seeds agree?")
    head = [r for r in rows
            if r["family"] == "literature_baselines" and r["N"] == 30 and r["K"] == 5]
    per = defaultdict(list)
    for r in head:
        v = r["conv"].get("rounds_to_abs_60")
        per[r["method"]].append(v)
    print(f"{'method':<16}{'rounds to 60%':>34}{'mean':>8}{'sd':>8}")
    for m in ("Uniform random", "DivFL", "SubTrunc", "UnionFL", "Oort",
              "Debt-only", "SCOPE-FD"):
        v = per.get(m)
        if not v:
            continue
        got = [x for x in v if x is not None]
        sd = st.stdev(got) if len(got) > 1 else 0.0
        shown = str(sorted(x for x in v if x is not None)) + ('' if all(x is not None for x in v) else ' +miss')
        print(f"{m:<16}{shown:>34}{(st.mean(got) if got else float('nan')):>8.1f}{sd:>8.2f}")


# ------------------------------------------------------------------ question 5
def q5_headline(rows):
    banner("5. The headline accuracy, per family")
    per = defaultdict(list)
    for r in rows:
        if r["N"] == 30 and r["K"] == 5 and r["R"] == 100 and r["dataset"] == "Fashion-MNIST" \
                and r["alpha"] == 0.5 and r["method"] in ("SCOPE-FD", "Uniform random", "Debt-only"):
            per[(r["method"], r["family"])].append(
                (r["seed"], 100 * float(r["last"].get("accuracy", float("nan")))))
    for m in ("SCOPE-FD", "Uniform random", "Debt-only"):
        print(f"-- {m}")
        pooled = []
        for k in sorted([k for k in per if k[0] == m], key=lambda x: x[1]):
            v = per[k]
            accs = [x[1] for x in v]
            pooled.extend(accs)
            sd = st.stdev(accs) if len(accs) > 1 else 0.0
            print(f"   {k[1]:<34} n={len(accs):>2}  {st.mean(accs):6.2f} +- {sd:4.2f}"
                  f"   seeds {sorted(x[0] for x in v)}")
        if pooled:
            sd = st.stdev(pooled) if len(pooled) > 1 else 0.0
            print(f"   {'POOLED across families':<34} n={len(pooled):>2}  "
                  f"{st.mean(pooled):6.2f} +- {sd:4.2f}")
        print()


if __name__ == "__main__":
    rows = load()
    print(f"loaded {len(rows)} method-arms from {ROOT}")
    fams = sorted({r['family'] for r in rows})
    print(f"{len(fams)} families: {', '.join(fams)}")
    q1_pool_size()
    q2_unionfl(rows)
    q3_client_dispersion(rows)
    q4_rounds_dispersion(rows)
    q5_headline(rows)
