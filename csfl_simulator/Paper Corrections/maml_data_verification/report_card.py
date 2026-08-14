"""Re-derive every quantity the R2 report card asks for, from the run logs.

Five questions, in the order the report card raises them.

1. Does the simulator implement Eq. (5)?  Wh per TFLOP is a constant within a
   dataset if it does not depend on the model, which is the test.
2. Is the compute saving a small-shard artifact?  Cumulative TFLOPs is exactly
   proportional to the number of selected training samples, so the mean
   selected shard size follows from the TFLOPs column with no extra assumption.
3. What is the achieved round latency?
4. What is the reduced protocol behind the "diagnostic" accuracies?
5. How many distinct MAML-Select runs enter the fairness recheck?
"""

import json
import math
import os
import statistics as st
from collections import defaultdict

ROOT = r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator"
RUNS = os.path.join(ROOT, "runs")
NUMBERS = os.path.join(
    ROOT, "csfl_simulator", "Paper Corrections", "MAML_REVISION_2", "build",
    "revision_numbers.json")

METHOD = {
    "baseline.fedavg": "FedAvg", "system_aware.fedcs": "FedCS",
    "system_aware.oort": "Oort", "system_aware.tifl": "TiFL",
    "ml.fedcor": "FedCor", "research.criticalfl": "CriticalFL",
    "research.fedgcs": "FedGCS", "research.maml_select": "MAML-Select",
}
ORDER = ["FedAvg", "FedCS", "Oort", "TiFL", "FedCor", "CriticalFL", "FedGCS",
         "MAML-Select"]

# pool composition and the tier constants the simulator uses
TIER_FRACTION = (0.20, 0.50, 0.30)
TIER_POWER = (4.0, 7.0, 12.0)
TIER_SPEED = (1.0, 2.0, 4.0)

TRAIN_SAMPLES = {"Fashion-MNIST": 60000, "CIFAR-10": 50000, "CIFAR-100": 50000}


def rows(run_dir):
    out = []
    p = os.path.join(run_dir, "round_metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load(family):
    base = os.path.join(RUNS, family)
    runs = []
    if not os.path.isdir(base):
        return runs
    for d in sorted(os.listdir(base)):
        rp = os.path.join(base, d, "result.json")
        if not os.path.exists(rp):
            continue
        res = json.load(open(rp, encoding="utf-8"))
        sim = res.get("simulation", {})
        runs.append({
            "dir": os.path.join(base, d), "name": d,
            "cfg": sim.get("config", {}), "seed": res.get("seed"),
            "method_key": res.get("method_key"),
            "method": METHOD.get(res.get("method_key")),
            "rows": rows(os.path.join(base, d)),
        })
    return runs


def at(rs, horizon):
    got = [r for r in rs if int(r["round"]) <= horizon]
    return got[-1] if got else None


def last_eval(rs, horizon):
    got = [r for r in rs if r.get("evaluated") and int(r["round"]) <= horizon]
    return got[-1] if got else None


def line(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- question 1
def q1_energy_model():
    line("1. Wh per TFLOP, the test of whether latency follows FLOPs")
    print("Eq. (5) as written puts C_model in the latency.  If it did, Wh per")
    print("TFLOP would be a device-mix constant and identical across datasets.")
    print()
    table = {  # the published Table I FedAvg row
        "Fashion-MNIST": (140.0, 5752.0, 200),
        "CIFAR-10": (8341.0, 4798.0, 200),
        "CIFAR-100": (6331.0, 3626.0, 150),
    }
    print(f"{'dataset':<16}{'TFLOPs':>9}{'Wh':>9}{'Wh/TFLOP':>11}{'Wh/round':>11}")
    for ds, (tf, wh, rounds) in table.items():
        print(f"{ds:<16}{tf:>9.0f}{wh:>9.0f}{wh / tf:>11.3f}{wh / rounds:>11.2f}")

    # the closed form the code implements: E * n_i * P_i / f_i / 3600
    mean_p_over_f = sum(f * p / s for f, p, s
                        in zip(TIER_FRACTION, TIER_POWER, TIER_SPEED))
    print()
    print(f"pool mean of P_i/f_i = {mean_p_over_f:.4f} W per unit rate")
    print("predicted Wh per round for random selection, E=5, K=10:")
    for ds in table:
        n_bar = TRAIN_SAMPLES[ds] / 100.0
        pred = 10 * mean_p_over_f * 5 * n_bar / 3600.0
        print(f"   {ds:<16} n_bar={n_bar:>6.0f}  predicted={pred:>7.2f} Wh"
              f"   observed={table[ds][1] / table[ds][2]:>7.2f} Wh")
    print()
    print("The closed form with NO C_model reproduces the observed energy, so")
    print("the simulator's latency is E*n_i/f_i and Eq. (5) must drop C_model.")


# ---------------------------------------------------------------- question 2
def q2_shard_size():
    line("2. Mean selected shard size, from the TFLOPs identity")
    print("TFLOPs_total = 3 * C_model * E * sum over selected of n_i, so within a")
    print("dataset the cumulative TFLOPs ratio IS the selected-sample ratio.")
    print()
    published = {  # dataset -> method -> (TFLOPs, Wh, acc)
        "Fashion-MNIST": {"FedAvg": (140, 5752, 90.22), "FedCS": (76, 2772, 84.11),
                          "Oort": (149, 5963, 86.70), "TiFL": (147, 6062, 86.73),
                          "FedCor": (121, 4845, 89.35), "CriticalFL": (304, 12552, 90.16),
                          "FedGCS": (136, 5595, 90.65), "MAML-Select": (122, 4809, 90.11)},
        "CIFAR-10": {"FedAvg": (8341, 4798, 79.43), "FedCS": (4081, 2073, 50.45),
                     "Oort": (8303, 4677, 60.19), "TiFL": (8200, 4800, 61.93),
                     "FedCor": (7748, 4452, 67.20), "CriticalFL": (14239, 8205, 77.34),
                     "FedGCS": (7656, 4428, 77.88), "MAML-Select": (6549, 3610, 75.63)},
        "CIFAR-100": {"FedAvg": (6331, 3626, 59.66), "FedCS": (5334, 2659, 27.52),
                      "Oort": (5955, 3421, 29.41), "TiFL": (5961, 3480, 28.52),
                      "FedCor": (6473, 3663, 30.86), "CriticalFL": (10235, 5879, 45.37),
                      "FedGCS": (6140, 3526, 58.66), "MAML-Select": (6224, 3457, 58.15)},
    }
    for ds, meths in published.items():
        pool = TRAIN_SAMPLES[ds] / 100.0
        base_tf, base_wh, base_acc = meths["FedAvg"]
        print(f"-- {ds}   pool mean shard = {pool:.0f} samples")
        print(f"   {'method':<13}{'mean shard':>11}{'vs pool':>9}"
              f"{'Wh/sample':>11}{'tier term':>10}{'acc':>8}")
        for m in ORDER:
            tf, wh, acc = meths[m]
            shard = pool * tf / base_tf
            # energy per selected sample, normalized to FedAvg -> the tier term
            tier = (wh / tf) / (base_wh / base_tf)
            print(f"   {m:<13}{shard:>11.1f}{100 * (tf / base_tf - 1):>8.1f}%"
                  f"{wh / tf:>11.4f}{tier:>9.3f}x{acc:>8.2f}")
        m = meths["MAML-Select"]
        print(f"   decomposition of the energy saving against FedAvg:")
        s_term = m[0] / base_tf
        t_term = (m[1] / m[0]) / (base_wh / base_tf)
        print(f"      samples {100 * (1 - s_term):.1f}%  x  tier-power "
              f"{100 * (1 - t_term):.1f}%  ->  {100 * (1 - s_term * t_term):.1f}%"
              f"   (published {100 * (1 - m[1] / base_wh):.1f}%)")
        print()

    # cross-check the tier term against the measured tier shares on CIFAR-100
    print("cross-check on CIFAR-100 using the measured tier shares:")
    pool_pf = sum(f * p / s for f, p, s in zip(TIER_FRACTION, TIER_POWER, TIER_SPEED))
    for name, shares in (("pool", TIER_FRACTION), ("FedAvg", (0.196, 0.495, 0.309)),
                         ("FedGCS", (0.217, 0.472, 0.311)),
                         ("MAML-Select", (0.116, 0.446, 0.437))):
        pf = sum(f * p / s for f, p, s in zip(shares, TIER_POWER, TIER_SPEED))
        print(f"   {name:<13} mean P/f = {pf:.4f}   ratio to pool = {pf / pool_pf:.4f}")


# ---------------------------------------------------------------- question 3
def q3_latency():
    line("3. Achieved round latency and time to target, CIFAR-100")
    runs = load("maml_select_cifar100") + [
        r for r in load("MAML-Revision-2/cifar100")]
    # the revision runs live one level down
    extra = os.path.join(RUNS, "MAML-Revision-2", "cifar100")
    if os.path.isdir(extra):
        for d in sorted(os.listdir(extra)):
            rp = os.path.join(extra, d, "result.json")
            if not os.path.exists(rp):
                continue
            res = json.load(open(rp, encoding="utf-8"))
            runs.append({
                "dir": os.path.join(extra, d), "name": d,
                "cfg": res.get("simulation", {}).get("config", {}),
                "seed": res.get("seed"), "method_key": res.get("method_key"),
                "method": METHOD.get(res.get("method_key")),
                "rows": rows(os.path.join(extra, d))})

    H = 150
    per = defaultdict(list)
    for r in runs:
        if not r["rows"] or not r["method"]:
            continue
        rs = [x for x in r["rows"] if int(x["round"]) <= H]
        if not rs:
            continue
        mean_round = st.mean(float(x["round_time"]) for x in rs)
        cum = float(rs[-1]["cum_time"])
        ev = last_eval(r["rows"], H)
        acc = float(ev["accuracy"]) * 100 if ev else float("nan")
        # time to 50 percent accuracy
        t50 = None
        for x in r["rows"]:
            if x.get("evaluated") and float(x["accuracy"]) >= 0.50:
                t50 = float(x["cum_time"])
                break
        per[r["method"]].append((mean_round, cum, acc, t50, r["seed"], r["name"]))

    print(f"{'method':<13}{'mean T_round':>14}{'cum time (s)':>14}"
          f"{'acc@150':>9}{'t to 50%':>11}{'seeds':>7}")
    for m in ORDER:
        v = per.get(m)
        if not v:
            continue
        mr = st.mean(x[0] for x in v)
        cu = st.mean(x[1] for x in v)
        ac = st.mean(x[2] for x in v)
        t5 = [x[3] for x in v if x[3] is not None]
        t5s = f"{st.mean(t5):.0f}" if len(t5) == len(v) else "not reached"
        print(f"{m:<13}{mr:>14.1f}{cu:>14.0f}{ac:>9.2f}{t5s:>11}{len(v):>7}")
    print()
    print("mean T_round relative to FedAvg:")
    base = st.mean(x[0] for x in per["FedAvg"])
    for m in ORDER:
        if per.get(m):
            print(f"   {m:<13}{st.mean(x[0] for x in per[m]) / base:>7.3f}x")


# ---------------------------------------------------------------- question 4
def q4_diagnostic():
    line("4. The reduced protocol behind the CIFAR-10 ablation numbers")
    for fam in ("maml_select_review_hardening", "maml_select"):
        for r in load(fam):
            c = r["cfg"]
            if not c:
                continue
            print(f"{r['name'][:66]:<68} {c.get('dataset'):<14}"
                  f"R={c.get('rounds')} N={c.get('total_clients')} "
                  f"K={c.get('clients_per_round')} a={c.get('dirichlet_alpha')}")
        print()

    print("grouped accuracy at the run's own horizon:")
    groups = defaultdict(list)
    for fam in ("maml_select_review_hardening", "maml_select"):
        for r in load(fam):
            if not r["rows"]:
                continue
            ev = last_eval(r["rows"], 10 ** 9)
            if ev is None:
                continue
            key = (r["cfg"].get("dataset"), r["cfg"].get("rounds"),
                   r["method_key"])
            groups[key].append((float(ev["accuracy"]) * 100,
                                float(r["rows"][-1]["cum_training_tflops"]),
                                float(r["rows"][-1]["fairness_jain"])))
    for k in sorted(groups, key=lambda x: (str(x[0]), x[1], str(x[2]))):
        v = groups[k]
        acc = [x[0] for x in v]
        print(f"   {str(k[0]):<14} R={k[1]:<5} {k[2]:<46} "
              f"acc={st.mean(acc):6.2f} +- {(st.stdev(acc) if len(acc) > 1 else 0):4.2f}"
              f"  tflops={st.mean(x[1] for x in v):8.1f}"
              f"  jain={st.mean(x[2] for x in v):5.3f}  n={len(v)}")


# ---------------------------------------------------------------- question 5
def q5_run_count():
    line("5. Distinct MAML-Select runs in the fairness recheck")
    d = json.load(open(NUMBERS, encoding="utf-8"))
    pool = d["fairness_without_cold_start"]
    grand_listed = grand_distinct = 0
    for ds in ("Fashion-MNIST", "CIFAR-10", "CIFAR-100"):
        runs = pool[ds]["runs"]
        seen, uniq = set(), []
        for r in runs:
            if r["run"] in seen:
                continue
            seen.add(r["run"])
            uniq.append(r)
        ja = st.mean(r["jain_all"] for r in uniq)
        jp = st.mean(r["jain_post"] for r in uniq)
        cov = min(r["coverage_post_pct"] for r in uniq)
        print(f"{ds:<15} listed={len(runs):>3}  distinct={len(uniq):>3}   "
              f"jain_all={ja:.4f}  jain_post={jp:.4f}  "
              f"min coverage after cold start={cov:.0f}%")
        print(f"{'':<15} published pair was "
              f"{pool[ds]['jain_all']:.3f} -> {pool[ds]['jain_post']:.3f}")
        fams = defaultdict(int)
        for r in uniq:
            fams[r["run"].split("_research.maml_select.")[0].rsplit("_s", 1)[0]] += 1
        for f, n in sorted(fams.items()):
            print(f"{'':<19}{n:>3}  {f}")
        grand_listed += len(runs)
        grand_distinct += len(uniq)
    print()
    print(f"TOTAL listed {grand_listed}, distinct {grand_distinct}")


# ---------------------------------------------------------------- question 6
def q6_equivalence():
    line("6. Confidence intervals for the accuracy differences")
    print("A large p-value at n=3 is uninformative, so report the interval.")
    pairs = [
        ("Fashion-MNIST", "MAML-Select vs FedAvg", [90.11, 90.22], [0.47, 0.58]),
        ("CIFAR-10", "MAML-Select vs FedGCS", [75.63, 77.88], [1.43, 1.75]),
        ("CIFAR-100", "MAML-Select vs FedGCS", [58.15, 58.66], [0.51, 0.11]),
        ("CIFAR-100", "MAML-Select vs FedAvg", [58.15, 59.66], [0.51, 0.29]),
    ]
    n = 3
    tcrit = 4.303  # two sided 95 percent, 2 degrees of freedom
    for ds, label, means, sds in pairs:
        diff = means[0] - means[1]
        se = math.sqrt(sds[0] ** 2 / n + sds[1] ** 2 / n)
        half = tcrit * se
        print(f"{ds:<15}{label:<26} diff={diff:+6.2f} pp   "
              f"95% CI [{diff - half:+6.2f}, {diff + half:+6.2f}]")
    print()
    print("Unpaired intervals from the published summary statistics, so they are")
    print("wider than the paired test.  They show the honest wording: a")
    print("difference was not detected, and the interval says how wide the")
    print("undetected difference could be.")


if __name__ == "__main__":
    q1_energy_model()
    q2_shard_size()
    q3_latency()
    q4_diagnostic()
    q5_run_count()
    q6_equivalence()
