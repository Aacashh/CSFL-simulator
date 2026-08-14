"""Exactly what the sparse-regime accuracy data will and will not support.

The abstract claims the selector "leads it where participation is sparsest".
The sweeps run three seeds. This works out what a paired test on three seeds can
attain at best, what the realized differences are, and therefore which wording
is defensible.
"""

import glob
import importlib.util
import itertools
import json
import math
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rc", os.path.join(HERE, "reportcard.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def wilcoxon_min_two_sided(n):
    """Smallest attainable two-sided p for a signed-rank test on n pairs.

    With n pairs there are 2^n sign assignments, all equally likely under the
    null. The most extreme outcome, every difference the same sign, therefore
    carries one-sided probability 2^-n and two-sided 2^(1-n).
    """
    return 2.0 ** (1 - n)


def sign_test_two_sided(diffs):
    """Exact two-sided sign test. Makes no normality assumption, which matters
    at this sample size."""
    n = len([d for d in diffs if d != 0])
    k = len([d for d in diffs if d > 0])
    if n == 0:
        return 1.0
    tail = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, tail + 1)) / 2.0 ** n
    return min(1.0, 2 * p)


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = st.mean(diffs)
    s = st.stdev(diffs)
    if s == 0:
        return float("inf"), 0.0
    return m, m / (s / math.sqrt(n))


print("=" * 78)
print("What a paired test on three seeds can attain")
print("=" * 78)
for n in (3, 5, 10, 15):
    print("  n = %-3d  smallest attainable two-sided p = %.4f%s"
          % (n, wilcoxon_min_two_sided(n),
             "   cannot reach 0.05" if wilcoxon_min_two_sided(n) > 0.05 else ""))
print()
print("  The sweeps use three seeds, so no paired test on them can produce")
print("  p < 0.25. A significance claim in the sparse regime is unreachable")
print("  with the data in hand, whatever the effect size.")

rows = rc.load()

print()
print("=" * 78)
print("Realized paired differences, matched seeds, N=30")
print("=" * 78)

for family in ("literature_baselines_k_sweep", "ablation_k_sweep"):
    sel = [r for r in rows if r["family"] == family and r["N"] == 30]
    if not sel:
        continue
    per = defaultdict(dict)
    for r in sel:
        per[(r["K"], r["method"])][r["seed"]] = (
            100 * float(r["last"]["accuracy"]),
            100 * float(r["last"].get("client_accuracy_std", float("nan"))),
        )
    print()
    print("-- %s" % family)
    for K in sorted({k[0] for k in per}):
        base = per.get((K, "Uniform random"))
        prop = per.get((K, "SCOPE-FD"))
        if not base or not prop:
            continue
        seeds = sorted(set(base) & set(prop))
        acc_d = [prop[s][0] - base[s][0] for s in seeds]
        sd_d = [base[s][1] - prop[s][1] for s in seeds]
        m, t = paired_t(acc_d)
        print("   K=%-3d seeds=%s" % (K, seeds))
        print("        accuracy   random %s" % ["%.2f" % base[s][0] for s in seeds])
        print("                 SCOPE-FD %s" % ["%.2f" % prop[s][0] for s in seeds])
        print("        paired differences %s   mean %+.2f pp"
              % (["%+.2f" % d for d in acc_d], m))
        print("        all same sign: %-5s   exact sign test p = %.3f"
              % (all(d > 0 for d in acc_d) or all(d < 0 for d in acc_d),
                 sign_test_two_sided(acc_d)))
        print("        client-sd reduction %s   mean %+.2f"
              % (["%+.2f" % d for d in sd_d], st.mean(sd_d)))


print()
print("=" * 78)
print("Effect size against seed spread, K=1")
print("=" * 78)
sel = [r for r in rows if r["family"] == "literature_baselines_k_sweep"
       and r["N"] == 30 and r["K"] == 1]
per = defaultdict(dict)
for r in sel:
    per[r["method"]][r["seed"]] = (
        100 * float(r["last"]["accuracy"]),
        100 * float(r["last"].get("client_accuracy_std", float("nan"))))
seeds = sorted(per["Uniform random"])
acc_d = [per["SCOPE-FD"][s][0] - per["Uniform random"][s][0] for s in seeds]
sd_d = [per["Uniform random"][s][1] - per["SCOPE-FD"][s][1] for s in seeds]
for label, d, sds in (("accuracy gain", acc_d,
                       [st.stdev([per[m][s][0] for s in seeds])
                        for m in ("Uniform random", "SCOPE-FD")]),
                      ("client-sd reduction", sd_d,
                       [st.stdev([per[m][s][1] for s in seeds])
                        for m in ("Uniform random", "SCOPE-FD")])):
    print("  %-22s mean %+.2f   seed sds %s   ratio to larger sd %.2f"
          % (label, st.mean(d), ["%.2f" % x for x in sds],
             abs(st.mean(d)) / max(sds)))

print()
print("=" * 78)
print("Verdict")
print("=" * 78)
print("  The accuracy gain at K=1 is %+.2f pp against seed standard deviations"
      % st.mean(acc_d))
print("  of 2.28 and 1.68, so the effect is smaller than the noise it sits in.")
print("  Three seeds cannot produce p < 0.25 under any paired test. The abstract")
print("  must therefore say accuracy is preserved, not that it leads.")
