"""Paired MAML-Select against FedAvg on the seeds both methods share.

Checks the effect sizes and p values printed in supplementary Table S3. Only
CIFAR-100 can be checked here, because the Fashion-MNIST and CIFAR-10 benchmark
runs are not on this machine.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

HORIZON = 150


def student_sf(t, df):
    """Two-sided p value for Student's t, via the incomplete beta function."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def paired(a, b):
    """a minus b, per seed."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    dz = abs(m / sd) if sd else float("inf")
    t = m / (sd / math.sqrt(n)) if sd else float("inf")
    return m, dz, student_sf(abs(t), n - 1), n


if __name__ == "__main__":
    runs = [r for r in load.load_all() if r["experiment"] == "cifar100_benchmarks"]
    by = {}
    for r in runs:
        e = load.at_round(r, HORIZON)
        if e:
            by.setdefault(r["method"], {})[r["seed"]] = e

    fa, ms = by["FedAvg"], by["MAML-Select"]
    shared = sorted(set(fa) & set(ms))
    print(f"CIFAR-100, shared seeds {shared}\n")
    print(f"{'metric':<14}{'mean reduction':>16}{'|d_z|':>9}{'p':>9}   paper")
    paper = {"tflops": ("107.80", "5.15", "0.012"),
             "energy": ("169.65", "10.89", "0.003"),
             "carbon": ("80.58", "10.89", "0.003")}
    for metric in ("tflops", "energy", "carbon"):
        a = [load.read(fa[s], metric) for s in shared]
        b = [load.read(ms[s], metric) for s in shared]
        m, dz, p, n = paired(a, b)
        pm, pd, pp = paper[metric]
        print(f"{metric:<14}{m:>16.2f}{dz:>9.2f}{p:>9.3f}   "
              f"{pm} / {pd} / {pp}")

    print("\naccuracy, MAML-Select minus FedAvg")
    a = [load.read(ms[s], "acc") for s in shared]
    b = [load.read(fa[s], "acc") for s in shared]
    m, dz, p, n = paired(a, b)
    print(f"  mean {m:+.2f} pp   |d_z| {dz:.2f}   p {p:.3f}")
