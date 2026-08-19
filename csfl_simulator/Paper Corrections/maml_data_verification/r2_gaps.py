"""Read the two campaigns that closed the R2 evidence gaps.

Stage A is the no-adaptation control, twelve runs, inner_steps 0 against
inner_steps 1 on CIFAR-10 at 100 rounds and CIFAR-100 at 150 rounds. It answers
whether the inner adaptation step earns its place, since at zero steps the
support set is never touched and the outer step is a plain Adam step on the
query loss, which is online regression with the same network.

Stage B is the benchmark sweep. Table II reports one thing for MAML-Select on
Fashion-MNIST and every other run of that configuration reports another, so the
sweep settles which is right.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

GAPS = os.path.join(load.ROOT, "runs", "maml_r2")

HORIZON = {"cifar10_review_100": 100, "cifar100_review_150": 150,
           "fashion_main": 200, "cifar10_main": 200}


def paired_t(a, b):
    """Two-sided paired t-test over the shared seeds."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0:
        return float("inf") if m else 0.0, 0.0
    t = m / math.sqrt(var / n)
    # two-sided p from the t distribution, by the incomplete beta
    df = n - 1
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return t, p


def _betainc(a, b, x):
    """Regularized incomplete beta, enough for a t-test tail."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
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
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def collect(stage):
    """label -> run, for one stage directory."""
    out = {}
    root = os.path.join(GAPS, stage)
    if not os.path.isdir(root):
        return out
    for path in load.find_results([os.path.join("..", "runs", "maml_r2", stage)]) \
            if False else _walk(root):
        run = load.load_run(path)
        if run:
            out[run["label"] or os.path.basename(os.path.dirname(path))] = run
    return out


def _walk(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "result.json" in filenames:
            yield os.path.join(dirpath, "result.json")


def value(run, field, horizon):
    entry = load.at_round(run, horizon) or load.final(run)
    return load.read(entry, field) if entry else None


def stage_a():
    print("=" * 74)
    print("STAGE A  THE NO-ADAPTATION CONTROL")
    print("=" * 74)
    runs = collect("A")
    print("  runs returned: %d" % len(runs))

    for scenario, label in (("cifar10_review_100", "CIFAR-10, 100 rounds"),
                            ("cifar100_review_150", "CIFAR-100, 150 rounds")):
        horizon = HORIZON[scenario]
        arms = {}
        for steps in (0, 1):
            key = "inner_steps_%d" % steps
            rows = {}
            for name, run in runs.items():
                if scenario in name and key in name:
                    rows[str(run["seed"])] = run
            arms[steps] = rows

        seeds = sorted(set(arms[0]) & set(arms[1]))
        print()
        print("  %s, seeds %s" % (label, ", ".join(seeds)))
        print("    %-6s %10s %10s %10s %8s" %
              ("seed", "acc 0-step", "acc 1-step", "delta", "Jain 1"))
        a0, a1 = [], []
        for s in seeds:
            v0 = value(arms[0][s], "acc", horizon)
            v1 = value(arms[1][s], "acc", horizon)
            j1 = value(arms[1][s], "jain", horizon)
            if v0 is None or v1 is None:
                print("    %-6s  incomplete" % s)
                continue
            a0.append(v0)
            a1.append(v1)
            print("    %-6s %10.2f %10.2f %+10.2f %8.3f" % (s, v0, v1, v1 - v0, j1))
        if len(a0) >= 2:
            m0, s0, _ = load.mean_sd(a0)
            m1, s1, _ = load.mean_sd(a1)
            t, p = paired_t(a1, a0)
            print("    %-6s %6.2f+-%-3.2f %6.2f+-%-3.2f %+10.2f" %
                  ("mean", m0, s0, m1, s1, m1 - m0))
            print("    paired t = %.3f, two-sided p = %.4f over n=%d"
                  % (t, p, len(a0)))
            for field, name in (("tflops", "TFLOPs"), ("jain", "Jain")):
                v0s = [value(arms[0][s], field, horizon) for s in seeds]
                v1s = [value(arms[1][s], field, horizon) for s in seeds]
                v0s = [v for v in v0s if v is not None]
                v1s = [v for v in v1s if v is not None]
                if v0s and v1s:
                    q0, _, _ = load.mean_sd(v0s)
                    q1, _, _ = load.mean_sd(v1s)
                    print("    %-8s 0-step %10.3f   1-step %10.3f" % (name, q0, q1))


def stage_b():
    print()
    print("=" * 74)
    print("STAGE B  THE BENCHMARK SWEEP")
    print("=" * 74)
    runs = collect("B")
    print("  runs returned: %d" % len(runs))

    for scenario, title in (("fashion_main", "Fashion-MNIST"),
                            ("cifar10_main", "CIFAR-10")):
        horizon = HORIZON[scenario]
        print()
        print("  %s at round %d" % (title, horizon))
        print("    %-13s %14s %14s %10s %8s %7s %6s" %
              ("method", "accuracy", "F1", "TFLOPs", "energy", "Jain", "cov"))
        by_method = {}
        for name, run in runs.items():
            if scenario not in name:
                continue
            by_method.setdefault(run["method"], []).append(run)
        for method in load.METHOD_ORDER:
            rs = by_method.get(method)
            if not rs:
                continue
            cells = []
            for field in ("acc", "f1", "tflops", "energy", "jain", "cov"):
                vals = [value(r, field, horizon) for r in rs]
                vals = [v for v in vals if v is not None]
                cells.append(load.mean_sd(vals)[:2] if vals else (None, None))
            acc, f1, tf, en, ja, cv = cells
            print("    %-13s %7.2f+-%-5.2f %7.2f+-%-5.2f %10.0f %8.0f %7.2f %6.0f  n=%d"
                  % (method, acc[0], acc[1], f1[0], f1[1], tf[0], en[0],
                     ja[0], cv[0], len(rs)))


stage_a()
stage_b()
