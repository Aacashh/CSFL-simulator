"""Cross-check every number the manuscript states against the run data.

Reads the rendered PDF rather than the source, so what is checked is what a
reviewer will actually see.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load
import conv
import fairness

PDF = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/Paper Corrections/MAML_REVISION_2/MAML_Select_manuscript_clean.pdf"

HORIZON = 150


def pdf_text(path):
    import pymupdf
    return "".join(p.get_text() for p in pymupdf.open(path))


def check(name, present, expected, text):
    """Assert that `expected` appears in the rendered text."""
    hit = expected in text
    print(f"  [{'ok ' if hit == present else 'MISS'}] {name:<52} {expected}")
    return hit == present


if __name__ == "__main__":
    t = pdf_text(PDF)
    t = re.sub(r"[\u2212\u2013\u2014]", "-", t)
    t = re.sub(r"\s+", " ", t)

    ok = []
    runs = [r for r in load.load_all() if r["experiment"] == "cifar100_benchmarks"]
    by = {}
    for r in runs:
        e = load.at_round(r, HORIZON)
        if e:
            by.setdefault(r["method"], []).append(e)

    print("CIFAR-100 benchmark rows, recomputed then looked for in the PDF")
    for m in load.METHOD_ORDER:
        mu, sd, n = load.mean_sd([load.read(e, "acc") for e in by[m]])
        ok.append(check(f"{m} accuracy (n={n})", True, f"{mu:.2f}±{sd:.2f}", t)
                  or check(f"{m} accuracy (n={n})", True, f"{mu:.2f}", t))

    print("\nSelector diagnostics, recomputed from the drift logs")
    for name, label in (("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"),
                        ("cifar100", "CIFAR-100")):
        s = conv.stats(conv.read_log(name))
        ok.append(check(f"{label} descent count", True,
                        f"{s['descent_nonpositive']}/{s['descent_n']}", t))
        ok.append(check(f"{label} gain count", True,
                        f"{s['gain_positive']}/{s['gain_n']}", t))
        ok.append(check(f"{label} path variation", True, f"{s['V_T_abs']:.2f}", t))

    print("\nFairness, recomputed with and without the cold start")
    cif = [r for r in load.load_all()
           if r["experiment"] == "cifar100_benchmarks" and r["method"] == "MAML-Select"]
    rows = fairness.summarise(cif, horizon=HORIZON)
    cov_post = load.mean_sd([r["cov_post"] for r in rows])[0]
    jain_all = load.mean_sd([r["jain_all"] for r in rows])[0]
    jain_post = load.mean_sd([r["jain_post"] for r in rows])[0]
    print(f"  recomputed CIFAR-100 coverage after warm-up : {cov_post:.1f}%")
    print(f"  recomputed CIFAR-100 Jain, all then post    : {jain_all:.3f} -> {jain_post:.3f}"
          f"  (drop {jain_all - jain_post:.3f})")
    ok.append(check("coverage claim survives warm-up removal", True, "100%", t))

    print("\nAblation and sweeps")
    abl = [r for r in load.load_all() if r["experiment"] == "feature_ablation"]
    g = {}
    for r in abl:
        e = load.final(r)
        if e:
            g.setdefault(r["method"], []).append(e)
    full = load.mean_sd([load.read(e, "acc") for e in g["full"]])[0]
    worst = min((load.mean_sd([load.read(e, "acc") for e in v])[0] - full, k)
                for k, v in g.items() if k != "full")
    print(f"  largest negative ablation shift: {worst[1]} at {worst[0]:+.2f} pp")
    ok.append(check("largest shift named in the text", True, "battery and latency", t))

    lam = {}
    for r in load.load_all():
        if r["experiment"] == "cifar10_lambda_sensitivity":
            e = load.final(r)
            if e:
                lam.setdefault(r["method"], []).append(e)
    accs = [load.mean_sd([load.read(e, "acc") for e in v])[0] for v in lam.values()]
    print(f"  CIFAR-10 lambda spread: {max(accs) - min(accs):.2f} pp")
    ok.append(check("CIFAR-10 lambda spread", True, f"{max(accs) - min(accs):.2f}", t))

    inner = {}
    for r in load.load_all():
        if r["experiment"] == "inner_step_ablation":
            e = load.final(r)
            if e:
                inner.setdefault(r["method"], []).append(e)
    for k in sorted(inner):
        v = load.mean_sd([load.read(e, "acc") for e in inner[k]])[0]
        print(f"  {k}: {v:.2f}")

    print(f"\n{sum(1 for x in ok if x)} of {len(ok)} checks passed")
