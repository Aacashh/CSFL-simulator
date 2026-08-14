"""Recompute Table II, the state-feature ablation, and the lambda sweep."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

ABL_ORDER = ["full", "no_loss", "no_grad_norm", "no_latency",
             "no_battery", "no_frequency", "no_staleness"]
ABL_LABEL = {"full": "Full state vector", "no_loss": "w/o loss",
             "no_grad_norm": "w/o gradient norm", "no_latency": "w/o latency",
             "no_battery": "w/o battery", "no_frequency": "w/o frequency",
             "no_staleness": "w/o staleness"}


def group(runs, key="method"):
    g = {}
    for r in runs:
        g.setdefault(r[key], []).append(r)
    return g


def table(runs, order, labels, horizon=None):
    g = group(runs)
    rows = []
    for name in order:
        rs = g.get(name, [])
        ent = [(load.at_round(r, horizon) if horizon else load.final(r)) for r in rs]
        ent = [e for e in ent if e]
        if not ent:
            continue
        acc = load.mean_sd([load.read(e, "acc") for e in ent])
        tfl = load.mean_sd([load.read(e, "tflops") for e in ent])
        jai = load.mean_sd([load.read(e, "jain") for e in ent])
        cov = load.mean_sd([load.read(e, "cov") for e in ent])
        rows.append((labels.get(name, name), len(ent), acc, tfl, jai, cov,
                     sorted(r["seed"] for r in rs)))
    return rows


def show(title, rows, ref_acc=None):
    print(f"\n=== {title}")
    print(f"{'Variant':<20}{'n':>3}{'Acc (%)':>18}{'TFLOPs':>12}"
          f"{'Jain':>14}{'Cov':>10}{'dAcc':>9}   seeds")
    for label, n, acc, tfl, jai, cov, seeds in rows:
        d = "" if ref_acc is None else f"{acc[0] - ref_acc:>+9.2f}"
        print(f"{label:<20}{n:>3}{acc[0]:>12.2f}+-{acc[1]:<4.2f}"
              f"{tfl[0]:>9.0f}+-{tfl[1]:<3.0f}{jai[0]:>10.3f}+-{jai[1]:<4.3f}"
              f"{cov[0]:>7.0f}+-{cov[1]:<3.0f}{d}   {seeds}")


if __name__ == "__main__":
    runs = load.load_all()

    abl = [r for r in runs if r["experiment"] == "feature_ablation"]
    rows = table(abl, ABL_ORDER, ABL_LABEL)
    ref = [r for r in rows if r[0] == "Full state vector"][0][2][0]
    show("Table II, state-feature ablation on Fashion-MNIST", rows, ref)
    span = max(r[2][0] for r in rows) - min(r[2][0] for r in rows)
    print(f"\n    accuracy span across variants: {span:.2f} pp")
    print(f"    largest drop vs full: "
          f"{min(r[2][0] - ref for r in rows):+.2f} pp "
          f"({min(rows, key=lambda r: r[2][0])[0]})")

    for exp, name in (("lambda_sensitivity", "Fashion-MNIST"),
                      ("cifar10_lambda_sensitivity", "CIFAR-10")):
        lam = [r for r in runs if r["experiment"] == exp]
        order = sorted({r["method"] for r in lam})
        rows = table(lam, order, {o: o for o in order})
        show(f"lambda sweep on {name}", rows)
        accs = [r[2][0] for r in rows]
        print(f"\n    accuracy spread across lambda: {max(accs) - min(accs):.2f} pp")
        jains = [r[4][0] for r in rows]
        tfls = [r[3][0] for r in rows]
        print(f"    Jain    {min(jains):.3f} to {max(jains):.3f}")
        print(f"    TFLOPs  {min(tfls):.0f} to {max(tfls):.0f}"
              f"   ({100 * (1 - min(tfls) / max(tfls)):.1f}% reduction across the sweep)")

    inner = [r for r in runs if r["experiment"] == "inner_step_ablation"]
    order = sorted({r["method"] for r in inner})
    show("inner-step ablation on CIFAR-10", table(inner, order, {o: o for o in order}))
