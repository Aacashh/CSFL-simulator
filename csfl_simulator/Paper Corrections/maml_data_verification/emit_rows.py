"""Emit the CIFAR-100 block of Table I as LaTeX, straight from the run data.

Printing the rows from the same code that verifies them removes the chance of a
transcription error between the harness and the manuscript.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load

HORIZON = 150
DEC = {"acc": 2, "prec": 2, "rec": 2, "f1": 2, "tflops": 0,
       "energy": 0, "carbon": 0, "jain": 2, "cov": 0}
COLS = ["acc", "prec", "rec", "f1", "tflops", "energy", "carbon", "jain", "cov"]


def cell(entries, col):
    mu, sd, _n = load.mean_sd([load.read(e, col) for e in entries])
    d = DEC[col]
    return f"{mu:.{d}f}$\\pm${sd:.{d}f}"


if __name__ == "__main__":
    runs = [r for r in load.load_all() if r["experiment"] == "cifar100_benchmarks"]
    by = {}
    for r in runs:
        e = load.at_round(r, HORIZON)
        if e:
            by.setdefault(r["method"], []).append(e)

    print("% CIFAR-100 block, regenerated from runs/ at horizon round 150")
    print("% by maml_data_verification/emit_rows.py, do not hand edit")
    for m in load.METHOD_ORDER:
        if m not in by:
            continue
        name = f"\\textbf{{{m}}}" if m == "MAML-Select" else m
        cells = " & ".join(cell(by[m], c) for c in COLS)
        print(f"{name:<20} & {cells} \\\\")

    print()
    print("% seed counts")
    for m in load.METHOD_ORDER:
        if m in by:
            print(f"%   {m:<13} n = {len(by[m])}")

    fa = by["FedAvg"]
    ms = by["MAML-Select"]
    print()
    print("% MAML-Select against FedAvg on CIFAR-100 at round 150")
    for c in ("acc", "tflops", "energy", "carbon"):
        a = load.mean_sd([load.read(e, c) for e in fa])[0]
        b = load.mean_sd([load.read(e, c) for e in ms])[0]
        if c == "acc":
            print(f"%   {c:<8} {b - a:+.2f} pp")
        else:
            print(f"%   {c:<8} {100 * (a - b) / a:.1f}% lower")
