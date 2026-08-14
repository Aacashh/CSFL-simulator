"""Check acronym expansion in the manuscript source.

Works from the .tex so that section headings and table captions, which IEEEtran
renders in small caps and which therefore look like acronyms in extracted PDF
text, do not drown the real candidates.
"""

import io
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/Paper Corrections/MAML_REVISION_2/build/manuscript_r2_clean.tex"

# Acronyms a reader could reasonably need expanded, and their expected expansion.
WATCH = {
    "FL": "Federated Learning",
    "MAML": "Model-Agnostic Meta-Learning",
    "IID": "independent and identically distributed",
    "MLP": "multilayer perceptron",
    "TFLOPs": "Tera Floating-Point Operations",
    "FLOPs": "floating-point operations",
    "GFLOPs": "giga floating-point operations",
    "SGD": "stochastic gradient descent",
    "MSE": "mean squared error",
    "DQN": "deep Q-network",
    "LSH": "locality-sensitive hashing",
    "RL": "reinforcement learning",
    "GP": "Gaussian Process",
    "IoT": "Internet of Things",
    "IoMT": "Internet of Medical Things",
    "CNN": "convolutional neural network",
    "ReLU": "rectified linear unit",
    "Wh": "watt-hour",
    "SNR": "signal-to-noise ratio",
}


def body(path):
    s = io.open(path, encoding="utf-8").read()
    start = s.find(r"\begin{abstract}")
    end = s.find(r"\begin{thebibliography}")
    s = s[start:end if end > 0 else len(s)]
    B = chr(92)
    s = re.sub("(?<!" + B + B + ")%[^\n]*", "", s)     # drop comments
    return s


if __name__ == "__main__":
    s = body(TEX)
    flat = re.sub(r"\s+", " ", s)

    print(f"{'acronym':<10}{'uses':>6}{'first at':>10}   status")
    print("-" * 76)
    problems = []
    for acr, expansion in sorted(WATCH.items()):
        uses = [m.start() for m in re.finditer(r"\b" + re.escape(acr) + r"\b", flat)]
        if not uses:
            continue
        first = uses[0]
        # look for the expansion anywhere before or at the first use
        head = flat[:first + len(acr) + 60].lower()
        key = expansion.lower().split()[0]
        has_full = expansion.lower() in head
        loose = key in head and len(expansion.split()) > 1
        if has_full:
            status = "expanded before or at first use"
        elif loose:
            status = "PARTIAL, check wording"
            problems.append((acr, expansion, first, flat[max(0, first - 130):first + 70]))
        else:
            status = "NOT EXPANDED"
            problems.append((acr, expansion, first, flat[max(0, first - 130):first + 70]))
        print(f"{acr:<10}{len(uses):>6}{first:>10}   {status}")

    print()
    if not problems:
        print("every watched acronym is expanded before or at its first use")
    for acr, exp, pos, ctx in problems:
        print(f"\n  {acr}  expected expansion: {exp}")
        print(f"    ...{ctx.strip()}...")

    # An acronym expanded twice is also a fault.
    print("\nre-expansions after the first use")
    for acr, expansion in sorted(WATCH.items()):
        hits = [m.start() for m in re.finditer(re.escape(expansion), flat, re.I)]
        if len(hits) > 1:
            print(f"  {acr:<8} '{expansion}' written out {len(hits)} times")
