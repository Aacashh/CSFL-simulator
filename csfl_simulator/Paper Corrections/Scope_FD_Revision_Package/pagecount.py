#!/usr/bin/env python3
"""Estimate the compiled length. Calibrated against the 16-page build of 8 Aug.

There is no TeX toolchain here, so length is modelled from running prose plus
float area. The constants below were fitted so the model returns 16.0 for the
manuscript as compiled on 8 August 2026.
"""
import re, sys
from pathlib import Path

TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("main_scope_clean.tex")
BIB = Path(__file__).with_name("references.bib")

WORDS_PER_PAGE = 876.0
COST = {"figure": 150, "figure*": 300, "table": 150, "table*": 300,
        "algorithm": 200, "equation": 32}
REF_WORDS = 13.0

def body(t):
    return t.split(r"\begin{document}", 1)[1].split(r"\bibliographystyle", 1)[0]

def prose_words(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    for env in ("table", r"table\*", "figure", r"figure\*", "algorithm"):
        t = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{(equation|align)\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\\&_^~]", " ", t)
    return len([w for w in t.split() if any(c.isalpha() for c in w)])

def main():
    b = body(TEX.read_text())
    w = prose_words(b)
    counts = {k: len(re.findall(r"\\begin\{" + k.replace("*", r"\*") + r"\}", b)) for k in COST}
    refs = len(re.findall(r"@\w+\{", BIB.read_text()))
    float_cost = sum(counts[k] * COST[k] for k in COST)
    total = w + float_cost + refs * REF_WORDS
    pages = total / WORDS_PER_PAGE
    print(f"  prose words        {w}")
    for k in COST:
        if counts[k]:
            print(f"  {k:<18} {counts[k]:>3}  x {COST[k]:>3} = {counts[k]*COST[k]:>5}")
    print(f"  references         {refs:>3}  x {REF_WORDS:>3.0f} = {refs*REF_WORDS:>5.0f}")
    print(f"  ---")
    print(f"  estimated pages    {pages:.2f}")
    return pages

if __name__ == "__main__":
    main()
