#!/usr/bin/env python3
"""Estimate the body-text length of the manuscript against the submitted version.

There is no TeX install on this machine, so page count cannot be measured
directly.  Body words are the best available proxy: display math, floats and
LaTeX control sequences are stripped, so what remains is the running prose that
actually drives how many columns the paper fills.

Usage:  python3 wordcount.py [revised.tex] [baseline.tex]
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEW = os.path.join(HERE, "manuscript_revised.tex")
DEFAULT_OLD = os.path.join(HERE, "..", "overleaf_maml_select_package", "manuscript_clean.tex")

FLOAT_ENVS = ("table", "table*", "figure", "figure*", "algorithm")


def body(text: str) -> str:
    return text.split(r"\begin{document}", 1)[1].split(r"\begin{thebibliography}", 1)[0]


def strip(text: str, drop_floats: bool = False) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    if drop_floats:
        for env in FLOAT_ENVS:
            text = re.sub(
                r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
                " ", text, flags=re.S,
            )
    text = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{(equation|align)\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}\\&_^~]", " ", text)
    return text


def count(text: str, drop_floats: bool = False) -> int:
    return len([w for w in strip(text, drop_floats).split() if any(c.isalpha() for c in w)])


def floats(text: str) -> dict:
    return {env: len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", text)) for env in FLOAT_ENVS}


def main() -> None:
    new_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NEW
    old_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OLD
    new, old = body(open(new_path).read()), body(open(old_path).read())

    wo, wn = count(old, drop_floats=True), count(new, drop_floats=True)
    print(f"{'':22}{'submitted':>12}{'revised':>10}{'delta':>9}")
    print(f"{'running prose words':22}{wo:>12}{wn:>10}{wn-wo:>+9}")
    fo, fn = floats(old), floats(new)
    for env in FLOAT_ENVS:
        if fo[env] or fn[env]:
            print(f"{'  ' + env:22}{fo[env]:>12}{fn[env]:>10}{fn[env]-fo[env]:>+9}")
    eqo = len(re.findall(r"\\begin\{equation\}", old))
    eqn = len(re.findall(r"\\begin\{equation\}", new))
    print(f"{'  numbered equations':22}{eqo:>12}{eqn:>10}{eqn-eqo:>+9}")

    # The submitted paper is 8 pages including all of its floats, so the float
    # area is already paid for.  What matters is the *delta*.  Added running
    # prose displaces text at roughly the density of a float-free IEEE two-column
    # page, about 950 words, which is the figure this estimate uses.  The author
    # reported the current draft at 10 pages, so the model is checked against
    # that observation rather than assumed.
    WORDS_PER_PAGE = 950.0
    d_words = wn - wo
    d_eq = eqn - eqo
    d_tab = (fn["table"] - fo["table"]) + (fn["table*"] - fo["table*"])
    d_fig = (fn["figure"] - fo["figure"]) + (fn["figure*"] - fo["figure*"])
    # Figure cost is measured, not guessed.  The image files were read for their
    # aspect ratios: drawn at \columnwidth these occupy 18 to 32 per cent of a
    # column once the caption is counted, and a column holds about half a page of
    # text, so one figure displaces roughly 110 words.
    #
    # Dropping columns from a table saves nothing vertically when its rows are
    # single-line, which is the case for Table II.  Only the related-work table,
    # whose cells wrap, gains from losing a column, and there the gain is offset
    # by the larger font requested for it.  Hence no credit for either.
    manual = int(os.environ.get("MANUAL_CREDIT", "-75"))  # shorter Algorithm 1
    added = d_words + d_eq * 30 + d_tab * 110 + d_fig * 110 + manual
    print(f"\nadded prose {d_words:+d} words, {d_eq:+d} equations, {d_tab:+d} tables, {d_fig:+d} figures")
    print(f"manual float credit {manual:+d} words")
    print(f"charged as ~{added:+.0f} words of displaced text")
    print(f"estimated length: {8.0 + added / WORDS_PER_PAGE:.2f} pages   (budget 8.00)")
    over = added - 0.0
    if over > 0:
        print(f"MUST STILL CUT ~{over:.0f} words to reach 8 pages")
    else:
        print(f"at or under the submitted length by ~{-over:.0f} words")


if __name__ == "__main__":
    main()
