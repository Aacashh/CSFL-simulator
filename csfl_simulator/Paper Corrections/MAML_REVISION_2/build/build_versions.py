#!/usr/bin/env python3
"""Generate the marked and clean copies of the revised manuscript.

Single source of truth is ``manuscript_revised.tex``.  Only the revision-markup
lines in the preamble differ between the two outputs, so the body text can never
drift apart:

    manuscript_r2_marked.tex   second-round changes in red
    manuscript_r2_clean.tex    everything plain, this is the submission copy

First-round ``\\revision{}`` text stays plain in both, because it is already part
of the accepted record and marking it again only obscures what is new.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "manuscript_revised.tex")

PLAIN_REVTWO = r"\newcommand{\revtwo}[1]{#1}"
MARKED_REVTWO = r"\newcommand{\revtwo}[1]{\textcolor{blue}{#1}}"
# Display math uses a colour *switch* rather than \textcolor, because wrapping a
# whole equation body in an argument disturbs limit placement and spacing.
PLAIN_REVTWOEQ = r"\newcommand{\revtwoeq}{}"
MARKED_REVTWOEQ = r"\newcommand{\revtwoeq}{\color{blue}}"

ENVIRONMENTS = ("Lemma", "corollary", "proposition", "proof", "table", "table*")


def build(text: str, marked: bool) -> str:
    for line in (PLAIN_REVTWO, PLAIN_REVTWOEQ):
        if line not in text:
            raise SystemExit(f"marker line not found in {SOURCE}:\n  {line}")
    out = text.replace(PLAIN_REVTWO, MARKED_REVTWO if marked else PLAIN_REVTWO)
    out = out.replace(PLAIN_REVTWOEQ, MARKED_REVTWOEQ if marked else PLAIN_REVTWOEQ)
    # Theorem-like and float environments carry their own colour hook; leave them
    # black in both copies.  New second-round blocks colour themselves via
    # \revtwo, so nothing else needs switching here.
    for env in ENVIRONMENTS:
        hook = "\\AtBeginEnvironment{%s}{\\color{black}}" % env
        if hook not in out:
            raise SystemExit(f"colour hook for '{env}' missing; preamble changed?")
    return out


def main() -> None:
    with open(SOURCE, encoding="utf-8") as handle:
        text = handle.read()

    targets = [("manuscript_r2_marked.tex", True), ("manuscript_r2_clean.tex", False)]
    for name, marked in targets:
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(build(text, marked))
        print(f"wrote {name}  ({'marked' if marked else 'clean'})")

    # A body-text mismatch would mean the two copies say different things, which
    # is exactly the failure the reviewer raised about the last submission.
    bodies = []
    for name, _ in targets:
        with open(os.path.join(HERE, name), encoding="utf-8") as handle:
            body = handle.read().split(r"\begin{document}", 1)
        bodies.append(body[1] if len(body) > 1 else body[0])
    if bodies[0] != bodies[1]:
        print("ERROR: the two copies differ after \\begin{document}", file=sys.stderr)
        raise SystemExit(1)
    print("verified: both copies share an identical body")


if __name__ == "__main__":
    main()
