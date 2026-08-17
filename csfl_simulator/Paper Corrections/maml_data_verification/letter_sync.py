"""Check the reply letter against the manuscript it points into.

Three things go wrong between a letter and a paper. A pointer names a section or
an equation that has moved, a number is quoted in one document and corrected only
in the other, and a reviewer comment is reproduced with a word changed. This
checks all three, reading the manuscript from the compiled aux file and the
rendered PDF so the numbering is the one the reader sees.
"""

import io
import os
import re
import sys

import pymupdf

BASE = ("c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/"
        "Paper Corrections/MAML_REVISION_2/build/")
LETTER = BASE + "response_to_reviewers_r2.tex"

# Reviewer 2 wrote one paragraph. The letter cuts it into these extracts, and
# joining them has to give the paragraph back.
SOURCE = (
    "The manuscript requires major revision because several foundational "
    "aspects of the proposed method remain internally inconsistent or "
    "insufficiently justified. Most importantly, the main manuscript and "
    "supplementary material describe different optimization procedures, and "
    "the latency-cost function is not defined consistently. The authors must "
    "establish one definitive algorithm and ensure that the mathematical "
    "equations, pseudocode, implementation, experiments, and theoretical "
    "analysis all correspond to it. In addition, the per-client ranking score "
    "is not currently derived from the cohort-level optimization objective and "
    "should either be justified as a valid surrogate or clearly presented as a "
    "heuristic. The fairness and coverage results must be re-evaluated because "
    "the supplementary procedure includes forced client coverage, and "
    "contradictory CriticalFL coverage values appear across the main and "
    "supplementary tables. Finally, the approximation-error and convergence "
    "claims require substantial strengthening, particularly because the "
    "theoretical analysis assumes stationarity while the method is motivated "
    "by non-stationary client utility. These issues affect the validity, "
    "reproducibility, and interpretation of the central contribution and must "
    "be resolved before the manuscript can be considered further."
)

# Numbers both documents quote. A value present in one and absent from the
# other is where the two have drifted apart.
SHARED = [
    ("74", "pooled MAML-Select run count"),
    ("0.755", "Fashion Jain, full run"),
    ("0.737", "Fashion Jain, post cold start"),
    ("0.411", "CIFAR-10 Jain, full run"),
    ("0.367", "CIFAR-10 Jain, post cold start"),
    ("0.802", "CIFAR-100 Jain, full run"),
    ("0.785", "CIFAR-100 Jain, post cold start"),
    ("-2.89", "V_T on Fashion-MNIST"),
    ("1.79", "V_T on CIFAR-100"),
    ("198", "adapted rounds checked"),
    ("55.8", "alpha=0.1 Fashion TFLOPs"),
    ("140.0", "alpha=0.1 Fashion FedAvg TFLOPs"),
    ("83.4", "alpha=0.1 Fashion accuracy"),
    ("41.2", "alpha=0.1 CIFAR-10 accuracy"),
    ("58.6", "alpha=0.1 CIFAR-10 FedAvg accuracy"),
]


def aux_labels():
    """label -> printed number, from the compiled manuscript."""
    out = {}
    src = io.open(BASE + "manuscript_r2_clean.aux", encoding="utf-8",
                  errors="ignore").read()
    # The printed value is the first brace group. IEEEtran wraps a subsection
    # in \mbox, which gives {{\mbox {V-C}}{6}{}...}, so the group is read up to
    # the brace that precedes the page number.
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{\d", src):
        val = m.group(2).replace(r"\mbox", "")
        val = val.replace("{", "").replace("}", "")
        out[m.group(1)] = " ".join(val.split())
    return out


def manuscript_text():
    doc = pymupdf.open(BASE + "manuscript_r2_clean.pdf")
    text = " ".join(" ".join(p.get_text() for p in doc).split())
    # the PDF carries a Unicode minus, which no source file contains
    return text.replace("−", "-")


def quoted_spans(text):
    """Character mask over every \\textit{...}, which holds the Reviewer's words."""
    mask = [False] * len(text)
    i = 0
    while True:
        j = text.find(r"\textit{", i)
        if j < 0:
            return mask
        depth, k = 1, j + len(r"\textit{")
        while k < len(text) and depth > 0:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        for x in range(j, k):
            mask[x] = True
        i = k


def main():
    letter = io.open(LETTER, encoding="utf-8").read()
    labels = aux_labels()
    printed = manuscript_text()
    bad = 0

    print("=" * 70)
    print("1. THE REVIEWER'S COMMENT, REPRODUCED")
    print("=" * 70)
    mask = quoted_spans(letter)
    extracts = []
    for m in re.finditer(r"\\textit\{", letter):
        j = m.start()
        depth, k = 1, m.end()
        while k < len(letter) and depth > 0:
            if letter[k] == "{":
                depth += 1
            elif letter[k] == "}":
                depth -= 1
            k += 1
        extracts.append(" ".join(letter[m.end():k - 1].split()))
    # the word "italicized" in the preamble is not a comment
    extracts = [e for e in extracts if len(e) > 40]
    joined = " ".join(extracts)
    if joined == SOURCE:
        print("  the %d extracts reproduce the comment exactly" % len(extracts))
    else:
        bad += 1
        print("  DIFFERS from the source comment")
        import difflib
        for line in difflib.unified_diff(SOURCE.split(), joined.split(),
                                         "source", "letter", lineterm="", n=2):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                print("      %s" % line)

    print()
    print("=" * 70)
    print("2. POINTERS FROM THE LETTER INTO THE MANUSCRIPT")
    print("=" * 70)
    wanted = {
        "Section~IV-B": ("From the Cohort Objective", None),
        "Section~IV-C": (None, "sec:method_selection"),
        "Section~V-A": (None, "sec:results_acc"),
        "Section~V-B": (None, "sec:tier"),
        "Section~V-C": (None, "sec:fairness_recheck"),
        "Section~V-D": (None, "sec:selector_convergence"),
        "Section~V-F": (None, "sec:alpha01"),
        "Section~III": (None, None),
        "Eq.~(4)": (None, "eq:zscore"),
        "Eq.~(5)": (None, "eq:sets"),
        "Eq.~(6)": (None, "eq:straggler_bound"),
        "Eq.~(7)": (None, "eq:cost"),
        "Eq.~(13)": (None, "eq:energy"),
        "Table~II": (None, "tab:main_summary"),
        "Algorithm~1": (None, "alg:maml_select"),
        "Proposition~1": (None, "prop:topk"),
        "Proposition~2": (None, "prop:coverage"),
        "Corollary~1": (None, "cor:rate"),
        "Remark~1": (None, "rem:reading"),
        "Lemma~1": (None, "stmt:inner"),
        "Eq.~(3)": (None, "eq:bi_objective"),
    }
    used = sorted({p for p in wanted if p in letter})
    for p in used:
        _, label = wanted[p]
        number = p.split("~")[-1].strip("()")
        ok = True
        detail = ""
        if label:
            got = labels.get(label)
            ok = got == number
            detail = "%s prints as %s" % (label, got)
        else:
            detail = "no label, checked by hand"
        if not ok:
            bad += 1
        print("  %-4s %-16s %s" % ("ok" if ok else "FAIL", p, detail))
    unused = sorted(set(wanted) - set(used))
    if unused:
        print("  (not cited by the letter: %s)" % ", ".join(unused))

    print()
    print("=" * 70)
    print("3. NUMBERS QUOTED IN BOTH DOCUMENTS")
    print("=" * 70)
    for token, what in SHARED:
        inl = token in letter
        inm = token in printed
        ok = inl and inm
        if not ok:
            bad += 1
        print("  %-5s %-8s %-34s letter %s   manuscript %s"
              % ("ok" if ok else "FAIL", token, what,
                 "yes" if inl else "NO ", "yes" if inm else "NO "))

    print()
    print("%d problems" % bad)
    return 1 if bad else 0


sys.exit(main())
