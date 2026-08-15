"""Validate every pointer the reply letter makes into the manuscript.

A letter that sends a referee to the wrong section is a small error with a large
cost, and section letters move whenever a subsection is added or removed. This
reads the headings out of the rendered manuscript and checks each reference in
the letter against them.

It also checks the bracketed reference numbers. Those move too. The Reviewers
cite FedTSKD as [13], its number in the original submission, but the revision
added references ahead of it and the same paper is now [12]. Any number the
letter types by hand is checked against the rendered reference list.
"""

import io
import re

import pymupdf

BASE = ("c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/"
        "Paper Corrections/SCOPE_FD_13PAGE/")

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}


def manuscript_headings():
    """Section and subsection headings as the reader sees them."""
    doc = pymupdf.open(BASE + "main.pdf")
    text = "\n".join(p.get_text() for p in doc)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    sections, subs = {}, {}
    current = None
    expected = 1
    for line in text.split("\n"):
        s = " ".join(line.split())
        # IEEEtran renders the ninth subsection as "I." which is also a roman
        # numeral, so a heading only opens a new section when it is the next one
        # in sequence. Otherwise "I. Client Dropout" would end section VI early.
        m = re.match(r"^([IVX]+)\.\s+([A-Z][A-Za-z].{2,70})$", s)
        if m and ROMAN.get(m.group(1)) == expected:
            current = m.group(1)
            sections[current] = m.group(2).strip()
            expected += 1
            continue
        m = re.match(r"^([A-Z])\.\s+([A-Z][A-Za-z].{2,70})$", s)
        if m and current:
            subs["%s-%s" % (current, m.group(1))] = m.group(2).strip()
    return sections, subs


def letter_references():
    src = io.open(BASE + "response_to_reviewers.tex", encoding="utf-8").read()
    refs = []
    for m in re.finditer(r"Section~([IVX]+)(?:-([A-Z]))?\b", src):
        refs.append((m.group(1) + ("-" + m.group(2) if m.group(2) else ""),
                     src[max(0, m.start() - 70):m.start()].replace("\n", " ")[-58:]))
    for m in re.finditer(r"(Table|Fig\.)~([IVX]+|\d)", src):
        refs.append((m.group(1) + " " + m.group(2),
                     src[max(0, m.start() - 70):m.start()].replace("\n", " ")[-58:]))
    return refs


def manuscript_floats():
    doc = pymupdf.open(BASE + "main.pdf")
    text = "\n".join(p.get_text() for p in doc).upper()
    found = set()
    for m in re.finditer(r"TABLE\s+([IVX]+)\b", text):
        found.add("Table " + m.group(1))
    for m in re.finditer(r"FIG\.\s*(\d)", text):
        found.add("Fig. " + m.group(1))
    return found


def reference_list():
    """number -> the first sixty characters of the entry, as rendered."""
    doc = pymupdf.open(BASE + "main.pdf")
    text = " ".join(" ".join(p.get_text() for p in doc).split())
    tail = text[text.rfind("REFERENCES"):]
    out = {}
    for m in re.finditer(r"\[(\d+)\] ([^\[]{10,})", tail):
        out[int(m.group(1))] = " ".join(m.group(2).split())[:60]
    return out


def letter_reference_numbers():
    """Bracketed numbers in our own replies, with the Reviewers' quotes removed.

    A number inside \\textit{} is the Reviewer's, and it refers to the version
    they read, so it is not ours to check or to change.
    """
    src = io.open(BASE + "response_to_reviewers.tex", encoding="utf-8").read()
    quoted = [False] * len(src)
    i = 0
    while True:
        j = src.find(r"\textit{", i)
        if j < 0:
            break
        depth, k = 1, j + len(r"\textit{")
        while k < len(src) and depth > 0:
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
            k += 1
        for x in range(j, k):
            quoted[x] = True
        i = k
    return sorted({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", src)
                   if not quoted[m.start()]})


# The paper the letter argues about, and the words that identify it.
EXPECTED = {12: "Mu"}

sections, subs = manuscript_headings()
floats = manuscript_floats()
refs = reference_list()

print("MANUSCRIPT HEADINGS AS RENDERED")
for k in sorted(sections, key=lambda x: ROMAN[x]):
    print("  %-6s %s" % (k + ".", sections[k]))
    for sk in sorted(s for s in subs if s.startswith(k + "-")):
        print("      %-8s %s" % (sk, subs[sk]))
print()
print("FLOATS PRESENT:", ", ".join(sorted(floats)))

print()
print("LETTER POINTERS")
bad = 0
seen = {}
for ref, ctx in letter_references():
    if ref in seen:
        continue
    if ref.startswith(("Table", "Fig")):
        ok = ref in floats
        target = ref if ok else "NOT IN MANUSCRIPT"
    elif "-" in ref:
        ok = ref in subs
        target = subs.get(ref, "NO SUCH SUBSECTION")
    else:
        ok = ref in sections
        target = sections.get(ref, "NO SUCH SECTION")
    seen[ref] = ok
    if not ok:
        bad += 1
    print("  %-4s %-12s -> %s" % ("ok" if ok else "FAIL", ref, target))

print()
print("REFERENCE NUMBERS THE LETTER TYPES IN OUR OWN REPLIES")
for n in letter_reference_numbers():
    entry = refs.get(n, "NO SUCH REFERENCE")
    want = EXPECTED.get(n)
    ok = n in refs and (want is None or want in entry)
    if not ok:
        bad += 1
    print("  %-4s [%d] -> %s" % ("ok" if ok else "FAIL", n, entry))
if not letter_reference_numbers():
    print("  the letter types no reference numbers of its own")

print()
print("%d pointers checked, %d broken" % (len(seen), bad))
raise SystemExit(1 if bad else 0)
