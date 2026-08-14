"""Final screening of the SCOPE-FD submission.

Two jobs.

1.  Punctuation that reads as machine-written. Colons, semicolons and dashes
    used as prose connectives. LaTeX uses all three legitimately, in optional
    arguments, in \\hskip, in ranges and in hyphenated compounds, so the scan
    strips maths, commands and comments before looking.
2.  The vocabulary and sentence habits that mark generated text, plus the
    mechanical checks that matter at submission, namely spacing defects, word
    limits and number agreement across the two documents.
"""

import io
import os
import re
import sys
from collections import Counter

BASE = ("c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/"
        "Paper Corrections/SCOPE_FD_13PAGE/")
FILES = ["main.tex", "response_to_reviewers.tex"]

# Words and phrases that mark generated academic prose.
TELLS = [
    "delve", "delves", "delving", "leverage", "leverages", "leveraging",
    "seamless", "seamlessly", "pivotal", "crucial", "crucially",
    "underscore", "underscores", "underscoring", "testament",
    "landscape", "realm", "tapestry", "intricate", "myriad",
    "furthermore", "moreover", "additionally", "notably", "importantly",
    "it is worth noting", "it is important to note", "it should be noted",
    "in conclusion", "in summary", "overall,", "comprehensive",
    "cutting-edge", "state-of-the-art", "paradigm shift", "game-changer",
    "robustly", "holistic", "nuanced", "multifaceted", "elevate",
    "unlock", "harness", "navigate", "foster", "showcase", "showcases",
    "meticulous", "meticulously", "profound", "vital", "paramount",
    "not only", "but also", "a wide range of", "a variety of",
    "plays a role", "plays a key role", "significant improvement",
]


def strip_latex(text):
    """Leave prose behind. Maths, commands, comments and citations go."""
    out = []
    for raw in text.split("\n"):
        line = raw
        if line.lstrip().startswith("%"):
            out.append("")
            continue
        line = re.sub(r"(?<!\\)%.*$", "", line)          # trailing comments
        line = re.sub(r"\$[^$]*\$", " MATH ", line)       # inline maths
        line = re.sub(r"\\\[.*?\\\]", " MATH ", line)     # display maths
        line = re.sub(r"\\cite\w*\{[^}]*\}", " CITE ", line)
        line = re.sub(r"\\ref\{[^}]*\}", " REF ", line)
        line = re.sub(r"\\label\{[^}]*\}", " ", line)
        line = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", line)
        line = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", line)  # commands
        line = re.sub(r"[{}&]", " ", line)
        out.append(line)
    return out


def in_environment(lines, envs=("tabular", "tabular*", "table", "algorithmic",
                                "thebibliography", "equation", "align")):
    """Mark lines that sit inside a table or similar, where punctuation differs."""
    flags = [False] * len(lines)
    depth = 0
    for i, line in enumerate(lines):
        for e in envs:
            depth += len(re.findall(r"\\begin\{" + re.escape(e) + r"\}", line))
            depth -= len(re.findall(r"\\end\{" + re.escape(e) + r"\}", line))
        flags[i] = depth > 0
    return flags


def screen_punctuation(name, raw):
    print()
    print("-" * 74)
    print("PUNCTUATION,", name)
    print("-" * 74)
    lines = strip_latex(raw)
    rawlines = raw.split("\n")
    inenv = in_environment(rawlines)

    findings = Counter()
    for i, line in enumerate(lines):
        if inenv[i]:
            continue
        # a colon or semicolon between two words is a prose connective
        for m in re.finditer(r"\w\s*([;:])\s+\w", line):
            findings[m.group(1)] += 1
            print("  line %-5d %-3s ...%s..." % (i + 1, m.group(1),
                                                 line[max(0, m.start() - 45):m.start() + 30].strip()))
        # em dash and en dash used as punctuation rather than in a range
        for m in re.finditer(r"(?<![0-9])\s(-{2,3})\s(?![0-9])", line):
            findings["dash"] += 1
            print("  line %-5d %-3s ...%s..." % (i + 1, m.group(1),
                                                 line[max(0, m.start() - 45):m.start() + 30].strip()))
        for m in re.finditer(r"\w(\u2014|\u2013)\w|\s(\u2014|\u2013)\s", line):
            findings["unicode dash"] += 1
            print("  line %-5d dash  ...%s..." % (i + 1,
                                                  line[max(0, m.start() - 45):m.start() + 30].strip()))
    if not findings:
        print("  none in prose")
    else:
        print("  totals:", dict(findings))
    return sum(findings.values())


def screen_vocabulary(name, raw):
    print()
    print("-" * 74)
    print("VOCABULARY AND HABITS,", name)
    print("-" * 74)
    prose = " ".join(strip_latex(raw)).lower()
    prose = re.sub(r"\s+", " ", prose)
    hits = 0
    for t in TELLS:
        n = len(re.findall(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", prose))
        if n:
            hits += n
            print("  %-28s %d" % (t, n))
    if not hits:
        print("  none of the %d tracked markers appear" % len(TELLS))
    return hits


def screen_mechanics(name, raw):
    print()
    print("-" * 74)
    print("MECHANICS,", name)
    print("-" * 74)
    problems = 0

    # a full stop followed straight by a capital with no space
    for m in re.finditer(r"[a-z\]\)]\.[A-Z][a-z]", raw):
        problems += 1
        print("  missing space after full stop: ...%s..." %
              raw[max(0, m.start() - 30):m.start() + 25].replace("\n", " "))
    # doubled spaces inside a sentence
    for m in re.finditer(r"[a-z],  +[a-z]", raw):
        problems += 1
        print("  double space: ...%s..." % raw[max(0, m.start() - 25):m.start() + 25])
    # doubled words
    for m in re.finditer(r"(?<![a-zA-Z])([a-z]{3,})\s+\1(?![a-zA-Z])", raw):
        problems += 1
        print("  doubled word %r" % m.group(1))
    # straight quotes where TeX wants directional ones
    n = len(re.findall(r'(?<!\\)"', raw))
    if n:
        problems += n
        print("  straight double quotes:", n)
    if not problems:
        print("  clean")
    return problems


def screen_limits():
    print()
    print("-" * 74)
    print("SUBMISSION LIMITS")
    print("-" * 74)
    src = io.open(BASE + "main.tex", encoding="utf-8").read()
    blocks = re.findall(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    for i, a in enumerate(blocks):
        plain = re.sub(r"\\[a-zA-Z]+", " ", a)
        plain = re.sub(r"[{}$\\]", " ", plain)
        words = [w for w in plain.split() if any(c.isalnum() for c in w)]
        if i == 0:
            print("  abstract          %3d words   limit 250   %s"
                  % (len(words), "ok" if len(words) <= 250 else "OVER"))
        else:
            ok = 100 <= len(words) <= 150
            print("  impact statement  %3d words   100 to 150  %s"
                  % (len(words), "ok" if ok else "OUT OF RANGE"))
    title = re.search(r"\\title\{(.*?)\}\s*\n", src, re.S)
    if title:
        print("  title:", " ".join(title.group(1).split()))
    for f in ("main.pdf", "main_marked.pdf", "response_to_reviewers.pdf"):
        p = BASE + f
        if os.path.exists(p):
            import pymupdf
            print("  %-30s %2d pages" % (f, pymupdf.open(p).page_count))
    for f in ("main.log", "main_marked.log"):
        p = BASE + f
        if os.path.exists(p):
            log = io.open(p, encoding="utf-8", errors="ignore").read()
            print("  %-30s errors %d  overfull %d  undefined %d"
                  % (f, log.count("\n!"), log.count("Overfull"),
                     len(re.findall(r"undefined", log, re.I))))


def screen_numbers():
    """Numbers the manuscript and the letter both quote have to agree."""
    print()
    print("-" * 74)
    print("NUMBER AGREEMENT BETWEEN MANUSCRIPT AND LETTER")
    print("-" * 74)
    man = io.open(BASE + "main.tex", encoding="utf-8").read()
    let = io.open(BASE + "response_to_reviewers.tex", encoding="utf-8").read()
    checks = [
        ("71.21", "headline accuracy"), ("70.99", "uniform random accuracy"),
        ("71.18", "UnionFL accuracy"), ("21.14", "Oort accuracy"),
        ("12.49", "uniform random Gini"), ("83.33", "Oort Gini"),
        ("1.40", "N=47 K=6 Gini"), ("1.25", "N=53 K=7 Gini"),
        ("55.56", "UnionFL Gini at K=1"), ("22.89", "UnionFL Gini at K=10"),
        ("13.94", "client sd, uniform random at K=1"),
        ("9.11", "client sd, SCOPE-FD at K=1"),
        ("459", "invariance count"), ("0.815", "rank correlation"),
    ]
    for token, what in checks:
        a, b = man.count(token), let.count(token)
        flag = "" if (a and b) or (a and not b) else "  <-- in letter only"
        print("  %-8s %-34s manuscript %d   letter %d%s" % (token, what, a, b, flag))
    for bad in ("0.0002", "0.0009", "strengthens rather than weakens",
                "strengthens as the pool grows"):
        a, b = man.count(bad), let.count(bad)
        if a or b:
            print("  STALE  %-40s manuscript %d   letter %d" % (bad, a, b))
    print("  no stale p-values or pool-size claims"
          if not any(man.count(x) + let.count(x)
                     for x in ("0.0002", "0.0009", "strengthens as the pool grows"))
          else "  ATTENTION above")


if __name__ == "__main__":
    total_punct = total_vocab = total_mech = 0
    for f in FILES:
        raw = io.open(BASE + f, encoding="utf-8").read()
        total_punct += screen_punctuation(f, raw)
        total_vocab += screen_vocabulary(f, raw)
        total_mech += screen_mechanics(f, raw)
    screen_limits()
    screen_numbers()
    print()
    print("=" * 74)
    print("prose punctuation findings %d, vocabulary markers %d, mechanics %d"
          % (total_punct, total_vocab, total_mech))
    print("=" * 74)
