#!/usr/bin/env python3
"""Static checks on the revised manuscript, for use when no TeX install is present.

Catches the failure modes that actually bite when editing by hand: unbalanced
braces, unbalanced environments, references with no matching label, duplicate
labels, and revision markup that spans a display-math block (which disturbs limit
placement and spacing).  It is not a TeX parser and does not claim to be; a clean
run here means "nothing obviously broken", not "compiles".

Usage:  python3 check_latex.py [file.tex ...]
"""

from __future__ import annotations

import os
import re
import sys

VERBATIM_ENVS = {"verbatim", "lstlisting", "Verbatim"}


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        idx, escaped = None, False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == "%":
                idx = i
                break
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def check_braces(text: str) -> list[str]:
    problems, depth, escaped, line = [], 0, False, 1
    for ch in text:
        if ch == "\n":
            line += 1
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                problems.append(f"line {line}: closing brace with no opener")
                depth = 0
    if depth:
        problems.append(f"unbalanced braces: {depth} left open at end of file")
    return problems


def check_environments(text: str) -> list[str]:
    problems, stack = [], []
    pattern = re.compile(r"\\(begin|end)\{([^}]+)\}")
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        kind, env = match.group(1), match.group(2)
        if kind == "begin":
            stack.append((env, line))
        else:
            if not stack:
                problems.append(f"line {line}: \\end{{{env}}} with no matching \\begin")
            elif stack[-1][0] != env:
                open_env, open_line = stack[-1]
                problems.append(
                    f"line {line}: \\end{{{env}}} closes \\begin{{{open_env}}} from line {open_line}"
                )
                stack.pop()
            else:
                stack.pop()
    for env, line in stack:
        problems.append(f"line {line}: \\begin{{{env}}} never closed")
    return problems


def check_labels(text: str) -> list[str]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    duplicates = []
    seen = set()
    for label in re.findall(r"\\label\{([^}]+)\}", text):
        if label in seen:
            duplicates.append(f"duplicate label: {label}")
        seen.add(label)
    refs = set()
    for command in ("ref", "eqref", "autoref", "cref"):
        refs |= set(re.findall(r"\\" + command + r"\{([^}]+)\}", text))
    missing = sorted(r for r in refs if r not in labels)
    unused = sorted(l for l in labels if l not in refs)
    problems = duplicates + [f"reference to undefined label: {r}" for r in missing]
    if unused:
        problems.append(f"note: labels never referenced: {', '.join(unused)}")
    return problems


def check_revision_markup(text: str) -> list[str]:
    """\\revtwo{...} must not contain a display block; use \\revtwoeq instead."""
    problems = []
    for match in re.finditer(r"\\revtwo\{", text):
        start = match.end() - 1
        depth, i = 0, start
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[start:i]
        for marker in (r"\begin{equation}", r"\begin{align}", r"\[", r"\begin{table"):
            if marker in body:
                line = text.count("\n", 0, match.start()) + 1
                problems.append(
                    f"line {line}: \\revtwo{{...}} spans '{marker}'; split it and use \\revtwoeq"
                )
                break
    return problems


def check_math_delimiters(text: str) -> list[str]:
    problems = []
    for number, line in enumerate(text.split("\n"), 1):
        cleaned = re.sub(r"\\[\$]", "", line)
        if cleaned.count("$") % 2:
            problems.append(f"line {number}: odd number of '$' delimiters")
    return problems


def main() -> None:
    files = sys.argv[1:] or ["manuscript_revised.tex"]
    here = os.path.dirname(os.path.abspath(__file__))
    failed = False
    for name in files:
        path = name if os.path.isabs(name) else os.path.join(here, name)
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        text = strip_comments(raw)
        problems = (
            check_braces(text)
            + check_environments(text)
            + check_math_delimiters(text)
            + check_revision_markup(text)
            + check_labels(text)
        )
        real = [p for p in problems if not p.startswith("note:")]
        notes = [p for p in problems if p.startswith("note:")]
        print(f"=== {os.path.basename(path)}")
        if real:
            failed = True
            for problem in real:
                print(f"  FAIL  {problem}")
        else:
            print("  ok    braces, environments, math delimiters, labels, markup")
        for note in notes:
            print(f"  {note}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
