#!/usr/bin/env bash
# Build the marked and clean manuscripts from the single revised source.
#
#   bash build_versions.sh
#
# Both outputs are generated from main_scope_revised.tex and differ only in the
# definition of \rev{}, so the two versions can never drift apart.
#   main_scope_marked.tex  ->  \rev{...} renders blue
#   main_scope_clean.tex   ->  \rev{...} renders as normal text
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC="main_scope_revised.tex"
[[ -f "$SRC" ]] || { echo "missing $SRC"; exit 1; }

python3 - <<'PY'
src = open("main_scope_revised.tex").read()
NOOP = "\\newcommand{\\rev}[1]{#1}"
assert NOOP in src, "revision macro not found"

# \color is a switch, so it is legal across a paragraph break. \textcolor
# is not \long and silently loses text when its argument contains \par,
# which is how the marked copy came to differ from the clean one.
marked = src.replace(NOOP, "\\newcommand{\\rev}[1]{{\\color{blue}#1}}", 1)
open("main_scope_marked.tex", "w").write(marked)
open("main_scope_clean.tex", "w").write(src)

n = src.count("\\rev{")

# Guard: a marked-only rendering difference is invisible in a text diff, so
# check the two failure modes that caused one before.
import re as _re
def _blocks(t, m="\\rev{"):
    out, i = [], 0
    while True:
        j = t.find(m, i)
        if j < 0:
            return out
        k, d = j + len(m), 1
        while k < len(t) and d:
            if t[k] == "\\":
                k += 2
                continue
            d += (t[k] == "{") - (t[k] == "}")
            k += 1
        out.append((j, k))
        i = k
_bad_par = [b for b in _blocks(src) if _re.search(r"\n\s*\n", src[b[0]+5:b[1]-1])]
_bad_str = [b for b in _blocks(src)
            if _re.search(r"\\\\(?:sub)*section\\*?\\{|\\\\begin\\{(?:figure|table|algorithm)",
                          src[b[0]+5:b[1]-1])]
if _bad_par:
    print(f"  !! {len(_bad_par)} \\rev{{}} block(s) span a paragraph break")
if _bad_str:
    print(f"  !! {len(_bad_str)} \\rev{{}} block(s) wrap a float or a heading")
print(f"  \\rev{{}} blocks: {n}   paragraph-spanning: {len(_bad_par)}   float-wrapping: {len(_bad_str)}")
print("  wrote main_scope_marked.tex (blue) and main_scope_clean.tex")
PY

for f in main_scope_marked.tex main_scope_clean.tex; do
    printf "  %-26s %s lines\n" "$f" "$(wc -l < "$f" | tr -d ' ')"
done
echo
echo "Compile with:  pdflatex <file> && bibtex <file> && pdflatex <file> && pdflatex <file>"
