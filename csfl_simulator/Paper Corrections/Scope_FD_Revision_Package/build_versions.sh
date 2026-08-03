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

marked = src.replace(NOOP, "\\newcommand{\\rev}[1]{\\textcolor{blue}{#1}}", 1)
open("main_scope_marked.tex", "w").write(marked)
open("main_scope_clean.tex", "w").write(src)

n = src.count("\\rev{")
print(f"  \\rev{{}} blocks: {n}")
print("  wrote main_scope_marked.tex (blue) and main_scope_clean.tex")
PY

for f in main_scope_marked.tex main_scope_clean.tex; do
    printf "  %-26s %s lines\n" "$f" "$(wc -l < "$f" | tr -d ' ')"
done
echo
echo "Compile with:  pdflatex <file> && bibtex <file> && pdflatex <file> && pdflatex <file>"
