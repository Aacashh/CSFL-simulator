"""Find paragraphs whose last line carries only one or two words.

Each such paragraph wastes a whole line. Pulling two or three words out of the
paragraph reclaims it. Works on the rendered PDF, since only the renderer knows
where the lines actually break.
"""

import sys

import pymupdf

PDF = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/Paper Corrections/MAML_REVISION_2/build/manuscript_r2_clean.pdf"

MAX_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def paragraph_last_lines(path):
    doc = pymupdf.open(path)
    out = []
    for pno, page in enumerate(doc, 1):
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            if len(lines) < 2:
                continue
            widths = []
            for ln in lines:
                text = "".join(sp["text"] for sp in ln["spans"])
                widths.append((ln["bbox"][2] - ln["bbox"][0], text))
            full = max(w for w, _ in widths)
            last_w, last_t = widths[-1]
            # a short last line in a multi-line block is a widow candidate
            if last_w < 0.42 * full and last_t.strip():
                out.append((pno, len(last_t.split()), last_t.strip(),
                            " ".join(t for _, t in widths[-3:-1])[-70:]))
    return out


if __name__ == "__main__":
    rows = paragraph_last_lines(PDF)
    rows = [r for r in rows if r[1] <= MAX_WORDS]
    print(f"{len(rows)} paragraphs end in {MAX_WORDS} words or fewer\n")
    print(f"{'page':>5}{'words':>7}   trailing text")
    print("-" * 92)
    for pno, n, tail, prev in sorted(rows):
        print(f"{pno:>5}{n:>7}   {tail[:44]:<46}| ...{prev[-42:]}")
    print()
    print(f"reclaiming each is worth one line, so about {len(rows)} lines in total")
