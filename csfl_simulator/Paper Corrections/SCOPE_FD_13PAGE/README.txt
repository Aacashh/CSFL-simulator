SCOPE-FD revision package
Manuscript ID TAI-2026-May-A-00956

Upload this zip to Overleaf with New Project -> Upload Project.


CONTENTS
--------
  main.tex                    revised manuscript, clean
  main_marked.tex             the same manuscript with round-2 changes in blue
  response_to_reviewers.tex   point-by-point response
  convergence_proof_note.tex  standalone note on the partial-participation proof
  references.bib              45 entries, 33 of them cited
  figures/                    2 PDF figures, drawn at final printed size

The class is \documentclass[journal]{IEEEtran}, which is the IEEE journal
template Overleaf ships. Nothing else needs uploading. No .cls or .bst file is
bundled, because supplying an unverified copy is worse than using the one
Overleaf maintains. Compile with pdfLaTeX and Overleaf runs the BibTeX passes
itself. Locally:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main


LENGTH
------
Thirteen pages, measured from a compile rather than estimated. Zero errors, zero
undefined references, zero overfull boxes and zero BibTeX warnings. Abstract 227
words against the 250 limit, written to the six-part structure the journal asks
for and carrying no formulas. Impact statement 133 words against the 100 to 150
range.

The previous build ran to fifteen pages. The reduction came from consolidating
the figures into multi-panel floats, setting the bibliography at scriptsize,
abbreviating author lists and venue strings, and tightening prose. No section was
removed, no reference was dropped and no result was cut.


FIGURE LIST
-----------
  Fig. 1   system model, drawn inline with TikZ, no external file
  Fig. 2   fig_sweeps    four panels. (a) and (b) sweep the Dirichlet
                         concentration, (c) and (d) sweep the cohort size
  Fig. 3   fig_robust    three panels. (a) downlink SNR sweep, (b) fairness
                         across pool and cohort sizes, (c) accuracy gap
                         against the participation ratio

Panel tags are drawn into the artwork itself, so a reference to Fig. 2(a) in the
text matches what the reader sees on the page. The figures are regenerated from
the campaign data by figure_source/make_combined_figures.py, which is kept in the
project folder and is not part of the upload.


THE CLEAN AND MARKED COPIES
---------------------------
Both are generated from one source and differ by exactly one preamble line, the
definition of \rev. Their bodies are byte identical, so they cannot drift apart.
Do not edit them separately.

The marked copy defines \rev as {\color{blue}#1} rather than \textcolor. This
matters. \textcolor is not \long, so when its argument contains a paragraph
break TeX raises "Paragraph ended before \textcolor was complete" and drops the
text. That is what made an earlier marked build shorter than the clean one.
\color is a switch and is legal across paragraphs.
