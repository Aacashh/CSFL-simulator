SCOPE-FD revision package
Manuscript ID TAI-2026-May-A-00956

Upload this zip to Overleaf with New Project -> Upload Project.


LAYOUT
------
Standard IEEE layout. Overleaf will pick main.tex as the main document on its
own, since that is the name it looks for first.

  main.tex                    revised manuscript, clean
  main_marked.tex             the same manuscript with round-2 changes in blue
  response_to_reviewers.tex   point-by-point response
  references.bib              45 entries, all 45 cited, no orphans
  figures/                    6 figures, PDF, drawn at final printed size

The class is \documentclass[journal]{IEEEtran}, which is the IEEE journal
template Overleaf ships. Nothing else needs uploading. No .cls or .bst file is
bundled, because supplying an unverified copy is worse than using the one
Overleaf maintains. Compile with pdfLaTeX and Overleaf runs the BibTeX passes
itself. Locally:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main


THE CLEAN AND MARKED COPIES
---------------------------
Both are generated from one source and differ by exactly one preamble line, the
definition of \rev. Their bodies are byte identical, so they cannot drift apart.
Do not edit them separately. Edit main_scope_revised.tex in the project folder
and run build_versions.sh.

The marked copy defines \rev as {\color{blue}#1} rather than \textcolor. This
matters. \textcolor is not \long, so when its argument contains a paragraph
break TeX raises "Paragraph ended before \textcolor was complete" and drops the
text. That is what made the previous marked build shorter than the clean one.
\color is a switch and is legal across paragraphs. build_versions.sh now also
reports how many \rev blocks span a paragraph break or wrap a float, and both
counts should stay at zero.


LENGTH
------
Abstract 249 words against the 250 limit, written to the six-part structure the
journal asks for and carrying no formulas.
Impact statement 150 words against the 100 to 150 range.
Estimated length about 14.6 pages against the 15-page limit, which leaves room
for roughly 350 more words of new results.

The estimate is modelled from running prose plus float area and calibrated
against the 16-page build of 8 August. It is not a compile, since no TeX
toolchain was available. Check it on the first Overleaf build.

Two figures were removed to reach this length. They reproduced Table II and
Table III number for number, so nothing was lost. The remaining figures are
renumbered, and every cross-reference in the manuscript and in the response
letter was remapped to match.


FIGURE LIST
-----------
  Fig. 1   system model, drawn inline with TikZ, no external file
  Fig. 2   fig_r5_coefficient_grid     sensitivity to the two coefficients
  Fig. 3   fig_r4_dirichlet_severity   Dirichlet sweep, full width
  Fig. 4   fig_r3_k_sweep              sparsity sweep, full width
  Fig. 5   fig_r7_channel_sweep        downlink SNR sweep
  Fig. 6   fig_r6_scale_fairness       fairness across pool and cohort sizes
  Fig. 7   fig_r8_ratio_gap            accuracy gap against K/N


STILL OPEN BEFORE SUBMISSION
----------------------------
Three replies in the response letter are marked [PENDING] in red. All three are
the same question from different reviewers, namely whether the FedTSKD
convergence result carries over to deterministic partial participation. They are
R1 comment 1, R2 comment 3, and comment 1 of the attached review. No experiment
settles this. It needs either a partial-participation theorem or the more modest
statement Reviewer 2 offers. Search the letter for PENDING.
