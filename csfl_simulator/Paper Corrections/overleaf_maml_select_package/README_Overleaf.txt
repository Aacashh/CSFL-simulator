MAML-Select — Overleaf upload package (IEEE TAI revision)
=========================================================

HOW TO USE
----------
1. Zip this whole folder (keep the images/ subfolder) and upload to a new
   Overleaf project, or drag the contents in directly.
2. Compile each document with pdfLaTeX. They are independent root files:
     - manuscript_clean.tex          main paper, black text          [IEEEtran]
     - manuscript_marked_edits.tex   main paper, revisions in blue    [IEEEtran]
     - response_to_reviewers.tex     point-by-point response          [article]
     - supplementary_material.tex    supplement S1-S5                 [article]
3. All figures are in images/ (the .tex files set \graphicspath to images/).
   The .bbl files are included, so the bibliography renders even if Overleaf
   does not re-run BibTeX. References_letter.bib is the source bibliography.

WHAT CHANGED THIS ROUND (second-round ablations)
------------------------------------------------
Main manuscript (clean + marked):
  - Lambda sensitivity now on BOTH Fashion-MNIST and CIFAR-10:
      * old Fashion-only figure replaced by a two-panel figure
        (images/fig_lambda_two_dataset_boxed.pdf);
      * a two-dataset lambda table added.
  - Inner-step (CIFAR-10) and selector-width (Fashion-MNIST) ablations:
      summarized in one short paragraph in the main text; full tables/figures
      are in the Supplementary Material (kept out of the main text for length).
  - Dataset-asymmetry justification paragraph added.
  - Formal definition of Jain's fairness index added.
  - Abstract, contributions, and conclusion updated to list the new ablations.

Response letter (response_to_reviewers.tex):
  - Reviewer 2 / Comment 3 (lambda): reply now reports both datasets.
  - Reviewer 1 / Comment 2 (architecture): reply now reports the selector-width
    and inner-step ablations, with one pointer to the supplement.
  - Summary bullets and Reviewer 3 / Comment 5 updated.

Supplement (supplementary_material.tex):
  - Section S3 now includes the inner-step and selector-width ablation figures
    (images/fig_inner_step_boxed.pdf, images/fig_arch_width_boxed.pdf).

NOTES
-----
- The *.pdf at the top level and in pdfs/ are PRE-REVISION builds; Overleaf will
  regenerate current PDFs from the .tex files on compile.
- All new ablation values are mean +/- std over seeds 42/123/2026.
