MAML-Select — Overleaf upload package (IEEE TAI revision)
=========================================================

HOW TO USE
----------
1. Upload this whole folder to a new Overleaf project (keep the images/ subfolder).
2. Compile each root document with pdfLaTeX. They are independent:
     - manuscript_clean.tex          main paper, black text          [IEEEtran]  (7 pages)
     - manuscript_marked_edits.tex   main paper, revisions in blue    [IEEEtran]  (7 pages)
     - response_to_reviewers.tex     point-by-point response          [article]
     - supplementary_material.tex    supplement                       [article]  (5 pages)
3. NO BibTeX is needed. The bibliography is INLINED as a thebibliography block
   in both manuscripts, so a single pdfLaTeX pass resolves all citations.
   (References_letter.bib is included for reference only; it is not used by the build.)

KEY POINTS THIS REVISION
------------------------
- ENERGY ONLY, NO CARBON. All carbon / carbon-footprint / Green-AI reporting was
  removed (it was only a constant rescaling of energy). The paper reports modelled
  ENERGY (Wh) and compute (TFLOPs). Do not reintroduce carbon.
- Energy equation: E_total = sum_t sum_i P_i * T_i^comp / 3600  (Wh).
- Fig. 2 panel (c) is "Energy" (cumulative energy, Wh).
- Table II columns: Acc / Prec / Rec / F1 / TFLOPs / Energy / Jain / Cov. (no Carbon).
- The CodeCarbon citation was removed entirely.
- marked = clean with revisions in blue; clean is generated from marked by turning
  the colour off (\revision -> identity, \color{blue} -> \color{black}).

PENDING (supplement figure)
---------------------------
- The cumulative-trajectories figure in supplementary_material.tex is COMMENTED OUT.
  Its old image (supp_resource_trajectories) had a 3rd "carbon" column and is NOT
  shipped. Rebuild it as a compute+energy-only figure with build_supplement_plots.py
  (the carbon metric is already removed from that script), drop the new PDF into
  images/, then uncomment the figure block. It is not referenced elsewhere, so the
  supplement compiles fine without it.

NOTES
-----
- All figures are in images/ (the .tex files set \graphicspath to images/).
- All result values are mean +/- std over seeds 42 / 123 / 2026.
