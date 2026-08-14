MAML-Select revision package, round 2
Manuscript ID TAI-2026-Mar-L-00619

Upload this zip to Overleaf with New Project -> Upload Project.


CONTENTS
--------
  main.tex                     revised manuscript, clean
  main_marked.tex              the same manuscript with revisions in blue
  Response_to_Reviewers.tex    point-by-point response
  Supplementary_Material.tex   extended evidence
  References_letter.bib        24 entries, all cited
  images/                      5 PDF figures, drawn at final printed size

The class is \documentclass[journal]{IEEEtran}. Compile with pdfLaTeX and
Overleaf runs the BibTeX passes itself. Locally:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Do the same for Response_to_Reviewers and Supplementary_Material, which need no
BibTeX pass.


LENGTH
------
Six pages, measured from a compile rather than estimated. Zero errors, zero
undefined references, zero overfull boxes and zero BibTeX warnings. The
response letter is eight pages and the supplementary is four.


FIGURE AND TABLE LIST
---------------------
  Fig. 1   system model and workflow
  Fig. 2   CIFAR-100 convergence over 150 rounds
  Fig. 3   efficiency and accuracy trade-off relative to FedAvg
  Fig. 4   selector scaling from N = 20 to N = 1000
  Fig. 5   lambda sensitivity on Fashion-MNIST

  Table I    comparison of representative methods
  Table II   benchmark summary across the three datasets
  Table III  selector diagnostics, new in this round
  Table IV   state-feature ablation

The feature-ablation bar chart was removed from the main paper because it
plotted the same seven numbers Table IV prints. The supplementary retains it.


WHAT CHANGED IN THIS ROUND
--------------------------
Three new CIFAR-100 seeds were run for TiFL, FedCor and CriticalFL, so every
row of Table II now rests on repeated runs. Those three rows were recomputed.

A new subsection reports per-round selector diagnostics. The inner step lowered
the support objective in every round that took one, and the mean adaptation
gain rises with the measured path variation across the three datasets.

Coverage and the Jain index were recomputed with the opening round-robin rounds
removed, to show that full coverage is a property of the policy.

The selector-scaling claim was reconciled with the data behind its figure. The
figure was regenerated because its labels collided and it plotted a different
campaign from the one the text quoted.

Every number in the manuscript is reproduced by
Paper Corrections/maml_data_verification/, which reads only runs/.
