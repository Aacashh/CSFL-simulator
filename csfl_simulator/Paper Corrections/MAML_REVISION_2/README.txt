MAML-Select revision package, round 2
Manuscript ID TAI-2026-Mar-L-00619

Upload this zip to Overleaf with New Project -> Upload Project.


NO SUPPLEMENTARY MATERIAL
-------------------------
This submission has no supplement. The manuscript is self-contained and every
claim it makes is supported inside it. The earlier supplement is kept in
not_submitted/ for reference only and is not part of the submission.


CONTENTS
--------
  main.tex                     revised manuscript, clean
  main_marked.tex              the same manuscript with revisions in blue
  Response_to_Reviewers.tex    point-by-point response
  References_letter.bib        24 entries, all cited
  images/                      5 PDF figures, drawn at final printed size

The class is \documentclass[journal]{IEEEtran}. Compile with pdfLaTeX and
Overleaf runs the BibTeX passes itself. Locally:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Response_to_Reviewers needs no BibTeX pass.


LENGTH
------
Seven pages, measured from a compile rather than estimated. Zero errors, zero
undefined references, zero overfull boxes and zero BibTeX warnings. The
response letter is eight pages. The agreed ceiling is eight pages.


FIGURES AND TABLES
------------------
  Fig. 1   system model and workflow
  Fig. 2   CIFAR-100 convergence over 150 rounds
  Fig. 3   efficiency and accuracy trade-off relative to FedAvg
  Fig. 4   selector scaling from N = 20 to N = 1000
  Fig. 5   lambda sensitivity on Fashion-MNIST

  Table I    comparison of representative methods
  Table II   benchmark summary across the three datasets
  Table III  paired significance tests against FedAvg
  Table IV   selector diagnostics
  Table V    lambda sweep and inner-step count on CIFAR-10
  Table VI   state-feature ablation

The feature-ablation bar chart was dropped because it plotted the same seven
numbers Table VI prints.


WHAT CHANGED IN THIS ROUND
--------------------------
Three new CIFAR-100 seeds were run for TiFL, FedCor and CriticalFL, so every
row of Table II now rests on repeated runs. Those three rows were recomputed.

A new subsection reports per-round selector diagnostics. The inner step lowered
the support objective in every round that took one, and the mean adaptation
gain rises with the measured path variation across the three datasets.

Coverage and the Jain index were recomputed with the opening round-robin rounds
removed, to show that full coverage is a property of the policy.

The selector-scaling claim was reconciled with the data behind its figure, and
the figure was regenerated because its labels collided and it plotted a
different campaign from the one the text quoted.

Because the supplement is withdrawn, the paired significance tests, the energy
model constants, the communication accounting, and the CIFAR-10 sensitivity and
inner-step studies were all moved into the manuscript.


REPRODUCING THE NUMBERS
-----------------------
Paper Corrections/maml_data_verification/ recomputes every number from runs/.

    python audit.py     # checks the rendered PDF against freshly computed values
