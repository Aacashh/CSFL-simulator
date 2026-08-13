# SCOPE-FD — page reduction to 13 pages

Handoff note. Read this first if you are a new session picking up this work.

## What this task is

Cut the SCOPE-FD manuscript from **15 pages to 13**, because IEEE TAI charges
per page beyond 10. Quality must not drop. Advait's five levers, in his words:

1. Combine figures into shared rows so 2 floats become 1. He proposed Fig 2+3
   in one row, Fig 4+5 in one row, Fig 6+7 in one row.
2. Merge paragraphs to reclaim the partial last line of each.
3. Shorten long sentences that say a simple thing.
4. Cut references from 45 down to 30–35.
5. Fix paragraphs whose last line holds only 2–3 words.

## Hard rules

- **Never edit the 15-page version.** It lives in
  `Paper Corrections/SCOPE_FD__A_Client_Selection_Method_in_Federated_Distillation_for_Massive_MIMO_Systems_10_August_2026/`
  and is the submittable fallback. All work happens in
  `Paper Corrections/SCOPE_FD_13PAGE/`.
- The 13-page zip is `Paper Corrections/SCOPE_FD_FINAL_13PAGE.zip`.
  The 15-page zip is `Paper Corrections/SCOPE_FD_FINAL.zip`. Keep both.
- **Verify by compiling, never by reading source.** Every claim about pages,
  layout or figures must come from a rendered PDF. This has bitten before.
- Author block at `main.tex` lines 34–36 is deliberately empty. Advait fills it.
  Do not flag it again.

## Build

MiKTeX is not on PATH.

```bash
export PATH="/c/Users/drash/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd <build dir>
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python -c "import pymupdf; print(pymupdf.open('main.pdf').page_count)"
```

Build in a scratch copy, not in the source folder. `main_marked.tex` is
regenerated from `main.tex` and differs only at line 26, where `\rev` is
redefined to print blue.

## Layout facts that matter

- **The layout is float-constrained, not text-constrained.** Scaling all four
  single-column figures to `0.96\columnwidth` once recovered *exactly zero*
  lines, measured. Freed vertical space gets eaten by float glue instead of
  propagating to the end. Prose cuts yield roughly 40 percent of what the raw
  line count predicts. Budget accordingly and re-measure after every change.
- Figure natural sizes, in points:

  | file | w × h | aspect | current use |
  |---|---|---|---|
  | `fig_r4_dirichlet_severity` | 515.5 × 198 | 2.60 | Fig 3, `figure*`, 2 panels |
  | `fig_r3_k_sweep` | 515.5 × 198 | 2.60 | Fig 4, `figure*`, 2 panels |
  | `fig_r5_coefficient_grid` | 252 × 183.6 | 1.37 | Fig 2, 1 col |
  | `fig_r7_channel_sweep` | 252 × 169.2 | 1.49 | Fig 5, 1 col |
  | `fig_r6_scale_fairness` | 252 × 219.6 | 1.15 | Fig 6, 1 col |
  | `fig_r8_ratio_gap` | 252 × 248.4 | 1.01 | Fig 7, 1 col |

  `\columnwidth` ≈ 252 pt, `\textwidth` ≈ 516 pt, column height ≈ 700 pt.
- **Combining two single-column figures into one full-width row does not save
  space.** The row takes the height of the taller one and the shorter one
  leaves whitespace. Computed for Fig 6 + Fig 7: 567 pt of column-space now
  against 605 pt combined. That specific part of lever 1 is a loss. Trim their
  captions instead, which run 5 and 6 lines.
- Section VI runs A through N and `response_to_reviewers.tex` cites those
  letters about fifteen times. **Do not merge or renumber VI subsections.**

## Do not touch — load-bearing honesty

Advait ruled these non-negotiable. They are the things that make the paper
defensible rather than the things that make it look good.

- The exact Gini law and its campaign evidence in VI-M: 459 of 459, 276 plus
  183 arms, seventeen families, 1.33 percent in 313 runs.
- The public-set overlap disclosure in VI-A.
- The dropout carve-out (eighteen runs held out of the Gini-law test).
- The K ≥ 20 trailing result and the cohort-size-versus-ratio confound note.
- The shared-seed-set caveat in VI-K.
- The three-value reconciliation in VI-A (71.21 / 71.75 / 71.99).
- Proposition 1, the two corollaries and their proofs.

## Symbols already fixed — do not reintroduce

- `\varepsilon` is the numerical floor in Eq. (9) only. The differential-privacy
  budget is `\varepsilon_{\mathrm{dp}}`. They collided before and the collision
  made the Proposition 1 proof look broken.
- `n_{\min}` is the minimum participation count in the Prop. 1 proof. It used to
  be `L`, which collided with the local loss `L_n`.
- MIMO expands as "multiple-input multiple-output", not "multi-input".

## Style rules Advait set

Plain English. Short sentences. **No colons, semicolons or dashes in prose.**
Numeric en-dashes like `Assumptions~1--4` are fine. No reviewer-reply register,
the paper addresses the general reader. Every acronym expanded on first use.

## Verification scripts

`Paper Corrections/scope_fd_data_verification/` re-derives every number from
`runs/runs_scope_revised`.

```bash
python load.py       # inventory
python tables.py     # Table II, both blocks
python sweeps.py     # Dirichlet, K, IID
python rest2.py      # sub-grouped by swept parameter
python ginilaw2.py   # the exact Gini law, campaign-wide
```

Two counting rules the numbers depend on: a run counts as complete if it has
`compare_results.json` (filtering on `status == "complete"` silently drops 8 of
17 families), and a "selector run" is one (config, method, seed) arm with method
in {`scope_fd`, `scope_fd_debt_only`}.

## Progress log

Measured as last-page slack in points, where 748 pt is the bottom of the text
area. Higher slack means more free space.

| step | pages | slack | gain |
|---|---|---|---|
| baseline from the 15-page version | 15 | 126 | — |
| references 45 → 35, no 2026 entry touched | 15 | 242 | +116 |
| merged the two full-width sweep figures, trimmed 2 captions | 15 | 610 | +368 |
| ten paragraph merges | 15 | 628 | +18 |
| references 35 → 33 | 15 | 646 | +18 |
| abbreviated 17 author lists to et al. | **14** | 1 | crossed |
| merged Figs. 4, 5 and 6 into one three-panel row | 14 | 181 | +180 |
| conclusion cut to one paragraph, introduction trimmed | 14 | 341 | +160 |
| contributions block trimmed | 14 | 413 | +72 |
| eleven paragraphs tightened | 14 | 493 | +80 |
| three large paragraphs trimmed | 14 | 574 | +81 |
| four more paragraphs tightened | 14 | 628 | +54 |
| final wording pass, Fig. 1 caption | 14 | 655 | +27 |
| abbreviated 12 venue strings in the bib | **13** | 3 | crossed |

**Target met. 13 pages, 0 errors, 0 undefined references, 0 overfull boxes.**

Rejected after measurement:

- Scaling every figure down 15 percent gained only 85 pt and cost legibility.
  Reverted. This reproduces the float-constrained finding above.
- Combining the single-column figures into shared full-width rows. Computed and
  confirmed a net loss, because the row takes the height of the taller figure.
  Only the two full-width 2-panel figures were worth merging, and that one
  change produced the single largest gain of the whole exercise.

## Figures are now regenerated, not composed

The paper carries three floats. Fig. 1 is the TikZ system model. Fig. 2 and
Fig. 3 are generated by `figure_source/make_combined_figures.py`, which imports
the loader and helpers from `figure_source/make_revision_figures.py`.

```bash
cd figure_source && python make_combined_figures.py   # writes figures_combined/
```

- **Fig. 2** `fig_sweeps.pdf`, four panels (a) to (d). Dirichlet sweep then
  cohort-size sweep, each as accuracy and Gini with a margin strip below.
- **Fig. 3** `fig_robust.pdf`, three panels (a) to (c) at identical box height,
  so no panel is padded with whitespace to match a taller neighbour.

Panel tags are drawn into the artwork, so `Fig. 2(a)` in the text matches what
the reader sees. The old per-figure PDFs were deleted. The coefficient-grid
figure was removed entirely and Section VI-C now carries every one of its
numbers in prose.

**Two generator bugs found and fixed while doing this. Do not reintroduce them.**

1. `make_revision_figures.load()` filtered on `man["status"] != "complete"`.
   Older orchestrator runs write no `status` key, so this silently dropped 8 of
   the 17 families and made the figures disagree with the verified tables. The
   filter is now `status == "failed"`, matching `scope_fd_data_verification/load.py`.
2. `_spearman()` in that file ranks with `np.argsort`, which gives tied values
   distinct ranks. Nine of the fifteen `K/N` ratios are tied, so it returns
   -0.754 where the correct average-rank treatment in `ratio.py` returns -0.815.
   **`ratio.py` is right and the paper's -0.815 and -0.764 stand.** Fig. 3(c) no
   longer prints rho at all, so there is nothing to disagree.

Fig. 3(c) also gathers its fifteen configurations from three families, because
the four `N=30` points live in the cohort-size families rather than in
`scale_and_nondivisible`. Its aggregation mirrors `ratio.py` exactly, taking
paired per-seed differences. Verified to reproduce 12/15 positive gaps and the
-1.42 to +2.43 and +0.05 to +2.39 ranges the text cites.

## What actually worked, ranked

1. **Merging figures into shared rows.** The two full-width 2-panel sweep
   figures merged into one four-panel row gave +368 pt. The three
   single-column robustness figures merged into one three-panel row gave
   +180 pt. Together this was more than half the whole saving.
2. **Abbreviating the bibliography.** Author lists to et al. gave the jump from
   15 to 14. Venue strings gave the jump from 14 to 13. Neither drops a
   reference.
3. **Prose tightening**, roughly 25 paragraphs, gave about 470 pt in total.

Note the ordering. Layout beat prose by a wide margin, and prose beat paragraph
merging by a wide margin. Ten paragraph merges gave 18 pt. Do the figures first.

## Superseded, kept for the record

The section below was written when the target looked unreachable at 14 pages.
It was wrong about the ceiling, because it did not account for the three-figure
row or for bibliography abbreviation. It is kept because the cost table is
still the right guide if the paper ever has to go below 13.

## Why 13 looked unreachable at the 14-page stage

A page is about 1400 pt of column space. At 14 pages the last page is full,
slack 1 pt, so reaching 13 means freeing a further ~1400 pt. What is left:

- all six figures together hold about 1620 pt, so even deleting every figure
  barely clears one page
- prose cuts realise only 16 to 30 percent of their raw size here, measured
  twice in this session, so 1400 pt realised needs 5000 pt cut, roughly 40
  percent of the body

**13 pages therefore requires removing content, not rearranging it.** Options,
with what each costs:

| option | frees | cost |
|---|---|---|
| delete Fig. 1, the TikZ system model | ~440 pt | loses the architecture diagram |
| compress Section II background to one page | ~700 pt | thins related-work coverage |
| delete a Section VI subsection (H, I or J) | ~600 pt each | undoes reviewer-responsive work, and `response_to_reviewers.tex` cites VI letters ~15 times |
| move the proofs to supplementary | ~520 pt | paper stops being self-contained |

The cheapest combination that reaches 13 is deleting Fig. 1 plus compressing
Section II plus a further prose pass. Advait has to authorise that. Do not do it
unprompted.

## Known gotcha

Bash heredocs collapse `\\` to `\` even when quoted with `<<'EOF'`. This has
caused `re.PatternError: bad escape` repeatedly. Build regex backslashes with
`chr(92)*2`, or locate bib entries by key rather than by matching their accented
bodies.

## Resync of 13 August, after the figure consolidation

The consolidation from seven floats to three left three files behind. A future
session that changes figures again must check all of these, because none of
them is caught by a LaTeX compile.

- `response_to_reviewers.tex` hard-codes figure numbers, it does not use
  `\ref`. It pointed at Fig. 2 through Fig. 7. Remapped to Fig. 2(a) to (d) and
  Fig. 3(a) to (c). **Quoted Reviewer text keeps the original numbering and must
  stay verbatim**, so leave the "gains in Fig. 3" and "resolution of Figure 1"
  comments alone.
- The Associate Editor reply claimed fifteen pages. It now states thirteen and
  explains the consolidation, so nobody wonders where four figures went.
- `README.txt` ships inside the zip and described six figures, forty-five cited
  references, an estimated rather than compiled page count, and three
  unresolved `[PENDING]` replies. There are now zero PENDING markers.

## arXiv citations

Advait asked for none, because unreviewed work is a weak citation. Status:

- `hsu2019measuring` for the Dirichlet partition was replaced by
  `yurochkin2019bayesian`, ICML 2019, pp. 7252--7261, which introduces the same
  construction and is peer reviewed.
- **`castillo2024unionfl` is still arXiv:2408.13683 and this is deliberate.**
  That preprint proposes both SubTrunc and UnionFL. The peer-reviewed CDC 2024
  paper by the same four authors, `castillo2024subtrunc`, covers SubTrunc only.
  UnionFL therefore has no published venue. Citing the CDC paper for it would be
  a misattribution, and dropping the citation means dropping a baseline that
  Table II reports and that the letter promises to the Associate Editor.
  **Advait decided on 13 August to keep it.** Do not remove it.

The bibliography carries 45 entries of which 33 are cited. The 12 orphans do not
print and are kept as spares.
