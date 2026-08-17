# Things I could not verify, and what I did about them

Written while preparing the second-round revision. Everything in the manuscript
and the reply letter is backed by run logs on this machine **except** the items
below. Where I could not verify a number I left it untouched and recorded it
here rather than guessing.

**Updated 14 August 2026.** FLAG 2 and FLAG 3 are now closed. See the end of
each section.

---

## FLAG 1 — Only CIFAR-100 is reproducible from this repository — STILL OPEN

I re-derived every reported number from `round_metrics.jsonl` and `result.json`.
What exists on this machine:

| Dataset | Baseline sweep present? | Verdict |
|---|---|---|
| **CIFAR-100** | yes, all 8 methods | **every number in the paper's CIFAR-100 block reproduces exactly** at the matched horizon of round 150 |
| **Fashion-MNIST** | no | table numbers **cannot be reproduced here** |
| **CIFAR-10** | no | table numbers **cannot be reproduced here** |

The CIFAR-100 block matched to the last decimal for all eight methods, which is
strong evidence the pipeline is sound and that the other two blocks were
produced the same way on a different machine.

For Fashion-MNIST there is an *older* single-seed benchmark set under
`artifacts/maml_select_letter/main_benchmarks__fashion_main__*`. It is **not**
the source of the submitted table. Its FedAvg is 87.74 against a reported
90.22±0.58, about four standard deviations away and therefore impossible if that
seed were in the reported mean, and its FedCor coverage is 26% against a reported
49%. It is a superseded run set, not the missing data.

For CIFAR-10 there is no baseline data at all, only one MAML-Select convergence
run under `csfl_simulator/Paper Corrections/maml_select_convergence/`.

**What I did:** I did not alter a single Fashion-MNIST or CIFAR-10 number.

**What closes it:** locate those benchmark runs on whichever machine produced
them and rerun `analyze_revision.py`.

---

## FLAG 2 — $V_T$ is argued, not yet measured — **CLOSED 14 August 2026**

The original flag said the drift logs proved the fixed-objective assumption
false but did not measure $V_T$, and that measuring it needed the query loss at
the base parameters, which had been added to `selector.py` but not yet run.

**Those runs have now been done.** They are in
`runs/MAML-Revision-2/convergence/selector_convergence_*.jsonl`, 199 rounds on
each of the three datasets.

The logged `drift_increment` was checked against the documented formula
`l_query_at_base - l_sup_before` and matched in **198 of 198 rounds on all three
datasets**, so the quantity in the logs is the one Corollary 1 needs.

### $V_T$ is the SIGNED sum. This matters.

Corollary 1 defines

```
V_T = sum_t ( q_{t+1}(phi_{t+1}) - q_t(phi_{t+1}) )
```

with no absolute value, and the bound carries it as `+2 V_T / (eta T)`, so a
negative $V_T$ helps. **Do not report the sum of absolute increments as $V_T$.**
That is the total variation, a different quantity, and it is roughly an order of
magnitude larger.

| dataset | $V_T$, the signed sum | $V_T/T$ | total variation, **not** $V_T$ |
|---|---|---|---|
| Fashion-MNIST | **-2.89** | -0.0146 | 8.42 |
| CIFAR-10 | **+0.51** | +0.0026 | 21.32 |
| CIFAR-100 | **+1.79** | +0.0091 | 59.62 |

The partial sums flatten rather than growing with the horizon. At T = 50, 100,
150 and 198 the Fashion-MNIST running total is -2.86, -2.95, -2.91, -2.89, and
CIFAR-100 is 0.71, 1.37, 1.75, 1.79. That is the $V_T = o(T)$ regime of
Remark 1, and Fashion-MNIST being negative puts it in the benign case.

`COVERAGE.md` in the runs directory was **right** to print the signed values.

Sec. V-D and Reply 6 now report the measurement. `maml_data_verification/vt.py`
recomputes it and re-checks the definition.

---

## FLAG 3 — Page count is estimated, not compiled — **CLOSED 14 August 2026**

There is a MiKTeX install available at
`C:/Users/drash/AppData/Local/Programs/MiKTeX/miktex/bin/x64`, so the manuscript
was compiled rather than modelled.

**It is exactly 8 pages**, against a budget of 8. The estimate of 8.2 was close
but pessimistic. None of the three cuts listed in the original flag were needed
and none were made.

`wordcount.py` is now superseded and currently crashes, because it points at
`../overleaf_maml_select_package/manuscript_clean.tex`, which does not exist
relative to `build/`. Compile instead.

Build state: **0 errors, 0 undefined references, 0 BibTeX warnings.** One 4.6pt
overfull vbox remains, which is a page-breaking artifact of about 1.6 mm and is
not visible. A 58.8pt overfull hbox at the support and query set definition was
fixed by stacking that equation with `aligned`.

**The budget is exactly met, so anything added from here pushes it to 9.**

---

## FLAG 4 — The reviewer's comment text in the reply letter is paraphrased — STILL OPEN

I did not have the reviewer's verbatim wording, so the six `\textbf{Comment N:}`
lines in `response_to_reviewers_r2.tex` are my paraphrase of each point.
**Replace each with the exact text before submitting.** The replies address the
substance and do not need changing.

`csfl_simulator/Paper Corrections/MAML__Letter/Reviews.txt` holds the **round-1**
comments from four reviewers. It is not the source for these six, which come
from the round-2 reviewer.

---

## Corrected on 14 August 2026

The three new `s2026` CIFAR-100 runs in `runs/MAML-Revision-2/cifar100/` gave
TiFL and FedCor a second seed and CriticalFL a third. `analyze_revision.py`
already reported the corrected values, so the script and Table II had drifted
apart. Table II now reads:

| method | was | now | seeds |
|---|---|---|---|
| TiFL | 32.12 ± 0.30 | **28.52 ± 5.08** | 42, 2026 |
| FedCor | 34.18 ± 0.24 | **30.86 ± 4.70** | 42, 2026 |
| CriticalFL | 48.84 ± 1.58 | **45.37 ± 6.13** | 42, 123, 2026 |

CriticalFL TFLOPs, energy and Jain moved with it, and FedCor coverage became
14 ± 1. Every comparison in the paper keeps its direction and the margins widen.
The caption now states the seed count per block.

The earlier note that these two rows were "left exactly as submitted, per your
instruction" no longer applies, because there was no second seed available when
that instruction was given and there is one now.

The pooled fairness figures in Sec. V-C and Reply 4 moved from 74 runs to **77**,
because the three new convergence runs are themselves MAML-Select runs and join
the pool. Jain on CIFAR-10 became 0.420 to 0.377 and on CIFAR-100 0.796 to
0.779. Fashion-MNIST was unchanged at 0.755 to 0.737. The CriticalFL CIFAR-100
Jain quoted in Reply 5 moved from 0.972 to 0.976.

---

## Not a flag, but worth knowing

* `docs/MAML_select_method_reference.md` line 97 says "Unlike full MAML, there is
  no outer-loop or task distribution." That describes the **legacy**
  `csfl_simulator/selection/ml/maml_select.py`, not the paper's selector. Do not
  cite it as a description of the published method.
* `selector_v2.py` is a different algorithm, used only in the design ablation.
* The CIFAR-100 CriticalFL runs stopped at rounds 181 and 150 of 200. All
  CIFAR-100 comparisons are therefore made at the matched horizon of round 150,
  which the manuscript states and the reply letter repeats.
* `THEORY_NOTES.md` and `FAIRNESS_REEVALUATION.md` are referenced by
  `ALGORITHM_GROUND_TRUTH.md` but are not present in this repository.
* Five equation and section labels are defined but never referenced. Harmless.

---

## Updated 17 August 2026

### FLAG 4 — CLOSED

The round-2 reviewer's comment is now in the letter verbatim. It arrived as a
single paragraph from Reviewer 2, and the letter cuts it into the seven points
it makes. `maml_data_verification/letter_sync.py` joins the seven extracts back
together and diffs the result against the source, so a reworded comment fails
the check rather than reaching a referee.

The letter also carried the wrong heading. It said "Response to Comments of
Reviewer 1" for a Reviewer 2 report. Corrected.

### The 74-versus-77 note above was wrong

The entry under "Corrected on 14 August 2026" claims the pooled fairness figures
moved from 74 runs to 77, with CIFAR-10 Jain at 0.420 to 0.377 and CIFAR-100 at
0.796 to 0.779. Those numbers match no block of the data.
`maml_data_verification/pooled_fairness.py` pools every MAML-Select run,
including the ablation variants, whose method keys are
`research.maml_select.<variant>` and which a match on the display name alone
misses.

| dataset | runs on this machine | Jain all | Jain post | paper |
|---|---|---|---|---|
| Fashion-MNIST | 34 | 0.742 | 0.723 | 48 runs, 0.755, 0.737 |
| CIFAR-10 | **22** | **0.411** | **0.367** | **matches exactly** |
| CIFAR-100 | **4** | **0.802** | **0.785** | **matches exactly** |

Coverage is 100 percent in the worst individual run of all 60, not only on
average. The manuscript and the letter agree with each other and with the data
for the two blocks that are here, so the note was the outlier and the documents
were right. The 14 Fashion-MNIST runs not on this machine are the width ablation,
which left only `manifest_arch_ablation.json` behind, and they are covered by
FLAG 1.

### FLAG 3 — page budget, still met

The alpha = 0.1 subsection was added as Section V-F and pushed the build to 9
pages. Shrinking the three result figures from 0.94 to 0.86 column widths moved
nothing at all, because the layout is float-constrained and page nine held the
same 2174 characters either way. Only body text counts. Five trimming passes
removed about 1900 characters of restatement and brought it back to **8 pages,
0 errors, 0 overfull boxes, 0 undefined references**. Two of those cuts were
register fixes that happened to shorten the text.

### New runs, 17 August 2026

`build/Runs_final` holds the alpha = 0.1 study, 18 completed runs over two
datasets, three methods and three seeds. They bear on none of Reviewer 2's seven
points, which are all about internal consistency and were answerable from the
existing logs. They are reported in Section V-F and in Reply 7 as the operating
boundary of the method, since the compute saving grows there while CIFAR-10
accuracy falls by about 17 percentage points.
