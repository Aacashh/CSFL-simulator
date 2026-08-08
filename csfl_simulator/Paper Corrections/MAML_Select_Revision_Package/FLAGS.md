# Things I could not verify, and what I did about them

Written while preparing the second-round revision. Everything in the manuscript
and the reply letter is backed by run logs on this machine **except** the items
below. Where I could not verify a number I left it untouched and recorded it
here rather than guessing.

---

## FLAG 1 — Only CIFAR-100 is reproducible from this repository

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
them and run

```
python3 "csfl_simulator/Paper Corrections/MAML_Select_Revision_Package/analyze_revision.py" <repo-root>
```

which regenerates `revision_numbers.json` for all three datasets.

---

## FLAG 2 — $V_T$ is argued, not yet measured

Corollary 1 carries a drift term $V_T$. The current logs show the query objective
is strongly non-monotone, with jumps of 2.67, 0.70 and 3.61, which *proves the
fixed-objective assumption is false* and justifies replacing the old corollary.
It does **not** measure $V_T$.

Measuring it needs $q_t$ and $q_{t+1}$ at the same iterate. Because the support
set already *is* the previous query set, the drift increment is exactly

```
q_{t+1}(phi_{t+1}) - q_t(phi_{t+1})  =  loss(base params, query) - loss(base params, support)
```

`l_sup_before` was already logged; the query loss at the *base* parameters was
not. I added it, two lines in `_outer_step`, unit-tested, no behaviour change:

```
csfl_simulator/experiments/maml_select/selector.py
    l_query_at_base = float(_loss(model, base_params, qx, qy))
    drift_increment = l_query_at_base - l_sup_before
```

Running `run_maml_revision2.sh --only-drift` yields $V_T$ exactly. Until then the
manuscript says plainly that $V_T$ is not measured and why, at the end of
Sec. V-D. **Do not let that paragraph ship claiming otherwise unless the runs
have actually been done.** The released reference implementation exposes the same
quantity as `Diagnostics.drift_increment`.

---

## FLAG 3 — The paper is about 8.8 pages, and you chose to submit at that length

There is no TeX install on this machine, so I could not compile. `wordcount.py`
models length from running prose at 950 words per float-free two-column page,
plus measured float costs. It reproduced the 10-page figure you observed before
the cuts, which is its calibration point.

Current estimate is **8.8 pages** against an 8-page budget. You elected to submit
at this length rather than cut further, so **confirm with the editor that IEEE
TAI permits overlength pages for this Letter, and at what charge**, before
submitting.

Two corrections to my earlier estimate are worth recording, because both made the
paper longer than I first reported:

* Dropping the precision and recall columns from Table II saved **nothing**
  vertically. Its rows are single-line, so removing columns only widens the
  survivors. I had credited it with about 0.15 page.
* Figures cost about 110 words of displaced text each, not the 170 I assumed.
  This is now measured from the aspect ratios of the image files rather than
  guessed, so removing the schematic bought less than I first reported.

If you later need the two pages after all, the levers in descending value are the
related-work table in Sec. II (about 0.35 page, the single largest float, and it
can be reduced from 13 method rows to about 7 rather than deleted), Fig. 4 (about
0.1 page, and the lambda results are fully stated in the text), and Fig. 2 (about
0.13 page, but it is referenced by the convergence discussion).

---

## FLAG 4 — The reviewer's comment text in the reply letter is paraphrased

I did not have the reviewer's verbatim wording, so the six `\textbf{Comment N:}`
lines in `response_to_reviewers_r2.tex` are my paraphrase of each point.
**Replace each with the exact text before submitting.** The replies address the
substance and do not need changing.

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
* The two single-seed CIFAR-100 rows in Table II are left exactly as submitted,
  per your instruction. Removing the precision and recall columns has the side
  effect of removing the accuracy-versus-recall dispersion mismatch that would
  otherwise have been visible in those rows.
* The abstract, impact statement, introduction and conclusion are byte-identical
  to the round-1 text, apart from two colons in the introduction that you asked
  me to fix. No round-2 markup appears in any of them, so the new contributions
  are not advertised in the abstract or the contribution list. That is a
  deliberate choice, not an oversight.
