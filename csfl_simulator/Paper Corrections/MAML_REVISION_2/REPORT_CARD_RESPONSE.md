# Response to the R2 report card

Written 14 August 2026 against `MAML-Select_R2_report_card.md`. Every number
below was re-derived from the run logs by
`csfl_simulator/Paper Corrections/maml_data_verification/report_card.py`.

State after this pass: manuscript **8 pages** (clean and marked), reply letter
4 pages, 0 LaTeX errors, 0 overfull boxes, 0 undefined references, all 23
references, 4 figures, 3 tables and Algorithm 1 present.

---

## Blocking items

### 1. The energy model did not implement Eq. (5) — FIXED

The report card is right, and the diagnosis is right. `simulator.py:312` is

```python
compute_seconds = epochs * float(client.data_size) / max(1e-6, float(client.compute_speed))
```

There is no `C_model`. The FLOP accounting at `simulator.py:463` does carry it
(`3.0 * model_macs_per_sample * local_epochs * data_size`), which is why Wh per
TFLOP is 41.1 on Fashion-MNIST and 0.573 on the two ResNet18 datasets.

Confirmed by closed form. With the pool mean of `P_i/f_i` equal to 3.45 W per
unit rate, the predicted energy per round with **no** `C_model` is

| dataset | predicted | observed |
|---|---|---|
| Fashion-MNIST | 28.75 Wh | 28.76 Wh |
| CIFAR-10 | 23.96 Wh | 23.99 Wh |
| CIFAR-100 | 23.96 Wh | 24.17 Wh |

Three significant figures. The equations were wrong, not the runs.

**What changed.** Eq. (5) is now `T_comp = E n_i / f_{i,t}` with `f` a
throughput in samples per second. Eq. (13) accumulates over `T_{i,t}` rather
than `T_{i,t}^{comp}`, which is what the code does. `C_model` is now defined at
Eq. (12) where it is actually used. The false sentence "`C_model` matches
`T_comp`, so cost and latency stay consistent" is gone. `T_comm` is stated in
the form the simulator uses, with the measured note that it is under 1% of
`T_comp`. Reply 2 of the letter records the correction, since the Reviewer's
Comment 2 is exactly about inconsistent latency definitions.

Side effect worth noting: dropping the fixed delay `d_{i,t}` from Eq. (5)
removed the last `delta` symbol collision, which was previously resolved by
renaming rather than by deletion.

### 2. Small-shard artifact — ANSWERED, and it is an identity

`TFLOPs_total = 3 C_model E sum(n_i)`, so within a dataset the cumulative TFLOPs
ratio **is** the selected-sample ratio. No estimation needed.

| dataset | pool mean shard | FedAvg | MAML-Select |
|---|---|---|---|
| Fashion-MNIST | 600 | 600 | 523 (−12.9%) |
| CIFAR-10 | 500 | 500 | 393 (−21.5%) |
| CIFAR-100 | 500 | 500 | 492 (−1.7%) |

Energy adds exactly one more factor, the tier power per unit throughput. The
two compose to the published reductions:

| dataset | sample term | tier term | product | published |
|---|---|---|---|---|
| Fashion-MNIST | 12.9% | 4.1% | 16.4% | 16.4% |
| CIFAR-10 | 21.5% | 4.2% | 24.8% | 24.7% |
| CIFAR-100 | 1.7% | 3.0% | 4.7% | 4.7% |

Cross-checked independently against the measured tier shares. Mean `P_i/f_i` is
3.4500 for the pool, 3.4435 for FedAvg, 3.4530 for FedGCS and 3.3360 for
MAML-Select, so MAML-Select sits at 0.967 of the pool average. That reproduces
the tier term.

The paper now states the mechanism plainly and prices it, and notes that the
sample budget alone does not determine accuracy, since FedCS trains on 46%
fewer Fashion-MNIST samples and loses 6.1 points where MAML-Select trains on
12.9% fewer and loses 0.1.

### 3. No zero-inner-step control — NOT DONE, blocked on compute

Two obstacles, both real.

- `selector.py:136` is `for _ in range(max(1, int(inner_steps)))`, so
  `inner_steps=0` is silently clamped to 1. The control needs a code change
  before it needs a run.
- `torch` here is `2.12.0+cpu` with no CUDA. Three seeds of ResNet18 on CIFAR-10
  for 100 rounds is not feasible on this machine in any useful time.

The Conclusion now states that a no-adaptation control is left open, so the
paper does not claim what it has not measured.

### 4. Ablation protocol undefined — FIXED

The CIFAR-10 ablations run at **T = 100 rounds**, against 200 in Table II.
Everything else is identical, N = 100, K = 10, alpha = 0.5, three seeds. That is
the whole of "diagnostic". Verified from every `config.json` in
`runs/maml_select_review_hardening`.

The Fashion-MNIST groups, including the width ablation in
`runs/maml_select_arch_ablation`, all run the full 200 rounds and are directly
comparable.

Table III's caption and the Fig. 4 caption now state the horizon, the
state-feature group header carries "200 rounds", and the body text says "63.8%
at 100 rounds".

The three Jain values are now reconciled. They were never inconsistent, they
answer different questions. 0.77 in Table II is the default configuration over
three seeds. 0.78 in Table III is the feature-ablation campaign's full-state
runs. 0.755 in Section V-C is **pooled over every run and every setting**,
including the large-lambda runs, which is why it is lower. Section V-C now says
so.

### 5. CIFAR-100 near-dominated by FedGCS — STATED, and the latency answers it

Correct as reported. FedGCS leads on accuracy, F1, TFLOPs and Jain. What the
report card could not see is that the paper never reported the metric its own
objective optimizes. Mean round latency on CIFAR-100, over the same runs that
produce the Table II accuracies:

| method | mean T_round | vs FedAvg | time to 50% |
|---|---|---|---|
| FedAvg | 2624 s | 1.000x | 47.7 h |
| FedGCS | 2538 s | 0.967x | 56.1 h |
| **MAML-Select** | **2246 s** | **0.856x** | 48.6 h |

MAML-Select is the fastest per round of every method that keeps full coverage,
14.4% below FedAvg and 11.5% below FedGCS. Now in Section V-A and the abstract.

The abstract no longer says "significantly lower" anywhere. It gives the three
per-dataset numbers, says the saving is small on CIFAR-100, names FedGCS as
stronger there on accuracy, and marks coverage as holding **by construction**.

---

## Substantive items

### 6. Baseline fidelity — DISCLOSED, cannot be repaired here

The report card is right and it is worse than stated.

- `selection/system_aware/tifl.py` sorts each tier's ids ascending and takes
  `pop(0)` from a list rebuilt every round, so the same lowest-id clients are
  chosen forever. There is no probabilistic tier sampling, which is TiFL's
  defining mechanism.
- `selection/system_aware/oort.py` gives `ucb = 0` when `n_i == 0`. Standard UCB
  gives an unplayed arm an unbounded bonus. Here an unselected client gets
  nothing, so the exploration term rewards clients already selected. The pacer
  is absent.

That fully explains 10% and 12% coverage. Repairing it means re-running every
baseline on three datasets, which this machine cannot do. The setup paragraph
now discloses both implementations and says their accuracies are those of these
variants. This is the honest option, and it pre-empts the referee.

### 7. Statistics — FIXED

Equivalence-from-a-large-p is gone. The text now says a difference "was not
detected" and gives the width of what could still be hiding. Intervals computed
from the published summary statistics at n = 3, two-sided 95%:

| comparison | difference | 95% CI |
|---|---|---|
| Fashion-MNIST, vs FedAvg | −0.11 pp | [−1.96, +1.74] |
| CIFAR-10, vs FedGCS | −2.25 pp | [−7.86, +3.36] |
| CIFAR-100, vs FedGCS | −0.51 pp | [−1.81, +0.79] |
| CIFAR-100, vs FedAvg | −1.51 pp | [−2.97, −0.05] |

Five or more seeds is a run, not an edit. Not done.

### 8. Latency never reported — FIXED. See item 5.

### 9. Coverage as a headline — FIXED. The abstract says by construction.

### 10. Scope, one alpha and one K — STATED, not fixed

alpha = 0.1 needs runs. The Conclusion now states the scope limit explicitly
rather than leaving it to the reader.

---

## Correctness notes

- **Exploration sign error.** Confirmed against `selector.py:290`, which sorts
  ascending on `(-recency, participation_count, coverage_rank)` and takes the
  first, i.e. the **stalest** client. The prose and Algorithm 1 said "largest in
  (−tau, nu)", which is the least stale. Both are fixed, the prose now naming
  the rule in words and the algorithm saying "smallest". The third tie-break by
  coverage order is now mentioned, which is what makes the slot deterministic.
- **Proposition 2 window.** Statement now says N−1, matching its proof, and the
  proof notes that selection through the learned ranking also resets staleness.
  Reply 4 of the letter follows.
- **"one ResNet18 round"** is now "one client-round". 8341 TFLOPs over 200
  rounds and 10 clients is 4.17 per client-round.
- **The 77 runs were 74.** The pool in `revision_numbers.json` lists each of the
  three instrumented convergence runs twice. Distinct counts are 48
  Fashion-MNIST, 22 CIFAR-10, 4 CIFAR-100. Recomputed over distinct runs the
  pooled Jain pairs are 0.755 to 0.737, **0.411 to 0.367** and **0.802 to
  0.785**, so two of the three published pairs moved. Manuscript and letter both
  updated.
- "six parameters" is now "six client-state features", subject-verb agreement
  fixed, index terms no longer end in "and", tier direction stated.

---

## Venue and formatting

- `\subsubsection` under `\section` promoted to `\subsection` in Related Work
  and the Methodology. Section IV now reads A State Space, B From the Cohort
  Objective, C The MAML-Select Algorithm, D Computational Complexity. **This
  moved the phase ordering from IV-A to IV-C**, and Reply 1 of the letter was
  updated to match.
- Table III moved from `[H]` to `[!t]`.
- **Reply 2 pointed at Eq. (8), which is the inner-loop update.** The cost
  function is Eq. (7). This was wrong before this pass as well. Fixed.
- **Submission category is not mine to change.** The report card says TAI has no
  Letters type and that this manuscript overruns a Brief. The manuscript number
  is TAI-2026-Mar-L-00619 and the 8-page target has been held throughout, so the
  category question goes to the author.

---

## Space

The new evidence cost about 720 pt of column height and the paper had 41 pt of
slack. It was recovered without deleting any result, by pulling stray last lines
out of 30 paragraphs, cutting three sentences that were already made by a
caption or an earlier paragraph, trimming five captions, and taking the four
figures down by 6 to 8 percent of column width. Page 8 is now full to the last
line, so anything added from here costs a ninth page.
