# Report-card evidence, what to run and what comes back

Everything here is code and configuration only. Nothing was executed on the
Windows box, which has `torch 2.12.0+cpu` and no CUDA.

## The one command

```bash
bash csfl_simulator/experiments/maml_select/run_report_card_evidence.sh
```

Runs stages A, B and C. Resumable, so an interrupted campaign restarts where it
stopped and a finished stage costs nothing to re-enter.

```bash
STAGES=A      bash .../run_report_card_evidence.sh   # blocking control only, 12 runs
STAGES=AB     bash .../run_report_card_evidence.sh   # + severe non-IID, 30 runs
STAGES=ABCD   bash .../run_report_card_evidence.sh   # everything, 123 runs
DEVICE=cuda   bash .../run_report_card_evidence.sh   # pin the device
ANALYZE_ONLY=1 bash .../run_report_card_evidence.sh  # rebuild tables, no training
SEEDS="42 123 2026 7 99" bash .../run_report_card_evidence.sh   # five seeds
```

It refuses to start on CPU unless you pass `ALLOW_CPU=1`, because stage C alone
is 69 runs.

## Stages

| stage | runs | what it answers | list item |
|---|---|---|---|
| **A** | 12 | inner steps 0 against 1, CIFAR-10 at 100 rounds and CIFAR-100 at 150, three seeds | 1, blocking |
| **B** | 18 | alpha = 0.1 on Fashion-MNIST and CIFAR-10, FedAvg and FedGCS alongside | 2 |
| **C** | 69 | all eight methods on Fashion-MNIST and CIFAR-10, plus the seven feature ablations | 3, 4, 5, blocking |
| **D** | 24 | CIFAR-100 benchmarks again, optional | 3, optional |

Run counts verified against `--dry-run`, not estimated.

Stage C carries three of the five items at once. It produces the round-latency
and time-to-target columns, it logs the selected shard sizes, and it puts the
benchmark MAML-Select row and the full-state ablation row in **one campaign** so
they reconcile by construction.

Stage D is optional. The CIFAR-100 shard sizes are already recoverable from the
TFLOPs column without re-running anything, and the analysis does that
automatically for any run that predates the new logging.

## Code changes

### `selector.py`, the control was impossible to run

```python
for _ in range(max(1, int(inner_steps))):   # before
for _ in range(max(0, int(inner_steps))):   # after
```

`inner_steps=0` was silently clamped to 1, so the control would have been a
relabelled copy of the default. `_outer_step` now also skips the support set
entirely at 0, which makes the outer update a plain Adam step on the query loss
at the un-adapted weights. That is online regression with the same
6-64-64-1 network, which is the comparison the report card asks for.

**Values of 1 and above are untouched, so every published ablation number stands.**
`test_selector_inner_steps.py` pins both halves of that, and the bash script
asserts it in preflight before spending any GPU time.

### `simulator.py`, the shard sizes were never recorded

Per round: `selected_sample_counts` and `mean_selected_samples`.
Per run: `client_sample_counts` and `client_tiers` for the whole pool, so the
selected shards are compared against the population they came from rather than
against an assumed mean.

### `configs.yaml`

Three scenarios, `cifar10_alpha_0p1`, `cifar100_review_150`, and a
`report_accuracy_target` on `fashion_alpha_0p1` so time-to-target is defined.
Two experiments, `no_adaptation_control` and `heterogeneity_alpha_0p1`.
`main_benchmarks` and `feature_ablation` gain the `report_card_main` profile so
stage C runs them into a fresh directory without touching the existing runs.

## What comes back

`artifacts/maml_select/report_card/report_card_tables.txt`, five sections in the
order of your list, with LaTeX-ready cells at the end of each. Send that file
back and I will fold it into the manuscript. Send the run directories too if you
want the numbers re-derived here rather than trusted.

The analysis already works against the runs on this machine, so its output is
not speculative. Today it prints, for CIFAR-100:

```
   method         mean T_round (s)  vs FedAvg  time to target (h)   n
   FedAvg                     2624     1.000x                47.7   3
   FedGCS                     2538     0.967x                56.1   2
   MAML-Select                2245     0.856x                48.6   3
```

and it reports the two reconciliation candidates side by side, currently

```
   benchmark table, research.maml_select        no runs found
   ablation table, research.maml_select.full    acc 90.23+-0.55  jain 0.776  n=3
```

which is item 4 exactly. The 90.11 in the benchmark table came from a campaign
that is not on this machine.

## On item 4, the reconciliation

The two rows are the same configuration, so the 0.12 pp and 0.01 Jain gap is not
a design difference. Seeding is by seed alone, not by run label, so identical
configurations should agree bit for bit. They do not, and the likely cause is
`performance_mode: true` in the scenario defaults, which enables cuDNN benchmark
mode and TF32 and gives up bit-for-bit reproducibility on GPU.

The fix is structural rather than cosmetic. **Report both rows from the same
runs.** Stage C makes that possible, and `report_card_tables.py` prints both
candidates and states the gap so it can never drift silently again.

## One thing I found that is not on your list

The manuscript's Phase 3 does not describe what the code does.

Paper: Phase 1 gives `phi'_t` by adapting `phi_t` on `D_sup(t)`, and Phase 3
scores with `h_{phi'_t}`.

Code: the outer step adapts on the support set and updates the model, and then
scoring adapts the **updated** weights one step on `D_query(t)`, the newest
completed round. So the weights that score the cohort are a third vector the
paper never defines, adapted on `t-1` feedback rather than `t-2`.

This is the same class of issue as Reviewer Comment 1. It is arguably the better
algorithm, since scoring on the freshest feedback is what you want, so the fix is
to the text and not to the code. It needs one sentence in Phase 3 and a matching
line in Algorithm 1. I have not touched it, because changing how the method is
described is your call and it does not block the runs.
