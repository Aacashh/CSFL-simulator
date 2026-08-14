# MAML-Select, revision round 2

Handoff note. Read this first if you are a new session picking up this work.

## What this paper is

**MAML-Select: An Online Adaptive Client Selection Method for Federated Learning
via Meta-Learning.** IEEE Transactions on Artificial Intelligence, **Letter**,
Manuscript ID **TAI-2026-Mar-L-00619**. Four reviewers in round 1. The round-1
response is already written and was submitted. This round adds the evidence from
`runs/MAML-Revision-2`.

## Hard rules

- **Never edit the round-1 version.** It lives in
  `Paper Corrections/MAML__Letter/` and is the submitted fallback. All work
  happens in `Paper Corrections/MAML_REVISION_2/`.
- **Verify by compiling, never by reading source.** Every claim about pages,
  layout or figures must come from a rendered PDF.
- **Every number in the paper must come from `maml_data_verification/`.** If the
  script and the paper disagree, the paper is wrong.
- Author block at `main.tex` lines 48--50 is commented out. Advait fills it.

## Style rules Advait set

Plain English. Short sentences. **No colons, semicolons or dashes in prose.**
Numeric en-dashes like `0.61--0.78` are fine. The manuscript is written as a
paper, **not as an answer to a reviewer**. The reply letter carries the
reviewer-facing register, the paper does not.

## Build

MiKTeX is not on PATH. Build in a scratch copy, not in the source folder.

```bash
export PATH="/c/Users/drash/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
python -c "import pymupdf; print(pymupdf.open('main.pdf').page_count)"
```

`main.tex` and `main_marked.tex` differ by exactly one line, the definition of
`\revision`. Keep them in step.

## Layout facts

- Baseline before this round: **6 pages, 0 errors, 0 undefined refs, 0 overfull
  boxes**, and the last page is **exactly full, 0 pt slack**. Anything added
  must be paid for by trimming, or the paper goes to 7.
- The initial submission was typeset by IEEE at 8 pages
  (`TAI-2026-Mar-L-00619_Proof_hi.pdf`). Atypon typesetting runs longer than
  IEEEtran, so 6 IEEEtran pages is not 6 typeset pages.
- `\graphicspath` was repointed to `{images/}{./}` for this folder.

## Verification harness

`Paper Corrections/maml_data_verification/` recomputes every number from
`runs/`. It reads nothing from the manuscript, so a disagreement is a paper bug.

```bash
python load.py       # inventory of every run with a round log
python bench.py      # CIFAR-100 Table I block, before and after MAML-Revision-2
python conv.py       # Statement 1, adaptation gain, path variation
python fairness.py   # Jain and coverage, with and without the cold start
python ablation.py   # Table II, the lambda sweeps, the inner-step ablation
```

### Which runs are actually on this machine

**Only CIFAR-100 has main-benchmark runs here.** The Fashion-MNIST and CIFAR-10
benchmark rows of Table I were produced on another machine
(`C:/Users/rickt/...`) and were never copied over. They cannot be re-derived.

| experiment | runs | verifiable |
|---|---|---|
| `cifar100_benchmarks` | 19 | yes, the whole CIFAR-100 block |
| `feature_ablation` (Fashion) | 21 | yes, all of Table II |
| `lambda_sensitivity` (Fashion) | 12 | yes |
| `cifar10_lambda_sensitivity` | 12 | yes, and **not yet in the paper** |
| `inner_step_ablation` (CIFAR-10) | 9 | yes, and **not yet in the paper** |
| `conv_*` (3 datasets) | 3 | yes, the new drift logs |
| Fashion and CIFAR-10 benchmarks | 0 | **no, absent** |

The Fashion and CIFAR-10 rows were checked for internal consistency instead.
Every derived percentage in the abstract and Section V recomputes correctly from
the table as printed, so the rows are self-consistent even though the raw runs
are gone.

## What round 2 found

### 1. Two rows of Table I reported a standard deviation from one seed

`accuracy` and `recall` are **the same quantity** in this schema. Verified on
all 19 CIFAR-100 runs, they never differ. So their standard deviations must be
identical in every row. In the round-1 manuscript they were not:

| row | acc | rec | seeds actually present |
|---|---|---|---|
| TiFL, CIFAR-100 | 32.12 ± **0.30** | 32.12 ± **0.33** | seed 42 only |
| FedCor, CIFAR-100 | 34.18 ± **0.24** | 34.18 ± **0.60** | seed 42 only |

Every other row in all three dataset blocks is self-consistent. The two
inconsistent rows are exactly the two single-seed rows. The new s2026 runs give
both a genuine second seed, so this is now fixed by data rather than by
deletion.

### 2. The three new CIFAR-100 seeds move three rows

At the matched horizon of round 150, over all runs now present:

| method | round 1 | round 2 | seeds |
|---|---|---|---|
| TiFL | 32.12 ± 0.30 | **28.52 ± 5.08** | 42, 2026 |
| FedCor | 34.18 ± 0.24 | **30.86 ± 4.70** | 42, 2026 |
| CriticalFL | 48.84 ± 1.58 | **45.37 ± 6.13** | 42, 123, 2026 |

CriticalFL TFLOPs go 9469 ± 392 to **10235 ± 1354**, energy 5438 ± 214 to
**5879 ± 779**, carbon 2583 ± 102 to **2793 ± 370**, Jain 0.97 to **0.98 ± 0.01**.
FedCor TFLOPs go 6428 to **6473 ± 64** and coverage 14 ± 0 to **14 ± 1**.

MAML-Select, FedAvg, FedGCS, FedCS and Oort are unchanged. **The direction of
every comparison in the paper is unchanged and the margins widen.**

### 3. The drift logs are the answer to Reviewer 2 comment 2

`runs/MAML-Revision-2/convergence/selector_convergence_*.jsonl`, 199 rounds each.

**The code that wrote `l_query_at_base` and `drift_increment` is not in this
repository.** `selector.py` here emits only six fields. The definition was
recovered numerically instead and matched **198 of 198 rounds on all three
datasets**, so it is certain:

```
drift_increment = L_query(phi_t) - L_sup(phi_t)
```

Both evaluated at the **same** parameter. `L_sup` is round t-1's objective and
`L_query` is round t's, so this is the change of the objective at a fixed
parameter. That is the standard non-stationarity increment.

**Statement 1 holds empirically in 198 of 198 rounds on every dataset.**

| dataset | descent <= 0 | mean descent | worst |
|---|---|---|---|
| Fashion-MNIST | 198/198 | -0.002660 | -0.221047 |
| CIFAR-10 | 198/198 | -0.009045 | -0.082856 |
| CIFAR-100 | 198/198 | -0.034039 | -0.391953 |

**Adaptation gain**, `L_query` at the base parameter minus `L_query` after the
inner step. Positive means the inner step helped that round's actual decision.

| dataset | gain > 0 | mean gain | median | V_T | V_T / T |
|---|---|---|---|---|---|
| Fashion-MNIST | 117/199 | 0.000619 | 0.000088 | 8.4204 | 0.0425 |
| CIFAR-10 | 122/199 | 0.003689 | 0.000624 | 21.3206 | 0.1077 |
| CIFAR-100 | 117/199 | 0.005633 | 0.002904 | 59.6178 | 0.3011 |

**The mean gain rises with the measured non-stationarity, in the same order on
all three datasets.** That is the empirical content of the claim that
meta-adaptation helps when the objective moves. With three datasets this is an
ordering, **not a significance result**, and must be written that way.

Adaptation helps in about 60 percent of rounds, not all of them. Say so.

### 4. Coverage survives the cold start, so the fairness claim holds

MAML-Select opens every run with `ceil(N/K) = 10` rounds of deterministic
round-robin (`_cold_start_selection` in `selector.py`). Those rounds alone touch
every client once, so a coverage number over the whole run partly measures the
warm-up. Recomputed over post-warm-up rounds only:

| dataset | cov all | cov post | Jain all | Jain post |
|---|---|---|---|---|
| CIFAR-100, 3 seeds, horizon 150 | 100.0 | **100.0** | 0.782 ± 0.031 | 0.758 ± 0.033 |
| Fashion-MNIST, 3 seeds | 100.0 | **100.0** | 0.776 ± 0.056 | 0.758 ± 0.059 |
| CIFAR-10, conv run, seed 42 | 100.0 | **100.0** | 0.616 | 0.592 |

Coverage stays at exactly 100 percent and Jain falls by about 0.02. The
coverage result is a property of the policy, not of the warm start.

My recomputation independently reproduces the Jain values printed in Table I
(0.78 CIFAR-100, 0.77 Fashion, 0.61 CIFAR-10), which is a good check on the
whole harness.

### 5. Do not use `fairness_without_cold_start` from `revision_numbers.json`

It reports Jain 0.42 for CIFAR-10 over `n_runs = 23`. That pools the
lambda-sweep runs, where lambda = 5.0 drives Jain to 0.22. It is not the main
configuration and quoting it would understate the method. Use the per-
configuration numbers above.

### 6. `COVERAGE.md` mislabels V_T

`runs/MAML-Revision-2/COVERAGE.md` prints a "V_T" column holding the **signed**
drift sum, including **-2.892** for Fashion-MNIST. A path variation is a sum of
absolute increments and cannot be negative. The correct values are 8.42, 21.32
and 59.62. Do not copy that column into the paper.

### 7. A small factual error in Section V

The manuscript says the largest negative shifts in the feature ablation occur
when **battery and loss** are removed. Recomputed, they are battery at -0.33 and
**latency** at -0.09. Loss is -0.07.

### 8. Two studies exist that the paper never mentions

- **CIFAR-10 lambda sweep**, 12 runs. Accuracy spread across lambda is
  **8.56 pp** against **1.02 pp** on Fashion-MNIST. The manuscript generalizes
  lambda insensitivity from Fashion alone. This is a real limitation and
  disclosing it is the honest move.
- **Inner-step ablation** on CIFAR-10, 9 runs. One inner step 63.84, two 63.07,
  five 62.15. This justifies the `inner_steps = 1` design choice, which the
  paper currently asserts without evidence.

## Style debt in the round-1 manuscript

Measured on `main.tex` with the equation and table bodies stripped: **22 colons,
12 semicolons, 3 prose double dashes**, 130 sentences, mean 21.2 words, 9 over
40 words. Many colons are the conventional lead-in to a display equation. Advait
wanted zero in SCOPE-FD, so the same bar applies here.

## Known gotcha

Bash heredocs collapse `\\` to `\` even when quoted with `<<'EOF'`. This has
caused `re.PatternError: bad escape` and once corrupted a manuscript. Write
Python scripts with the Write tool, or build regex backslashes with `chr(92)*2`.

## Progress log

| step | pages | slack | note |
|---|---|---|---|
| round-1 baseline copied into MAML_REVISION_2 | 6 | 0 | clean build |
