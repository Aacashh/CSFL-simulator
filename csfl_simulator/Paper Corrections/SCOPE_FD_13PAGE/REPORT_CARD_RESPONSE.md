# Response to the SCOPE-FD report card

Written 15 August 2026 against `scope-fd-report-card.md`. Every number was
re-derived from `runs/runs_scope_revised` by
`Paper Corrections/scope_fd_data_verification/reportcard.py`. **No new run was
executed.** Everything below is either a correction to something already wrong,
or a quantity the campaign already contained and the paper did not report.

State after this pass: manuscript **13 pages** clean and marked, reply letter 13
pages, 0 errors, 0 overfull boxes, 0 undefined references, 33 references,
abstract 247 words against the 250 limit, impact statement 140 words inside the
100 to 150 range.

---

## Corrections, things that were wrong

### M3, Section VI-K contradicted Section V-A. **Fixed, manuscript and letter.**

The paper said the guarantee "strengthens rather than weakens as the pool
grows", citing $0.00\%$ at $N{=}200, K{=}10$ against $1.33\%$ at the headline.
Equation (17) is arithmetic, so this is checkable exactly:

| $N$, $K$ | $R=97$ | $R=98$ | $R=99$ | $R=100$ | $R=101$ |
|---|---|---|---|---|---|
| 30, 5 | 0.86% | 1.36% | 1.52% | **1.33%** | 0.83% |
| 200, 10 | **2.63%** | 1.84% | 0.96% | **0.00%** | 0.94% |
| 200, 20 | 2.16% | 1.63% | 0.91% | **0.00%** | 0.89% |
| 100, 5 | 2.63% | 1.84% | 0.96% | **0.00%** | 0.94% |

The zeros are horizon alignment, since $KR$ is an exact multiple of $N$ at
$R = 100$. Move one round either side and the large pools are the *worst* rows.
Section V-A already said the decay is not smooth, so the paper contradicted
itself. Over $R \in [25,120]$ the mean is 1.63% at $(30,5)$ and **5.77%** at
$(200,10)$, so if anything the large pool is worse on average.

The letter repeated the same claim to Reviewer 1 and is corrected too.

### m4, Corollary 1 dropped a term. **Fixed.**

$\Gamma_R$ omitted the $+1$ of the $t+1+\beta_1$ denominator of (18). Restored,
which is both correct and tighter.

### m5, Table II implied five seeds agreed. **Fixed.**

Rounds-to-60% was printed as an exact 13.0 and 14.0. The seeds did not agree:

| | rounds to 60% | mean ± sd |
|---|---|---|
| complete score | 11, 11, 11, 16, 16 | 13.0 ± 2.7 |
| debt only | 11, 11, 16, 16, 16 | 14.0 ± 2.7 |
| uniform random | 16, 16, 21, 21, 21 | 19.0 ± 2.7 |
| DivFL | 11, 16, 16, 16, 21 | 16.0 ± 3.5 |
| SubTrunc | 11, 11, 16, 21, 21 | 16.0 ± 5.0 |

Accuracy is evaluated every five rounds, so the metric can only take 11, 16 or
21. The column now carries its dispersion and the caption says why it is
quantized.

### m1, three headline numbers for one configuration. **Fixed.**

Not three configurations, one seed set. Every five-seed family returns
$71.21 \pm 0.99$ exactly, and restricting those same runs to seeds
$\{11,22,33\}$ returns $71.75 \pm 0.81$ exactly, which is the second value. The
privacy family's $71.99 \pm 0.63$ sits 0.24 points from it as run-to-run
variation. The five-seed value is now stated as canonical.

### M4, a p-value on non-independent configurations. **Fixed, manuscript and letter.**

The rank correlations over fifteen configurations carried $p = 0.0002$ and
$p = 0.0009$. The paper itself notes those configurations share a seed set. Both
p-values are dropped and the caveat is attached to both correlations.

### Section IV-D contradicted Section VI-H. **Fixed.**

Section IV-D said a quantitative characterization of the privacy remedies "is
left to future investigation". Section VI-H evaluates both. The stale sentence
is gone.

### Smaller ones

m2 off-by-one in (9), stated as cancelling under min-max normalization · m3
degenerate case of (10) in the body · m6 the 1.33 percentage-point span that
collided numerically with the 1.33% Gini · m9 what the privacy study does not
establish · m10 the uncited regulatory-duty claim, softened · m11 $E_{\min}$
tied to the constant schedule · m12 "count form" to "counting argument" · m16
the 459-of-459 check framed as an identity check rather than a hypothesis test.

---

## Additions, all from data already collected

### M2, UnionFL matched on every column. **Answered with evidence.**

At the headline it returns 71.18 ± 1.40 accuracy, **1.33%** Gini and **13.0**
rounds against SCOPE-FD's 71.21 ± 0.99, 1.33% and 13.0. A tie on all three. The
paper's claim that "the separation is in participation fairness alone" was false
with respect to UnionFL.

The cohort sweep separates them decisively:

| $K$ | law (17) | SCOPE-FD | UnionFL | UnionFL seed sd |
|---|---|---|---|---|
| 1 | 6.67% | 6.67% | **55.56%** | 0.37 |
| 3 | 0.00% | 0.00% | **29.44%** | 0.98 |
| 5 | 1.33% | 1.33% | 1.33% | 0.00 |
| 10 | 0.67% | 0.67% | **22.89%** | 3.11 |

SCOPE-FD returns the law's value with a seed standard deviation of zero at every
configuration in the campaign. UnionFL agrees at one and diverges by up to 29
points elsewhere. Across all its configurations UnionFL's Gini spread is 20.75
against 0.96 for SCOPE-FD.

### M5, participation fairness connected to outcome fairness. **Answered.**

The report card called this the highest-value missing item. `client_accuracy_std`
was already logged, so it needed no run. Matched seeds, $N=30$:

| $K$ | uniform random | SCOPE-FD | debt only |
|---|---|---|---|
| 1 | 13.94 ± 1.24 | **9.11 ± 2.00** | 8.95 ± 1.28 |
| 3 | 7.93 | 7.47 | 7.77 |
| 5 | 6.63 | 6.54 | 6.71 |
| 10 | 6.03 | 5.99 | 6.17 |

At $K=1$ the spread of accuracy across clients falls by about a third, in the
same direction on all three seeds (differences 1.65, 6.13, 6.71). The effect is
confined to the sparse regime and tracks the participation imbalance it removes,
which is the empirical counterpart of Corollary 1. Added to Section VI-E and to
Reply 2 of Reviewer 4, who asked for broader fairness evidence.

### M8, the Oort collapse given a mechanism. **Answered.**

A Gini of 83.33% over $N=30$ with a seed standard deviation of **zero** is
exactly what appears when the same five clients take every slot of every round.
The per-client accuracy standard deviation is 25.78 against 6.71 for uniform
random. The row is now read as a utility-only rule ported without a coverage
mechanism rather than as a tuned comparison against Oort as published.

### M6, the gap between what is proved and what is measured. **Stated.**

Two sentences at the end of Section V-B. (18) measures each local loss against
its own optimum on a separable objective, so the corollaries do not separate
federated from isolated local training, and the server-model accuracy reported
throughout Section VI is not a quantity they bound. Reviewer 2 comment 3 and
Reviewer 4 comment 1 both asked for this care.

### M1, the rotation stated as the design. **Done.**

Proposition 1 preserves the debt ordering for *every* admissible coefficient
pair, not only at the origin, so the complete score is a rotation too. Section
IV-A now says the contribution is not the ordering, which round-robin also
gives, but the closed-form guarantee, the composition with the substrate bound,
and carrying the information terms without disturbing either. Reviewer 2 comment
1 raised exactly this.

### m7 and m8, the privacy claim made precise.

$\bm{h}_i$ is L1-normalized, so replacing one client's dataset moves it by at
most 2 in L1. Laplace noise of scale $2/\varepsilon_{\mathrm{dp}}$ per entry with
renormalization gives that client $(\varepsilon_{\mathrm{dp}}, 0)$-differential
privacy, which is now stated. The surrogate's bootstrap rule is stated, and so is
the fact that it narrows the disclosure rather than ending it.

---

## Not done, and why

**M7, the public set overlapping the evaluation split.** Removing the public
inputs and re-running the headline is a run. The disjoint-corpus control already
exists at 67.10 ± 1.05 and Section VI-A already discloses the overlap.

**Item 5, CIFAR-10.** `runs/runs_scope_revised/cifar10_multiseed` exists but
holds **two seeds** and reaches 28.15% for uniform random against 28.54% for
SCOPE-FD, with the Gini at 1.33% against 11.65%. Reporting a two-seed result to
reviewers who complained about single-seed results would cost more than it buys.
The fairness ordering it shows is the same as everywhere else.

**Item 6, a non-FedTSKD substrate.** A run.

**Retitling away from massive MIMO, item 7.** The manuscript is under review as
submitted. That is the author's call, not a correction.

**m13**, renaming $N_{BS}$ to $M$ touches every equation in Section III for a
readability gain. **m14**, moving the Figure 1 caption content into the body
costs space the 13-page budget does not have.

---

## Space

The additions and corrections cost about 890 pt of column height against 15 pt
of slack. It was recovered by compressing the new text, pulling stray last lines
out of roughly thirty paragraphs, tightening both table row heights, and setting
the bibliography at `\scriptsize`. IEEEtran fixes the bibliography font inside
`thebibliography`, so an outer size declaration does nothing and the environment
itself had to be patched. No result, figure, table or reference was removed.
