# Report Card — *MAML-Select* (R2 marked copy), targeted at IEEE TAI

**Assessment basis:** full read of `manuscript_r2_marked.tex`, checked against the current IEEE TAI *Information for Authors*. Grades reflect the standard a TAI referee would apply on a second-round submission, where the bar is "are the remaining objections fatal?" rather than "is this promising?"

**Bottom line: Major Revision.** The R2 theory work is genuinely good and would survive review. What will not survive is a cluster of internal-consistency problems in the empirical section — most seriously an energy model that contradicts the equation the paper says it implements — plus one missing ablation that goes directly to whether the method's central idea does anything.

---

## Grades

| Dimension | Grade | One-line verdict |
|---|---|---|
| Venue & category fit | **B−** | TAI has no "Letters"; length fits Regular, not Brief |
| Novelty & positioning | **B−** | MAML framing is defensible but under-defended; the decisive control is missing |
| Problem formulation | **A−** | Cohort→per-client derivation is clean and honestly bounded |
| Theoretical rigor | **A−** | Correct, modest, unusually candid about scope — strongest part of the paper |
| Experimental design | **C+** | 3 seeds, one α, one K, no naturally-federated data, baseline fidelity doubts |
| Results & claim calibration | **C** | CIFAR-100 is near-dominated by FedGCS; abstract overstates; p-values misread |
| Internal consistency | **C−** | Energy model contradicts Eq. (5); three different Jain values; undefined protocol |
| Fairness treatment | **B+** | Much improved and honest; coverage claim is now correctly framed as engineered |
| Writing & presentation | **B** | Clear, but sectioning levels and a notation sign error need fixing |
| References | **B** | Good 2026 TAI coverage; no post-2024 method is actually benchmarked |

---

## Blocking issues (fix before resubmission)

### 1. The energy model does not implement Eq. (5), and Table I proves it

Eq. (5) states `T_comp = E·n_i·C_model / f_i`, and Eq. (9) accumulates energy as `P_i·T_comp/3600`. If both held, energy per TFLOP would equal `P_i·10¹²/(3600·f_i)` — a constant that depends only on the device-tier mix, **not on the dataset or model**.

Table I says otherwise:

| Dataset (FedAvg row) | TFLOPs | Energy (Wh) | Wh per TFLOP |
|---|---|---|---|
| Fashion-MNIST | 140 | 5752 | **41.1** |
| CIFAR-10 | 8341 | 4798 | **0.575** |
| CIFAR-100 | 6331 | 3626 | **0.573** |

The two ResNet18 datasets agree to three significant figures; Fashion-MNIST is off by a factor of ~72. Per-round energy tells you why: 28.8, 24.0 and 24.2 Wh/round respectively, and 28.8/24.0 = 1.20 = 60,000/50,000 — exactly the ratio of dataset sizes. **The simulator's latency is proportional to samples, not to FLOPs.** `C_model` enters the TFLOPs accounting (Eq. 8) but not the timing model, despite the text at line 415 asserting "`C_model` matches `T_comp`, so cost and latency stay consistent."

This is not cosmetic. `T_comp` feeds `T_i,t`, which feeds `ρ_i,t`, which is half of the per-client cost `c_i,t = λρ_i,t − Δ_i,t`. So the objective the selector actually optimizes is not the objective the paper writes down. Either correct Eq. (5) to the samples-based form (and state that `f_i` is in samples/s, which is defensible), or re-run with a FLOP-based timing model. A referee who divides two columns will find this, and it reads as a modelling error until explained.

**Related:** because both TFLOPs and energy reduce to Σn_i over selected clients within a dataset, they are near-collinear. Reporting both as separate wins is close to double-counting one quantity. Keep both, but say plainly that energy adds only the tier-power weighting.

### 2. The compute saving may be an artifact of preferring small-data clients

Following from the above: `TFLOPs ∝ Σ n_i` and `ρ_i,t` increases with `n_i`. A selector that minimizes `λρ − Δ` therefore has a structural bias toward clients with fewer samples, which mechanically lowers TFLOPs *and* lowers accuracy. That is a complete alternative explanation for every headline number, and the paper never rules it out.

**Required:** report the mean (or distribution of) `n_i` among selected clients for MAML-Select versus FedAvg, per dataset. If the selector is *not* just picking small shards, this is a one-figure rebuttal and it strengthens the paper considerably. If it is, the contribution needs reframing — "a principled way to trade sample budget for accuracy" is still publishable, but it is a different claim.

### 3. No ε=0 / no-adaptation control

The ablation sweeps inner steps over {1, 2, 5} but never **0**. The single most obvious referee question is: *does the MAML inner step do anything at all, or would the same 6-64-64-1 MLP trained by plain online regression on the previous round's feedback perform identically?* Since `D_sup(t) = D_query(t−1)`, that baseline is nearly the same computation, and the paper's own defence (line 230, "this lag is what makes it meta-learning rather than online regression") is an argument where a measurement is needed.

Add a row: **0 inner steps (online regression on `D_query`)**, on at least CIFAR-10 and CIFAR-100, three seeds. Without it, "meta-learning" reads as framing rather than mechanism, and that is the difference between a B− and a B+ novelty grade.

### 4. Ablation numbers are not comparable to Table I, and the protocol is undefined

- Table II reports CIFAR-10 accuracy of **63.8%** for the default configuration; Table I reports **75.63%** for the same configuration.
- The λ sweep spans 64.9% (λ=0.1) to 56.3% (λ=5), so the *default* λ=0.5 sits near 63.8% — internally consistent with Table II but not with Table I.
- The only hint is the phrase "CIFAR-10 **diagnostic** accuracy" at line 425, and "diagnostic" is never defined.

Define the reduced protocol explicitly (rounds, seeds, any other change), state it in the caption of Table II and the λ figure, and add one sentence in the text saying the sensitivity runs are shortened and therefore not comparable to Table I. As written this looks like a discrepancy rather than a design choice.

Same class of problem, smaller: the Fashion-MNIST full-state row of Table II (90.23 ± 0.55, Jain 0.78) does not match the Table I MAML-Select row (90.11 ± 0.47, Jain 0.77), and §V-C reports Fashion-MNIST Jain as 0.755 → 0.737. Three different values for what should be one configuration. Reconcile or explain.

### 5. CIFAR-100: FedGCS dominates on almost every axis

| CIFAR-100 | Acc | F1 | TFLOPs | Energy | Jain |
|---|---|---|---|---|---|
| FedGCS | **58.66** | **58.08** | **6140** | 3526 | **0.82** |
| MAML-Select | 58.15 | 57.66 | 6224 | **3457** | 0.78 |

FedGCS wins on accuracy, F1, TFLOPs and fairness; MAML-Select wins only on modelled energy, by 2%. Given issue #1, that single win rests entirely on the tier-power weighting. Meanwhile the abstract claims "significantly lower cumulative TFLOPs and modelled energy" on all three datasets, when CIFAR-100 delivers a 1.7% TFLOPs reduction against FedAvg and a *loss* against FedGCS.

Either explain why CIFAR-100 behaves differently (a real explanation — 100 classes over ~500 samples/client leaves little utility signal for `Δ_i,t` to separate on, which is a plausible and interesting one), or restate the claim per-dataset. Referees are tolerant of a method that wins on two of three benchmarks and says so. They are not tolerant of an abstract that flattens the third.

---

## Substantive issues (expect a referee to raise these)

### 6. Baseline fidelity

Oort at 10% coverage and Jain 0.10, and TiFL at 12%/0.11, mean each selected exactly 10–12 distinct clients out of 100 across 200 rounds. Oort has an explicit exploration–exploitation mechanism; TiFL samples tiers probabilistically. Neither should collapse onto a fixed cohort. This will read as a degenerate reimplementation that inflates the proposed method's standing. Either document the implementation (exploration factor, pacer settings, tier-sampling probabilities) or fix it. Given TAI's guidance to compare against "the top two or three most recent and competitive algorithms," you would lose nothing by cutting to FedAvg / FedGCS / CriticalFL / FedCor and doing those four properly.

Also: the related work cites 2026 TAI papers, but every benchmarked baseline is 2024 or earlier. Expect "why is nothing from 2025–2026 in Table I?"

### 7. Statistics

- Paired *t*-tests on **n = 3** are very fragile; normality is untestable at that size.
- More importantly, the equivalence claims are inverted: "indistinguishable on CIFAR-10 (p = 0.10)" and "equivalent on Fashion-MNIST (p = 0.84)" treat failure to reject as evidence for the null. At n = 3 the test has almost no power, so a large p-value is uninformative. Use TOST with a stated equivalence margin, or report the difference with a confidence interval and say only that no difference was detected.
- Five to ten seeds would cost little at this scale and would remove the objection entirely.

### 8. The latency outcome is never reported

The objective in Eq. (6) explicitly penalizes `T_round`, and the method's selling point is straggler-aware selection — yet no table or figure reports achieved round latency, wall-clock time, or time-to-target-accuracy. This is the metric most client-selection referees look for first. Add mean `T_round` and time-to-X% accuracy to Table I, even as a supplementary column.

### 9. Coverage as a headline result

§V-B and Proposition 2 are now correctly honest: coverage is *guaranteed by construction* via the exploration slot, independent of the learned policy. That is the right framing. But the abstract still advertises "retains full participation coverage on every dataset" as a finding. It is a design guarantee, not an empirical result, and one that any baseline could adopt in three lines of code. Either drop it from the abstract or state it as "by construction."

Meanwhile the Jain index (0.61 on CIFAR-10 against 0.90–0.98 for FedAvg/FedGCS/CriticalFL) is the fairness number that *is* a result, and it is the weakest column in the table. §V-A and §V-B handle this well; consider moving that framing earlier so the reader meets the explanation before the number.

### 10. Scope of the empirical study

One α (0.5), one K (10), one N (100) for all accuracy results; three vision datasets, none naturally federated. The conclusion promises heterogeneity and pool-size sweeps as future work — but α ∈ {0.1, 0.5, 1.0} is the standard robustness check for any non-IID selection paper and its absence is a predictable referee request. At minimum add α = 0.1 on one dataset. FEMNIST or Shakespeare would substantially raise the paper's credibility for edge deployment claims.

---

## Correctness notes

**A sign error in the exploration rule, appearing twice.** Line 309 and Algorithm 1 line 379 both order the exploration slot by "largest in `(−τ_i,t, ν_i,t)`". Largest `−τ` is *smallest* staleness, i.e. the least stale client — the opposite of the stated intent ("the stalest and least-used"), and the opposite of what Proposition 2's proof assumes. It should be largest in `(τ_i,t, −ν_i,t)`, or smallest in `(−τ_i,t, ν_i,t)`. Fix in both places.

**Proposition 2 statement vs. proof.** The statement says "any window of N consecutive rounds"; the proof establishes N−1. Harmless (the statement is weaker) but tidy it up. The proof also does not account for clients whose staleness resets because they were chosen by the Top-(K−ε) ranking rather than the slot — that only helps, but a sentence saying so would close the gap.

**Everything else checks.** Lemma 1 (descent lemma application), Lemma 2 (completing the square, with `δ_t ≤ L_q β‖∇g_t‖`), Corollary 1 (telescoping with the drift term at a common iterate), and Proposition 1 (exchange argument for top-K) are all correct as written. Remark 1 is the best paragraph in the paper — explicitly disclaiming convergence to a stationary point of a fixed objective, and naming the Adam gap, is exactly the honesty that gets a theory section through review. Keep it verbatim.

**Two small factual slips.** "One ResNet18 round on CIFAR-10 is about 4.13 TFLOPs" should be "one client-round" (Table I implies ~41.7 TFLOPs per round across K = 10). And §V-C refers to "all 77 MAML-Select runs" without ever establishing where 77 comes from — add the accounting.

---

## Venue and formatting

**There is no "Letters" category at IEEE TAI.** The three types are Original Research **Regular** Manuscripts (10 pages normal, 15 max), Original Research **Review** Manuscripts, and Original Research **Briefs** Manuscripts (6 normal, 9 max). Category cannot be changed after submission, so this decision matters.

At an estimated 8.5–9.5 typeset pages (two full-width `table*`s, three figures, an algorithm, four theorem-environment blocks, 22 references), the manuscript **fits Regular comfortably and overruns Brief** — as a Brief it would incur $200/page over six pages, or require cutting roughly a third. Submit as a Regular Manuscript.

Other checks:
- Title 12 words (limit 15) ✓. Abstract 175 words (limit 250) ✓. Impact Statement 129 words (100–150) ✓. Index terms: 4 (min 3, max 6) ✓ — but delete the trailing "and".
- Double-anonymous review: the author block is correctly commented out. Before submission also confirm no identifying content in figure files, and note that code/data links must wait for camera-ready.
- **Sectioning levels are wrong.** Related Work and the Methodology section use `\subsubsection` directly under `\section`, skipping `\subsection`. IEEEtran will number these as `a)`, `b)` at the wrong depth. Promote them to `\subsection`.
- `\begin{table}[H]` (Table II) forces placement in a two-column layout and frequently breaks pagination; use `[!t]`.
- Abstract currently carries no numbers. TAI's own abstract guideline asks for one to two sentences summarizing the main result. Restoring per-dataset figures — with the CIFAR-100 caveat — would be both more compliant and more honest than "significantly lower."
- Grammar, line 120: "the classification of existing client selection methods ... are presented" → "is presented". Line 122: "learns the selection policy based on six parameters" → "six features" (parameters means something else two paragraphs later).
- Clarify the tier convention: Tier 1 at 1.0× is the *slowest* and Tier 3 at 4.0× the *fastest*, which is the reverse of most readers' expectation and matters for reading §V-B.

---

## What is working well

Worth saying plainly, because it should not be lost in revision:

1. **The R2 theory is the right kind of theory.** Two descent lemmas, an exact combinatorial step, a tracking corollary, and a remark that refuses to overclaim. Nothing is inflated. The explicit statement that `δ_t` is *not* the term dropped by the first-order approximation is the sort of precision reviewers rarely see and always reward.
2. **The cohort→per-client derivation** (Eq. 7 → Eq. 8, with exactly one flagged approximation) turns what was probably a heuristic score into a justified one. This addresses the most common objection to learned selectors.
3. **Measuring `V_T` empirically** to check the regime its own corollary assumes is well above the norm for this venue.
4. **§V-C** (coverage without the cold start) anticipates the artifact objection and kills it with both a proof and a re-measurement. Do more of this.
5. The z-scoring justification in Eq. (4) — dimensionless input, relative ranking, which is all top-K needs — is a small point handled exactly right.

---

## Prioritized action list

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Resolve the energy/latency model contradiction (Eq. 5 vs Table I) | Medium | Blocking |
| 2 | Add the 0-inner-step (online regression) ablation | Low | Blocking |
| 3 | Report selected-client sample counts to rule out the small-shard explanation | Low | Blocking |
| 4 | Define the "diagnostic" protocol; reconcile Table II / §V-C / Table I numbers | Low | Blocking |
| 5 | Restate the abstract's efficiency claim per-dataset; acknowledge CIFAR-100 vs FedGCS | Low | Blocking |
| 6 | Fix the `(−τ, ν)` sign error in both places | Trivial | High |
| 7 | Replace equivalence-from-large-p with TOST or CIs; raise to 5+ seeds | Low–Med | High |
| 8 | Document or repair Oort/TiFL implementations; consider trimming to 4 baselines | Medium | High |
| 9 | Add achieved round latency / time-to-accuracy | Low | High |
| 10 | Add α = 0.1 on one dataset | Medium | Medium |
| 11 | Fix `\subsubsection` → `\subsection`, `[H]` → `[!t]`, index-term "and", grammar | Trivial | Medium |
| 12 | Confirm submission category: Original Research Regular Manuscript | Trivial | Medium |

**Realistic outcome as submitted:** major revision, with items 1, 3 and 5 the likeliest grounds for a hostile second reviewer. **With items 1–9 addressed:** a solid accept-with-minor-revision candidate. The theoretical contribution is already at the required standard; the empirical section has not yet caught up to it.
