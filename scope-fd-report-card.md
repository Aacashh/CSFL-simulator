# Report Card — *SCOPE-FD: A Client Selection Method in Federated Distillation for Massive-MIMO Systems*

**Target venue:** IEEE Transactions on Artificial Intelligence
**Assessment type:** Simulated referee report / pre-submission triage
**Overall recommendation:** **Major Revision** (currently borderline; revisable to Accept)
**Reviewer confidence:** High on methodology and internal consistency; medium on venue-specific novelty thresholds

---

## 1. Scorecard

| Criterion | Grade | /10 | Note |
|---|---|---|---|
| Novelty & originality | C+ | 5.0 | Core mechanism is debt-ordered round-robin; the *closed-form* Gini is the real novelty, not the selector |
| Theoretical contribution | C+ | 5.5 | Prop. 1 is correct but near-trivial; Cor. 1–2 are arithmetic composition of an existing bound |
| Technical soundness | B | 7.5 | Proofs check out; one empirical claim (§VI-K) is not supported by the paper's own formula |
| Experimental design | B+ | 8.0 | Unusually thorough campaign, honest negatives, good ablation and controls |
| Statistical rigour | B− | 6.5 | 3–5 seeds, self-admitted inability to reach significance, three different headline numbers |
| Reproducibility | B | 7.0 | Setup table is good; no code/seed/artifact statement |
| Related-work coverage | B+ | 8.0 | Comprehensive and current; but UnionFL's near-identical result is under-confronted |
| Clarity & writing | A− | 9.0 | Genuinely excellent prose; among the clearest FL/FD manuscripts I have read |
| Significance / impact | C+ | 5.5 | Fairness gain is largely definitional; no accuracy gain established at the headline setting |
| Fit for IEEE TAI | B− | 6.5 | Plausible fit, but the mMIMO framing points toward TCCN/TWC |
| **Composite** | **B−** | **6.9** | **Major Revision** |

---

## 2. What the paper does

The manuscript adds partial-participation client selection to FedTSKD (ref. [12], the authors' own prior work) for federated distillation over an mMIMO backbone. The selector scores each client by a participation-debt term plus a server under-prediction bonus minus a class-coverage penalty, with coefficients small enough that the debt term strictly dominates. The central claim is that this makes the cumulative participation Gini coefficient exactly $m(N-m)/(NKR)$ with $m = KR \bmod N$ — computable before deployment rather than measured after — while accuracy is unchanged relative to uniform random selection.

---

## 3. Strengths worth protecting in revision

1. **The exact-Gini result (17) is the paper's genuine contribution.** Moving from a bound to a closed form, and identifying $N \mid KR$ as the exact vanishing condition, is a clean and useful result. I verified (17) numerically against every value quoted in the text — 1.33% at (30, 5, 100), 3.33% at (30, 5, 25), 1.40% at (47, 6, 100), 1.25% at (53, 7, 100), 0.00% at (50, 3, 100) and (200, 10, 100) — and all are correct. The bound $N/(4KR)$ is also respected in every case.
2. **Intellectual honesty is high and rare.** §VI-J reports that the channel-aware variant is *worse on both axes* and diagnoses why. §VI-K reports that the complete score *loses* to its own ablation at $K \geq 20$. §VI-A discloses the public-set/evaluation-split overlap unprompted. §VI-A concedes that five seeds cannot produce $p < 0.0625$. Reviewers reward this; do not sand it down under revision pressure.
3. **The reporting protocol paragraph in §VI-A** (explaining why the headline number differs across families) is the right instinct, even if the execution needs fixing (see m1).
4. **Writing quality.** The prose is compressed, unhedged and precise. This should be preserved verbatim wherever possible.
5. **The negative-result sections are the most interesting content in the paper** and are currently buried at §VI-J and §VI-K. Consider promoting them.

---

## 4. Major issues

### M1 — The selector reduces to round-robin, and the paper does not confront this
Under the paper's own Proposition 1, whenever $\alpha_u + \alpha_d < 1$ the debt ordering is strictly preserved, so SCOPE-FD selects the $K$ clients with the smallest cumulative counts, breaking ties by bonus/penalty. That is deterministic round-robin with a tie-break rule. Proposition 1 then proves what round-robin gives by construction. A reviewer will ask directly: *what is the contribution beyond round-robin?*

**Action.** Add plain round-robin (smallest count, ties by index) as an explicit baseline in Table II. State up front — ideally in §IV-A — that the debt term *is* a rotation and that the contribution is (i) the closed-form fairness law, (ii) the composition with the FedTSKD convergence bound, and (iii) the demonstration that information terms can be layered on without destroying the guarantee. Framing this as a feature ("the guarantee holds because the mechanism is simple") is far stronger than leaving it for a reviewer to find.

### M2 — UnionFL matches SCOPE-FD on both axes in the paper's own headline table
Table II: UnionFL 71.18 ± 1.40 / 1.33% / 13.0 rounds against SCOPE-FD 71.21 ± 0.99 / 1.33% / 13.0 rounds. That is a statistical tie on accuracy, an *exact* tie on fairness, and a tie on convergence speed, against a published 2024 method. As written, the headline table shows the proposed method being matched by prior art, and the text disposes of this in one clause.

**Action.** This needs a dedicated paragraph, not a clause. The defensible differentiator is predictability: SCOPE-FD's Gini is a closed form with zero seed variance, whereas UnionFL's diversity objective admits no such statement. Demonstrate it — report UnionFL's Gini standard deviation across seeds and across the fifteen $(N, K)$ configurations of Fig. 3(c) and show it fluctuates where SCOPE-FD's does not. If UnionFL also returns exactly 1.33% at every configuration, the differentiation claim is in serious trouble and you need to know that before a referee does.

### M3 — §VI-K's headline claim is contradicted by the paper's own formula
§VI-K states: *"the guarantee strengthens rather than weakens as the pool grows,"* citing 0.00% at $N = 200, K = 10$ against 1.33% at $N = 30, K = 5$. This is an artifact of $R = 100$ happening to align with those pools, not a pool-size effect. From (17), at $R = 97$:

| Configuration | $R = 100$ | $R = 99$ | $R = 97$ |
|---|---|---|---|
| $N=30, K=5$ | 1.33% | 1.52% | 0.86% |
| $N=200, K=10$ | **0.00%** | 0.96% | **2.63%** |
| $N=200, K=20$ | **0.00%** | 0.91% | **2.17%** |
| $N=100, K=5$ | **0.00%** | 0.96% | **2.63%** |

At $R = 97$ the large-pool cases are *worse* than the headline. The paper already says this in §V-A (*"the decay in R is not smooth… touching zero whenever the horizon completes a whole number of rotations"*), so §VI-K contradicts §V-A.

**Action.** Rewrite the first result of §VI-K. The correct statement is that the *law* holds at every pool size, and that large pools happened to be horizon-aligned at $R = 100$. Better: sweep $R$ over a non-aligned range for the large-pool cases and show the measured oscillation tracking (17). That turns a flawed claim into a stronger validation of the formula.

### M4 — No accuracy benefit is established, and the theory does not predict one
At the headline setting the result is a tie (71.21 ± 0.99 vs 70.99 ± 1.38). The sparse-regime gains ($K=1$: +1.94 pp with SDs of 1.68/2.28 over three seeds; $K=3$: +1.01 pp) are inside noise. The Spearman correlation of −0.815 ($p = 0.0002$) is computed over fifteen configurations that the paper itself notes share a seed set and are therefore not independent, so that $p$-value is not interpretable as stated. Corollaries 1–2 give $O(N/(KR))$ for both SCOPE-FD and uniform random, so the theory explicitly predicts no order improvement.

**Action.** Either (a) commit fully to the fairness-first framing — remove "leads it where participation is sparsest" from the abstract and the Impact Statement, and present accuracy as *preserved*, which is a clean and defensible claim; or (b) invest in enough seeds (10–15) at $K \in \{1, 3\}$ to make the sparse-regime gain statistically supportable. Option (a) is cheaper and I would recommend it. Also recompute the rank correlation with a permutation test that respects the shared-seed structure, or drop the $p$-value and report the trend descriptively.

### M5 — Participation fairness is never connected to outcome fairness
The Impact Statement promises *"fair treatment of the people they serve"* and invokes regulatory duty. The paper then measures only a Gini coefficient over selection counts. A referee — especially at an AI venue rather than a comms venue — will ask whether balanced participation actually improves anything a user experiences.

**Action.** This is the single highest-value missing experiment and it costs no new runs, only new logging. Report **per-client accuracy dispersion**: worst-client accuracy, 10th-percentile client accuracy, and the standard deviation of accuracy across the $N$ clients, for SCOPE-FD vs uniform random. If balanced participation lifts the worst-served client — which it plausibly does at $K = 1$ and $\alpha = 0.05$ — that is a genuine result and it converts the fairness claim from procedural to substantive. It also aligns the experiments with Corollary 1, which is a *per-client* statement that currently has no empirical counterpart.

### M6 — The convergence corollaries are weaker than they appear
Three problems compound:
- (18) bounds $\mathbb{E}[L_n(w_n)] - L_n^*$, the *local* loss against the *local* optimum. A client that ignores distillation entirely and runs pure local SGD would satisfy this bound faster. The bound therefore does not distinguish federated training from isolated training.
- Since $F^* = \sum_n \lambda_n L_n^*$ and (2) is a sum of separable per-client minimizations, Corollary 2 inherits the same problem at the global level.
- All reported accuracy is server-model or aggregate accuracy, which neither corollary bounds.

The composition itself is sound and the honesty of §V-B ("a consistency check rather than a new result") is appreciated — but the gap between what is proved and what is measured needs stating.

**Action.** Add two or three sentences at the end of §V-B acknowledging that (18)–(20) inherit the separable objective of [12] and bound local convergence rather than server-model generalization, and that the empirical accuracy is not covered by them. Reviewers penalize unacknowledged gaps far more than acknowledged ones. Do not attempt to overclaim here.

### M7 — Public set drawn from the evaluation split
§VI-A: the 2000 public inputs are drawn from the held-out split on which accuracy is measured and are not removed from it. The defence (unlabeled, identical across selectors, disjoint control in §VI-G) is reasonable for *comparative* claims but the headline 71.21% is transductively inflated — the MNIST-public control at 67.10% suggests by roughly 4 pp.

**Action.** Either remove the public inputs from the evaluation split and re-run the headline (preferred, and probably a modest change), or promote the disjoint-corpus cell of §VI-G to a headline-adjacent position and state clearly that absolute numbers under the overlapping setting are not comparable to the literature. The current placement — a defence buried mid-paragraph in the setup section — reads as something the authors hoped would go unnoticed, which is unfair to a paper that is elsewhere so forthcoming.

### M8 — The Oort result is a strawman as presented
21.14 ± 0.60 against a uniform-random baseline of 70.99 is a ~50 pp collapse. The mechanistic explanation is plausible, but a result this extreme is more commonly an implementation or porting defect than a scientific finding, and a referee will suspect exactly that.

**Action.** Either strengthen the evidence (report Oort's realized per-client selection distribution, show the effect of its exploration parameter, and demonstrate that the collapse persists under a tuned exploration factor), or soften the claim to "a utility-only rule ported without a coverage mechanism degrades severely in FD" and add a *fairness-constrained* Oort variant as the fair comparison. As it stands this row invites a reviewer to doubt the whole baseline suite, including the DivFL and SubTrunc numbers which are otherwise credible.

---

## 5. Minor issues and technical corrections

| # | Location | Issue |
|---|---|---|
| m1 | §VI-A | Three headline values (71.21 / 71.75 / 71.99) for the same configuration is a red flag regardless of the explanation. Pool all seeds and report one canonical number with per-family values in a footnote or appendix table. |
| m2 | Eq. (9) | Target is $r \cdot K/N$ but $n_i^{(r-1)}$ covers $r-1$ rounds; $(r-1)K/N$ is the consistent choice. The offset cancels under min–max normalization so nothing breaks, but state this explicitly or fix it — a referee will read it as an off-by-one. |
| m3 | Eq. (10) | The degenerate case $\max_j d_j = \min_j d_j$ (all counts equal) gives $0/\varepsilon = 0$ for all clients. The proof handles it; the text of §IV-B does not. Add one clause. |
| m4 | Cor. 1 | $\Gamma_R$ silently drops the $+1$ from the $t+1+\beta_1$ denominator of (18). The result stays valid (the bound is loosened) but write it as an inequality or restore the term. |
| m5 | Table II | Rounds-to-60% are reported as exactly 14.0 and 13.0 with no dispersion, implying zero variance across five seeds. Report the standard deviation or state that all seeds agreed. |
| m6 | §VI-C | "the entire grid spans 1.33 percentage points" collides numerically with the 1.33% Gini value used throughout. Report one to two decimals differently or reword to avoid the coincidence confusing readers. |
| m7 | §IV-D | The Laplace mechanism is invoked without stating the $\ell_1$ sensitivity of the normalized histogram or the resulting formal $(\varepsilon, 0)$-DP claim. For an AI venue this must be made precise; "calibrated Laplace mechanism" is not a guarantee. |
| m8 | §IV-D / §VI-H | The server-side surrogate requires each client to have been selected at least once, so it is undefined for round 1 and for any never-selected client. State the bootstrap rule. Also note the surrogate is derived from client-submitted logits and is therefore not disclosure-free in the strict sense. |
| m9 | §VI-H | The privacy result (no cost at any $\varepsilon_{dp}$) is *predicted* by the ablation, which is a nice argument, but it also means the experiment is uninformative about the mechanism's utility. Say so. |
| m10 | Impact Statement | The regulatory-duty claim ("growing legal duty to evidence fair treatment") carries no citation. Cite the EU AI Act or equivalent, or soften. |
| m11 | Cor. 1 | $E_{\min} = \min_{r \le R} E_r$ is defined, but §III-A states $E_r$ is constant in all reported experiments. Either exercise a varying schedule or simplify the notation. |
| m12 | §V-B | "the count form behind (17)" — likely intended as "the counting argument behind (17)". |
| m13 | §III-A | $N$ (pool), $N_{BS}$ (antennas) and $N_D$ (streams) are visually close. Consider $M$ for the antenna count. |
| m14 | Fig. 1 | The caption carries substantial load-bearing content (that the cloud workstation is wired to the BS, so only BS-to-client is over the air). Move that into §III-A body text; referees skim captions. |
| m15 | Refs | Four 2026 TAI references ([2], [16], [26], [28]) suggests venue-targeted citing. [28] and [26] are genuinely relevant; check that [2] and [16] are load-bearing rather than decorative. |
| m16 | §VI-M | "459 of 459" is a strong claim. State explicitly that this is a check of an analytic identity against a deterministic procedure, not an empirical hypothesis test — otherwise it reads as inflated. |

---

## 6. Missing experiments, by return on effort

1. **Per-client accuracy dispersion** (M5). No new runs, only logging. Highest value in the paper.
2. **Plain round-robin baseline** (M1). One extra row in Table II. Pre-empts the most likely rejection argument.
3. **UnionFL seed and configuration variance** (M2). Reuses existing runs. Establishes the differentiator.
4. **Non-aligned horizon sweep** for large pools (M3). Cheap; converts a flawed claim into a validated law.
5. **CIFAR-10 under Dirichlet.** FMNIST/MNIST/FSDD with small CNNs is below the current expectation at TAI. One CIFAR-10 column would substantially raise the perceived weight of the empirical work. FSDD at 46% accuracy with ~90 samples per client does not carry much load.
6. **SCOPE-FD on a non-FedTSKD substrate** (e.g. DS-FL [6] without the mMIMO layer). Demonstrates the selector is a contribution in its own right rather than an extension of the group's prior paper. This directly addresses the incrementality concern and is probably the strongest single move available.

---

## 7. Fit for IEEE TAI specifically

**In favour:** the Impact Statement is present and reasonably well-written; fairness, participation equity and auditability are squarely AI-journal concerns; the reference list engages recent TAI work; federated learning is well within scope.

**Against:** the mMIMO apparatus is largely decorative. §VI-J concedes the score reads no channel or energy quantity; §VI-F concedes the guarantee is decoupled from the wireless layer; the channel model is *disabled by default* in most experiments. What the mMIMO substrate actually contributes is the constraint $K \le N_{BS}$ and a shared lineage with [12], which appeared in IEEE TCCN. A TAI referee may reasonably say the paper belongs at TCCN or TWC.

**Recommendation on framing.** Lead with the AI-side contribution — *ex ante* auditable participation fairness in federated distillation — and demote mMIMO to a deployment context in §III. Concretely: retitle away from "for Massive-MIMO Systems"; open the abstract on predictable-versus-verifiable fairness rather than on wireless; and add the DS-FL substrate experiment (item 6 above) so the result is not tied to one physical layer. This also defuses M1 and the incrementality concern in one move, because the contribution becomes a *property* (computable fairness) rather than a *scheduler*.

---

## 8. Anticipated referee questions — prepare answers now

1. How does SCOPE-FD differ from round-robin scheduling, and what does Proposition 1 prove that round-robin does not give trivially?
2. UnionFL matches you on accuracy, Gini and convergence rounds in Table II. What is the contribution?
3. You claim fairness but measure only selection counts. Does any client end up better served?
4. Your headline accuracy is a statistical tie and you state you cannot reach $p < 0.0625$. On what basis is the accuracy claim made?
5. Why does the public set overlap the evaluation split, and what is the headline number without that overlap?
6. Why is Oort 50 points below random?
7. Corollary 2 bounds a separable sum of local losses. What does it say about the server model you actually evaluate?
8. Section VI-J shows the natural extension for an mMIMO deployment (channel/energy awareness) breaks the guarantee. Does the method serve its stated motivation?

Question 8 is the sharpest one and the paper answers it honestly but unfavourably. Consider addressing it constructively in §VII: a soft energy penalty inside the debt-equal cohort, rather than a hard feasibility filter, would preserve Proposition 1's selectability assumption. Even a small proof-of-concept experiment here would convert the paper's weakest section into a contribution.

---

## 9. Revision plan

**Must do before resubmission (blocking):**
M1 round-robin baseline and reframing · M2 UnionFL confrontation · M3 §VI-K correction · M5 per-client dispersion · M7 public-set handling · m1 canonical headline number

**Should do (materially improves acceptance odds):**
M4 accuracy claim discipline · M6 theory–experiment gap acknowledgement · M8 Oort · Item 5 CIFAR-10 · Item 6 non-mMIMO substrate · §7 reframing

**Nice to have:**
Remaining minor items · soft energy penalty in §VII

**Realistic outcome if all blocking items are addressed:** Accept with minor revision, or a second-round major revision on novelty grounds if the reviewer pool is unsympathetic to simplicity-as-a-feature. **If only cosmetic changes are made:** reject on incremental-contribution grounds, most likely citing M1 and M2 together.

---

## 10. Closing note

The manuscript's problem is not quality — it is framing. It presents itself as a three-term selector that improves accuracy and fairness, and its own evidence says it is a rotation that makes fairness *computable in advance* at no accuracy cost. The second claim is more interesting, more defensible and more unusual than the first, and the paper already contains everything needed to make it. The revision is largely a matter of leading with the result you actually have.