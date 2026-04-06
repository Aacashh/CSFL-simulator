# Critical Review of APEX Paper for IEEE TAI Submission

## Preamble: IEEE TAI Standards Applied

IEEE TAI requires: (1) technical rigor with correct, sound investigations; (2) self-contained and evidence-based work; (3) sufficient information for replication; (4) results supported by adequate data; (5) novel contributions advancing the state of knowledge. TAI is Q1 and explicitly warns that "authors should expect to be challenged by reviewers if results are not supported by adequate data and critical details." The analysis below applies these standards.

---

## CRITICAL ISSUES (Likely Rejection Triggers)

### 1. Absolute Accuracy Numbers Are Alarmingly Low

**The Problem:** APEX achieves **53.47% on CIFAR-10** after 50 rounds. Even standard centralized training on CIFAR-10 with a similar 2-layer CNN reaches ~75-80%. The baseline FedAvg at 46.15% is also far below what is expected. This suggests something is fundamentally wrong with the experimental setup — either the model is too weak, the number of rounds too few, or the data split too extreme.

**Why reviewers will flag this:** Recent IEEE TAI and peer FL papers (e.g., Mai et al. 2024, Vahidian et al. 2024) report FedAvg at 60-75% on CIFAR-10 with Dirichlet α=0.3. A reviewer familiar with the literature will immediately question why ALL methods, including FedAvg, underperform so dramatically. This undercuts the entire experimental contribution.

**What to do:**
- Increase the number of communication rounds to at least 200-500 (50 is unusually low for CIFAR-10).
- Use a stronger model (e.g., ResNet-18 or a 4-layer CNN), which is standard in the FL client selection literature.
- If you keep 50 rounds for a specific reason, justify it explicitly and compare with longer training.

---

### 2. The Ablation Study Shows Full APEX Is *Worse* Than Ablated Versions

**The Problem:** Table III shows that **removing** adaptive recency, hysteresis, or heterogeneity scaling each **increases** final accuracy by 5-6 percentage points. The full APEX model achieves 35.45% while "No Het Scaling" gets 41.98%. This is a devastating result.

**Why reviewers will flag this:** This directly contradicts the paper's core claim that these mechanisms are beneficial. The paper's defense — "the accuracy gap is the cost of fairness" — will not satisfy a rigorous reviewer because:
- The paper frames APEX primarily as an accuracy-competitive method (the abstract highlights "5.32 percentage points above the next best baseline").
- A 5-6 pp drop is enormous, especially at these low accuracy levels.
- The claim that this is a fairness-accuracy tradeoff is not backed by any formal Pareto analysis. There's no evidence that this particular tradeoff point is optimal.
- The full APEX Gini (0.3962) is actually *worse* than "No Hysteresis" (0.2479), meaning the ablation without hysteresis achieves both better accuracy AND better fairness.

**What to do:**
- Run the ablation at the same N=50, K=10 setting as your main benchmark (not N=100).
- Provide a Pareto frontier plot of accuracy vs. fairness for all variants.
- Reconsider whether hysteresis and adaptive recency are truly helpful, or whether simpler configurations work better.
- If the tradeoff is real, frame the paper around the tradeoff rather than accuracy leadership.

---

### 3. Theoretical Claims Are Informal ("Remark" and "Proposition") But Lack Proofs

**The Problem:** Section V contains "Remark 1" (dual bound reduction) and "Proposition 2" (regret bound), but neither has a formal proof. Remark 1 is essentially a qualitative argument ("APEX targets reduction of both terms... through the following mechanisms"). Proposition 2 states an O(√(NT log T)) bound but defers to standard Thompson sampling analysis from Agrawal & Goyal (2012), then immediately lists two caveats that weaken the claim:
- The bound assumes single-arm selection, but APEX selects K arms.
- The variance floor changes the concentration rate.

**Why reviewers will flag this:** IEEE TAI explicitly values "correct and sound investigations." A proposition without proof, followed by caveats saying the assumptions don't hold, will be seen as hand-waving. The combinatorial bandit regret analysis is non-trivial and the "decomposition into K independent pulls" assumption is known to be loose. The paper acknowledges this but doesn't resolve it.

**What to do:**
- Either provide a formal proof that accounts for the combinatorial selection and variance floor, or downgrade the claim to an informal discussion clearly labeled as such.
- Consider citing combinatorial Thompson sampling results (e.g., from Chen et al., ICML 2013) that handle the K-arm case properly.
- At minimum, add an empirical regret plot showing the actual cumulative regret over rounds to validate the theoretical claim empirically.

---

### 4. Only 50 Communication Rounds — Insufficient for Drawing Conclusions

**The Problem:** All experiments run for only T=50 rounds. This is extremely short for FL experiments. Most FL papers use 200-1000 rounds. At T=50, models are far from convergence, and the results reflect early-training dynamics rather than final performance.

**Why reviewers will flag this:**
- The "late-stage surge" between rounds 45-50 (3.48 pp gain) looks like the model just started converging — not a signature of phase-adaptive design.
- Comparing methods at T=50 when none have converged tells us about early-phase behavior, not about which method produces the best final model.
- The claim "APEX is the only method to exceed 50% within 50 rounds" is a weak claim because 50% on CIFAR-10 is low.

**What to do:**
- Run experiments for at least 200 rounds, ideally 500.
- Show convergence curves that demonstrate plateau behavior.
- If you want to emphasize communication efficiency, show the round at which each method reaches a target accuracy (e.g., 60%, 70%).

---

### 5. Missing Important Baselines

**The Problem:** The baselines are FedAvg, FedCS, FedCor, and TiFL. Several critical baselines from the related work are missing:
- **Power-of-Choice (PoC)** — discussed extensively in the paper as a key prior work, but never compared against experimentally.
- **DivFL** — the main diversity-based method in the related work, also never compared.
- **FedProx** — a standard FL baseline for non-IID settings.
- **SCAFFOLD** — widely used convergence-improved FL method.

**Why reviewers will flag this:** The paper's positioning is that APEX combines the strengths of PoC (loss-biased) and DivFL (diversity) while being lightweight. Not comparing against either of them is a glaring omission. A reviewer will ask: "If your method combines the ideas of PoC and DivFL, shouldn't you at least show it outperforms both?"

**What to do:**
- Add PoC and DivFL as baselines. PoC is easy to implement (it's just selecting high-loss clients). DivFL requires gradient communication, so you can note its overhead disadvantage while still comparing accuracy.
- Consider adding FedProx as a complementary aggregation baseline.

---

## MAJOR CONCERNS (Likely Major Revision)

### 6. The "Zero Trainable Parameters" Claim Is Misleading

The paper repeatedly emphasizes "zero trainable parameters" as a key advantage. However, APEX has many manually tuned hyperparameters: phase thresholds (τ_c=0.05, τ_u=0.10, τ_e=0.01), contextual weights (w_l=0.4, w_g=0.2, w_s=0.2, w_q=0.2), EMA coefficient (α_e=0.3), blending weight (γ=0.3), variance floor constant (c_f=0.1), window size (W=5), dwell time (δ_min=3), phase-dependent weight triplets (three sets of three weights), and the recency constant formula. That's roughly 15-20 hand-tuned values.

While the paper correctly distinguishes between "trainable" (gradient-updated) and "manually specified" parameters, the practical burden is similar — these need to be set, and their sensitivity isn't studied. A reviewer will argue that a method with 20 hyperparameters isn't truly "lightweight" in terms of tuning effort.

**What to do:**
- Add a sensitivity analysis for the key hyperparameters (at least the phase thresholds and phase weights).
- Reduce the emphasis on "zero trainable parameters" — instead frame it as "no neural network training overhead" or "no backpropagation required."

---

### 7. The Composite Score (Eq. 20) Is Non-Standard and Self-Serving

**The Problem:** The composite score C = 0.6·Acc + 0.2·(round time penalty) + 0.1·(fairness term) + 0.1·(privacy term) is a custom metric with arbitrary weights. The paper's claim "APEX obtains the best composite score in 4 out of 5 settings" is based on a metric the authors themselves designed. The 0.6 weight on accuracy and the inclusion of a differential privacy budget term (ε̄) — which is not a focus of the paper — seems chosen to favor APEX.

**Why reviewers will flag this:** Custom evaluation metrics are acceptable if well-justified, but reviewers will question why standard metrics (accuracy, convergence speed, Gini coefficient) aren't sufficient. The composite score obscures rather than clarifies the results.

**What to do:**
- Report and discuss standard metrics separately (accuracy, convergence round to target, Gini coefficient).
- If you keep the composite score, provide sensitivity analysis over the weight choices.
- Remove the DP budget term from the composite unless DP is actually part of the framework.

---

### 8. Privacy Concern With Label Histograms Is Inadequately Addressed

**The Problem:** APEX requires each client to share its label histogram h_i with the server. The paper acknowledges this is a privacy concern (Section IV-C) but defers the solution to "future investigation." This is a significant gap because the paper is positioned as a method for FL, where privacy is a core requirement.

**Why reviewers will flag this:** Sharing label histograms reveals the exact class distribution of each client's data. In many FL applications (e.g., healthcare, finance), this is unacceptable. The claim that "the diversity proxy is robust to moderate noise" from DP is unsubstantiated — no experiments or analysis support this claim.

**What to do:**
- At minimum, run an experiment with noisy histograms (add Laplace or Gaussian noise at different privacy budgets) and show how the accuracy and diversity proxy quality degrade.
- If the degradation is graceful, this becomes a strength. If not, acknowledge it honestly.

---

### 9. Statistical Rigor Is Missing

**The Problem:** All results are reported from single runs (seed 42). There are no error bars, confidence intervals, or statistical significance tests. IEEE TAI requires results "supported by adequate data."

**What to do:**
- Run each experiment with at least 3-5 random seeds.
- Report mean ± standard deviation.
- Use statistical tests (e.g., paired t-test) for key comparisons.

---

### 10. The Reward Signal Design Is Questionable

**The Problem:** The reward for each client (Eq. 16) is defined as the *composite score improvement* divided equally among all selected clients: r_i = ΔC/|S_t|. This means:
- All selected clients get the same reward, regardless of their individual contribution.
- The reward is based on the custom composite score, not on a direct measure of client utility.
- Good clients are punished when paired with bad clients (and vice versa).

**Why reviewers will flag this:** Equal credit assignment is a known problem in cooperative multi-agent settings. It makes the Thompson sampling posteriors noisy and slow to converge. The paper doesn't discuss alternatives like Shapley value-based attribution or per-client loss improvement.

**What to do:**
- Discuss the credit assignment problem explicitly.
- Consider using per-client loss decrease as the reward signal instead.
- At minimum, compare the equal-credit reward against a per-client reward in an ablation.

---

## MODERATE CONCERNS (Likely Minor Revision)

### 11. The Phase Detector Thresholds Are Fixed But Dataset-Dependent
The thresholds τ_c=0.05, τ_u=0.10, τ_e=0.01 are described as "standard statistical significance levels," but loss improvement rates are highly dataset- and model-dependent. What counts as "rapid decrease" (ρ > 0.05) on CIFAR-10 may be completely different on a medical imaging dataset. No sensitivity analysis is provided.

### 12. Scalability Claim Is Weakly Supported
The "scalability test" goes from N=50 to N=100 — this is a modest increase. True scalability testing would go to N=500 or N=1000, which is common in the FL literature.

### 13. Limited Dataset Diversity
Only image classification datasets (CIFAR-10, MNIST, Fashion-MNIST) are used. MNIST and Fashion-MNIST are considered too simple for modern FL research. Consider adding a more challenging dataset (e.g., CIFAR-100, FEMNIST, or a non-vision dataset like a text classification task).

### 14. Section Ordering: Future Work Before Conclusion
The paper has Section VII as "Conclusion" and Section VIII as "Future Work," but the numbering in the text says Section VII is Future Work and Section VIII is Conclusion. This is a formatting error that suggests rushed preparation.

### 15. The "11.3× acceleration rate" Claim Is Cherry-Picked
Comparing the last-10-round acceleration rate of APEX vs. TiFL is methodologically questionable. A method could show a high late-stage acceleration simply because it was underperforming earlier and is now catching up. This metric isn't standard in the FL literature.

### 16. EMA Smoothing May Mask Important Signal
The EMA coefficient α_e = 0.3 gives 70% weight to historical rewards. This means the Thompson posteriors respond slowly to changes in client utility — which directly conflicts with the goal of phase-adaptive behavior where the strategy needs to respond quickly to phase transitions.

---

## PRESENTATION ISSUES

### 17. Title Contains "Novel"-Style Language
IEEE TAI explicitly warns against using phrases like "a novel methodology" — the title "APEX: Adaptive Phase-Aware Exploration" is fine, but the paper text uses "proposed" excessively. This is acceptable but should be toned down.

### 18. Figure Quality
The system model figure (Fig. 1) references an "APEX_infographic_exported.png" — ensure this is high-resolution and camera-ready. The EPS figures for the plots should be vector graphics.

### 19. Missing Reproducibility Details
- No code or data availability statement.
- The LightCNN architecture details are minimal.
- Batch size, optimizer details (plain SGD? momentum?), and learning rate schedules are not fully specified.

---

## SUMMARY: PRIORITIZED ACTION ITEMS

| Priority | Issue | Impact on Decision |
|----------|-------|-------------------|
| **P0** | Low absolute accuracy (53% on CIFAR-10) | Likely rejection |
| **P0** | Ablation shows APEX worse than its variants | Likely rejection |
| **P0** | Missing key baselines (PoC, DivFL) | Likely rejection |
| **P0** | Only 50 rounds, no convergence | Likely rejection |
| **P1** | No error bars / single seed | Major revision |
| **P1** | Theoretical claims without proofs | Major revision |
| **P1** | Privacy gap with label histograms | Major revision |
| **P1** | Composite score is non-standard | Major revision |
| **P2** | "Zero parameters" claim misleading | Minor revision |
| **P2** | Reward design (equal credit) | Minor revision |
| **P2** | Limited dataset diversity | Minor revision |
| **P2** | Scalability only to N=100 | Minor revision |
| **P3** | Section ordering error | Easy fix |
| **P3** | Reproducibility details | Easy fix |
| **P3** | Sensitivity analysis missing | Moderate effort |

---

## BOTTOM LINE

The paper has a well-motivated idea — combining loss-biased selection with diversity in a phase-adaptive way — and the framework design is creative. However, **the experimental evaluation falls significantly below IEEE TAI standards**. The four P0 issues (low accuracy, ablation contradiction, missing baselines, insufficient rounds) would likely lead to rejection in the current form. The strongest recommendation is to:

1. **Extend training to 200+ rounds** with a stronger model.
2. **Add PoC and DivFL baselines** — these are essential given the paper's positioning.
3. **Fix the ablation study** at the main benchmark setting and address the accuracy gap honestly.
4. **Run multiple seeds** and report statistics.

If these are addressed, the paper has a reasonable chance at IEEE TAI, especially given the lightweight nature of the algorithm and the fairness analysis, which is a timely topic.
