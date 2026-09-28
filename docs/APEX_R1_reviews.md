# APEX — Round 1 Reviews (verbatim)

Source: decision letter as pasted by the authors on 2026-09-28. Kept verbatim so
the response letter can quote each comment exactly. The bracketed labels
[R2.1]–[R2.6] and the heading levels are ours, for cross-referencing in
[APEX_revision_plan.md](APEX_revision_plan.md).

---

## Associate Editor Comments to Author

Associate Editor
Comments to the Author:
APEX presents a lightweight and potentially useful phase-aware client-selection strategy for federated learning.

However, there are concerns regarding experiments and presentations.
In revision, the authors should reconcile all experimental discrepancies, strengthen statistical reporting and baseline/runtime comparisons, clarify or temper the theoretical claims, and improve notation and presentation.

---

## Reviewer 1

Comments to the Author
The paper proposes a lightweight client selection algorithm named APEX, short for Adaptive Phase-Aware Exploration, to address the client selection problem in federated learning under non-identically distributed, or non-IID, data. The method combines contextual Thompson sampling with a label-histogram-based diversity proxy to score and select clients. Its core idea is to emphasize client diversity during the early or unstable stages of training, while shifting toward the exploitation of high-utility clients in later stages. In this way, the method aims to mitigate both client drift and gradient variance caused by partial participation under non-IID conditions.

### Main Contributions

1. **Phase-aware client selection framework:**

   The paper designs a phase detector based on loss trajectory analysis. It classifies the training process into three phases: critical, transition, and exploitation. Different weights are assigned to Thompson sampling, diversity, and recency in different phases. The algorithm also introduces hysteresis and a minimum dwell time to reduce frequent oscillations between phases.

2. **Heterogeneity-aware scaling mechanism:**

   The algorithm estimates the level of data heterogeneity by computing the average Jensen-Shannon divergence among client label distributions. This estimate is then used to scale the diversity weight. Under near-IID conditions, the diversity weight is reduced; under severe non-IID conditions, the diversity weight is largely preserved. This mechanism reduces the need to manually tune the diversity weight across different heterogeneity levels.

3. **Posterior regularization and confidence-aware blending:**

   The paper introduces a variance floor that decays with the number of client observations, preventing the Thompson sampling posterior from collapsing too early during training. It also uses a confidence-aware blending parameter, so that newly observed clients rely more on contextual feature scores, while well-observed clients rely more heavily on Thompson sampling posteriors.

4. **Experimental evaluation:**

   The paper evaluates APEX on CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST, and compares it with methods including FedAvg, FedCS, FedCor, TiFL, Oort, PoC, and MMR-Diverse. The experiments claim that APEX achieves the highest accuracy in most settings and provides better participation fairness than system-aware baselines. However, there are several inconsistencies and insufficiently supported claims in the tables, text, and ablation analysis that require clarification.

### Issues Identified

1. **The improvement in the main experiment is too small to support a strong claim of significant superiority.**

   In Table II, APEX achieves a final accuracy of 70.7%, while MMR-Diverse achieves 70.5% and FedAvg achieves 69.5%. The improvement over MMR-Diverse is only about 0.2 percentage points, and the standard deviation ranges of the two methods clearly overlap. Even compared with FedAvg, the gain is only about 1.2 percentage points. Without statistical significance tests or confidence interval analysis, the paper may state that APEX “ranked first” or “slightly outperformed” the baselines, but it should avoid strong claims such as “significantly outperforms all baselines.”

2. **The APEX results in Table II and Table V are inconsistent.**

   Both Table II and Table V claim to use the same setting: CIFAR-10, α = 0.3, N = 50, K = 10, T = 200, and three random seeds. However, Table II reports the final accuracy of APEX as 0.707 ± 0.012, while Table V reports the full APEX model as 0.694 ± 0.015. Since the experimental settings appear identical but the results differ, this discrepancy affects the interpretation of both the main benchmark and the ablation study. The paper should clarify whether this difference is due to different runs, different random seeds, different implementation versions, or a table error.

3. **The description of Fig. 2(b) is inconsistent with the loss curves.**

   The text states that APEX and FedAvg achieve the lowest final training loss. However, Fig. 2(b) appears to show that APEX and MMR-Diverse have the lowest final loss, while FedAvg’s final loss is higher than that of MMR-Diverse. This statement should be corrected to match the figure.

4. **The ablation study weakens the claimed necessity of several core mechanisms.**

   Table V shows that, under α = 0.3, N = 50, and K = 10, removing heterogeneity-aware scaling, posterior regularization, adaptive recency, or adaptive gamma actually leads to higher final accuracy than the full APEX model. Only removing hysteresis slightly reduces final accuracy and substantially increases variance. Therefore, it would be inaccurate to say that removing any core mechanism improves accuracy. A more accurate statement is that, except for hysteresis, removing most components improves the mean final accuracy under this setting.

   This suggests that the paper does not sufficiently prove the necessity of these mechanisms under moderate heterogeneity. Although the authors argue that these components are designed for harder conditions, such as α = 0.1 or larger client pools, the paper does not provide corresponding ablation experiments under those harder settings. Therefore, the claim that each component contributes positively to performance is insufficiently supported.

5. **The statement that α = 0.3 is the most benign setting is not rigorous.**

   In the ablation discussion, the paper states that α = 0.3 is the most benign heterogeneity level in the evaluation suite. However, the earlier experiments clearly include α = 0.6. Since a larger Dirichlet α usually corresponds to a data distribution closer to IID, α = 0.6 should be more benign than α = 0.3. This statement is inconsistent with the experimental setup.

6. **The number and scope of baselines are not consistently reported.**

   The experimental setup lists seven baselines: FedAvg, FedCS, FedCor, TiFL, Oort, PoC, and MMR-Diverse. Table II contains APEX plus these seven baselines, for a total of eight methods. However, some parts of the paper use the phrase “eight baselines,” which is confusing.

   In addition, Table III compares only APEX, FedAvg, PoC, MMR-Diverse, and Oort; Table IV compares only APEX, FedAvg, Oort, and PoC. The paper does not explain why FedCS, FedCor, TiFL, or MMR-Diverse are excluded from some experiments. This may give the impression of selective reporting. The authors should either provide a clear explanation for excluding certain baselines or keep the baseline set consistent across all core experiments.

7. **The value of K in Table IV may conflict with the experimental setup.**

   The experimental setup states that N ∈ {50, 100, 200}, with K = 10 clients selected per round, except that K = 20 when N = 200. However, the title of Table IV states α = 0.3 and K = 10 while also reporting results for N = 200. The paper should clarify whether K = 10 or K = 20 was used when N = 200. Otherwise, the scalability experiment is ambiguous.

8. **The reward allocation in Equation (16) is overly coarse.**

   Equation (16) evenly distributes the global accuracy improvement ΔAcc(t) among all selected clients. This means that a client contributing low-quality gradients, or even harming the global model, may receive the same positive reward as a genuinely useful client. This credit assignment mechanism may weaken the ability of Thompson sampling to accurately estimate each client’s true value. The paper should explain why equal reward allocation is reasonable or provide a more fine-grained estimate of client contribution.

   In addition, the equation uses global test accuracy as the reward signal. In real federated learning deployments, the server may not have access to a representative global test set, which could limit the practical applicability of the method.

9. **F1 score is reported but not sufficiently discussed.**

   Both Table II and Fig. 2(c) report F1 score, but the main text primarily discusses accuracy, training loss, and Gini fairness. It provides little analysis of the F1 trends and does not clearly explain whether APEX truly outperforms other methods in terms of F1. Since F1 is an important classification metric, especially under non-IID and class-imbalanced settings, the paper should add a discussion of the F1 results in Fig. 2(c) and Table II.

10. **The theoretical analysis is more intuitive than rigorous.**

    In Section V.A, the discussion around Equation (5) mainly explains that APEX targets client drift through loss-biased selection and gradient variance through the diversity proxy. However, the paper does not rigorously prove that the greedy selection based on label histograms can achieve any optimal or suboptimal gradient variance reduction rate in expectation.

    The paper cites the submodular gradient selection results from DivFL, but APEX does not directly use true gradients. It also does not prove that cosine distance between label histograms reliably approximates gradient diversity. Therefore, directly connecting APEX’s diversity proxy with theoretical results based on true gradients is insufficiently justified.

11. **The applicability of the Thompson sampling regret bound is not fully established.**

    Section V.B claims that the Thompson sampling component achieves an O(√NT log T) Bayesian regret bound. However, APEX selects K clients per round, and the selected clients interact through model aggregation, which is not equivalent to a standard independent single-arm pull setting. The paper itself acknowledges that the stated bound depends on the simplifying assumption that K-arm selection can be decomposed into K independent pulls. Moreover, APEX uses global model accuracy improvement as the reward, which is non-stationary and strongly coupled with the overall training process. Therefore, this regret bound should be presented as a heuristic reference rather than a strict theoretical guarantee for the full APEX algorithm.

12. **There is a gap between the theoretical assumptions and the experimental models.**

    The convergence bound in Equation (5) relies on strongly convex and smooth local objectives, whereas the experiments use ResNet18 and CNN models, which are generally non-convex neural networks. The paper may use this bound as a design motivation, but it should explicitly acknowledge the gap between the theoretical assumptions and the experimental models. Otherwise, readers may mistakenly infer that the convergence bound directly applies to the experimental setting.

13. **Static caching of label histograms limits applicability to dynamic scenarios.**

    The paper states that label histograms and the heterogeneity scalar are computed and cached in the first round, and then reused thereafter. This is reasonable for the static partitioned datasets used in the experiments. However, the Impact Statement emphasizes potential applications in healthcare monitoring, mobile networks, and Internet of Things systems. In such scenarios, client data distributions may drift over time. If the algorithm continues to rely on label histograms cached in the first round, its diversity assessment may gradually become invalid.

    Therefore, this should not be framed as a contradiction with a streaming-data setting, because the original system model does not explicitly define a streaming-data scenario. A more accurate criticism is that the static caching mechanism limits the external validity of APEX in dynamic or distribution-drift scenarios. The paper should either weaken its real-world applicability claims or add mechanisms such as periodic histogram updates, drift detection, or sliding-window histograms.

14. **The counting basis for “8 out of 9 settings” is unclear.**

    The paper repeatedly claims that APEX ranks first in 8 out of 9 experimental settings. However, based on the organization of Table III and Table IV, it is not clear how these nine settings are counted. Some settings also appear to overlap, such as CIFAR-10 with α = 0.3 and N = 50, which appears in the main benchmark, heterogeneity robustness, scalability, and cross-dataset results. The authors should explicitly define which nine experimental conditions are included in this count and avoid double counting or selective counting.

---

## Reviewer 2

Comments to the Author
This paper introduces APEX, a lightweight client selection algorithm for FL under non-iid data that combines contextual Thompson sampling, a label-histogram diversity proxy, and phase detector to dynamically balance exploration and diversity. Operating with zero trainable parameters and requiring no gradient uploads. APEX adds only 3.9ms of per-round overhead while consistently outperform existing baselines in accuracy and participation fairness.
I have identified a few topics for improvement in the manuscript:

**[R2.1]** Variables introduced before their definitions: Variables are frequently mentioned before being defined, which hinders readability. For example, in Section III-B, Eq. 5 uses terms such as gamma and sigma^2, but the text only briefly notes that these terms "are defined below". I recommend defining all variables immediately before or right after the equation in which they first appear or use a notation table.

**[R2.2]** 2) Confusing variable notation: The current notation overloads certain letters, leading to potential confusion. For instance, w is used in eq. 3 to denote the global and local model parameters. However, in Section IV-B eq. 12, the authors use w_l, w_g, w_s,... to denote constant scalar weights. I suggest using a distinct notation for the scalar weights (e.g., alpha, beta, or lambda) to avoid ambiguity with the model weights.

**[R2.3]** 3) Please add the standard deviations to add standard deviations to tables iii e iv.

**[R2.4]** 4) Table I compares the algorithms theoretically using Big-O notation. While the authors claim a 3.9 ms overhead per round for the proposed method, the paper would benefit from an empirical evaluation showing the actual selection time per round across all evaluated algorithms on identical hardware.

Other observations:

**[R2.5]** The results in Table II show that baseline methods such as Power-of-Choice (PoC) and Oort perform significantly worse than standard FedAvg (65.30% and 64.50% vs. 69.52%). While the authors attribute this degradation to PoC’s over-selection of high-loss clients and Oort’s bias toward faster clients under non-IID conditions, the paper should clarify if FedAvg uses uniform random partial sampling (K=10 out of N) in every round and discuss why these established baselines degrade so respectively compared to their original published results.

**[R2.6]** Although the manuscript discusses learning-based methods, such as FedAEB, which also uses a proxy for the histogram, a comparison may be beneficial for the work.
