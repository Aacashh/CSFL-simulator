## CSFL Simulator: Client Selection Methods (Math, Intuition, and References)

This document describes all client-selection methods implemented in this repository, with formulas, intuition, and references. When we write scores, higher is better unless noted, and selection is typically “pick top-K by score,” or greedy where specified.

### Common notation
- Clients: \(\{c_i\}_{i=1}^n\)
- Data/features per client \(c_i\):
  - \(d_i\): data_size; \(\ell_i\): last_loss; \(g_i\): grad_norm
  - \(v_i\): compute_speed; \(q_i\): channel_quality; \(p_i\): participation_count
  - \(r_i\): recency in rounds since last selection
  - \(\hat t_i\): expected round duration proxy
  - \(h_i\in\mathbb{R}^L\): label histogram vector (if available)
  - \(\epsilon_i\): DP epsilon remaining (if DP is enabled)
- Expected duration proxy (when not directly provided):
  \[\hat t_i \approx \frac{d_i}{\max(v_i,10^{-6})} + \frac{d_i}{1000}\cdot \frac{1}{\max(q_i,10^{-6})}.\]
- Label entropy (used as a feature in some ML methods):
  \[H(h_i)= -\sum_{\ell} p_{i\ell}\,\log p_{i\ell},\quad p_{i\ell} = \frac{h_{i\ell}}{\sum_\ell h_{i\ell}}.\]
- Simple min–max normalization over the current cohort is used where noted.

For implementation references, see the indicated file under `csfl_simulator/selection/...`.

---

## Heuristic methods

### Random (uniform)
- Impl: `heuristic/random_select.py`
- Rule: sample \(K\) clients uniformly without replacement.
- Intuition: unbiased baseline; maximally explores; convergence slow; fair on average.

### Top-K Loss
- Impl: `heuristic/topk_loss.py`
- Score: \(s_i = \ell_i\). Select top \(K\).
- Intuition: prioritize hard clients (higher loss) to accelerate learning; can overuse a few clients.

### Proportional to Data Size
- Impl: `heuristic/proportional_data.py`
- Sampling without replacement with probabilities \(p_i = d_i/\sum_j d_j\).
- Intuition: more data → potentially larger contribution; may pick stragglers without time awareness.

### Gradient Norm
- Impl: `heuristic/gradient_norm.py`
- Score: \(s_i = g_i\). Select top \(K\).
- Intuition: larger gradient norms suggest larger potential update; can be noisy/outlier-prone.

### Fairness Adjusted
- Impl: `heuristic/fairness_adjusted.py`
- Score: \(s_i = \ell_i - \lambda\, p_i\) (default \(\lambda=0.1\)). Top \(K\).
- Intuition: retain utility while penalizing frequent participants to improve fairness.

### Cluster Balanced
- Impl: `heuristic/cluster_balanced.py`
- Steps: KMeans on features \([\ell_i, g_i, d_i]\); select roughly equally from clusters, fill remainder randomly.
- Intuition: enforce diversity across client types to reduce redundancy.

### Round Robin
- Impl: `heuristic/round_robin.py`
- Rule: cycle deterministically over sorted IDs with a persistent pointer.
- Intuition: strict participation fairness; ignores utility and system constraints.

### MMR Diverse (utility + diversity)
- Impl: `heuristic/mmr_diverse.py`
- Utility (normalized) combines signals (loss, grad_norm, speed, channel) with an inverse-frequency fairness factor; embeddings \(x_i\) are label histograms when available, else simple stats; diversity via cosine similarity \(s_{ij}=x_i^\top x_j\).
- Greedy MMR: pick \(i\) maximizing \(\lambda u_i - (1-\lambda)\max_{j\in S}s_{ij}\). \(\lambda\in[0,1]\).
- Intuition: balance relevance and anti-redundancy for strong one-shot subsets.
- Ref: Carbonell & Goldstein, 1998 (MMR) [ACM DOI](https://dl.acm.org/doi/10.1145/290941.291025)

### Label Coverage (greedy set cover)
- Impl: `heuristic/label_coverage.py`
- Build label incidence matrix from \(h_i\); scarcity weights \(w_\ell = 1/(\text{freq}_\ell+10^{-6})\) (IDF-like).
- Greedy gain for candidate \(i\): \((1-\alpha)\sum_{\ell\notin C} w_\ell \mathbf{1}\{V_{i\ell}>0\} + \alpha\,u_i\), with \(u_i=\) normalized \(\ell_i\), covered set \(C\).
- Intuition: maximize label coverage per round; mixing a utility term avoids trivial picks.
- Ref: Greedy set cover approx. (Chvátal, 1979) [Springer](https://link.springer.com/article/10.1007/BF01585954)

### DP-Budget Aware
- Impl: `heuristic/dp_budget_aware.py`
- Base utility: \(b_i = \text{norm}(\ell_i/\hat t_i)\). Penalty from remaining \(\epsilon_i\): either relative shortfall vs. required \(\epsilon^*\) or \(\propto 1/(1+\epsilon_i)\).
- Score: \(s_i = b_i/(1+\text{penalty}_i)\).
- Intuition: sustain learning quality across more rounds under DP budgets.
- Ref: DP-SGD (Abadi et al., 2016) [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

---

## System-aware methods

### FedCS (deadline-aware)
- Impl: `system_aware/fedcs.py`
- With time budget \(B\): greedy by \(\ell_i/\hat t_i\) (utility per unit time), packing until cumulative time \(\le B\). Fallback to Top-K Loss when \(B\) is absent.
- Intuition: knapsack-like selection to fit deadlines while maximizing utility/time.
- Ref: Nishio & Yonetani, “Client Selection for Federated Learning with Heterogeneous Resources,” ICC 2019 (FedCS). [arXiv (preprint)](https://arxiv.org/abs/1909.13014)

### TiFL (tiers)
- Impl: `system_aware/tifl.py`
- Tier clients (e.g., by resource class); round-robin across tiers; fill remainder randomly.
- Intuition: enforce cross-tier fairness and stability under heterogeneity.
- Ref: Li et al., “TiFL: A Tier-based FL System,” HPDC 2020. [arXiv](https://arxiv.org/abs/2001.09243)

### Oort-style Utility
- Impl: `system_aware/oort.py`
- Maintain selection counts \(n_i\) and global step \(t\). Score:
  \[ s_i = \frac{\ell_i}{\hat t_i} + \alpha\,\sqrt{\frac{2\log t}{\max(1,n_i)}}. \]
- Intuition: exploit high utility/time and explore underused clients via UCB.
- Ref: Oort (OSDI 2021) [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)

### Power-of-Choice (two-stage)
- Impl: `system_aware/poc.py`
- Stage 1: random pool of size \(m\). Stage 2 score in pool:
  \[ s_i = w_u\,\tilde \ell_i + w_s\,\widetilde{1/\hat t_i} + w_r\,\tilde r_i, \]
  select top-\(K\); under budget \(B\), greedily keep top-ranked while cumulative time \(\le B\).
- Intuition: light exploration, then exploit utility, speed, and recency for efficient rounds.
- Ref (conceptual): “Power of two choices” scheduling heuristics (Mitzenmacher & Upfal, survey).

### Oort-Plus (fairness + recency)
- Impl: `system_aware/oort_plus.py`
- Base: \(b_i = (\ell_i)^\beta/\hat t_i\); UCB term like Oort; penalties using recency gap \(g_i=r_i\): fairness \(\propto 1/(1+g_i)\), and recency \(\propto e^{-g_i/H}\).
- Score: \( s_i = \dfrac{b_i\,(1+\alpha_{ucb}\,\text{ucb}_i)}{1 + \text{fairness}_i + \text{recency}_i}. \)
- Intuition: stabilize usage via fairness/recency while keeping utility/time and mild exploration.
- Ref: Builds on Oort (OSDI 2021) [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)

---

## Bandit / ML methods

### Bandit: Epsilon-Greedy
- Impl: `ml/bandit/epsilon_greedy.py`
- Maintain per-client estimates \(Q_i\) (incremental average of last rewards). With prob. \(\epsilon\): explore (random); else exploit top \(Q_i\).
- Intuition: simplest explore/exploit baseline; no context.
- Ref: Sutton & Barto, RL basics (multi-armed bandit).

### Bandit: LinUCB
- Impl: `ml/bandit/linucb.py`
- Context \(x_i\in\mathbb{R}^d\); maintain \(A\in\mathbb{R}^{d\times d}, b\in\mathbb{R}^d\). Score:
  \[ p_i = \theta^\top x_i + \alpha\,\sqrt{x_i^\top A^{-1} x_i},\quad \theta=A^{-1}b. \]
- Intuition: balances exploitation and confidence-driven exploration with linear models.
- Ref: Li et al., WWW 2010 (LinUCB) [PDF](https://www.cs.princeton.edu/~rlivni/courses/linucb.pdf)

### Bandit: RFF-LinUCB
- Impl: `ml/rff_linucb.py`
- Map features with Random Fourier Features (RBF kernel approx) \(z_i=\phi(x_i)\); LinUCB on \(z_i\) with Sherman–Morrison updates.
- Intuition: capture mild nonlinearity with minimal overhead.
- Refs: Rahimi & Recht, NIPS 2007 [arXiv:0711.2521](https://arxiv.org/abs/0711.2521); LinUCB as above.

### Meta Ranker (SGDRegressor)
- Impl: `ml/meta_ranker.py`
- Standardized features; online SGDRegressor predicts reward \(\hat r_i\). Cold start fallback: \(\hat r_i \approx \ell_i/\hat t_i\).
- Intuition: supervised ranking from simulator feedback; lightweight and adaptive.
- Ref: Online linear models (SGD) in scikit-learn docs (background resource).

### NeuralLinear-UCB (tiny MLP + Bayesian head)
- Impl: `ml/neural_linear_ucb.py`
- Representation \(z_i=f_\theta(x_i)\) via tiny MLP; Bayesian linear head with \(A^{-1}, b\). Score \(p_i=\mu(z_i) + \alpha\,\sigma(z_i)\) with \(\mu=\theta^\top z_i\), \(\sigma=\sqrt{z_i^\top A^{-1} z_i}\).
- Intuition: representation learning + principled uncertainty; robust in nonlinear regimes.
- Ref: Riquelme et al., “Deep Bayesian Bandits Showdown,” 2018 [arXiv:1807.02009](https://arxiv.org/abs/1807.02009)

### DeepSets Ranker (permutation-invariant)
- Impl: `ml/deepset_ranker.py`
- DeepSets: embeddings \(\phi(x_i)\); set summary \(s=\sum_j \phi(x_j)\); score \(y_i=\psi([\phi(x_i), s])\). Train online with shaped reward that encourages fairness and speed.
- Intuition: learns context-aware, order-invariant ranking across the cohort.
- Ref: Zaheer et al., NeurIPS 2017 (Deep Sets) [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)

### GAT Selector (demo)
- Impl: `ml/gat/selector.py`
- Fully connected client graph; GAT produces per-node scores; pick top \(K\). Falls back to random if PyG unavailable.
- Intuition: leverage attention over client–client relations.
- Ref: Veličković et al., ICLR 2018 (GAT) [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

### Graph Transformer (demo)
- Impl: `ml/gt_ppcs/selector.py`
- Transformer encoder over per-client features; linear head scores nodes; small fairness correction post-score.
- Intuition: capture interactions across clients with self-attention.
- Ref: Vaswani et al., 2017 (Transformer) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### RL-GNN Policy (GCN + REINFORCE stub)
- Impl: `ml/rl_gnn/policy.py`, trainer in `ml/rl_gnn/trainer.py`
- Policy network: GCN over fully connected client graph; train with a simple REINFORCE objective using episode reward \(R=\text{final\_acc}-\text{initial\_acc}\).
- Intuition: directly optimize a long-term global metric via policy gradients.
- Refs: Williams, 1992 (REINFORCE) [Springer](https://link.springer.com/article/10.1007/BF00992696); GCN (Kipf & Welling) [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

### RankFormer-Tiny (Transformer ranker)
- Impl: `ml/rankformer_tiny.py`
- Transformer encoder scores each client; select top \(K\); online MSE training toward scalar reward on last selected indices.
- Intuition: sequence model approximating ranking with small capacity; learns cross-client patterns.
- Ref: Vaswani et al., 2017 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## Practical guidance
- Best overall without strict deadlines: `system_aware.oort_plus`.
- With heterogeneity/time budget: `system_aware.poc` (two-stage) or `system_aware.fedcs`.
- Severe label skew (histograms available): warm up 3–5 rounds with `heuristic.label_coverage`, then `oort_plus`. `heuristic.mmr_diverse` is a strong one-shot alternative.
- After short warm-up, for max late-round accuracy: `ml.bandit.rff_linucb` or `ml.neural_linear_ucb`.
- Fairness-critical: `system_aware.oort_plus`; simpler alternatives: `heuristic.fairness_adjusted`, `heuristic.round_robin`.
- Strict DP budgets: `heuristic.dp_budget_aware`.

---

## Bibliography (selected)
- Oort: Efficient Federated Learning via Guided Participant Selection (OSDI 2021). [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)
- FedCS: Client Selection for Federated Learning with Heterogeneous Resources (ICC 2019). [arXiv:1909.13014](https://arxiv.org/abs/1909.13014)
- TiFL: A Tier-based Federated Learning System (HPDC 2020). [arXiv:2001.09243](https://arxiv.org/abs/2001.09243)
- Maximal Marginal Relevance (SIGIR 1998). [ACM DOI](https://dl.acm.org/doi/10.1145/290941.291025)
- Greedy Set Cover Approximation (Chvátal, 1979). [Springer](https://link.springer.com/article/10.1007/BF01585954)
- LinUCB (WWW 2010). [PDF](https://www.cs.princeton.edu/~rlivni/courses/linucb.pdf)
- Random Fourier Features (NIPS 2007). [arXiv:0711.2521](https://arxiv.org/abs/0711.2521)
- Deep Bayesian Bandits Showdown (2018). [arXiv:1807.02009](https://arxiv.org/abs/1807.02009)
- Deep Sets (NeurIPS 2017). [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- Graph Attention Networks (ICLR 2018). [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Attention Is All You Need (NeurIPS 2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- REINFORCE (1992). [Springer link](https://link.springer.com/article/10.1007/BF00992696)
- DP-SGD (CCS 2016). [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

### FedCor (approximate correlation-aware active selection)
- Impl: `ml/fedcor_approx.py`
- Greedy objective: \(s(i\mid S) = u_i - \alpha\max_{j\in S} \text{cos}(h_i, h_j)\), where \(u_i=\text{norm}(\ell_i/\hat t_i)\), \(h_i\) label-hist vectors.
- Intuition: prefer high-utility clients with low correlation to selected ones; fast approximation of FedCor’s correlation modeling.
- Ref: Li et al., “FedCor: Correlation-based Active Client Selection,” 2021. [arXiv:2103.13822](https://arxiv.org/abs/2103.13822)

### DP-EIG (DP-aware information-gain greedy)
- Impl: `ml/dp_eig.py`
- Base EIG proxy: \(\text{eig}_i \propto d_i( g_i^2 + \ell_i ) / (\hat t_i(1+\sigma^2))\); greedy with label-coverage gain and optional time-budget constraint.
- Intuition: approximate Bayesian experimental design under DP noise; favors informative and fast clients with complementary labels.
- Refs: DP-SGD (Abadi et al., 2016); submodular greedy (Nemhauser et al., 1978, background theory).

### GNN-DPP (diverse graph scoring + DPP-style sampling)
- Impl: `ml/gnn_dpp.py`
- Attention-aggregated utility from cosine-neighborhoods, with DPP-style diversity via MMR: \(\text{score}_i = \tilde u_i - \lambda\max_{j\in S}\cos(x_i,x_j)\).
- Intuition: capture cross-client relations and enforce diversity to be robust under non-IID and DP noise.
- Refs: Graph attention (Veličković et al., 2018); DPP selection (Kulesza & Taskar, 2012, background).

### ParetoRL (multi-objective constrained greedy)
- Impl: `ml/pareto_rl.py`
- Composite score: \( s_i = w_{acc}\,\text{norm}(\ell_i/\hat t_i) + w_{time}\,\text{norm}(1/\hat t_i) + w_{fair}\,\tilde f_i + w_{dp}\,\tilde \epsilon_i \). Optional safety: include a minimum number of low-participation clients first; respects time budget greedily.
- Intuition: practical approximation to multi-objective RL with a safety layer; stable and strong under heterogeneity and DP.
- Refs: Constrained RL (Altman, 1999, background); Oort for reward shaping under FL.
