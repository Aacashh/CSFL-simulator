# Cross-Device Federated Learning: Client Selection under DP and Heterogeneity

This guide outlines an IEEE-ready paper structure, baselines, proposed methods, experiment design, and figure checklist for a compelling submission.

## 1. Title
- "Practical and Robust Client Selection for Cross-Device Federated Learning under Differential Privacy and System Heterogeneity"

## 2. Abstract (150–200 words)
- Problem: Cross-device FL needs client selection robust to non-IID data, DP noise, stragglers, and bandwidth variability.
- Contribution: Three new methods (ParetoRL, DP-EIG, GNN-DPP) that jointly optimize utility, speed, fairness, and DP; strong baselines included.
- Results: Faster time-to-X accuracy and higher worst-client accuracy at ε ∈ {1,3,8} vs. Oort/FedCS/TiFL and bandits.

## 3. Introduction
- Importance of client selection in cross-device FL; practical constraints (availability, stragglers, network/computation heterogeneity, DP-SGD noise).
- Gaps: Many methods optimize single objectives; few are DP-aware; diversity often neglected; fairness rarely explicit; ablation rigor limited.
- Contributions:
  1) ParetoRL (multi-objective constrained selection).
  2) DP-EIG (DP-aware information-gain greedy).
  3) GNN-DPP (attention-aggregated utility + diversity sampling).
  4) A unified evaluation at ε ∈ {1,3,8}, budget K, dropouts, and heterogeneity with strong baselines.

## 4. Related Work
- Baselines (with canonical refs):
  - FedAvg; FedCS (deadline-aware); TiFL (tiers); Oort (utility/time + UCB);
  - Bandits: ε-greedy, LinUCB; (optional) NeuralLinear.
  - Correlation-aware: FedCor (GP-based active selection).
  - Generative/greedy: GreedyFed (Shapley approx); FedGCS (generative selection).
- DP-SGD, RDP accounting, FL systems on mobile/edge.

## 5. Problem Setting and Objectives
- Cross-device FL: 1k–10k clients; per-round K ∈ {50,100,200}; dropouts; availability from on/off process; compute/bandwidth heterogeneity.
- Objectives: maximize accuracy/time, respect time budgets, improve worst-client accuracy, operate under DP-SGD noise.
- Notation: see `SELECTION_METHODS.md` common notation (link readers there if supplemental).

## 6. Methods (with intuition and key equations)
- ParetoRL (multi-objective constrained greedy)
  - Score: s_i = w_acc·norm(ℓ_i/ t̂_i) + w_time·norm(1/t̂_i) + w_fair·f_i + w_dp·ε̃_i
  - Safety: include a minimum quota of low-participation clients; greedy packing under budget.
  - Robust to heterogeneity; stable without training.
- DP-EIG (DP-aware info-gain greedy)
  - EIG proxy: ∝ d_i (g_i² + ℓ_i) / (t̂_i (1+σ²)) + λ_cov·coverage_gain − λ_time·budget_penalty
  - Lazy-greedy-style selection; encourages label complementarity and fast clients.
- GNN-DPP (diverse graph scoring + DPP sampling)
  - Attention-aggregated utility from cosine neighborhoods; DPP-style (MMR) diversity term to avoid redundancy.
  - Robustness to non-IID and DP noise by promoting complementary updates.

Implementation paths in repo:
- `csfl_simulator/selection/ml/pareto_rl.py`
- `csfl_simulator/selection/ml/dp_eig.py`
- `csfl_simulator/selection/ml/gnn_dpp.py`

## 7. Experimental Setup
- Datasets: CIFAR-10 (Dirichlet α∈{0.1,0.3,1.0}), EMNIST-62; Shakespeare (LSTM) optional.
- Models: CNN-MNIST / LightCIFAR; MobileNetV2-lite optional.
- Client pool: 1k–10k; budget K ∈ {50,100,200}; dropout ∈ {0.1,0.3,0.5}.
- Heterogeneity: compute tiers (A/B/C) and bandwidth (low/med/high).
- DP: DP-SGD noise σ; RDP accounting; report ε ∈ {1,3,8}; gradient clipping C ∈ {0.5,1,2}.
- Seeds: 3; report mean ± 95% CI.

Repro scripts:
- Use the Streamlit app or directly call `FLSimulator` with different `method_key` and `SimConfig`.
- Outputs stored under `artifacts/runs/<run_id>/metrics.json`.

## 8. Metrics
- Convergence: rounds and wall-clock to reach accuracy targets (70/80/85%).
- Final accuracy; robustness vs ε; AUC over rounds.
- Fairness: worst-client accuracy (approx. per-client eval or proxy), 10th-percentile accuracy, Jain’s index.
- Efficiency: bytes up/down per round (proxy), selected-client-seconds per round.
- Stability: variance across seeds; failure under dropouts.

## 9. Baselines and Our Methods (Keys for Reproducibility)
- Baselines (already in presets): `system_aware.fedcs`, `system_aware.tifl`, `system_aware.oort`, `ml.bandit.epsilon_greedy`, `ml.bandit.linucb`, `heuristic.random`, `heuristic.topk_loss`, etc.
- Added baselines: `ml.fedcor` (approximate).
- Proposed: `ml.pareto_rl`, `ml.dp_eig`, `ml.gnn_dpp`.

## 10. Figures and Tables
- Curves: accuracy vs rounds; accuracy vs wall-clock proxy; fairness vs rounds.
- Pareto fronts: bytes vs accuracy; time-to-X vs worst-client accuracy.
- Ablations: (a) feature contributions; (b) effect of ε and K; (c) diversity vs utility (λ params).
- Tables: wall-clock to reach 80%; final accuracy; worst-client accuracy; ε; communication.

## 11. Results Summary (template)
- Our methods achieve X% faster time-to-80% accuracy (median) and Y% higher worst-client accuracy at ε=3 compared to Oort and FedCS, with comparable or lower communication.

## 12. Discussion and Limitations
- ParetoRL is training-free but not a learned policy; DP-EIG uses proxies for EIG; GNN-DPP uses MMR approximation—not exact k-DPP MAP.
- Nonetheless, all three are practical, stable, and performant under strong constraints.

## 13. Conclusion
- Multi-objective, DP-aware, and diversity-promoting selection is crucial in cross-device FL. The proposed methods provide significant gains in speed and robustness.

## References (primary)
- Oort (NSDI’21) — Guided participant selection.
- FedCS (ICC’19) — Deadline-aware selection.
- TiFL (HPDC’20) — Tier-based FL.
- FedCor (2021) — Correlation-based active selection.
- GreedyFed (2023) — Shapley-approx selection.
- FedGCS (2024) — Generative client selection.
- DP-SGD (CCS’16) — Differential privacy training.
- RDP accounting (Mironov, 2017).

For inline references and details, see `SELECTION_METHODS.md`.

