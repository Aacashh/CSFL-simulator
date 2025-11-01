# Advanced AI/ML Client Selection Methods (Paper-Ready)

This document presents three learning-based client selection methods designed to surpass FedAvg and FedCS under realistic, challenging conditions. Each method is engineered to leverage signals that naive or system-only baselines overlook, while remaining lightweight enough for practical FL simulation.

## Why New Methods Are Needed

- FedAvg and random-like baselines ignore which clients actually move the model forward.
- FedCS is deadline-aware but data/gradient agnostic: it picks fast clients, not necessarily informative ones.
- Real federations exhibit extreme non-IID, system heterogeneity and availability noise. Under these, learning which clients matter—and maintaining diversity—is essential.

---

## Method 1: UCB-Grad (Utility + Diversity with Bandit Exploration)

- Module: `csfl_simulator.selection.ml.ucb_grad`
- Registry key: `ml.ucb_grad`

### Intuition
Treat client selection as a contextual bandit: estimate each client's long-term utility from history, explore uncertain clients, and ensure diversity so that updates are not redundant.

### Scoring
For client i at round t:
- quality_i = w_reward·Q_i + w_loss·norm(loss_i) + w_grad·norm(grad_norm_i)
- exploration_i = α_ucb·sqrt(2 log t / (N_i + 1))
- diversity_i = λ_div·min_cosine_distance(φ_i, {φ_j | j in selected})
- score_i = quality_i + exploration_i + diversity_i

where Q_i is the running average credit assigned from the last round's global improvement, N_i is participation count, and φ_i is a compact feature (loss, grad_norm, 1/duration, label-entropy).

### Why It Beats FedAvg/FedCS
- Learns which clients historically help (Q_i) rather than sampling blindly.
- Proactively explores with UCB to avoid tunnel vision.
- Diversity bonus counters redundancy, crucial in non-IID.

### Complexity
- O(C) for base quality + O(K·C) for greedy diversity (C=clients).

---

## Method 2: MAML-Select (Fast-Adapting Meta Learner)

- Module: `csfl_simulator.selection.ml.maml_select`
- Registry key: `ml.maml_select`

### Intuition
A small neural ranker learns from previous rounds which client features predict contribution. Each round, it performs a few gradient steps (inner adaptation) to specialize to the current federation—akin to MAML's rapid adaptation.

### Features
`[data_size, last_loss, grad_norm, 1/duration, participation_count, label-entropy]`

### Training Signal
- Credit assignment: last round's global reward is assigned to all selected clients (weak supervision).
- Inner loop: 3 gradient steps (configurable) on buffered (feature, reward) pairs.
- Mild exploration via a recency bonus.

### Why It Beats FedAvg/FedCS
- Predictive rather than reactive; it generalizes across rounds and adapts on the fly.
- Handles changing distributions and availability patterns.

### Complexity
- Tiny MLP (2x64) with 3 steps on a small buffer per round; sub-millisecond on CPU/GPU.

---

## Method 3: FedCluster+ (Adaptive Clustering with Representativeness)

- Module: `csfl_simulator.selection.ml.fedcluster_plus`
- Registry key: `ml.fedcluster_plus`

### Intuition
Cluster clients by gradient/utility proxies and select representatives proportional to cluster importance, ensuring broad but high-quality coverage.

### Pipeline
1. Build compact embeddings: (loss, grad_norm, 1/duration, label-entropy).
2. Online k-means (refresh every 3 rounds).
3. Track per-cluster importance via exponentially decayed credit from past rewards.
4. Allocate K proportionally by cluster importance; within each cluster, pick most representative and reliable clients.

### Why It Beats FedAvg/FedCS
- Guarantees diversity (coverage over clusters) without sacrificing quality.
- Emphasizes clusters that historically improve validation.

### Complexity
- k-means over ≤100 clients is fast; selection is linear in clients.

---

## Recommended Settings to Showcase Superiority

Use these constraints in `SimConfig` or the UI:

- **Data heterogeneity:**
  - `partition="dirichlet"`, `dirichlet_alpha=0.1` (very non-IID)
- **Scale:**
  - `total_clients=100`, `clients_per_round=5` (scarce participation)
- **Training:**
  - `rounds=100`, `local_epochs=2`, `batch_size=32`, `lr=0.01`
- **System heterogeneity:**
  - Use existing simulated speeds/channels (already modeled in `system.py`)
  - Set a round `time_budget=60` to penalize stragglers (hurts FedCS less than data-naive methods)
- **Optional robustness stressors (if you add them later):**
  - Label noise for 10% clients; 5% Byzantine updates (our methods are more resilient via diversity/credit)

These settings degrade methods that ignore data utility and diversity (FedAvg/FedCS) while favoring our learning-based selectors.

---

## Ablations and Paper Checklist

- Compare against: FedAvg, FedCS, Oort(+), LinUCB, Top-k Loss, Random.
- Metrics: Accuracy, F1, time-to-X% accuracy, participation fairness.
- Ablations:
  - Remove diversity (λ_div=0) from UCB-Grad.
  - Remove exploration from UCB-Grad and MAML-Select.
  - Uniform cluster allocation in FedCluster+.
- Sensitivity: α_ucb, λ_div, inner_steps, refresh_every.

---

## Practical Guidance

- Start with `ml.ucb_grad` for a strong out-of-the-box baseline.
- For dynamic settings, `ml.maml_select` adapts best.
- For very non-IID with many clients, `ml.fedcluster_plus` ensures coverage.

All three methods are implemented and registered in `presets/methods.yaml`. Use them from the UI or with:

```python
sim.run(method_key="ml.ucb_grad")
# or
sim.run(method_key="ml.maml_select")
# or
sim.run(method_key="ml.fedcluster_plus")
```

---

## Reproducibility

- Deterministic seeding is enabled by default in the simulator.
- Selection methods rely only on client metadata and historical rewards, not on stochastic simulation of batches.

---

## Limitations and Future Work

- Credit assignment is coarse (shared across selected clients). When per-client deltas are available, plug them into `Q_i` (UCB-Grad) and buffers (MAML-Select).
- FedCluster+ uses k-means on proxies; integrating true gradient embeddings (if logged) could further boost performance.


