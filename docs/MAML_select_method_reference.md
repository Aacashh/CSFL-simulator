# MAML-Select: Meta-Adaptive MLP for Client Selection in Federated Learning

## Overview

MAML-Select is a **trainable, online-learning** client selection method for federated learning. It uses a small multi-layer perceptron (MLP) to predict each client's expected contribution to the global model, adapting rapidly via a few gradient steps each round (inspired by Model-Agnostic Meta-Learning). The method maintains an experience buffer and performs lightweight inner-loop adaptation, enabling it to track shifting client utility over the course of training.

**Key:** `ml.maml_select`
**Category:** ML-based (trainable)
**Module:** `csfl_simulator.selection.ml.maml_select`

---

## Algorithm

### Notation

| Symbol | Description |
|--------|-------------|
| $N$ | Total number of available clients |
| $K$ | Number of clients to select per round |
| $t$ | Current round index |
| $\mathbf{x}_i^{(t)}$ | Feature vector for client $i$ at round $t$ |
| $f_\theta$ | MLP with parameters $\theta$ |
| $\mathcal{B}$ | Experience replay buffer of $(\mathbf{x}, r)$ pairs |
| $H$ | Maximum buffer size (`max_history`) |
| $J$ | Number of inner adaptation steps (`inner_steps`) |
| $\eta$ | Learning rate (`lr`) |
| $\alpha$ | Exploration coefficient (`exploration_ucb`) |
| $r^{(t)}$ | Reward signal at round $t$ |
| $\Delta_i^{(t)}$ | Recency gap: rounds since client $i$ was last selected |

### Input Features

For each client $i$, a 6-dimensional feature vector is constructed:

$$\mathbf{x}_i = \left[ d_i,\ \ell_i,\ g_i,\ s_i^{-1},\ p_i,\ e_i \right]$$

where:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Data size | $d_i$ | Number of local training samples |
| Last loss | $\ell_i$ | Loss reported after last local training |
| Gradient norm | $g_i$ | Norm of the gradient from last training |
| Inverse duration | $s_i^{-1}$ | $1 / \max(\epsilon, \hat{T}_i)$ where $\hat{T}_i$ is estimated round time |
| Participation count | $p_i$ | Number of rounds client $i$ has been selected |
| Label entropy | $e_i$ | Shannon entropy of the client's label distribution |

**Standardization:** Features are z-score normalized per round (zero mean, unit variance) across all available clients:

$$\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \mu}{\sigma + \epsilon}$$

where $\mu$ and $\sigma$ are computed over the current client pool.

### Scoring Network (MLP)

The contribution predictor is a 3-layer MLP:

$$f_\theta(\mathbf{x}) : \mathbb{R}^6 \to \mathbb{R}$$

Architecture:
```
Linear(6, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
```

Output is a scalar predicted contribution score for each client. The network is optimized with Adam.

### Reward Signal

The reward $r^{(t)}$ is the **round-over-round improvement in composite score**:

$$r^{(t)} = C^{(t)} - C^{(t-1)}$$

The composite score $C^{(t)}$ is a weighted combination:

$$C^{(t)} = w_\text{acc} \cdot \text{acc}^{(t)} + w_\text{time} \cdot \phi(\text{time}^{(t)}) + w_\text{fair} \cdot \phi(\text{fair\_var}^{(t)}) + w_\text{dp} \cdot \phi(\text{dp\_used}^{(t)})$$

where $\phi(v) = 1 - \frac{v}{v + 1}$ is a diminishing-returns normalization, and the default weights are $w_\text{acc}=0.6$, $w_\text{time}=0.2$, $w_\text{fair}=0.1$, $w_\text{dp}=0.1$.

This reward is **shared equally** across all clients that were selected in the previous round (credit assignment is uniform).

### Experience Buffer

The buffer $\mathcal{B}$ stores $(\mathbf{x}_i, r)$ tuples for every client selected in every previous round:

- After each round, for each selected client $i$: append $(\tilde{\mathbf{x}}_i^{(t)}, r^{(t)})$ to $\mathcal{B}$
- If $|\mathcal{B}| > H$, discard the oldest entries (FIFO)

### Inner-Loop Adaptation (MAML-Inspired)

Each round, the MLP is updated via $J$ gradient steps on the full buffer (or as much as available):

$$\text{For } j = 1, \ldots, J: \quad \theta \leftarrow \theta - \eta \nabla_\theta \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{x}, r) \in \mathcal{B}} \left( f_\theta(\mathbf{x}) - r \right)^2$$

This adaptation is triggered only when $|\mathcal{B}| \geq \max(10, K)$ (enough data to learn from). The loss function is **mean squared error** between predicted scores and observed rewards.

Note: Unlike full MAML, there is no outer-loop or task distribution. The "meta" aspect is that the MLP maintains a persistent initialization across rounds and performs rapid few-shot adaptation each round, allowing it to quickly adjust to changing client dynamics.

### Exploration via UCB-Style Recency Bonus

After computing MLP predictions $\hat{y}_i = f_\theta(\tilde{\mathbf{x}}_i)$, a recency-based exploration bonus is added:

$$\text{score}_i = \hat{y}_i + \alpha \cdot \frac{\Delta_i^{(t)}}{\Delta_i^{(t)} + 5}$$

where:
- $\Delta_i^{(t)} = t - t_i^{\text{last}}$ is the number of rounds since client $i$ was last selected
- The term $\frac{\Delta}{\ \Delta + 5}$ saturates at 1 for large gaps, providing a bounded exploration boost
- $\alpha$ (`exploration_ucb`, default 0.2) controls the exploration-exploitation trade-off

This encourages selecting clients that haven't been seen recently, preventing starvation and promoting diverse information gathering.

### Selection

The final selection is **greedy top-K** on the augmented scores:

$$S^{(t)} = \underset{|S|=K}{\arg\text{top-}K}\ \text{score}_i$$

---

## Full Algorithm (Pseudocode)

```
Algorithm: MAML-Select
Input: clients C, K, round t, history, hyperparameters (H, J, eta, alpha)

1.  Compute features X = [x_1, ..., x_N] for all clients
2.  Z-score normalize: X <- (X - mean(X)) / (std(X) + eps)

3.  If first round:
4.      Initialize MLP f_theta with random weights
5.      Initialize Adam optimizer, empty buffer B

6.  // --- Online adaptation ---
7.  r <- history.last_reward  (composite improvement from previous round)
8.  S_prev <- clients selected last round
9.  For each client i in S_prev:
10.     Append (x_i, r) to buffer B
11. If |B| > H: trim oldest entries

12. If |B| >= max(10, K):
13.     For j = 1 to J:
14.         loss = MSE(f_theta(B.X), B.y)
15.         theta <- theta - eta * grad(loss)   [Adam step]

16. // --- Scoring ---
17. For each client i:
18.     y_hat_i = f_theta(x_i)              [MLP prediction]
19.     delta_i = t - last_selected_round(i) [recency gap]
20.     score_i = y_hat_i + alpha * delta_i / (delta_i + 5)

21. // --- Selection ---
22. Return top-K clients by score_i
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_history` | 5000 | Maximum experience buffer size $H$ |
| `inner_steps` | 3 | Adaptation gradient steps $J$ per round |
| `lr` | 0.001 | Adam learning rate $\eta$ |
| `exploration_ucb` | 0.2 | Recency exploration coefficient $\alpha$ |

### Network Architecture (Fixed)

| Layer | Dimensions |
|-------|-----------|
| Input | 6 (feature dim) |
| Hidden 1 | 64, ReLU |
| Hidden 2 | 64, ReLU |
| Output | 1 (scalar score) |

**Optimizer:** Adam (PyTorch default betas)

---

## Design Rationale

### Why "MAML-Inspired"?

Standard MAML learns an initialization $\theta_0$ from a distribution of tasks such that a few gradient steps on a new task yield good performance. MAML-Select borrows this principle in a different setting:

- **Persistent initialization:** The MLP carries forward its parameters from round to round, building cumulative knowledge about client-reward relationships.
- **Few-shot adaptation:** Each round, only $J$ (default 3) gradient steps are taken, enabling rapid adjustment to the latest reward signal without catastrophic forgetting of past patterns.
- **Non-stationary environment:** Client utilities shift as the global model trains (e.g., a client with high loss early on may become less informative later). The few-step adaptation allows the model to track these dynamics.

### Exploration-Exploitation Balance

The UCB-style recency bonus ensures that:
- Clients that haven't been observed recently get a boost, preventing the MLP from permanently ignoring clients whose features looked unfavorable in earlier rounds.
- The bonus is bounded (saturates at $\alpha$), so it cannot override strong MLP signals.
- The denominator constant (5) controls how quickly the bonus ramps up; within ~5 rounds of non-selection, a client receives roughly half the maximum bonus.

### Uniform Credit Assignment

All clients selected in a round receive the same reward (the composite improvement). This is a simplification:
- **Pro:** Simple, no need for per-client counterfactual estimation.
- **Con:** Cannot distinguish which specific clients drove improvement.
- **Mitigating factor:** Over many rounds, clients that are consistently selected in high-reward rounds will accumulate a pattern in the buffer that the MLP can learn, even under uniform credit.

---

## Computational Complexity

| Component | Complexity per Round |
|-----------|---------------------|
| Feature extraction | $O(N)$ |
| Z-score normalization | $O(N \cdot d)$ where $d=6$ |
| Buffer append | $O(K)$ |
| Inner-loop training | $O(J \cdot |\mathcal{B}| \cdot P)$ where $P$ = MLP params (4,545) |
| MLP inference | $O(N \cdot P)$ |
| Top-K selection | $O(N \log N)$ (argsort) |
| **Total** | $O(J \cdot H \cdot P + N \cdot P)$ |

The MLP has $6 \times 64 + 64 + 64 \times 64 + 64 + 64 \times 1 + 1 = 4{,}545$ parameters, making training extremely lightweight.

---

## Usage

### CLI
```bash
python -m csfl_simulator run --name maml_experiment \
    --method ml.maml_select \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 30
```

### Programmatic
```python
from csfl_simulator.core.simulator import FLSimulator, SimConfig

cfg = SimConfig(dataset="CIFAR-10", partition="dirichlet", dirichlet_alpha=0.3,
                model="LightCNN", total_clients=50, clients_per_round=10, rounds=30)
sim = FLSimulator(cfg)
sim.setup()
result = sim.run("ml.maml_select")
```

### Custom Hyperparameters (via methods.yaml)
```yaml
- key: ml.maml_select
  module: csfl_simulator.selection.ml.maml_select
  display_name: "MAML-Select (adaptive MLP)"
  origin: custom
  params: {max_history: 5000, inner_steps: 3, lr: 0.001, exploration_ucb: 0.2}
  type: ml
  trainable: true
```
