### Fairness-Adjusted Client Selection: Logic and Literature Survey

This report documents the implementation of the fairness-adjusted client selection method in this repository, and compares it with prior client-selection methods/papers. It also clarifies how similar ideas appear in the literature and where this specific implementation differs.

---

### 1) Implementation in this repository

The heuristic called "Fairness Adjusted" is implemented as:

```1:15:csfl_simulator/selection/heuristic/fairness_adjusted.py
from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo

LMBDA = 0.1  # fairness penalty weight


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    scores = []
    for c in clients:
        s = (c.last_loss or 0.0) - LMBDA * c.participation_count
        scores.append(s)
    ranked = sorted(range(len(clients)), key=lambda i: (scores[i], rng.random()), reverse=True)
    sel = [clients[i].id for i in ranked[:K]]
    return sel, scores, {}
```

- Score used per client i: s_i = ℓ_i − λ · p_i, where ℓ_i is last_loss and p_i is participation_count; default λ = 0.1.
- Selection: pick top-K by s_i (with a random tie-breaker).
- Intent: retain a utility signal (loss) but discourage repeatedly selecting the same clients, improving participation fairness by promoting under-selected clients over time.

Contrast with the repository’s Oort-Plus variant, which integrates time-awareness, exploration, and fairness/recency penalties:

```11:53:csfl_simulator/selection/system_aware/oort_plus.py
def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, beta: float = 0.5, fairness_gamma: float = 0.3,
                   recency_delta: float = 0.3, half_life_rounds: int = 10, alpha_ucb: float = 0.1,
                   time_awareness: bool = True) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Oort-Plus: base utility/time with fairness and recency penalties and mild UCB.
    - beta: utility exponent for diminishing returns (0..1)
    - fairness_gamma: penalizes over-selected clients (via small gap)
    - recency_delta: extra recency penalty
    - half_life_rounds: exponential decay horizon for recency penalty
    - alpha_ucb: exploration weight (LinUCB-style term)
    - time_awareness: if True, divide score by expected duration
    """
    st = history.get("state", {}).get(STATE, {"N": {}, "t": 0})
    last_reward = history.get("state", {}).get("last_reward", 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []

    # Update counts
    for cid in last_sel:
        st["N"][cid] = int(st["N"].get(cid, 0)) + 1
    st["t"] = int(st.get("t", 0)) + 1

    # Compute scores
    scores = []
    t_global = max(1, st["t"])  # time for UCB
    for c in clients:
        util = float(c.last_loss or 0.0)
        dur = expected_duration(c)
        base = util ** max(0.0, min(1.0, beta))
        if time_awareness:
            base = base / max(1e-6, dur)
        # UCB component for exploration (based on selection counts)
        n_i = int(st["N"].get(c.id, 0))
        ucb = math.sqrt(2.0 * math.log(t_global + 1.0) / (n_i + 1.0))
        # Fairness and recency penalties
        gap = float(recency(round_idx, c))
        fairness_penalty = fairness_gamma * (1.0 / (1.0 + gap))
        recency_penalty = recency_delta * math.exp(-gap / max(1.0, float(half_life_rounds)))
        score = base * (1.0 + alpha_ucb * ucb) / (1.0 + fairness_penalty + recency_penalty)
        scores.append((c.id, score))

    scores.sort(key=lambda t: t[1], reverse=True)
    sel = [cid for cid, _ in scores[:K]]
    return sel, [s for _, s in scores], {STATE: st}
```

Key differences versus fairness_adjusted:
- Adds time-awareness (utility per expected duration) and explicit exploration (UCB over per-client selection counts).
- Uses a recency-based fairness penalty (decays with the gap since last selection), rather than a lifetime participation count.

---

### 2) What kind of fairness does this implement?

- The heuristic directly targets participation fairness by penalizing frequent participants linearly. This tends to reduce the variance of selection counts across clients over many rounds (i.e., more equitable participation), while still exploiting high-loss clients.
- It does not directly pursue statistical fairness in predictions (e.g., minimizing performance disparity across subpopulations); it is purely a participation (scheduling) fairness heuristic.

Practical implications:
- Pros: extremely simple, low-overhead, helps avoid starvation of under-selected clients while keeping a utility signal.
- Cons: no notion of system heterogeneity (time budget, stragglers), no explicit exploration term, and fairness pressure can overwhelm utility if λ is not tuned.

---

### 3) Has this exact method appeared in prior papers?

- I did not find a mainstream paper that uses the exact form s_i = loss_i − λ · participation_count_i as its core client-selection rule.
- Very similar ideas appear in the literature via exploration/fairness terms that down-weight frequently selected clients or up-weight under-used ones:
  - UCB-style exploration terms that decrease with selection counts n_i (e.g., Oort) implicitly counterbalance over-selection.
  - Recency-based penalties or rotations (e.g., tier rotations) also encourage equitable participation.
- In short, the specific linear penalty on lifetime participation count is a natural, simple instantiation of the broader idea “reduce overuse of already-selected clients,” but it is not a widely cited canonical formula by itself.

---

### 4) Comparison to notable client-selection papers

Below, I summarize well-cited selection methods and relate them to this heuristic. Links point to the official preprints referenced in this repository’s documentation.

- **Oort: Efficient Federated Learning via Guided Participant Selection (OSDI 2021)**  — emphasizes utility/time and exploration
  - Paper: [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)
  - Essence: score combines utility per time with a UCB-style exploration bonus that is larger for clients with fewer prior selections. This implicitly pushes fairness by avoiding starvation, but does not use a direct lifetime-count penalty.
  - Relation: conceptually similar in discouraging repeated selections, but Oort’s exploration is sublinear (∝ √(log t / n_i)) and combined with time-awareness; fairness_adjusted is purely linear in participation_count and ignores duration.

- **FedCS: Client Selection for Federated Learning with Heterogeneous Resources (ICC 2019)**  — deadline-aware packing
  - Paper: [arXiv:1909.13014](https://arxiv.org/abs/1909.13014)
  - Essence: choose clients to meet a round time budget while maximizing utility per unit time, focusing on system constraints. Fairness is not an explicit objective.
  - Relation: orthogonal. fairness_adjusted could be seen as a fairness add-on to utility-based rules used in FedCS; however, FedCS does not include a participation penalty.

- **TiFL: A Tier-based Federated Learning System (HPDC 2020)**  — cross-tier fairness and straggler awareness
  - Paper: [arXiv:2001.09243](https://arxiv.org/abs/2001.09243)
  - Essence: group clients into resource tiers and rotate among tiers to mitigate stragglers and improve fairness across system classes (slow vs. fast). Fairness is implemented via tier scheduling/rotation rather than client-level lifetime penalties.
  - Relation: both aim to avoid consistently excluding certain clients; TiFL does so structurally (across tiers), fairness_adjusted does so via a per-client penalty on repeated participation.

- **Oort-Plus (this repository)**  — utility/time + UCB + fairness/recency penalties
  - Code: `csfl_simulator/selection/system_aware/oort_plus.py` (see reference above)
  - Essence: extends Oort with explicit fairness (recency) penalty and a time-aware base score, plus mild UCB. This is closer in spirit to “fairness adjusted” but uses recency rather than a lifetime count and is normalized by time.
  - Relation: compared to fairness_adjusted, Oort-Plus tends to be more stable in heterogeneous systems and less sensitive to the single λ hyperparameter.

Notes on fairness beyond selection:
- Other works (e.g., optimizing fairness in the training objective like Q-FFL or AFL) address statistical fairness or fairness across client performance, but they are not client-selection algorithms per se; they can be combined with selection rules like the one here. These are outside the scope of this brief since they change the optimization objective rather than the selector.

---

### 5) Similarities and differences at a glance

- **Shared principle:** discourage overusing the same clients to reduce participation imbalance.
- **Form of the correction:**
  - This heuristic: subtract λ · participation_count (linear, lifetime-based).
  - Oort: add exploration ∝ √(log t / n_i) (diminishing returns, count-based, but as a bonus to utility/time).
  - Oort-Plus: penalize small recency gaps and include UCB, normalized by expected duration.
  - TiFL: fairness via tier rotation (structure), not a numeric penalty.
- **System awareness:**
  - fairness_adjusted: none (no duration/budget awareness).
  - Oort/FedCS/TiFL: explicitly system-aware (time budget, stragglers, tiers).

---

### 6) Practical guidance

- If you need a minimal fairness nudge with no extra bookkeeping, fairness_adjusted is a good starting point; tune λ to balance convergence vs. participation equity.
- In heterogeneous settings or when round time matters, prefer Oort(-Plus) or FedCS-like methods; you can still add a small fairness/recency correction.
- For strict participation balancing, Round-Robin ensures fairness but sacrifices utility; fairness_adjusted is a middle ground.

---

### References (client selection)

- Oort: Efficient Federated Learning via Guided Participant Selection (OSDI 2021). [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)
- FedCS: Client Selection for Federated Learning with Heterogeneous Resources (ICC 2019). [arXiv:1909.13014](https://arxiv.org/abs/1909.13014)
- TiFL: A Tier-based Federated Learning System (HPDC 2020). [arXiv:2001.09243](https://arxiv.org/abs/2001.09243)

These are the primary, widely cited client-selection papers that relate closely to the themes embodied by fairness_adjusted (utility-vs-fairness trade-offs and avoiding client starvation). Where possible, this report references their official preprints.
