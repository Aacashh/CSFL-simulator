# Selection methods

This directory contains the client selection methods used by the CSFL Simulator. Each method implements the interface:

```
from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo

def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, **params) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """
    Return selected client ids, optional per-client scores, and an optional state update dict.
    """
```

New advanced methods added

- heuristic.mmr_diverse (utility + diversity)
  - Greedy Maximal Marginal Relevance: score(i | S) = λ·utility_i − (1−λ)·max_{j∈S} cos_sim(emb_i, emb_j)
  - Embeddings prefer label histograms if available, else simple client stats
  - Great when you want strong convergence with fewer redundant clients per round

- heuristic.label_coverage (greedy set cover)
  - Maximizes label coverage in each round with scarcity weighting (IDF-like)
  - Optionally mixes in a small utility term to avoid picking only tiny/easy clients
  - Useful on skewed data to accelerate accuracy in fewer rounds

- heuristic.dp_budget_aware
  - Prefers clients with higher DP epsilon remaining by penalizing those close to exhaustion
  - Helps sustain learning quality across more rounds under DP budgets

- system_aware.poc (Power-of-Choice)
  - Two-stage: randomly sample a pool, then rank by utility, speed (1/duration), and recency
  - Optional time_budget-aware greedy packing of top-K within budget
  - Improves round efficiency and convergence speed in heterogeneous systems

- system_aware.oort_plus
  - Builds on Oort (utility/time + exploration) with fairness and recency penalties
  - Stabilizes selection to avoid overusing a few clients, while keeping exploration

- ml.bandit.rff_linucb
  - Contextual bandit using LinUCB on Random Fourier Features (RBFSampler)
  - Captures simple nonlinearities in client utility with minimal overhead

- ml.meta_ranker
  - Lightweight supervised ranker (SGDRegressor) trained online on past rounds’ rewards
  - Predicts per-client reward each round and ranks accordingly; falls back gracefully when cold

Notes and usage

- CPU-first design: all methods avoid GPU dependencies. Two ML methods require scikit-learn.
- State across rounds: use the `history` dict’s "state" key to persist method state; return `{STATE_KEY: state}`.
- Per-client info available (ClientInfo): data_size, last_loss, grad_norm, compute_speed, channel_quality, participation_count, estimated_duration, last_selected_round, label_histogram, dp_epsilon_remaining.
- Time-awareness: For methods supporting `time_budget`, they will favor faster clients when a budget is provided.
- Registration: methods are listed in presets/methods.yaml so they appear in the app’s selection dropdown.

