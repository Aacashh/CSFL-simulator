from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, recency

STATE = "oort_plus_state"


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

