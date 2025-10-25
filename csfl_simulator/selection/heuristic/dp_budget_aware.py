from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, penalty_eta: float = 1.0,
                   required_epsilon: Optional[float] = None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """DP-budget-aware utility selector.
    Score = utility / (1 + penalty(dp_epsilon_remaining)).
    - penalty_eta: scaling for DP penalty
    - required_epsilon: if set, penalize shortfall relative to this threshold
    """
    if not clients:
        return [], None, {}
    n = len(clients)

    # Base utility: normalized (loss / expected_duration)
    base = []
    for c in clients:
        util = float(c.last_loss or 0.0) / max(1e-6, expected_duration(c))
        base.append(util)
    base_n = normalize_list(base)

    scores = []
    for i, c in enumerate(clients):
        remaining = float(getattr(c, "dp_epsilon_remaining", 0.0) or 0.0)
        if required_epsilon is not None and required_epsilon > 0:
            shortfall = max(0.0, required_epsilon - remaining)
            penalty = penalty_eta * (shortfall / (required_epsilon + 1e-8))
        else:
            penalty = penalty_eta * (1.0 / (1.0 + remaining))  # lower remaining => higher penalty
        final = base_n[i] / (1.0 + penalty)
        scores.append(final)

    ranked = sorted(range(n), key=lambda i: (scores[i], rng.random()), reverse=True)
    sel = [clients[i].id for i in ranked[:K]]
    return sel, scores, {}

