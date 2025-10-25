from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    # Sort by last_loss descending; fallback to random tie-breakers
    ranked = sorted(clients, key=lambda c: (c.last_loss, rng.random()), reverse=True)
    sel = [c.id for c in ranked[:K]]
    scores = [c.last_loss for c in clients]
    return sel, scores, {}
