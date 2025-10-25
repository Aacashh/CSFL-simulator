from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    ranked = sorted(clients, key=lambda c: (c.grad_norm, rng.random()), reverse=True)
    sel = [c.id for c in ranked[:K]]
    scores = [c.grad_norm for c in clients]
    return sel, scores, {}
