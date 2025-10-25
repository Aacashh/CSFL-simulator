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
