from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    ids = [c.id for c in clients]
    rng.shuffle(ids)
    sel = ids[:K]
    scores = [0.0 for _ in clients]
    return sel, scores, {}
