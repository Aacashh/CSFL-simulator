from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo

STATE_KEY = "round_robin_idx"


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    ids = [c.id for c in clients]
    ids.sort()
    idx = history.get("state", {}).get(STATE_KEY, 0)
    sel = []
    for i in range(K):
        sel.append(ids[(idx + i) % len(ids)])
    idx = (idx + K) % len(ids)
    return sel, None, {STATE_KEY: idx}
