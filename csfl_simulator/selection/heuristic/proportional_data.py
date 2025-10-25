from typing import List, Dict, Optional, Tuple
import random as pyrand
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    sizes = [max(1, c.data_size) for c in clients]
    total = float(sum(sizes))
    probs = [s/total for s in sizes]
    ids = [c.id for c in clients]
    # sample without replacement proportional to probs
    chosen = []
    available = list(range(len(ids)))
    p = probs[:]
    for _ in range(min(K, len(ids))):
        r = rng.random()
        cum = 0.0
        pick = 0
        for i, idx in enumerate(available):
            cum += p[idx]
            if r <= cum:
                pick = idx
                break
        chosen.append(ids[pick])
        # remove picked
        removed = available.index(pick)
        available.pop(removed)
        # renormalize remaining
        rem_sum = sum(p[idx] for idx in available) or 1.0
        for idx in available:
            p[idx] = p[idx]/rem_sum
    scores = probs
    return chosen, scores, {}
