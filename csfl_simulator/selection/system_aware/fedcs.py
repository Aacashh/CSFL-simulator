from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo

# Greedy knapsack by utility per unit time

def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    if time_budget is None:
        # fallback to top-K loss
        ranked = sorted(clients, key=lambda c: (c.last_loss, rng.random()), reverse=True)
        return [c.id for c in ranked[:K]], None, {}
    items = []
    for c in clients:
        util = (c.last_loss or 0.0) + 1e-6
        t = max(1e-6, c.estimated_duration or 1.0)
        items.append((c.id, util/t, t))
    items.sort(key=lambda x: x[1], reverse=True)
    sel = []
    budget = float(time_budget)
    for cid, ratio, t in items:
        if len(sel) >= K:
            break
        if t <= budget:
            sel.append(cid)
            budget -= t
    if len(sel) < K:
        remaining = [c.id for c in clients if c.id not in sel]
        rng.shuffle(remaining)
        sel.extend(remaining[:K-len(sel)])
    return sel[:K], None, {}
