from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from csfl_simulator.core.client import ClientInfo

STATE_KEY = "tifl_tier_ptr"


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    tiers = defaultdict(list)
    for c in clients:
        tiers[int(c.tier or 0)].append(c.id)
    for t in tiers:
        tiers[t].sort()
    order = sorted(tiers.keys())
    ptr = history.get("state", {}).get(STATE_KEY, 0)
    sel = []
    while len(sel) < K and order:
        t = order[ptr % len(order)]
        if tiers[t]:
            sel.append(tiers[t].pop(0))
        ptr += 1
        # prevent infinite loop if tiers empty
        if all(len(v) == 0 for v in tiers.values()):
            break
    # fill remaining randomly
    if len(sel) < K:
        rest = [c.id for c in clients if c.id not in sel]
        rng.shuffle(rest)
        sel.extend(rest[:K-len(sel)])
    return sel[:K], None, {STATE_KEY: ptr}
