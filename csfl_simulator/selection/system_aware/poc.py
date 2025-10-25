from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration, recency


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, over_sample_factor: float = 4.0,
                   weights: Optional[Dict[str, float]] = None, min_pool: Optional[int] = None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Power-of-Choice (two-stage) selector.
    Stage 1: randomly sample a candidate pool of size m = ceil(K*over_sample_factor).
    Stage 2: rank by weighted sum of utility, speed (1/duration), and recency, then pick top-K.
    If time_budget provided, greedily keep the top-ranked while fitting in budget.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    if weights is None:
        weights = {"utility": 0.5, "speed": 0.3, "recency": 0.2}
    if min_pool is None:
        min_pool = max(K + 2, int(K * over_sample_factor))
    m = min(n, max(K, min_pool))

    # Stage 1: random pool
    idxs = list(range(n))
    rng.shuffle(idxs)
    pool = idxs[:m]

    # Build components on pool
    util = []
    speed = []
    recs = []
    for i in pool:
        c = clients[i]
        util.append(float(c.last_loss or 0.0))  # larger loss => potentially higher utility
        speed.append(1.0 / max(1e-6, expected_duration(c)))
        recs.append(float(recency(round_idx, c)))
    util_n = normalize_list(util)
    speed_n = normalize_list(speed)
    rec_n = normalize_list(recs)

    pool_scores = []
    for j, i in enumerate(pool):
        s = (weights.get("utility", 0.5) * util_n[j] +
             weights.get("speed", 0.3) * speed_n[j] +
             weights.get("recency", 0.2) * rec_n[j])
        pool_scores.append((i, s))
    pool_scores.sort(key=lambda t: t[1], reverse=True)

    # Stage 2: respect time budget if given
    selected = []
    used_time = 0.0
    if time_budget is not None:
        for i, s in pool_scores:
            dur = expected_duration(clients[i])
            if used_time + dur <= float(time_budget) + 1e-9:
                selected.append(i)
                used_time += dur
            if len(selected) >= K:
                break
        if len(selected) < K:
            # Fill without budget (best-effort)
            for i, s in pool_scores:
                if i not in selected:
                    selected.append(i)
                if len(selected) >= K:
                    break
    else:
        selected = [i for i, _ in pool_scores[:K]]

    sel_ids = [clients[i].id for i in selected]
    return sel_ids, None, {}

