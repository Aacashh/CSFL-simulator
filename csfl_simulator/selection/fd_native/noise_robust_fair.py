"""Noise-Robust Fair Scheduler (NRFS) for Federated Distillation.

Round-robin backbone with channel-aware deferral. Guarantees near-equal
participation (exploiting the fairness-accuracy correlation in FD) while
avoiding transmitting through the worst channel states.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # --- NRFS hyperparameters ---
    max_defer: int = 3,
    channel_percentile: float = 20.0,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    state = history.get("state", {}).get("nrfs", {})
    deficit = state.get("deficit", {})
    defer_count = state.get("defer_count", {})

    # --- Step 1: Update deficits ---
    expected_rate = 1.0 / max(K, 1)
    for c in clients:
        cid = str(c.id)
        deficit[cid] = deficit.get(cid, 0.0) + expected_rate

    # --- Step 2: Channel threshold (bottom percentile) ---
    channel_vals = [c.channel_quality for c in clients]
    threshold = float(np.percentile(channel_vals, channel_percentile))

    # --- Step 3: Sort by deficit (priority) ---
    client_priority = []
    for c in clients:
        cid = str(c.id)
        client_priority.append((deficit.get(cid, 0.0), c))
    client_priority.sort(key=lambda x: -x[0])  # highest deficit first

    # --- Step 4: Greedy selection with deferral ---
    selected = []
    deferred = []
    scores_out = []

    for priority, c in client_priority:
        if len(selected) >= K:
            break

        cid = str(c.id)
        dc = defer_count.get(cid, 0)

        if c.channel_quality < threshold and dc < max_defer:
            # Defer: bad channel and not yet hit max deferral
            defer_count[cid] = dc + 1
            deferred.append((priority, c))
        else:
            # Select: good channel or fairness override
            selected.append(c.id)
            scores_out.append(priority)
            deficit[cid] = deficit.get(cid, 0.0) - 1.0
            defer_count[cid] = 0

    # --- Step 5: Backfill if needed ---
    if len(selected) < K:
        # Fill from deferred clients by priority
        deferred.sort(key=lambda x: -x[0])
        for priority, c in deferred:
            if len(selected) >= K:
                break
            if c.id not in selected:
                selected.append(c.id)
                scores_out.append(priority)
                cid = str(c.id)
                deficit[cid] = deficit.get(cid, 0.0) - 1.0
                defer_count[cid] = 0

    new_state = {
        "nrfs": {
            "deficit": deficit,
            "defer_count": defer_count,
        }
    }

    return selected, scores_out, new_state
