from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.registry import MethodRegistry


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget: Optional[float] = None,
    device=None,
    *,
    base_key: str = "system_aware.oort_plus",
    window: int = 10,
    q_min: int = 1,
    energy_budget: Optional[float] = None,
    bytes_budget: Optional[float] = None,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """QuotaSafety wrapper.

    Ensures each under-participating client (within a sliding window of rounds) gets
    at least q_min selections. It preselects a set of such clients (up to K), then
    delegates the remaining slots to a base selector.
    """
    if not clients:
        return [], None, {}

    # Build sliding-window participation counts
    recent = history.get("selected", [])[-int(max(0, int(window))):]
    counts: Dict[int, int] = {}
    for ids in recent:
        for cid in ids:
            counts[cid] = counts.get(cid, 0) + 1

    deficit = []
    for c in clients:
        have = int(counts.get(c.id, 0))
        if have < int(q_min):
            deficit.append((c.id, have))
    # Sort by largest deficit (smallest have), tie-break by random
    deficit.sort(key=lambda t: (t[1], rng.random()))

    preselect: List[int] = [cid for cid, _ in deficit[: min(K, len(deficit))]]
    remaining = max(0, K - len(preselect))
    if remaining <= 0:
        return preselect[:K], None, {}

    # Delegate remaining slots to base selector on the remaining pool
    pool = [c for c in clients if c.id not in preselect]
    reg = MethodRegistry(); reg.load_presets()
    ids_rest, scores, state = reg.invoke(
        base_key,
        round_idx,
        remaining,
        pool,
        history,
        rng,
        time_budget,
        device,
        energy_budget=energy_budget,
        bytes_budget=bytes_budget,
        **kwargs,
    )
    all_ids = list(preselect) + list(ids_rest)
    return all_ids[:K], scores, state


