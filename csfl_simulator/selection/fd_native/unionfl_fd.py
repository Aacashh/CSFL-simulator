"""Optional FD port of UnionFL using recent effective-participation history."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from .submodular_utils import (
    cosine_facility_matrix,
    greedy_facility_location,
    select_for_signal_exploration,
    selector_output,
)


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    overlap_penalty: float = 0.1,
    history_window: int = 5,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict[str, Any]]]:
    if not clients or K <= 0:
        return [], None, {}
    K = min(K, len(clients))
    selected, exploring = select_for_signal_exploration(clients, K, history, rng)
    gains = {}
    recent = set()
    if not exploring:
        source = history.get("responded") or history.get("selected", [])
        for ids in source[-max(1, int(history_window)):]:
            recent.update(int(cid) for cid in ids)
        ids = [c.id for c in clients]
        similarity = cosine_facility_matrix(ids, history)
        penalty = max(0.0, float(overlap_penalty))
        selected, gains = greedy_facility_location(
            ids,
            similarity,
            K,
            extra_marginal=lambda chosen, cid: -penalty if cid in recent else 0.0,
        )
    diag = {
        "exploration_round": exploring,
        "recent_union_size": len(recent),
        "history_window": int(history_window),
    }
    return selected, selector_output(clients, selected, gains), {"unionfl_fd": diag}
