"""FD adaptation of DivFL using public-set logit vectors.

The original DivFL facility-location objective operates in client-gradient
space. Federated distillation does not exchange gradients, so this registered
baseline substitutes the flattened public-set logit vector already submitted by
each client. Selection explores clients until all representations are observed,
then applies the original greedy facility-location structure.
"""
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
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict[str, Any]]]:
    if not clients or K <= 0:
        return [], None, {}
    K = min(K, len(clients))
    selected, exploring = select_for_signal_exploration(clients, K, history, rng)
    gains = {}
    if not exploring:
        ids = [c.id for c in clients]
        similarity = cosine_facility_matrix(ids, history)
        selected, gains = greedy_facility_location(ids, similarity, K)
    diag = {"exploration_round": exploring, "representations": len(clients) if not exploring else None}
    return selected, selector_output(clients, selected, gains), {"divfl_fd": diag}
