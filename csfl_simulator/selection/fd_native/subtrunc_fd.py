"""FD adaptation of SubTrunc with public-set client loss.

The representativeness term is DivFL's facility-location objective over public
logits. The fairness term follows SubTrunc:
    lambda * min(b, sum_i log(1 + public_loss_i)).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from .submodular_utils import (
    cosine_facility_matrix,
    greedy_facility_location,
    select_for_signal_exploration,
    selector_output,
    signal_for,
)


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    lambda_loss: float = 0.25,
    truncation_b: float = 1.0,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict[str, Any]]]:
    if not clients or K <= 0:
        return [], None, {}
    K = min(K, len(clients))
    selected, exploring = select_for_signal_exploration(clients, K, history, rng)
    gains = {}
    observed_losses = 0
    if not exploring:
        ids = [c.id for c in clients]
        similarity = cosine_facility_matrix(ids, history)
        phi = {}
        for c in clients:
            signal = signal_for(history, c.id)
            public_loss = signal.get("public_loss") if isinstance(signal, dict) else None
            if public_loss is not None:
                observed_losses += 1
                value = max(0.0, float(public_loss))
            else:
                value = max(0.0, float(c.last_loss))
            phi[c.id] = math.log1p(value)

        cap = max(0.0, float(truncation_b))
        weight = max(0.0, float(lambda_loss))

        def loss_marginal(chosen, cid):
            before = min(cap, sum(phi[x] for x in chosen))
            after = min(cap, before + phi[cid])
            return weight * (after - before)

        selected, gains = greedy_facility_location(
            ids,
            similarity,
            K,
            extra_marginal=loss_marginal,
        )

    diag = {
        "exploration_round": exploring,
        "observed_public_losses": observed_losses,
        "lambda_loss": float(lambda_loss),
        "truncation_b": float(truncation_b),
    }
    return selected, selector_output(clients, selected, gains), {"subtrunc_fd": diag}
