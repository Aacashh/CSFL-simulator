"""In-simulator reproduction of CriticalFL's random client selection.

CriticalFL augments a base FL selector during critical learning periods.  Its
paper uses the relative change of the federated gradient norm (FGN) and grows
the participating cohort geometrically while the critical period persists.
The experiment simulator performs the post-training FGN update, geometric
cohort adjustment, and critical-period update sparsification. This selector
implements Algorithm 2's random subset selection using the resulting cohort.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from csfl_simulator.core.client import ClientInfo


STATE_KEY = "research_criticalfl_state"


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    *,
    delta: float = 0.01,
    growth_factor: float = 2.0,
    sparse_fraction: float = 0.20,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    state = history.get("state", {}).get(
        STATE_KEY,
        {"previous_fgn": None, "cohort_size": int(K), "critical_rounds": 0},
    )
    if not clients:
        return [], None, {STATE_KEY: state}

    cohort_size = min(len(clients), max(1, int(state.get("cohort_size", K))))
    ids = [client.id for client in clients]
    rng.shuffle(ids)
    selected = ids[:cohort_size]
    state.update(
        {
            "cohort_size": cohort_size,
            "delta": float(delta),
            "growth_factor": float(growth_factor),
            "sparse_fraction": float(sparse_fraction),
        }
    )
    return selected, None, {STATE_KEY: state}
