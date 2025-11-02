from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget: Optional[float] = None,
    device=None,
    *,
    energy_budget: Optional[float] = None,
    bytes_budget: Optional[float] = None,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """FedCS-style greedy under time + energy (+ optional bytes) budgets.

    - Rank by utility per unit time (loss / duration) with random tie-breakers.
    - Greedily accept if remaining budgets allow.
    - If time_budget is None, behaves like Top-K by utility/time without budget.
    """
    if not clients:
        return [], None, {}

    items = []
    for c in clients:
        util = float(c.last_loss or 0.0) + 1e-9
        t = max(1e-9, expected_duration(c))
        e = float(getattr(c, "estimated_energy", 0.0) or (getattr(c, "energy_rate", 1.0) * t))
        b = float(getattr(c, "estimated_bytes", 0.0) or float(c.data_size or 0.0))
        items.append((c.id, util / t, t, e, b))
    items.sort(key=lambda x: (x[1], rng.random()), reverse=True)

    sel: List[int] = []
    rem_t = float(time_budget) if time_budget is not None else float("inf")
    rem_e = float(energy_budget) if energy_budget is not None else float("inf")
    rem_b = float(bytes_budget) if bytes_budget is not None else float("inf")

    for cid, _, t, e, b in items:
        if len(sel) >= K:
            break
        fits_t = (time_budget is None) or (t <= rem_t)
        fits_e = (energy_budget is None) or (e <= rem_e)
        fits_b = (bytes_budget is None) or (b <= rem_b)
        if fits_t and fits_e and fits_b:
            sel.append(cid)
            rem_t -= t
            rem_e -= e
            rem_b -= b

    # If underfilled, add by ranking ignoring budgets
    if len(sel) < K:
        remaining = [cid for cid, *_ in items if cid not in sel]
        sel.extend(remaining[: (K - len(sel))])

    return sel[:K], None, {}


