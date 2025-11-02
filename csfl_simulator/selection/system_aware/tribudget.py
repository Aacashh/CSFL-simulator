from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, normalize_list


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
    a_time: float = 1.0,
    b_energy: float = 1.0,
    c_bytes: float = 1.0,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Multi-constraint greedy packing by density.

    - density(i) = util_i / (a*dur_i + b*energy_i + c*bytes_i)
    - Accept while remaining budgets allow (time, energy, bytes). If a budget is None, it is ignored.
    - Utility defaults to last_loss; duration uses expected_duration.
    - If packing yields <K, fill by density ignoring budgets (best-effort).
    """
    if not clients:
        return [], None, {}

    n = len(clients)
    util = [float(c.last_loss or 0.0) for c in clients]
    dur = [expected_duration(c) for c in clients]
    eng = [float(getattr(c, "estimated_energy", 0.0) or (getattr(c, "energy_rate", 1.0) * d)) for c, d in zip(clients, dur)]
    byt = [float(getattr(c, "estimated_bytes", 0.0) or float(c.data_size or 0.0)) for c in clients]

    # Normalize denominators to mitigate scale imbalance
    dur_n = normalize_list(dur)
    eng_n = normalize_list(eng)
    byt_n = normalize_list(byt)

    scores = []
    for i in range(n):
        denom = (
            a_time * (dur_n[i] if dur_n else 0.0)
            + b_energy * (eng_n[i] if eng_n else 0.0)
            + c_bytes * (byt_n[i] if byt_n else 0.0)
            + 1e-9
        )
        s = util[i] / denom
        scores.append((i, s))
    scores.sort(key=lambda t: t[1], reverse=True)

    sel: List[int] = []
    rem_t = float(time_budget) if time_budget is not None else float("inf")
    rem_e = float(energy_budget) if energy_budget is not None else float("inf")
    rem_b = float(bytes_budget) if bytes_budget is not None else float("inf")

    for i, _ in scores:
        if len(sel) >= K:
            break
        need_t, need_e, need_b = dur[i], eng[i], byt[i]
        fits_t = (time_budget is None) or (need_t <= rem_t)
        fits_e = (energy_budget is None) or (need_e <= rem_e)
        fits_b = (bytes_budget is None) or (need_b <= rem_b)
        if fits_t and fits_e and fits_b:
            sel.append(i)
            rem_t -= need_t
            rem_e -= need_e
            rem_b -= need_b

    # Best-effort fill if underfull
    if len(sel) < K:
        chosen = set(sel)
        for i, _ in scores:
            if i in chosen:
                continue
            sel.append(i)
            if len(sel) >= K:
                break

    ids = [clients[i].id for i in sel[:K]]
    return ids, [s for _, s in scores], {}


