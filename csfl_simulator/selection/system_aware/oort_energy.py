from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, normalize_list, recency


STATE = "oort_energy_state"


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
    beta: float = 0.5,
    alpha_ucb: float = 0.1,
    rho_energy: float = 0.5,
    kappa_bytes: float = 0.3,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Oort with energy/bytes-aware penalties and optional budget filter.

    score_i = (util_i**beta / dur_i) * (1 + alpha_ucb * UCB_i) / (1 + rho*E_i + kappa*B_i)
    where E_i, B_i are min-max normalized energy/bytes across current cohort.
    """
    st = history.get("state", {}).get(STATE, {"N": {}, "t": 0})

    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    for cid in last_sel:
        st["N"][cid] = int(st["N"].get(cid, 0)) + 1
    st["t"] = int(st.get("t", 0)) + 1

    n = len(clients)
    if n == 0:
        return [], None, {STATE: st}

    util = [float(c.last_loss or 0.0) for c in clients]
    dur = [expected_duration(c) for c in clients]
    eng = [float(getattr(c, "estimated_energy", 0.0) or (getattr(c, "energy_rate", 1.0) * d)) for c, d in zip(clients, dur)]
    byt = [float(getattr(c, "estimated_bytes", 0.0) or float(c.data_size or 0.0)) for c in clients]
    eng_n = normalize_list(eng)
    byt_n = normalize_list(byt)

    scores = []
    t_global = max(1, st["t"])  # for UCB
    for i, c in enumerate(clients):
        base = (util[i] ** max(0.0, min(1.0, float(beta)))) / max(1e-6, float(dur[i]))
        n_i = int(st["N"].get(c.id, 0))
        ucb = math.sqrt(2.0 * math.log(t_global + 1.0) / (n_i + 1.0))
        penalty = 1.0 + float(rho_energy) * (eng_n[i] if eng_n else 0.0) + float(kappa_bytes) * (byt_n[i] if byt_n else 0.0)
        score = base * (1.0 + float(alpha_ucb) * ucb) / penalty
        scores.append((i, score))
    scores.sort(key=lambda t: t[1], reverse=True)

    # Optional budget packing
    sel_idx: List[int] = []
    rem_t = float(time_budget) if time_budget is not None else float("inf")
    rem_e = float(energy_budget) if energy_budget is not None else float("inf")
    rem_b = float(bytes_budget) if bytes_budget is not None else float("inf")
    for i, _ in scores:
        if len(sel_idx) >= K:
            break
        t_need, e_need, b_need = dur[i], eng[i], byt[i]
        fits_t = (time_budget is None) or (t_need <= rem_t)
        fits_e = (energy_budget is None) or (e_need <= rem_e)
        fits_b = (bytes_budget is None) or (b_need <= rem_b)
        if fits_t and fits_e and fits_b:
            sel_idx.append(i)
            rem_t -= t_need
            rem_e -= e_need
            rem_b -= b_need

    if len(sel_idx) < K:
        chosen = set(sel_idx)
        for i, _ in scores:
            if i in chosen:
                continue
            sel_idx.append(i)
            if len(sel_idx) >= K:
                break

    ids = [clients[i].id for i in sel_idx[:K]]
    return ids, [s for _, s in scores], {STATE: st}


