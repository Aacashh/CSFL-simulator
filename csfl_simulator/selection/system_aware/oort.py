from typing import List, Dict, Optional, Tuple
import math
from csfl_simulator.core.client import ClientInfo

STATE = "oort_state"


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    st = history.get("state", {}).get(STATE, {"N": {}, "U": {}})
    # Get previous reward (global acc improvement) if present
    reward = history.get("state", {}).get("last_reward", 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    # Update utilities for last selected clients
    for cid in last_sel:
        st["N"][cid] = st["N"].get(cid, 0) + 1
        st["U"][cid] = st["U"].get(cid, 0.0) * 0.9 + 0.1 * reward
    # Compute scores for current selection
    scores = []
    t = max(1, sum(st["N"].values()))
    for c in clients:
        loss_term = (c.last_loss or 0.0)
        time_penalty = (c.estimated_duration or 1.0)
        base = loss_term / max(1e-6, time_penalty)
        n_i = st["N"].get(c.id, 0)
        ucb = 0.0
        if n_i > 0:
            ucb = math.sqrt(2.0 * math.log(t) / n_i)
        scores.append((c.id, base + 0.1 * ucb))
    scores.sort(key=lambda x: x[1], reverse=True)
    sel = [cid for cid, _ in scores[:K]]
    return sel, [s for _, s in scores], {STATE: st}
