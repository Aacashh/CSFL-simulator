from typing import List, Dict, Optional, Tuple
import random
from csfl_simulator.core.client import ClientInfo

STATE = "bandit_epsg"


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, epsilon: float = 0.1) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    st = history.get("state", {}).get(STATE, {"N": {}, "Q": {}, "epsilon": float(epsilon)})
    # If preset provided epsilon, override state epsilon on first use
    if "epsilon" in st:
        try:
            st["epsilon"] = float(epsilon)
        except Exception:
            pass
    # Update with last reward
    reward = history.get("state", {}).get("last_reward", 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    for cid in last_sel:
        n = st["N"].get(cid, 0) + 1
        q = st["Q"].get(cid, 0.0)
        # incremental update
        q = q + (reward - q) / n
        st["N"][cid] = n
        st["Q"][cid] = q
    # Selection
    eps = float(st.get("epsilon", epsilon))
    ids = [c.id for c in clients]
    if rng.random() < eps:
        rng.shuffle(ids)
        sel = ids[:K]
    else:
        ranked = sorted(ids, key=lambda cid: st["Q"].get(cid, 0.0), reverse=True)
        sel = ranked[:K]
    scores = [st["Q"].get(c.id, 0.0) for c in clients]
    return sel, scores, {STATE: st}
