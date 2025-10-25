from typing import List, Dict, Optional, Tuple
import numpy as np
from csfl_simulator.core.client import ClientInfo

STATE = "bandit_linucb"
ALPHA = 1.0


def _features(c: ClientInfo):
    return np.array([
        float(c.data_size),
        float(c.last_loss or 0.0),
        float(c.grad_norm or 0.0),
        float(c.compute_speed or 1.0),
        float(c.channel_quality or 1.0),
        float(c.participation_count or 0.0)
    ], dtype=float)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    st = history.get("state", {}).get(STATE, {"A": None, "b": None})
    d = 6
    if st["A"] is None:
        st["A"] = np.eye(d)
        st["b"] = np.zeros((d,))
    A = st["A"]
    b = st["b"]
    A_inv = np.linalg.inv(A)
    scores = []
    for c in clients:
        x = _features(c)
        theta = A_inv @ b
        p = theta.dot(x) + ALPHA * np.sqrt(x.T @ A_inv @ x)
        scores.append((c.id, p, x))
    scores.sort(key=lambda t: t[1], reverse=True)
    sel = []
    for cid, p, x in scores[:K]:
        sel.append(cid)
    # Update with last reward
    reward = history.get("state", {}).get("last_reward", 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    for cid in last_sel:
        x = next((_x for (_cid, _p, _x) in scores if _cid == cid), None)
        if x is None: continue
        A = A + np.outer(x, x)
        b = b + reward * x
    st["A"], st["b"] = A, b
    return sel, [p for _, p, _ in scores], {STATE: st}
