from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDRegressor

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy

STATE = "meta_ranker_state"


def _features(clients: List[ClientInfo]) -> np.ndarray:
    data = np.array([float(c.data_size or 0.0) for c in clients], dtype=float)
    loss = np.array([float(c.last_loss or 0.0) for c in clients], dtype=float)
    gnorm = np.array([float(c.grad_norm or 0.0) for c in clients], dtype=float)
    inv_dur = np.array([1.0 / max(1e-6, expected_duration(c)) for c in clients], dtype=float)
    part = np.array([float(c.participation_count or 0.0) for c in clients], dtype=float)
    chq = np.array([float(c.channel_quality or 1.0) for c in clients], dtype=float)
    spd = np.array([float(c.compute_speed or 1.0) for c in clients], dtype=float)
    ent = []
    for c in clients:
        if isinstance(c.label_histogram, dict) and c.label_histogram:
            L = int(max(c.label_histogram.keys()) + 1)
            vec = [0.0] * L
            for k, v in c.label_histogram.items():
                idx = int(k)
                if 0 <= idx < L:
                    vec[idx] = float(v)
            ent.append(label_entropy(vec))
        else:
            ent.append(0.0)
    ent = np.array(ent, dtype=float)
    X = np.stack([data, loss, gnorm, inv_dur, part, chq, spd, ent], axis=1)
    # Standardize columns (per-call) for stable regression
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    Xn = (X - mu) / sigma
    return Xn


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, max_history: int = 5000, alpha: float = 1e-4,
                   learning_rate: str = "optimal") -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Meta Ranker: stateful SGDRegressor that predicts expected reward per client and ranks accordingly.
    - Accumulates (features, reward) from previous rounds for selected clients.
    - Cold-start: fallback to utility proxy (loss / duration).
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        st = {
            "model": SGDRegressor(alpha=alpha, learning_rate=learning_rate, max_iter=1, tol=None, penalty="l2"),
            "fitted": False,
            "X_buf": [],
            "y_buf": [],
            "last_feat": {},
        }

    # Update using last round's reward and last features
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        for cid in last_sel:
            x = st["last_feat"].get(cid, None)
            if x is None:
                continue
            st["X_buf"].append(x)
            st["y_buf"].append(reward)
        # Keep buffer bounded
        if len(st["X_buf"]) > max_history:
            overflow = len(st["X_buf"]) - max_history
            st["X_buf"] = st["X_buf"][overflow:]
            st["y_buf"] = st["y_buf"][overflow:]
        # Train incrementally
        Xb = np.array(st["X_buf"], dtype=float)
        yb = np.array(st["y_buf"], dtype=float)
        if Xb.shape[0] >= 10:  # require a small warm-up
            st["model"].partial_fit(Xb, yb)
            st["fitted"] = True

    # Build current features
    X = _features(clients)
    # Cache per-cid
    st["last_feat"] = {c.id: X[i] for i, c in enumerate(clients)}

    # Predict or fallback
    if st["fitted"]:
        y_hat = st["model"].predict(X)
    else:
        # Fallback utility: loss / duration
        y_hat = np.array([float(c.last_loss or 0.0) / max(1e-6, expected_duration(c)) for c in clients], dtype=float)

    ranked = np.argsort(-y_hat)  # descending
    sel_idxs = ranked[:K].tolist()
    sel = [clients[i].id for i in sel_idxs]

    # Per-client scores in original order
    per_client = [float(y_hat[i]) for i in range(n)]

    return sel, per_client, {STATE: st}

