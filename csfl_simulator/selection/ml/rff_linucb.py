from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration, label_entropy

STATE = "bandit_rff_linucb"


def _features(clients: List[ClientInfo]) -> np.ndarray:
    # Build per-client feature vectors with simple normalization across current cohort
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

    def _mm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        m, M = float(np.min(x)), float(np.max(x))
        if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
            return np.zeros_like(x)
        return (x - m) / (M - m + 1e-12)

    cols = [
        _mm(data), _mm(loss), _mm(gnorm), _mm(inv_dur), _mm(part), _mm(chq), _mm(spd), _mm(ent)
    ]
    X = np.stack(cols, axis=1)
    return X


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, n_components: int = 128, rbf_gamma: float = 0.5,
                   reg_lambda: float = 1e-2, alpha_ucb: float = 1.0) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Contextual bandit using LinUCB over Random Fourier Features.
    Maintains state across rounds under STATE.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        # Lazy init
        rff = RBFSampler(gamma=rbf_gamma, n_components=n_components, random_state=42)
        st = {
            "A": np.eye(n_components, dtype=float) * reg_lambda,
            "A_inv": np.eye(n_components, dtype=float) / max(reg_lambda, 1e-6),
            "b": np.zeros((n_components,), dtype=float),
            "rff": rff,
            "rff_fitted": False,
            "last_selected": [],
            "z_cache": {},
        }

    # Update from last round's reward
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        for cid in last_sel:
            z = st["z_cache"].get(cid, None)
            if z is None:
                continue
            # Sherman-Morrison for A_inv update
            A_inv = st["A_inv"]
            z = z.reshape(-1, 1)
            denom = 1.0 + float((z.T @ A_inv @ z).squeeze())
            A_inv = A_inv - (A_inv @ z @ z.T @ A_inv) / max(denom, 1e-12)
            st["A_inv"] = A_inv
            st["A"] = None  # no longer needed
            st["b"] = st["b"] + reward * z.flatten()

    # Build current design
    X = _features(clients)
    # Ensure RBFSampler is fitted once to fix random features across rounds
    rff = st["rff"]
    if not st.get("rff_fitted", False):
        rff.fit(X)
        st["rff_fitted"] = True
    Z = rff.transform(X)

    theta = st["A_inv"] @ st["b"]

    # Score UCB
    scores = []
    z_cache = {}
    for i, c in enumerate(clients):
        z = Z[i]
        z_cache[c.id] = z
        mean = float(theta @ z)
        # exploration term
        A_inv = st["A_inv"]
        conf = float(np.sqrt(z.T @ A_inv @ z))
        p = mean + alpha_ucb * conf
        scores.append((c.id, p))

    scores.sort(key=lambda t: t[1], reverse=True)
    sel = [cid for cid, _ in scores[:K]]

    # Stash features for next round's update
    st["last_selected"] = sel
    st["z_cache"] = {cid: z_cache[cid] for cid in sel}

    # Return per-client scores in original order
    per_client = []
    score_map = {cid: s for cid, s in scores}
    for c in clients:
        per_client.append(float(score_map.get(c.id, 0.0)))

    return sel, per_client, {STATE: st}

