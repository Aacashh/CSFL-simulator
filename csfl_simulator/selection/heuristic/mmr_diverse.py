from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list


def _build_utilities(clients: List[ClientInfo]) -> List[float]:
    # Combine multiple signals into a utility score and normalize
    loss = [float(c.last_loss or 0.0) for c in clients]
    gnorm = [float(c.grad_norm or 0.0) for c in clients]
    speed = [float(c.compute_speed or 1.0) for c in clients]
    chq = [float(c.channel_quality or 1.0) for c in clients]
    part = [float(c.participation_count or 0.0) for c in clients]

    loss_n = normalize_list(loss)
    g_n = normalize_list(gnorm)
    s_n = normalize_list(speed)
    q_n = normalize_list(chq)
    inv_part = [1.0 / (1.0 + p) for p in part]

    u = [0.4 * l + 0.3 * g + 0.15 * s + 0.15 * q for l, g, s, q in zip(loss_n, g_n, s_n, q_n)]
    u = [ui * inv_part[i] for i, ui in enumerate(u)]
    return normalize_list(u)


def _build_embeddings(clients: List[ClientInfo]) -> np.ndarray:
    # Prefer label_histogram vectors if available; else fall back to simple stats
    hists: List[Optional[Dict[int, int]]] = [c.label_histogram for c in clients]
    keys = set()
    for h in hists:
        if isinstance(h, dict):
            keys.update(h.keys())
    if keys:
        L = int(max(keys) + 1)
        X = np.zeros((len(clients), L), dtype=float)
        for i, h in enumerate(hists):
            if not isinstance(h, dict):
                continue
            for lbl, cnt in h.items():
                if 0 <= int(lbl) < L:
                    X[i, int(lbl)] = float(cnt)
    else:
        X = np.array([[float(c.compute_speed or 1.0), float(c.channel_quality or 1.0), float(c.data_size or 0.0)] for c in clients], dtype=float)
    # L2 normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, lambda_relevance: float = 0.7) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """MMR-style selector: greedily balances utility and diversity (cosine similarity).
    - lambda_relevance: weight on relevance vs diversity (0..1)
    Returns: (selected_client_ids, utility_scores_per_client, state_update)
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    u = _build_utilities(clients)
    X = _build_embeddings(clients)
    S: List[int] = []  # indices into clients
    sims = _cosine_sim_matrix(X)

    remaining = set(range(n))
    # First pick: highest utility
    first = max(remaining, key=lambda i: u[i])
    S.append(first)
    remaining.remove(first)

    while len(S) < K and remaining:
        best_i = None
        best_score = -1e9
        for i in remaining:
            max_sim = max(sims[i, j] for j in S) if S else 0.0
            score = lambda_relevance * u[i] - (1.0 - lambda_relevance) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        S.append(best_i)
        remaining.remove(best_i)

    sel_ids = [clients[i].id for i in S]
    return sel_ids, u, {}

