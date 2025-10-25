from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list


def _collect_hist_info(clients: List[ClientInfo]):
    # Build a consistent label space across clients
    keys = set()
    vecs = []
    for c in clients:
        h = c.label_histogram
        if isinstance(h, dict):
            keys.update(h.keys())
    if not keys:
        return None, None, None
    L = int(max(keys) + 1)
    for c in clients:
        v = np.zeros(L, dtype=float)
        h = c.label_histogram
        if isinstance(h, dict):
            for k, cnt in h.items():
                idx = int(k)
                if 0 <= idx < L:
                    v[idx] = float(cnt)
        vecs.append(v)
    V = np.vstack(vecs) if vecs else np.zeros((len(clients), L), dtype=float)
    return V, L, keys


def _base_utilities(clients: List[ClientInfo]) -> List[float]:
    loss = [float(c.last_loss or 0.0) for c in clients]
    loss_n = normalize_list(loss)
    return loss_n


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, scarcity_weighting: str = "idf", mix_alpha: float = 0.2,
                   require_histogram: bool = False) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Greedy label-coverage selector.
    - scarcity_weighting: 'idf' (1/freq) or 'uniform'
    - mix_alpha: weight to mix utility with coverage gain (0..1)
    - require_histogram: if True, fallback to random when histograms missing
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    V, L, keys = _collect_hist_info(clients)
    if V is None:
        if require_histogram:
            ids = [c.id for c in clients]
            rng.shuffle(ids)
            return ids[:K], None, {}
        # Fallback to utility-only top-k
        u = _base_utilities(clients)
        ranked = sorted(range(n), key=lambda i: u[i], reverse=True)
        return [clients[i].id for i in ranked[:K]], u, {}

    # Compute scarcity weights
    freqs = V.sum(axis=0)  # total counts per label
    if scarcity_weighting == "idf":
        w = 1.0 / (freqs + 1e-6)
    else:
        w = np.ones_like(freqs)

    # Utility mix
    u = np.array(_base_utilities(clients), dtype=float)

    covered = np.zeros(L, dtype=bool)
    selected: List[int] = []
    remaining = set(range(n))

    while len(selected) < K and remaining:
        best_i = None
        best_gain = -1e9
        for i in remaining:
            # Binary coverage gain for labels not yet covered
            hist = V[i]
            mask = (hist > 0) & (~covered)
            cov_gain = float((w[mask]).sum())
            score = (1.0 - mix_alpha) * cov_gain + mix_alpha * float(u[i])
            if score > best_gain:
                best_gain = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        # Update covered labels
        covered = covered | (V[best_i] > 0)

    if len(selected) < K and remaining:
        # Fill the rest by utility
        rest = sorted(list(remaining), key=lambda i: u[i], reverse=True)
        selected.extend(rest[: K - len(selected)])

    return [clients[i].id for i in selected], u.tolist(), {}

