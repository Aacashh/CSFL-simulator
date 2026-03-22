"""DELTA: Diversity-Enhanced Loss-Temporal Adaptive Selection.

A lightweight O(K*N) client selection method that combines:
  - EMA-smoothed loss/duration utility (information density)
  - UCB1 exploration bonus (prevents starvation)
  - Label-histogram cosine diversity (greedy complementarity)
  - Optional time-budget packing

Theoretical motivation (non-IID FL convergence bound):
  E[||w_t - w*||^2] <= (1 - eta*mu)^t ||w_0 - w*||^2 + C1*Gamma/(eta*mu) + C2*sigma^2/(eta*K)

  DELTA minimizes Gamma (client drift) via loss-biased selection
  and sigma^2/K (gradient variance) via label diversity.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import choose_random, expected_duration

_STATE_KEY = "delta_state"


def _collect_label_space(clients: List[ClientInfo]) -> int:
    """Determine label space size L from all client histograms."""
    mx = -1
    for c in clients:
        if isinstance(c.label_histogram, dict):
            for k in c.label_histogram:
                try:
                    mx = max(mx, int(k))
                except (ValueError, TypeError):
                    pass
    return mx + 1 if mx >= 0 else 0


def _hist_to_vec(hist: Optional[Dict], L: int) -> Optional[np.ndarray]:
    """Convert label histogram dict to L2-normalized dense vector."""
    if not isinstance(hist, dict) or not hist or L <= 0:
        return None
    v = np.zeros(L, dtype=np.float64)
    for k, cnt in hist.items():
        try:
            idx = int(k)
            if 0 <= idx < L:
                v[idx] = float(cnt)
        except (ValueError, TypeError):
            pass
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    v /= norm
    return v


def _min_cosine_distance(
    cand_vec: np.ndarray, selected_vecs: List[np.ndarray]
) -> float:
    """Min cosine distance from candidate to any selected client.

    Returns value in [0, 2]. Higher = more diverse from selected set.
    Returns 1.0 (neutral) when selected set is empty.
    """
    if not selected_vecs:
        return 1.0
    min_sim = min(float(np.dot(cand_vec, sv)) for sv in selected_vecs)
    return 1.0 - min_sim


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # Hyperparameters (overridable via YAML params)
    decay: float = 0.7,
    beta: float = 1.0,
    c_ucb: float = 0.5,
    lam: float = 0.3,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """DELTA: Diversity-Enhanced Loss-Temporal Adaptive Selection.

    Parameters
    ----------
    decay : float
        EMA decay factor for utility smoothing (0 = no memory, 1 = infinite memory).
    beta : float
        Exponent on loss term (higher = more aggressive loss-biasing).
    c_ucb : float
        UCB exploration constant (0 = no exploration).
    lam : float
        Diversity bonus weight (0 = no diversity consideration).
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    # Retrieve persistent state
    st = history.get("state", {}).get(_STATE_KEY, {})
    ema: Dict[int, float] = dict(st.get("ema", {}))
    counts: Dict[int, int] = dict(st.get("counts", {}))
    t = round_idx + 1

    # Cold start: if no client has loss info yet, select randomly
    has_loss = any(
        getattr(c, "last_loss", 0.0) and c.last_loss > 0 for c in clients
    )
    if not has_loss:
        selected = choose_random(clients, K, rng)
        for cid in selected:
            counts[cid] = counts.get(cid, 0) + 1
        return selected, None, {_STATE_KEY: {"ema": ema, "counts": counts}}

    # Precompute label vectors for diversity
    L = _collect_label_space(clients)
    client_vecs: Dict[int, Optional[np.ndarray]] = {}
    for c in clients:
        client_vecs[c.id] = _hist_to_vec(c.label_histogram, L)

    # Phase 1: Score all clients (utility x exploration)
    id_to_client: Dict[int, ClientInfo] = {c.id: c for c in clients}
    scores: Dict[int, float] = {}

    for c in clients:
        dur = expected_duration(c)
        loss_val = min(float(c.last_loss or 0.0), 1e6)
        raw_util = (loss_val ** beta) / max(dur, 1e-6)

        # EMA update
        prev = ema.get(c.id)
        if prev is None:
            mu = raw_util  # first encounter
        elif loss_val > 0:
            mu = decay * prev + (1.0 - decay) * raw_util
        else:
            mu = prev  # no fresh data, keep estimate
        ema[c.id] = mu

        # UCB exploration bonus
        ni = counts.get(c.id, 0)
        explore = c_ucb * math.sqrt(math.log(t + 1) / (ni + 1))

        scores[c.id] = mu * (1.0 + explore)

    # Phase 2: Greedy selection with diversity
    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []
    remaining_budget = float(time_budget) if time_budget is not None else None
    pool = set(id_to_client.keys())

    for _ in range(K):
        best_id, best_score = -1, -float("inf")

        for cid in pool:
            if cid in selected:
                continue
            c = id_to_client[cid]
            if remaining_budget is not None and expected_duration(c) > remaining_budget:
                continue

            # Diversity bonus
            div_bonus = 0.0
            cv = client_vecs.get(cid)
            if cv is not None and lam > 0:
                div_bonus = _min_cosine_distance(cv, selected_vecs)

            total = scores[cid] * (1.0 + lam * div_bonus)

            if total > best_score:
                best_score = total
                best_id = cid

        if best_id < 0:
            break

        selected.append(best_id)
        pool.discard(best_id)
        cv = client_vecs.get(best_id)
        if cv is not None:
            selected_vecs.append(cv)
        if remaining_budget is not None:
            remaining_budget -= expected_duration(id_to_client[best_id])
        counts[best_id] = counts.get(best_id, 0) + 1

    # Fill remaining slots randomly if budget prevented filling K
    if len(selected) < K:
        remaining = [cid for cid in pool if cid not in set(selected)]
        rng.shuffle(remaining)
        for cid in remaining[: K - len(selected)]:
            selected.append(cid)
            counts[cid] = counts.get(cid, 0) + 1

    per_scores = [scores.get(cid, 0.0) for cid in selected]
    state_update = {_STATE_KEY: {"ema": ema, "counts": counts}}

    return selected, per_scores, state_update
