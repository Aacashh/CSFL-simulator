from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration


def _hist_vector(c: ClientInfo, L: int = 128) -> List[float]:
    """Turn label_histogram into a fixed-length dense vector (best-effort).
    If histogram is missing, return zeros.
    """
    h = getattr(c, "label_histogram", None)
    if not h:
        return [0.0] * L
    # Infer max label
    try:
        max_label = int(max(h.keys()))
    except Exception:
        max_label = 0
    dim = min(max(L, max_label + 1), L)
    v = [0.0] * dim
    for k, vcount in h.items():
        idx = int(k)
        if 0 <= idx < dim:
            v[idx] = float(vcount)
    s = float(sum(v))
    if s > 0:
        v = [x / s for x in v]
    return v


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / math.sqrt(na * nb))


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, corr_alpha: float = 0.5,
                   utility_weight: float = 1.0) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """FedCor-inspired correlation-aware active selection (approximation).
    Greedy objective: utility(i) - corr_alpha * max_{j in S} sim(i, j),
    where sim is cosine between label-hist vectors; utility prefers loss/time.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    # Precompute features
    embeds = [_hist_vector(c) for c in clients]
    util_raw = []
    for c in clients:
        loss_term = float(c.last_loss or 0.0)
        dur = max(1e-6, expected_duration(c))
        util_raw.append(loss_term / dur)
    util = normalize_list(util_raw)

    selected: List[int] = []
    selected_idx: List[int] = []
    picked = set()
    scores: List[float] = [0.0] * n

    for _ in range(min(K, n)):
        best_i = -1
        best_score = -1e18
        for i, c in enumerate(clients):
            if i in picked:
                continue
            # Diversity penalty = max similarity to selected
            if selected_idx:
                max_sim = 0.0
                for j in selected_idx:
                    s = _cosine(embeds[i], embeds[j])
                    if s > max_sim:
                        max_sim = s
            else:
                max_sim = 0.0
            score_i = utility_weight * util[i] - float(corr_alpha) * max_sim
            if score_i > best_score:
                best_score = score_i
                best_i = i
        if best_i < 0:
            break
        picked.add(best_i)
        selected_idx.append(best_i)
        selected.append(clients[best_i].id)
        scores[best_i] = best_score

    # Fill if needed
    if len(selected) < K:
        remain = [i for i in range(n) if i not in picked]
        rng.shuffle(remain)
        for i in remain[: K - len(selected)]:
            selected.append(clients[i].id)
            scores[i] = utility_weight * util[i]

    return selected, scores, {}


