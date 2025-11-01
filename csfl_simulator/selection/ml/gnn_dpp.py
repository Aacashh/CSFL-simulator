from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration


def _feat(c: ClientInfo) -> List[float]:
    return [
        float(c.last_loss or 0.0),
        float(c.grad_norm or 0.0),
        float(c.data_size or 0.0),
        1.0 / max(1e-6, float(c.compute_speed or 1.0)),  # slower => larger
        1.0 / max(1e-6, float(c.channel_quality or 1.0)),  # weaker => larger
        float(c.participation_count or 0.0),
    ]


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cos(a: List[float], b: List[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, lambda_div: float = 0.5,
                   attn_temp: float = 1.0) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """GNN-DPP: attention-aggregated utility with DPP-style diversity (MMR approximation).
    - Build feature vectors, compute per-client base utility ~ loss/duration.
    - Neighborhood attention: each client’s score includes a soft aggregate of neighbors’ utilities via cosine attention.
    - Diversity via MMR: penalize similarity to selected set.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    feats = [_feat(c) for c in clients]
    dur = [max(1e-6, expected_duration(c)) for c in clients]
    util_raw = [(float(c.last_loss or 0.0) / d) for c, d in zip(clients, dur)]
    util = normalize_list(util_raw)

    # Attention-aggregated utility
    agg = [0.0] * n
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            if i == j:
                continue
            s = _cos(feats[i], feats[j])
            # temperature-scaled attention weight (softmax-like without exp overflow)
            w = math.exp(attn_temp * s)
            num += w * util[j]
            den += w
        neigh = (num / den) if den > 0 else 0.0
        agg[i] = 0.7 * util[i] + 0.3 * neigh

    selected: List[int] = []
    picked = set()
    scores: List[float] = [0.0] * n
    remaining_budget = float(time_budget) if time_budget is not None else float("inf")

    for _ in range(min(K, n)):
        best_i = -1
        best_sc = -1e18
        for i in range(n):
            if i in picked:
                continue
            # DPP-like diversity penalty: max similarity to selected
            if picked:
                max_sim = 0.0
                for j in picked:
                    s = _cos(feats[i], feats[j])
                    if s > max_sim:
                        max_sim = s
            else:
                max_sim = 0.0
            sc = agg[i] - float(lambda_div) * max_sim
            if time_budget is not None and dur[i] > remaining_budget:
                # lightly penalize infeasible under budget
                sc -= 0.25
            if sc > best_sc:
                best_sc = sc
                best_i = i
        if best_i < 0:
            break
        if time_budget is not None and dur[best_i] > remaining_budget:
            # find next feasible
            feas = [i for i in range(n) if i not in picked and dur[i] <= remaining_budget]
            if not feas:
                break
            # choose best feasible by score
            bf = max(feas, key=lambda i: agg[i])
            best_i = bf
            best_sc = agg[bf]
        picked.add(best_i)
        selected.append(clients[best_i].id)
        scores[best_i] = best_sc
        remaining_budget -= dur[best_i]

    # Fill remaining if any
    if len(selected) < K:
        remain = [i for i in range(n) if i not in picked]
        remain.sort(key=lambda i: agg[i], reverse=True)
        for i in remain[: K - len(selected)]:
            if time_budget is not None and dur[i] > remaining_budget:
                continue
            selected.append(clients[i].id)
            scores[i] = agg[i]
            remaining_budget -= dur[i]
            if time_budget is not None and remaining_budget <= 0:
                break

    return selected, scores, {}


