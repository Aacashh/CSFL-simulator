from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration, recency


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None,
                   w_acc: float = 0.6, w_time: float = 0.2,
                   w_fair: float = 0.1, w_dp: float = 0.1,
                   min_low_participation: int = 0) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """ParetoRL (practical multi-objective greedy with safety constraints).
    Approximates a constrained RL policy with a weighted scoring and fairness guard.
    - Accuracy utility: loss / duration
    - Time utility: 1 / duration
    - Fairness: prioritize clients with lowest participation_count and longest recency
    - DP: prefer clients with more epsilon remaining
    - Optional time_budget greedy packing
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n and time_budget is None:
        return [c.id for c in clients], None, {}

    # Core components
    util_loss = []
    util_time = []
    fair_raw = []
    dp_raw = []
    durations = []
    for c in clients:
        dur = max(1e-6, expected_duration(c))
        durations.append(dur)
        util_loss.append(float(c.last_loss or 0.0) / dur)
        util_time.append(1.0 / dur)
        # Fairness: low participation and long recency => higher
        fair_raw.append(1.0 / (1.0 + float(c.participation_count or 0.0)) + 0.1 * float(recency(round_idx, c)))
        dp_raw.append(float(getattr(c, 'dp_epsilon_remaining', 0.0) or 0.0))
    util_loss = normalize_list(util_loss)
    util_time = normalize_list(util_time)
    fair = normalize_list(fair_raw)
    dpn = normalize_list(dp_raw)

    base_score = [
        float(w_acc) * util_loss[i]
        + float(w_time) * util_time[i]
        + float(w_fair) * fair[i]
        + float(w_dp) * dpn[i]
        for i in range(n)
    ]

    # Safety: ensure minimum low-participation clients are included first
    ids_sorted_by_participation = sorted(range(n), key=lambda i: (float(clients[i].participation_count or 0.0), -base_score[i]))
    selected: List[int] = []
    picked = set()
    remaining_budget = float(time_budget) if time_budget is not None else float('inf')
    if min_low_participation and min_low_participation > 0:
        for i in ids_sorted_by_participation[: min(min_low_participation, n)]:
            if time_budget is not None and durations[i] > remaining_budget:
                continue
            picked.add(i)
            selected.append(clients[i].id)
            remaining_budget -= durations[i]
            if len(selected) >= K:
                break

    # Greedy by composite score with optional budget
    rest = [i for i in range(n) if i not in picked]
    rest.sort(key=lambda i: (base_score[i], rng.random()), reverse=True)
    for i in rest:
        if len(selected) >= K:
            break
        if time_budget is not None and durations[i] > remaining_budget:
            continue
        picked.add(i)
        selected.append(clients[i].id)
        remaining_budget -= durations[i]

    # Per-client scores for diagnostics
    scores = [0.0] * n
    for i in range(n):
        scores[i] = base_score[i]

    return selected, scores, {}


