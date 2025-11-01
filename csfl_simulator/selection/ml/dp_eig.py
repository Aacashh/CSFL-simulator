from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list, expected_duration


def _label_set_gain(current_cov: Dict[int, float], new_hist: Dict[int, int] | None) -> float:
    """Coverage gain proxy: favor clients adding rare/uncovered labels.
    current_cov counts labels already covered in S; new_hist is label histogram.
    """
    if not new_hist:
        return 0.0
    gain = 0.0
    total = float(sum(new_hist.values())) or 1.0
    for lbl, cnt in new_hist.items():
        p = float(cnt) / total
        # Scarcity weight: inverse to current coverage (add epsilon)
        scarcity = 1.0 / (1.0 + float(current_cov.get(int(lbl), 0.0)))
        gain += scarcity * p
    return gain


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, sigma: float = 0.0,
                   lambda_cov: float = 0.3, lambda_time: float = 0.2) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """DP-aware Expected Information Gain (greedy knapsack-like selection).
    EIG_i ∝ (data_size * (grad_norm^2 + loss)) / duration / (1 + sigma^2).
    We add a label-coverage gain and a time-budget penalty.
    Greedy lazy selection approximates submodular maximization.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n and time_budget is None:
        return [c.id for c in clients], None, {}

    # Base utility terms
    base = []
    durations = []
    for c in clients:
        gn = float(c.grad_norm or 0.0)
        loss = float(c.last_loss or 0.0)
        data = float(c.data_size or 1.0)
        dur = max(1e-6, expected_duration(c))
        durations.append(dur)
        info = data * (gn * gn + loss)
        base.append(info / dur)
    base = normalize_list(base)

    # DP attenuation factor
    dp_att = 1.0 / (1.0 + float(sigma) * float(sigma)) if sigma and sigma > 0 else 1.0

    selected: List[int] = []
    picked = set()
    scores: List[float] = [0.0] * n
    label_cov: Dict[int, float] = {}
    remaining_budget = float(time_budget) if time_budget is not None else float("inf")

    for _ in range(min(K, n)):
        best_i = -1
        best_gain = -1e18
        for i, c in enumerate(clients):
            if i in picked:
                continue
            cov_gain = lambda_cov * _label_set_gain(label_cov, getattr(c, "label_histogram", None))
            time_penalty = 0.0
            if time_budget is not None:
                # Penalize if likely to exceed budget
                if durations[i] > remaining_budget:
                    time_penalty = lambda_time
                else:
                    time_penalty = lambda_time * (durations[i] / max(1e-6, remaining_budget))
            gain = dp_att * base[i] + cov_gain - time_penalty
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        # Feasibility under time budget
        if time_budget is not None and durations[best_i] > remaining_budget:
            # Try to find next feasible
            feas_i = -1
            feas_gain = -1e18
            for i, c in enumerate(clients):
                if i in picked or durations[i] > remaining_budget:
                    continue
                cov_gain = lambda_cov * _label_set_gain(label_cov, getattr(c, "label_histogram", None))
                gain = dp_att * base[i] + cov_gain
                if gain > feas_gain:
                    feas_gain = gain
                    feas_i = i
            if feas_i < 0:
                break
            best_i = feas_i

        picked.add(best_i)
        selected.append(clients[best_i].id)
        scores[best_i] = float(best_gain)
        remaining_budget -= durations[best_i]
        # Update coverage state
        h = getattr(clients[best_i], "label_histogram", None)
        if h:
            for lbl, cnt in h.items():
                label_cov[int(lbl)] = label_cov.get(int(lbl), 0.0) + float(cnt)

    # Fill if still short and no budget constraint
    if len(selected) < K and not (time_budget is not None and remaining_budget <= 0):
        rest = [i for i in range(n) if i not in picked]
        rest.sort(key=lambda i: base[i], reverse=True)
        for i in rest[: K - len(selected)]:
            if time_budget is not None and durations[i] > remaining_budget:
                continue
            selected.append(clients[i].id)
            scores[i] = dp_att * base[i]
            remaining_budget -= durations[i]
            if remaining_budget <= 0 and time_budget is not None:
                break

    return selected, scores, {}


