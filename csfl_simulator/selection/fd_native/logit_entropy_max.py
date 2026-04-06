"""Logit Entropy Maximization (LEM) for Federated Distillation.

Selects clients whose logit predictions on the public dataset have the highest
informative entropy. Distinguishes genuine model uncertainty (high entropy with
high variance) from noise-corrupted logits (near-uniform entropy with low variance).
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import normalize_list


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # --- LEM hyperparameters ---
    noise_penalty: float = 0.5,
    entropy_ema_alpha: float = 0.3,
    w_fairness: float = 0.15,
    max_entropy_ratio: float = 0.95,
    min_entropy_var: float = 0.01,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    state = history.get("state", {}).get("lem", {})
    ent_ema = state.get("entropy_ema", {})
    ent_var_ema = state.get("entropy_var_ema", {})

    # --- Step 1: Retrieve logit stats from last round ---
    fd_stats = history.get("state", {}).get("fd_logit_stats", {})
    for cid_int, stats in fd_stats.items():
        cid = str(cid_int)
        ent_mean = stats.get("entropy_mean", 0.0)
        ent_var = stats.get("entropy_var", 0.0)
        old_ent = ent_ema.get(cid, ent_mean)
        old_var = ent_var_ema.get(cid, ent_var)
        ent_ema[cid] = entropy_ema_alpha * ent_mean + (1 - entropy_ema_alpha) * old_ent
        ent_var_ema[cid] = entropy_ema_alpha * ent_var + (1 - entropy_ema_alpha) * old_var

    # --- Step 2: Noise detection + scoring ---
    # Determine number of classes for max entropy
    num_classes = 10
    for c in clients:
        h = c.label_histogram
        if isinstance(h, dict) and h:
            num_classes = max(num_classes, max((int(k) for k in h.keys()), default=0) + 1)

    max_entropy = math.log(max(num_classes, 2))
    C_rec = max(n / K, 3.0)

    scores = {}
    for c in clients:
        cid = str(c.id)
        ent = ent_ema.get(cid, -1.0)
        var = ent_var_ema.get(cid, -1.0)

        if ent < 0:
            # No data yet: assign median score + exploration bonus
            info_score = 0.5
            exploration_bonus = 0.2
        else:
            ratio = ent / max_entropy
            is_noisy = (ratio > max_entropy_ratio) and (var < min_entropy_var)

            if is_noisy:
                info_score = ent * (1.0 - noise_penalty)
            else:
                # Reward high entropy with high variance (genuine uncertainty)
                info_score = ent * (1.0 + var)
            exploration_bonus = 0.0

        # Fairness
        gap = max(0, round_idx - (c.last_selected_round if c.last_selected_round is not None else -1))
        fairness = gap / (gap + C_rec)

        scores[c.id] = (1.0 - w_fairness) * info_score + w_fairness * fairness + exploration_bonus

    # Normalize scores
    all_scores = [scores[c.id] for c in clients]
    s_min = min(all_scores) if all_scores else 0
    s_max = max(all_scores) if all_scores else 1
    s_range = s_max - s_min if (s_max - s_min) > 1e-8 else 1.0
    norm_scores = {c.id: (scores[c.id] - s_min) / s_range for c in clients}

    # --- Step 3: Select top-K ---
    ranked = sorted(clients, key=lambda c: norm_scores[c.id], reverse=True)
    selected = [c.id for c in ranked[:K]]
    scores_out = [norm_scores[c.id] for c in ranked[:K]]

    new_state = {
        "lem": {
            "entropy_ema": ent_ema,
            "entropy_var_ema": ent_var_ema,
        }
    }

    return selected, scores_out, new_state
