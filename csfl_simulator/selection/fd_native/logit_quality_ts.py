"""Logit-Quality Thompson Sampling (LQTS) for Federated Distillation.

Extends Thompson sampling by using per-client logit quality (cosine similarity
to aggregated mean) as the reward signal instead of global accuracy delta.
This directly measures each client's contribution to the distillation pool.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

import numpy as np

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
    # --- LQTS hyperparameters ---
    ema_alpha: float = 0.3,
    variance_floor_scale: float = 0.1,
    w_diversity: float = 0.25,
    w_recency: float = 0.15,
    use_global_reward: bool = False,  # Ablation: use global accuracy delta
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    state = history.get("state", {}).get("lqts", {})
    mu = state.get("mu", {})          # posterior means
    sigma2 = state.get("sigma2", {})  # posterior variances
    obs_n = state.get("obs_n", {})    # observation counts
    ema_rew = state.get("ema_rew", {})

    # --- Step 1: Retrieve logit-quality rewards from last round ---
    if use_global_reward:
        # Ablation: distribute global reward uniformly
        last_selected = history.get("selected", [[]])[-1] if history.get("selected") else []
        global_reward = history.get("state", {}).get("last_reward", 0.5)
        rewards = {cid: global_reward for cid in last_selected}
    else:
        rewards = history.get("state", {}).get("fd_logit_rewards", {})

    # --- Step 2: Update Thompson posteriors ---
    for cid_int, raw_reward in rewards.items():
        cid = str(cid_int)
        raw = float(raw_reward)
        old_ema = ema_rew.get(cid, 0.5)
        new_ema = ema_alpha * raw + (1 - ema_alpha) * old_ema
        ema_rew[cid] = new_ema

        count = obs_n.get(cid, 0) + 1
        obs_n[cid] = count
        mu[cid] = new_ema

        # Variance: floor decays with observations
        floor = variance_floor_scale / math.sqrt(max(count, 1))
        # Online variance estimate
        old_var = sigma2.get(cid, 1.0)
        sigma2[cid] = max(floor, 0.5 * old_var + 0.5 * (raw - new_ema) ** 2)

    # --- Step 3: Thompson sampling ---
    samples = {}
    for c in clients:
        cid = str(c.id)
        m = mu.get(cid, 0.5)
        s2 = sigma2.get(cid, 1.0)
        count = max(obs_n.get(cid, 0), 1)
        std = math.sqrt(s2 / count)
        sample = rng.gauss(m, std)
        samples[c.id] = sample

    # --- Step 4: Diversity-augmented greedy selection ---
    # Build label proxy vectors for diversity
    num_classes = 0
    proxy_vecs = {}
    for c in clients:
        h = c.label_histogram
        if isinstance(h, dict):
            num_classes = max(num_classes, max((int(k) for k in h.keys()), default=0) + 1)
    if num_classes == 0:
        num_classes = 10

    for c in clients:
        vec = np.zeros(num_classes + 2, dtype=float)
        h = c.label_histogram
        if isinstance(h, dict):
            for k, v in h.items():
                idx = int(k)
                if 0 <= idx < num_classes:
                    vec[idx] = float(v)
        # Append normalized loss and grad_norm
        vec[num_classes] = float(c.last_loss or 0.0)
        vec[num_classes + 1] = float(c.grad_norm or 0.0)
        norm = np.linalg.norm(vec) + 1e-8
        proxy_vecs[c.id] = vec / norm

    C_rec = max(n / K, 3.0)
    w_ts = 1.0 - w_diversity - w_recency

    selected = []
    selected_set = set()
    selected_vecs = []
    scores_out = []

    for _ in range(min(K, n)):
        best_id = -1
        best_score = -float("inf")
        for c in clients:
            if c.id in selected_set:
                continue
            # Recency
            gap = max(0, round_idx - (c.last_selected_round if c.last_selected_round is not None else -1))
            recency = gap / (gap + C_rec)

            # Diversity
            if selected_vecs:
                dists = [1.0 - float(np.dot(proxy_vecs[c.id], sv)) for sv in selected_vecs]
                div = min(dists)
            else:
                div = 1.0

            score = w_ts * samples[c.id] + w_diversity * div + w_recency * recency
            if score > best_score:
                best_score = score
                best_id = c.id

        if best_id < 0:
            break
        selected.append(best_id)
        selected_set.add(best_id)
        selected_vecs.append(proxy_vecs[best_id])
        scores_out.append(best_score)

    new_state = {
        "lqts": {
            "mu": mu,
            "sigma2": sigma2,
            "obs_n": obs_n,
            "ema_rew": ema_rew,
        }
    }

    return selected, scores_out, new_state
