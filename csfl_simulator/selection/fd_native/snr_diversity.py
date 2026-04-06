"""SNR-Aware Diversity Selector (SNRD) for Federated Distillation.

Jointly optimizes for label diversity and channel quality with an adaptive
tradeoff weight that shifts based on the estimated noise environment.

When channels are harsh -> prioritize channel quality (ensure logits arrive intact).
When channels are clean -> prioritize label diversity (maximize information content).
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
    # --- SNRD hyperparameters ---
    w_fairness: float = 0.15,
    noise_threshold: float = 1.0,
    channel_ema_alpha: float = 0.3,
    fixed_w_channel: Optional[float] = None,  # For ablation: override adaptive weight
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    state = history.get("state", {}).get("snrd", {})
    channel_ema = state.get("channel_ema", {})
    last_logit_var = state.get("last_logit_var", 1.0)

    # --- Step 1: Noise estimation & adaptive tradeoff ---
    if fixed_w_channel is not None:
        w_channel = float(fixed_w_channel)
    else:
        # Estimate effective noise from last round's logit variance
        # Higher noise_ratio -> more weight on channel quality
        noise_ratio = 0.0
        fd_stats = history.get("state", {})
        if "fd_logit_stats" in fd_stats:
            # Use effective noise variance if available
            eff_noise = sum(
                s.get("entropy_var", 0.0) for s in fd_stats["fd_logit_stats"].values()
            ) / max(len(fd_stats["fd_logit_stats"]), 1)
            noise_ratio = eff_noise / (last_logit_var + 1e-8)
        w_channel = 1.0 / (1.0 + math.exp(-(noise_ratio - noise_threshold)))

    w_diversity = max(0.0, 1.0 - w_channel - w_fairness)

    # --- Step 2: Per-client scoring ---
    # Channel quality EMA
    for c in clients:
        cid_str = str(c.id)
        old = channel_ema.get(cid_str, c.channel_quality)
        channel_ema[cid_str] = channel_ema_alpha * c.channel_quality + (1 - channel_ema_alpha) * old

    ch_vals = [channel_ema.get(str(c.id), c.channel_quality) for c in clients]
    ch_norm = normalize_list(ch_vals)

    # Fairness: recency bonus
    C_rec = max(n / K, 3.0)
    fairness_scores = []
    for c in clients:
        gap = max(0, round_idx - (c.last_selected_round if c.last_selected_round is not None else -1))
        fairness_scores.append(gap / (gap + C_rec))

    # --- Step 3: Greedy diversity-aware selection ---
    # Build label histograms
    num_classes = 0
    hists = []
    for c in clients:
        h = c.label_histogram
        if isinstance(h, dict):
            num_classes = max(num_classes, max((int(k) for k in h.keys()), default=0) + 1)
    if num_classes == 0:
        num_classes = 10  # fallback

    for c in clients:
        vec = np.zeros(num_classes, dtype=float)
        h = c.label_histogram
        if isinstance(h, dict):
            for k, v in h.items():
                idx = int(k)
                if 0 <= idx < num_classes:
                    vec[idx] = float(v)
        hists.append(vec)

    # IDF weights: upweight rare classes
    doc_freq = np.zeros(num_classes, dtype=float)
    for vec in hists:
        doc_freq += (vec > 0).astype(float)
    idf = np.log(n / (doc_freq + 1.0)) + 1.0

    selected = []
    selected_set = set()
    label_coverage = np.zeros(num_classes, dtype=float)
    scores_out = []

    for _ in range(min(K, n)):
        best_id = -1
        best_score = -float("inf")
        for idx, c in enumerate(clients):
            if c.id in selected_set:
                continue
            # Marginal diversity gain
            gain = 0.0
            vec = hists[idx]
            for cls in range(num_classes):
                if vec[cls] > 0 and label_coverage[cls] < 2:
                    gain += idf[cls] * (1.0 if label_coverage[cls] == 0 else 0.3)
            div_score = gain / (idf.sum() + 1e-8)

            score = (
                w_channel * ch_norm[idx]
                + w_diversity * div_score
                + w_fairness * fairness_scores[idx]
            )
            if score > best_score:
                best_score = score
                best_id = c.id
                best_idx = idx

        if best_id < 0:
            break
        selected.append(best_id)
        selected_set.add(best_id)
        scores_out.append(best_score)
        label_coverage += (hists[best_idx] > 0).astype(float)

    # Update state
    # Estimate logit variance from channel quality dispersion as proxy
    new_logit_var = float(np.var(ch_vals)) + 1e-8
    new_state = {
        "snrd": {
            "channel_ema": channel_ema,
            "last_logit_var": new_logit_var,
        }
    }

    return selected, scores_out, new_state
