"""CALM-FD: Confidence-blended Adaptive Logit-quality Matching for Federated Distillation.

Direct successor to LQTS (Logit-Quality Thompson Sampling), designed to address
four empirical weaknesses observed in LQTS on the 2026-04 batch:

1. **Reward is cosine-only.** LQTS rewards agreement with the aggregated logit
   (cosine sim to mean). Under severe DL noise (-20 dB), "agreement" can mean
   "commonly corrupted in the same direction". CALM-FD blends cosine-agreement
   with a **confidence** term derived from logit entropy — confident+agreeing
   logits are the desired distillation target; diffuse+agreeing logits are
   noise-consensus and should be down-weighted.

2. **Fixed exploration variance.** LQTS uses a constant `variance_floor_scale`
   regardless of noise regime. CALM-FD scales the posterior variance floor by
   an online **noise-regime estimator** (tanh of the round-mean
   effective_noise_var), so the bandit explores more aggressively when the
   channel is noisy and commits faster in clean regimes.

3. **No stale-posterior guard.** LQTS's posterior for a long-unselected client
   drifts to an arbitrary stale mean. CALM-FD re-inflates σ² for any client
   not selected in the last 2·N/K rounds, forcing re-examination when
   conditions may have changed.

4. **Channel signal is absent.** LQTS is purely logit-driven and ignores
   channel quality. But pure channel-aware methods (SNR-Diversity,
   Noise-Robust-Fair) underperform because they over-trust a noisy signal.
   CALM-FD uses channel quality only as a **hard exclusion** filter — drop
   only the bottom ``channel_floor_percentile`` % of clients per round, leave
   the rest to Thompson sampling. This is a minimal-intervention use of the
   channel that preserves LQTS's main objective.

5. **Anti-collusion is on logit-space in LQTS.** The empirical finding
   (Pearson r = −0.57) is that chasing *logit* diversity hurts accuracy.
   CALM-FD keeps anti-collusion but computes it on a **proxy feature space**
   (loss, grad-norm, label histogram) — never on logits. This makes diversity
   a regulariser against redundant selection rather than a reward in itself.

Canonical signature matches all other selectors in the repo.

Observed reward source (set by FDSimulator each round):
    history["state"]["fd_logit_rewards"][cid] ∈ [−1, 1]   (cosine to mean, data-weighted)
    history["state"]["fd_logit_stats"][cid]["entropy_mean"]
    history["state"]["fd_logit_stats"][cid]["entropy_var"]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), p))


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # --- reward blending ---
    w_cosine: float = 0.65,      # weight on cos-sim-to-mean
    w_confidence: float = 0.35,  # weight on (1 - normalised entropy)
    # --- posterior & exploration ---
    ema_alpha: float = 0.3,
    base_variance_floor: float = 0.05,
    noise_expand_gain: float = 2.0,     # multiplier range for σ² floor
    stale_rounds_factor: float = 2.0,   # stale cutoff = factor * N/K
    stale_variance_inflate: float = 0.30,
    # --- selection scoring ---
    w_ts: float = 0.70,
    w_recency: float = 0.15,
    w_collusion_penalty: float = 0.15,
    channel_floor_percentile: float = 5.0,
    # --- ablation toggles ---
    disable_confidence_blend: bool = False,
    disable_adaptive_variance: bool = False,
    disable_stale_guard: bool = False,
    disable_channel_filter: bool = False,
    disable_collusion_penalty: bool = False,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    state = history.get("state", {}).get("calm_fd", {})
    mu: Dict[str, float] = state.get("mu", {})
    sigma2: Dict[str, float] = state.get("sigma2", {})
    obs_n: Dict[str, int] = state.get("obs_n", {})
    ema_rew: Dict[str, float] = state.get("ema_rew", {})
    noise_ema: float = state.get("noise_ema", 0.0)

    # ---- 1. Noise-regime estimator (global) ----
    metrics_hist = history.get("metrics", [])
    if metrics_hist and not disable_adaptive_variance:
        last_noise = float(metrics_hist[-1].get("effective_noise_var", 0.0) or 0.0)
        # tanh squash to [0, 1] — noise_var ~ 10 saturates
        squashed = math.tanh(last_noise / 10.0)
        noise_ema = 0.9 * noise_ema + 0.1 * squashed

    var_floor = base_variance_floor
    if not disable_adaptive_variance:
        var_floor *= (1.0 + noise_expand_gain * noise_ema)

    # ---- 2. Reward extraction (with confidence blend) ----
    raw_rewards: Dict[str, float] = history.get("state", {}).get("fd_logit_rewards", {}) or {}
    logit_stats: Dict[str, Dict[str, float]] = history.get("state", {}).get("fd_logit_stats", {}) or {}

    num_classes = 0
    for c in clients:
        h = c.label_histogram
        if isinstance(h, dict) and h:
            num_classes = max(num_classes, max(int(k) for k in h.keys()) + 1)
    num_classes = max(num_classes, 10)
    max_entropy = math.log(num_classes)

    blended_rewards: Dict[str, float] = {}
    for cid_raw, r_cos in raw_rewards.items():
        cid = str(cid_raw)
        r = float(r_cos)
        if not disable_confidence_blend:
            stats = logit_stats.get(int(cid_raw)) or logit_stats.get(cid_raw) or logit_stats.get(cid) or {}
            if isinstance(stats, dict) and "entropy_mean" in stats:
                ent = float(stats["entropy_mean"])
                confidence = 1.0 - min(max(ent / max_entropy, 0.0), 1.0)
                r = w_cosine * r + w_confidence * confidence
        blended_rewards[cid] = r

    # ---- 3. Posterior update for last-round selections ----
    for cid, r in blended_rewards.items():
        old_ema = ema_rew.get(cid, 0.5)
        new_ema = ema_alpha * r + (1 - ema_alpha) * old_ema
        ema_rew[cid] = new_ema
        count = obs_n.get(cid, 0) + 1
        obs_n[cid] = count
        mu[cid] = new_ema
        old_var = sigma2.get(cid, 1.0)
        sigma2[cid] = max(var_floor, 0.5 * old_var + 0.5 * (r - new_ema) ** 2)

    # ---- 4. Stale-posterior guard ----
    stale_cutoff = stale_rounds_factor * (n / max(K, 1))
    if not disable_stale_guard:
        for c in clients:
            cid = str(c.id)
            last_sel = c.last_selected_round if c.last_selected_round is not None else -1
            gap = round_idx - last_sel
            if gap > stale_cutoff and cid in sigma2:
                sigma2[cid] = max(sigma2[cid], stale_variance_inflate)

    # ---- 5. Thompson draw per client ----
    samples: Dict[int, float] = {}
    for c in clients:
        cid = str(c.id)
        m = mu.get(cid, 0.5)
        s2 = sigma2.get(cid, 1.0)
        count = max(obs_n.get(cid, 0), 1)
        std = math.sqrt(s2 / count)
        samples[c.id] = rng.gauss(m, std)

    # ---- 6. Channel floor exclusion ----
    eligible_ids = set(c.id for c in clients)
    if not disable_channel_filter and n > K + 2:
        channel_vals = [float(c.channel_quality or 0.0) for c in clients]
        q_thr = _percentile(channel_vals, channel_floor_percentile)
        excluded = set()
        for c in clients:
            if float(c.channel_quality or 0.0) < q_thr:
                # But only if they've been observed recently (don't starve cold starters)
                if obs_n.get(str(c.id), 0) > 0:
                    excluded.add(c.id)
        # Never exclude so many that we can't fill K
        if len(eligible_ids - excluded) >= K:
            eligible_ids = eligible_ids - excluded

    # ---- 7. Proxy feature vectors (for anti-collusion) ----
    proxy_vecs: Dict[int, np.ndarray] = {}
    for c in clients:
        vec = np.zeros(num_classes + 2, dtype=float)
        h = c.label_histogram
        if isinstance(h, dict):
            for k, v in h.items():
                idx = int(k)
                if 0 <= idx < num_classes:
                    vec[idx] = float(v)
        vec[num_classes] = float(c.last_loss or 0.0)
        vec[num_classes + 1] = float(c.grad_norm or 0.0)
        norm = np.linalg.norm(vec) + 1e-8
        proxy_vecs[c.id] = vec / norm

    C_rec = max(n / max(K, 1), 3.0)

    # ---- 8. Greedy top-K with TS score + recency + anti-collusion penalty ----
    selected: List[int] = []
    selected_set: set = set()
    selected_vecs: List[np.ndarray] = []
    scores_out: List[float] = []

    for _ in range(min(K, n)):
        best_id = -1
        best_score = -float("inf")
        for c in clients:
            if c.id in selected_set or c.id not in eligible_ids:
                continue

            gap = max(0, round_idx - (c.last_selected_round if c.last_selected_round is not None else -1))
            recency = gap / (gap + C_rec)

            if selected_vecs and not disable_collusion_penalty:
                max_sim = max(float(np.dot(proxy_vecs[c.id], sv)) for sv in selected_vecs)
                # penalise when we're about to pick a proxy-similar client
                collusion = max_sim
            else:
                collusion = 0.0

            score = (
                w_ts * samples[c.id]
                + w_recency * recency
                - w_collusion_penalty * collusion
            )
            if score > best_score:
                best_score = score
                best_id = c.id

        if best_id < 0:
            break
        selected.append(best_id)
        selected_set.add(best_id)
        selected_vecs.append(proxy_vecs[best_id])
        scores_out.append(best_score)

    # Safety fallback: if channel filter left us short, refill from excluded
    while len(selected) < K:
        for c in clients:
            if c.id not in selected_set:
                selected.append(c.id)
                selected_set.add(c.id)
                scores_out.append(-1.0)
                if len(selected) >= K:
                    break
        if len(selected) >= K:
            break

    new_state = {
        "calm_fd": {
            "mu": mu,
            "sigma2": sigma2,
            "obs_n": obs_n,
            "ema_rew": ema_rew,
            "noise_ema": noise_ema,
        }
    }
    return selected, scores_out, new_state
