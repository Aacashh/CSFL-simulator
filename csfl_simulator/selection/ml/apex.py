"""APEX: Adaptive Phase-aware EXploration for Federated Client Selection.

A lightweight O(N*L + K^2*L) client selection method combining:
  - Online phase detection (critical/transition/exploitation)
  - Contextual Thompson Sampling with Bayesian posteriors
  - Label-histogram-based gradient diversity proxy
  - Phase-adaptive weight blending

Theoretical motivation (non-IID FL convergence bound):
  E[||w_t - w*||^2] <= (1 - eta*mu)^t ||w_0 - w*||^2 + C1*Gamma/(eta*mu) + C2*sigma^2/(eta*K)

  APEX minimizes Gamma (client drift) via loss-biased Thompson sampling
  and sigma^2/K (gradient variance) via label-diversity-based proxy selection.
  Phase detection adapts the balance between these objectives across training.

Reference: docs/APEX_method_reference.md
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import choose_random, expected_duration

_STATE_KEY = "apex_state"

# ---------------------------------------------------------------------------
# Phase-dependent weight presets: (thompson, diversity, recency)
# ---------------------------------------------------------------------------
_PHASE_WEIGHTS = {
    "critical":      (0.20, 0.60, 0.20),
    "transition":    (0.50, 0.30, 0.20),
    "exploitation":  (0.70, 0.15, 0.15),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(x: float) -> float:
    """Coerce to finite float, default 0."""
    if x is None:
        return 0.0
    v = float(x)
    return v if math.isfinite(v) else 0.0


def _norm(xs: List[float]) -> List[float]:
    """Min-max normalize a list to [0, 1]."""
    if not xs:
        return xs
    lo = min(xs)
    hi = max(xs)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo + 1e-12) for x in xs]


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
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    v /= n
    return v


def _build_proxy(loss_n: float, gnorm_n: float, hist_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build gradient diversity proxy vector: [norm_loss, norm_grad, label_hist...]."""
    if hist_vec is None:
        return None
    return np.concatenate([[loss_n, gnorm_n], hist_vec])


def _min_cosine_distance(cand: np.ndarray, selected: List[np.ndarray]) -> float:
    """Min cosine distance from candidate to any selected vector. Higher = more diverse."""
    if not selected:
        return 1.0
    min_sim = min(float(np.dot(cand, sv)) for sv in selected)
    return 1.0 - min_sim


def _detect_phase(
    loss_history: List[float],
    W: int,
    tau_critical: float,
    tau_unstable: float,
    tau_exploit: float,
) -> str:
    """Detect training phase from loss trajectory."""
    if len(loss_history) < W:
        return "critical"

    recent = loss_history[-W:]
    older = loss_history[-2 * W:-W] if len(loss_history) >= 2 * W else loss_history[:W]

    mean_recent = sum(recent) / len(recent)
    mean_older = sum(older) / len(older)

    # Relative improvement rate
    rate = (mean_older - mean_recent) / (mean_older + 1e-12)

    # Coefficient of variation of recent window (stability)
    if mean_recent > 1e-12:
        var_recent = sum((x - mean_recent) ** 2 for x in recent) / len(recent)
        cv = math.sqrt(var_recent) / (mean_recent + 1e-12)
    else:
        cv = 0.0

    if rate > tau_critical and cv > tau_unstable:
        return "critical"
    elif rate > tau_exploit:
        return "transition"
    else:
        return "exploitation"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # Phase detection hyperparameters
    W_phase: int = 5,
    tau_critical: float = 0.05,
    tau_unstable: float = 0.10,
    tau_exploit: float = 0.01,
    # Thompson sampling hyperparameters
    gamma: float = 0.3,
    w_loss: float = 0.4,
    w_grad: float = 0.2,
    w_speed: float = 0.2,
    w_data: float = 0.2,
    # Phase weight overrides (tuples: thompson, diversity, recency)
    phase_weights_critical: Optional[Tuple[float, float, float]] = None,
    phase_weights_transition: Optional[Tuple[float, float, float]] = None,
    phase_weights_exploit: Optional[Tuple[float, float, float]] = None,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """APEX: Adaptive Phase-aware EXploration for client selection.

    Parameters
    ----------
    W_phase : int
        Window size for phase detection (rounds).
    tau_critical, tau_unstable, tau_exploit : float
        Phase classification thresholds.
    gamma : float
        Blend weight for Thompson sample vs contextual utility (0=pure context, 1=pure Thompson).
    w_loss, w_grad, w_speed, w_data : float
        Weights for contextual utility components.
    phase_weights_critical/transition/exploit : tuple of 3 floats, optional
        Override (thompson_wt, diversity_wt, recency_wt) for each phase.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    # === 1. Retrieve / initialize persistent state ===
    st = history.get("state", {}).get(_STATE_KEY, None)
    if st is None:
        st = {
            "loss_history": [],
            "ts_alpha": {},
            "ts_beta": {},
            "ts_mu": {},
            "ts_sigma2": {},
            "ts_n": {},
        }
    # Ensure all dicts exist (guard against partial state)
    for key in ("ts_alpha", "ts_beta", "ts_mu", "ts_sigma2", "ts_n"):
        if key not in st:
            st[key] = {}
    if "loss_history" not in st:
        st["loss_history"] = []

    # === 2. Update Thompson posteriors from last round's reward ===
    last_reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        credit = last_reward / max(1, len(last_sel))
        for cid in last_sel:
            cid_s = str(cid)  # JSON keys may be strings
            n_i = int(st["ts_n"].get(cid_s, st["ts_n"].get(cid, 0))) + 1
            old_mu = float(st["ts_mu"].get(cid_s, st["ts_mu"].get(cid, 0.0)))

            delta = credit - old_mu
            new_mu = old_mu + delta / n_i
            old_var = float(st["ts_sigma2"].get(cid_s, st["ts_sigma2"].get(cid, 1.0)))
            new_var = ((n_i - 1) * old_var + delta * (credit - new_mu)) / max(n_i, 1)

            st["ts_n"][cid] = n_i
            st["ts_mu"][cid] = new_mu
            st["ts_sigma2"][cid] = max(new_var, 1e-8)

            # Beta update for cold-start fallback
            if credit > 0:
                st["ts_alpha"][cid] = float(st["ts_alpha"].get(cid_s, st["ts_alpha"].get(cid, 1.0))) + min(credit, 1.0)
            else:
                st["ts_beta"][cid] = float(st["ts_beta"].get(cid_s, st["ts_beta"].get(cid, 1.0))) + min(abs(credit), 1.0)

    # === 3. Phase detection ===
    # Compute average loss from clients selected last round
    if last_sel:
        loss_vals = []
        id_map = {c.id: c for c in clients}
        for cid in last_sel:
            c = id_map.get(cid)
            if c and c.last_loss and c.last_loss > 0:
                loss_vals.append(float(c.last_loss))
        if loss_vals:
            st["loss_history"].append(sum(loss_vals) / len(loss_vals))

    phase = _detect_phase(
        st["loss_history"], W_phase, tau_critical, tau_unstable, tau_exploit
    )

    # === 4. Get phase-dependent weights ===
    pw_crit = phase_weights_critical or _PHASE_WEIGHTS["critical"]
    pw_trans = phase_weights_transition or _PHASE_WEIGHTS["transition"]
    pw_expl = phase_weights_exploit or _PHASE_WEIGHTS["exploitation"]

    if phase == "critical":
        w_ts, w_div, w_rec = pw_crit
    elif phase == "transition":
        w_ts, w_div, w_rec = pw_trans
    else:
        w_ts, w_div, w_rec = pw_expl

    # === 5. Cold start: if no client has loss info, do data-size-weighted random ===
    has_loss = any(c.last_loss and c.last_loss > 0 for c in clients)
    if not has_loss:
        # Weight by data size for a smarter cold start than uniform
        weights = [max(float(c.data_size or 1), 1.0) for c in clients]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        ids = [c.id for c in clients]
        selected = []
        remaining_ids = list(ids)
        remaining_probs = list(probs)
        for _ in range(min(K, n)):
            if not remaining_ids:
                break
            # Weighted sampling without replacement
            r = rng.random()
            cumulative = 0.0
            chosen_idx = len(remaining_ids) - 1
            for j, p in enumerate(remaining_probs):
                cumulative += p
                if r <= cumulative:
                    chosen_idx = j
                    break
            selected.append(remaining_ids[chosen_idx])
            remaining_ids.pop(chosen_idx)
            remaining_probs.pop(chosen_idx)
            # Renormalize
            s = sum(remaining_probs)
            if s > 0:
                remaining_probs = [p / s for p in remaining_probs]
        return selected, None, {_STATE_KEY: st}

    # === 6. Compute normalized features ===
    losses = _norm([_safe(c.last_loss) for c in clients])
    gnorms = _norm([_safe(c.grad_norm) for c in clients])
    speeds = _norm([1.0 / max(1e-6, expected_duration(c)) for c in clients])
    dsizes = _norm([float(c.data_size or 0) for c in clients])

    # === 7. Build gradient diversity proxy vectors ===
    L = _collect_label_space(clients)
    proxy_vecs: Dict[int, Optional[np.ndarray]] = {}
    for i, c in enumerate(clients):
        hv = _hist_to_vec(c.label_histogram, L)
        proxy_vecs[c.id] = _build_proxy(losses[i], gnorms[i], hv)

    # === 8. Score all clients (Thompson + contextual) ===
    id_to_client: Dict[int, ClientInfo] = {c.id: c for c in clients}
    base_scores: Dict[int, float] = {}

    for i, c in enumerate(clients):
        # Contextual utility
        ctx = w_loss * losses[i] + w_grad * gnorms[i] + w_speed * speeds[i] + w_data * dsizes[i]

        # Thompson sample
        cid = c.id
        n_i = int(st["ts_n"].get(cid, 0))
        if n_i >= 2:
            mu_i = float(st["ts_mu"].get(cid, 0.0))
            var_i = max(float(st["ts_sigma2"].get(cid, 1.0)), 1e-8)
            ts_sample = rng.gauss(mu_i, math.sqrt(var_i / max(n_i, 1)))
        else:
            a = float(st["ts_alpha"].get(cid, 1.0))
            b = float(st["ts_beta"].get(cid, 1.0))
            ts_sample = rng.betavariate(max(a, 0.01), max(b, 0.01))

        # Blend contextual + Thompson
        blended = (1.0 - gamma) * ctx + gamma * ts_sample

        # Recency bonus (starvation prevention)
        last_sel_round = getattr(c, "last_selected_round", -1)
        if last_sel_round is None or last_sel_round < 0:
            gap = float(round_idx + 1)
        else:
            gap = float(round_idx - last_sel_round)
        recency = gap / (gap + 5.0)

        # Phase-weighted combination (diversity added during greedy)
        base_scores[cid] = w_ts * blended + w_rec * recency

    # === 9. Greedy selection with diversity ===
    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []
    remaining_budget = float(time_budget) if time_budget is not None else None
    pool = set(id_to_client.keys())

    for _ in range(K):
        best_id = -1
        best_score = -float("inf")

        for cid in pool:
            # Time budget feasibility
            if remaining_budget is not None:
                dur = expected_duration(id_to_client[cid])
                if dur > remaining_budget:
                    continue

            # Diversity bonus
            div_bonus = 0.0
            pv = proxy_vecs.get(cid)
            if pv is not None and w_div > 0:
                div_bonus = _min_cosine_distance(pv, selected_vecs)

            score = base_scores[cid] + w_div * div_bonus

            if score > best_score:
                best_score = score
                best_id = cid

        if best_id < 0:
            break

        selected.append(best_id)
        pool.discard(best_id)
        pv = proxy_vecs.get(best_id)
        if pv is not None:
            selected_vecs.append(pv)
        if remaining_budget is not None:
            remaining_budget -= expected_duration(id_to_client[best_id])

    # Fill remaining slots randomly if budget prevented filling K
    if len(selected) < K:
        remaining = [cid for cid in pool if cid not in set(selected)]
        rng.shuffle(remaining)
        for cid in remaining[: K - len(selected)]:
            selected.append(cid)

    # Per-client scores for visualization (aligned to clients list)
    scores_out = [base_scores.get(c.id, 0.0) for c in clients]

    return selected, scores_out, {_STATE_KEY: st}
