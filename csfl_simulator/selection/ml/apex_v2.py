"""APEX v2: Adaptive Phase-aware EXploration for Federated Client Selection.

Builds on APEX v1 with five targeted fixes derived from empirical failure analysis:

  Fix 1 - Adaptive recency scaling: `gap/(gap + N/K)` instead of `gap/(gap + 5)`
           Prevents round-robin collapse at large client pools.

  Fix 2 - Phase hysteresis: minimum dwell time + no phase skipping
           (critical <-> exploitation must pass through transition).
           Eliminates destructive boom-bust oscillations under extreme non-IID.

  Fix 3 - Heterogeneity-aware diversity: diversity weight is scaled by an
           online Jensen-Shannon divergence estimate of data heterogeneity.
           Avoids over-exploration when data is near-IID.

  Fix 4 - Thompson posterior regularization: variance floor 0.1/sqrt(n)
           instead of 1e-8, plus EMA-smoothed reward signals.
           Prevents overconfident exploitation with sparse observations.

  Fix 5 - Confidence-scaled gamma: blend weight adapts per-client based
           on observation count. New clients trust context; observed clients
           trust posteriors.

Complexity: O(N*L + K^2*L)  (same as v1, heterogeneity estimate cached)
Trainable parameters: 0

Reference: docs/APEX_v2_proposal.md
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import choose_random, expected_duration

_STATE_KEY = "apex_v2_state"

# ---------------------------------------------------------------------------
# Phase-dependent weight presets: (thompson, diversity, recency)
# ---------------------------------------------------------------------------
_PHASE_WEIGHTS = {
    "critical":      (0.20, 0.60, 0.20),
    "transition":    (0.50, 0.30, 0.20),
    "exploitation":  (0.70, 0.15, 0.15),
}

_PHASE_ORDER = {"critical": 0, "transition": 1, "exploitation": 2}


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


# ---------------------------------------------------------------------------
# Fix 2: Phase detection with hysteresis
# ---------------------------------------------------------------------------

def _detect_phase_with_hysteresis(
    loss_history: List[float],
    W: int,
    tau_critical: float,
    tau_unstable: float,
    tau_exploit: float,
    prev_phase: str,
    phase_age: int,
    min_dwell: int = 3,
) -> Tuple[str, int]:
    """Detect training phase from loss trajectory with hysteresis.

    Returns (phase, new_phase_age).

    Hysteresis rules:
      - Must stay in a phase for at least `min_dwell` rounds before transitioning.
      - Cannot jump critical <-> exploitation; must pass through transition.
    """
    if len(loss_history) < W:
        if prev_phase == "critical":
            return "critical", phase_age + 1
        return "critical", 0

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

    # Raw phase classification (same logic as v1)
    if rate > tau_critical and cv > tau_unstable:
        raw_phase = "critical"
    elif rate > tau_exploit:
        raw_phase = "transition"
    else:
        raw_phase = "exploitation"

    # Hysteresis: stay in current phase if dwell time not met
    if raw_phase == prev_phase:
        return prev_phase, phase_age + 1

    if phase_age < min_dwell:
        return prev_phase, phase_age + 1

    # No jumping: critical <-> exploitation must pass through transition
    if abs(_PHASE_ORDER[raw_phase] - _PHASE_ORDER[prev_phase]) > 1:
        return "transition", 0

    return raw_phase, 0


# ---------------------------------------------------------------------------
# Fix 3: Online heterogeneity estimation
# ---------------------------------------------------------------------------

def _estimate_heterogeneity(clients: List[ClientInfo], L: int) -> float:
    """Estimate data heterogeneity from label histogram divergence.

    Returns a value in [0, 1] where 0 = IID, 1 = extreme non-IID.
    Uses average pairwise Jensen-Shannon divergence (sampled).
    """
    if L <= 0:
        return 0.5  # Unknown

    # Build probability distributions per client
    dists = []
    for c in clients:
        if not isinstance(c.label_histogram, dict) or not c.label_histogram:
            continue
        v = np.zeros(L, dtype=np.float64)
        for k, cnt in c.label_histogram.items():
            try:
                idx = int(k)
                if 0 <= idx < L:
                    v[idx] = float(cnt)
            except (ValueError, TypeError):
                pass
        s = v.sum()
        if s > 0:
            dists.append(v / s)

    if len(dists) < 2:
        return 0.5

    # Sample pairwise JSD (cap at 200 pairs for efficiency)
    n_clients = len(dists)
    max_pairs = min(200, n_clients * (n_clients - 1) // 2)
    jsds = []

    # Deterministic sampling: evenly spaced pairs
    step = max(1, (n_clients * (n_clients - 1) // 2) // max_pairs)
    pair_idx = 0
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            if pair_idx % step == 0:
                m = 0.5 * (dists[i] + dists[j])
                # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)
                # Use safe log to avoid log(0)
                kl_pm = 0.0
                kl_qm = 0.0
                for k in range(L):
                    if dists[i][k] > 1e-12 and m[k] > 1e-12:
                        kl_pm += dists[i][k] * math.log(dists[i][k] / m[k])
                    if dists[j][k] > 1e-12 and m[k] > 1e-12:
                        kl_qm += dists[j][k] * math.log(dists[j][k] / m[k])
                jsd = 0.5 * kl_pm + 0.5 * kl_qm
                jsds.append(math.sqrt(max(jsd, 0.0)))  # JSD distance (sqrt)
                if len(jsds) >= max_pairs:
                    break
            pair_idx += 1
        if len(jsds) >= max_pairs:
            break

    if not jsds:
        return 0.5

    avg_jsd = sum(jsds) / len(jsds)
    # JSD distance (sqrt of JSD) ranges from 0 to ~0.83 for 10-class
    # Normalize to [0, 1] with a sigmoid-like mapping
    return min(avg_jsd / 0.6, 1.0)


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
    min_dwell: int = 3,               # Fix 2: hysteresis dwell time
    # Thompson sampling hyperparameters
    gamma: float = 0.3,
    w_loss: float = 0.4,
    w_grad: float = 0.2,
    w_speed: float = 0.2,
    w_data: float = 0.2,
    reward_ema_alpha: float = 0.3,     # Fix 4: EMA smoothing for rewards
    # Phase weight overrides (tuples: thompson, diversity, recency)
    phase_weights_critical: Optional[Tuple[float, float, float]] = None,
    phase_weights_transition: Optional[Tuple[float, float, float]] = None,
    phase_weights_exploit: Optional[Tuple[float, float, float]] = None,
    # Fix toggles (for ablation)
    use_adaptive_recency: bool = True,  # Fix 1
    use_phase_hysteresis: bool = True,  # Fix 2
    use_het_scaling: bool = True,       # Fix 3
    use_posterior_reg: bool = True,     # Fix 4
    use_adaptive_gamma: bool = True,    # Fix 5
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """APEX v2: Adaptive Phase-aware EXploration with self-calibration.

    All five fixes from the empirical failure analysis are applied by default.
    Individual fixes can be toggled off for ablation studies.
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
            "ts_reward_ema": {},      # Fix 4: per-client EMA of rewards
            "prev_phase": "critical",  # Fix 2: hysteresis state
            "phase_age": 0,            # Fix 2: rounds in current phase
            "heterogeneity": None,     # Fix 3: cached JSD estimate
        }
    # Ensure all keys exist (guard against partial state from v1)
    for key in ("ts_alpha", "ts_beta", "ts_mu", "ts_sigma2", "ts_n", "ts_reward_ema"):
        if key not in st:
            st[key] = {}
    if "loss_history" not in st:
        st["loss_history"] = []
    if "prev_phase" not in st:
        st["prev_phase"] = "critical"
    if "phase_age" not in st:
        st["phase_age"] = 0
    if "heterogeneity" not in st:
        st["heterogeneity"] = None

    # === 2. Update Thompson posteriors from last round's reward ===
    last_reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        raw_credit = last_reward / max(1, len(last_sel))
        for cid in last_sel:
            cid_s = str(cid)

            # Fix 4: EMA-smoothed reward signal
            if use_posterior_reg:
                old_ema = float(st["ts_reward_ema"].get(cid_s, st["ts_reward_ema"].get(cid, raw_credit)))
                credit = reward_ema_alpha * raw_credit + (1.0 - reward_ema_alpha) * old_ema
                st["ts_reward_ema"][cid] = credit
            else:
                credit = raw_credit

            n_i = int(st["ts_n"].get(cid_s, st["ts_n"].get(cid, 0))) + 1
            old_mu = float(st["ts_mu"].get(cid_s, st["ts_mu"].get(cid, 0.0)))

            delta = credit - old_mu
            new_mu = old_mu + delta / n_i
            old_var = float(st["ts_sigma2"].get(cid_s, st["ts_sigma2"].get(cid, 1.0)))
            new_var = ((n_i - 1) * old_var + delta * (credit - new_mu)) / max(n_i, 1)

            st["ts_n"][cid] = n_i
            st["ts_mu"][cid] = new_mu

            # Fix 4: Variance floor 0.1/sqrt(n) instead of 1e-8
            if use_posterior_reg:
                sigma2_floor = 0.1 / math.sqrt(max(n_i, 1))
            else:
                sigma2_floor = 1e-8
            st["ts_sigma2"][cid] = max(new_var, sigma2_floor)

            # Beta update for cold-start fallback
            if credit > 0:
                st["ts_alpha"][cid] = float(st["ts_alpha"].get(cid_s, st["ts_alpha"].get(cid, 1.0))) + min(credit, 1.0)
            else:
                st["ts_beta"][cid] = float(st["ts_beta"].get(cid_s, st["ts_beta"].get(cid, 1.0))) + min(abs(credit), 1.0)

    # === 3. Phase detection ===
    if last_sel:
        loss_vals = []
        id_map = {c.id: c for c in clients}
        for cid in last_sel:
            c = id_map.get(cid)
            if c and c.last_loss and c.last_loss > 0:
                loss_vals.append(float(c.last_loss))
        if loss_vals:
            st["loss_history"].append(sum(loss_vals) / len(loss_vals))

    # Fix 2: Phase detection with hysteresis
    if use_phase_hysteresis:
        phase, new_age = _detect_phase_with_hysteresis(
            st["loss_history"], W_phase, tau_critical, tau_unstable, tau_exploit,
            prev_phase=st["prev_phase"],
            phase_age=st["phase_age"],
            min_dwell=min_dwell,
        )
        st["prev_phase"] = phase
        st["phase_age"] = new_age
    else:
        # v1 behavior: stateless phase detection
        if len(st["loss_history"]) < W_phase:
            phase = "critical"
        else:
            recent = st["loss_history"][-W_phase:]
            older = st["loss_history"][-2 * W_phase:-W_phase] if len(st["loss_history"]) >= 2 * W_phase else st["loss_history"][:W_phase]
            mean_recent = sum(recent) / len(recent)
            mean_older = sum(older) / len(older)
            rate = (mean_older - mean_recent) / (mean_older + 1e-12)
            if mean_recent > 1e-12:
                var_recent = sum((x - mean_recent) ** 2 for x in recent) / len(recent)
                cv = math.sqrt(var_recent) / (mean_recent + 1e-12)
            else:
                cv = 0.0
            if rate > tau_critical and cv > tau_unstable:
                phase = "critical"
            elif rate > tau_exploit:
                phase = "transition"
            else:
                phase = "exploitation"

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

    # === 4b. Fix 3: Scale diversity weight by heterogeneity ===
    L = _collect_label_space(clients)
    if use_het_scaling:
        # Compute once and cache (heterogeneity doesn't change across rounds)
        if st["heterogeneity"] is None:
            st["heterogeneity"] = _estimate_heterogeneity(clients, L)
        het = st["heterogeneity"]

        # Scale diversity weight; redistribute to Thompson
        w_div_original = w_div
        w_div = w_div * het
        w_ts = w_ts + (w_div_original - w_div)

    # === 5. Cold start: if no client has loss info, do data-size-weighted random ===
    has_loss = any(c.last_loss and c.last_loss > 0 for c in clients)
    if not has_loss:
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
    proxy_vecs: Dict[int, Optional[np.ndarray]] = {}
    for i, c in enumerate(clients):
        hv = _hist_to_vec(c.label_histogram, L)
        proxy_vecs[c.id] = _build_proxy(losses[i], gnorms[i], hv)

    # === 8. Score all clients (Thompson + contextual) ===
    id_to_client: Dict[int, ClientInfo] = {c.id: c for c in clients}
    base_scores: Dict[int, float] = {}

    # Fix 1: Adaptive recency constant
    if use_adaptive_recency:
        C_rec = max(float(n) / max(K, 1), 3.0)
    else:
        C_rec = 5.0

    for i, c in enumerate(clients):
        # Contextual utility
        ctx = w_loss * losses[i] + w_grad * gnorms[i] + w_speed * speeds[i] + w_data * dsizes[i]

        # Thompson sample
        cid = c.id
        n_i = int(st["ts_n"].get(cid, 0))

        # Fix 5: Confidence-scaled gamma
        if use_adaptive_gamma:
            confidence = 1.0 - 1.0 / (1.0 + n_i)  # 0 at n=0, 0.5 at n=1, 0.9 at n=9
            effective_gamma = gamma * confidence
        else:
            effective_gamma = gamma

        if n_i >= 2:
            mu_i = float(st["ts_mu"].get(cid, 0.0))
            var_i = float(st["ts_sigma2"].get(cid, 1.0))
            # Fix 4: floor already applied during update, but guard here too
            if use_posterior_reg:
                var_i = max(var_i, 0.1 / math.sqrt(max(n_i, 1)))
            else:
                var_i = max(var_i, 1e-8)
            ts_sample = rng.gauss(mu_i, math.sqrt(var_i / max(n_i, 1)))
        else:
            a = float(st["ts_alpha"].get(cid, 1.0))
            b = float(st["ts_beta"].get(cid, 1.0))
            ts_sample = rng.betavariate(max(a, 0.01), max(b, 0.01))

        # Blend contextual + Thompson
        blended = (1.0 - effective_gamma) * ctx + effective_gamma * ts_sample

        # Fix 1: Adaptive recency bonus
        last_sel_round = getattr(c, "last_selected_round", -1)
        if last_sel_round is None or last_sel_round < 0:
            gap = float(round_idx + 1)
        else:
            gap = float(round_idx - last_sel_round)
        recency = gap / (gap + C_rec)

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
            if remaining_budget is not None:
                dur = expected_duration(id_to_client[cid])
                if dur > remaining_budget:
                    continue

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

    scores_out = [base_scores.get(c.id, 0.0) for c in clients]

    return selected, scores_out, {_STATE_KEY: st}
