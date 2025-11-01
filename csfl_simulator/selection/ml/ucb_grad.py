from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import math
import numpy as np

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy, recency


STATE = "ucb_grad_state"


def _safe(x: float) -> float:
    if x is None or not math.isfinite(float(x)):
        return 0.0
    return float(x)


def _norm(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    m = float(min(xs))
    M = float(max(xs))
    if not math.isfinite(m) or not math.isfinite(M) or abs(M - m) < 1e-12:
        return [0.0 for _ in xs]
    return [(x - m) / (M - m + 1e-12) for x in xs]


def _label_dist_entropy(c: ClientInfo) -> float:
    if isinstance(c.label_histogram, dict) and c.label_histogram:
        L = int(max(c.label_histogram.keys()) + 1)
        vec = [0.0] * L
        for k, v in c.label_histogram.items():
            idx = int(k)
            if 0 <= idx < L:
                vec[idx] = float(v)
        return label_entropy(vec)
    return 0.0


def _diversity_bonus(sel_ids: List[int], cand: ClientInfo, id_to_feat: Dict[int, np.ndarray]) -> float:
    """Compute a diversity bonus for candidate w.r.t. already selected (min cosine distance).
    Uses a compact feature embedding built from (loss, grad_norm, inv_duration, entropy).
    """
    z_cand = id_to_feat.get(cand.id)
    if z_cand is None or not sel_ids:
        return 0.0
    # cosine distance = 1 - cosine similarity
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a) + 1e-12)
        nb = float(np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / (na * nb))

    sims = []
    for sid in sel_ids:
        z = id_to_feat.get(sid)
        if z is None:
            continue
        sims.append(_cos(z, z_cand))
    if not sims:
        return 0.0
    return 1.0 - float(max(sims))  # prefer farthest from the most similar selected


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None,
                   w_reward: float = 0.6, w_loss: float = 0.2, w_grad: float = 0.2,
                   alpha_ucb: float = 0.5, lambda_div: float = 0.2,
                   time_awareness: bool = True) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """UCB-Grad: Bandit-style client selection with gradient- and loss-aware utility and
    an explicit diversity bonus.

    Maintains per-client counts and a running average utility estimate that credits
    last round's composite reward equally among selected clients (credit assignment).

    Score = quality + exploration + diversity, where
      quality   = w_reward * Q_i + w_loss * norm(loss) + w_grad * norm(grad_norm)
      exploration = alpha_ucb * sqrt(2 log t / (N_i + 1))
      diversity = lambda_div * min_cosine_distance(feature(c), feature(Sel))
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        st = {"N": {}, "Q": {}, "t": 0}

    # Update state from previous round
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    last_reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    st["t"] = int(st.get("t", 0)) + 1
    if last_sel:
        credit = last_reward / max(1, len(last_sel))
        for cid in last_sel:
            n_i = int(st["N"].get(cid, 0)) + 1
            st["N"][cid] = n_i
            q_old = float(st["Q"].get(cid, 0.0))
            # Incremental average
            st["Q"][cid] = q_old + (credit - q_old) / float(n_i)

    # Build features for current clients
    losses = [_safe(c.last_loss) for c in clients]
    gnorms = [_safe(c.grad_norm) for c in clients]
    inv_durs = [1.0 / max(1e-6, expected_duration(c)) for c in clients]
    ents = [_label_dist_entropy(c) for c in clients]

    # Normalize components that need it
    loss_n = _norm(losses)
    gnorm_n = _norm(gnorms)
    inv_dur_n = _norm(inv_durs)
    ent_n = _norm(ents)

    # Compact feature for diversity
    feats: Dict[int, np.ndarray] = {}
    for i, c in enumerate(clients):
        feats[c.id] = np.array([loss_n[i], gnorm_n[i], inv_dur_n[i], ent_n[i]], dtype=float)

    # Greedy selection with diversity term
    selected: List[int] = []
    per_client_scores: Dict[int, float] = {}

    t_global = max(1, int(st.get("t", 1)))
    # Precompute base qualities and ucb
    base_quality: Dict[int, float] = {}
    ucb_bonus: Dict[int, float] = {}
    for i, c in enumerate(clients):
        q_i = float(st["Q"].get(c.id, 0.0))
        base = w_reward * q_i + w_loss * loss_n[i] + w_grad * gnorm_n[i]
        # Time awareness: divide by expected duration proxy
        if time_awareness:
            base = base * (inv_dur_n[i] + 1e-6)
        base_quality[c.id] = base
        n_i = int(st["N"].get(c.id, 0))
        ucb_bonus[c.id] = alpha_ucb * math.sqrt(2.0 * math.log(t_global + 1.0) / (n_i + 1.0))

    # Greedy add K clients
    pool = {c.id for c in clients}
    while len(selected) < min(K, n) and pool:
        best_id = None
        best_score = -1e9
        for cid in list(pool):
            c = next(cc for cc in clients if cc.id == cid)
            score = base_quality[cid] + ucb_bonus[cid] + lambda_div * _diversity_bonus(selected, c, feats)
            # Light recency preference for long-unselected clients
            gap = float(recency(round_idx, c))
            score = score * (1.0 + 0.05 * (gap / (gap + 5.0)))
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        pool.remove(best_id)
        per_client_scores[best_id] = best_score

    # Per-client scores aligned with clients order (for plotting/analysis)
    scores_vec = [float(per_client_scores.get(c.id, base_quality.get(c.id, 0.0))) for c in clients]

    return selected, scores_vec, {STATE: st}


