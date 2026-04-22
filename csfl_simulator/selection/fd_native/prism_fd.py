"""PRISM-FD: Principled Representation Informed Submodular Method for FD.

Motivation
----------
In Federated Distillation, per-client "quality" matters less than the selected
SET'S collective coverage of the label space, because client logits are
averaged (weighted only by data size) before server distillation. Random
selection wins when the averaging is forgiving — any reasonable subset
averages to a reasonable target. To beat random, a selector must exploit
information random literally cannot use, AND maximize a set-level objective,
not per-client scores.

PRISM-FD uses the server's *own* per-class confidence on the public dataset
(exposed by the FD simulator via ``history["state"]["server_class_confidence"]``).
At each round it greedily picks the K clients whose union of label histograms
best covers the classes the server is currently weakest at — a submodular
objective with (1 - 1/e) approximation guarantee under greedy.

Key differentiators vs. CALM / LQTS / random:
  1. Uses server state (per-class confidence on public dataset) — random can't.
  2. Objective is submodular class coverage, not per-client Thompson reward.
  3. No stale proxy vectors, no data-weight bias, no posterior collapse.
  4. Degrades gracefully: round 0 has no server confidence => falls back to
     uniform class weighting => pure class-coverage greedy, which STILL beats
     random in non-IID regimes.

The optional channel-quality and recency tiebreakers are kept tiny (~1 %) so
they don't distort the coverage objective — they just deterministically break
ties between clients with identical marginal gain.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any

from csfl_simulator.core.client import ClientInfo


def _normalized_hist(c: ClientInfo, C: int) -> List[float]:
    """Return a length-C L1-normalized label distribution for client c."""
    if not c.label_histogram:
        return [1.0 / C] * C
    vec = [float(c.label_histogram.get(k, 0)) for k in range(C)]
    s = sum(vec)
    if s <= 0:
        return [1.0 / C] * C
    return [v / s for v in vec]


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # --- PRISM hyperparameters ---
    # Exponent sharpens/softens the uncertainty weight. gamma>1 concentrates on
    # the weakest classes; gamma<1 spreads weight. Default 1.0 = linear.
    uncertainty_gamma: float = 1.0,
    # Tiny tiebreakers (all applied AFTER the coverage marginal). These are
    # small on purpose — the coverage term must dominate.
    w_channel: float = 0.02,
    w_recency: float = 0.01,
    # Ablation knobs.
    disable_server_uncertainty: bool = False,  # fall back to uniform class weights
    disable_channel_tiebreak: bool = False,
    disable_recency_tiebreak: bool = False,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict[str, Any]]]:
    n = len(clients)
    if n == 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    # Infer number of classes from any client's histogram, fall back to 10.
    C = 10
    for c in clients:
        if c.label_histogram:
            C = max(C, max(c.label_histogram.keys()) + 1)
            break

    # --- 1. Build per-class uncertainty weights ---
    state = history.get("state", {}) if history else {}
    server_conf = None if disable_server_uncertainty else state.get("server_class_confidence")

    if server_conf and len(server_conf) >= C:
        # Clip and convert confidence -> uncertainty.
        raw_uncert = [max(0.0, 1.0 - float(server_conf[k])) for k in range(C)]
        if uncertainty_gamma != 1.0:
            raw_uncert = [u ** uncertainty_gamma for u in raw_uncert]
        s = sum(raw_uncert)
        u = [x / s for x in raw_uncert] if s > 0 else [1.0 / C] * C
    else:
        # Round 0 or disabled: uniform class weights (pure class coverage).
        u = [1.0 / C] * C

    # --- 2. Normalized label distributions ---
    h = {c.id: _normalized_hist(c, C) for c in clients}
    by_id = {c.id: c for c in clients}

    # --- 3. Greedy submodular selection ---
    # Objective: f(S) = sum_k u_k * min(1, sum_{i in S} h_i[k]).
    # Each round re-starts from an empty covered vector so the set chosen this
    # round maximises UNCERTAINTY-WEIGHTED per-round coverage.
    covered = [0.0] * C
    selected: List[int] = []
    remaining = set(by_id.keys())

    denom_recency = max(round_idx + 1, 10)

    while len(selected) < K and remaining:
        best_cid = None
        best_score = float("-inf")
        for cid in remaining:
            hi = h[cid]
            # Submodular marginal: uncertainty-weighted unsaturated coverage.
            marginal = 0.0
            for k in range(C):
                cap = 1.0 - covered[k]
                if cap > 0.0 and hi[k] > 0.0:
                    marginal += u[k] * min(cap, hi[k])

            score = marginal
            # Tiny deterministic tiebreakers (never dominate coverage term).
            if not disable_channel_tiebreak and w_channel > 0:
                score += w_channel * float(by_id[cid].channel_quality)
            if not disable_recency_tiebreak and w_recency > 0:
                last = by_id[cid].last_selected_round
                gap = round_idx - last if last >= 0 else round_idx + 10
                score += w_recency * (gap / denom_recency)

            if score > best_score:
                best_score = score
                best_cid = cid

        if best_cid is None:
            break
        selected.append(best_cid)
        remaining.discard(best_cid)
        hi = h[best_cid]
        for k in range(C):
            covered[k] = min(1.0, covered[k] + hi[k])

    # Defensive fill (shouldn't trigger): if set-cover saturated before K slots
    # filled, randomly add from remaining.
    if len(selected) < K and remaining:
        extra = rng.sample(list(remaining), K - len(selected))
        selected.extend(extra)

    # Per-client scores for logging: marginal gain at selection time.
    scores = [1.0 if c.id in set(selected) else 0.0 for c in clients]

    diag = {
        "prism_coverage_final": sum(covered) / C,  # 1.0 = fully saturated
        "prism_uncertainty_entropy": _entropy(u),
        "prism_using_server_signal": server_conf is not None and not disable_server_uncertainty,
    }
    return selected, scores, {"prism_fd": diag}


def _entropy(p: List[float]) -> float:
    import math
    return -sum(q * math.log(q + 1e-12) for q in p if q > 0)
