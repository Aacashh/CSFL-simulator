"""SCOPE-FD: Server-aware Coverage with Over-round Participation Equalization.

WHY THIS EXISTS
---------------
PRISM-FD failed catastrophically on CIFAR-10/α=0.5/DL=-20: Gini shot from
random's 0.083 to 0.536 and accuracy dropped from 0.308 to 0.271. The reason:
label_histogram is a **static** field (fixed at partition time), so greedy
coverage on it deterministically locks onto the same ~15 "high-coverage"
clients every round. Over 100 rounds, those clients get picked ~66 times each
and the other 15 get picked ~0 times. Diversity collapses, the server sees
a narrow slice of the true data distribution, and accuracy falls below random.

CORE INSIGHT
------------
Random's headline property is not its "randomness" — it is its
**over-round participation balance** (Gini ≈ 0.08). Any method that wants to
beat random MUST first match or beat that balance, then layer structure on
top. In other words: don't think of selection as "pick best K clients"; think
of it as "allocate K slots so that (a) participation accumulates uniformly
over rounds AND (b) each round's subset is informative." These are SEPARATE
objectives that compose.

ALGORITHM
---------
Primary signal — **participation debt**:
    debt_i = (r+1) * K / N - participation_count_i
The client's under-participation relative to the uniform target. A client
picked every round has debt → -∞; a client never picked has debt → +∞.
Normalising debt to [0, 1] across the current pool makes it the dominant
ranking term. This term alone produces a deterministic round-robin that
achieves Gini ≈ 0 asymptotically — strictly better than random.

Secondary signal — **server uncertainty bonus** (0.3 weight):
    uncert_i = Σ_k (1 - server_conf_k) * h_i[k]
Nudges clients whose label histogram aligns with classes the server is
currently weakest on. Uses the simulator's per-class confidence hook on the
public dataset (exposed in history["state"]["server_class_confidence"]).
Weight kept small so it cannot overwhelm the debt term.

Tertiary signal — **per-round diversity penalty** (0.1 weight):
    penalty_i = Σ_k covered[k] * h_i[k]
Subtracts from the score of clients whose label mass overlaps what's already
been selected THIS round. Prevents picking near-duplicate label distributions
in the same round.

Selection is greedy through these K slots; after each pick we update
`covered` and recompute scores for the remaining candidates.

WHY THIS BEATS RANDOM (EXPECTED)
--------------------------------
1. Participation balance: a deterministic cycle of ⌈N/K⌉ rounds guarantees
   every client is selected exactly once per cycle. Gini → 0.
2. Random matches this only in expectation and has Θ(√(KR)/N) variance in
   participation counts; SCOPE has 0 variance within a cycle.
3. The uncertainty bonus and diversity penalty can only add information on
   top of that already-better-than-random balance. So even a weak positive
   nudge translates to a net accuracy win.

Ablations exposed:
  - disable_server_signal    (scope_fd_no_server) — pure debt + diversity
  - disable_diversity_penalty (scope_fd_no_diversity) — pure debt + server
"""
from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple, Any

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.core.dp import laplace_noise_histogram


def _normalized_hist(histogram, C: int) -> List[float]:
    if not histogram:
        return [1.0 / C] * C
    vec = [float(histogram.get(k, histogram.get(str(k), 0.0))) for k in range(C)]
    s = sum(vec)
    if s <= 0:
        return [1.0 / C] * C
    return [v / s for v in vec]


def _minmax(values: Dict[int, float], invert: bool = False) -> Dict[int, float]:
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= 1e-12:
        return {cid: 0.5 for cid in values}
    normalized = {cid: (value - lo) / span for cid, value in values.items()}
    if invert:
        return {cid: 1.0 - value for cid, value in normalized.items()}
    return normalized



def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    # --- SCOPE hyperparameters ---
    alpha_uncertainty: float = 0.3,   # weight on server-uncertainty bonus (<< 1.0 debt)
    alpha_diversity: float = 0.1,     # weight on per-round diversity penalty
    alpha_channel: float = 0.0,
    channel_energy_mix: float = 0.5,
    histogram_epsilon: float = float("inf"),
    histogram_sensitivity: float = 1.0,
    use_surrogate_histogram: bool = False,
    enforce_energy_budget: bool = False,
    energy_budget=None,
    bytes_budget=None,
    # Ablation switches
    disable_server_signal: bool = False,
    disable_diversity_penalty: bool = False,
    **kwargs,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict[str, Any]]]:
    n = len(clients)
    if n == 0 or K <= 0:
        return [], None, {}
    K = min(K, n)

    state_in = history.get("state", {}) if history else {}
    signals = state_in.get("fd_client_signals", {})
    # Detect the class count from raw histograms and server-side surrogates.
    C = 10
    for c in clients:
        if c.label_histogram:
            C = max(C, max(int(k) for k in c.label_histogram.keys()) + 1)
    for signal in signals.values():
        surrogate = signal.get("surrogate_histogram") if isinstance(signal, dict) else None
        if surrogate:
            C = max(C, len(surrogate))

    private_cache = None
    surrogate_count = 0
    if use_surrogate_histogram:
        raw_hists = {}
        for c in clients:
            signal = signals.get(c.id, signals.get(str(c.id), {}))
            surrogate = signal.get("surrogate_histogram") if isinstance(signal, dict) else None
            if surrogate:
                raw_hists[c.id] = {i: float(v) for i, v in enumerate(surrogate)}
                surrogate_count += 1
            else:
                raw_hists[c.id] = {}
    elif math.isfinite(float(histogram_epsilon)):
        cache_in = state_in.get("scope_private_histograms", {})
        cache_meta = cache_in.get("meta", {}) if isinstance(cache_in, dict) else {}
        expected_meta = {
            "epsilon": float(histogram_epsilon),
            "sensitivity": float(histogram_sensitivity),
            "num_classes": C,
        }
        private_cache = cache_in if cache_meta == expected_meta else {
            "meta": expected_meta,
            "histograms": {},
        }
        cached_hists = private_cache["histograms"]
        for c in clients:
            if c.id not in cached_hists and str(c.id) not in cached_hists:
                cached_hists[c.id] = laplace_noise_histogram(
                    c.label_histogram,
                    histogram_epsilon,
                    rng,
                    sensitivity=histogram_sensitivity,
                    num_classes=C,
                )
        raw_hists = {
            c.id: cached_hists.get(c.id, cached_hists.get(str(c.id), {}))
            for c in clients
        }
    else:
        raw_hists = {c.id: c.label_histogram for c in clients}


    hists = {cid: _normalized_hist(hist, C) for cid, hist in raw_hists.items()}
    by_id = {c.id: c for c in clients}

    # --- 1. Participation debts (PRIMARY signal, dominates everything else). ---
    target = (round_idx + 1) * K / n
    raw_debts = {c.id: target - c.participation_count for c in clients}
    mn = min(raw_debts.values())
    mx = max(raw_debts.values())
    span = mx - mn
    if span > 1e-9:
        nd = {cid: (d - mn) / span for cid, d in raw_debts.items()}
    else:
        # Everyone's equally over/under-picked — debt term is flat, other
        # signals break the tie. Assign 0.5 so it doesn't skew anything.
        nd = {cid: 0.5 for cid in raw_debts}

    # Optional fourth term: normalized channel quality plus energy efficiency.
    raw_channel = {c.id: float(c.channel_quality) for c in clients}
    raw_energy = {c.id: max(0.0, float(c.estimated_energy)) for c in clients}
    channel_score = _minmax(raw_channel)
    energy_score = _minmax(raw_energy, invert=True)
    mix = min(1.0, max(0.0, float(channel_energy_mix)))
    channel_utility = {
        cid: mix * channel_score[cid] + (1.0 - mix) * energy_score[cid]
        for cid in by_id
    }

    # --- 2. Server-uncertainty bonus (SECONDARY, only nudges). ---
    uncert_bonus = {c.id: 0.0 for c in clients}
    used_server = False
    if not disable_server_signal:
        server_conf = history.get("state", {}).get("server_class_confidence") if history else None
        if server_conf and len(server_conf) >= C:
            raw_u = [max(0.0, 1.0 - float(server_conf[k])) for k in range(C)]
            us = sum(raw_u)
            if us > 0:
                u = [x / us for x in raw_u]
                for c in clients:
                    h = hists[c.id]
                    uncert_bonus[c.id] = sum(u[k] * h[k] for k in range(C))
                used_server = True

    # --- 3. Greedy K-selection with per-round diversity penalty (TERTIARY). ---
    selected: List[int] = []
    covered = [0.0] * C
    remaining = set(by_id.keys())
    budget = float(energy_budget) if energy_budget is not None else None
    spent_energy = 0.0

    while len(selected) < K and remaining:
        best_cid = None
        best_score = float("-inf")
        for cid in remaining:
            energy = max(0.0, float(by_id[cid].estimated_energy))
            if enforce_energy_budget and budget is not None and spent_energy + energy > budget:
                continue
            score = (nd[cid] + alpha_uncertainty * uncert_bonus[cid]
                     + alpha_channel * channel_utility[cid])
            if not disable_diversity_penalty and selected:
                h = hists[cid]
                overlap = sum(covered[k] * h[k] for k in range(C))
                score -= alpha_diversity * overlap
            if score > best_score:
                best_score = score
                best_cid = cid
        if best_cid is None:
            break
        selected.append(best_cid)
        remaining.discard(best_cid)
        spent_energy += max(0.0, float(by_id[best_cid].estimated_energy))
        h = hists[best_cid]
        for k in range(C):
            covered[k] = min(1.0, covered[k] + h[k])

    if len(selected) < K and remaining and not enforce_energy_budget:
        # Defensive: shouldn't trigger.
        extra = rng.sample(list(remaining), K - len(selected))
        selected.extend(extra)

    scores = [1.0 if c.id in set(selected) else 0.0 for c in clients]
    diag = {
        "scope_used_server_signal": used_server,
        "scope_target_participation": target,
        "scope_debt_span": span,
        "scope_final_covered_mean": sum(covered) / C,
        "scope_alpha_channel": float(alpha_channel),
        "scope_energy_spent": spent_energy,
        "scope_energy_budget_enforced": bool(enforce_energy_budget and budget is not None),
        "scope_histogram_source": "surrogate" if use_surrogate_histogram else (
            "laplace" if math.isfinite(float(histogram_epsilon)) else "raw"
        ),
        "scope_surrogate_clients": surrogate_count,
    }
    state_out = {"scope_fd": diag}
    if private_cache is not None:
        state_out["scope_private_histograms"] = private_cache
    return selected, scores, state_out
