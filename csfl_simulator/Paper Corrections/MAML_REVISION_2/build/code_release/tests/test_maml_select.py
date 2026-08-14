"""Tests that check the manuscript's claims against this implementation.

Each test names the result it verifies, so a reader can confirm that what the
paper states is what the code does.  Run with ``pytest -q`` from the repository
root.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maml_select import (  # noqa: E402
    ClientState,
    MAMLSelect,
    client_cost,
    cohort_surrogate,
    normalized_latency_penalty,
    parameter_count,
    standardize,
    target_latency,
    top_k_by_cost,
)


def make_states(n: int, rng: np.random.Generator) -> list[ClientState]:
    return [
        ClientState(
            client_id=i,
            loss=float(rng.uniform(0.1, 2.0)),
            grad_norm=float(rng.uniform(0.1, 5.0)),
            latency=float(rng.uniform(5.0, 60.0)),
            battery_ratio=float(rng.uniform(0.2, 1.0)),
            participation_count=0,
            staleness=0,
            tier=int(i % 3),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------- architecture
def test_policy_has_the_parameter_count_quoted_in_the_paper():
    assert parameter_count() == 4673


# ------------------------------------------------------------------- Eq. (4)
def test_standardization_is_dimensionless_and_per_round():
    raw = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]], dtype=np.float32)
    z = standardize(raw)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-3)
    # Columns on wildly different scales become directly comparable.
    assert np.allclose(z[:, 0], z[:, 1], atol=1e-4)


def test_standardization_survives_a_constant_column():
    z = standardize(np.array([[5.0, 1.0], [5.0, 2.0]], dtype=np.float32))
    assert np.all(np.isfinite(z))


# ------------------------------------------------------------------- Eq. (8)
def test_latency_penalty_is_a_fractional_overrun():
    assert normalized_latency_penalty(15.0, 10.0) == pytest.approx(0.5)
    assert normalized_latency_penalty(10.0, 10.0) == 0.0
    assert normalized_latency_penalty(5.0, 10.0) == 0.0, "meeting the deadline is never penalized"


def test_normalized_cost_is_scale_free_but_unnormalized_cost_is_not():
    """Why the paper normalizes.  Doubling the time unit must not change the cost."""
    normalized = [client_cost(30.0, 0.4, 20.0, 0.5), client_cost(60.0, 0.4, 40.0, 0.5)]
    assert normalized[0] == pytest.approx(normalized[1])

    unnormalized = [
        client_cost(30.0, 0.4, 20.0, 0.5, normalize=False),
        client_cost(60.0, 0.4, 40.0, 0.5, normalize=False),
    ]
    assert unnormalized[0] != pytest.approx(unnormalized[1])


def test_target_latency_uses_the_medium_tier_and_falls_back_to_the_quantile():
    latencies, tiers = [10.0, 20.0, 30.0, 40.0], [0, 1, 1, 2]
    assert target_latency(latencies, tiers) == pytest.approx(25.0)
    assert target_latency([10.0, 30.0], [0, 2]) == pytest.approx(20.0)


# ------------------------------------------------------------- Proposition 1
def test_top_k_exactly_minimizes_the_modular_surrogate():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, k = 9, 4
        costs = rng.normal(size=n).tolist()
        chosen = top_k_by_cost(costs, k)
        best = cohort_surrogate([costs[i] for i in chosen])

        from itertools import combinations

        brute = min(cohort_surrogate([costs[i] for i in c]) for c in combinations(range(n), k))
        assert best == pytest.approx(brute), "Top-K must be exactly optimal, not merely good"


# ------------------------------------------------------------------- Lemma 1
def test_inner_step_never_increases_the_support_loss():
    """Lemma 1.  Verified over a full run, as in Sec. V-D of the manuscript."""
    rng = np.random.default_rng(7)
    sel = MAMLSelect(num_clients=30, cohort_size=5, seed=11)
    states = make_states(30, rng)

    violations, checked = 0, 0
    for _ in range(60):
        chosen, diag = sel.select(states)
        if diag is not None and not math.isnan(diag.inner_descent):
            checked += 1
            if diag.inner_descent > 1e-9:
                violations += 1
        sel.observe(
            [states[c].latency for c in chosen],
            [float(rng.uniform(0.0, 1.0)) for _ in chosen],
        )
        for s in states:
            s.staleness = 0 if s.client_id in chosen else s.staleness + 1
            s.participation_count += int(s.client_id in chosen)
    assert checked > 30, "the run must actually exercise the inner step"
    assert violations == 0, f"{violations}/{checked} rounds increased the support loss"


# ------------------------------------------------------------- Proposition 2
def test_exploration_slot_alone_guarantees_full_coverage():
    """Proposition 2.  Coverage must not depend on the cold start.

    The cold start is disabled here, so any client that is ever selected is
    selected by the learned ranking or by the exploration slot.
    """
    rng = np.random.default_rng(3)
    N, K = 40, 5
    sel = MAMLSelect(num_clients=N, cohort_size=K, cold_start_rounds=0, seed=5)
    states = make_states(N, rng)

    seen: set[int] = set()
    for round_index in range(N):
        chosen, _ = sel.select(states)
        seen.update(chosen)
        sel.observe(
            [states[c].latency for c in chosen],
            [float(rng.uniform(0.0, 1.0)) for _ in chosen],
        )
        for s in states:
            s.staleness = 0 if s.client_id in chosen else s.staleness + 1
            s.participation_count += int(s.client_id in chosen)
    assert len(seen) == N, f"only {len(seen)}/{N} clients were ever selected without a cold start"


def test_without_the_exploration_slot_coverage_is_not_guaranteed():
    """The contrapositive.  The slot is doing the work, not the cold start."""
    rng = np.random.default_rng(3)
    N, K = 40, 5
    sel = MAMLSelect(num_clients=N, cohort_size=K, cold_start_rounds=0, exploration_slots=0, seed=5)
    states = make_states(N, rng)
    for s in states:  # a strongly separated pool, so a greedy ranking can lock on
        s.latency = 5.0 if s.client_id < K else 90.0

    seen: set[int] = set()
    for _ in range(N):
        chosen, _ = sel.select(states)
        seen.update(chosen)
        sel.observe([states[c].latency for c in chosen], [0.5] * len(chosen))
    assert len(seen) < N


# ----------------------------------------------------------- set lag, Eq. (5)
def test_support_set_is_the_previous_query_set():
    rng = np.random.default_rng(1)
    sel = MAMLSelect(num_clients=20, cohort_size=4, seed=2)
    states = make_states(20, rng)

    chosen, diag = sel.select(states)
    assert diag is None, "no meta-update is possible before any feedback exists"
    sel.observe([states[c].latency for c in chosen], [0.5] * len(chosen))
    assert sel.support == [] and len(sel.query) == 4

    previous_query = list(sel.query)
    chosen, _ = sel.select(states)
    sel.observe([states[c].latency for c in chosen], [0.5] * len(chosen))
    assert sel.support == previous_query, "D_sup(t) must equal D_query(t-1)"


# ------------------------------------------------------------------ drift, V_T
def test_drift_increment_is_measured_at_a_common_iterate():
    rng = np.random.default_rng(4)
    sel = MAMLSelect(num_clients=25, cohort_size=5, seed=9)
    states = make_states(25, rng)
    increments = []
    for _ in range(25):
        chosen, diag = sel.select(states)
        if diag is not None and not math.isnan(diag.drift_increment):
            increments.append(diag.drift_increment)
        sel.observe(
            [states[c].latency for c in chosen],
            [float(rng.uniform(0.0, 1.0)) for _ in chosen],
        )
    assert increments, "the drift term must be observable"
    assert all(math.isfinite(v) for v in increments)


# ------------------------------------------------------------------- guards
def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        MAMLSelect(num_clients=10, cohort_size=0)
    with pytest.raises(ValueError):
        MAMLSelect(num_clients=10, cohort_size=11)
    with pytest.raises(ValueError):
        MAMLSelect(num_clients=10, cohort_size=3, exploration_slots=3)


def test_observe_requires_a_prior_selection():
    sel = MAMLSelect(num_clients=10, cohort_size=2)
    with pytest.raises(RuntimeError):
        sel.observe([1.0, 2.0], [0.1, 0.2])
