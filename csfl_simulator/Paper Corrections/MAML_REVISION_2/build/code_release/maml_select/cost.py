"""The per-client utility of Eq. (8) and the deadline it is measured against.

The latency penalty is *normalized* by the target latency.  This is the form used
for every reported result.  Without the division the penalty carries units of
seconds while the loss reduction is dimensionless, so the single weight lambda
would depend on the units of the latency model and would not transfer between
datasets with different round times.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

TIER_MEDIUM = 1  # zero-indexed tier identifiers, so Tier 2 of the paper is index 1


def target_latency(
    latencies: Sequence[float],
    tiers: Sequence[int],
    quantile: float = 0.5,
) -> float:
    """The soft deadline T_target, recomputed from the live pool each round.

    It is the mean expected round latency of the medium tier.  If no medium-tier
    device is available the pool quantile is used instead, so the deadline stays
    defined for any pool composition.
    """
    latencies = [float(x) for x in latencies]
    if not latencies:
        raise ValueError("target_latency needs a non-empty pool")
    medium = [lat for lat, tier in zip(latencies, tiers) if int(tier) == TIER_MEDIUM]
    value = float(np.mean(medium)) if medium else float(np.quantile(latencies, min(1.0, max(0.0, quantile))))
    return max(1e-8, value)


def normalized_latency_penalty(latency: float, target: float) -> float:
    """rho_{i,t} of Eq. (6), a fractional overrun of the deadline."""
    target = max(1e-8, float(target))
    return max(0.0, float(latency) - target) / target


def client_cost(
    latency: float,
    local_loss_reduction: float,
    target: float,
    lambda_latency: float = 0.5,
    normalize: bool = True,
) -> float:
    """Eq. (8).  Lower is better, so Top-K selects the K smallest costs.

    ``normalize=False`` reproduces the unnormalized variant only for ablation.
    It is not the configuration behind any reported number.
    """
    penalty = max(0.0, float(latency) - float(target))
    if normalize:
        penalty /= max(1e-8, float(target))
    return float(lambda_latency) * penalty - float(local_loss_reduction)


def cohort_surrogate(costs: Sequence[float]) -> float:
    """The modular cohort surrogate F-tilde of Eq. (8), summed over a cohort."""
    return float(sum(float(c) for c in costs))
