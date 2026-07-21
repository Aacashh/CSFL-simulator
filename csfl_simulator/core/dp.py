from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping

import torch

def apply_gaussian_noise(tensor: torch.Tensor, sigma: float):
    if sigma <= 0:
        return tensor
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def clip_gradients(model: torch.nn.Module, max_norm: float):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def consume_epsilon(client, eps_cost: float):
    client.dp_epsilon_used += eps_cost
    client.dp_epsilon_remaining = max(0.0, client.dp_epsilon_remaining - eps_cost)


def _laplace_sample(rng, scale: float) -> float:
    """Draw Laplace(0, scale) using the simulator's seeded RNG."""
    if scale <= 0:
        return 0.0
    u = min(max(float(rng.random()), 1e-12), 1.0 - 1e-12) - 0.5
    return -scale * math.copysign(1.0, u) * math.log1p(-2.0 * abs(u))


def laplace_noise_histogram(
    histogram: Mapping[int, float] | None,
    epsilon: float,
    rng,
    *,
    sensitivity: float = 1.0,
    num_classes: int | None = None,
) -> Dict[int, float]:
    """Return a non-negative Laplace-perturbed label histogram.

    The default L1 sensitivity of one corresponds to add/remove-one-example
    adjacency. Callers using replace-one adjacency should pass sensitivity=2.
    The selector caches this release, avoiding fresh privacy spend each round.
    ``epsilon=inf`` is the explicit non-private control.
    """
    if epsilon is None or math.isinf(float(epsilon)):
        return {int(k): float(v) for k, v in (histogram or {}).items()}
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("histogram epsilon must be positive or infinity")
    if sensitivity <= 0:
        raise ValueError("histogram sensitivity must be positive")

    source: MutableMapping[int, float] = {
        int(k): float(v) for k, v in (histogram or {}).items()
    }
    inferred = (max(source.keys()) + 1) if source else 0
    classes = max(int(num_classes or 0), inferred)
    scale = float(sensitivity) / epsilon
    return {
        cls: max(0.0, source.get(cls, 0.0) + _laplace_sample(rng, scale))
        for cls in range(classes)
    }
