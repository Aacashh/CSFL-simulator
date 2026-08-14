"""The MAML-Select meta-policy and its state representation.

The policy is the 6-64-64-1 MLP of the manuscript, and the state builder applies
the per-round standardization of Eq. (4).  Both mirror the implementation that
produced the reported results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:  # older supported PyTorch releases
    from torch.nn.utils.stateless import functional_call


FEATURE_NAMES = ("loss", "grad_norm", "latency", "battery", "frequency", "staleness")


class MetaPolicy(nn.Module):
    """The 6-64-64-1 ranking policy described in the manuscript.

    At the default width this holds 4,673 trainable parameters, which is the
    figure quoted in the complexity analysis.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def parameter_count(hidden_dim: int = 64) -> int:
    return sum(p.numel() for p in MetaPolicy(hidden_dim=hidden_dim).parameters())


def seeded_policy(seed: int, device: str = "cpu", hidden_dim: int = 64) -> MetaPolicy:
    """Initialize the policy reproducibly without disturbing the training RNG.

    Forking the RNG matters because the selector must not shift the stream that
    drives model initialization and data partitioning.  Without it, changing the
    selector would silently change the federated run it is being compared in.
    """
    devices = list(range(torch.cuda.device_count())) if str(device).startswith("cuda") and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return MetaPolicy(hidden_dim=int(hidden_dim)).to(device)


@dataclass
class ClientState:
    """The observable per-client signals the server sees before selecting.

    No raw data ever leaves a client.  The server holds only these six scalars
    and, after a round, the scalar cost feedback of Eq. (8).
    """

    client_id: int
    loss: float = 0.0
    grad_norm: float = 0.0
    latency: float = 0.0
    battery_ratio: float = 1.0
    participation_count: int = 0
    staleness: int = 0
    tier: int = 1

    def raw_vector(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.loss),
                float(self.grad_norm),
                float(self.latency),
                float(self.battery_ratio),
                float(self.participation_count),
                float(self.staleness),
            ],
            dtype=np.float32,
        )


def standardize(rows: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Eq. (4).  Column-wise z-scoring across the clients available this round.

    The six raw features carry different units, so the policy would otherwise be
    fed seconds alongside counts.  Standardizing per round also makes the score a
    *relative* ranking signal, which is all the Top-K rule consumes.
    """
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D array of states, got shape {matrix.shape}")
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True)
    return (matrix - mu) / (sigma + epsilon)


def build_state_matrix(
    states: Sequence[ClientState],
    disabled_features: Iterable[str] = (),
) -> np.ndarray:
    """Stack, standardize, then zero any ablated feature column."""
    disabled = {str(name).strip().lower() for name in disabled_features}
    unknown = disabled.difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature name(s): {sorted(unknown)}")
    matrix = standardize(np.stack([s.raw_vector() for s in states]))
    mask = np.asarray([name not in disabled for name in FEATURE_NAMES], dtype=np.float32)
    return matrix * mask[None, :]


@dataclass
class Feedback:
    """One observed (state, cost) pair returned by a client that trained."""

    features: np.ndarray
    cost: float


def mse(model: MetaPolicy, params, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(functional_call(model, params, (x,)), y)
