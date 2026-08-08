"""MAML-Select, the online first-order MAML client selector of the manuscript.

This is Algorithm 1 in full.  Every round performs, in this order and all before
any client is scored:

    Phase 1  one plain gradient step on the support set          Eq. (9)
    Phase 2  a first-order meta-update on the query set          Eq. (10)
    Phase 3  scoring, cold start or Top-K with an exploration slot, Eq. (11)

The support and query sets are two *consecutive completed* rounds, because cost
feedback only exists once a cohort has trained:

    D_sup(t)   = feedback of round t-2
    D_query(t) = feedback of round t-1

so D_sup(t) = D_query(t-1).  The inner step adapts on the older round and the
meta-gradient is measured on the newer one, which is what makes this a
meta-learning problem rather than online regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .cost import client_cost, target_latency
from .policy import (
    ClientState,
    Feedback,
    MetaPolicy,
    build_state_matrix,
    mse,
    seeded_policy,
)

try:
    from torch.func import functional_call
except ImportError:  # older supported PyTorch releases
    from torch.nn.utils.stateless import functional_call


@dataclass
class Diagnostics:
    """Per-round quantities used to check the theory against the run."""

    support_loss_before: float = float("nan")
    support_loss_after: float = float("nan")
    query_loss_adapted: float = float("nan")
    query_loss_at_base: float = float("nan")
    meta_grad_norm: float = float("nan")
    update_norm: float = float("nan")

    @property
    def inner_descent(self) -> float:
        """g_t(phi'_t) - g_t(phi_t).  Lemma 1 predicts this is never positive."""
        return self.support_loss_after - self.support_loss_before

    @property
    def drift_increment(self) -> float:
        """One term of V_T, namely q_{t+1}(phi_{t+1}) - q_t(phi_{t+1}).

        The support set *is* the previous query set, so evaluating both at the
        same un-adapted parameters gives the drift exactly rather than by
        approximation.
        """
        return self.query_loss_at_base - self.support_loss_before


class MAMLSelect:
    """The selector.  Call :meth:`select` once per round, then :meth:`observe`."""

    def __init__(
        self,
        num_clients: int,
        cohort_size: int,
        *,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 1,
        lambda_latency: float = 0.5,
        exploration_slots: int = 1,
        cold_start_rounds: Optional[int] = None,
        hidden_dim: int = 64,
        seed: int = 2026,
        device: str = "cpu",
        disabled_features: Sequence[str] = (),
    ) -> None:
        if cohort_size <= 0 or cohort_size > num_clients:
            raise ValueError("cohort_size must be in 1..num_clients")
        if exploration_slots < 0 or exploration_slots >= cohort_size:
            raise ValueError("exploration_slots must leave at least one learned place")

        self.N = int(num_clients)
        self.K = int(cohort_size)
        self.inner_lr = float(inner_lr)
        self.inner_steps = max(1, int(inner_steps))
        self.lambda_latency = float(lambda_latency)
        self.epsilon = int(exploration_slots)
        self.device = str(device)
        self.disabled_features = tuple(disabled_features)
        self.cold_start_rounds = (
            int(cold_start_rounds) if cold_start_rounds is not None else math.ceil(self.N / self.K)
        )

        self.model = seeded_policy(seed, self.device, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(outer_lr))

        # A fixed, seed-shuffled visiting order used only during the cold start,
        # and as the final tie-break of the exploration slot.
        self.coverage_order: List[int] = list(range(self.N))
        np.random.default_rng(int(seed)).shuffle(self.coverage_order)
        self.coverage_rank = {cid: i for i, cid in enumerate(self.coverage_order)}

        self.round_index = 0
        self.support: List[Feedback] = []   # feedback of round t-2
        self.query: List[Feedback] = []     # feedback of round t-1
        self._pending: Optional[Dict] = None
        self.last_selection_mode = "uninitialized"

    # ------------------------------------------------------------------ phases
    def _adapt(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Phase 1.  Plain gradient descent, differentiable through the step."""
        params = dict(self.model.named_parameters())
        for _ in range(self.inner_steps):
            grads = torch.autograd.grad(mse(self.model, params, x, y), tuple(params.values()))
            params = {
                name: value - self.inner_lr * grad
                for (name, value), grad in zip(params.items(), grads)
            }
        return params

    def _meta_update(self) -> Optional[Diagnostics]:
        """Phases 1 and 2.  Returns None until a query set exists."""
        if not self.query:
            return None

        diag = Diagnostics()
        base = {name: v.detach().clone() for name, v in self.model.named_parameters()}
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)

        if self.support:
            sx = tensor(np.stack([f.features for f in self.support]))
            sy = tensor(np.asarray([f.cost for f in self.support], dtype=np.float32))
            with torch.no_grad():
                diag.support_loss_before = float(mse(self.model, base, sx, sy))
            adapted = self._adapt(sx, sy)
            with torch.no_grad():
                diag.support_loss_after = float(mse(self.model, adapted, sx, sy))
        else:
            adapted = dict(self.model.named_parameters())

        qx = tensor(np.stack([f.features for f in self.query]))
        qy = tensor(np.asarray([f.cost for f in self.query], dtype=np.float32))
        with torch.no_grad():
            diag.query_loss_adapted = float(mse(self.model, adapted, qx, qy))
            diag.query_loss_at_base = float(mse(self.model, base, qx, qy))

        # First-order MAML.  The gradient is taken at the adapted weights and
        # applied to the base weights, which drops the second-order term.
        grads = torch.autograd.grad(mse(self.model, adapted, qx, qy), tuple(adapted.values()))
        diag.meta_grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))

        self.optimizer.zero_grad(set_to_none=True)
        for parameter, grad in zip(self.model.parameters(), grads):
            parameter.grad = grad.detach().clone()
        self.optimizer.step()

        with torch.no_grad():
            diag.update_norm = float(
                torch.sqrt(sum(((v.detach() - base[n]) ** 2).sum() for n, v in self.model.named_parameters()))
            )
        return diag

    def _explorers(self, states: Sequence[ClientState]) -> List[int]:
        """The deterministic exploration slot.

        The place goes to the stalest, then least-used, client.  It is not random.
        This is what makes full participation coverage a property of the method
        rather than of the cold start, see Proposition 2 in the manuscript.
        """
        if self.epsilon <= 0:
            return []
        ordered = sorted(
            states,
            key=lambda s: (-int(s.staleness), int(s.participation_count), self.coverage_rank.get(s.client_id, s.client_id)),
        )
        return [s.client_id for s in ordered[: self.epsilon]]

    # ------------------------------------------------------------------- public
    def select(self, states: Sequence[ClientState]) -> tuple[List[int], Optional[Diagnostics]]:
        """Run one full round of Algorithm 1 and return the chosen cohort."""
        if not states:
            raise ValueError("no available clients")
        if len(states) < self.K:
            raise ValueError(f"need at least K={self.K} available clients, got {len(states)}")

        diag = self._meta_update()

        features = build_state_matrix(states, self.disabled_features)
        params = dict(self.model.named_parameters())
        if self.support and self.query:
            tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
            sx = tensor(np.stack([f.features for f in self.support]))
            sy = tensor(np.asarray([f.cost for f in self.support], dtype=np.float32))
            params = self._adapt(sx, sy)
        with torch.no_grad():
            scores = functional_call(
                self.model, params, (torch.as_tensor(features, dtype=torch.float32, device=self.device),)
            ).cpu().numpy()

        if self.round_index < self.cold_start_rounds:
            offset = (self.round_index * self.K) % self.N
            chosen = [self.coverage_order[(offset + i) % self.N] for i in range(self.K)]
            self.last_selection_mode = "coverage_cold_start"
        else:
            explorers = self._explorers(states)
            ranked = [states[int(i)].client_id for i in np.argsort(scores)]
            exploit = [cid for cid in ranked if cid not in set(explorers)]
            chosen = exploit[: self.K - len(explorers)] + explorers
            self.last_selection_mode = "policy_with_staleness_exploration"

        index_of = {s.client_id: i for i, s in enumerate(states)}
        self._pending = {
            "client_ids": list(chosen),
            "features": [features[index_of[cid]] for cid in chosen],
            "target": target_latency([s.latency for s in states], [s.tier for s in states]),
        }
        self.round_index += 1
        return chosen, diag

    def observe(self, latencies: Sequence[float], loss_reductions: Sequence[float]) -> List[float]:
        """Record the realized cost of the cohort returned by the last ``select``.

        Shifting support to the previous query set here is what creates the
        one-round lag between the two sets.
        """
        if self._pending is None:
            raise RuntimeError("observe() called before select()")
        if not (len(latencies) == len(loss_reductions) == len(self._pending["client_ids"])):
            raise ValueError("feedback length must match the selected cohort")

        target = self._pending["target"]
        costs = [
            client_cost(lat, red, target, self.lambda_latency)
            for lat, red in zip(latencies, loss_reductions)
        ]
        self.support = self.query
        self.query = [
            Feedback(features=f, cost=c) for f, c in zip(self._pending["features"], costs)
        ]
        self._pending = None
        return costs


def top_k_by_cost(costs: Sequence[float], k: int) -> List[int]:
    """Proposition 1.  The exact minimizer of the modular cohort surrogate."""
    if k < 0 or k > len(costs):
        raise ValueError("k must be in 0..len(costs)")
    return sorted(range(len(costs)), key=lambda i: float(costs[i]))[:k]
