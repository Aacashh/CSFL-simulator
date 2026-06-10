"""Paper-aligned online FOMAML client selector used only by the experiment suite.

This module intentionally does not replace ``selection/ml/maml_select.py``.  The
original selector remains available for legacy simulator runs.  The research
suite registers this implementation dynamically as ``research.maml_select``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, recency

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - for older supported PyTorch releases
    from torch.nn.utils.stateless import functional_call


MAML_SELECT_KEY = "research.maml_select"
STATE_KEY = "research_maml_select_state"
FEATURE_NAMES = ("loss", "grad_norm", "latency", "battery", "frequency", "staleness")


class MetaPolicy(nn.Module):
    """The 6-64-64-1 policy described in the manuscript (4,673 parameters)."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
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


@dataclass
class ReplayRecord:
    features: np.ndarray
    cost: float


def parameter_count() -> int:
    return sum(p.numel() for p in MetaPolicy().parameters())


def _seeded_policy(selector_seed: int, device: str, hidden_dim: int = 64) -> MetaPolicy:
    """Initialize policy reproducibly without perturbing the FL training RNG."""
    devices = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(selector_seed))
        return MetaPolicy(hidden_dim=int(hidden_dim)).to(device)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True)
    return (matrix - mu) / (sigma + 1e-8)


def _enabled_mask(disabled_features: Sequence[str]) -> np.ndarray:
    disabled = {str(name).strip().lower() for name in disabled_features}
    unknown = disabled.difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown MAML-Select feature(s): {sorted(unknown)}")
    return np.asarray([0.0 if name in disabled else 1.0 for name in FEATURE_NAMES], dtype=np.float32)


def _battery_ratio(client: ClientInfo, battery_wh: Mapping[int, float]) -> float:
    capacity = max(1e-8, float(getattr(client, "battery_capacity", 1.0) or 1.0))
    remaining = float(battery_wh.get(client.id, capacity))
    return max(0.0, min(1.0, remaining / capacity))


def _features(
    round_idx: int,
    clients: Sequence[ClientInfo],
    battery_wh: Mapping[int, float],
    disabled_features: Sequence[str],
) -> np.ndarray:
    rows = []
    for client in clients:
        rows.append(
            [
                float(client.last_loss or 0.0),
                float(client.grad_norm or 0.0),
                float(expected_duration(client)),
                _battery_ratio(client, battery_wh),
                float(client.participation_count or 0),
                float(recency(round_idx, client)),
            ]
        )
    matrix = _zscore(np.asarray(rows, dtype=np.float32))
    return matrix * _enabled_mask(disabled_features)[None, :]


def _as_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _loss(
    model: MetaPolicy,
    params: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(functional_call(model, params, (x,)), y)


def _adapt(
    model: MetaPolicy,
    x: torch.Tensor,
    y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = dict(model.named_parameters())
    if x.numel() == 0:
        return params
    for _ in range(max(1, int(inner_steps))):
        gradients = torch.autograd.grad(_loss(model, params, x, y), tuple(params.values()))
        params = {
            name: value - float(inner_lr) * gradient
            for (name, value), gradient in zip(params.items(), gradients)
        }
    return params


def _maybe_log_convergence(round_idx: int, diag: Dict[str, float]) -> None:
    """Append per-round convergence diagnostics to a sidecar JSONL when requested.

    Activated only when the environment variable ``MAML_SELECT_CONV_LOG`` points to
    a file path.  This decouples convergence logging from the simulator and adds no
    cost to normal runs.  Used for the selector-convergence verification study.
    """
    path = os.environ.get("MAML_SELECT_CONV_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": int(round_idx), **diag}) + "\n")
    except OSError:
        pass


def _outer_step(
    model: MetaPolicy,
    optimizer: torch.optim.Optimizer,
    support: Sequence[ReplayRecord],
    query: Sequence[ReplayRecord],
    device: str,
    inner_lr: float,
    inner_steps: int,
) -> Optional[Dict[str, float]]:
    """Apply one first-order MAML update and return convergence diagnostics.

    The query gradient is evaluated at the adapted parameters and copied onto the
    persistent initialization.  This deliberately omits second derivatives.  The
    returned dictionary records the inner-step support descent (Statement~1), the
    query objective (Statement~2), the meta-gradient norm, and the meta-update
    magnitude, all measured before the optimizer mutates the parameters.
    """
    if not query:
        return None
    base_params = {name: value.detach().clone() for name, value in model.named_parameters()}
    if support:
        sx = _as_tensor(np.stack([r.features for r in support]), device)
        sy = _as_tensor(np.asarray([r.cost for r in support], dtype=np.float32), device)
        with torch.no_grad():
            l_sup_before = float(_loss(model, base_params, sx, sy))
        adapted = _adapt(model, sx, sy, inner_lr, inner_steps)
        with torch.no_grad():
            l_sup_after = float(_loss(model, adapted, sx, sy))
    else:
        adapted = dict(model.named_parameters())
        l_sup_before = float("nan")
        l_sup_after = float("nan")

    qx = _as_tensor(np.stack([r.features for r in query]), device)
    qy = _as_tensor(np.asarray([r.cost for r in query], dtype=np.float32), device)
    with torch.no_grad():
        l_query = float(_loss(model, adapted, qx, qy))
    query_gradients = torch.autograd.grad(_loss(model, adapted, qx, qy), tuple(adapted.values()))
    meta_grad_norm = float(
        torch.sqrt(sum((g.detach() ** 2).sum() for g in query_gradients)).item()
    )

    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(model.parameters(), query_gradients):
        parameter.grad = gradient.detach().clone()
    optimizer.step()

    with torch.no_grad():
        dphi_sq = sum(
            ((value.detach() - base_params[name]) ** 2).sum()
            for name, value in model.named_parameters()
        )
        dphi_norm = float(torch.sqrt(dphi_sq).item())

    return {
        "l_sup_before": l_sup_before,
        "l_sup_after": l_sup_after,
        "l_sup_descent": l_sup_after - l_sup_before,
        "l_query": l_query,
        "meta_grad_norm": meta_grad_norm,
        "dphi_norm": dphi_norm,
    }


def _ingest_previous_feedback(
    state: Dict,
    clients_by_id: Mapping[int, ClientInfo],
    lambda_latency: float,
    normalize_latency_penalty: bool,
) -> Tuple[List[ReplayRecord], List[ReplayRecord]]:
    pending = state.get("pending")
    if not pending:
        return [], []

    target = max(1e-8, float(pending["target_latency"]))
    query: List[ReplayRecord] = []
    for cid, features, duration, energy_wh in zip(
        pending["client_ids"],
        pending["features"],
        pending["durations"],
        pending["energy_wh"],
    ):
        client = clients_by_id.get(int(cid))
        if client is None:
            continue
        local_reduction = float(client.meta.get("last_local_loss_reduction", 0.0) or 0.0)
        latency_penalty = max(0.0, float(duration) - target)
        if normalize_latency_penalty:
            latency_penalty /= target
        cost = float(lambda_latency) * latency_penalty - local_reduction
        query.append(ReplayRecord(np.asarray(features, dtype=np.float32), cost))
        state["battery_wh"][int(cid)] = max(
            0.0,
            float(state["battery_wh"].get(int(cid), client.battery_capacity)) - float(energy_wh),
        )
    return query, list(pending.get("adaptation_support", []))


def _coverage_order(clients: Sequence[ClientInfo], selector_seed: int) -> List[int]:
    ids = [client.id for client in clients]
    np.random.default_rng(int(selector_seed)).shuffle(ids)
    return ids


def _cold_start_selection(round_idx: int, K: int, coverage_order: Sequence[int]) -> List[int]:
    if not coverage_order:
        return []
    offset = (int(round_idx) * int(K)) % len(coverage_order)
    return [
        int(coverage_order[(offset + index) % len(coverage_order)])
        for index in range(min(int(K), len(coverage_order)))
    ]


def _policy_selection(
    round_idx: int,
    K: int,
    clients: Sequence[ClientInfo],
    costs: np.ndarray,
    coverage_order: Sequence[int],
    exploration_clients: int,
) -> List[int]:
    """Combine learned ranking with a disclosed staleness-based exploration slot."""
    ranked_ids = [clients[int(index)].id for index in np.argsort(costs)]
    if exploration_clients <= 0:
        return ranked_ids[: min(int(K), len(ranked_ids))]

    coverage_rank = {int(cid): index for index, cid in enumerate(coverage_order)}
    explorers = sorted(
        clients,
        key=lambda client: (
            -int(recency(round_idx, client)),
            int(client.participation_count or 0),
            coverage_rank.get(client.id, client.id),
        ),
    )[: min(int(exploration_clients), int(K), len(clients))]
    explorer_ids = [client.id for client in explorers]
    explorer_set = set(explorer_ids)
    exploit_ids = [cid for cid in ranked_ids if cid not in explorer_set]
    return exploit_ids[: max(0, min(int(K), len(clients)) - len(explorer_ids))] + explorer_ids


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    *,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 1,
    lambda_latency: float = 0.5,
    target_latency_quantile: float = 0.5,
    normalize_latency_penalty: bool = True,
    cold_start_rounds: Optional[int] = None,
    exploration_clients: int = 1,
    disabled_features: Sequence[str] = (),
    selector_seed: int = 2026,
    hidden_dim: int = 64,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Select the lowest predicted-cost clients after online FOMAML adaptation."""
    if not clients:
        return [], None, {}
    if K <= 0:
        return [], None, {}

    dev = str(device or "cpu")
    clients_by_id = {client.id: client for client in clients}
    state = history.get("state", {}).get(STATE_KEY)
    if state is None:
        model = _seeded_policy(selector_seed, dev, hidden_dim)
        state = {
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=float(outer_lr)),
            "pending": None,
            "battery_wh": {client.id: float(client.battery_capacity) for client in clients},
            "coverage_order": _coverage_order(clients, selector_seed),
            "outer_updates": 0,
        }

    model: MetaPolicy = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]
    query, previous_adaptation_support = _ingest_previous_feedback(
        state,
        clients_by_id,
        lambda_latency,
        normalize_latency_penalty,
    )

    if query:
        diag = _outer_step(model, optimizer, previous_adaptation_support, query, dev, inner_lr, inner_steps)
        state["outer_updates"] = int(state.get("outer_updates", 0)) + 1
        if diag is not None:
            _maybe_log_convergence(round_idx, diag)

    current_features = _features(round_idx, clients, state["battery_wh"], disabled_features)
    if query:
        sx = _as_tensor(np.stack([record.features for record in query]), dev)
        sy = _as_tensor(np.asarray([record.cost for record in query], dtype=np.float32), dev)
        adapted = _adapt(model, sx, sy, inner_lr, inner_steps)
    else:
        adapted = dict(model.named_parameters())

    with torch.no_grad():
        predicted_cost = functional_call(model, adapted, (_as_tensor(current_features, dev),))
    costs = predicted_cost.detach().cpu().numpy()
    warmup_rounds = (
        max(0, int(cold_start_rounds))
        if cold_start_rounds is not None
        else int(math.ceil(len(clients) / max(1, int(K))))
    )
    if round_idx < warmup_rounds:
        selected_ids = _cold_start_selection(round_idx, K, state["coverage_order"])
        state["last_selection_mode"] = "coverage_cold_start"
    else:
        selected_ids = _policy_selection(
            round_idx,
            K,
            clients,
            costs,
            state["coverage_order"],
            max(0, int(exploration_clients)),
        )
        state["last_selection_mode"] = "policy_with_staleness_exploration"
    indices_by_id = {client.id: index for index, client in enumerate(clients)}
    selected_indices = [indices_by_id[cid] for cid in selected_ids]

    durations = [float(expected_duration(clients[int(index)])) for index in selected_indices]
    energies = [
        float(getattr(clients[int(index)], "estimated_energy", 0.0) or 0.0)
        for index in selected_indices
    ]
    quantile = min(1.0, max(0.0, float(target_latency_quantile)))
    tier_two_durations = [
        expected_duration(client)
        for client in clients
        if int(getattr(client, "tier", -1) or -1) == 1
    ]
    if tier_two_durations:
        target_latency = float(np.mean(tier_two_durations))
    else:
        target_latency = float(np.quantile([expected_duration(client) for client in clients], quantile))
    state["pending"] = {
        "client_ids": selected_ids,
        "features": [current_features[int(index)].tolist() for index in selected_indices],
        "durations": durations,
        "energy_wh": energies,
        "target_latency": target_latency,
        "adaptation_support": query,
    }
    state["last_predicted_cost_mean"] = float(np.mean(costs))
    state["last_normalized_latency_penalty"] = bool(normalize_latency_penalty)
    return selected_ids, costs.astype(float).tolist(), {STATE_KEY: state}
