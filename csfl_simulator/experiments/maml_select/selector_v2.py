"""MAML-Select v2: ranking-aware, diversity-aware client selection.

This selector is intentionally isolated from ``selector.py``.  It is an
experimental successor for post-campaign comparison, not a replacement for the
submitted MAML-Select implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, recency

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call


MAML_SELECT_V2_KEY = "research.maml_select_v2"
STATE_KEY = "research_maml_select_v2_state"
FEATURE_NAMES = (
    "loss",
    "grad_norm",
    "latency",
    "battery",
    "frequency",
    "staleness",
    "data_size",
    "label_entropy",
    "rare_class_score",
    "update_novelty",
    "reward_ema",
    "reward_uncertainty",
    "energy",
)


class MetaPolicyV2(nn.Module):
    """Three-layer MLP policy, widened only at the input layer."""

    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64):
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
    return sum(p.numel() for p in MetaPolicyV2().parameters())


def _seeded_policy(selector_seed: int, device: str) -> MetaPolicyV2:
    devices = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(selector_seed))
        return MetaPolicyV2().to(device)


def _enabled_mask(disabled_features: Sequence[str]) -> np.ndarray:
    disabled = {str(name).strip().lower() for name in disabled_features}
    unknown = disabled.difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown MAML-Select v2 feature(s): {sorted(unknown)}")
    return np.asarray([0.0 if name in disabled else 1.0 for name in FEATURE_NAMES], dtype=np.float32)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True)
    return (matrix - mu) / (sigma + 1e-8)


def _unit(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    lo, hi = float(np.min(array)), float(np.max(array))
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)


def _safe_histogram(client: ClientInfo) -> Dict[int, float]:
    hist = getattr(client, "label_histogram", None)
    if not isinstance(hist, dict):
        return {}
    return {int(k): float(v) for k, v in hist.items() if float(v) > 0.0}


def _label_entropy(hist: Mapping[int, float]) -> float:
    total = float(sum(hist.values()))
    if total <= 0.0 or len(hist) <= 1:
        return 0.0
    probs = np.asarray([float(v) / total for v in hist.values() if float(v) > 0.0], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum() / max(1e-8, math.log(len(probs))))


def _class_totals(clients: Sequence[ClientInfo]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for client in clients:
        for label, count in _safe_histogram(client).items():
            totals[label] = totals.get(label, 0.0) + float(count)
    return totals


def _rare_score(hist: Mapping[int, float], totals: Mapping[int, float]) -> float:
    if not hist or not totals:
        return 0.0
    weights = [1.0 / math.sqrt(max(1.0, float(totals.get(label, 1.0)))) for label in hist]
    return float(np.mean(weights)) if weights else 0.0


def _battery_ratio(client: ClientInfo, battery_wh: Mapping[int, float]) -> float:
    capacity = max(1e-8, float(getattr(client, "battery_capacity", 1.0) or 1.0))
    remaining = float(battery_wh.get(client.id, capacity))
    return max(0.0, min(1.0, remaining / capacity))


def _reward_stats(state: Dict, cid: int) -> Tuple[float, float]:
    stats = state.setdefault("reward_stats", {})
    item = stats.get(int(cid), {"ema": 0.0, "n": 0, "mean": 0.0, "m2": 0.0})
    n = int(item.get("n", 0))
    variance = float(item.get("m2", 0.0)) / max(1, n - 1) if n > 1 else 0.0
    return float(item.get("ema", 0.0)), math.sqrt(max(0.0, variance))


def _update_reward_stats(state: Dict, cid: int, reward: float, ema_alpha: float) -> None:
    stats = state.setdefault("reward_stats", {})
    item = stats.get(int(cid), {"ema": 0.0, "n": 0, "mean": 0.0, "m2": 0.0})
    n = int(item.get("n", 0)) + 1
    mean = float(item.get("mean", 0.0))
    delta = float(reward) - mean
    mean += delta / n
    m2 = float(item.get("m2", 0.0)) + delta * (float(reward) - mean)
    old_ema = float(item.get("ema", 0.0))
    ema = float(reward) if n == 1 else float(ema_alpha) * float(reward) + (1.0 - float(ema_alpha)) * old_ema
    stats[int(cid)] = {"ema": ema, "n": n, "mean": mean, "m2": m2}


def _client_descriptors(
    round_idx: int,
    clients: Sequence[ClientInfo],
    state: Dict,
    disabled_features: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    totals = _class_totals(clients)
    battery_wh = state["battery_wh"]
    rows = []
    raw = {name: [] for name in FEATURE_NAMES}
    for client in clients:
        hist = _safe_histogram(client)
        reward_ema, reward_uncertainty = _reward_stats(state, client.id)
        previous_grad = float(state.setdefault("last_grad_norm", {}).get(client.id, 0.0))
        previous_loss = float(state.setdefault("last_loss", {}).get(client.id, 0.0))
        update_novelty = abs(float(client.grad_norm or 0.0) - previous_grad) + abs(float(client.last_loss or 0.0) - previous_loss)
        values = {
            "loss": float(client.last_loss or 0.0),
            "grad_norm": float(client.grad_norm or 0.0),
            "latency": float(expected_duration(client)),
            "battery": _battery_ratio(client, battery_wh),
            "frequency": float(client.participation_count or 0),
            "staleness": float(recency(round_idx, client)),
            "data_size": float(client.data_size or 0),
            "label_entropy": _label_entropy(hist),
            "rare_class_score": _rare_score(hist, totals),
            "update_novelty": update_novelty,
            "reward_ema": reward_ema,
            "reward_uncertainty": reward_uncertainty,
            "energy": float(getattr(client, "estimated_energy", 0.0) or 0.0),
        }
        for name in FEATURE_NAMES:
            raw[name].append(values[name])
        rows.append([values[name] for name in FEATURE_NAMES])
    matrix = _zscore(np.asarray(rows, dtype=np.float32))
    matrix *= _enabled_mask(disabled_features)[None, :]
    return matrix, {name: np.asarray(values, dtype=np.float32) for name, values in raw.items()}


def _as_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _ranking_loss(
    model: MetaPolicyV2,
    params: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    pairwise_weight: float,
    listwise_weight: float,
    temperature: float,
) -> torch.Tensor:
    pred = functional_call(model, params, (x,))
    mse = F.mse_loss(pred, y)
    loss = mse
    if x.shape[0] > 1 and pairwise_weight > 0.0:
        target_diff = y[:, None] - y[None, :]
        pred_diff = pred[:, None] - pred[None, :]
        mask = target_diff < 0.0
        if torch.any(mask):
            loss = loss + float(pairwise_weight) * F.softplus(pred_diff[mask]).mean()
    if x.shape[0] > 1 and listwise_weight > 0.0:
        tau = max(1e-3, float(temperature))
        target_prob = torch.softmax(-y / tau, dim=0)
        pred_log_prob = torch.log_softmax(-pred / tau, dim=0)
        loss = loss + float(listwise_weight) * (-(target_prob.detach() * pred_log_prob).sum())
    return loss


def _adapt(
    model: MetaPolicyV2,
    x: torch.Tensor,
    y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    pairwise_weight: float,
    listwise_weight: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = dict(model.named_parameters())
    if x.numel() == 0:
        return params
    for _ in range(max(1, int(inner_steps))):
        loss = _ranking_loss(model, params, x, y, pairwise_weight, listwise_weight, temperature)
        gradients = torch.autograd.grad(loss, tuple(params.values()), allow_unused=False)
        params = {
            name: value - float(inner_lr) * gradient
            for (name, value), gradient in zip(params.items(), gradients)
        }
    return params


def _outer_step(
    model: MetaPolicyV2,
    optimizer: torch.optim.Optimizer,
    support: Sequence[ReplayRecord],
    query: Sequence[ReplayRecord],
    device: str,
    inner_lr: float,
    inner_steps: int,
    pairwise_weight: float,
    listwise_weight: float,
    temperature: float,
) -> None:
    if not query:
        return
    if support:
        sx = _as_tensor(np.stack([r.features for r in support]), device)
        sy = _as_tensor(np.asarray([r.cost for r in support], dtype=np.float32), device)
        adapted = _adapt(model, sx, sy, inner_lr, inner_steps, pairwise_weight, listwise_weight, temperature)
    else:
        adapted = dict(model.named_parameters())
    qx = _as_tensor(np.stack([r.features for r in query]), device)
    qy = _as_tensor(np.asarray([r.cost for r in query], dtype=np.float32), device)
    loss = _ranking_loss(model, adapted, qx, qy, pairwise_weight, listwise_weight, temperature)
    query_gradients = torch.autograd.grad(loss, tuple(adapted.values()), allow_unused=False)
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(model.parameters(), query_gradients):
        parameter.grad = gradient.detach().clone()
    optimizer.step()


def _ingest_previous_feedback(
    state: Dict,
    clients_by_id: Mapping[int, ClientInfo],
    global_reward: float,
    *,
    lambda_latency: float,
    energy_weight: float,
    diversity_weight: float,
    fairness_weight: float,
    novelty_weight: float,
    uncertainty_weight: float,
    global_reward_weight: float,
    reward_ema_alpha: float,
    normalize_latency_penalty: bool,
) -> Tuple[List[ReplayRecord], Dict[str, float]]:
    pending = state.get("pending")
    if not pending:
        return [], {}

    client_ids = [int(cid) for cid in pending["client_ids"]]
    reductions, novelty_values = [], []
    for cid in client_ids:
        client = clients_by_id.get(cid)
        if client is None:
            reductions.append(0.0)
            novelty_values.append(0.0)
            continue
        last_ema, _ = _reward_stats(state, cid)
        reduction = float(client.meta.get("last_local_loss_reduction", 0.0) or 0.0)
        reductions.append(reduction)
        novelty_values.append(abs(reduction - last_ema))

    reduction_u = _unit(reductions)
    novelty_u = _unit(novelty_values)
    energy_u = _unit(pending["energy_wh"])
    target = max(1e-8, float(pending["target_latency"]))
    query: List[ReplayRecord] = []
    component_totals = {
        "loss_utility": 0.0,
        "novelty_bonus": 0.0,
        "diversity_bonus": 0.0,
        "latency_penalty": 0.0,
        "energy_penalty": 0.0,
        "overuse_penalty": 0.0,
        "uncertainty_bonus": 0.0,
        "global_reward_bonus": 0.0,
    }

    for index, cid in enumerate(client_ids):
        client = clients_by_id.get(cid)
        if client is None:
            continue
        latency_penalty = max(0.0, float(pending["durations"][index]) - target)
        if normalize_latency_penalty:
            latency_penalty /= target
        overuse_penalty = float(pending["overuse"][index])
        diversity_bonus = float(pending["diversity_bonus"][index])
        _, reward_uncertainty = _reward_stats(state, cid)
        uncertainty_bonus = float(reward_uncertainty)
        utility = (
            float(reduction_u[index])
            + float(novelty_weight) * float(novelty_u[index])
            + float(diversity_weight) * diversity_bonus
            + float(uncertainty_weight) * uncertainty_bonus
            + float(global_reward_weight) * float(global_reward)
            - float(lambda_latency) * latency_penalty
            - float(energy_weight) * float(energy_u[index])
            - float(fairness_weight) * overuse_penalty
        )
        cost = -float(utility)
        query.append(ReplayRecord(np.asarray(pending["features"][index], dtype=np.float32), cost))
        _update_reward_stats(state, cid, utility, reward_ema_alpha)
        state["battery_wh"][cid] = max(
            0.0,
            float(state["battery_wh"].get(cid, client.battery_capacity)) - float(pending["energy_wh"][index]),
        )
        state.setdefault("last_grad_norm", {})[cid] = float(client.grad_norm or 0.0)
        state.setdefault("last_loss", {})[cid] = float(client.last_loss or 0.0)
        component_totals["loss_utility"] += float(reduction_u[index])
        component_totals["novelty_bonus"] += float(novelty_u[index])
        component_totals["diversity_bonus"] += diversity_bonus
        component_totals["latency_penalty"] += latency_penalty
        component_totals["energy_penalty"] += float(energy_u[index])
        component_totals["overuse_penalty"] += overuse_penalty
        component_totals["uncertainty_bonus"] += uncertainty_bonus
        component_totals["global_reward_bonus"] += float(global_reward)

    denom = max(1.0, float(len(query)))
    return query, {key: value / denom for key, value in component_totals.items()}


def _coverage_order(clients: Sequence[ClientInfo], selector_seed: int) -> List[int]:
    ids = [client.id for client in clients]
    np.random.default_rng(int(selector_seed)).shuffle(ids)
    return ids


def _cold_start_selection(round_idx: int, K: int, coverage_order: Sequence[int]) -> List[int]:
    if not coverage_order:
        return []
    offset = (int(round_idx) * int(K)) % len(coverage_order)
    return [int(coverage_order[(offset + index) % len(coverage_order)]) for index in range(min(int(K), len(coverage_order)))]


def _diversity_order(clients: Sequence[ClientInfo], candidate_ids: Sequence[int], raw: Mapping[str, np.ndarray]) -> List[int]:
    indices_by_id = {client.id: index for index, client in enumerate(clients)}
    covered = set()
    selected = []
    remaining = list(candidate_ids)
    while remaining:
        best_id, best_score = None, -float("inf")
        for cid in remaining:
            idx = indices_by_id[int(cid)]
            hist = _safe_histogram(clients[idx])
            new_labels = [label for label in hist if label not in covered]
            score = float(raw["rare_class_score"][idx]) + 0.25 * float(raw["label_entropy"][idx]) + 0.05 * len(new_labels)
            if score > best_score:
                best_id, best_score = int(cid), score
        if best_id is None:
            break
        selected.append(best_id)
        covered.update(_safe_histogram(clients[indices_by_id[best_id]]).keys())
        remaining.remove(best_id)
    return selected


def _take_unique(target: List[int], ordered: Sequence[int], count: int, selected: set[int]) -> None:
    for cid in ordered:
        cid = int(cid)
        if cid in selected:
            continue
        target.append(cid)
        selected.add(cid)
        if len(target) >= count:
            return


def _bucket_counts(K: int, learned_clients: Optional[int], diversity_clients: int, fairness_clients: int, cost_clients: int) -> Tuple[int, int, int, int]:
    diversity_n = max(0, min(int(diversity_clients), int(K)))
    fairness_n = max(0, min(int(fairness_clients), int(K) - diversity_n))
    cost_n = max(0, min(int(cost_clients), int(K) - diversity_n - fairness_n))
    if learned_clients is None:
        learned_n = max(0, int(K) - diversity_n - fairness_n - cost_n)
    else:
        learned_n = max(0, min(int(learned_clients), int(K) - diversity_n - fairness_n - cost_n))
    leftover = int(K) - (learned_n + diversity_n + fairness_n + cost_n)
    learned_n += max(0, leftover)
    return learned_n, diversity_n, fairness_n, cost_n


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
    lambda_latency: float = 0.25,
    target_latency_quantile: float = 0.5,
    normalize_latency_penalty: bool = True,
    cold_start_rounds: Optional[int] = None,
    exploration_clients: int = 1,
    learned_clients: Optional[int] = 6,
    diversity_clients: int = 2,
    fairness_clients: int = 1,
    cost_clients: int = 1,
    overuse_factor: float = 1.5,
    energy_weight: float = 0.10,
    diversity_weight: float = 0.35,
    fairness_weight: float = 0.35,
    novelty_weight: float = 0.20,
    uncertainty_weight: float = 0.05,
    global_reward_weight: float = 0.0,
    reward_ema_alpha: float = 0.30,
    replay_capacity: int = 128,
    support_size: int = 32,
    pairwise_weight: float = 0.30,
    listwise_weight: float = 0.10,
    ranking_temperature: float = 0.50,
    disabled_features: Sequence[str] = (),
    selector_seed: int = 2026,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Select clients with ranking-aware FOMAML and cohort-level constraints."""
    if not clients or K <= 0:
        return [], None, {}

    dev = str(device or "cpu")
    clients_by_id = {client.id: client for client in clients}
    state = history.get("state", {}).get(STATE_KEY)
    if state is None:
        model = _seeded_policy(selector_seed, dev)
        state = {
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=float(outer_lr)),
            "pending": None,
            "battery_wh": {client.id: float(client.battery_capacity) for client in clients},
            "coverage_order": _coverage_order(clients, selector_seed),
            "replay": [],
            "outer_updates": 0,
            "reward_stats": {},
            "last_grad_norm": {},
            "last_loss": {},
            "last_components": {},
        }

    global_reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    query, components = _ingest_previous_feedback(
        state,
        clients_by_id,
        global_reward,
        lambda_latency=lambda_latency,
        energy_weight=energy_weight,
        diversity_weight=diversity_weight,
        fairness_weight=fairness_weight,
        novelty_weight=novelty_weight,
        uncertainty_weight=uncertainty_weight,
        global_reward_weight=global_reward_weight,
        reward_ema_alpha=reward_ema_alpha,
        normalize_latency_penalty=normalize_latency_penalty,
    )
    replay: List[ReplayRecord] = list(state.get("replay", []))
    support = replay[-max(0, int(support_size)):] if replay else []
    model: MetaPolicyV2 = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]
    if query:
        _outer_step(
            model,
            optimizer,
            support,
            query,
            dev,
            inner_lr,
            inner_steps,
            pairwise_weight,
            listwise_weight,
            ranking_temperature,
        )
        state["outer_updates"] = int(state.get("outer_updates", 0)) + 1
        replay.extend(query)
        state["replay"] = replay[-max(1, int(replay_capacity)):]
        state["last_components"] = components

    features, raw = _client_descriptors(round_idx, clients, state, disabled_features)
    adaptation_records = list(state.get("replay", []))[-max(0, int(support_size)):]
    if adaptation_records:
        sx = _as_tensor(np.stack([record.features for record in adaptation_records]), dev)
        sy = _as_tensor(np.asarray([record.cost for record in adaptation_records], dtype=np.float32), dev)
        adapted = _adapt(model, sx, sy, inner_lr, inner_steps, pairwise_weight, listwise_weight, ranking_temperature)
    else:
        adapted = dict(model.named_parameters())

    with torch.no_grad():
        predicted_cost = functional_call(model, adapted, (_as_tensor(features, dev),))
    model_cost = predicted_cost.detach().cpu().numpy().astype(np.float64)

    expected_count = (float(round_idx) + 1.0) * float(K) / max(1.0, float(len(clients)))
    overuse = np.asarray([
        max(0.0, float(client.participation_count or 0) - float(overuse_factor) * expected_count) / max(1.0, expected_count)
        for client in clients
    ], dtype=np.float64)
    latency_u = _unit(raw["latency"])
    energy_u = _unit(raw["energy"])
    rare_u = _unit(raw["rare_class_score"])
    entropy_u = _unit(raw["label_entropy"])
    novelty_u = _unit(raw["update_novelty"])
    uncertainty_u = _unit(raw["reward_uncertainty"])
    adjusted_cost = (
        model_cost
        + float(fairness_weight) * overuse
        + float(lambda_latency) * latency_u
        + float(energy_weight) * energy_u
        - float(diversity_weight) * (0.7 * rare_u + 0.3 * entropy_u)
        - float(novelty_weight) * novelty_u
        - float(uncertainty_weight) * uncertainty_u
    )

    warmup_rounds = (
        max(0, int(cold_start_rounds))
        if cold_start_rounds is not None
        else int(math.ceil(len(clients) / max(1, int(K))))
    )
    if round_idx < warmup_rounds:
        selected_ids = _cold_start_selection(round_idx, K, state["coverage_order"])
        state["last_selection_mode"] = "coverage_cold_start"
    else:
        ranked_ids = [clients[int(index)].id for index in np.argsort(adjusted_cost)]
        diversity_ids = _diversity_order(clients, ranked_ids, raw)
        fairness_order = [
            clients[int(index)].id
            for index in np.argsort(-(np.asarray([recency(round_idx, c) for c in clients], dtype=np.float64) - overuse))
        ]
        cost_order = [
            clients[int(index)].id
            for index in np.argsort(0.6 * latency_u + 0.4 * energy_u)
        ]
        learned_n, diversity_n, fairness_n, cost_n = _bucket_counts(K, learned_clients, diversity_clients, fairness_clients, cost_clients)
        selected_ids = []
        selected_set: set[int] = set()
        _take_unique(selected_ids, ranked_ids, learned_n, selected_set)
        _take_unique(selected_ids, diversity_ids, learned_n + diversity_n, selected_set)
        _take_unique(selected_ids, fairness_order, learned_n + diversity_n + fairness_n, selected_set)
        _take_unique(selected_ids, cost_order, learned_n + diversity_n + fairness_n + cost_n, selected_set)
        _take_unique(selected_ids, ranked_ids, int(K), selected_set)
        if exploration_clients > 0 and len(selected_ids) >= int(K):
            explore_order = [
                cid for cid in fairness_order
                if cid not in set(selected_ids[:-int(exploration_clients)])
            ]
            if explore_order:
                keep = selected_ids[: max(0, int(K) - int(exploration_clients))]
                selected_ids = keep
                selected_set = set(keep)
                _take_unique(selected_ids, explore_order, int(K), selected_set)
                _take_unique(selected_ids, ranked_ids, int(K), selected_set)
        state["last_selection_mode"] = "ranked_diversity_fairness_cost"

    selected_ids = [int(cid) for cid in selected_ids[: min(int(K), len(clients))]]
    indices_by_id = {client.id: index for index, client in enumerate(clients)}
    selected_indices = [indices_by_id[cid] for cid in selected_ids]
    tier_two_durations = [
        expected_duration(client)
        for client in clients
        if int(getattr(client, "tier", -1) or -1) == 1
    ]
    if tier_two_durations:
        target_latency = float(np.mean(tier_two_durations))
    else:
        quantile = min(1.0, max(0.0, float(target_latency_quantile)))
        target_latency = float(np.quantile([expected_duration(client) for client in clients], quantile))
    diversity_bonus = (0.7 * rare_u + 0.3 * entropy_u).astype(float)
    state["pending"] = {
        "client_ids": selected_ids,
        "features": [features[int(index)].tolist() for index in selected_indices],
        "durations": [float(expected_duration(clients[int(index)])) for index in selected_indices],
        "energy_wh": [float(getattr(clients[int(index)], "estimated_energy", 0.0) or 0.0) for index in selected_indices],
        "target_latency": target_latency,
        "diversity_bonus": [float(diversity_bonus[int(index)]) for index in selected_indices],
        "overuse": [float(overuse[int(index)]) for index in selected_indices],
    }
    state["last_predicted_cost_mean"] = float(np.mean(model_cost))
    state["last_adjusted_cost_mean"] = float(np.mean(adjusted_cost))
    state["last_overuse_mean"] = float(np.mean(overuse))
    state["last_bucket_counts"] = {
        "learned": int(_bucket_counts(K, learned_clients, diversity_clients, fairness_clients, cost_clients)[0]),
        "diversity": int(_bucket_counts(K, learned_clients, diversity_clients, fairness_clients, cost_clients)[1]),
        "fairness": int(_bucket_counts(K, learned_clients, diversity_clients, fairness_clients, cost_clients)[2]),
        "cost": int(_bucket_counts(K, learned_clients, diversity_clients, fairness_clients, cost_clients)[3]),
    }
    return selected_ids, adjusted_cost.astype(float).tolist(), {STATE_KEY: state}
