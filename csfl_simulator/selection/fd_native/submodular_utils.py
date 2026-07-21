"""Shared facility-location helpers for FD-native literature baselines."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from csfl_simulator.core.client import ClientInfo


def signal_for(history: Mapping, cid: int) -> Mapping:
    signals = history.get("state", {}).get("fd_client_signals", {}) if history else {}
    return signals.get(cid, signals.get(str(cid), {}))


def representation_for(history: Mapping, cid: int):
    signal = signal_for(history, cid)
    value = signal.get("logit_representation") if isinstance(signal, Mapping) else None
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array if array.size else None


def select_for_signal_exploration(
    clients: Sequence[ClientInfo],
    K: int,
    history: Mapping,
    rng,
) -> Tuple[List[int], bool]:
    """Select unseen clients first until every client has an FD representation."""
    unseen = [c for c in clients if representation_for(history, c.id) is None]
    if not unseen:
        return [], False
    tie_break = {c.id: rng.random() for c in clients}
    ranked = sorted(
        clients,
        key=lambda c: (
            representation_for(history, c.id) is not None,
            c.participation_count,
            tie_break[c.id],
        ),
    )
    return [c.id for c in ranked[:K]], True


def cosine_facility_matrix(client_ids: Sequence[int], history: Mapping) -> np.ndarray:
    reps = [representation_for(history, cid) for cid in client_ids]
    if any(rep is None for rep in reps):
        raise ValueError("facility-location selection requires one representation per client")
    lengths = {int(rep.size) for rep in reps}
    if len(lengths) != 1:
        raise ValueError("client logit representations have inconsistent dimensions")
    matrix = np.stack(reps, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    # Cosine similarity mapped to [0, 1], preserving facility-location monotonicity.
    return np.clip((matrix @ matrix.T + 1.0) * 0.5, 0.0, 1.0)


def greedy_facility_location(
    client_ids: Sequence[int],
    similarity: np.ndarray,
    K: int,
    *,
    extra_marginal: Callable[[List[int], int], float] | None = None,
) -> Tuple[List[int], Dict[int, float]]:
    """Naive greedy maximization of normalized facility-location utility."""
    ids = list(client_ids)
    if not ids or K <= 0:
        return [], {}
    K = min(int(K), len(ids))
    index = {cid: pos for pos, cid in enumerate(ids)}
    covered = np.zeros(len(ids), dtype=np.float64)
    remaining = set(ids)
    selected: List[int] = []
    gains: Dict[int, float] = {}

    while remaining and len(selected) < K:
        best_cid = None
        best_gain = float("-inf")
        for cid in sorted(remaining):
            col = similarity[:, index[cid]]
            facility_gain = float(np.maximum(covered, col).sum() - covered.sum()) / len(ids)
            gain = facility_gain
            if extra_marginal is not None:
                gain += float(extra_marginal(selected, cid))
            if gain > best_gain:
                best_cid, best_gain = cid, gain
        if best_cid is None:
            break
        selected.append(best_cid)
        remaining.remove(best_cid)
        covered = np.maximum(covered, similarity[:, index[best_cid]])
        gains[best_cid] = best_gain
    return selected, gains


def selector_output(
    clients: Sequence[ClientInfo],
    selected: Iterable[int],
    gains: Mapping[int, float] | None = None,
) -> List[float]:
    selected_set = set(selected)
    gains = gains or {}
    return [float(gains.get(c.id, 1.0 if c.id in selected_set else 0.0)) for c in clients]
