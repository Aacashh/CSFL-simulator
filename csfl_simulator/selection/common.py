from __future__ import annotations
from typing import List
import math

from csfl_simulator.core.client import ClientInfo


def normalize_list(xs: List[float], eps: float = 1e-8) -> List[float]:
    if not xs:
        return xs
    m = min(xs)
    M = max(xs)
    if not math.isfinite(m) or not math.isfinite(M) or abs(M - m) < eps:
        return [0.0 for _ in xs]
    return [(x - m) / (M - m + eps) for x in xs]


def choose_random(clients: List[ClientInfo], K: int, rng) -> List[int]:
    ids = [c.id for c in clients]
    rng.shuffle(ids)
    return ids[: min(K, len(ids))]


def expected_duration(c: ClientInfo) -> float:
    # Prefer provided estimate
    if c.estimated_duration and c.estimated_duration > 0:
        return float(c.estimated_duration)
    # Fallback estimate from simple compute/network model
    data = float(c.data_size or 0.0)
    comp = data / max(1e-6, float(c.compute_speed or 1.0))
    net = (data / 1000.0) * (1.0 / max(1e-6, float(c.channel_quality or 1.0)))
    return comp + net


def recency(round_idx: int, c: ClientInfo) -> int:
    try:
        last = int(c.last_selected_round)
    except Exception:
        last = -1
    return max(0, int(round_idx) - last)


def label_entropy(counts: List[float]) -> float:
    s = float(sum(counts))
    if s <= 0:
        return 0.0
    ent = 0.0
    for v in counts:
        if v <= 0:
            continue
        p = float(v) / s
        ent += -p * math.log(max(p, 1e-12))
    return ent

