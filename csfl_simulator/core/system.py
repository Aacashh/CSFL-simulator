from __future__ import annotations
import math
import random
from typing import List, Dict, Any

from .client import ClientInfo


def init_system_state(clients: List[ClientInfo], knobs: Dict[str, Any]):
    # Sample compute_speed (lognormal), channel_quality (uniform)
    for c in clients:
        c.compute_speed = max(0.1, random.lognormvariate(0.0, 0.5))
        c.channel_quality = random.uniform(0.5, 1.0)
        c.tier = 0 if c.compute_speed < 0.8 else (1 if c.compute_speed < 1.2 else 2)


def simulate_round_env(clients: List[ClientInfo], knobs: Dict[str, Any], round_idx: int):
    # Update channel quality with small random walk; estimate durations
    net_scale = knobs.get("network_scale", 1.0)
    comp_scale = knobs.get("compute_scale", 1.0)
    for c in clients:
        c.channel_quality = min(1.5, max(0.1, c.channel_quality + random.uniform(-0.05, 0.05)))
        est_compute = (c.data_size / max(1e-6, c.compute_speed)) * comp_scale
        est_net = (c.data_size / 1000.0) * (1.0 / max(1e-6, c.channel_quality)) * net_scale
        c.estimated_duration = est_compute + est_net
