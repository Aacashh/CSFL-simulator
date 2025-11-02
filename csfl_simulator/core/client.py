from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ClientInfo:
    id: int
    data_size: int
    last_loss: float = 0.0
    grad_norm: float = 0.0
    compute_speed: float = 1.0
    dp_epsilon_remaining: float = 10.0
    dp_epsilon_used: float = 0.0
    participation_count: int = 0
    position: Optional[List[float]] = None
    channel_quality: float = 1.0
    estimated_duration: float = 0.0
    # Energy/bandwidth-related fields
    energy_rate: float = 1.0  # abstract units of energy per unit time
    battery_capacity: float = 100.0  # abstract units; not enforced unless a policy uses it
    region_carbon_g_per_kwh: float = 400.0  # rough regional carbon intensity proxy
    estimated_energy: float = 0.0  # energy estimate for this round (computed each round)
    estimated_bytes: float = 0.0   # bytes to communicate this round (approx.)
    tier: Optional[int] = None
    utility_estimate: float = 0.0
    last_selected_round: int = -1
    label_histogram: Optional[Dict[int, int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
