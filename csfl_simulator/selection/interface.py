from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Interface function signature for client selection methods.
    - round_idx: current round number
    - K: number of clients to select
    - clients: list of ClientInfo states
    - history: dict with "state" (method state) and "selected" (history list)
    - rng: random-like object with .shuffle, .random, etc.
    - time_budget: optional round time budget
    - device: torch device string
    Return: (selected_ids, optional_scores_per_client, optional_state_update)
    """
    raise NotImplementedError
