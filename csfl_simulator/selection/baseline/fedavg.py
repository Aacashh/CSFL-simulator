from typing import List, Dict, Optional, Tuple

from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, fraction: Optional[float] = None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """FedAvg client selection: uniform random sampling without replacement.

    If 'fraction' (C) is provided, the effective cohort size becomes
    ceil(C * N) clipped to [1, N]; otherwise use K.

    Reference:
    - McMahan, H.B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A.
      Communication-Efficient Learning of Deep Networks from Decentralized Data.
      AISTATS 2017 (PMLR 54:1273–1282). arXiv:1602.05629.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    k_eff = int(K)
    if fraction is not None:
        try:
            c = float(fraction)
            if c > 0:
                k_eff = max(1, min(n, int(round(c * n))))
        except Exception:
            pass
    ids = [c.id for c in clients]
    rng.shuffle(ids)
    sel = ids[: min(k_eff, n)]
    return sel, None, {}


