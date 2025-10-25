from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import random

from csfl_simulator.core.client import ClientInfo

try:
    import torch
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False


class SimpleGAT(torch.nn.Module):
    def __init__(self, in_dim=6, hidden=32, heads=2):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden, heads=heads)
        self.g2 = GATConv(hidden*heads, hidden, heads=1)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.g1(x, edge_index))
        h = torch.relu(self.g2(h, edge_index))
        return self.lin(h).squeeze(-1)


def _features(c: ClientInfo):
    # Align with resource-focused features seen in cifar10-gat.py
    # [compute_speed, channel_quality, normalized_data_size, participation, last_contribution (approx via grad_norm)]
    data_norm = float(c.data_size or 0.0)
    contrib = float(c.grad_norm or 0.0)
    return [
        float(c.compute_speed or 1.0),
        float(c.channel_quality or 1.0),
        data_norm,
        float(c.participation_count or 0.0),
        contrib,
    ]


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    if not PYG_AVAILABLE or len(clients) == 0:
        ids = [c.id for c in clients]
        rng.shuffle(ids)
        return ids[:K], None, {}
    import itertools
    dev = device or 'cpu'
    x = torch.tensor([_features(c) for c in clients], dtype=torch.float, device=dev)
    n = len(clients)
    edges = torch.tensor([[i, j] for i, j in itertools.permutations(range(n), 2) if i != j], dtype=torch.long).t().contiguous()
    model = SimpleGAT(in_dim=x.shape[1]).to(dev)
    with torch.no_grad():
        scores = model(x, edges).detach().cpu().numpy().tolist()
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sel = [clients[i].id for i in ranked[:K]]
    return sel, scores, {}
