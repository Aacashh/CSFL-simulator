from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

from csfl_simulator.core.client import ClientInfo


class GTSelector(nn.Module):
    def __init__(self, in_dim=6, hidden=64, heads=4, layers=2, dropout=0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hidden, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.scorer = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.scorer(h).squeeze(-1)


def _features(c: ClientInfo):
    return [
        float(c.data_size or 0.0),
        float(c.compute_speed or 1.0),
        float(c.channel_quality or 1.0),
        float(c.participation_count or 0.0),
        float(c.last_loss or 0.0),
        float(c.grad_norm or 0.0),
    ]


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    if len(clients) == 0:
        return [], None, {}
    dev = device or 'cpu'
    x = torch.tensor([_features(c) for c in clients], dtype=torch.float, device=dev)
    model = GTSelector(in_dim=x.shape[1]).to(dev)
    with torch.no_grad():
        raw_scores = model(x).detach().cpu().numpy().tolist()
    # Fairness correction similar to GT-PPCS: penalize frequent participants
    fairness_alpha = 0.3
    scores = [s * (1 - fairness_alpha) + fairness_alpha * (1.0 / (1.0 + c.participation_count)) for s, c in zip(raw_scores, clients)]
    ranked = sorted(range(len(clients)), key=lambda i: scores[i], reverse=True)
    sel = [clients[i].id for i in ranked[:K]]
    return sel, scores, {}
