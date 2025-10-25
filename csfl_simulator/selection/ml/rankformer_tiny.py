from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy

STATE = "rankformer_tiny_state"


def _features(clients: List[ClientInfo]) -> np.ndarray:
    data = np.array([float(c.data_size or 0.0) for c in clients], dtype=float)
    loss = np.array([float(c.last_loss or 0.0) for c in clients], dtype=float)
    gnorm = np.array([float(c.grad_norm or 0.0) for c in clients], dtype=float)
    inv_dur = np.array([1.0 / max(1e-6, expected_duration(c)) for c in clients], dtype=float)
    part = np.array([float(c.participation_count or 0.0) for c in clients], dtype=float)
    chq = np.array([float(c.channel_quality or 1.0) for c in clients], dtype=float)
    spd = np.array([float(c.compute_speed or 1.0) for c in clients], dtype=float)
    ent = []
    for c in clients:
        if isinstance(c.label_histogram, dict) and c.label_histogram:
            L = int(max(c.label_histogram.keys()) + 1)
            vec = [0.0] * L
            for k, v in c.label_histogram.items():
                idx = int(k)
                if 0 <= idx < L:
                    vec[idx] = float(v)
            ent.append(label_entropy(vec))
        else:
            ent.append(0.0)
    ent = np.array(ent, dtype=float)

    def _mm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        m, M = float(np.min(x)), float(np.max(x))
        if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
            return np.zeros_like(x)
        return (x - m) / (M - m + 1e-12)

    cols = [_mm(data), _mm(loss), _mm(gnorm), _mm(inv_dur), _mm(part), _mm(chq), _mm(spd), _mm(ent)]
    X = np.stack(cols, axis=1)
    return X


class TinyRankFormer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 32, nhead: int = 2, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        return self.scorer(h).squeeze(-1)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, d_model: int = 32, nhead: int = 2, layers: int = 1,
                   lr: float = 1e-3, exploration_dropout: float = 0.1) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    dev = device or 'cpu'
    X = _features(clients)
    X_t = torch.tensor(X, dtype=torch.float, device=dev).unsqueeze(0)  # (1, N, F)

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        model = TinyRankFormer(in_dim=X.shape[1], d_model=d_model, nhead=nhead, layers=layers, dropout=exploration_dropout).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        st = {"model": model, "opt": opt}

    model: TinyRankFormer = st["model"]
    opt: torch.optim.Optimizer = st["opt"]

    model.train(False)
    with torch.no_grad():
        scores_t = model(X_t).squeeze(0)
        scores = scores_t.detach().cpu().numpy().tolist()

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sel = [clients[i].id for i in ranked[:K]]

    # Online training with last reward
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        idxs = [next((j for j, c in enumerate(clients) if c.id == cid), None) for cid in last_sel]
        idxs = [i for i in idxs if i is not None]
        if idxs:
            model.train(True)
            opt.zero_grad()
            out = model(X_t).squeeze(0)[idxs]
            target = torch.full_like(out, fill_value=reward)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            opt.step()

    per_client = [float(scores[i]) for i in range(n)]
    return sel, per_client, {STATE: st}


