from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy

STATE = "deepset_ranker_state"


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


class Phi(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Psi(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, rep_dim: int = 32, lr: float = 1e-3,
                   fairness_alpha: float = 0.2, time_alpha: float = 0.2) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    dev = device or 'cpu'
    X = _features(clients)
    X_t = torch.tensor(X, dtype=torch.float, device=dev)

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        phi = Phi(in_dim=X.shape[1], hidden=32, out_dim=rep_dim).to(dev)
        psi = Psi(in_dim=rep_dim * 2, hidden=32).to(dev)
        opt = torch.optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=lr)
        st = {"phi": phi, "psi": psi, "opt": opt, "last_phi": {}}

    phi: Phi = st["phi"]
    psi: Psi = st["psi"]
    opt: torch.optim.Optimizer = st["opt"]

    with torch.no_grad():
        H = phi(X_t)
        s = H.sum(dim=0, keepdim=True).repeat(H.shape[0], 1)
        scores_t = psi(torch.cat([H, s], dim=1))
        scores = scores_t.detach().cpu().numpy().tolist()

    # Rank and select
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sel = [clients[i].id for i in ranked[:K]]

    # Online training with last reward; shape target using fairness/time proxies
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        ids = [cid for cid in last_sel]
        if ids:
            opt.zero_grad()
            # Approx targets per selected using shaped reward
            targ = []
            idxs = []
            for cid in ids:
                i = next((j for j, c in enumerate(clients) if c.id == cid), None)
                if i is None:
                    continue
                # Penalize frequent participants, encourage faster clients
                part_pen = 1.0 / (1.0 + float(clients[i].participation_count or 0.0))
                inv_time = 1.0 / (1.0 + float(getattr(clients[i], 'estimated_duration', 0.0) or 0.0))
                shaped = reward * (1.0 + time_alpha * inv_time) * (1.0 + fairness_alpha * part_pen)
                targ.append(shaped)
                idxs.append(i)
            if idxs:
                H = phi(X_t)
                s = H.sum(dim=0, keepdim=True).repeat(H.shape[0], 1)
                pred = psi(torch.cat([H[idxs], s[idxs]], dim=1))
                target = torch.tensor(targ, dtype=torch.float, device=dev)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                opt.step()

    # Per-client scores in original order
    per_client = [float(scores[i]) for i in range(n)]
    st["last_phi"] = {c.id: float(scores[i]) for i, c in enumerate(clients)}
    return sel, per_client, {STATE: st}


