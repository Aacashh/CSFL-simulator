from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy

STATE = "neural_linear_ucb_state"


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


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, rep_dim: int = 32, alpha_ucb: float = 0.5,
                   lr: float = 1e-3) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
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
        mlp = TinyMLP(in_dim=X.shape[1], hidden=32, out_dim=rep_dim).to(dev)
        A_inv = np.eye(rep_dim, dtype=float)
        b = np.zeros((rep_dim,), dtype=float)
        st = {"mlp": mlp, "A_inv": A_inv, "b": b, "opt": torch.optim.Adam(mlp.parameters(), lr=lr), "last_z": {}}

    mlp: TinyMLP = st["mlp"]
    opt: torch.optim.Optimizer = st["opt"]

    with torch.no_grad():
        Z = mlp(X_t).cpu().numpy()

    theta = st["A_inv"] @ st["b"]
    scores = []
    z_cache = {}
    for i, c in enumerate(clients):
        z = Z[i]
        z_cache[c.id] = z
        mean = float(theta @ z)
        conf = float(np.sqrt(z.T @ st["A_inv"] @ z))
        p = mean + alpha_ucb * conf
        scores.append((c.id, p))
    scores.sort(key=lambda t: t[1], reverse=True)
    sel = [cid for cid, _ in scores[:K]]

    # Online update using last reward (composite) and last representations
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        # Train MLP to predict reward from features via simple MSE on selected z->linear prediction
        opt.zero_grad()
        ids = [cid for cid in last_sel if cid in st.get("last_z", {})]
        if ids:
            zs = torch.tensor([st["last_z"][cid] for cid in ids], dtype=torch.float, device=dev)
            with torch.no_grad():
                theta_t = torch.tensor(theta, dtype=torch.float, device=dev)
            pred = (zs @ theta_t).view(-1)
            target = torch.full_like(pred, fill_value=reward)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            opt.step()
        # Update Bayesian linear head with Sherman–Morrison for each selected
        for cid in last_sel:
            z = st["last_z"].get(cid, None)
            if z is None:
                continue
            z = z.reshape(-1, 1)
            A_inv = st["A_inv"]
            denom = 1.0 + float((z.T @ A_inv @ z).squeeze())
            A_inv = A_inv - (A_inv @ z @ z.T @ A_inv) / max(denom, 1e-12)
            st["A_inv"] = A_inv
            st["b"] = st["b"] + reward * z.flatten()

    st["last_z"] = {cid: z_cache[cid] for cid in sel}

    # Per-client scores in original order
    per_client = []
    m = {cid: s for cid, s in scores}
    for c in clients:
        per_client.append(float(m.get(c.id, 0.0)))

    return sel, per_client, {STATE: st}


