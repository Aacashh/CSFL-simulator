from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy, recency


STATE = "maml_select_state"


def _features(clients: List[ClientInfo]) -> np.ndarray:
    data = np.array([float(c.data_size or 0.0) for c in clients], dtype=float)
    loss = np.array([float(c.last_loss or 0.0) for c in clients], dtype=float)
    gnorm = np.array([float(c.grad_norm or 0.0) for c in clients], dtype=float)
    inv_dur = np.array([1.0 / max(1e-6, expected_duration(c)) for c in clients], dtype=float)
    part = np.array([float(c.participation_count or 0.0) for c in clients], dtype=float)
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

    cols = [data, loss, gnorm, inv_dur, part, ent]
    X = np.stack(cols, axis=1)
    # Standardize per call
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sigma


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None,
                   max_history: int = 5000, inner_steps: int = 3, lr: float = 1e-3,
                   exploration_ucb: float = 0.2) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """MAML-Select: a tiny MLP that learns to predict contribution and adapts every round.

    - Maintains a buffer of (features, reward) from previous rounds for selected clients.
    - Performs a few gradient steps (inner_steps) each round to adapt quickly.
    - Uses a mild UCB-like term based on recency for exploration.
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    dev = device or "cpu"
    X_np = _features(clients)
    X = torch.tensor(X_np, dtype=torch.float, device=dev)

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        model = MLP(in_dim=X.shape[1], hidden=64).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        st = {
            "model": model,
            "opt": opt,
            "buf_X": [],
            "buf_y": [],
        }

    model: MLP = st["model"]
    opt: torch.optim.Optimizer = st["opt"]

    # Online adaptation using last round credit (reward equally to last selected)
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = history.get("selected", [])[-1] if history.get("selected") else []
    if last_sel:
        credit = reward  # supervised target for selected clients
        # Append training samples
        for cid in last_sel:
            try:
                idx = next(i for i, c in enumerate(clients) if c.id == cid)
                st["buf_X"].append(X_np[idx])
                st["buf_y"].append([credit])
            except StopIteration:
                # If the client is not in current pool, skip (e.g., due to filtering)
                pass
        # Bound buffer size
        if len(st["buf_X"]) > max_history:
            overflow = len(st["buf_X"]) - max_history
            st["buf_X"] = st["buf_X"][overflow:]
            st["buf_y"] = st["buf_y"][overflow:]
        # Perform a few adaptation steps
        if len(st["buf_X"]) >= max(10, K):
            Xtr = torch.tensor(np.array(st["buf_X"], dtype=float), dtype=torch.float, device=dev)
            ytr = torch.tensor(np.array(st["buf_y"], dtype=float).reshape(-1), dtype=torch.float, device=dev)
            for _ in range(int(inner_steps)):
                opt.zero_grad()
                pred = model(Xtr)
                loss = torch.nn.functional.mse_loss(pred, ytr)
                loss.backward()
                opt.step()

    # Predict contribution scores
    with torch.no_grad():
        y_hat = model(X).detach().cpu().numpy()

    # Add mild UCB-style exploration based on recency (less recent => small boost)
    boosts = []
    for c in clients:
        gap = float(recency(round_idx, c))
        boosts.append(exploration_ucb * (gap / (gap + 5.0)))
    y_hat = y_hat + np.array(boosts, dtype=float)

    ranked = np.argsort(-y_hat)
    sel = [clients[i].id for i in ranked[:K].tolist()]
    per_client = [float(y_hat[i]) for i in range(n)]

    return sel, per_client, {STATE: st}


