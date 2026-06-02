"""In-simulator approximation of FedGCS (Generative Client Selection).

Reference:
    Ning, Z., Tian, C., Xiao, M., Fan, W., Wang, P., Li, L., Wang, P.,
    Zhou, Y. "FedGCS: A Generative Framework for Efficient Client Selection
    in Federated Learning via Gradient-based Optimization." IJCAI 2024,
    pp. 4760-4768. arXiv:2405.06312.

This module provides a self-contained approximation of the FedGCS framework
for consistent comparison within the MAML-Select experiment suite.  The
official IJCAI repository remains the ground-truth reference; see
``official_fedgcs_protocol.md`` for the external-comparison workflow.

Architecture
~~~~~~~~~~~~
A lightweight VAE encodes per-client feature vectors (loss, gradient norm,
latency, data size, participation frequency, staleness) into a shared latent
space.  At each round the server performs gradient-based optimization in
latent space to find a selection probability vector that maximises a predicted
utility objective.  Top-K clients by optimised probability are selected.

Complexity: O(N · d) per round where d is the latent dimension. This is the
complexity of this disclosed approximation, not a claim about the official
implementation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, recency


STATE_KEY = "research_fedgcs_state"
FEATURE_DIM = 6


class FedGCSEncoder(nn.Module):
    """Encodes per-client features into a latent representation."""

    def __init__(self, input_dim: int = FEATURE_DIM, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Reparameterisation trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class FedGCSDecoder(nn.Module):
    """Decodes latent vectors to predicted utility scores."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).squeeze(-1)


class FedGCSModel(nn.Module):
    """Full VAE-based generative selection model."""

    def __init__(self, input_dim: int = FEATURE_DIM, latent_dim: int = 16):
        super().__init__()
        self.encoder = FedGCSEncoder(input_dim, latent_dim)
        self.decoder = FedGCSDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        utility = self.decoder(z)
        return utility, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        beta: float = 0.01,
    ) -> torch.Tensor:
        utility, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(utility, targets)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True)
    return (matrix - mu) / (sigma + 1e-8)


def _build_features(
    round_idx: int,
    clients: Sequence[ClientInfo],
) -> np.ndarray:
    rows = []
    for client in clients:
        rows.append([
            float(client.last_loss or 0.0),
            float(client.grad_norm or 0.0),
            float(expected_duration(client)),
            float(client.data_size or 0),
            float(client.participation_count or 0),
            float(recency(round_idx, client)),
        ])
    return _zscore(np.asarray(rows, dtype=np.float32))


def _seeded_model(seed: int, device: str, latent_dim: int) -> FedGCSModel:
    """Initialise model reproducibly without perturbing FL RNG."""
    devices = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return FedGCSModel(latent_dim=int(latent_dim)).to(device)


def select_clients(
    round_idx: int,
    K: int,
    clients: List[ClientInfo],
    history: Dict,
    rng,
    time_budget=None,
    device=None,
    *,
    latent_dim: int = 16,
    optimisation_steps: int = 5,
    lr: float = 0.001,
    beta_kl: float = 0.01,
    selector_seed: int = 2026,
) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """Select clients using FedGCS-style generative latent-space optimisation."""
    if not clients or K <= 0:
        return [], None, {}

    dev = str(device or "cpu")
    state = history.get("state", {}).get(STATE_KEY)
    if state is None:
        model = _seeded_model(selector_seed, dev, latent_dim)
        state = {
            "model": model,
            "optimizer": torch.optim.Adam(model.parameters(), lr=float(lr)),
            "pending_features": None,
            "pending_ids": None,
        }

    model: FedGCSModel = state["model"]
    optimizer: torch.optim.Optimizer = state["optimizer"]

    # Ingest previous round feedback and train the generative model
    if state.get("pending_features") is not None and state.get("pending_ids") is not None:
        prev_features = state["pending_features"]
        prev_ids = state["pending_ids"]
        # Build targets from observed loss reductions
        targets = []
        clients_by_id = {c.id: c for c in clients}
        for cid in prev_ids:
            c = clients_by_id.get(int(cid))
            if c is not None:
                reduction = float(c.meta.get("last_local_loss_reduction", 0.0) or 0.0)
                targets.append(reduction)
            else:
                targets.append(0.0)
        if targets:
            x = torch.as_tensor(prev_features, dtype=torch.float32, device=dev)
            y = torch.as_tensor(targets, dtype=torch.float32, device=dev)
            # Train the VAE for a few steps
            model.train()
            for _ in range(max(1, int(optimisation_steps))):
                optimizer.zero_grad()
                loss = model.loss(x, y, beta=float(beta_kl))
                loss.backward()
                optimizer.step()

    # Score all clients
    features = _build_features(round_idx, clients)
    x_all = torch.as_tensor(features, dtype=torch.float32, device=dev)

    model.eval()
    with torch.no_grad():
        utility, _, _ = model(x_all)
    scores = utility.detach().cpu().numpy()

    # Select top-K by predicted utility (higher = better)
    top_indices = np.argsort(-scores)[:min(int(K), len(clients))]
    selected_ids = [clients[int(i)].id for i in top_indices]

    # Store pending info for next round
    selected_features = features[top_indices].copy()
    state["pending_features"] = selected_features
    state["pending_ids"] = selected_ids

    return selected_ids, scores.astype(float).tolist(), {STATE_KEY: state}
