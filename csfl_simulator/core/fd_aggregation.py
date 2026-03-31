"""Logit-level aggregation for Federated Distillation.

Implements:
  - Weighted logit averaging  (Eq. 17, Mu et al. IEEE TCCN 2024)
  - Group-based aggregation   (Algorithm 2, FedTSKD-G)
"""
from __future__ import annotations
from typing import List, Optional

import torch


def logit_avg(
    logits_list: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    """Weighted average of logit tensors.

    Each tensor in *logits_list* has shape ``(N_public, num_classes)``.
    Weights are typically ``lambda_n = D_n / sum(D_j)`` (Eq. 17).

    Returns a tensor of the same shape as each element.
    """
    total_w = sum(weights)
    if total_w == 0:
        return torch.zeros_like(logits_list[0])
    result = torch.zeros_like(logits_list[0])
    for logits, w in zip(logits_list, weights):
        result.add_(logits, alpha=w / total_w)
    return result


def logit_avg_grouped(
    logits_list: List[torch.Tensor],
    weights: List[float],
    group_labels: List[str],
) -> torch.Tensor:
    """Group-based logit aggregation for FedTSKD-G (Algorithm 2).

    Clients are split into ``"good"`` and ``"bad"`` channel groups.
    Logits from each group are averaged independently, then concatenated
    along the class dimension:

        z_hat^[u] = [z_hat^[u,bad], z_hat^[u,good]]

    This doubles the output dimension to ``2 * num_classes``.  The server
    model's last FC layer must be adapted accordingly.

    Returns:
        Tensor of shape ``(N_public, 2 * num_classes)``.
    """
    good_logits = [l for l, g in zip(logits_list, group_labels) if g == "good"]
    good_weights = [w for w, g in zip(weights, group_labels) if g == "good"]
    bad_logits = [l for l, g in zip(logits_list, group_labels) if g == "bad"]
    bad_weights = [w for w, g in zip(weights, group_labels) if g == "bad"]

    if good_logits:
        z_good = logit_avg(good_logits, good_weights)
    else:
        z_good = torch.zeros_like(logits_list[0])

    if bad_logits:
        z_bad = logit_avg(bad_logits, bad_weights)
    else:
        z_bad = torch.zeros_like(logits_list[0])

    return torch.cat([z_bad, z_good], dim=-1)  # (N_public, 2*C)
