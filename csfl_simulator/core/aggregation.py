from __future__ import annotations
from collections import OrderedDict
import torch


def fedavg(state_dicts, weights, keep_on_device: bool = True):
    """Federated Averaging (FedAvg) aggregation.

    Computes the weighted average of client model parameters as in the original
    FedAvg paper, using data-size weights. Implemented with in-place weighted
    additions under torch.no_grad for speed and minimal memory overhead.
    
    Args:
        state_dicts: List of model state dictionaries from clients
        weights: List of weights (typically data sizes) for weighted averaging
        keep_on_device: If True, keeps tensors on their original device (GPU) during
                       aggregation for better performance. Set to False to force CPU.
    
    Returns:
        Aggregated state dictionary
    """
    new_sd = OrderedDict()
    total = float(sum(weights))
    if total == 0.0:
        scales = [1.0 / max(1, len(weights))] * len(weights)
    else:
        scales = [float(w) / total for w in weights]

    with torch.no_grad():
        for k in state_dicts[0].keys():
            v0 = state_dicts[0][k]
            if isinstance(v0, torch.Tensor):
                # Keep tensors on device for GPU-accelerated aggregation
                if keep_on_device:
                    acc = v0.detach().clone().mul_(scales[0])
                    for i in range(1, len(state_dicts)):
                        # Ensure all tensors are on the same device
                        vi = state_dicts[i][k]
                        if vi.device != acc.device:
                            vi = vi.to(acc.device)
                        acc.add_(vi, alpha=scales[i])
                else:
                    # Original behavior: may move to CPU
                    acc = v0.detach().clone().mul_(scales[0])
                    for i in range(1, len(state_dicts)):
                        acc.add_(state_dicts[i][k], alpha=scales[i])
                new_sd[k] = acc
            else:
                # Non-tensor entries (e.g., counters) are carried from the first state
                # to preserve dtype and avoid invalid weighted mixing.
                new_sd[k] = v0
    return new_sd
