from __future__ import annotations
from collections import OrderedDict
import torch


def fedavg(state_dicts, weights):
    new_sd = OrderedDict()
    total = sum(weights)
    for k in state_dicts[0].keys():
        acc = None
        for sd, w in zip(state_dicts, weights):
            v = sd[k]
            if acc is None:
                acc = v * (w / total)
            else:
                acc = acc + v * (w / total)
        new_sd[k] = acc.clone() if isinstance(acc, torch.Tensor) else acc
    return new_sd
