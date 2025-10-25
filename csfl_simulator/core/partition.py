from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List


def iid_partition(labels: list[int], num_clients: int) -> Dict[int, List[int]]:
    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)
    return {i: idxs[i::num_clients].tolist() for i in range(num_clients)}


def dirichlet_partition(labels: list[int], num_clients: int, alpha: float = 0.5, num_classes: int | None = None) -> Dict[int, List[int]]:
    y = np.array(labels)
    if num_classes is None:
        num_classes = int(y.max() + 1)
    cls_idx = [np.where(y == c)[0] for c in range(num_classes)]
    client_bins: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        n = len(cls_idx[c])
        np.random.shuffle(cls_idx[c])
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(proportions) * n).astype(int)[:-1]
        split = np.split(cls_idx[c], cuts)
        for i, part in enumerate(split):
            client_bins[i].extend(part.tolist())
    return client_bins


def label_shard_partition(labels: list[int], num_clients: int, shards_per_client: int = 2, num_classes: int | None = None) -> Dict[int, List[int]]:
    y = np.array(labels)
    if num_classes is None:
        num_classes = int(y.max() + 1)
    # Create shards: shuffle all indices and split into (num_clients*shards_per_client) shards
    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)
    num_shards = num_clients * shards_per_client
    shard_size = len(idxs) // num_shards
    shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    # Assign shards randomly to clients
    assignments: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    order = np.random.permutation(num_shards)
    for i, s in enumerate(order):
        assignments[i % num_clients].extend(shards[s].tolist())
    return assignments


def partition_dataset(dataset, strategy: str, **kwargs) -> Dict[int, List[int]]:
    labels = [dataset[i][1] for i in range(len(dataset))]
    if strategy == "iid":
        return iid_partition(labels, kwargs["num_clients"])
    if strategy == "dirichlet":
        return dirichlet_partition(labels, kwargs["num_clients"], kwargs.get("alpha", 0.5), kwargs.get("num_classes"))
    if strategy in ("label-shard", "label_shard"):
        return label_shard_partition(labels, kwargs["num_clients"], kwargs.get("shards_per_client", 2), kwargs.get("num_classes"))
    raise ValueError(f"Unknown partition strategy: {strategy}")
