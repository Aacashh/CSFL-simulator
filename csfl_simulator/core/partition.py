from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List
from math import ceil


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
    # Sort indices by label so each shard contains samples from 1-2 classes (non-IID)
    idxs = np.arange(len(labels))
    idxs = idxs[np.argsort(y)]
    num_shards = num_clients * shards_per_client
    shard_size = len(idxs) // num_shards
    shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    # Assign shards randomly to clients
    assignments: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    order = np.random.permutation(num_shards)
    for i, s in enumerate(order):
        assignments[i % num_clients].extend(shards[s].tolist())
    return assignments


def _generate_size_targets(num_clients: int, total_size: int, size_distribution: str = "uniform",
                           mu: float = 0.0, sigma: float = 0.5, alpha: float = 1.5) -> List[int]:
    """Generate integer target sizes per client that sum to total_size."""
    size_distribution = (size_distribution or "uniform").lower()
    if size_distribution == "uniform":
        base = total_size // num_clients
        sizes = [base] * num_clients
        rem = total_size - base * num_clients
        # Distribute remainder to first few clients
        for i in range(rem):
            sizes[i] += 1
        return sizes
    # Draw positive weights then normalize to total_size
    if size_distribution in ("lognormal", "log-normal", "log_norm"):
        weights = np.random.lognormal(mean=mu, sigma=sigma, size=num_clients)
    elif size_distribution in ("power_law", "power-law", "pareto", "zipf"):
        # Pareto(alpha) has heavy-tail; shift by +1 so min>0
        weights = np.random.pareto(alpha, size=num_clients) + 1.0
    else:
        # Fallback to uniform if unknown
        weights = np.ones(num_clients, dtype=float)
    weights = np.maximum(weights, 1e-9)
    probs = weights / np.sum(weights)
    # Initial rounding
    sizes = np.floor(probs * total_size).astype(int).tolist()
    # Fix rounding to match total_size exactly
    diff = total_size - int(sum(sizes))
    if diff > 0:
        # assign the remaining items to clients with largest fractional parts
        fracs = (probs * total_size) - np.floor(probs * total_size)
        order = np.argsort(-fracs)  # descending
        for i in range(int(diff)):
            sizes[int(order[i % num_clients])] += 1
    elif diff < 0:
        # remove extras from largest allocations
        order = np.argsort(-(probs * total_size))
        to_remove = -int(diff)
        i = 0
        while to_remove > 0 and i < len(order):
            idx = int(order[i])
            can = min(to_remove, max(0, sizes[idx] - 0))
            if can > 0:
                sizes[idx] -= can
                to_remove -= can
            i += 1
    # Ensure no negative sizes
    sizes = [max(0, int(s)) for s in sizes]
    # Last guard: enforce exact sum by adjusting first entry
    adj = total_size - sum(sizes)
    if adj != 0 and num_clients > 0:
        sizes[0] += adj
    return sizes


def apply_size_distribution(mapping: Dict[int, List[int]], total_size: int,
                            size_distribution: str = "uniform",
                            mu: float = 0.0, sigma: float = 0.5, alpha: float = 1.5) -> Dict[int, List[int]]:
    """Rebalance an existing mapping to match a target size distribution.
    Strategy:
    1) Compute target sizes that sum to total_size.
    2) Shrink donors (current_size > target) by random sampling → pool surplus.
    3) Fill deficit clients from the pooled surplus, preserving uniqueness of indices.
    """
    num_clients = len(mapping)
    if num_clients == 0:
        return mapping
    targets = _generate_size_targets(num_clients, total_size, size_distribution, mu, sigma, alpha)
    # Copy mapping to avoid mutating caller
    new_map: Dict[int, List[int]] = {cid: list(idxs) for cid, idxs in mapping.items()}
    rng = np.random.default_rng(np.random.randint(0, 2**31))
    pool: List[int] = []
    # Shrink donors and build pool
    for cid in range(num_clients):
        cur = new_map.get(cid, [])
        t = int(targets[cid])
        if len(cur) > t:
            # random choose t to keep; move surplus to pool
            arr = np.array(cur, dtype=int)
            sel = rng.choice(arr, size=t, replace=False).tolist() if t > 0 else []
            rem_count = len(cur) - t
            if rem_count > 0:
                # collect the remaining (those not in sel)
                mask_keep = np.zeros(len(arr), dtype=bool)
                if t > 0:
                    # mark kept positions by indices in arr
                    keep_set = set(sel)
                    for i, v in enumerate(arr):
                        if v in keep_set and not mask_keep[i]:
                            mask_keep[i] = True
                            # remove one instance from keep_set to handle potential duplicates safely
                            try:
                                keep_set.remove(v)
                            except KeyError:
                                pass
                surplus = arr[~mask_keep].tolist()
                pool.extend(surplus)
            new_map[cid] = sel
    # Fill deficits from pool
    rng.shuffle(pool)
    pool_ptr = 0
    for cid in range(num_clients):
        cur = new_map.get(cid, [])
        t = int(targets[cid])
        need = max(0, t - len(cur))
        if need > 0:
            take = pool[pool_ptr: pool_ptr + need]
            pool_ptr += len(take)
            new_map[cid] = cur + take
    # If we still have leftovers or deficits due to rounding, do a final pass
    # Recompute totals and fix by moving items between clients if necessary
    total_now = sum(len(v) for v in new_map.values())
    if total_now != total_size:
        # Adjust first client to fix any residual mismatch without duplicates
        # If excess: drop from first client
        # If deficit: append from leftover pool (if any)
        first = 0
        if total_now > total_size:
            drop = total_now - total_size
            new_map[first] = new_map[first][:-drop] if drop < len(new_map[first]) else []
        elif total_now < total_size:
            add = total_size - total_now
            extras = pool[pool_ptr: pool_ptr + add]
            new_map[first].extend(extras)
    return new_map


def partition_dataset(dataset, strategy: str, **kwargs) -> Dict[int, List[int]]:
    labels = [dataset[i][1] for i in range(len(dataset))]
    if strategy == "iid":
        return iid_partition(labels, kwargs["num_clients"])
    if strategy == "dirichlet":
        return dirichlet_partition(labels, kwargs["num_clients"], kwargs.get("alpha", 0.5), kwargs.get("num_classes"))
    if strategy in ("label-shard", "label_shard"):
        return label_shard_partition(labels, kwargs["num_clients"], kwargs.get("shards_per_client", 2), kwargs.get("num_classes"))
    raise ValueError(f"Unknown partition strategy: {strategy}")
