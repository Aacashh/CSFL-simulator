from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from csfl_simulator.core.client import ClientInfo


def _feature(c: ClientInfo):
    return [c.last_loss or 0.0, c.grad_norm or 0.0, float(c.data_size)]


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    if len(clients) <= K:
        return [c.id for c in clients], None, {}
    X = np.array([_feature(c) for c in clients], dtype=float)
    n_clusters = min(K, max(1, int(np.sqrt(len(clients)))))
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = kmeans.fit_predict(X)
    selected = []
    per_cluster = max(1, K // n_clusters)
    for cl in range(n_clusters):
        ids = [i for i, lab in enumerate(labels) if lab == cl]
        rng.shuffle(ids)
        take = ids[:per_cluster]
        selected.extend(take)
    if len(selected) < K:
        remaining = [i for i in range(len(clients)) if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:K-len(selected)])
    sel_ids = [clients[i].id for i in selected[:K]]
    return sel_ids, None, {}
