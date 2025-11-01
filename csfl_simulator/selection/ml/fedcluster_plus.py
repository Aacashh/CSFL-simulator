from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import math
import numpy as np
from sklearn.cluster import KMeans

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.selection.common import expected_duration, label_entropy, recency


STATE = "fedcluster_plus_state"


def _features(clients: List[ClientInfo]) -> np.ndarray:
    loss = np.array([float(c.last_loss or 0.0) for c in clients], dtype=float)
    gnorm = np.array([float(c.grad_norm or 0.0) for c in clients], dtype=float)
    inv_dur = np.array([1.0 / max(1e-6, expected_duration(c)) for c in clients], dtype=float)
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

    X = np.stack([loss, gnorm, inv_dur, ent], axis=1)
    # Normalize columns
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sigma


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None, refresh_every: int = 3,
                   cluster_k: Optional[int] = None, rep_weight: float = 0.5,
                   cluster_importance_decay: float = 0.95) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    """FedCluster+: adaptive clustering based on gradient/utility proxies.

    - Online k-means over compact embeddings (loss, grad_norm, 1/duration, label entropy)
    - Per-cluster importance tracked from historical reward contributions
    - Select representative clients proportionally to cluster importance
    """
    n = len(clients)
    if n <= 0:
        return [], None, {}
    if K >= n:
        return [c.id for c in clients], None, {}

    X = _features(clients)

    st = history.get("state", {}).get(STATE, None)
    if st is None:
        k = cluster_k if cluster_k and cluster_k > 1 else max(2, int(math.ceil(math.sqrt(max(K, 2)))) * 2)
        km = KMeans(n_clusters=int(k), n_init=5, random_state=0)
        st = {"kmeans": km, "fitted": False, "cluster_weight": None, "t": 0}

    km: KMeans = st["kmeans"]
    st["t"] = int(st.get("t", 0)) + 1

    # (Re)fit clusters periodically or if not yet fitted
    if (not st.get("fitted", False)) or (int(st["t"]) % max(1, int(refresh_every)) == 0):
        try:
            km.fit(X)
            st["fitted"] = True
            # Initialize cluster importance weights uniformly
            if st.get("cluster_weight") is None or len(st["cluster_weight"]) != km.n_clusters:
                st["cluster_weight"] = np.ones((km.n_clusters,), dtype=float)
        except Exception:
            # If kmeans fails, fall back to single cluster
            st["fitted"] = False
            st["cluster_weight"] = np.ones((1,), dtype=float)

    if st.get("fitted", False):
        labels = km.predict(X)
        centers = km.cluster_centers_
        n_clusters = int(km.n_clusters)
    else:
        labels = np.zeros((n,), dtype=int)
        centers = np.zeros((1, X.shape[1]), dtype=float)
        n_clusters = 1

    # Update cluster importance with last round reward
    reward = float(history.get("state", {}).get("last_reward", 0.0) or 0.0)
    last_sel = set(history.get("selected", [])[-1] if history.get("selected") else [])
    if reward != 0.0 and last_sel:
        # Exponential decay to keep recent performance more important
        st["cluster_weight"] = st["cluster_weight"] * float(cluster_importance_decay)
        # Credit clusters of selected cids
        for i, c in enumerate(clients):
            if c.id in last_sel:
                cid_cluster = int(labels[i])
                st["cluster_weight"][cid_cluster] += reward / max(1, len(last_sel))

    # Selection: proportional by cluster weight, representative by distance to center + reliability
    weight = st["cluster_weight"]
    weight = weight / max(1e-12, float(weight.sum()))
    # Compute how many to draw from each cluster
    alloc = np.floor(weight * K).astype(int)
    # Distribute remaining slots greedily by largest fractional part
    remain = int(K - int(alloc.sum()))
    fracs = (weight * K) - alloc
    for idx in np.argsort(-fracs):
        if remain <= 0:
            break
        alloc[idx] += 1
        remain -= 1

    # Build per-cluster candidate lists scored by representativeness and reliability
    selected: List[int] = []
    per_client_score: Dict[int, float] = {}
    for cl in range(n_clusters):
        need = int(alloc[cl]) if cl < len(alloc) else 0
        if need <= 0:
            continue
        members = [(i, clients[i]) for i in range(n) if int(labels[i]) == cl]
        if not members:
            continue
        center = centers[cl] if cl < len(centers) else np.zeros((X.shape[1],), dtype=float)
        scores = []
        for i, c in members:
            # Representativeness: inverse distance to centroid
            dist = float(np.linalg.norm(X[i] - center) + 1e-12)
            rep = 1.0 / dist
            # Reliability: prefer higher channel quality and compute speed, and longer recency gap
            rel = float(c.channel_quality or 1.0) * float(c.compute_speed or 1.0)
            gap = float(recency(round_idx, c))
            rel = rel * (1.0 + 0.05 * (gap / (gap + 5.0)))
            score = rep_weight * rep + (1.0 - rep_weight) * rel
            scores.append((c.id, score))
        scores.sort(key=lambda t: t[1], reverse=True)
        picks = [cid for cid, _ in scores[:need]]
        selected.extend(picks)
        for cid, s in scores[:need]:
            per_client_score[cid] = float(s)

    # If due to rounding we selected less/more than K, adjust
    if len(selected) < K:
        # Fill remaining by highest scores across all clusters
        rest: List[Tuple[int, float]] = []
        for cl in range(n_clusters):
            members = [(i, clients[i]) for i in range(n) if int(labels[i]) == cl]
            center = centers[cl] if cl < len(centers) else np.zeros((X.shape[1],), dtype=float)
            for i, c in members:
                dist = float(np.linalg.norm(X[i] - center) + 1e-12)
                rep = 1.0 / dist
                rel = float(c.channel_quality or 1.0) * float(c.compute_speed or 1.0)
                rest.append((c.id, rep_weight * rep + (1.0 - rep_weight) * rel))
        rest.sort(key=lambda t: t[1], reverse=True)
        for cid, s in rest:
            if cid in selected:
                continue
            selected.append(cid)
            per_client_score[cid] = float(s)
            if len(selected) >= K:
                break
    elif len(selected) > K:
        # Trim to K by lowest scores
        selected = sorted(selected, key=lambda cid: per_client_score.get(cid, 0.0), reverse=True)[:K]

    scores_vec = [float(per_client_score.get(c.id, 0.0)) for c in clients]
    return selected, scores_vec, {STATE: st}


