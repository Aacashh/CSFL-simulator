from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import itertools
import torch

try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

from csfl_simulator.core.client import ClientInfo

# Global policy state
POLICY = None
TRAIN_MODE = False
LOG_PROBS: list[torch.Tensor] = []


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim=6, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze(-1)


def _features(c: ClientInfo):
    """Feature vector adapted from GRL-client-selection.ipynb ideas:
    [compute_speed, channel_quality, normalized_data_size, inv_loss, grad_norm, participation]
    inv_loss = 1/(1+loss) to keep within [0,1] when loss>0
    """
    inv_loss = 1.0 / (1.0 + float(c.last_loss or 0.0))
    data_norm = float(c.data_size or 0.0)
    return [
        float(c.compute_speed or 1.0),
        float(c.channel_quality or 1.0),
        data_norm,
        inv_loss,
        float(c.grad_norm or 0.0),
        float(c.participation_count or 0.0),
    ]


def _build_graph(clients: List[ClientInfo], device: str):
    x = torch.tensor([_features(c) for c in clients], dtype=torch.float, device=device)
    n = len(clients)
    if n <= 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        idx = [[i, j] for i in range(n) for j in range(n) if i != j]
        edge_index = torch.tensor(idx, dtype=torch.long, device=device).t().contiguous()
    return x, edge_index


def init_policy(in_dim: int, device: str = 'cpu'):
    global POLICY
    if not PYG_AVAILABLE:
        POLICY = None
        return
    POLICY = SimpleGCN(in_dim=in_dim).to(device)


def set_train_mode(flag: bool):
    global TRAIN_MODE
    TRAIN_MODE = flag


def zero_logs():
    global LOG_PROBS
    LOG_PROBS = []


def get_logs_and_clear():
    global LOG_PROBS
    logs = LOG_PROBS
    LOG_PROBS = []
    return logs


def save_checkpoint(path: str):
    if POLICY is not None:
        torch.save(POLICY.state_dict(), path)


def load_checkpoint(path: str, in_dim: int, device: str = 'cpu'):
    if not PYG_AVAILABLE:
        return False
    global POLICY
    if POLICY is None:
        POLICY = SimpleGCN(in_dim=in_dim).to(device)
    POLICY.load_state_dict(torch.load(path, map_location=device))
    return True


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    if not PYG_AVAILABLE or len(clients) == 0:
        ids = [c.id for c in clients]
        rng.shuffle(ids)
        return ids[:K], None, {}
    dev = device or 'cpu'
    x, edge_index = _build_graph(clients, dev)
    global POLICY
    if POLICY is None:
        init_policy(in_dim=x.shape[1], device=dev)
    scores = POLICY(x, edge_index)
    if TRAIN_MODE:
        probs = torch.softmax(scores, dim=0)
        # sample K without replacement (approx)
        n = len(clients)
        idxs = torch.multinomial(probs, num_samples=min(K, n), replacement=False)
        log_prob = torch.log(probs[idxs]).sum()
        LOG_PROBS.append(log_prob)
        sel = [clients[i.item()].id for i in idxs]
        return sel, probs.detach().cpu().tolist(), {}
    else:
        ranked = torch.argsort(scores, descending=True).tolist()
        sel = [clients[i].id for i in ranked[:K]]
        return sel, scores.detach().cpu().tolist(), {}
