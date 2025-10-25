from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Tuple, Optional
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from . import datasets as dset
from . import partition as part
from .models import get_model, maybe_load_pretrained
from .client import ClientInfo
from .aggregation import fedavg
from .metrics import eval_model
from .utils import new_run_dir, save_json, set_seed, autodetect_device
from ..selection.registry import MethodRegistry
from .system import init_system_state, simulate_round_env


@dataclass
class SimConfig:
    dataset: str = "MNIST"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    shards_per_client: int = 2
    total_clients: int = 10
    clients_per_round: int = 3
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    model: str = "CNN-MNIST"
    device: str = "auto"
    seed: int = 42
    fast_mode: bool = True
    pretrained: bool = False
    time_budget: Optional[float] = None
    dp_sigma: float = 0.0
    dp_epsilon_per_round: float = 0.0
    reward_weights: Dict[str, float] = field(default_factory=lambda: {"acc": 0.6, "time": 0.2, "fair": 0.1, "dp": 0.1})


class FLSimulator:
    def __init__(self, config: SimConfig):
        self.cfg = config
        set_seed(self.cfg.seed)
        self.device = autodetect_device(True) if self.cfg.device == "auto" else self.cfg.device
        self.run_dir, self.run_id = new_run_dir("sim")
        self.registry = MethodRegistry()
        self.registry.load_presets()
        self.history: Dict[str, Any] = {"state": {}, "selected": []}
        # Ensure these attributes exist even before setup() to avoid attribute errors in UI
        self.clients: List[ClientInfo] = []
        self.client_loaders: Dict[int, Any] = {}

    def setup(self):
        # Datasets
        train_ds, test_ds = dset.get_full_data(self.cfg.dataset)
        # Partition
        if self.cfg.partition == "iid":
            mapping = part.iid_partition([train_ds[i][1] for i in range(len(train_ds))], self.cfg.total_clients)
        elif self.cfg.partition == "dirichlet":
            mapping = part.dirichlet_partition([train_ds[i][1] for i in range(len(train_ds))], self.cfg.total_clients, self.cfg.dirichlet_alpha)
        else:
            mapping = part.label_shard_partition([train_ds[i][1] for i in range(len(train_ds))], self.cfg.total_clients, self.cfg.shards_per_client)
        # Model
        # Infer num_classes from test_ds targets
        try:
            num_classes = len(getattr(test_ds, 'classes', [])) or int(max(test_ds.targets) + 1)
        except Exception:
            num_classes = 10
        self.model = get_model(self.cfg.model, self.cfg.dataset, num_classes, device=self.device, pretrained=self.cfg.pretrained)
        maybe_load_pretrained(self.model, self.cfg.model, self.cfg.dataset)
        self.criterion = nn.CrossEntropyLoss()
        self.train_ds = train_ds
        self.test_loader = dset.make_loader(test_ds, batch_size=128, shuffle=False)
        # Build client loaders and infos
        self.client_loaders = {}
        self.clients: List[ClientInfo] = []
        for cid in range(self.cfg.total_clients):
            idxs = mapping[cid]
            self.client_loaders[cid] = dset.make_loaders_from_indices(train_ds, idxs, batch_size=self.cfg.batch_size)
            # Compute label histogram for this client if possible
            # Try common dataset attributes first
            hist = None
            try:
                import numpy as _np
                # Many torchvision datasets have .targets
                targets = getattr(train_ds, 'targets', None)
                if targets is None:
                    targets = getattr(train_ds, 'labels', None)
                if targets is not None:
                    # targets may be a list or tensor
                    ys = [int(targets[i]) for i in idxs]
                else:
                    # Fallback: index into dataset
                    ys = [int(train_ds[i][1]) for i in idxs]
                # Infer num_classes from earlier computation or ys
                try:
                    L = int(max(getattr(train_ds, 'classes', []) and len(getattr(train_ds, 'classes', [])) or (max(ys) + 1)))
                except Exception:
                    L = int(max(ys) + 1) if ys else 0
                if L > 0:
                    bc = _np.bincount(ys, minlength=L).astype(float)
                    hist = {int(i): float(v) for i, v in enumerate(bc) if v > 0}
            except Exception:
                hist = None
            self.clients.append(ClientInfo(id=cid, data_size=len(idxs), label_histogram=hist))
        init_system_state(self.clients, {})
        # Save config
        save_json({"config": asdict(self.cfg), "device": self.device}, self.run_dir / "config.json")

    def _local_train(self, cid: int) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        import copy
        loader = self.client_loaders[cid]
        # Deep-copy the current global model to preserve architecture/num_classes
        model = copy.deepcopy(self.model).to(self.device)
        model.train()
        opt = optim.SGD(model.parameters(), lr=self.cfg.lr)
        last_loss = 0.0
        for e in range(self.cfg.local_epochs):
            for bi, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = model(x)
                loss = self.criterion(out, y)
                loss.backward()
                opt.step()
                last_loss = float(loss.item())
                if self.cfg.fast_mode and bi > 1:
                    break
        # grad norm proxy via final layer
        grad_norm = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.norm().item())
        return model.state_dict(), len(loader.dataset), last_loss, grad_norm

    def run(self, method_key: str = "heuristic.random", on_progress=None, is_cancelled=None) -> Dict[str, Any]:
        self.setup()
        # Baseline evaluation before rounds
        metrics = []
        base = eval_model(self.model, self.test_loader, self.device)
        base["round"] = -1
        metrics.append(base)
        # Composite reward initialization
        prev_composite = 0.0
        prev_acc = base["accuracy"]
        stopped_early = False
        for rnd in range(self.cfg.rounds):
            if is_cancelled and is_cancelled():
                stopped_early = True
                break
            simulate_round_env(self.clients, {}, rnd)
            # Selection
            # Pass time_budget to selector and preset params via registry
            ids, scores, state = self.registry.invoke(
                method_key,
                rnd,
                self.cfg.clients_per_round,
                self.clients,
                self.history,
                random,
                self.cfg.time_budget,
                self.device,
            )
            self.history["selected"].append(ids)
            # Update recency info on clients
            for cid in ids:
                try:
                    self.clients[cid].last_selected_round = rnd
                except Exception:
                    pass
            if state:
                self.history["state"].update(state)
            # Local training
            updates, weights = [], []
            for cid in ids:
                sd, n, loss, gnorm = self._local_train(cid)
                self.clients[cid].last_loss = loss
                self.clients[cid].grad_norm = gnorm
                self.clients[cid].participation_count += 1
                # Apply DP accounting per client
                if self.cfg.dp_epsilon_per_round and self.cfg.dp_epsilon_per_round > 0:
                    try:
                        from . import dp as _dp
                        _dp.consume_epsilon(self.clients[cid], float(self.cfg.dp_epsilon_per_round))
                    except Exception:
                        pass
                updates.append(sd)
                weights.append(n)
            # Aggregate
            if updates:
                # Optional DP noise before aggregation
                if self.cfg.dp_sigma and self.cfg.dp_sigma > 0:
                    try:
                        from .dp import apply_gaussian_noise as _dp_noise
                        noisy_updates = []
                        for sd in updates:
                            nsd = {k: _dp_noise(v, float(self.cfg.dp_sigma)) if hasattr(v, 'shape') else v for k, v in sd.items()}
                            noisy_updates.append(nsd)
                        updates = noisy_updates
                    except Exception:
                        pass
                new_sd = fedavg(updates, weights)
                self.model.load_state_dict(new_sd)
            # Evaluate
            m = eval_model(self.model, self.test_loader, self.device)
            m["round"] = rnd
            # Measure round time proxy (number of local batches approximated by data size/speed)
            round_time = sum(float(getattr(self.clients[cid], 'estimated_duration', 0.0) or 0.0) for cid in ids)
            # Fairness: participation variance
            try:
                import numpy as _np
                parts = _np.array([float(c.participation_count or 0.0) for c in self.clients], dtype=float)
                fairness_var = float(_np.var(parts))
            except Exception:
                fairness_var = 0.0
            # DP usage avg
            dp_used_avg = float(sum(float(getattr(c, 'dp_epsilon_used', 0.0) or 0.0) for c in self.clients) / max(1, len(self.clients)))
            # Composite metric per config weights
            w = self.cfg.reward_weights or {"acc": 1.0}
            acc_score = float(m.get("accuracy", 0.0))
            time_score = 1.0 - (round_time / (round_time + 1.0))
            fair_score = 1.0 - (fairness_var / (fairness_var + 1.0))
            dp_score = 1.0 - (dp_used_avg / (dp_used_avg + 1.0))
            composite = (
                float(w.get("acc", 0.6)) * acc_score +
                float(w.get("time", 0.2)) * time_score +
                float(w.get("fair", 0.1)) * fair_score +
                float(w.get("dp", 0.1)) * dp_score
            )
            # Reward = composite improvement
            reward = composite - prev_composite
            self.history["state"]["last_reward"] = reward
            prev_acc = m["accuracy"]
            prev_composite = composite
            # Log metrics
            m["round_time"] = round_time
            m["fairness_var"] = fairness_var
            m["dp_used_avg"] = dp_used_avg
            m["composite"] = composite
            prev_acc = m["accuracy"]
            metrics.append(m)
            if on_progress:
                try:
                    on_progress(rnd, {"accuracy": m["accuracy"], "metrics": m, "selected": ids, "reward": reward, "composite": composite})
                except Exception:
                    pass
        # Save metrics
        save_json({"metrics": metrics}, self.run_dir / "metrics.json")
        return {"run_id": self.run_id, "metrics": metrics, "config": asdict(self.cfg), "device": self.device, "stopped_early": stopped_early}
