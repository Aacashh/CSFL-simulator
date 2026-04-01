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
from .utils import new_run_dir, save_json, set_seed, autodetect_device, cleanup_memory, check_memory_critical
from ..selection.registry import MethodRegistry
from .system import init_system_state, simulate_round_env
from .parallel import create_trainer


@dataclass
class SimConfig:
    dataset: str = "MNIST"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    shards_per_client: int = 2
    # Data quantity heterogeneity
    size_distribution: str = "uniform"  # "uniform" | "lognormal" | "power_law"
    size_lognormal_mu: float = 0.0
    size_lognormal_sigma: float = 0.5
    size_powerlaw_alpha: float = 1.5
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
    energy_budget: Optional[float] = None
    bytes_budget: Optional[float] = None
    dp_sigma: float = 0.0
    dp_epsilon_per_round: float = 0.0
    dp_clip_norm: float = 0.0
    reward_weights: Dict[str, float] = field(default_factory=lambda: {"acc": 0.6, "time": 0.2, "fair": 0.1, "dp": 0.1})
    # Performance/diagnostics knobs
    track_grad_norm: bool = False
    parallel_clients: int = 0  # 0 = off (sequential)
    name: Optional[str] = None  # Named run (used as folder name under artifacts/runs/)
    # Paradigm selection: "fl" (Federated Learning) or "fd" (Federated Distillation)
    paradigm: str = "fl"
    # --- Federated Distillation (FD) settings ---
    # Public dataset for logit exchange
    public_dataset: str = "same"         # "same" = sample from test split; or dataset name e.g. "STL-10"
    public_dataset_size: int = 2000      # Number of public samples
    # Distillation hyperparameters
    distillation_epochs: int = 2         # S distillation steps per round (paper: 2)
    distillation_batch_size: int = 500   # Batch size for distillation (paper: 500)
    temperature: float = 1.0             # Softmax temperature for KL divergence
    distillation_lr: float = 0.001       # Learning rate for distillation (paper: Adam 0.001)
    # Dynamic training steps (FedTSKD)
    dynamic_steps: bool = True           # Decrease local steps K_r over rounds
    dynamic_steps_base: int = 5          # Initial multiplier (paper: 5)
    dynamic_steps_period: int = 25       # Rounds per step decrease (paper: 25)
    # Model heterogeneity
    model_heterogeneous: bool = False    # Different architectures per client
    model_pool: str = ""                 # Comma-separated model names, e.g. "FD-CNN1,FD-CNN2,FD-CNN3"
    # Communication channel (mMIMO)
    channel_noise: bool = False          # Enable mMIMO channel noise
    n_bs_antennas: int = 64             # N_BS base station antennas (paper: 64)
    n_device_antennas: int = 1          # N_D device antennas (paper: 1)
    ul_snr_db: float = -8.0            # Uplink SNR in dB (paper: -8)
    dl_snr_db: float = -20.0           # Downlink SNR in dB (paper: -20)
    quantization_bits: int = 8          # Logit quantization bits (paper: 8)
    # Group-based FD (FedTSKD-G)
    group_based: bool = False            # Enable channel-aware group splitting
    channel_threshold: float = 0.5       # Threshold for good/bad channel groups
    # FD optimizer
    fd_optimizer: str = "adam"           # Optimizer for FD local training (paper: adam)
    # Performance tuning
    eval_every: int = 5                  # Evaluate every N rounds (0 = every round)
    use_amp: bool = False                # Mixed precision (AMP) — faster on Turing+ GPUs


class FLSimulator:
    def __init__(self, config: SimConfig):
        self.cfg = config
        set_seed(self.cfg.seed, deterministic=True)
        self.device = autodetect_device(True) if self.cfg.device == "auto" else self.cfg.device
        self.run_dir, self.run_id = new_run_dir(self.cfg.name or "sim")
        self.registry = MethodRegistry()
        self.registry.load_presets()
        self.history: Dict[str, Any] = {"state": {}, "selected": []}
        # Ensure these attributes exist even before setup() to avoid attribute errors in UI
        self.clients: List[ClientInfo] = []
        self.client_loaders: Dict[int, Any] = {}
        self._scratch_model = None
        self._parallel_trainer = None
        self._partition_mapping: Dict[int, List[int]] = {}  # Store for emergency cleanup

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
        # Apply data size skew (quantity heterogeneity) if requested
        if (self.cfg.size_distribution or "uniform").lower() != "uniform":
            try:
                total_size = len(train_ds)
                mapping = part.apply_size_distribution(
                    mapping,
                    total_size=total_size,
                    size_distribution=self.cfg.size_distribution,
                    mu=float(self.cfg.size_lognormal_mu),
                    sigma=float(self.cfg.size_lognormal_sigma),
                    alpha=float(self.cfg.size_powerlaw_alpha),
                )
            except Exception as e:
                # Non-fatal: fall back to original mapping
                print(f"Warning: Failed to apply size distribution '{self.cfg.size_distribution}': {e}")
        # Model
        # Infer num_classes from test_ds targets
        try:
            num_classes = len(getattr(test_ds, 'classes', [])) or int(max(test_ds.targets) + 1)
        except Exception:
            num_classes = 10
        self.num_classes = num_classes
        self.model = get_model(self.cfg.model, self.cfg.dataset, num_classes, device=self.device, pretrained=self.cfg.pretrained)
        maybe_load_pretrained(self.model, self.cfg.model, self.cfg.dataset)
        self.criterion = nn.CrossEntropyLoss()
        self.train_ds = train_ds
        self.test_loader = dset.make_loader(test_ds, batch_size=128, shuffle=False)
        # Adjust model to actual data shape (channels, image size) to avoid FC and channel mismatches
        try:
            xb0, yb0 = next(iter(self.test_loader))
            c0, h0, w0 = int(xb0.shape[1]), int(xb0.shape[2]), int(xb0.shape[3])
            name_l = self.cfg.model.lower()
            if name_l in ("cnn-mnist", "cnn_mnist"):
                from .models import CNNMnist
                self.model = CNNMnist(num_classes=num_classes, in_channels=c0, image_size=h0).to(self.device)
            elif name_l in ("cnn-mnist (fedavg)", "cnn_mnist_fedavg", "cnn-mnist-fedavg"):
                from .models import CNNMnistFedAvg
                self.model = CNNMnistFedAvg(num_classes=num_classes, in_channels=c0, image_size=h0).to(self.device)
            elif name_l in ("lightcnn", "light-cifar"):
                from .models import LightCIFAR
                self.model = LightCIFAR(num_classes=num_classes, in_channels=c0, image_size=h0).to(self.device)
            elif name_l == "resnet18" and c0 != 3:
                try:
                    # Rebuild conv1 for grayscale inputs
                    self.model.conv1 = nn.Conv2d(c0, 64, kernel_size=7, stride=2, padding=3, bias=False).to(self.device)
                except Exception:
                    pass
            # Generic safety: ensure the very first Conv2d matches input channels
            try:
                def _first_conv_and_parent(m: nn.Module):
                    parent = None
                    attr_name = None
                    for n, mod in m.named_modules():
                        if isinstance(mod, nn.Conv2d):
                            # find parent via attribute walk
                            parts = n.split(".")
                            p = m
                            for part in parts[:-1]:
                                p = getattr(p, part)
                            return p, parts[-1], mod
                    return None, None, None
                p, aname, conv = _first_conv_and_parent(self.model)
                if conv is not None and int(conv.in_channels) != c0:
                    new_conv = nn.Conv2d(c0, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                                         padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                                         bias=(conv.bias is not None), padding_mode=conv.padding_mode).to(self.device)
                    # Initialize new weights by tiling/averaging to preserve scale
                    with torch.no_grad():
                        w_old = conv.weight.data
                        in_old = int(w_old.shape[1])
                        if c0 == in_old:
                            new_conv.weight.copy_(w_old)
                        elif c0 > in_old:
                            reps = (c0 + in_old - 1) // in_old
                            w_rep = w_old.repeat(1, reps, 1, 1)[:, :c0]
                            # scale to keep variance roughly similar
                            w_rep = w_rep * (in_old / float(c0))
                            new_conv.weight.copy_(w_rep)
                        else:  # c0 < in_old
                            # average first in_old -> c0 groups
                            step = in_old / float(c0)
                            idxs = [int(i * step) for i in range(c0)]
                            w_sel = w_old[:, idxs, :, :].clone()
                            new_conv.weight.copy_(w_sel)
                        if new_conv.bias is not None and conv.bias is not None:
                            new_conv.bias.copy_(conv.bias.data)
                    if p is not None and aname is not None:
                        setattr(p, aname, new_conv)
            except Exception:
                pass
        except Exception:
            pass
        # Preflight: validate model <-> data compatibility early (channels, shapes, class count)
        try:
            xb, yb = next(iter(self.test_loader))
            xb = xb.to(self.device)
            with torch.no_grad():
                out = self.model(xb)
            # If class count mismatches, adjust final layer when possible
            if out.dim() == 2 and out.shape[1] != num_classes:
                try:
                    if hasattr(self.model, 'fc2') and isinstance(self.model.fc2, nn.Linear):
                        self.model.fc2 = nn.Linear(self.model.fc2.in_features, num_classes).to(self.device)
                    elif hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
                        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes).to(self.device)
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"Model/data preflight failed: {e}. This likely indicates a mismatch between dataset channels/size and the selected model.") from e
        # Build client loaders and infos
        self.client_loaders = {}
        self.clients: List[ClientInfo] = []
        self._partition_mapping = mapping  # Store for emergency cleanup
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
                    classes_attr = getattr(train_ds, 'classes', None)
                    L = len(classes_attr) if classes_attr else (int(max(ys) + 1) if ys else 0)
                except Exception:
                    L = int(max(ys) + 1) if ys else 0
                if L > 0:
                    bc = _np.bincount(ys, minlength=L).astype(float)
                    hist = {int(i): float(v) for i, v in enumerate(bc) if v > 0}
            except Exception:
                hist = None
            self.clients.append(ClientInfo(id=cid, data_size=len(idxs), label_histogram=hist))
        init_system_state(self.clients, {})
        
        # Initialize parallel trainer if parallelization is enabled
        if self.cfg.parallel_clients != 0:
            self._parallel_trainer = create_trainer(
                model=self.model,
                device=self.device,
                criterion=self.criterion,
                lr=self.cfg.lr,
                parallel_clients=self.cfg.parallel_clients,
                track_grad_norm=self.cfg.track_grad_norm
            )
        
        # Save config
        save_json({"config": asdict(self.cfg), "device": self.device}, self.run_dir / "config.json")

        # Profile model for efficiency metrics
        self.model_macs_per_sample = 0.0
        self.model_params_count = 0.0
        try:
            import thop
            # Create a dummy input with correct shape
            xb, _ = next(iter(self.test_loader))
            xb = xb[:1].to(self.device)  # use batch size 1 for per-sample MACs
            macs, params = thop.profile(self.model, inputs=(xb,), verbose=False)
            self.model_macs_per_sample = macs
            self.model_params_count = params
        except ImportError:
            print("Warning: 'thop' not installed. Research efficiency metrics will be 0.")
        except Exception as e:
            print(f"Warning: Model profiling failed: {e}")

        # Save initial model state for fair multi-method comparisons
        with torch.no_grad():
            self._initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

    def _local_train(self, cid: int) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        # Lazily recreate loader if it was deleted during emergency cleanup
        if cid not in self.client_loaders or self.client_loaders[cid] is None:
            if cid in self._partition_mapping:
                from . import datasets as dset
                idxs = self._partition_mapping[cid]
                self.client_loaders[cid] = dset.make_loaders_from_indices(
                    self.train_ds, idxs, batch_size=self.cfg.batch_size, num_workers=0
                )
        
        loader = self.client_loaders[cid]
        # Reuse a scratch model to avoid per-client deepcopy cost
        if self._scratch_model is None:
            import copy as _copy
            self._scratch_model = _copy.deepcopy(self.model).to(self.device)
        model = self._scratch_model
        # Reset weights to current global state
        model.load_state_dict(self.model.state_dict())
        model.train()
        opt = optim.SGD(model.parameters(), lr=self.cfg.lr)
        last_loss = 0.0
        last_grad_norm = 0.0
        for e in range(self.cfg.local_epochs):
            for bi, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = model(x)
                loss = self.criterion(out, y)
                loss.backward()
                # Optional DP gradient clipping
                if self.cfg.dp_clip_norm and self.cfg.dp_clip_norm > 0:
                    try:
                        from .dp import clip_gradients as _dp_clip
                        _dp_clip(model, float(self.cfg.dp_clip_norm))
                    except Exception:
                        pass
                if self.cfg.track_grad_norm:
                    try:
                        total_norm_sq = sum(
                            p.grad.data.norm(2).item() ** 2
                            for p in model.parameters() if p.grad is not None
                        )
                        last_grad_norm = total_norm_sq ** 0.5
                    except Exception:
                        pass
                opt.step()
                last_loss = float(loss.item())
                if self.cfg.fast_mode and bi > 1:
                    break
        # Export weights safely (clone tensors)
        with torch.no_grad():
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return sd, len(loader.dataset), last_loss, (last_grad_norm if self.cfg.track_grad_norm else 0.0)

    @staticmethod
    def _selection_ops_estimate(key: str, N: int) -> float:
        """Rough complexity estimate for a selection method given N clients."""
        k = (key or "").lower()
        if "random" in k or "heuristic" in k:
            p = 1.0
        elif "bandit" in k:
            p = 1.5
        elif ("ml." in k) or ("gnn" in k) or ("gat" in k) or ("gcn" in k) or ("rl" in k):
            p = 2.0
        else:
            p = 1.0
        try:
            return float(N ** p)
        except Exception:
            return float(N)

    def run(self, method_key: str = "heuristic.random", on_progress=None, is_cancelled=None) -> Dict[str, Any]:
        # Only setup if not already done (allows reuse for fair comparisons)
        if not hasattr(self, 'train_ds') or self.train_ds is None:
            self.setup()
        # Reset state for this run: restore initial model, clear history & client stats
        if hasattr(self, '_initial_state_dict'):
            self.model.load_state_dict(self._initial_state_dict)
        self.history = {"state": {}, "selected": []}
        for c in self.clients:
            c.last_loss = 0.0
            c.grad_norm = 0.0
            c.participation_count = 0
            c.last_selected_round = -1
        set_seed(self.cfg.seed, deterministic=True)
        self._scratch_model = None
        # Re-init parallel trainer for clean state
        if self.cfg.parallel_clients != 0 and self._parallel_trainer is not None:
            from .parallel import create_trainer
            self._parallel_trainer = create_trainer(
                model=self.model, device=self.device, criterion=self.criterion,
                lr=self.cfg.lr, parallel_clients=self.cfg.parallel_clients,
                track_grad_norm=self.cfg.track_grad_norm
            )
        # Baseline evaluation before rounds
        metrics = []
        base = eval_model(self.model, self.test_loader, self.device)
        base["round"] = -1
        # Initialize cumulative efficiency metrics in base
        base["cum_flops"] = 0.0
        base["cum_comm"] = 0.0
        base["cum_time"] = 0.0
        base["cum_tflops"] = 0.0
        metrics.append(base)
        # Composite reward initialization
        prev_composite = 0.0
        prev_acc = base["accuracy"]
        
        cumulative_flops = 0.0
        cumulative_total_flops = 0.0
        cumulative_comm_mb = 0.0
        cumulative_time = 0.0

        stopped_early = False
        for rnd in range(self.cfg.rounds):
            if is_cancelled and is_cancelled():
                stopped_early = True
                break
            simulate_round_env(self.clients, {}, rnd)
            # Selection
            # Pass time_budget to selector and preset params via registry
            _t0 = time.perf_counter()
            ids, scores, state = self.registry.invoke(
                    method_key,
                    rnd,
                    self.cfg.clients_per_round,
                    self.clients,
                    self.history,
                    random,
                    self.cfg.time_budget,
                    self.device,
                    # Additional budgets (selectors may ignore if not supported)
                    energy_budget=self.cfg.energy_budget,
                    bytes_budget=self.cfg.bytes_budget,
                )
            selection_time = float(time.perf_counter() - _t0)
            self.history["selected"].append(ids)
            # Update recency info on clients
            for cid in ids:
                try:
                    self.clients[cid].last_selected_round = rnd
                except Exception:
                    pass
            if state:
                self.history["state"].update(state)
            # Local training - use parallel trainer if enabled, otherwise sequential
            updates, weights = [], []
            
            # Ensure loaders exist for selected clients (in case emergency cleanup removed them)
            for cid in ids:
                if cid not in self.client_loaders or self.client_loaders[cid] is None:
                    if cid in self._partition_mapping:
                        idxs = self._partition_mapping[cid]
                        self.client_loaders[cid] = dset.make_loaders_from_indices(
                            self.train_ds, idxs, batch_size=self.cfg.batch_size, num_workers=0
                        )
            
            if self._parallel_trainer is not None:
                # Parallel training path
                results = self._parallel_trainer.train_clients_parallel(
                    client_ids=ids,
                    client_loaders=self.client_loaders,
                    local_epochs=self.cfg.local_epochs,
                    fast_mode=self.cfg.fast_mode,
                    seed_offset=rnd
                )
                # Process results
                for (sd, n, loss, gnorm), cid in zip(results, ids):
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
            else:
                # Sequential training path (backward compatible)
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
                # Keep aggregation on GPU for better performance
                use_gpu_aggregation = self.device.startswith('cuda')
                new_sd = fedavg(updates, weights, keep_on_device=use_gpu_aggregation)
                self.model.load_state_dict(new_sd)
                
                # Update parallel trainer's global model if using parallel mode
                if self._parallel_trainer is not None:
                    self._parallel_trainer.update_global_model(new_sd)
            # Evaluate
            m = eval_model(self.model, self.test_loader, self.device)
            m["round"] = rnd
            # Measure round time (astute): selection_time + max(selected_client_durations)
            sel_durations = [float(getattr(self.clients[cid], 'estimated_duration', 0.0) or 0.0) for cid in ids]
            compute_time = max(sel_durations) if sel_durations else 0.0
            round_time = float(selection_time) + float(compute_time)
            round_energy = sum(float(getattr(self.clients[cid], 'estimated_energy', 0.0) or 0.0) for cid in ids)
            round_bytes = sum(float(getattr(self.clients[cid], 'estimated_bytes', 0.0) or 0.0) for cid in ids)
            # Keep cumulative wall-clock
            prev_wc = float(metrics[-1].get('wall_clock', 0.0)) if metrics else 0.0
            wall_clock = prev_wc + round_time
            # Fairness: participation variance and Gini
            try:
                import numpy as _np
                parts = _np.array([float(c.participation_count or 0.0) for c in self.clients], dtype=float)
                fairness_var = float(_np.var(parts))
                s = parts.sum()
                if s > 0 and len(parts) > 0:
                    diffs = _np.abs(parts[:, None] - parts[None, :])
                    fairness_gini = float(diffs.sum() / (2.0 * len(parts) * s))
                else:
                    fairness_gini = 0.0
            except Exception:
                fairness_var = 0.0
                fairness_gini = 0.0
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

            # Calculate efficiency metrics (FLOPs and Comm)
            # FLOPs approx: 3 * MACs * local_epochs * data_size (for all participants)
            round_flops = 0.0
            for cid in ids:
                c_data_size = self.clients[cid].data_size
                round_flops += 3.0 * self.model_macs_per_sample * self.cfg.local_epochs * c_data_size
            training_tflops = round_flops / 1e12
            total_round_flops = round_flops  # training FLOPs only; selection cost tracked via selection_time
            selection_ops = self._selection_ops_estimate(method_key, len(self.clients))
            total_round_ops = round_flops + selection_ops  # kept for research/debug; mixed units by design
            
            # Communication: 2 * K * ModelSize (MB)
            # ModelSize in MB (assuming float32 = 4 bytes)
            model_size_mb = (self.model_params_count * 4) / (1024 * 1024)
            round_comm_mb = 2.0 * len(ids) * model_size_mb
            
            cumulative_flops += round_flops
            cumulative_total_flops += total_round_flops
            cumulative_comm_mb += round_comm_mb
            cumulative_time += round_time

            # Log metrics
            m["selection_time"] = float(selection_time)
            m["compute_time"] = float(compute_time)
            m["round_time"] = round_time
            m["round_energy"] = round_energy
            m["round_bytes"] = round_bytes
            m["wall_clock"] = wall_clock
            m["cum_flops"] = cumulative_flops
            # Backward-compat + astute naming
            m["cum_tflops"] = cumulative_flops / 1e12
            m["training_tflops"] = training_tflops
            m["cum_training_tflops"] = cumulative_flops / 1e12
            m["total_tflops"] = total_round_flops / 1e12
            m["cum_total_tflops"] = cumulative_total_flops / 1e12
            m["selection_ops_estimate"] = float(selection_ops)
            m["total_round_ops"] = float(total_round_ops)
            m["cum_comm"] = cumulative_comm_mb
            m["cum_time"] = cumulative_time
            m["clients_per_hour"] = (len(ids) * 3600.0 / round_time) if round_time > 0 else 0.0
            m["fairness_var"] = fairness_var
            m["dp_used_avg"] = dp_used_avg
            m["composite"] = composite
            m["fairness_gini"] = fairness_gini

            # --- New DELTA-showcase metrics ---
            # 1. Label Coverage Ratio: fraction of classes covered by selected clients
            try:
                covered_labels = set()
                for cid in ids:
                    h = getattr(self.clients[cid], 'label_histogram', None)
                    if isinstance(h, dict):
                        for lbl, cnt in h.items():
                            if cnt and cnt > 0:
                                covered_labels.add(int(lbl))
                m["label_coverage_ratio"] = len(covered_labels) / max(1, self.num_classes)
            except Exception:
                m["label_coverage_ratio"] = 0.0

            # 2. Selection Overhead %: selection_time as % of round_time
            m["selection_overhead_pct"] = (float(selection_time) / round_time * 100.0) if round_time > 0 else 0.0

            # 3. Accuracy Stability: 1 - CV of accuracy over last 5 rounds
            try:
                recent_accs = [float(row.get("accuracy", 0.0) or 0.0) for row in metrics if int(row.get("round", -1)) >= 0]
                recent_accs.append(float(m.get("accuracy", 0.0)))
                window = recent_accs[-5:]
                if len(window) >= 2:
                    _mean = sum(window) / len(window)
                    _std = (sum((x - _mean) ** 2 for x in window) / len(window)) ** 0.5
                    m["acc_stability"] = max(0.0, min(1.0, 1.0 - (_std / (_mean + 1e-8))))
                else:
                    m["acc_stability"] = 1.0
            except Exception:
                m["acc_stability"] = 1.0

            # 4. Client Utilization Entropy: normalized Shannon entropy of participation counts
            try:
                import numpy as _np2
                parts2 = _np2.array([float(c.participation_count or 0.0) for c in self.clients], dtype=float)
                total_p = parts2.sum()
                n_clients = len(self.clients)
                if total_p > 0 and n_clients > 1:
                    probs = parts2 / total_p
                    probs = probs[probs > 0]
                    entropy = -float(_np2.sum(probs * _np2.log(probs)))
                    max_entropy = float(_np2.log(n_clients))
                    m["utilization_entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0
                else:
                    m["utilization_entropy"] = 0.0
            except Exception:
                m["utilization_entropy"] = 0.0

            # 5. Convergence Efficiency: accuracy gain per cumulative TFLOP
            try:
                base_accuracy = float(metrics[0].get("accuracy", 0.0)) if metrics else 0.0
                curr_acc = float(m.get("accuracy", 0.0))
                ctf = cumulative_flops / 1e12
                m["convergence_efficiency"] = (curr_acc - base_accuracy) / max(ctf, 1e-9)
            except Exception:
                m["convergence_efficiency"] = 0.0

            prev_acc = m["accuracy"]
            metrics.append(m)
            if on_progress:
                try:
                    on_progress(rnd, {"accuracy": m["accuracy"], "metrics": m, "selected": ids, "reward": reward, "composite": composite})
                except Exception:
                    pass
            
            # Memory management: aggressive cleanup to prevent system hangs
            # CRITICAL: Clean up every single round to prevent accumulation
            cleanup_memory(force_cuda_empty=False, verbose=False)
            
            # Check memory status every 3 rounds
            if rnd % 3 == 0:
                is_critical, msg = check_memory_critical(threshold_percent=70.0)
                if is_critical:
                    print(f"WARNING at round {rnd}: {msg}")
                    # EMERGENCY: Aggressive cleanup when critical
                    self._emergency_cleanup()
                    cleanup_memory(force_cuda_empty=True, verbose=True)
            
            # Heavy cleanup every 5 rounds
            if rnd % 5 == 0:
                cleanup_memory(force_cuda_empty=True, verbose=False)
        # Final cleanup after simulation completes
        cleanup_memory(force_cuda_empty=True, verbose=False)
        
        # Convergence and time-accuracy efficiency summaries
        try:
            # Build time/accuracy series from base + rounds
            base_acc = float(metrics[0].get("accuracy", 0.0) or 0.0) if metrics else 0.0
            times: List[float] = [0.0]
            accs: List[float] = [base_acc]
            for row in metrics:
                try:
                    r = int(row.get("round", -1))
                except Exception:
                    r = -1
                if r >= 0:
                    try:
                        times.append(float(row.get("wall_clock", 0.0) or 0.0))
                    except Exception:
                        times.append(0.0)
                    try:
                        accs.append(float(row.get("accuracy", 0.0) or 0.0))
                    except Exception:
                        accs.append(0.0)
            def _first_time_to(target: float) -> float:
                # Return earliest time where acc >= target (linear interpolate across segments)
                if not times or not accs or len(times) != len(accs):
                    return float("nan")
                for i in range(1, len(times)):
                    a0, a1 = accs[i-1], accs[i]
                    if a1 >= target:
                        t0, t1 = times[i-1], times[i]
                        # linear interpolation safeguard
                        da = max(1e-12, (a1 - a0))
                        frac = min(1.0, max(0.0, (target - a0) / da))
                        return float(t0 + frac * (t1 - t0))
                return float("nan")
            def _trapz(xs: List[float], ys: List[float]) -> float:
                area = 0.0
                for i in range(1, len(xs)):
                    dx = float(xs[i] - xs[i-1])
                    area += dx * (float(ys[i]) + float(ys[i-1])) * 0.5
                return area
            final_acc = float(accs[-1]) if accs else base_acc
            total_time = float(times[-1]) if times else 0.0
            # Targets as fraction of improvement above baseline
            improv = max(0.0, final_acc - base_acc)
            targets = {
                "50": base_acc + 0.50 * improv,
                "80": base_acc + 0.80 * improv,
                "90": base_acc + 0.90 * improv,
            }
            t50 = _first_time_to(targets["50"]) if improv > 1e-12 else float("nan")
            t80 = _first_time_to(targets["80"]) if improv > 1e-12 else float("nan")
            t90 = _first_time_to(targets["90"]) if improv > 1e-12 else float("nan")
            # Rounds to reach targets
            def _rounds_to(target: float) -> float:
                best = float("nan")
                for row in metrics:
                    try:
                        r = int(row.get("round", -1))
                        a = float(row.get("accuracy", 0.0) or 0.0)
                        if r >= 0 and a >= target:
                            best = float(r + 1)  # +1 to be 1-based count of rounds executed
                            break
                    except Exception:
                        continue
                return best
            r50 = _rounds_to(targets["50"]) if improv > 1e-12 else float("nan")
            r80 = _rounds_to(targets["80"]) if improv > 1e-12 else float("nan")
            r90 = _rounds_to(targets["90"]) if improv > 1e-12 else float("nan")
            # AUC of accuracy over wall-clock; normalized by ideal rectangular area at final acc
            auc = _trapz(times, accs) if len(times) >= 2 else 0.0
            ideal_area = final_acc * total_time
            auc_norm = (auc / ideal_area) if ideal_area > 1e-12 else 0.0
            # Accuracy gain per hour
            acc_gain = max(0.0, final_acc - base_acc)
            acc_gain_per_hour = (3600.0 * acc_gain / total_time) if total_time > 1e-9 else 0.0
            # Attach to final row (for ranking/exports)
            if metrics:
                metrics[-1]["time_to_50pct_final"] = float(t50)
                metrics[-1]["time_to_80pct_final"] = float(t80)
                metrics[-1]["time_to_90pct_final"] = float(t90)
                metrics[-1]["rounds_to_50pct_final"] = float(r50)
                metrics[-1]["rounds_to_80pct_final"] = float(r80)
                metrics[-1]["rounds_to_90pct_final"] = float(r90)
                metrics[-1]["auc_acc_time"] = float(auc)
                metrics[-1]["auc_acc_time_norm"] = float(auc_norm)
                metrics[-1]["acc_gain_per_hour"] = float(acc_gain_per_hour)
        except Exception:
            # Non-fatal; keep results
            pass
        
        # Save metrics
        save_json({"metrics": metrics}, self.run_dir / "metrics.json")
        # Prepare participation counts snapshot
        try:
            participation_counts = [int(c.participation_count or 0) for c in self.clients]
        except Exception:
            participation_counts = []
        return {
            "run_id": self.run_id,
            "metrics": metrics,
            "config": asdict(self.cfg),
            "device": self.device,
            "stopped_early": stopped_early,
            "method": method_key,
            "history": {"selected": list(self.history.get("selected", []))},
            "participation_counts": participation_counts,
        }
    
    def _emergency_cleanup(self):
        """Emergency cleanup when memory is critical - very aggressive."""
        import gc
        
        print(f"EMERGENCY CLEANUP: Aggressively freeing RAM...")
        
        # Force delete all client loaders to free RAM
        if hasattr(self, 'client_loaders') and self.client_loaders:
            loader_ids = list(self.client_loaders.keys())
            for cid in loader_ids:
                try:
                    loader = self.client_loaders[cid]
                    # Delete loader's dataset and workers
                    if hasattr(loader, 'dataset'):
                        del loader.dataset
                    if hasattr(loader, '_iterator'):
                        del loader._iterator
                    del loader
                    self.client_loaders[cid] = None
                except Exception:
                    pass
            
            # Force multiple GC passes before recreating
            for _ in range(3):
                gc.collect()
            
            self.client_loaders.clear()
            
            # Recreate client loaders with REDUCED settings to use less RAM
            try:
                from . import datasets as dset
                # Only recreate loaders we'll actually need (reduce memory footprint)
                # We'll lazily recreate others if needed
                for cid in range(min(self.cfg.total_clients, 50)):  # Limit to first 50
                    if cid in self._partition_mapping:
                        idxs = self._partition_mapping[cid]
                        # Use smaller batch size and fewer workers to reduce RAM
                        batch_size = min(self.cfg.batch_size, 16)
                        self.client_loaders[cid] = dset.make_loaders_from_indices(
                            self.train_ds, idxs, batch_size=batch_size, num_workers=0
                        )
            except Exception as e:
                print(f"WARNING: Failed to recreate loaders: {e}")
        
        # Clear optimizer states if any
        if self._scratch_model is not None:
            del self._scratch_model
            self._scratch_model = None
        
        # Clear parallel trainer caches
        if self._parallel_trainer is not None:
            try:
                # Sync and clear any cached states
                if hasattr(self._parallel_trainer, 'model_replicas'):
                    for replica in self._parallel_trainer.model_replicas:
                        if hasattr(replica, 'zero_grad'):
                            replica.zero_grad(set_to_none=True)
            except Exception:
                pass
        
        # Force garbage collection multiple times
        for _ in range(5):
            gc.collect()
        
        # Clear CUDA cache aggressively
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        print(f"Emergency cleanup completed")
    
    def cleanup(self):
        """Cleanup resources to free memory. Call this between method comparisons."""
        # Clean up parallel trainer
        if self._parallel_trainer is not None:
            if hasattr(self._parallel_trainer, 'cleanup'):
                self._parallel_trainer.cleanup()
            self._parallel_trainer = None
        
        # Clean up scratch model
        if self._scratch_model is not None:
            del self._scratch_model
            self._scratch_model = None
        
        # Clean up client loaders and their datasets
        if hasattr(self, 'client_loaders'):
            for cid in list(self.client_loaders.keys()):
                try:
                    loader = self.client_loaders.get(cid)
                    if loader and hasattr(loader, 'dataset'):
                        del loader.dataset
                    del loader
                except Exception:
                    pass
            self.client_loaders.clear()
        
        # Clean up datasets
        if hasattr(self, 'train_ds'):
            try:
                del self.train_ds
            except Exception:
                pass
        
        if hasattr(self, 'test_loader'):
            try:
                if hasattr(self.test_loader, 'dataset'):
                    del self.test_loader.dataset
                del self.test_loader
            except Exception:
                pass
        
        # Force garbage collection
        import gc
        for _ in range(3):
            gc.collect()
        
        # Aggressive memory cleanup
        cleanup_memory(force_cuda_empty=True, verbose=False)
