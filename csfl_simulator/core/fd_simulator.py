"""Federated Distillation (FD) Simulator.

Implements the FedTSKD and FedTSKD-G algorithms from:
  Mu et al., "Federated Distillation in Massive MIMO Networks:
  Dynamic Training Convergence Analysis and Communication
  Channel-Aware Learning", IEEE TCCN, vol. 10, no. 4, Aug 2024.

The selection interface is identical to FL — all 47+ existing client
selection methods work unchanged.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import datasets as dset
from . import partition as part
from .models import get_model, _dataset_image_spec
from .client import ClientInfo
from .channel import MIMOChannel
from .fd_aggregation import logit_avg, logit_avg_grouped
from .metrics import eval_model
from .utils import new_run_dir, save_json, set_seed, autodetect_device, cleanup_memory, check_memory_critical
from ..selection.registry import MethodRegistry
from .system import init_system_state, simulate_round_env
from .simulator import SimConfig


class FDSimulator:
    """Federated Distillation simulator following Algorithms 1 & 2 of the paper.

    Unlike FL (which exchanges model weights), FD exchanges **logits** on a
    shared public dataset.  Each client maintains its own model (supporting
    heterogeneous architectures) and learns via KL-divergence distillation
    from aggregated logits received from the server.
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        set_seed(self.cfg.seed, deterministic=True)
        self.device = autodetect_device(True) if self.cfg.device == "auto" else self.cfg.device
        self.run_dir, self.run_id = new_run_dir(self.cfg.name or "fd_sim")
        self.registry = MethodRegistry()
        self.registry.load_presets()
        self.history: Dict[str, Any] = {"state": {}, "selected": []}
        self.clients: List[ClientInfo] = []
        self.client_loaders: Dict[int, Any] = {}
        self.client_models: Dict[int, nn.Module] = {}
        self.server_model: Optional[nn.Module] = None
        self._partition_mapping: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        cfg = self.cfg

        # --- Private datasets & partitioning (same as FL) ---
        train_ds, test_ds = dset.get_full_data(cfg.dataset)
        labels = dset.get_labels(train_ds)
        if cfg.partition == "iid":
            mapping = part.iid_partition(labels, cfg.total_clients)
        elif cfg.partition == "dirichlet":
            mapping = part.dirichlet_partition(labels, cfg.total_clients, cfg.dirichlet_alpha)
        else:
            mapping = part.label_shard_partition(labels, cfg.total_clients, cfg.shards_per_client)

        if (cfg.size_distribution or "uniform").lower() != "uniform":
            try:
                mapping = part.apply_size_distribution(
                    mapping, total_size=len(train_ds),
                    size_distribution=cfg.size_distribution,
                    mu=float(cfg.size_lognormal_mu),
                    sigma=float(cfg.size_lognormal_sigma),
                    alpha=float(cfg.size_powerlaw_alpha),
                )
            except Exception as e:
                print(f"Warning: size distribution failed: {e}")

        self._partition_mapping = mapping
        self.train_ds = train_ds

        # --- Infer num_classes ---
        try:
            num_classes = len(getattr(test_ds, 'classes', [])) or int(max(test_ds.targets) + 1)
        except Exception:
            num_classes = 10
        self.num_classes = num_classes

        # --- Test loader ---
        self.test_loader = dset.make_loader(test_ds, batch_size=128, shuffle=False)

        # --- Public dataset for logit exchange ---
        self.public_ds = dset.get_public_dataset(
            name=cfg.public_dataset,
            size=cfg.public_dataset_size,
            training_dataset=cfg.dataset,
            seed=cfg.seed,
        )
        self.public_loader = dset.make_loader(
            self.public_ds, batch_size=cfg.distillation_batch_size, shuffle=False,
        )

        # --- Model pool for heterogeneity ---
        if cfg.model_heterogeneous and cfg.model_pool:
            pool_names = [m.strip() for m in cfg.model_pool.split(",") if m.strip()]
        else:
            pool_names = [cfg.model]

        # --- Build per-client models and loaders ---
        import numpy as _np
        self.client_loaders = {}
        self.clients = []
        for cid in range(cfg.total_clients):
            idxs = mapping[cid]
            self.client_loaders[cid] = dset.make_loaders_from_indices(
                train_ds, idxs, batch_size=cfg.batch_size,
            )
            # Label histogram
            hist = None
            try:
                targets = getattr(train_ds, 'targets', None) or getattr(train_ds, 'labels', None)
                if targets is not None:
                    ys = [int(targets[i]) for i in idxs]
                else:
                    ys = [int(train_ds[i][1]) for i in idxs]
                bc = _np.bincount(ys, minlength=num_classes).astype(float)
                hist = {int(i): float(v) for i, v in enumerate(bc) if v > 0}
            except Exception:
                pass

            arch_name = pool_names[cid % len(pool_names)]
            self.clients.append(ClientInfo(
                id=cid, data_size=len(idxs),
                label_histogram=hist, model_arch=arch_name,
            ))

            # Per-client model
            model = get_model(arch_name, cfg.dataset, num_classes, device=self.device)
            self.client_models[cid] = model

        init_system_state(self.clients, {})

        # --- Server model (same architecture as first client for simplicity) ---
        server_arch = pool_names[0]
        self.server_model = get_model(server_arch, cfg.dataset, num_classes, device=self.device)

        # If group-based (FedTSKD-G), double the server model's last FC layer
        if cfg.group_based:
            self._adapt_server_for_groups()

        # --- Channel model ---
        self.channel: Optional[MIMOChannel] = None
        if cfg.channel_noise:
            self.channel = MIMOChannel(
                n_bs=cfg.n_bs_antennas,
                n_device=cfg.n_device_antennas,
                ul_snr_db=cfg.ul_snr_db,
                dl_snr_db=cfg.dl_snr_db,
                quantization_bits=cfg.quantization_bits,
            )

        # --- Save initial states for fair multi-method comparisons ---
        with torch.no_grad():
            self._initial_client_states = {
                cid: {k: v.clone() for k, v in m.state_dict().items()}
                for cid, m in self.client_models.items()
            }
            self._initial_server_state = {
                k: v.clone() for k, v in self.server_model.state_dict().items()
            }

        # --- Save config ---
        save_json({
            "config": asdict(cfg),
            "device": self.device,
            "paradigm": "fd",
            "model_pool": pool_names,
            "num_classes": num_classes,
            "public_dataset_size": len(self.public_ds),
        }, self.run_dir / "config.json")

        # --- Profiling ---
        self.model_params_count = sum(
            p.numel() for p in self.server_model.parameters()
        )

        # --- Performance: cache public data on GPU ---
        self._public_x_cache: Optional[torch.Tensor] = None
        try:
            xs = [x for x, *_ in self.public_loader]
            self._public_x_cache = torch.cat(xs, dim=0).to(self.device)
        except Exception:
            pass  # fallback to loader-based iteration

        # --- Performance: cache test set on GPU ---
        self._test_x_cache: Optional[torch.Tensor] = None
        self._test_y_cache: Optional[torch.Tensor] = None
        try:
            xs, ys = [], []
            for x, y in self.test_loader:
                xs.append(x)
                ys.append(y)
            self._test_x_cache = torch.cat(xs, dim=0).to(self.device)
            self._test_y_cache = torch.cat(ys, dim=0).to(self.device)
        except Exception:
            pass

        # --- Performance: cache training data on GPU (skip if insufficient VRAM) ---
        self._client_data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.device.startswith("cuda"):
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                # Estimate total training data size
                sample_x, _ = train_ds[0]
                if isinstance(sample_x, torch.Tensor):
                    bytes_per_sample = sample_x.numel() * sample_x.element_size() + 8  # +8 for label
                else:
                    bytes_per_sample = 3 * 32 * 32 * 4 + 8  # fallback estimate
                total_bytes = len(train_ds) * bytes_per_sample
                # Only cache if >1GB remains free after caching
                if total_bytes < free_mem - 1024 ** 3:
                    for cid in range(cfg.total_clients):
                        xs, ys = [], []
                        for x, y in self.client_loaders[cid]:
                            xs.append(x)
                            ys.append(y)
                        self._client_data_cache[cid] = (
                            torch.cat(xs, dim=0).to(self.device),
                            torch.cat(ys, dim=0).to(self.device),
                        )
            except Exception:
                self._client_data_cache.clear()

        # --- Performance: AMP (mixed precision) ---
        self._use_amp = cfg.use_amp and self.device.startswith("cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp) if self._use_amp else None

        # --- Performance: pre-allocated client optimizers ---
        self._client_train_opts: Dict[int, optim.Optimizer] = {}
        self._client_distill_opts: Dict[int, optim.Optimizer] = {}
        self._server_distill_opt: Optional[optim.Optimizer] = None

        # --- Performance: CUDA parallelism detection ---
        self._max_parallel = 1
        if self.device.startswith("cuda") and cfg.parallel_clients != 0:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info()
                model_mem = sum(p.numel() * p.element_size() for p in self.server_model.parameters())
                per_client = model_mem * 4  # model + grads + optimizer + activations
                usable = int(free_mem * 0.5)
                auto = max(1, min(usable // max(per_client, 1), 8))
                if cfg.parallel_clients == -1:
                    self._max_parallel = auto
                elif cfg.parallel_clients > 0:
                    self._max_parallel = cfg.parallel_clients
                else:
                    self._max_parallel = auto
            except Exception:
                self._max_parallel = 2

    def _adapt_server_for_groups(self):
        """Double the server model's last FC layer for FedTSKD-G."""
        # Find the last Linear layer
        last_linear_name = None
        last_linear = None
        for name, mod in self.server_model.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear_name = name
                last_linear = mod
        if last_linear is not None and last_linear_name is not None:
            new_fc = nn.Linear(last_linear.in_features, 2 * self.num_classes).to(self.device)
            parts = last_linear_name.split(".")
            parent = self.server_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_fc)

    # ------------------------------------------------------------------
    # Dynamic training steps (Section V-A)
    # ------------------------------------------------------------------

    def _compute_dynamic_steps(self, round_idx: int, client_data_size: int) -> int:
        """Compute K_r: the number of local training steps for round r.

        From the paper:
            K_r = ceil(D_n / |D_{n,t}|) * max(1, base - floor((r-1) / period))
        where |D_{n,t}| is the batch size.
        """
        if not self.cfg.dynamic_steps:
            return max(1, self.cfg.local_epochs * math.ceil(client_data_size / self.cfg.batch_size))

        base = self.cfg.dynamic_steps_base
        period = self.cfg.dynamic_steps_period
        multiplier = max(1, base - (round_idx // period))
        steps_per_epoch = max(1, math.ceil(client_data_size / self.cfg.batch_size))
        return steps_per_epoch * multiplier

    # ------------------------------------------------------------------
    # Local training on private data
    # ------------------------------------------------------------------

    def _get_train_opt(self, cid: int) -> optim.Optimizer:
        """Get or create the training optimizer for a client (reused across rounds)."""
        if cid not in self._client_train_opts:
            model = self.client_models[cid]
            if self.cfg.fd_optimizer == "adam":
                self._client_train_opts[cid] = optim.Adam(model.parameters(), lr=self.cfg.distillation_lr)
            else:
                self._client_train_opts[cid] = optim.SGD(model.parameters(), lr=self.cfg.lr)
        return self._client_train_opts[cid]

    def _get_distill_opt(self, cid: int) -> optim.Optimizer:
        """Get or create the distillation optimizer for a client."""
        if cid not in self._client_distill_opts:
            self._client_distill_opts[cid] = optim.Adam(
                self.client_models[cid].parameters(), lr=self.cfg.distillation_lr,
            )
        return self._client_distill_opts[cid]

    def _get_server_distill_opt(self) -> optim.Optimizer:
        """Get or create the server distillation optimizer."""
        if self._server_distill_opt is None:
            self._server_distill_opt = optim.Adam(
                self.server_model.parameters(), lr=self.cfg.distillation_lr,
            )
        return self._server_distill_opt

    def _local_train(self, cid: int, num_steps: int) -> Tuple[float, float]:
        """Train client cid's model on private data for num_steps SGD/Adam steps.

        Returns (last_loss, grad_norm).
        """
        cfg = self.cfg
        model = self.client_models[cid]
        model.train()
        criterion = nn.CrossEntropyLoss()
        opt = self._get_train_opt(cid)

        last_loss = 0.0
        grad_norm = 0.0
        use_amp = self._use_amp
        scaler = self._scaler

        # Fast path: GPU-cached training data (no DataLoader overhead)
        if cid in self._client_data_cache:
            x_all, y_all = self._client_data_cache[cid]
            n = x_all.size(0)
            bs = cfg.batch_size
            for step in range(num_steps):
                idx = (step * bs) % n
                x = x_all[idx:idx + bs]
                y = y_all[idx:idx + bs]
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.track_grad_norm:
                        scaler.unscale_(opt)
                        try:
                            total_sq = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
                            grad_norm = total_sq ** 0.5
                        except Exception:
                            pass
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.track_grad_norm:
                        try:
                            total_sq = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
                            grad_norm = total_sq ** 0.5
                        except Exception:
                            pass
                    opt.step()
                last_loss = float(loss.item())
                if cfg.fast_mode and step > 0:
                    return last_loss, grad_norm
            return last_loss, grad_norm

        # Fallback: DataLoader-based training
        if cid not in self.client_loaders or self.client_loaders[cid] is None:
            if cid in self._partition_mapping:
                self.client_loaders[cid] = dset.make_loaders_from_indices(
                    self.train_ds, self._partition_mapping[cid],
                    batch_size=cfg.batch_size, num_workers=0,
                )

        loader = self.client_loaders[cid]
        step = 0
        while step < num_steps:
            for x, y in loader:
                if step >= num_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.track_grad_norm:
                        scaler.unscale_(opt)
                        try:
                            total_sq = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
                            grad_norm = total_sq ** 0.5
                        except Exception:
                            pass
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.track_grad_norm:
                        try:
                            total_sq = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
                            grad_norm = total_sq ** 0.5
                        except Exception:
                            pass
                    opt.step()
                last_loss = float(loss.item())
                step += 1
                if cfg.fast_mode and step > 1:
                    return last_loss, grad_norm
        return last_loss, grad_norm

    # ------------------------------------------------------------------
    # Inference on public dataset (Eq. 13)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_logits(self, cid: int) -> torch.Tensor:
        """Run client model on public dataset, returning logits of shape (N_pub, C)."""
        model = self.client_models[cid]
        model.eval()
        if self._public_x_cache is not None:
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                return model(self._public_x_cache).float()  # always return fp32 logits
        all_logits = []
        for x, *_ in self.public_loader:
            x = x.to(self.device)
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                all_logits.append(model(x).float())
        return torch.cat(all_logits, dim=0)

    # ------------------------------------------------------------------
    # Local distillation (Eq. 3 / 14)
    # ------------------------------------------------------------------

    def _local_distill(self, cid: int, target_logits: torch.Tensor, num_epochs: int) -> float:
        """Distill aggregated logits into client cid's model via KL divergence.

        Q_n(w; Z) = (1/|Z|) * sum_p KL(log_softmax(F_n(x_p;w)/T), softmax(z_p/T))

        Returns average distillation loss.
        """
        cfg = self.cfg
        T = cfg.temperature
        model = self.client_models[cid]
        model.train()
        opt = self._get_distill_opt(cid)
        total_loss = 0.0
        n_batches = 0
        target_logits = target_logits.to(self.device)
        use_amp = self._use_amp
        scaler = self._scaler

        # Use cached public data if available (avoids CPU->GPU transfer)
        if self._public_x_cache is not None:
            public_x = self._public_x_cache
            for epoch in range(num_epochs):
                for i in range(0, public_x.size(0), cfg.distillation_batch_size):
                    x_batch = public_x[i:i + cfg.distillation_batch_size]
                    t_batch = target_logits[i:i + cfg.distillation_batch_size]
                    opt.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        log_p = F.log_softmax(model(x_batch) / T, dim=-1)
                        q = F.softmax(t_batch / T, dim=-1)
                        loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    total_loss += loss.item()
                    n_batches += 1
                    if cfg.fast_mode and n_batches > 1:
                        return total_loss / n_batches
        else:
            for epoch in range(num_epochs):
                offset = 0
                for x, *_ in self.public_loader:
                    x = x.to(self.device)
                    bs = x.size(0)
                    t_batch = target_logits[offset:offset + bs]
                    offset += bs
                    opt.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        log_p = F.log_softmax(model(x) / T, dim=-1)
                        q = F.softmax(t_batch / T, dim=-1)
                        loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    total_loss += loss.item()
                    n_batches += 1
                    if cfg.fast_mode and n_batches > 1:
                        return total_loss / n_batches

        return total_loss / max(1, n_batches)

    # ------------------------------------------------------------------
    # Server-side distillation
    # ------------------------------------------------------------------

    def _server_distill(self, aggregated_logits: torch.Tensor, num_epochs: int) -> float:
        """Server distills its own model using aggregated logits from clients."""
        cfg = self.cfg
        T = cfg.temperature
        model = self.server_model
        model.train()
        opt = self._get_server_distill_opt()
        total_loss = 0.0
        n_batches = 0
        aggregated_logits = aggregated_logits.to(self.device)
        use_amp = self._use_amp
        scaler = self._scaler

        if self._public_x_cache is not None:
            public_x = self._public_x_cache
            for epoch in range(num_epochs):
                for i in range(0, public_x.size(0), cfg.distillation_batch_size):
                    x_batch = public_x[i:i + cfg.distillation_batch_size]
                    t_batch = aggregated_logits[i:i + cfg.distillation_batch_size]
                    opt.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        log_p = F.log_softmax(model(x_batch) / T, dim=-1)
                        q = F.softmax(t_batch / T, dim=-1)
                        loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    total_loss += loss.item()
                    n_batches += 1
                    if cfg.fast_mode and n_batches > 1:
                        return total_loss / n_batches
        else:
            for epoch in range(num_epochs):
                offset = 0
                for x, *_ in self.public_loader:
                    x = x.to(self.device)
                    bs = x.size(0)
                    t_batch = aggregated_logits[offset:offset + bs]
                    offset += bs
                    opt.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        log_p = F.log_softmax(model(x) / T, dim=-1)
                        q = F.softmax(t_batch / T, dim=-1)
                        loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    total_loss += loss.item()
                    n_batches += 1
                    if cfg.fast_mode and n_batches > 1:
                        return total_loss / n_batches

        return total_loss / max(1, n_batches)

    # ------------------------------------------------------------------
    # Main FD training loop (Algorithm 1 / 2)
    # ------------------------------------------------------------------

    def run(self, method_key: str = "heuristic.random", on_progress=None, is_cancelled=None) -> Dict[str, Any]:
        """Execute the full FD training loop.

        Follows Algorithm 1 (FedTSKD) or Algorithm 2 (FedTSKD-G) from the
        paper, with client selection plugged in at the start of each round.
        """
        if not hasattr(self, 'train_ds') or self.train_ds is None:
            self.setup()

        cfg = self.cfg

        # --- Reset state for fair multi-method comparison ---
        if hasattr(self, '_initial_client_states'):
            for cid, sd in self._initial_client_states.items():
                self.client_models[cid].load_state_dict(sd)
        if hasattr(self, '_initial_server_state'):
            self.server_model.load_state_dict(self._initial_server_state)
        self.history = {"state": {}, "selected": []}
        for c in self.clients:
            c.last_loss = 0.0
            c.grad_norm = 0.0
            c.participation_count = 0
            c.last_selected_round = -1
        set_seed(cfg.seed, deterministic=True)
        # Reset cached optimizers and AMP scaler
        self._client_train_opts.clear()
        self._client_distill_opts.clear()
        self._server_distill_opt = None
        if self._use_amp:
            self._scaler = torch.amp.GradScaler("cuda", enabled=True)

        # --- Baseline evaluation (round -1) ---
        metrics = []
        base = self._evaluate_clients()
        base["round"] = -1
        base["cum_comm"] = 0.0
        base["cum_tflops"] = 0.0
        base["wall_clock"] = 0.0
        metrics.append(base)

        # Tracking across rounds
        cum_comm_kb = 0.0
        cum_tflops = 0.0
        prev_wall_clock = 0.0
        prev_composite = 0.0
        aggregated_logits: Optional[torch.Tensor] = None
        last_eval: Dict[str, float] = {}

        # --- Round loop ---
        for rnd in range(cfg.rounds):
            if is_cancelled and is_cancelled():
                break

            rnd_start = time.perf_counter()

            # Phase 1: Update environment & select clients
            simulate_round_env(self.clients, {"paradigm": "fd", "channel_threshold": cfg.channel_threshold}, rnd)

            sel_start = time.perf_counter()
            ids, scores, state = self.registry.invoke(
                method_key, rnd, cfg.clients_per_round,
                self.clients, self.history, random,
                cfg.time_budget, self.device,
                energy_budget=cfg.energy_budget,
                bytes_budget=cfg.bytes_budget,
            )
            selection_time = time.perf_counter() - sel_start

            self.history["selected"].append(ids)
            if state:
                self.history["state"].update(state)
            for cid in ids:
                self.clients[cid].last_selected_round = rnd

            # Phase 2-4: Per-client distillation + training + inference
            # Parallelised across clients using CUDA streams when available
            kl_divs = []
            client_logits = {}

            def _process_client(cid: int) -> Tuple[float, torch.Tensor]:
                """Run distill -> train -> inference for one client. Returns (kl, logits)."""
                kl = 0.0
                # Phase 2: Local distillation (skip round 0)
                if aggregated_logits is not None and rnd > 0:
                    if cfg.group_based and self.channel is not None:
                        group = self.clients[cid].meta.get("channel_group", "good")
                        target = aggregated_logits[:, :self.num_classes] if group == "bad" else aggregated_logits[:, self.num_classes:]
                    else:
                        target = aggregated_logits
                    if self.channel is not None:
                        target = self.channel.quantize(target.clone())
                        target = self.channel.downlink_noise(target)
                    kl = self._local_distill(cid, target, cfg.distillation_epochs)

                # Phase 3: Local training
                K_r = self._compute_dynamic_steps(rnd, self.clients[cid].data_size)
                loss, gnorm = self._local_train(cid, K_r)
                self.clients[cid].last_loss = loss
                self.clients[cid].grad_norm = gnorm
                self.clients[cid].participation_count += 1

                # Phase 4: Inference on public dataset + uplink
                logits = self._generate_logits(cid)
                if self.channel is not None:
                    logits = self.channel.quantize(logits)
                    logits = self.channel.uplink_noise(logits)
                return kl, logits

            # Run clients in parallel batches using CUDA streams
            is_cuda = self.device.startswith("cuda")
            for batch_start in range(0, len(ids), self._max_parallel):
                batch_ids = ids[batch_start:batch_start + self._max_parallel]

                if is_cuda and len(batch_ids) > 1:
                    streams = [torch.cuda.Stream() for _ in batch_ids]
                    results: List[Tuple[float, torch.Tensor]] = [None] * len(batch_ids)  # type: ignore
                    for i, cid in enumerate(batch_ids):
                        with torch.cuda.stream(streams[i]):
                            results[i] = _process_client(cid)
                    torch.cuda.synchronize()
                    for i, cid in enumerate(batch_ids):
                        kl, logits = results[i]
                        if kl > 0:
                            kl_divs.append(kl)
                        client_logits[cid] = logits
                else:
                    for cid in batch_ids:
                        kl, logits = _process_client(cid)
                        if kl > 0:
                            kl_divs.append(kl)
                        client_logits[cid] = logits

            # Phase 5: Server aggregation
            weights = [float(self.clients[cid].data_size) for cid in ids]
            logit_list = [client_logits[cid] for cid in ids]

            if cfg.group_based:
                group_labels = [
                    self.clients[cid].meta.get("channel_group", "good")
                    for cid in ids
                ]
                aggregated_logits = logit_avg_grouped(logit_list, weights, group_labels)
            else:
                aggregated_logits = logit_avg(logit_list, weights)

            # Phase 6: Server-side distillation
            server_target = aggregated_logits
            server_kl = self._server_distill(server_target, cfg.distillation_epochs)

            # Server predicts on public dataset → logits for next round's downlink
            self.server_model.eval()
            with torch.no_grad():
                if self._public_x_cache is not None:
                    server_logits = self.server_model(self._public_x_cache)
                else:
                    server_preds = []
                    for x, *_ in self.public_loader:
                        x = x.to(self.device)
                        server_preds.append(self.server_model(x))
                    server_logits = torch.cat(server_preds, dim=0)

            # For next round, clients will receive the server's predicted logits
            if cfg.group_based:
                # Split server logits into bad/good groups
                aggregated_logits = server_logits  # shape: (N_pub, 2*C) for group-based
            else:
                aggregated_logits = server_logits

            # Phase 7: Evaluation
            rnd_time = time.perf_counter() - rnd_start
            compute_time = max(
                (self.clients[cid].estimated_duration for cid in ids),
                default=0.0,
            )
            wall_clock = prev_wall_clock + rnd_time

            # Communication metrics (FD overhead)
            K_r_current = self._compute_dynamic_steps(rnd, max((self.clients[cid].data_size for cid in ids), default=1))
            n_pub = len(self.public_ds)
            C = self.num_classes
            quant_bytes = cfg.quantization_bits / 8.0
            # Uplink: K clients send logits; Downlink: server broadcasts logits
            logit_comm_kb = (len(ids) * n_pub * C * quant_bytes + n_pub * C * quant_bytes) / 1024.0
            # FL equivalent: 2 * K * model_size_bytes
            fl_model_bytes = self.model_params_count * 4.0
            fl_equiv_comm_mb = 2.0 * len(ids) * fl_model_bytes / (1024.0 * 1024.0)
            comm_reduction = logit_comm_kb / 1024.0 / max(fl_equiv_comm_mb, 1e-10)

            cum_comm_kb += logit_comm_kb

            # Effective noise variance
            eff_noise_var = 0.0
            if self.channel is not None:
                logit_var = aggregated_logits.var().item() if aggregated_logits is not None else 0.0
                eff_noise_var = self.channel.effective_noise_variance(len(ids), logit_var)

            # Evaluate client models — skip most rounds for speed
            eval_every = max(1, cfg.eval_every)
            do_eval = (rnd % eval_every == 0) or (rnd == cfg.rounds - 1) or (rnd == 0)
            if do_eval:
                m = self._evaluate_clients(sample_ids=ids)
                server_m = self._evaluate_server()
                m["server_accuracy"] = server_m.get("accuracy", 0.0)
                m["server_loss"] = server_m.get("loss", 0.0)
                m["server_f1"] = server_m.get("f1", 0.0)
                last_eval = m.copy()
            else:
                m = last_eval.copy()

            # Fairness metrics
            counts = [c.participation_count for c in self.clients]
            mean_p = sum(counts) / max(len(counts), 1)
            fairness_var = sum((p - mean_p) ** 2 for p in counts) / max(len(counts), 1)
            pairs_sum = sum(abs(counts[i] - counts[j]) for i in range(len(counts)) for j in range(i + 1, len(counts)))
            fairness_gini = pairs_sum / (len(counts) ** 2 * max(mean_p, 1e-10)) if len(counts) > 0 else 0.0

            # Good/bad channel counts
            n_good = sum(1 for cid in ids if self.clients[cid].meta.get("channel_group", "good") == "good")
            n_bad = len(ids) - n_good

            m.update({
                "round": rnd,
                "selection_time": selection_time,
                "compute_time": compute_time,
                "round_time": rnd_time,
                "wall_clock": wall_clock,
                "cum_comm": cum_comm_kb / 1024.0,  # MB
                "cum_tflops": cum_tflops,
                "fairness_var": fairness_var,
                "fairness_gini": fairness_gini,
                # FD-specific metrics
                "kl_divergence_avg": sum(kl_divs) / max(len(kl_divs), 1) if kl_divs else 0.0,
                "distillation_loss_avg": server_kl,
                "logit_comm_kb": logit_comm_kb,
                "fl_equiv_comm_mb": fl_equiv_comm_mb,
                "comm_reduction_ratio": comm_reduction,
                "effective_noise_var": eff_noise_var,
                "dynamic_steps_kr": K_r_current,
                "num_good_channel": n_good,
                "num_bad_channel": n_bad,
                "client_accuracy_avg": m.get("accuracy", 0.0),
                "client_accuracy_std": m.get("accuracy_std", 0.0),
            })

            # Composite score (same formula as FL for comparability)
            rw = cfg.reward_weights
            acc_s = m.get("accuracy", 0.0)
            time_s = 1.0 - rnd_time / (rnd_time + 1.0)
            fair_s = 1.0 - fairness_var / (fairness_var + 1.0)
            composite = rw.get("acc", 0.6) * acc_s + rw.get("time", 0.2) * time_s + rw.get("fair", 0.1) * fair_s
            m["composite"] = composite
            m["reward"] = composite - prev_composite
            prev_composite = composite

            metrics.append(m)
            prev_wall_clock = wall_clock

            # Progress callback
            if on_progress:
                on_progress(rnd, {"accuracy": m.get("accuracy", 0.0), "metrics": m})

            # Memory management (infrequent — small models don't need aggressive cleanup)
            if rnd % 50 == 0:
                cleanup_memory(force_cuda_empty=True)
                if check_memory_critical():
                    self._emergency_cleanup()

        # --- Convergence summary ---
        convergence = self._convergence_summary(metrics)

        # Save metrics
        save_json({"metrics": metrics, "convergence": convergence}, self.run_dir / "metrics.json")

        return {
            "run_id": self.run_id,
            "metrics": metrics,
            "config": asdict(cfg),
            "device": self.device,
            "stopped_early": (is_cancelled() if is_cancelled else False),
            "method": method_key,
            "paradigm": "fd",
            "history": {"selected": self.history["selected"]},
            "participation_counts": [c.participation_count for c in self.clients],
            "convergence": convergence,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_clients(self, sample_ids: Optional[List[int]] = None) -> Dict[str, float]:
        """Evaluate client models on test set, returning averaged metrics.

        Uses GPU-cached test data when available to avoid DataLoader overhead.
        Iterates test data once per chunk, running all client models per chunk.
        """
        ids_to_eval = sample_ids or list(range(min(self.cfg.total_clients, 10)))
        n_clients = len(ids_to_eval)

        # Set all models to eval mode
        for cid in ids_to_eval:
            self.client_models[cid].eval()

        if self._test_x_cache is not None and self._test_y_cache is not None:
            return self._evaluate_clients_cached(ids_to_eval)

        # Fallback: DataLoader-based evaluation
        per_client_correct = [0] * n_clients
        per_client_total = [0] * n_clients
        all_ys: List[int] = []
        all_preds: List[int] = []
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for cid_idx, cid in enumerate(ids_to_eval):
                model = self.client_models[cid]
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    total_loss += F.cross_entropy(out, y, reduction='sum').item()
                    preds = out.argmax(dim=1)
                    per_client_correct[cid_idx] += (preds == y).sum().item()
                    per_client_total[cid_idx] += y.size(0)
                    all_ys.extend(y.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
                    n_samples += y.size(0)

        return self._compute_eval_metrics(ids_to_eval, per_client_correct, per_client_total,
                                          all_ys, all_preds, total_loss, n_samples)

    def _evaluate_clients_cached(self, ids_to_eval: List[int]) -> Dict[str, float]:
        """Fast evaluation using GPU-cached test data. Single data iteration, all models per chunk."""
        n_clients = len(ids_to_eval)
        N_test = self._test_x_cache.size(0)
        chunk_size = 512

        per_client_correct = [0] * n_clients
        per_client_total = [0] * n_clients
        # Accumulate all predictions as GPU tensors (single .cpu() at end)
        all_ys_parts: List[torch.Tensor] = []
        all_preds_parts: List[torch.Tensor] = []
        total_loss = 0.0

        with torch.no_grad():
            for start in range(0, N_test, chunk_size):
                x_chunk = self._test_x_cache[start:start + chunk_size]
                y_chunk = self._test_y_cache[start:start + chunk_size]
                bs = x_chunk.size(0)

                for i, cid in enumerate(ids_to_eval):
                    out = self.client_models[cid](x_chunk)
                    total_loss += F.cross_entropy(out, y_chunk, reduction='sum').item()
                    preds = out.argmax(dim=1)
                    per_client_correct[i] += (preds == y_chunk).sum().item()
                    per_client_total[i] += bs
                    all_ys_parts.append(y_chunk)
                    all_preds_parts.append(preds)

        # Single GPU->CPU transfer
        all_ys = torch.cat(all_ys_parts).cpu().tolist()
        all_preds = torch.cat(all_preds_parts).cpu().tolist()
        n_samples = N_test * n_clients

        return self._compute_eval_metrics(ids_to_eval, per_client_correct, per_client_total,
                                          all_ys, all_preds, total_loss, n_samples)

    def _compute_eval_metrics(self, ids_to_eval, per_client_correct, per_client_total,
                              all_ys, all_preds, total_loss, n_samples) -> Dict[str, float]:
        """Compute accuracy, F1, precision, recall from accumulated predictions."""
        all_accs = [c / max(t, 1) for c, t in zip(per_client_correct, per_client_total)]

        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            accuracy = accuracy_score(all_ys, all_preds)
            f1 = f1_score(all_ys, all_preds, average='macro', zero_division=0)
            precision = precision_score(all_ys, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_ys, all_preds, average='macro', zero_division=0)
        except ImportError:
            accuracy = sum(1 for y, p in zip(all_ys, all_preds) if y == p) / max(len(all_ys), 1)
            f1 = precision = recall = 0.0

        import statistics
        acc_std = statistics.stdev(all_accs) if len(all_accs) > 1 else 0.0

        return {
            "accuracy": accuracy,
            "loss": total_loss / max(n_samples, 1),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy_std": acc_std,
            "num_clients_evaluated": len(ids_to_eval),
        }

    def _evaluate_server(self) -> Dict[str, float]:
        """Evaluate the server model on the test set (selection-independent metric)."""
        return eval_model(self.server_model, self.test_loader, self.device)

    def _convergence_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Compute convergence efficiency metrics from the round-by-round metrics."""
        rounds = [m for m in metrics if m.get("round", -1) >= 0]
        if not rounds:
            return {}

        accs = [m.get("accuracy", 0.0) for m in rounds]
        times = [m.get("wall_clock", 0.0) for m in rounds]
        final_acc = accs[-1] if accs else 0.0
        base_acc = metrics[0].get("accuracy", 0.0) if metrics else 0.0
        improvement = final_acc - base_acc

        result: Dict[str, Any] = {
            "final_accuracy": final_acc,
            "total_improvement": improvement,
            "total_rounds": len(rounds),
        }

        # Time-to-X% of improvement
        for pct in (0.5, 0.8, 0.9):
            target = base_acc + pct * improvement
            for i, acc in enumerate(accs):
                if acc >= target:
                    result[f"rounds_to_{int(pct*100)}pct"] = i + 1
                    result[f"time_to_{int(pct*100)}pct"] = times[i] if i < len(times) else None
                    break
            else:
                result[f"rounds_to_{int(pct*100)}pct"] = None
                result[f"time_to_{int(pct*100)}pct"] = None

        return result

    def _emergency_cleanup(self):
        """Free memory aggressively when RAM is critical."""
        import gc
        for cid in list(self.client_loaders.keys()):
            self.client_loaders[cid] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        """Release all resources."""
        self.client_models.clear()
        self.client_loaders.clear()
        self.server_model = None
        cleanup_memory(force_cuda_empty=True)
