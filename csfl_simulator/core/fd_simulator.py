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

    def _local_train(self, cid: int, num_steps: int) -> Tuple[float, float]:
        """Train client cid's model on private data for num_steps SGD/Adam steps.

        Returns (last_loss, grad_norm).
        """
        cfg = self.cfg
        model = self.client_models[cid]
        model.train()

        # Recreate loader if missing (emergency cleanup recovery)
        if cid not in self.client_loaders or self.client_loaders[cid] is None:
            if cid in self._partition_mapping:
                self.client_loaders[cid] = dset.make_loaders_from_indices(
                    self.train_ds, self._partition_mapping[cid],
                    batch_size=cfg.batch_size, num_workers=0,
                )

        loader = self.client_loaders[cid]
        criterion = nn.CrossEntropyLoss()

        if cfg.fd_optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=cfg.distillation_lr)
        else:
            opt = optim.SGD(model.parameters(), lr=cfg.lr)

        last_loss = 0.0
        grad_norm = 0.0
        step = 0
        while step < num_steps:
            for x, y in loader:
                if step >= num_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                if cfg.track_grad_norm:
                    try:
                        total_sq = sum(
                            p.grad.data.norm(2).item() ** 2
                            for p in model.parameters() if p.grad is not None
                        )
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
        all_logits = []
        for x, *_ in self.public_loader:
            x = x.to(self.device)
            logits = model(x)
            all_logits.append(logits)
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

        opt = optim.Adam(model.parameters(), lr=cfg.distillation_lr)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(num_epochs):
            offset = 0
            for x, *_ in self.public_loader:
                x = x.to(self.device)
                bs = x.size(0)
                target_batch = target_logits[offset:offset + bs].to(self.device)
                offset += bs

                opt.zero_grad()
                student_logits = model(x)

                # KL(log_softmax(student/T), softmax(target/T)) * T^2
                log_p = F.log_softmax(student_logits / T, dim=-1)
                q = F.softmax(target_batch / T, dim=-1)
                loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)

                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

                if cfg.fast_mode and n_batches > 1:
                    return total_loss / max(1, n_batches)

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

        opt = optim.Adam(model.parameters(), lr=cfg.distillation_lr)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(num_epochs):
            offset = 0
            for x, *_ in self.public_loader:
                x = x.to(self.device)
                bs = x.size(0)
                target_batch = aggregated_logits[offset:offset + bs].to(self.device)
                offset += bs

                opt.zero_grad()
                server_logits = model(x)
                log_p = F.log_softmax(server_logits / T, dim=-1)
                q = F.softmax(target_batch / T, dim=-1)
                loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

                if cfg.fast_mode and n_batches > 1:
                    return total_loss / max(1, n_batches)

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

            # Phase 2: Local distillation with previously received logits (skip round 0)
            kl_divs = []
            if aggregated_logits is not None and rnd > 0:
                for cid in ids:
                    if cfg.group_based and self.channel is not None:
                        # FedTSKD-G: client receives group-specific logits
                        group = self.clients[cid].meta.get("channel_group", "good")
                        if group == "bad":
                            target = aggregated_logits[:, :self.num_classes]
                        else:
                            target = aggregated_logits[:, self.num_classes:]
                    else:
                        target = aggregated_logits

                    # Apply downlink channel noise per client
                    if self.channel is not None:
                        target = self.channel.downlink_noise(target.clone())

                    kl = self._local_distill(cid, target, cfg.distillation_epochs)
                    kl_divs.append(kl)

            # Phase 3: Local training with dynamic steps
            for cid in ids:
                K_r = self._compute_dynamic_steps(rnd, self.clients[cid].data_size)
                loss, gnorm = self._local_train(cid, K_r)
                self.clients[cid].last_loss = loss
                self.clients[cid].grad_norm = gnorm
                self.clients[cid].participation_count += 1

            # Phase 4: Inference on public dataset + uplink
            client_logits = {}
            for cid in ids:
                logits = self._generate_logits(cid)
                if self.channel is not None:
                    logits = self.channel.quantize(logits)
                    logits = self.channel.uplink_noise(logits)
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

            # Evaluate client models (average accuracy)
            m = self._evaluate_clients(sample_ids=ids)

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

            # Memory management
            cleanup_memory(force_cuda_empty=(rnd % 5 == 0))
            if rnd % 3 == 0 and check_memory_critical():
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
        """Evaluate client models on test set, returning averaged metrics."""
        ids_to_eval = sample_ids or list(range(min(self.cfg.total_clients, 10)))
        all_accs = []
        # Collect combined predictions from all client models
        all_ys = []
        all_preds = []
        total_loss = 0.0
        n_samples = 0

        for cid in ids_to_eval:
            model = self.client_models[cid]
            model.eval()
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    total_loss += F.cross_entropy(out, y, reduction='sum').item()
                    preds = out.argmax(dim=1)
                    all_ys.extend(y.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
                    n_samples += y.size(0)

            # Per-client accuracy
            m = eval_model(model, self.test_loader, self.device)
            all_accs.append(m.get("accuracy", 0.0))

        # Compute averaged metrics
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            accuracy = accuracy_score(all_ys, all_preds)
            f1 = f1_score(all_ys, all_preds, average='macro', zero_division=0)
            precision = precision_score(all_ys, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_ys, all_preds, average='macro', zero_division=0)
        except ImportError:
            accuracy = sum(1 for y, p in zip(all_ys, all_preds) if y == p) / max(len(all_ys), 1)
            f1 = precision = recall = 0.0

        avg_loss = total_loss / max(n_samples, 1)

        # Client accuracy std
        import statistics
        acc_std = statistics.stdev(all_accs) if len(all_accs) > 1 else 0.0

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy_std": acc_std,
            "num_clients_evaluated": len(ids_to_eval),
        }

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
