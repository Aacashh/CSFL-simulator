"""Additive simulator instrumentation for the MAML-Select letter experiments."""
from __future__ import annotations

import math
import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from csfl_simulator.core.aggregation import fedavg
from csfl_simulator.core.metrics import eval_model
from csfl_simulator.core.simulator import FLSimulator, SimConfig
from csfl_simulator.core.system import simulate_round_env
from csfl_simulator.core.utils import cleanup_memory, save_json, set_seed


CRITICALFL_STATE_KEY = "research_criticalfl_state"


def _gini(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    if total <= 0.0 or len(array) == 0:
        return 0.0
    return float(np.abs(array[:, None] - array[None, :]).sum() / (2.0 * len(array) * total))


def _jain(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    denominator = float(len(array) * np.square(array).sum())
    return float(array.sum() ** 2 / denominator) if denominator > 0.0 else 0.0


def _entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    if total <= 0.0 or len(array) <= 1:
        return 0.0
    probabilities = array[array > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(array)))


def _label_coverage(clients, ids: Iterable[int], class_count: int) -> float:
    covered = set()
    for cid in ids:
        histogram = clients[int(cid)].label_histogram
        if isinstance(histogram, dict):
            covered.update(int(label) for label, count in histogram.items() if float(count) > 0.0)
    return float(len(covered) / max(1, int(class_count)))


def _first_time_to(metrics: Sequence[Dict[str, Any]], target: float) -> float:
    for row in metrics:
        if int(row.get("round", -1)) >= 0 and float(row.get("accuracy", 0.0)) >= target:
            return float(row.get("cum_time", 0.0))
    return float("nan")


def _first_metric_to(metrics: Sequence[Dict[str, Any]], target: float, metric_name: str) -> float:
    for row in metrics:
        if int(row.get("round", -1)) >= 0 and float(row.get("accuracy", 0.0)) >= target:
            return float(row.get(metric_name, 0.0))
    return float("nan")


def _criticalfl_sparse_fedavg(updates, weights, global_state, keep_fraction: float):
    """Aggregate top-magnitude client updates and preserve unreported parameters."""
    keep_fraction = min(1.0, max(0.0, float(keep_fraction)))
    masked_updates = []
    masks = []
    for state_dict in updates:
        magnitudes = []
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                magnitudes.append((value.detach() - global_state[key].detach()).abs().reshape(-1))
        flat = torch.cat(magnitudes) if magnitudes else torch.empty(0)
        keep_count = max(1, int(math.ceil(keep_fraction * flat.numel()))) if flat.numel() else 0
        threshold = torch.topk(flat, min(keep_count, flat.numel()), sorted=False).values.min() if keep_count else None
        client_masks = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and threshold is not None:
                client_masks[key] = (value.detach() - global_state[key].detach()).abs() >= threshold
        masked_updates.append(state_dict)
        masks.append(client_masks)

    total = float(sum(weights)) or 1.0
    aggregated = {}
    with torch.no_grad():
        for key, initial in global_state.items():
            if not isinstance(initial, torch.Tensor) or not initial.is_floating_point():
                aggregated[key] = initial.detach().clone() if isinstance(initial, torch.Tensor) else initial
                continue
            numerator = torch.zeros_like(initial)
            denominator = torch.zeros_like(initial)
            for state_dict, client_masks, weight in zip(masked_updates, masks, weights):
                mask = client_masks.get(key)
                if mask is None:
                    continue
                fraction = float(weight) / total
                numerator.add_(state_dict[key] * mask, alpha=fraction)
                denominator.add_(mask, alpha=fraction)
            aggregated[key] = torch.where(denominator > 0.0, numerator / denominator.clamp_min(1e-12), initial)
    return aggregated


class InstrumentedFLSimulator(FLSimulator):
    """FLSimulator extension with reviewer-facing efficiency instrumentation.

    The original simulator stays untouched. Sequential client execution is
    deliberate: the local-loss credit signal is captured from the shared scratch
    model after each selected client update.
    """

    def __init__(
        self,
        config: SimConfig,
        *,
        grid_carbon_g_per_kwh: float = 475.0,
        credit_batches: int = 1,
        report_accuracy_target: float | None = None,
        stop_on_accuracy_target: bool = False,
        cifar10_augment: bool = False,
        lr_scheduler: str | None = None,
        lr_warmup_rounds: int = 0,
        local_optimizer: str = "SGD",
        local_momentum: float = 0.9,
        local_weight_decay: float = 1e-4,
        model_initialization: str = "pytorch_default",
        scratch_root: Path | str | None = None,
    ):
        if config.parallel_clients != 0:
            raise ValueError("MAML-Select experiments require parallel_clients=0 for local credit capture.")
        if scratch_root is not None:
            from csfl_simulator.core import utils as core_utils
            core_utils.ART_ROOT = Path(scratch_root)
        super().__init__(config)
        self.grid_carbon_g_per_kwh = float(grid_carbon_g_per_kwh)
        self.credit_batches = max(1, int(credit_batches))
        self.report_accuracy_target = (
            float(report_accuracy_target) if report_accuracy_target is not None else None
        )
        self.stop_on_accuracy_target = bool(stop_on_accuracy_target)
        self.cifar10_augment = bool(cifar10_augment)
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_rounds = max(0, int(lr_warmup_rounds))
        self.local_optimizer = str(local_optimizer).upper()
        if self.local_optimizer != "SGD":
            raise ValueError(f"Unsupported local optimizer: {local_optimizer}. Expected SGD.")
        self.local_momentum = float(local_momentum)
        self.local_weight_decay = float(local_weight_decay)
        self.model_initialization = str(model_initialization)

    def _get_lr_for_round(self, round_idx: int) -> float:
        """Compute learning rate for the given round with optional scheduling."""
        base_lr = float(self.cfg.lr)
        if not self.lr_scheduler:
            return base_lr
        # Warmup phase: linear warmup from lr/10 to lr
        if round_idx < self.lr_warmup_rounds and self.lr_warmup_rounds > 0:
            return base_lr * (0.1 + 0.9 * round_idx / self.lr_warmup_rounds)
        # Cosine annealing after warmup
        if self.lr_scheduler == "cosine":
            effective_round = round_idx - self.lr_warmup_rounds
            effective_total = max(1, int(self.cfg.rounds) - self.lr_warmup_rounds)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * effective_round / effective_total))
        return base_lr

    def setup(self) -> None:
        super().setup()
        # Apply CIFAR-10 data augmentation if enabled
        if self.cifar10_augment and self.train_ds is not None:
            try:
                import torchvision.transforms as T
                augment_transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ])
                # Wrap the existing dataset with augmentation
                # Only if dataset has raw images (not already tensor-ized)
                if hasattr(self.train_ds, 'transform'):
                    self.train_ds.transform = augment_transform
                    # Rebuild client loaders to use augmented data
                    from csfl_simulator.core import datasets as dset
                    for cid in range(self.cfg.total_clients):
                        idxs = self._partition_mapping[cid]
                        self.client_loaders[cid] = dset.make_loaders_from_indices(
                            self.train_ds, idxs, batch_size=self.cfg.batch_size
                        )
                    print("[info] CIFAR-10 data augmentation enabled (RandomCrop+RandomHorizontalFlip)")
            except Exception as e:
                print(f"[warn] Failed to apply CIFAR-10 augmentation: {e}")

        # Device-level energy is an explicit modeled proxy used for per-round
        # attribution. Hardware energy is collected independently by CodeCarbon.
        tier_power_watts = {0: 4.0, 1: 7.0, 2: 12.0}
        tier_capacity_wh = {0: 18.0, 1: 30.0, 2: 48.0}
        tier_compute_speed = {0: 1.0, 1: 2.0, 2: 4.0}
        tier_order = [client.id for client in self.clients]
        random.Random(self.cfg.seed + 1701).shuffle(tier_order)
        first_cut = int(round(0.20 * len(tier_order)))
        second_cut = first_cut + int(round(0.50 * len(tier_order)))
        tier_by_client = {
            cid: 0 if index < first_cut else (1 if index < second_cut else 2)
            for index, cid in enumerate(tier_order)
        }
        for client in self.clients:
            tier = tier_by_client[client.id]
            client.tier = tier
            client.compute_speed = tier_compute_speed[tier]
            power_watts = tier_power_watts.get(tier, 7.0)
            client.meta["device_power_watts"] = power_watts
            client.energy_rate = power_watts / 3600.0
            client.battery_capacity = tier_capacity_wh.get(tier, 30.0)
            client.region_carbon_g_per_kwh = self.grid_carbon_g_per_kwh

    def _local_train(self, cid: int) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """Train one client and derive local credit without extra inference passes."""
        loader = self.client_loaders[cid]
        if self._scratch_model is None:
            self._scratch_model = deepcopy(self.model).to(self.device)
        model = self._scratch_model
        model.load_state_dict(self.model.state_dict())
        model.train()
        current_lr = self._get_lr_for_round(getattr(self, '_current_round_idx', 0))
        optimizer = optim.SGD(
            model.parameters(),
            lr=current_lr,
            momentum=self.local_momentum,
            weight_decay=self.local_weight_decay,
        )
        first_losses: List[torch.Tensor] = []
        tail_losses = deque(maxlen=self.credit_batches)
        last_loss: torch.Tensor | None = None
        last_grad_norm = 0.0
        for _ in range(self.cfg.local_epochs):
            for batch_index, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                # Keep scalar feedback on-device until the client finishes. Calling
                # item() for every batch serializes MPS execution and dominates the
                # ResNet18 runtime on Apple Silicon.
                current_loss = loss.detach()
                if len(first_losses) < self.credit_batches:
                    first_losses.append(current_loss)
                tail_losses.append(current_loss)
                if self.cfg.dp_clip_norm and self.cfg.dp_clip_norm > 0:
                    try:
                        from csfl_simulator.core.dp import clip_gradients

                        clip_gradients(model, float(self.cfg.dp_clip_norm))
                    except Exception:
                        pass
                optimizer.step()
                last_loss = current_loss
                if self.cfg.smoke_test_mode and batch_index > 1:
                    break
        if self.cfg.track_grad_norm:
            gradients = [
                parameter.grad.detach().norm(2)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if gradients:
                last_grad_norm = float(torch.linalg.vector_norm(torch.stack(gradients)).item())
        with torch.no_grad():
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        before = float(torch.stack(first_losses).mean().item()) if first_losses else 0.0
        after = float(torch.stack(list(tail_losses)).mean().item()) if tail_losses else 0.0
        final_loss = float(last_loss.item()) if last_loss is not None else 0.0
        client = self.clients[cid]
        client.meta["last_local_loss_before"] = before
        client.meta["last_local_loss_after"] = after
        client.meta["last_local_loss_reduction"] = before - after
        return state, len(loader.dataset), final_loss, (last_grad_norm if self.cfg.track_grad_norm else 0.0)

    def _simulate_round_environment(self, round_idx: int) -> None:
        """Apply the shared environment model with the manuscript's epoch factor."""
        simulate_round_env(self.clients, {}, round_idx)
        epochs = max(1, int(self.cfg.local_epochs))
        for client in self.clients:
            compute_seconds = epochs * float(client.data_size) / max(1e-6, float(client.compute_speed))
            network_seconds = float(client.data_size) / 1000.0 / max(1e-6, float(client.channel_quality))
            client.estimated_duration = compute_seconds + network_seconds
            client.estimated_energy = float(client.energy_rate) * client.estimated_duration

    def _selection_budget(self, method_key: str) -> float | None:
        if self.cfg.time_budget is not None:
            return float(self.cfg.time_budget)
        if method_key != "system_aware.fedcs":
            return None
        tier_two = [
            float(client.estimated_duration)
            for client in self.clients
            if int(client.tier if client.tier is not None else -1) == 1
        ]
        return self.cfg.clients_per_round * (float(np.mean(tier_two)) if tier_two else 0.0)

    def _evaluate(self, round_idx: int) -> Tuple[Dict[str, float], bool]:
        should_evaluate = (
            round_idx == self.cfg.rounds - 1
            or int(self.cfg.eval_every) <= 1
            or round_idx % int(self.cfg.eval_every) == 0
        )
        if should_evaluate:
            self._latest_evaluation = eval_model(self.model, self.test_loader, self.device)
        return dict(self._latest_evaluation), should_evaluate

    def _criticalfl_post_train(self, ids: Sequence[int], base_cohort_size: int) -> bool:
        state = self.history["state"].setdefault(
            CRITICALFL_STATE_KEY,
            {"previous_fgn": None, "cohort_size": int(base_cohort_size), "critical_rounds": 0},
        )
        weights = [float(self.clients[cid].data_size) for cid in ids]
        norms = [float(self.clients[cid].grad_norm or 0.0) for cid in ids]
        total_weight = float(sum(weights))
        fgn = float(sum(weight * norm for weight, norm in zip(weights, norms)) / total_weight) if total_weight else 0.0
        previous_fgn = state.get("previous_fgn")
        relative_change = 0.0
        if previous_fgn is not None and abs(float(previous_fgn)) > 1e-12:
            relative_change = (fgn - float(previous_fgn)) / abs(float(previous_fgn))
        critical = bool(previous_fgn is not None and relative_change >= float(state.get("delta", 0.01)))
        if critical:
            state["cohort_size"] = min(
                len(self.clients),
                max(1, int(round(float(state.get("growth_factor", 2.0)) * len(ids)))),
            )
            state["critical_rounds"] = int(state.get("critical_rounds", 0)) + 1
        else:
            state["cohort_size"] = max(1, int(math.ceil(len(ids) / 2.0)), int(math.ceil(base_cohort_size / 2.0)))
        state.update({"previous_fgn": fgn, "relative_fgn_change": relative_change, "critical": critical})
        return critical

    def run(self, method_key: str = "baseline.fedavg", on_progress=None, is_cancelled=None) -> Dict[str, Any]:
        if not hasattr(self, "train_ds") or self.train_ds is None:
            self.setup()
        self.model.load_state_dict(self._initial_state_dict)
        self.history = {"state": {}, "selected": []}
        set_seed(self.cfg.seed, deterministic=True)
        self._scratch_model = None
        for client in self.clients:
            client.last_loss = 0.0
            client.grad_norm = 0.0
            client.participation_count = 0
            client.last_selected_round = -1
            for key in ("last_local_loss_before", "last_local_loss_after", "last_local_loss_reduction"):
                client.meta.pop(key, None)

        self._latest_evaluation = eval_model(self.model, self.test_loader, self.device)
        base = dict(self._latest_evaluation)
        base.update(
            {
                "round": -1,
                "evaluated": True,
                "cum_time": 0.0,
                "cum_modelled_energy_wh": 0.0,
                "cum_modelled_carbon_g": 0.0,
                "cum_training_tflops": 0.0,
                "cum_comm_mb": 0.0,
            }
        )
        metrics: List[Dict[str, Any]] = [base]
        cumulative_time = 0.0
        cumulative_energy_wh = 0.0
        cumulative_flops = 0.0
        cumulative_comm_mb = 0.0
        previous_accuracy = float(base["accuracy"])
        stopped_early = False

        for round_idx in range(int(self.cfg.rounds)):
            if is_cancelled and is_cancelled():
                stopped_early = True
                break
            self._current_round_idx = round_idx
            self._simulate_round_environment(round_idx)
            start = time.perf_counter()
            ids, scores, state = self.registry.invoke(
                method_key,
                round_idx,
                self.cfg.clients_per_round,
                self.clients,
                self.history,
                random,
                self._selection_budget(method_key),
                self.device,
                energy_budget=self.cfg.energy_budget,
                bytes_budget=self.cfg.bytes_budget,
            )
            selection_seconds = float(time.perf_counter() - start)
            ids = [int(cid) for cid in ids]
            self.history["selected"].append(ids)
            if state:
                self.history["state"].update(state)
            for cid in ids:
                self.clients[cid].last_selected_round = round_idx

            updates, weights = [], []
            for cid in ids:
                state_dict, sample_count, loss, grad_norm = self._local_train(cid)
                client = self.clients[cid]
                client.last_loss = float(loss)
                client.grad_norm = float(grad_norm)
                client.participation_count += 1
                updates.append(state_dict)
                weights.append(sample_count)
            criticalfl_sparse_round = False
            if method_key == "research.criticalfl":
                criticalfl_sparse_round = self._criticalfl_post_train(ids, self.cfg.clients_per_round)
            if updates:
                if criticalfl_sparse_round:
                    criticalfl_state = self.history["state"][CRITICALFL_STATE_KEY]
                    new_state = _criticalfl_sparse_fedavg(
                        updates,
                        weights,
                        self.model.state_dict(),
                        keep_fraction=float(criticalfl_state.get("sparse_fraction", 0.20)),
                    )
                else:
                    new_state = fedavg(updates, weights, keep_on_device=self.device.startswith("cuda"))
                self.model.load_state_dict(new_state)

            evaluation, evaluated = self._evaluate(round_idx)
            durations = [float(self.clients[cid].estimated_duration or 0.0) for cid in ids]
            energies_wh = [float(self.clients[cid].estimated_energy or 0.0) for cid in ids]
            round_time = max(durations) if durations else 0.0
            round_energy_wh = float(sum(energies_wh))
            round_flops = sum(
                3.0 * float(self.model_macs_per_sample) * self.cfg.local_epochs * self.clients[cid].data_size
                for cid in ids
            )
            model_size_mb = float(self.model_params_count) * 4.0 / (1024.0 * 1024.0)
            uplink_fraction = (
                float(self.history["state"][CRITICALFL_STATE_KEY].get("sparse_fraction", 0.20))
                if criticalfl_sparse_round
                else 1.0
            )
            round_comm_mb = (1.0 + uplink_fraction) * len(ids) * model_size_mb
            cumulative_time += round_time
            cumulative_energy_wh += round_energy_wh
            cumulative_flops += round_flops
            cumulative_comm_mb += round_comm_mb

            counts = [float(client.participation_count) for client in self.clients]
            tier_totals = {
                tier: sum(float(client.participation_count) for client in self.clients if int(client.tier or 0) == tier)
                for tier in (0, 1, 2)
            }
            total_participation = max(1.0, float(sum(tier_totals.values())))
            local_credits = [
                float(self.clients[cid].meta.get("last_local_loss_reduction", 0.0))
                for cid in ids
            ]
            row: Dict[str, Any] = dict(evaluation)
            row.update(
                {
                    "round": round_idx,
                    "evaluated": evaluated,
                    "selected_clients": ids,
                    "selected_count": len(ids),
                    "selection_seconds": selection_seconds,
                    "selection_overhead_seconds": selection_seconds,
                    "round_time": round_time,
                    "cum_time": cumulative_time,
                    "round_modelled_energy_wh": round_energy_wh,
                    "cum_modelled_energy_wh": cumulative_energy_wh,
                    "round_modelled_carbon_g": round_energy_wh / 1000.0 * self.grid_carbon_g_per_kwh,
                    "cum_modelled_carbon_g": cumulative_energy_wh / 1000.0 * self.grid_carbon_g_per_kwh,
                    "training_tflops": round_flops / 1e12,
                    "cum_training_tflops": cumulative_flops / 1e12,
                    "round_comm_mb": round_comm_mb,
                    "cum_comm_mb": cumulative_comm_mb,
                    "fairness_variance": float(np.var(counts)),
                    "fairness_gini": _gini(counts),
                    "fairness_jain": _jain(counts),
                    "utilization_entropy": _entropy(counts),
                    "participation_coverage_ratio": float(sum(count > 0.0 for count in counts) / max(1, len(counts))),
                    "tier_0_selection_rate": tier_totals[0] / total_participation,
                    "tier_1_selection_rate": tier_totals[1] / total_participation,
                    "tier_2_selection_rate": tier_totals[2] / total_participation,
                    "label_coverage_ratio": _label_coverage(self.clients, ids, self.num_classes),
                    "mean_local_loss_reduction": float(np.mean(local_credits)) if local_credits else 0.0,
                    "accuracy_delta": float(evaluation["accuracy"]) - previous_accuracy if evaluated else 0.0,
                    "criticalfl_sparse_round": criticalfl_sparse_round,
                }
            )
            v2_state = self.history["state"].get("research_maml_select_v2_state")
            if isinstance(v2_state, dict):
                components = v2_state.get("last_components", {}) or {}
                for key, value in components.items():
                    row[f"maml_v2_{key}"] = float(value)
                row["maml_v2_selection_mode"] = str(v2_state.get("last_selection_mode", ""))
                row["maml_v2_adjusted_cost_mean"] = float(v2_state.get("last_adjusted_cost_mean", 0.0))
                row["maml_v2_overuse_mean"] = float(v2_state.get("last_overuse_mean", 0.0))
                bucket_counts = v2_state.get("last_bucket_counts", {}) or {}
                for key, value in bucket_counts.items():
                    row[f"maml_v2_bucket_{key}"] = int(value)
            self.history["state"]["last_reward"] = row["accuracy_delta"]
            if evaluated:
                previous_accuracy = float(evaluation["accuracy"])
            metrics.append(row)
            if on_progress:
                on_progress(round_idx, {"metrics": row, "selected": ids})
            if (
                evaluated
                and self.stop_on_accuracy_target
                and self.report_accuracy_target is not None
                and float(row["accuracy"]) >= self.report_accuracy_target
            ):
                break
            if round_idx % 10 == 0:
                cleanup_memory(force_cuda_empty=False, verbose=False)

        cleanup_memory(force_cuda_empty=True, verbose=False)
        final = metrics[-1]
        base_accuracy = float(metrics[0]["accuracy"])
        final_accuracy = float(final["accuracy"])
        improvement = max(0.0, final_accuracy - base_accuracy)
        final["time_to_80pct_final"] = _first_time_to(metrics, base_accuracy + 0.8 * improvement)
        final["mean_cohort_size"] = float(np.mean([len(ids) for ids in self.history["selected"]]))
        if self.report_accuracy_target is not None:
            final["report_accuracy_target"] = self.report_accuracy_target
            final["target_reached"] = bool(final_accuracy >= self.report_accuracy_target)
            final["rounds_to_target"] = _first_metric_to(metrics, self.report_accuracy_target, "round") + 1.0
            final["time_to_target"] = _first_metric_to(metrics, self.report_accuracy_target, "cum_time")
            final["training_tflops_to_target"] = _first_metric_to(
                metrics, self.report_accuracy_target, "cum_training_tflops"
            )
            final["modelled_energy_wh_to_target"] = _first_metric_to(
                metrics, self.report_accuracy_target, "cum_modelled_energy_wh"
            )
            final["comm_mb_to_target"] = _first_metric_to(metrics, self.report_accuracy_target, "cum_comm_mb")
        save_json({"metrics": metrics}, Path(self.run_dir) / "metrics.json")
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "metrics": metrics,
            "config": asdict(self.cfg),
            "device": self.device,
            "stopped_early": stopped_early,
            "stop_on_accuracy_target": self.stop_on_accuracy_target,
            "report_accuracy_target": self.report_accuracy_target,
            "rounds_completed": len(self.history["selected"]),
            "method": method_key,
            "history": {"selected": list(self.history["selected"])},
            "participation_counts": [int(client.participation_count) for client in self.clients],
            "modelled_energy_assumptions": {
                "unit": "Wh",
                "duration_unit": "modeled seconds",
                "tier_power_watts": {"0": 4.0, "1": 7.0, "2": 12.0},
                "carbon_note": "Per-round carbon is estimated from modeled Wh and the declared grid intensity.",
                "grid_intensity_g_per_kwh": self.grid_carbon_g_per_kwh,
            },
            "training_protocol": {
                "optimizer": self.local_optimizer,
                "base_lr": float(self.cfg.lr),
                "momentum": self.local_momentum,
                "weight_decay": self.local_weight_decay,
                "scheduler": self.lr_scheduler or "constant",
                "warmup_rounds": self.lr_warmup_rounds,
                "model_initialization": self.model_initialization,
            },
        }
