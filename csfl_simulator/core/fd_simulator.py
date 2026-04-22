"""Federated Distillation (FD) Simulator.

Implements the FedTSKD and FedTSKD-G algorithms from:
  Mu et al., "Federated Distillation in Massive MIMO Networks:
  Dynamic Training Convergence Analysis and Communication
  Channel-Aware Learning", IEEE TCCN, vol. 10, no. 4, Aug 2024.

The selection interface is identical to FL — all 47+ existing client
selection methods work unchanged.

NOTE: This implementation uses constant learning rates. Theorem 1 of the
paper assumes eta_t = beta_0 / (t + beta_1) decay for its O(1/t) convergence
bound. Empirical convergence rates may therefore differ from the theoretical
O(1/t) predicted in §IV; in particular, late-training behaviour is controlled
by Adam's running-moment scaling rather than by the theorem's step-size
decay. Swap in a time-decayed LR schedule if the theorem bound is being
empirically verified.

Reproducibility: the default ``performance_mode=True`` in ``SimConfig`` enables
``torch.backends.cudnn.benchmark`` so cuDNN picks the fastest kernel for each
input shape. This preserves **seed-level** reproducibility (the same seed yields
the same high-level results across runs on the same hardware) but NOT **bit-for-bit**
reproducibility. Set ``performance_mode=False`` (CLI: ``--no-performance-mode``) if
exact tensor-level reproducibility is required.
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


def _unwrap_compiled(m: nn.Module) -> nn.Module:
    """Return the underlying nn.Module, peeling off torch.compile's OptimizedModule.

    torch.compile wraps a module and prefixes every state_dict key with
    "_orig_mod.". Any save/load path that must stay agnostic to whether compile
    was applied routes through this helper so the key set stays consistent.
    Returns the argument unchanged for plain modules.
    """
    return getattr(m, "_orig_mod", m)


class FDSimulator:
    """Federated Distillation simulator following Algorithms 1 & 2 of the paper.

    Unlike FL (which exchanges model weights), FD exchanges **logits** on a
    shared public dataset.  Each client maintains its own model (supporting
    heterogeneous architectures) and learns via KL-divergence distillation
    from aggregated logits received from the server.
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        perf = bool(getattr(config, "performance_mode", False))
        set_seed(self.cfg.seed, deterministic=(not perf), performance_mode=perf)
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

        init_system_state(self.clients, {
            "paradigm": "fd",
            "channel_threshold": cfg.channel_threshold,
        })

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
                combining=cfg.combining_scheme,
            )

        # --- Save initial states for fair multi-method comparisons ---
        # Always target the underlying nn.Module, not the torch.compile wrapper,
        # so keys don't gain/lose the "_orig_mod." prefix if tier 3 compile is
        # applied later. _unwrap_compiled() is the inverse of torch.compile()
        # and a no-op on uncompiled modules.
        with torch.no_grad():
            self._initial_client_states = {
                cid: {k: v.clone() for k, v in _unwrap_compiled(m).state_dict().items()}
                for cid, m in self.client_models.items()
            }
            self._initial_server_state = {
                k: v.clone() for k, v in _unwrap_compiled(self.server_model).state_dict().items()
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
        # When the cache holds CIFAR tensors, augmentation is re-applied live on GPU
        # in _local_train; the cache itself is deliberately un-augmented so each
        # epoch sees fresh crops/flips instead of frozen-at-setup samples.
        self._cache_augments_on_gpu = False
        is_cifar = self.cfg.dataset.lower() in ("cifar10", "cifar-10", "cifar100", "cifar-100")
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
                    if is_cifar:
                        # Swap to a no-augment transform so cached tensors are clean.
                        from torchvision import transforms as _tvtfm
                        if self.cfg.dataset.lower() in ("cifar10", "cifar-10"):
                            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                        else:
                            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                        no_aug_tfm = _tvtfm.Compose([
                            _tvtfm.ToTensor(),
                            _tvtfm.Normalize(mean, std),
                        ])
                        original_tfm = getattr(train_ds, "transform", None)
                        train_ds.transform = no_aug_tfm
                        try:
                            for cid in range(cfg.total_clients):
                                xs, ys = [], []
                                for x, y in self.client_loaders[cid]:
                                    xs.append(x)
                                    ys.append(y)
                                self._client_data_cache[cid] = (
                                    torch.cat(xs, dim=0).to(self.device),
                                    torch.cat(ys, dim=0).to(self.device),
                                )
                            self._cache_augments_on_gpu = True
                        finally:
                            train_ds.transform = original_tfm
                    else:
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
                self._cache_augments_on_gpu = False

        # --- Performance: AMP (mixed precision) ---
        # bf16 on Ampere+ has the same dynamic range as fp32, so no GradScaler is
        # required. This eliminates the per-optimizer-step CPU<->GPU sync that
        # serialized all "parallel" clients through a single scaler object. Falls
        # back to fp16 + GradScaler on pre-Ampere so the code still works on
        # older hardware.
        self._use_amp = cfg.use_amp and self.device.startswith("cuda")
        self._amp_dtype = torch.float32  # sentinel when AMP is off
        self._scaler = None
        if self._use_amp:
            try:
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 8:
                    self._amp_dtype = torch.bfloat16
                else:
                    self._amp_dtype = torch.float16
                    self._scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                self._amp_dtype = torch.float16
                self._scaler = torch.amp.GradScaler("cuda", enabled=True)

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
                # Cap at 4 instead of 8. With heterogeneous architectures and per-client
                # model instances, the PyTorch caching-allocator global lock makes 8-wide
                # dispatch slower than 4-wide on a single GPU. Override via --parallel-clients.
                auto = max(1, min(usable // max(per_client, 1), 4))
                if cfg.parallel_clients == -1:
                    self._max_parallel = auto
                elif cfg.parallel_clients > 0:
                    self._max_parallel = cfg.parallel_clients
                else:
                    self._max_parallel = auto
            except Exception:
                self._max_parallel = 2

        # Pre-allocate CUDA streams once. Allocating per-round added measurable
        # overhead (200 rounds * up to 4 streams = 800 stream allocs).
        self._client_streams: List[torch.cuda.Stream] = []
        if self.device.startswith("cuda"):
            try:
                self._client_streams = [torch.cuda.Stream() for _ in range(self._max_parallel)]
            except Exception:
                self._client_streams = []

        # --- Tier 3 optional: torch.compile ---
        if cfg.use_torch_compile and self.device.startswith("cuda"):
            self._maybe_compile_models()

        # --- Tier 3 optional: channels_last memory format ---
        if cfg.channels_last and self.device.startswith("cuda"):
            self._maybe_apply_channels_last()

        # --- Tier 3 optional: fixed eval subsample indices ---
        self._eval_subsample_indices: Optional[torch.Tensor] = None
        self._eval_full_override: bool = False
        if cfg.eval_subsample > 0 and self._test_x_cache is not None:
            n_test = self._test_x_cache.size(0)
            k = min(int(cfg.eval_subsample), n_test)
            g = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
            perm = torch.randperm(n_test, generator=g)[:k]
            self._eval_subsample_indices = perm.to(self.device)

        # --- Tier 1.2: pre-warm cuDNN kernels ---
        # The first forward pass on a new input shape triggers cuDNN algorithm selection
        # (1-5s stall per model). With heterogeneous client architectures this stall would
        # hit on every architecture the first time it's used. Fire one dummy forward per
        # unique architecture and on the server model so the tax is paid at setup time.
        if self.device.startswith("cuda"):
            try:
                warm_sizes = sorted({
                    int(self.cfg.batch_size),
                    int(self.cfg.distillation_batch_size),
                    512,
                })
                ref_shape = None
                ref_dtype = torch.float32
                if self._public_x_cache is not None and self._public_x_cache.size(0) > 0:
                    ref_shape = tuple(self._public_x_cache.shape[1:])
                    ref_dtype = self._public_x_cache.dtype
                else:
                    sample_x, _ = train_ds[0]
                    if isinstance(sample_x, torch.Tensor):
                        ref_shape = tuple(sample_x.shape)
                        ref_dtype = sample_x.dtype
                if ref_shape is not None:
                    seen_ids: set = set()
                    warmed = 0
                    amp_dtype = getattr(self, "_amp_dtype", torch.float32)
                    with torch.inference_mode():
                        for cid, m in self.client_models.items():
                            if id(m) in seen_ids:
                                continue
                            seen_ids.add(id(m))
                            try:
                                m.eval()
                                for bs in warm_sizes:
                                    dummy = torch.zeros((bs,) + ref_shape, device=self.device, dtype=ref_dtype)
                                    if self._use_amp:
                                        with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                                            _ = m(dummy)
                                    else:
                                        _ = m(dummy)
                                    del dummy
                                warmed += 1
                            except Exception:
                                pass
                        try:
                            self.server_model.eval()
                            for bs in warm_sizes:
                                dummy = torch.zeros((bs,) + ref_shape, device=self.device, dtype=ref_dtype)
                                if self._use_amp:
                                    with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                                        _ = self.server_model(dummy)
                                else:
                                    _ = self.server_model(dummy)
                                del dummy
                            warmed += 1
                        except Exception:
                            pass
                    torch.cuda.synchronize()
                    print(f"[FD setup] pre-warmed cuDNN kernels on {warmed} model(s) across {len(warm_sizes)} batch sizes.", flush=True)
            except Exception:
                pass

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
    # Tier 3 optional performance helpers
    # ------------------------------------------------------------------

    def _maybe_compile_models(self):
        """Wrap each client model and the server model with torch.compile (tier 3.1).

        Off by default because compilation can fail on unusual layers. Each failure
        is caught and the uncompiled model is kept instead — the simulator stays
        functional even if only some models compile. Requires PyTorch >= 2.1.

        Mode choice — uses the default Inductor mode, NOT ``mode="reduce-overhead"``.
        The reduce-overhead mode uses CUDA graphs, which reuse a single output buffer
        across successive calls of the same graph. The FD loop calls multiple compiled
        models back-to-back (in _evaluate_clients_cached and the client phase) and
        needs to read each output after later calls, which the graph-captured buffer
        aliasing breaks with a "accessing tensor output of CUDAGraphs that has been
        overwritten" RuntimeError. The default mode still gives Inductor kernel
        fusion without the CUDA-graph aliasing hazard.
        """
        if not hasattr(torch, "compile"):
            print("[FD setup] torch.compile not available (requires PyTorch >= 2.1) — skipping.", flush=True)
            return
        n_compiled = 0
        n_failed = 0
        # Cache compile results by model identity to avoid recompiling shared arch instances.
        compiled_cache: Dict[int, nn.Module] = {}
        for cid, m in self.client_models.items():
            key = id(m)
            if key in compiled_cache:
                self.client_models[cid] = compiled_cache[key]
                continue
            try:
                cm = torch.compile(m)  # default mode, see docstring
                self.client_models[cid] = cm
                compiled_cache[key] = cm
                n_compiled += 1
            except Exception as e:
                n_failed += 1
                print(f"[FD setup] torch.compile failed for client {cid}: {e}", flush=True)
        try:
            self.server_model = torch.compile(self.server_model)  # default mode
            n_compiled += 1
        except Exception as e:
            n_failed += 1
            print(f"[FD setup] torch.compile failed for server: {e}", flush=True)
        print(f"[FD setup] torch.compile: {n_compiled} compiled, {n_failed} failed.", flush=True)

    def _maybe_apply_channels_last(self):
        """Convert CNN models and cached tensors to channels_last memory format (tier 3.2).

        Off by default; 10-20% speedup on Ampere+ GPUs for conv-heavy models.  Only
        applied when the cached public tensor is 4D (N,C,H,W).
        """
        cache = self._public_x_cache
        if cache is None or cache.dim() != 4:
            print("[FD setup] channels_last skipped: no 4D public tensor cache.", flush=True)
            return
        try:
            # Convert cached tensors
            self._public_x_cache = self._public_x_cache.contiguous(memory_format=torch.channels_last)
            if self._test_x_cache is not None and self._test_x_cache.dim() == 4:
                self._test_x_cache = self._test_x_cache.contiguous(memory_format=torch.channels_last)
            for cid, (xs, ys) in list(self._client_data_cache.items()):
                if xs.dim() == 4:
                    self._client_data_cache[cid] = (
                        xs.contiguous(memory_format=torch.channels_last), ys,
                    )
            # Convert models (skip already-converted shared instances)
            seen = set()
            for cid, m in self.client_models.items():
                if id(m) in seen:
                    continue
                seen.add(id(m))
                try:
                    m.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            try:
                self.server_model.to(memory_format=torch.channels_last)
            except Exception:
                pass
            print("[FD setup] channels_last applied.", flush=True)
        except Exception as e:
            print(f"[FD setup] channels_last failed: {e}", flush=True)

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
                self._client_train_opts[cid] = optim.Adam(
                    model.parameters(),
                    lr=self.cfg.distillation_lr,
                    fused=self.device.startswith("cuda"),
                )
            else:
                self._client_train_opts[cid] = optim.SGD(model.parameters(), lr=self.cfg.lr)
        return self._client_train_opts[cid]

    def _get_distill_opt(self, cid: int) -> optim.Optimizer:
        """Get or create the distillation optimizer for a client."""
        if cid not in self._client_distill_opts:
            self._client_distill_opts[cid] = optim.Adam(
                self.client_models[cid].parameters(), lr=self.cfg.distillation_lr,
                fused=self.device.startswith("cuda"),
            )
        return self._client_distill_opts[cid]

    def _get_server_distill_opt(self) -> optim.Optimizer:
        """Get or create the server distillation optimizer."""
        if self._server_distill_opt is None:
            self._server_distill_opt = optim.Adam(
                self.server_model.parameters(), lr=self.cfg.distillation_lr,
                fused=self.device.startswith("cuda"),
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

        grad_norm = 0.0
        use_amp = self._use_amp
        scaler = self._scaler
        # Defer loss->float conversion until the end of the call: one GPU->CPU sync
        # instead of one per SGD step. See FD_speed_optimization_notes.md tier 1.7.
        loss_buffer: Optional[torch.Tensor] = None

        # GPU-side augmentation for the un-augmented CIFAR cache built in setup().
        # Reflection-pad-4 + random 32x32 crop + 50% horizontal flip; matches the
        # torchvision transform used by the DataLoader fallback path.
        def _augment_gpu_batch(x: torch.Tensor) -> torch.Tensor:
            if not getattr(self, "_cache_augments_on_gpu", False):
                return x
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            h = torch.randint(0, 9, (1,), device=x.device).item()
            w = torch.randint(0, 9, (1,), device=x.device).item()
            x = x[:, :, h:h + 32, w:w + 32]
            if torch.rand((), device=x.device).item() < 0.5:
                x = torch.flip(x, dims=[-1])
            return x

        # Fast path: GPU-cached training data (no DataLoader overhead)
        if cid in self._client_data_cache:
            x_all, y_all = self._client_data_cache[cid]
            n = x_all.size(0)
            bs = cfg.batch_size
            for step in range(num_steps):
                idx = (step * bs) % n
                x = _augment_gpu_batch(x_all[idx:idx + bs])
                y = y_all[idx:idx + bs]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
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
                loss_buffer = loss.detach()
                if cfg.smoke_test_mode and step > 0:
                    last_loss = float(loss_buffer.item()) if loss_buffer is not None else 0.0
                    return last_loss, grad_norm
            last_loss = float(loss_buffer.item()) if loss_buffer is not None else 0.0
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
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
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
                loss_buffer = loss.detach()
                step += 1
                if cfg.smoke_test_mode and step > 1:
                    last_loss = float(loss_buffer.item()) if loss_buffer is not None else 0.0
                    return last_loss, grad_norm
        last_loss = float(loss_buffer.item()) if loss_buffer is not None else 0.0
        return last_loss, grad_norm

    # ------------------------------------------------------------------
    # Inference on public dataset (Eq. 13)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate_logits(self, cid: int) -> torch.Tensor:
        """Run client model on public dataset, returning logits of shape (N_pub, C).

        Chunked by ``cfg.distillation_batch_size`` so the forward passes stay inside
        a size that fits AMP pipelining and matches the batch size used by
        ``_local_distill``/``_server_distill``. On the full 2000-sample public set
        with 500-batch chunks this issues 4 launches per client, each of which can
        overlap with the rest of the client's CUDA stream — see
        FD_speed_optimization_notes.md follow-up pass.
        """
        model = self.client_models[cid]
        model.eval()
        bs = max(1, int(self.cfg.distillation_batch_size))
        if self._public_x_cache is not None:
            parts: List[torch.Tensor] = []
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                for i in range(0, self._public_x_cache.size(0), bs):
                    parts.append(model(self._public_x_cache[i:i + bs]).float())
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        all_logits = []
        for x, *_ in self.public_loader:
            x = x.to(self.device)
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
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
        # Public-set distillation has a distribution shift vs the test set; freezing
        # BN running stats stops them drifting toward public stats. No-op under GN.
        if bool(getattr(cfg, "freeze_bn_on_distill", True)):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        opt = self._get_distill_opt(cid)
        n_batches = 0
        target_logits = target_logits.to(self.device)
        use_amp = self._use_amp
        scaler = self._scaler
        # Precompute softmax targets once per call (tier 1.5): target_logits does not change
        # across epochs/batches, so F.softmax(t/T, dim=-1) only needs to run once.
        with torch.no_grad():
            q_full = F.softmax(target_logits / T, dim=-1)
        # Accumulate per-batch loss as a GPU tensor and convert once at the end (tier 1.7).
        loss_sum: Optional[torch.Tensor] = None

        # Use cached public data if available (avoids CPU->GPU transfer)
        if self._public_x_cache is not None:
            public_x = self._public_x_cache
            for epoch in range(num_epochs):
                for i in range(0, public_x.size(0), cfg.distillation_batch_size):
                    x_batch = public_x[i:i + cfg.distillation_batch_size]
                    q_batch = q_full[i:i + cfg.distillation_batch_size]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
                        log_p = F.log_softmax(model(x_batch) / T, dim=-1)
                        loss = F.kl_div(log_p, q_batch, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    detached = loss.detach()
                    loss_sum = detached if loss_sum is None else loss_sum + detached
                    n_batches += 1
                    if cfg.smoke_test_mode and n_batches > 1:
                        return float(loss_sum.item() / n_batches) if loss_sum is not None else 0.0
        else:
            for epoch in range(num_epochs):
                offset = 0
                for x, *_ in self.public_loader:
                    x = x.to(self.device)
                    bs = x.size(0)
                    q_batch = q_full[offset:offset + bs]
                    offset += bs
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
                        log_p = F.log_softmax(model(x) / T, dim=-1)
                        loss = F.kl_div(log_p, q_batch, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    detached = loss.detach()
                    loss_sum = detached if loss_sum is None else loss_sum + detached
                    n_batches += 1
                    if cfg.smoke_test_mode and n_batches > 1:
                        return float(loss_sum.item() / n_batches) if loss_sum is not None else 0.0

        if loss_sum is None or n_batches == 0:
            return 0.0
        return float(loss_sum.item() / n_batches)

    # ------------------------------------------------------------------
    # Server-side distillation
    # ------------------------------------------------------------------

    def _server_distill(self, aggregated_logits: torch.Tensor, num_epochs: int) -> float:
        """Server distills its own model using aggregated logits from clients."""
        cfg = self.cfg
        T = cfg.temperature
        model = self.server_model
        model.train()
        # Same rationale as _local_distill: freeze BN stats on public-set distill batches.
        if bool(getattr(self.cfg, "freeze_bn_on_distill", True)):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        opt = self._get_server_distill_opt()
        n_batches = 0
        aggregated_logits = aggregated_logits.to(self.device)
        use_amp = self._use_amp
        scaler = self._scaler
        # Precompute softmax targets once per call (tier 1.5): aggregated_logits is fixed
        # across the distillation epochs, so F.softmax(t/T, dim=-1) only needs to run once.
        with torch.no_grad():
            q_full = F.softmax(aggregated_logits / T, dim=-1)
        # Accumulate per-batch loss as a GPU tensor and convert once at the end (tier 1.7).
        loss_sum: Optional[torch.Tensor] = None

        if self._public_x_cache is not None:
            public_x = self._public_x_cache
            for epoch in range(num_epochs):
                for i in range(0, public_x.size(0), cfg.distillation_batch_size):
                    x_batch = public_x[i:i + cfg.distillation_batch_size]
                    q_batch = q_full[i:i + cfg.distillation_batch_size]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
                        log_p = F.log_softmax(model(x_batch) / T, dim=-1)
                        loss = F.kl_div(log_p, q_batch, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    detached = loss.detach()
                    loss_sum = detached if loss_sum is None else loss_sum + detached
                    n_batches += 1
                    if cfg.smoke_test_mode and n_batches > 1:
                        return float(loss_sum.item() / n_batches) if loss_sum is not None else 0.0
        else:
            for epoch in range(num_epochs):
                offset = 0
                for x, *_ in self.public_loader:
                    x = x.to(self.device)
                    bs = x.size(0)
                    q_batch = q_full[offset:offset + bs]
                    offset += bs
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=self._amp_dtype):
                        log_p = F.log_softmax(model(x) / T, dim=-1)
                        loss = F.kl_div(log_p, q_batch, reduction='batchmean') * (T * T)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    detached = loss.detach()
                    loss_sum = detached if loss_sum is None else loss_sum + detached
                    n_batches += 1
                    if cfg.smoke_test_mode and n_batches > 1:
                        return float(loss_sum.item() / n_batches) if loss_sum is not None else 0.0

        if loss_sum is None or n_batches == 0:
            return 0.0
        return float(loss_sum.item() / n_batches)

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

        # --- Guardrail: warn if smoke_test_mode is left on for non-trivial runs ---
        if cfg.smoke_test_mode and cfg.rounds > 20:
            import warnings
            warnings.warn(
                "smoke_test_mode=True with rounds>20 — results will not reflect paper-faithful FD. "
                "This flag is for debugging only.",
                stacklevel=2,
            )

        # --- Reset state for fair multi-method comparison ---
        # Unwrap torch.compile wrappers before load_state_dict so the saved keys
        # (no "_orig_mod." prefix; see _unwrap_compiled) match the target module.
        if hasattr(self, '_initial_client_states'):
            for cid, sd in self._initial_client_states.items():
                _unwrap_compiled(self.client_models[cid]).load_state_dict(sd)
        if hasattr(self, '_initial_server_state'):
            _unwrap_compiled(self.server_model).load_state_dict(self._initial_server_state)
        self.history = {"state": {}, "selected": []}
        for c in self.clients:
            c.last_loss = 0.0
            c.grad_norm = 0.0
            c.participation_count = 0
            c.last_selected_round = -1
        perf = bool(getattr(cfg, "performance_mode", False))
        set_seed(cfg.seed, deterministic=(not perf), performance_mode=perf)
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
            simulate_round_env(self.clients, {
                "paradigm": "fd",
                "channel_threshold": cfg.channel_threshold,
                "static_channel_groups": cfg.static_channel_groups,
            }, rnd)

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
            t_clients_start = time.perf_counter()

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
                    if self._client_streams and len(self._client_streams) >= len(batch_ids):
                        streams = self._client_streams[:len(batch_ids)]
                    else:
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

            t_clients_total = time.perf_counter() - t_clients_start

            # --- Per-client logit statistics (for FD-native selectors) ---
            fd_logit_stats = {}
            logit_entropy_vals = []
            logit_entropy_var_vals = []
            if client_logits and len(client_logits) > 1:
                all_logit_tensors = [client_logits[cid] for cid in ids]
                stacked = torch.stack(all_logit_tensors)  # (K, N_pub, C)
                mean_logits = stacked.mean(dim=0)  # (N_pub, C)
                # Pairwise cosine distances for diversity
                flat_vecs = stacked.view(len(ids), -1)  # (K, N_pub*C)
                norms = flat_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                normed = flat_vecs / norms
                cos_sim_matrix = normed @ normed.t()  # (K, K)
                # Mean pairwise cosine distance (1 - sim, excluding diagonal)
                K_sel = len(ids)
                if K_sel > 1:
                    mask = 1.0 - torch.eye(K_sel, device=cos_sim_matrix.device)
                    logit_cosine_div = ((1.0 - cos_sim_matrix) * mask).sum().item() / (K_sel * (K_sel - 1))
                else:
                    logit_cosine_div = 0.0

                for cid in ids:
                    cl = client_logits[cid]
                    # Cosine similarity to mean
                    cos_sim = F.cosine_similarity(
                        cl.reshape(1, -1), mean_logits.reshape(1, -1)
                    ).item()
                    # Logit entropy per sample
                    probs = F.softmax(cl, dim=-1).clamp(min=1e-10)
                    ent_per_sample = -(probs * probs.log()).sum(dim=-1)  # (N_pub,)
                    entropy_mean = ent_per_sample.mean().item()
                    entropy_var = ent_per_sample.var().item() if cl.shape[0] > 1 else 0.0
                    logit_entropy_vals.append(entropy_mean)
                    logit_entropy_var_vals.append(entropy_var)
                    fd_logit_stats[cid] = {
                        "cosine_to_mean": cos_sim,
                        "entropy_mean": entropy_mean,
                        "entropy_var": entropy_var,
                    }
                data_sizes = {cid: self.clients[cid].data_size for cid in ids}
                total_data = sum(data_sizes.values())
                self.history["state"]["fd_logit_stats"] = fd_logit_stats
                self.history["state"]["fd_logit_rewards"] = {
                    cid: stats["cosine_to_mean"] * (data_sizes[cid] / max(total_data, 1))
                    for cid, stats in fd_logit_stats.items()
                }
            else:
                logit_cosine_div = 0.0

            # Phase 5: Server aggregation
            t_server_start = time.perf_counter()
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

            # Phase 6: Server-side distillation.
            # Paper Eq. 18 introduces omega_D ~ N(0, sigma_D^2 * I) as a small-variance
            # Gaussian perturbation on the server's aggregated logits (representing the
            # residual distillation uncertainty after aggregation). We model it as an
            # additive noise term before server distillation. Set cfg.server_distill_sigma
            # to 0 to disable.
            server_target = aggregated_logits
            if self.channel is not None and cfg.server_distill_sigma > 0:
                server_target = server_target + torch.randn_like(server_target) * cfg.server_distill_sigma
            # Warmup: extra server-distill epochs in the first few rounds to escape the
            # near-uniform-logit fixed point that otherwise wastes 20-30 rounds.
            sd_epochs = (cfg.server_warmup_epochs if rnd < cfg.server_warmup_rounds
                         else cfg.distillation_epochs)
            server_kl = self._server_distill(server_target, sd_epochs)

            # Server predicts on public dataset → logits for next round's downlink.
            # Chunked by cfg.distillation_batch_size to match _generate_logits and keep
            # the forward pass stream-friendly (see follow-up pass in the speed notes).
            self.server_model.eval()
            bs = max(1, int(cfg.distillation_batch_size))
            with torch.inference_mode():
                if self._public_x_cache is not None:
                    server_parts: List[torch.Tensor] = []
                    for i in range(0, self._public_x_cache.size(0), bs):
                        server_parts.append(self.server_model(self._public_x_cache[i:i + bs]))
                    server_logits = (
                        server_parts[0] if len(server_parts) == 1
                        else torch.cat(server_parts, dim=0)
                    )
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

            # Expose server's per-class confidence on the public dataset so that
            # coverage-aware selectors (e.g. fd_native.prism_fd) can target the
            # classes the server is currently weakest at. This is a cheap signal
            # random selection cannot use.
            # Shape: List[float] of length num_classes; each entry is the mean
            # softmax probability the server assigns to that class across the
            # public samples. Low value => server rarely predicts that class =>
            # more gain from selecting clients that hold that class.
            with torch.inference_mode():
                base = (server_logits[:, :self.num_classes]
                        if cfg.group_based else server_logits)
                class_mass = F.softmax(base, dim=-1).mean(dim=0).detach().cpu().tolist()
            self.history["state"]["server_class_confidence"] = class_mass

            t_server_total = time.perf_counter() - t_server_start

            # Phase 7: Evaluation
            compute_time = max(
                (self.clients[cid].estimated_duration for cid in ids),
                default=0.0,
            )

            # Communication metrics (FD overhead)
            K_r_current = self._compute_dynamic_steps(rnd, max((self.clients[cid].data_size for cid in ids), default=1))
            n_pub = len(self.public_ds)
            C = self.num_classes
            quant_bytes = cfg.quantization_bits / 8.0
            # Uplink: K clients send logits; Downlink: server broadcasts logits
            logit_comm_kb = (len(ids) * n_pub * C * quant_bytes + n_pub * C * quant_bytes) / 1024.0
            # FL equivalent: 2 * K * model_size_bytes.
            # Paper Fig. 10 reports FL with 16-bit quantization (2 bytes/param), not fp32.
            # Using 4 bytes/param here would inflate FL comm by 2x, overstating FD's savings.
            fl_model_bytes = self.model_params_count * 2.0  # 16-bit quantization per paper Fig. 10
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
            is_final = (rnd == cfg.rounds - 1)
            do_eval = (rnd % eval_every == 0) or is_final or (rnd == 0)
            t_eval_start = time.perf_counter()
            if do_eval:
                # Evaluate ALL clients on eval rounds (paper's per-user avg metric).
                # Previously passed sample_ids=ids which only evaluated selected clients,
                # biasing `client_accuracy_avg` toward the selection policy.
                # Force full test set on the final round, even when eval_subsample is set.
                self._eval_full_override = is_final
                try:
                    m = self._evaluate_clients(sample_ids=None)
                    server_m = self._evaluate_server()
                finally:
                    self._eval_full_override = False
                m["server_accuracy"] = server_m.get("accuracy", 0.0)
                m["server_loss"] = server_m.get("loss", 0.0)
                m["server_f1"] = server_m.get("f1", 0.0)
                last_eval = m.copy()
            else:
                m = last_eval.copy()
            t_eval_total = time.perf_counter() - t_eval_start

            # Round wall-clock now includes client + server + eval (selection is its own field).
            rnd_time = time.perf_counter() - rnd_start
            wall_clock = prev_wall_clock + rnd_time

            # Fairness metrics
            counts = [c.participation_count for c in self.clients]
            mean_p = sum(counts) / max(len(counts), 1)
            fairness_var = sum((p - mean_p) ** 2 for p in counts) / max(len(counts), 1)
            pairs_sum = sum(abs(counts[i] - counts[j]) for i in range(len(counts)) for j in range(i + 1, len(counts)))
            fairness_gini = pairs_sum / (len(counts) ** 2 * max(mean_p, 1e-10)) if len(counts) > 0 else 0.0

            # Good/bad channel counts
            n_good = sum(1 for cid in ids if self.clients[cid].meta.get("channel_group", "good") == "good")
            n_bad = len(ids) - n_good

            t_other = max(0.0, rnd_time - selection_time - t_clients_total - t_server_total - t_eval_total)

            m.update({
                "round": rnd,
                "selection_time": selection_time,
                "compute_time": compute_time,
                "round_time": rnd_time,
                "wall_clock": wall_clock,
                # --- per-phase timing (always recorded; stdout printed when cfg.profile=True) ---
                "t_clients_phase": t_clients_total,   # LD + LT + LI across all selected clients
                "t_server_phase": t_server_total,     # aggregation + server distill + server inference
                "t_eval_phase": t_eval_total,         # evaluation (0 on non-eval rounds)
                "t_other_phase": t_other,             # metrics bookkeeping + memory cleanup + misc
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
                "client_accuracy_avg": m.get("avg_per_client_accuracy", 0.0),
                "client_accuracy_std": m.get("accuracy_std", 0.0),
                # New FD-native metrics
                "logit_entropy_avg": (sum(logit_entropy_vals) / len(logit_entropy_vals)) if logit_entropy_vals else 0.0,
                "logit_entropy_var": (sum(logit_entropy_var_vals) / len(logit_entropy_var_vals)) if logit_entropy_var_vals else 0.0,
                "logit_cosine_diversity": logit_cosine_div,
                "server_client_gap": m.get("server_accuracy", 0.0) - m.get("accuracy", 0.0),
                "channel_quality_selected_avg": sum(self.clients[cid].channel_quality for cid in ids) / max(len(ids), 1),
                "label_coverage_ratio": self._label_coverage_ratio(ids),
                "participation_gini": fairness_gini,
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

            # --- Per-phase profile print (opt-in) ---
            if cfg.profile:
                eval_tag = "eval" if do_eval else "----"
                print(
                    f"  [r={rnd:3d} {eval_tag}] total={rnd_time:6.2f}s "
                    f"| select={selection_time:5.2f} clients={t_clients_total:6.2f} "
                    f"server={t_server_total:5.2f} eval={t_eval_total:5.2f} other={t_other:4.2f}s "
                    f"| acc={m.get('server_accuracy', m.get('accuracy', 0.0)):.3f} "
                    f"| kl={m.get('kl_divergence_avg', 0.0):.2f}",
                    flush=True,
                )

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

    def _label_coverage_ratio(self, ids: List[int]) -> float:
        """Fraction of classes represented by at least one selected client."""
        covered = set()
        for cid in ids:
            h = self.clients[cid].label_histogram
            if isinstance(h, dict):
                for cls, cnt in h.items():
                    if cnt > 0:
                        covered.add(int(cls))
            elif isinstance(h, (list, tuple)):
                for cls, cnt in enumerate(h):
                    if cnt > 0:
                        covered.add(cls)
        return len(covered) / max(self.num_classes, 1)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_clients(self, sample_ids: Optional[List[int]] = None) -> Dict[str, float]:
        """Evaluate client models on test set, returning averaged metrics.

        Uses GPU-cached test data when available to avoid DataLoader overhead.
        Iterates test data once per chunk, running all client models per chunk.

        When sample_ids is None, evaluates ALL total_clients — the paper's per-user
        averaged accuracy metric. Previously defaulted to the first 10 clients,
        which biased metrics when total_clients > 10.
        """
        if sample_ids is None:
            ids_to_eval = list(range(self.cfg.total_clients))
        else:
            ids_to_eval = list(sample_ids)
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

        with torch.inference_mode():
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
        """Fast evaluation using GPU-cached test data. Single data iteration, all models per chunk.

        On CUDA with multiple models, client forward passes are dispatched across CUDA streams
        (mirroring the pattern used in ``run()`` for the client-phase). All per-chunk metrics
        are accumulated as GPU tensors and converted via a single ``.item()`` at the end to
        eliminate per-chunk GPU->CPU sync. See FD_speed_optimization_notes.md tier 1.3.
        """
        n_clients = len(ids_to_eval)
        cfg = self.cfg

        # Optional fixed subsample for non-final rounds (tier 3.3). Defaults to full test set.
        if (self._eval_subsample_indices is not None
                and not getattr(self, "_eval_full_override", False)):
            x_src = self._test_x_cache.index_select(0, self._eval_subsample_indices)
            y_src = self._test_y_cache.index_select(0, self._eval_subsample_indices)
        else:
            x_src = self._test_x_cache
            y_src = self._test_y_cache
        N_test = x_src.size(0)
        chunk_size = 512

        # All metrics accumulated on-GPU; single sync per client at the end.
        per_client_correct_t: List[Optional[torch.Tensor]] = [None] * n_clients
        per_client_total = [0] * n_clients
        all_ys_parts: List[torch.Tensor] = []
        all_preds_parts: List[torch.Tensor] = []
        total_loss_t: Optional[torch.Tensor] = None

        is_cuda = self.device.startswith("cuda")
        use_streams = is_cuda and n_clients > 1 and self._max_parallel > 1
        if use_streams:
            n_streams = min(self._max_parallel, n_clients)
            streams = [torch.cuda.Stream() for _ in range(n_streams)]
        else:
            n_streams = 1
            streams = []

        with torch.inference_mode():
            for start in range(0, N_test, chunk_size):
                x_chunk = x_src[start:start + chunk_size]
                y_chunk = y_src[start:start + chunk_size]
                bs = x_chunk.size(0)

                chunk_outs: List[Optional[torch.Tensor]] = [None] * n_clients

                if use_streams:
                    for i, cid in enumerate(ids_to_eval):
                        stream = streams[i % n_streams]
                        with torch.cuda.stream(stream):
                            chunk_outs[i] = self.client_models[cid](x_chunk)
                    torch.cuda.synchronize()
                else:
                    for i, cid in enumerate(ids_to_eval):
                        chunk_outs[i] = self.client_models[cid](x_chunk)

                for i in range(n_clients):
                    out = chunk_outs[i]
                    loss_i = F.cross_entropy(out, y_chunk, reduction='sum').detach()
                    total_loss_t = loss_i if total_loss_t is None else total_loss_t + loss_i
                    preds = out.argmax(dim=1)
                    correct_i = (preds == y_chunk).sum()
                    per_client_correct_t[i] = (
                        correct_i if per_client_correct_t[i] is None
                        else per_client_correct_t[i] + correct_i
                    )
                    per_client_total[i] += bs
                    all_ys_parts.append(y_chunk)
                    all_preds_parts.append(preds)

        total_loss = float(total_loss_t.item()) if total_loss_t is not None else 0.0
        per_client_correct = [int(t.item()) if t is not None else 0 for t in per_client_correct_t]

        # Skip the pooled GPU->CPU label/pred transfer on intermediate rounds: sklearn's
        # f1/precision/recall over up-to-500k pairs is expensive and unused for diagnostics.
        # Final rounds (and any forced full-eval) still produce full pooled arrays.
        is_final_or_forced = bool(getattr(self, "_eval_full_override", False))
        if is_final_or_forced:
            all_ys = torch.cat(all_ys_parts).cpu().tolist()
            all_preds = torch.cat(all_preds_parts).cpu().tolist()
        else:
            all_ys, all_preds = [], []
        n_samples = N_test * n_clients

        return self._compute_eval_metrics(ids_to_eval, per_client_correct, per_client_total,
                                          all_ys, all_preds, total_loss, n_samples)

    def _compute_eval_metrics(self, ids_to_eval, per_client_correct, per_client_total,
                              all_ys, all_preds, total_loss, n_samples) -> Dict[str, float]:
        """Compute accuracy, F1, precision, recall from accumulated predictions."""
        all_accs = [c / max(t, 1) for c, t in zip(per_client_correct, per_client_total)]

        if not all_ys:
            # Intermediate-round fast path: sklearn skipped. Pooled accuracy from per-client
            # counts is mathematically identical to accuracy_score on the pooled list because
            # every client evaluates the same test set, so per_client_total[i] == N_test.
            total_correct = sum(per_client_correct)
            total_seen = sum(per_client_total)
            accuracy = total_correct / max(total_seen, 1)
            f1 = precision = recall = 0.0
        else:
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

        # Per-client accuracy averaging (paper's metric: average of each client's own test acc).
        # Distinct from `accuracy` which is pooled across all (prediction, label) pairs and
        # double-counts each test sample once per evaluated client.
        avg_per_client_accuracy = sum(all_accs) / max(len(all_accs), 1)

        return {
            "accuracy": accuracy,                       # pooled: micro-averaged over all samples*clients
            "avg_per_client_accuracy": avg_per_client_accuracy,  # paper metric: mean of per-client accs
            "loss": total_loss / max(n_samples, 1),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy_std": acc_std,
            "num_clients_evaluated": len(ids_to_eval),
        }

    def _evaluate_server(self) -> Dict[str, float]:
        """Evaluate the server model on the test set (selection-independent metric)."""
        if self._test_x_cache is not None and self._test_y_cache is not None:
            return self._evaluate_server_cached()
        return eval_model(self.server_model, self.test_loader, self.device)

    def _evaluate_server_cached(self) -> Dict[str, float]:
        """Fast server evaluation using the GPU-cached test tensors.

        Mirrors _evaluate_clients_cached but for a single model: avoids the CPU->GPU
        transfer cost of iterating self.test_loader every eval round. See tier 1.4 in
        FD_speed_optimization_notes.md.
        """
        # Optional fixed subsample for non-final rounds (tier 3.3).
        if (self._eval_subsample_indices is not None
                and not getattr(self, "_eval_full_override", False)):
            x_src = self._test_x_cache.index_select(0, self._eval_subsample_indices)
            y_src = self._test_y_cache.index_select(0, self._eval_subsample_indices)
        else:
            x_src = self._test_x_cache
            y_src = self._test_y_cache
        N_test = x_src.size(0)
        chunk_size = 512

        self.server_model.eval()
        all_ys_parts: List[torch.Tensor] = []
        all_preds_parts: List[torch.Tensor] = []
        total_loss_t: Optional[torch.Tensor] = None
        correct_t: Optional[torch.Tensor] = None

        with torch.inference_mode():
            for start in range(0, N_test, chunk_size):
                x_chunk = x_src[start:start + chunk_size]
                y_chunk = y_src[start:start + chunk_size]
                out = self.server_model(x_chunk)
                loss_i = F.cross_entropy(out, y_chunk, reduction='sum').detach()
                total_loss_t = loss_i if total_loss_t is None else total_loss_t + loss_i
                preds = out.argmax(dim=1)
                correct_i = (preds == y_chunk).sum()
                correct_t = correct_i if correct_t is None else correct_t + correct_i
                all_ys_parts.append(y_chunk)
                all_preds_parts.append(preds)

        total_loss = float(total_loss_t.item()) if total_loss_t is not None else 0.0
        correct = int(correct_t.item()) if correct_t is not None else 0

        all_ys = torch.cat(all_ys_parts).cpu().tolist()
        all_preds = torch.cat(all_preds_parts).cpu().tolist()

        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            accuracy = accuracy_score(all_ys, all_preds)
            f1 = f1_score(all_ys, all_preds, average='macro', zero_division=0)
            precision = precision_score(all_ys, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_ys, all_preds, average='macro', zero_division=0)
        except ImportError:
            accuracy = (correct / max(N_test, 1)) if N_test > 0 else 0.0
            f1 = precision = recall = 0.0

        return {
            "accuracy": accuracy,
            "loss": total_loss / max(N_test, 1),
            "f1": f1,
            "precision": precision,
            "recall": recall,
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
