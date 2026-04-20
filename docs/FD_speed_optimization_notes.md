# FD Simulator Speed Optimization Notes

This document describes the speed-ups applied to `csfl_simulator/core/fd_simulator.py`
in the wake of the paper-faithfulness fixes that raised per-round wall-clock by
roughly 100–200×. None of the invariants introduced by those fixes (smoke-mode
off by default, full-client-pool evaluation, per-user accuracy as the headline
metric, KL + T² scaling, group logit layout, `σ_D` server-distill noise,
across-method reset path, seed-based reproducibility) are relaxed here.

## Change summary by tier

### Tier 1.1 — cuDNN autotune via `performance_mode`
- `csfl_simulator/core/utils.py::set_seed` now takes a third argument
  `performance_mode`. When true it sets
  `torch.backends.cudnn.benchmark = True`, turns off deterministic algorithms,
  and lets cuDNN pick the fastest kernel per input shape. Seed-based
  reproducibility of high-level results is preserved; bit-for-bit is not.
- `SimConfig.performance_mode: bool = True` added in
  `csfl_simulator/core/simulator.py`.
- `FDSimulator.__init__` and `FDSimulator.run` now pass the config value
  through to `set_seed`, so `deterministic=not performance_mode` when the user
  opts in.
- The module docstring of `fd_simulator.py` calls out that exact bit-level
  reproducibility requires `--no-performance-mode`.

### Tier 1.2 — cuDNN kernel pre-warm at setup end
- New block at the end of `FDSimulator.setup()` fires a single zero-tensor
  forward pass on each unique client architecture and on the server model. This
  forces cuDNN's first-time algorithm selection to happen once at setup
  instead of during round 0. Guarded by `device.startswith("cuda")` and a
  try/except so unusual model shapes degrade gracefully. Logs a single line
  like `[FD setup] pre-warmed cuDNN kernels on N model(s).`

### Tier 1.3 — CUDA-stream-parallel client evaluation
- `_evaluate_clients_cached` now dispatches each client's forward pass on its
  own CUDA stream (pool of `min(self._max_parallel, n_clients)` streams,
  mirroring the client-phase stream pattern in `run()`), with a single
  `torch.cuda.synchronize()` at the end of each chunk.
- The outer context is now `torch.inference_mode()` (see tier 1.8). All
  per-chunk metrics (`correct`, `total_loss`) are accumulated as GPU
  tensors, so the per-chunk `.item()` syncs are gone; a single `.item()` fires
  at the end of the function.
- Sequential fallback path retained for CPU device and single-client case.

### Tier 1.4 — Cached server evaluation
- New `_evaluate_server_cached` reuses `self._test_x_cache` /
  `self._test_y_cache` with chunked forward passes and
  `torch.inference_mode()`, avoiding the CPU→GPU transfer cost of
  `eval_model(self.test_loader, ...)`.
- `_evaluate_server` prefers the cached path when the cache is available and
  falls back to the `eval_model` DataLoader path otherwise.

### Tier 1.5 — Precomputed softmax distillation targets
- Both `_local_distill` and `_server_distill` now compute
  `q_full = F.softmax(target_logits / T, dim=-1)` **once** inside a
  `torch.no_grad()` block before the epoch loop, then slice
  `q_batch = q_full[i:i + bs]` inside the loop. The student-side
  `log_softmax` stays per-batch because it depends on live model weights.

### Tier 1.6 — `set_to_none=True` on every `zero_grad`
- All six `opt.zero_grad()` call sites in `_local_train`, `_local_distill`,
  and `_server_distill` now pass `set_to_none=True`. Identical numerics for
  Adam/SGD; cheaper because PyTorch assigns `None` instead of running a
  memset over every gradient buffer.

### Tier 1.7 — One `.item()` per function, not per step
- `_local_train` now holds the latest loss in a detached GPU tensor
  (`loss_buffer`) and only converts to a Python float at the very end (or in
  the smoke-test early-return branch).
- `_local_distill` and `_server_distill` accumulate a running `loss_sum`
  GPU tensor plus an integer `n_batches`, and convert once at return. This
  removes the per-SGD-step and per-distill-batch GPU→CPU syncs.
- `track_grad_norm=True` still triggers its existing per-step GPU→CPU norm
  readout — that path is opt-in and not on the hot path.

### Tier 1.8 — `torch.inference_mode()` in every eval path
- `_generate_logits` decorator changed from `@torch.no_grad()` to
  `@torch.inference_mode()`.
- `_evaluate_clients` DataLoader fallback, `_evaluate_clients_cached`,
  `_evaluate_server_cached`, and the server-on-public-dataset block inside
  `run()` (previously `with torch.no_grad():`) now use `torch.inference_mode()`.
  `inference_mode` additionally disables version-counter bookkeeping.
- Downstream use: eval outputs are never fed into `.backward()` or
  `.requires_grad_(True)`. The logits produced by `_generate_logits` later
  flow into `logit_avg` (aggregation, grad-free) and the next round's
  distillation targets — the channel layer always clones or adds noise,
  producing fresh non-inference tensors, and even the no-channel path only
  uses the tensor as a KL-divergence target which PyTorch treats as
  non-differentiable.

### Tier 2 — Config defaults & CLI
- `SimConfig.use_amp` default: `False` → `True` (a no-op on CPU thanks to
  the existing CUDA guard).
- `SimConfig.eval_every` default: `5` → `10`. Final round is still
  always evaluated.
- `SimConfig.performance_mode: bool = True` (new, tier 1.1).
- `csfl_simulator/__main__.py` gains `--performance-mode` /
  `--no-performance-mode`, flips `--use-amp` to default-True and adds
  `--no-amp`, adds `--use-torch-compile`, `--channels-last`,
  `--eval-subsample`, and bumps the default of `--eval-every` to 10.
- `FDSimulator.run()` warns when `smoke_test_mode=True` is combined with
  `rounds > 20`, catching silent re-enablement of the debug shortcut.

### Tier 3 — Opt-in features (default OFF)
- `SimConfig.use_torch_compile: bool = False`. When on plus CUDA plus
  PyTorch ≥ 2.1, `_maybe_compile_models` wraps each client model and the
  server model with `torch.compile(mode="reduce-overhead")`. Failures per
  model are caught and the uncompiled module is kept — the simulator stays
  functional even if only some models compile.
- `SimConfig.channels_last: bool = False`. When on plus CUDA,
  `_maybe_apply_channels_last` converts the cached public, test, and
  per-client training tensors along with every unique model instance to
  `torch.channels_last`. Applied only when the public tensor is 4D.
- `SimConfig.eval_subsample: int = 0`. When > 0, a seeded random subset of
  `eval_subsample` test samples is fixed at setup
  (`self._eval_subsample_indices`) and used for intermediate eval rounds.
  The final round always uses the full test set — `run()` sets
  `self._eval_full_override = True` on the final round before calling
  `_evaluate_clients` and `_evaluate_server`, and clears it in a `finally`.

## Added / changed public API

| Name | Kind | Location | Notes |
| --- | --- | --- | --- |
| `set_seed(seed, deterministic, performance_mode)` | function | `core/utils.py` | New `performance_mode` kwarg (default False). Back-compat preserved. |
| `SimConfig.performance_mode` | field | `core/simulator.py` | `bool = True` |
| `SimConfig.use_amp` | field | `core/simulator.py` | Default flipped to `True` |
| `SimConfig.eval_every` | field | `core/simulator.py` | Default 5 → 10 |
| `SimConfig.use_torch_compile` | field | `core/simulator.py` | New, default False |
| `SimConfig.channels_last` | field | `core/simulator.py` | New, default False |
| `SimConfig.eval_subsample` | field | `core/simulator.py` | New, default 0 |
| `--performance-mode` / `--no-performance-mode` | CLI | `__main__.py` | — |
| `--no-amp` | CLI | `__main__.py` | `--use-amp` default is now True |
| `--use-torch-compile` | CLI | `__main__.py` | Tier 3.1 |
| `--channels-last` | CLI | `__main__.py` | Tier 3.2 |
| `--eval-subsample N` | CLI | `__main__.py` | Tier 3.3, `N` samples for intermediate eval |
| `FDSimulator._evaluate_server_cached` | method | `core/fd_simulator.py` | Tier 1.4 |
| `FDSimulator._maybe_compile_models` | method | `core/fd_simulator.py` | Tier 3.1 |
| `FDSimulator._maybe_apply_channels_last` | method | `core/fd_simulator.py` | Tier 3.2 |
| `FDSimulator._eval_subsample_indices` | attribute | `core/fd_simulator.py` | `Optional[torch.Tensor]`, tier 3.3 |
| `FDSimulator._eval_full_override` | attribute | `core/fd_simulator.py` | `bool`, set by `run()` on the final round |

## Deviations from the prompt

- None that affect correctness. The prompt asked for `cudnn.benchmark` only
  when `performance_mode=True`; the implementation adds a third branch in
  `set_seed` for the "neither deterministic nor performance" case and sets
  both `deterministic` and `benchmark` to `False` there, which matches
  PyTorch's documented default behaviour.
- The pre-warm block (tier 1.2) is placed **after** the tier 3 hooks in
  `setup()`. This is intentional: if `torch.compile` or `channels_last` are
  enabled, they must rewrite / convert the models **before** the first
  forward pass so the pre-warm measures the final kernel layout.

## Known limitations and caveats

- `torch.compile` (`use_torch_compile=True`) can fail on some model-zoo
  layers, particularly when run against mixed-precision paths. Failures fall
  back silently — look for `[FD setup] torch.compile failed for ...` in
  stdout. First-round wall-clock will be 10–60 s higher the very first time.
- `channels_last` benefits are negligible on single-channel data (MNIST,
  Fashion-MNIST). Recommended for CIFAR-10/100 and larger CNNs only.
- `performance_mode=True` makes results non-deterministic at the bit level;
  per-round metrics and final accuracies are statistically equivalent across
  runs on the same hardware/seed, but tensor values will not match
  byte-for-byte between runs. Set `--no-performance-mode` for regression
  testing where bit-exact outputs matter.
- With `torch.compile` enabled, restoring `_initial_client_states` via
  `load_state_dict` still works because `OptimizedModule` forwards the call
  to the underlying module; the state-dict key prefix stays compatible.
- `inference_mode` tensors produced by the server-inference block flow into
  the next round's distillation target. The channel layer's `.clone()` call
  converts them back to regular tensors; the no-channel path uses them only
  as the `target` argument to `F.kl_div`, which does not compute gradients
  through that input. No `.backward()` or `.requires_grad_(True)` touches
  these tensors downstream.

## How to measure (before vs. after)

Run with `--profile` both before and after the changes and capture the
per-phase timings recorded in `metrics.json`:

```bash
python -m csfl_simulator run --paradigm fd --name fd_before \
  --method heuristic.random --dataset CIFAR-10 --rounds 50 --profile
# ... checkout the optimization branch ...
python -m csfl_simulator run --paradigm fd --name fd_after \
  --method heuristic.random --dataset CIFAR-10 --rounds 50 --profile
```

Then compare `t_clients_phase`, `t_server_phase`, `t_eval_phase`, and
`round_time` across the two runs. Expected improvements per round:

| Phase | Expected reduction |
| --- | --- |
| `t_clients_phase` | 20–40 % |
| `t_server_phase` | 30–50 % |
| `t_eval_phase` | 60–80 % |
| Total `round_time` | 40–60 % |

### If the measured speed-up is disappointing

1. **Check cuDNN autotune actually engaged.** Add a one-line
   `print("cudnn.benchmark=", torch.backends.cudnn.benchmark)` right after
   `setup()` returns. Expect `True` when `--performance-mode` is on.
2. **Confirm AMP is active.** In a Python shell with the simulator imported,
   `sim._use_amp` should be `True` on CUDA. If `False`, check whether
   `--no-amp` was passed or `device` is `cpu`.
3. **Confirm stream parallelism is engaging.** `sim._max_parallel` must be
   `> 1`. If it is `1`, either `--parallel-clients` is `0` (sequential) or
   the autodetection picked 1 because VRAM was tight. Try `--parallel-clients
   -1` explicitly.
4. **Tier 3 opt-ins are off by default.** If the models are conv-heavy and
   you are on Ampere or newer, `--channels-last` should give a further 10–20 %
   wins on top of the tier 1 / tier 2 defaults. `--use-torch-compile` gives
   another 20–40 % on sustained training but taxes the first round with 10–60 s
   of compilation.
5. **Intermediate eval dominates.** On small models / long eval loops,
   `t_eval_phase` can be the gating phase. Raise `--eval-every` further, or
   enable `--eval-subsample 2000` so intermediate rounds evaluate on a fixed
   2 000-sample subset (the final round still uses the full test set).
