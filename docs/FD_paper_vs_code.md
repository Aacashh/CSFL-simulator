# FD Implementation: Paper vs Code — Side-by-Side Comparison

**Paper:** Mu et al., "Federated Distillation in Massive MIMO Networks," IEEE TCCN, vol. 10, no. 4, Aug 2024.
**Code:** `csfl_simulator/core/fd_simulator.py`, `channel.py`, `fd_aggregation.py`

---

## 1. Round Structure (Algorithm 1: FedTSKD)

| Step | Paper (Algorithm 1) | Code (`run()` method) | Match? |
|------|--------------------|-----------------------|--------|
| 1 | **Local Training**: K_r steps of SGD on private data (Eq. 12) | `_local_train(cid, K_r)` — line 696 | YES |
| 2 | **Local Inference**: client predicts on public dataset (Eq. 13) | `_generate_logits(cid)` — line 702 | YES |
| 3 | **Uplink**: quantize + modulate + noisy channel | `channel.quantize()` then `channel.uplink_noise()` — lines 703-705 | YES |
| 4 | **Server Aggregation**: weighted average (Eq. 17) | `logit_avg()` — line 742 | YES |
| 5 | **Server Distillation**: S steps of KL distillation | `_server_distill()` — line 747 | YES |
| 6 | **Server Inference**: server predicts on public dataset | `server_model(public_x_cache)` — lines 750-759 | YES |
| 7 | **Downlink**: quantize + modulate + precoded channel | `channel.quantize()` then `channel.downlink_noise()` — lines 689-691 | YES |
| 8 | **Local Distillation** (next round): S steps via KL (Eq. 14) | `_local_distill()` — line 692 (runs at start of next round) | YES |

**Order within each round (r >= 1):**

| Paper (Fig. 2) | Code |
|-----------------|------|
| LD (local distillation from prev round's logits) | `_local_distill()` — line 683-692 |
| LT (local training on private data) | `_local_train()` — line 696 |
| LI (local inference on public dataset) | `_generate_logits()` — line 702 |
| UL (uplink transmission) | quantize + uplink_noise — lines 703-705 |
| Server Aggregation | `logit_avg()` — line 742 |
| SD (server distillation) | `_server_distill()` — line 747 |
| Server LI (server inference) | lines 750-759 |
| DL (downlink — applied at start of NEXT round) | quantize + downlink_noise — lines 689-691 |

**Round 0 special case:**
- Paper: No distillation in round 1 (first round), only LT + LI
- Code: `if aggregated_logits is not None and rnd > 0:` — line 683. Distillation skipped. **MATCH**

---

## 2. Loss Functions

### Local Training Loss (Eq. 1-2)

| | Paper | Code |
|-|-------|------|
| **Loss** | Cross-entropy: `L_n(w) = (1/D_n) sum l(F_n(x;w), y)` | `nn.CrossEntropyLoss()` — line 357 |
| **Match?** | | **YES** |

### Distillation Loss (Eq. 3)

| | Paper | Code |
|-|-------|------|
| **Formula** | `Q_n(w;Z) = (1/\|Z\|) sum l_KL(LSM(F_n(x_p;w)/T), SM(z_p/T))` | `F.kl_div(log_softmax(model(x)/T), softmax(target/T)) * T^2` |
| **KL direction** | l_KL(log_student, teacher) = KL(teacher \|\| student) | `F.kl_div(log_p, q)` = KL(q \|\| exp(log_p)) = KL(teacher \|\| student) |
| **T^2 scaling** | Standard distillation scaling | `* (T * T)` — line 491 |
| **Match?** | | **YES** |

### Server Distillation Loss

| | Paper | Code |
|-|-------|------|
| **Formula** | Same KL formula as local distillation | Identical formula — lines 555-557 |
| **Target** | Aggregated client logits | `server_target = aggregated_logits` — line 746 |
| **Match?** | | **YES** |

---

## 3. Hyperparameters

| Parameter | Paper (Section VI) | Code default | Experiment cmd | Match? |
|-----------|-------------------|--------------|----------------|--------|
| **N (total clients)** | **15** | 30 | 30 | **NO** — paper uses 15 |
| **K (selected/round)** | **15 (full participation)** | 10 | 10 | **NO** — paper uses full participation |
| **R (rounds)** | 200 | 200 | 200 | YES |
| **Optimizer** | Adam | Adam (`fd_optimizer=adam`) | adam | YES |
| **Learning rate** | eta = 0.001 | `distillation_lr=0.001` | 0.001 | YES |
| **Batch size (training)** | 128 | 128 | 128 | YES |
| **Batch size (distillation)** | 500 | 500 | 500 | YES |
| **Epochs (training)** | 2 (but dynamic steps override this) | dynamic | dynamic | YES |
| **Epochs (distillation)** | 2 | `distillation_epochs=2` | 2 | YES |
| **Temperature T** | Not specified (likely 1.0) | 1.0 | 1.0 (default) | ASSUMED |
| **Public dataset** | STL-10, 2000 images | STL-10, 2000 | STL-10, 2000 | YES |
| **Partition** | Dirichlet alpha=0.5 | 0.5 | 0.5 | YES |
| **N_BS (BS antennas)** | 64 | 64 | 64 | YES |
| **N_D (device antennas)** | 1 | 1 | 1 | YES |
| **UL SNR** | -8 dB | -8 dB | -8 dB | YES |
| **DL SNR** | -20 dB | -20 dB | -20 dB | YES |
| **Quantization** | 8-bit | 8-bit | 8-bit | YES |
| **Data augmentation** | Random horizontal flip (p=0.5) | Depends on dataset loader | VERIFY |
| **Dynamic steps base** | 5 | 5 | 5 | YES |
| **Dynamic steps period** | 25 | 25 | 25 | YES |
| **Dynamic steps rounds** | First 100 rounds | All rounds (multiplier reaches 1 by round 100) | YES |

---

## 4. Models (CRITICAL MISMATCH)

### Paper Table IV — Model Assignments by Dataset

| Dataset combo | Paper models | Paper param counts | Our old exp02 models | Our param counts |
|---------------|-------------|-------------------|---------------------|-----------------|
| **CIFAR-10 + STL-10** | **ResNet, MobileNet, ShuffleNet** | **~11M, ~3.4M, ~1M** | FD-CNN1/2/3 | 545K, 102K, 68K |
| MNIST + FMNIST | CNN_1, CNN_2, CNN_3 | 1.2M, 79K, 25K | FD-CNN1/2/3 | 545K, 102K, 68K |

**Paper's CIFAR-10 results (Table IV, error-prone mMIMO, alpha=0.5):**

| Model | Error-free | Error-prone (-8/-20 dB) |
|-------|-----------|------------------------|
| ResNet | 54.91% | 51.77% |
| MobileNet | 56.96% | 53.58% |
| ShuffleNet | 49.99% | 46.02% |
| **Overall** | **53.95%** | **50.46%** |

**Our exp02 result:** ~28-32% accuracy, plateauing and oscillating.

**Root cause:** FD-CNN1/2/3 are the paper's MNIST architectures (Table III). They lack capacity for CIFAR-10. Our models' centralized ceiling is ~65-78% vs paper's ~90-94%.

**FIX (now implemented):** New models added: `ResNet18-FD`, `MobileNetV2-FD`, `ShuffleNetV2-FD`

---

## 5. Dynamic Training Steps (Section V-A)

| | Paper | Code |
|-|-------|------|
| **Formula** | K_r = ceil(D_n / batch_size) * max(1, 5 - floor((r-1)/25)) | `max(1, base - (round_idx // period))` * `ceil(data_size / batch_size)` |
| **Base multiplier** | 5 | `cfg.dynamic_steps_base = 5` |
| **Decrease period** | Every 25 rounds | `cfg.dynamic_steps_period = 25` |
| **Active for** | First 100 rounds (multiplier=1 after round 100) | Multiplier reaches 1 at round 100, stays 1 |
| **Match?** | | **YES** |

**Concrete values (30 clients, ~1667 samples each, batch_size=128):**

| Round | Multiplier | Steps per epoch | Total K_r per client |
|-------|-----------|----------------|---------------------|
| 0-24 | 5 | 13 | 65 |
| 25-49 | 4 | 13 | 52 |
| 50-74 | 3 | 13 | 39 |
| 75-99 | 2 | 13 | 26 |
| 100+ | 1 | 13 | 13 |

---

## 6. Channel Noise Model

### Uplink (Eq. 16)

| | Paper | Code (`channel.py` lines 62-77) |
|-|-------|---------------------------------|
| **Formula** | sigma_UL^2 = N_D * sigma^2 / (2 * P_UL * N_BS) | `noise_var = n_device / (2 * snr_lin * n_bs)` |
| **SNR definition** | SNR_UL = P_UL / sigma^2 | `snr_lin = 10^(ul_snr_db / 10)` |
| **Array gain** | 1/N_BS factor from MIMO combining | Included via `n_bs` in denominator |
| **Match?** | | **YES** |

**Concrete value (-8 dB, N_BS=64, N_D=1):**
```
SNR_UL = 10^(-0.8) = 0.158
noise_var = 1 / (2 * 0.158 * 64) = 0.0494
noise_std = 0.222
```

### Downlink (Eq. 22)

| | Paper | Code (`channel.py` lines 83-97) |
|-|-------|---------------------------------|
| **Formula** | sigma_DL^2 = sigma_z^2 * sigma^2 / (2 * P_DL) | `noise_var = logit_var / (2 * snr_lin)` |
| **sigma_z^2** | Variance of transmitted logits | `logits.var().item()` |
| **Array gain** | NOT in paper's Appendix A DL term | NOT included |
| **Match?** | | **YES** (matches Appendix A Eq. 23d) |

**Concrete value (-20 dB, logit_var ~ 1.0):**
```
SNR_DL = 10^(-2.0) = 0.01
noise_var = 1.0 / (2 * 0.01) = 50.0
noise_std = 7.07   <-- THIS IS HUGE relative to logit magnitude (~2)
```

**IMPORTANT:** This noise level is correct per the paper. The paper's models handle it because they are much larger (ResNet/MobileNet/ShuffleNet produce higher-confidence logits). The paper's Fig. 4 shows -20 dB DL does cause significant accuracy degradation (65% error-free -> 50% with -20 dB).

### Quantization

| | Paper | Code (`channel.py` lines 44-56) |
|-|-------|---------------------------------|
| **Method** | Uniform quantization, 8 bits | Mid-tread uniform quantization |
| **Levels** | 2^8 - 1 = 255 | `n_levels = 2^bits - 1` |
| **Applied on** | UL: before uplink noise; DL: before downlink noise | UL: line 703; DL: line 689 |
| **Match?** | | **YES** |

---

## 7. Aggregation (Eq. 17)

| | Paper | Code (`fd_aggregation.py`) |
|-|-------|-----------------------------|
| **Formula** | z_bar = sum(lambda_n * z_n) where lambda_n = D_n / sum(D_j) | `result.add_(logits, alpha=w / total_w)` — line 29 |
| **Weights** | Data size proportional | `weights = [client.data_size for cid in ids]` — line 732 |
| **Match?** | | **YES** |

### Group-based (FedTSKD-G, Algorithm 2)

| | Paper | Code |
|-|-------|------|
| **Split** | Separate good/bad channel groups | `logit_avg_grouped()` separates by group label |
| **Output** | Concatenated [bad_logits, good_logits] shape (N_pub, 2C) | `torch.cat([z_bad, z_good], dim=-1)` |
| **Server FC** | Doubled last layer (2C outputs) | `_adapt_server_for_groups()` — line 231 |
| **Client receives** | Own group's slice | `[:, :C]` for bad, `[:, C:]` for good — line 686 |
| **Match?** | | **YES** |

---

## 8. What Clients Receive

| | Paper | Code |
|-|-------|------|
| **Sent to clients** | Server's predicted logits on public dataset (after server distillation) | `aggregated_logits = server_logits` — line 765 |
| **NOT** | Raw aggregated client logits | Server distills first, then generates its own predictions |
| **Match?** | | **YES** |

Flow: client logits → aggregate → server distills on aggregated → server predicts → server logits sent to clients

---

## 9. Optimizer Details

| Context | Paper | Code | Match? |
|---------|-------|------|--------|
| **Training optimizer** | Adam (Section VI) | Adam when `fd_optimizer=adam` | YES |
| **Training LR** | eta = 0.001 | `distillation_lr = 0.001` (shared) | YES |
| **Distillation optimizer** | Adam (Section VI) | Always Adam | YES |
| **Distillation LR** | eta = 0.001 | `distillation_lr = 0.001` | YES |
| **Server distillation optimizer** | Adam (implied) | Always Adam | YES |
| **Server distillation LR** | eta = 0.001 | `distillation_lr = 0.001` | YES |

**Note:** When `fd_optimizer=adam`, both training and distillation use Adam with the same LR (0.001). This matches the paper.

---

## 10. Participation Model (CRITICAL DIFFERENCE)

| | Paper | Our Experiments |
|-|-------|----------------|
| **Total clients N** | 15 | 30 |
| **Selected per round K** | **15 (ALL clients)** | 10 (33%) |
| **Participation rate** | **100%** | 33% |

**Impact of partial participation in FD:**
1. Clients not selected don't receive new logits → models become stale
2. Fewer client logits averaged → noisier aggregation
3. Some clients may never learn from others' knowledge
4. The paper's convergence analysis (Theorem 1) assumes all clients participate

**For our research:** We intentionally use K < N to study client selection. This is our contribution — the paper doesn't do client selection at all.

---

## 11. Potential Issues Causing Accuracy Drop Over Rounds

### Issue A: Training overwrites distillation

Each round: distill (2 epochs) → train (13-65 steps). Training on private non-IID data may overwrite the global knowledge injected by distillation, especially with dynamic steps giving 5x more training than distillation in early rounds.

### Issue B: Server bootstrapping from noisy round-0 logits

In round 0, clients are untrained. Their logits are near-random. The server distills on these noisy aggregated logits, learning poor patterns. These poor server logits are then sent to clients in round 1, potentially causing a negative feedback loop.

### Issue C: Noise accumulation in the distillation loop

Round r: server sends noisy logits → clients distill (absorb noise) → clients train (partially recover) → clients generate logits → quantize + UL noise → aggregate → server distills (absorbs noise) → server generates logits → repeat.

Each round adds noise through both UL and DL channels. With -20 dB DL (noise_std >> signal), the distillation signal degrades faster than learning can compensate.

### Issue D: Stale clients (partial participation only)

With K=10/N=30, clients selected in round 0 may not be selected again until round 3+. During that time, their models don't benefit from distillation, while other clients' models advance. When finally selected, their outdated logits pollute the aggregation.

### Issue E: Non-IID data + small models + noise = catastrophic forgetting cycle

1. Client A has only classes {0,1,2} (Dirichlet alpha=0.5)
2. Trains well on those classes
3. Receives distillation logits covering all 10 classes (but noisy)
4. Distillation slightly corrupts its knowledge of {0,1,2}
5. Next training round partially recovers, but each cycle loses a bit
6. Over 200 rounds, accuracy oscillates instead of converging

---

## 12. Data Augmentation

| | Paper | Code |
|-|-------|------|
| **Training data** | "each sample is randomly flipped horizontally with probability 0.5" | Depends on `datasets.py` transforms | VERIFY |
| **Public data** | Not mentioned (likely no augmentation) | Cached as-is on GPU (no augmentation) | ASSUMED MATCH |
| **Test data** | Standard (no augmentation) | Standard | YES |

**TODO:** Verify that `datasets.py` applies RandomHorizontalFlip for CIFAR-10 training data.

---

## Summary: What Matches vs What Doesn't

### CORRECT (verified)
- Round structure and order of operations
- KL divergence formula and direction
- Temperature scaling
- Dynamic steps formula
- Uplink noise (Eq. 16) with N_BS array gain
- Downlink noise (Eq. 22) matching Appendix A
- Quantization (8-bit uniform)
- Aggregation weights (data-size proportional)
- Group-based FedTSKD-G
- Server → client logit flow
- Optimizer choice and learning rate
- Distillation epochs

### MISMATCHED (causing poor results)
- **Models**: FD-CNN1/2/3 on CIFAR-10 instead of ResNet/MobileNet/ShuffleNet → **FIXED**
- **Participation**: K=10/N=30 (33%) instead of K=N=15 (100%) → **Intentional for client selection research**

### TO VERIFY
- Data augmentation in training transforms
- Whether temperature T=1.0 matches paper's intent (paper doesn't specify)

---

## 13. Fixed in revision 2026-04-20-b

These deviations from Mu et al. 2024 were identified during the Exp-1 paper-replication autopsy (server accuracy plateaued at 44–46% vs the paper's 50%+ target) and fixed in a single revision. Cross-reference each item to the paper section that motivated the fix.

### Correctness-critical

- **`fast_mode` renamed to `smoke_test_mode`; default flipped from `True` to `False`.**
  - Files: `core/simulator.py` (field + default), `core/fd_simulator.py` (6 inline uses), `core/parallel.py` (function parameter + 7 call sites), `__main__.py` (CLI wiring), `app/main.py` (Streamlit checkbox), `app/export.py` (backwards-compat read), `test_parallelization.py`, `reproduce_issue.py`, `scripts/run_matrix.py`.
  - CLI flags `--fast-mode` / `--no-fast-mode` kept as aliases for `--smoke-test-mode` / `--no-smoke-test-mode` so existing shell scripts (`run_all_experiments.sh`, `run_next_sota_experiments.sh`, etc.) continue to work unchanged.
  - Rationale: the True default silently capped training at 2 batches per round, producing ~45% accuracy ceilings in any run that didn't explicitly pass `--no-fast-mode`.

- **`_evaluate_clients` now evaluates ALL clients by default** (`core/fd_simulator.py`).
  - Changed call in `run()`: `_evaluate_clients(sample_ids=ids)` → `_evaluate_clients(sample_ids=None)`.
  - Changed default fallback inside `_evaluate_clients`: previously evaluated only `range(min(total_clients, 10))`, now evaluates `range(total_clients)`.
  - Rationale: evaluating only the selection-policy-chosen clients entangles the selection policy with the metric, giving different methods an unfair advantage. The paper's per-user averaged metric requires evaluating every client each eval round.

- **Paper-metric `avg_per_client_accuracy` added to `_compute_eval_metrics`** (`core/fd_simulator.py`).
  - Existing `accuracy` (micro-pooled over all `[client × sample]` predictions) retained for backwards compatibility.
  - New `avg_per_client_accuracy` = mean of each client's own accuracy — matches Mu et al. Table IV's "Overall" row semantics (average across clients, not across test samples).
  - `m["client_accuracy_avg"]` now sources from `avg_per_client_accuracy`.

### Paper-faithfulness

- **Server-side distillation uncertainty `σ_D²` (Eq. 18).**
  - New `SimConfig.server_distill_sigma: float = 0.01` (`core/simulator.py`).
  - In `run()`, `core/fd_simulator.py`: additive Gaussian noise `omega_D ~ N(0, sigma_D^2 I)` is applied to `server_target` immediately before `_server_distill(...)` when both the channel model is enabled and `server_distill_sigma > 0`.

- **FL-equivalent communication cost now uses 16-bit quantisation (2 bytes/param).**
  - `core/fd_simulator.py`: `fl_model_bytes = params_count * 4.0` → `fl_model_bytes = params_count * 2.0`, matching paper Fig. 10.
  - Prior 4-byte assumption overstated FL cost by 2×, inflating the reported FD comm-reduction ratio.

- **FedTSKD-G static channel groups (Algorithm 2).**
  - New `ClientInfo.meta["static_channel_group"]` assigned once in `init_system_state` (`core/system.py`) based on initial `channel_quality` and `channel_threshold`.
  - `simulate_round_env` now prefers the static group when `knobs["static_channel_groups"] == True` (default); legacy per-round re-classification stays available via `SimConfig.static_channel_groups = False`.
  - `FDSimulator.setup()` now passes the FD paradigm and channel threshold into `init_system_state` so the static group is correctly populated.
  - Rationale: the paper assigns groups once based on persistent channel-strength estimates, not the drifting per-round `channel_quality` random walk; per-round reclassification caused group flapping that undermines the paper's grouped-aggregation stability.

### Transparency

- **Temperature comment added** (`core/simulator.py`): "Paper implicitly uses T=1.0 (Eq. 3); KD literature suggests T=3–4 for soft-target benefit."

- **Constant-LR deviation note added** to `core/fd_simulator.py` module docstring: explicitly flags that the paper's Theorem 1 assumes η_t = β₀/(t+β₁) step-size decay for its O(1/t) convergence bound, whereas our implementation uses constant LRs (Adam's running-moment scaling compensates behaviourally but the theorem bound isn't directly empirically validated).

### Optional

- **MMSE combining scheme** added to `MIMOChannel` (`core/channel.py`).
  - New `combining: str = "zf"` constructor parameter; `"mmse"` attenuates uplink noise by `1 / (1 + 1/(SNR·N_BS))` relative to ZF, matching paper §VI-C Fig. 7.
  - New `SimConfig.combining_scheme: str = "zf"` wired through to `MIMOChannel(combining=cfg.combining_scheme)` in `FDSimulator.setup()`.
  - Enables paper-faithful reproduction of the ZF-vs-MMSE ablation (paper Fig. 7).

### What this revision does NOT change

- Round ordering LD → LT → LI (already correct).
- KL divergence formula + T² scaling (already correct).
- Uplink/downlink noise variance formulas (already match Eqs. 16 and 22 after fix 7 above for the FL-comm accounting; the channel formulas themselves were already right).
- Selection-method interface (unchanged — all 47+ existing selectors still work).
- Reproducibility: all new noise injection uses `torch.randn_like` which is seeded by the existing `set_seed` call in `FDSimulator.__init__`.

### Verification

Run after applying all fixes:
```bash
python test.py            # trivial print sanity
python test_partition.py  # Dirichlet partition check
# 5-round FD smoke test (non-trivial accuracy progression expected)
python -m csfl_simulator run --paradigm fd --method fd_native.calm_fd \
    --dataset MNIST --model FD-CNN1 --public-dataset FMNIST --public-dataset-size 500 \
    --total-clients 10 --clients-per-round 4 --rounds 5 --local-epochs 1 \
    --distillation-epochs 1 --distillation-batch-size 100 --batch-size 32 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 --group-based \
    --seed 42 --device cpu
```

Success criteria: metrics.json shows accuracy increasing over the 5 rounds (not flat at ~10% or NaN).
