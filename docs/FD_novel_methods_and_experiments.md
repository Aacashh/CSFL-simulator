# Novel Client Selection Methods for Federated Distillation: Design, Motivation, and Experiment Plan

**Date:** 2026-04-06
**Reference:** Mu et al., "Federated Distillation in Massive MIMO Networks," IEEE TCCN, vol. 10, no. 4, Aug 2024
**Codebase:** CSFL-Simulator (`csfl_simulator/`)

---

## Table of Contents

1. [Diagnosis: Why Accuracy is Stuck at 0.2–0.3](#1-diagnosis)
2. [Novel FD-Native Client Selection Methods](#2-novel-methods)
   - [Method 1: SNR-Aware Diversity Selector (SNRD)](#method-1-snrd)
   - [Method 2: Logit-Quality Thompson Sampling (LQTS)](#method-2-lqts)
   - [Method 3: Noise-Robust Fair Scheduler (NRFS)](#method-3-nrfs)
   - [Method 4: Logit Entropy Maximization (LEM)](#method-4-lem)
3. [Metrics for Holistic Evaluation](#3-metrics)
4. [Experiment Suite](#4-experiments)
5. [Paper Narrative](#5-narrative)

---

## 1. Diagnosis: Why Accuracy is Stuck at 0.2–0.3 {#1-diagnosis}

### The Gap

| Setting | Our Simulator | Mu et al. Paper | Gap |
|---------|:---:|:---:|:---:|
| CIFAR-10/STL-10, error-prone (-8/-20 dB) | **0.294** (best FD) | **~0.50** (Fig. 4, 200 rounds) | -0.21 |
| MNIST/FMNIST, error-prone | **0.840** | **~0.72** (Fig. 6) | +0.12 |

On CIFAR-10, we underperform the paper by ~20pp. On MNIST, we slightly exceed it. This asymmetry points to **model capacity** as the primary bottleneck, not the FD algorithm itself.

### Root Causes (Ranked by Impact)

**RC1 — Model capacity mismatch (PRIMARY).** Our experiments use FD-CNN1/2/3 (25K–1.2M parameters). The paper uses ResNet18 (~11.2M), MobileNetV2 (~2.2M), ShuffleNetV2 (~350K). On CIFAR-10 (3-channel 32×32 RGB), model capacity matters enormously. The FD-CNN models are too small to learn meaningful features from CIFAR-10, capping accuracy regardless of selection method. **Fix: Use ResNet18-FD, MobileNetV2-FD, ShuffleNetV2-FD (must implement MobileNetV2-FD and ShuffleNetV2-FD in `models.py`).**

**RC2 — Channel noise vicious cycle.** Greedy FL selectors (FedCS, Oort, LabelCov) chase "high loss" clients. In FD, high loss often stems from channel noise corruption, not data quality. Selected → noisy logits uploaded → higher loss → selected again. **Evidence:** Spearman rho between FL and FD rankings = -0.024 (zero correlation). FedCS selects 8/10 good-channel clients but achieves only 0.266 accuracy.

**RC3 — Diversity starvation.** Methods that concentrate on a client subset (FedCS: Gini=0.666, Oort: Gini=0.667) produce aggregated logits covering only a fraction of the class distribution. Missing classes degrade distillation. **Evidence:** The 3 fairest methods = the 3 most accurate (rho ≈ 0.85 between fairness rank and accuracy rank).

**RC4 — Per-user SNR degradation at K=N.** Including all N clients in mMIMO forces the base station to split spatial resources across more users, degrading per-user SNR. K=10 outperforms K=30 by 18%. **Evidence:** At K=30 (full participation), all 8 methods converge to ~0.250 (indistinguishable).

**RC5 — Logit fragility.** Unlike FL model weights (millions of parameters, robust to moderate noise), FD logits are 10-dimensional probability vectors directly corrupted by additive Gaussian noise. A single corrupted logit can flip the predicted class.

### What a Good FD Selector Must Do

1. **Avoid the vicious cycle** — never use client loss as the primary selection signal
2. **Maximize logit diversity** — ensure aggregated logits cover all classes
3. **Be channel-aware but not channel-greedy** — filter the worst channels, don't fixate on the best
4. **Promote fair participation** — equitable selection ensures all data distributions are represented
5. **Adapt to noise conditions** — behave differently at -4 dB vs -20 dB

---

## 2. Novel FD-Native Client Selection Methods {#2-novel-methods}

All methods follow the standard selector interface:
```python
def select_clients(round_idx, K, clients, history, rng,
                   time_budget=None, device=None, **kwargs):
    return (selected_ids, scores, state_dict)
```

New package: `csfl_simulator/selection/fd_native/`

---

### Method 1: SNR-Aware Diversity Selector (SNRD) {#method-1-snrd}

**Key Idea.** Jointly optimize for label diversity and channel quality, with an *adaptive tradeoff weight* that shifts based on the estimated noise environment. When channels are harsh → prioritize channel quality (ensure logits arrive intact). When channels are clean → prioritize label diversity (maximize information content).

#### Motivation

FedCS picks 8/10 good-channel clients → 0.266 accuracy. FedAvg picks 3/10 good-channel clients → 0.294 accuracy. Pure channel selection fails because it sacrifices diversity. Pure diversity selection (LabelCov) fails because it ignores channel corruption. **The optimal strategy is a noise-adaptive blend.**

The effective noise variance from Mu et al. Eq. 23d is:

$$\sigma_\omega^2 = \frac{\|\lambda\|_2^2 \sigma_z^2}{2} \cdot \frac{\sigma^2 N_D}{P_{UL} N_{BS}} + \sigma_D^2 + \frac{\|\lambda\|_2^2 \sigma_z^2}{2} \cdot \frac{\sigma^2}{P_{DL}}$$

When $\sigma_\omega^2$ is large relative to the logit variance, a noisy diverse logit is worse than a clean redundant one. SNRD reads the noise environment and smoothly shifts between channel-first and diversity-first selection.

#### Why It Should Work

- **Addresses RC2** (vicious cycle): Never uses client loss as a signal — only channel quality and label histograms
- **Addresses RC3** (diversity starvation): Diversity is always part of the objective, even in noisy conditions
- **Addresses RC5** (logit fragility): At high noise, channel-quality weight increases, filtering corrupted logits
- **Novel aspect**: No existing method adapts its selection criterion to the noise environment

#### Algorithm

```
Hyperparameters:
  w_fairness = 0.15          # fairness weight (always on)
  noise_threshold = 1.0      # sigmoid midpoint for noise ratio
  channel_ema_alpha = 0.3    # EMA smoothing for channel quality

State (persisted in history["state"]):
  channel_quality_ema[c]     # smoothed channel quality per client
  last_logit_var             # logit variance from last round

Each round:
  1. NOISE ESTIMATION
     effective_noise = channel.effective_noise_variance(K, last_logit_var)
     noise_ratio = effective_noise / (last_logit_var + 1e-8)
     w_channel = sigmoid(noise_ratio - noise_threshold)   # 0→1 as noise grows
     w_diversity = 1.0 - w_channel - w_fairness

  2. PER-CLIENT SCORING
     For each candidate client c:
       # Channel score: EMA-smoothed, normalized to [0,1]
       channel_ema[c] = alpha * c.channel_quality + (1-alpha) * channel_ema[c]
       channel_score = normalize(channel_ema[c])

       # Fairness score: recency bonus
       gap = round_idx - c.last_selected_round
       fairness_score = gap / (gap + max(N/K, 3))

  3. GREEDY DIVERSITY-AWARE SELECTION
     selected = []
     label_coverage = zeros(num_classes)
     For i in 1..K:
       For each unselected c:
         # Marginal label coverage gain (IDF-weighted)
         diversity_gain = sum(idf[cls] for cls in c.label_histogram
                              if label_coverage[cls] < threshold)
         diversity_score = diversity_gain / (sum(idf) + eps)

         score = w_channel * channel_score[c]
               + w_diversity * diversity_score
               + w_fairness * fairness_score[c]
       Select c* = argmax(score)
       selected.append(c*)
       label_coverage += c*.label_histogram

  4. STATE UPDATE
     Store: channel_ema, last_logit_var (updated post-aggregation via hook)

  Return: (selected, scores, state)
```

#### File: `csfl_simulator/selection/fd_native/snr_diversity.py`
#### Registry key: `fd_native.snr_diversity`

---

### Method 2: Logit-Quality Thompson Sampling (LQTS) {#method-2-lqts}

**Key Idea.** Extend APEX v2's Thompson sampling by replacing the reward signal. Instead of using global accuracy improvement (which is noisy and confounded by channel effects), use **per-client logit quality** as the reward: how much each client's logits improved the aggregated distribution. This directly measures what matters in FD — the information value of each client's contribution to the distillation pool.

#### Motivation

APEX v2 is already the most robust method (9.2% noise degradation, 0.4% heterogeneity degradation) thanks to Thompson sampling's exploration. But its reward signal (global composite score, distributed uniformly to all selected clients) is noisy and doesn't distinguish between a client that contributed valuable logits vs. one that contributed noise. LQTS makes Thompson sampling *FD-aware* by measuring each client's logit contribution directly.

**Key insight from experiments:** The server-client accuracy gap varies by method (MAML: 0.152 gap, APEX v2: 0.098, FedAvg: 0.089). This gap reflects how well selected clients' logits serve the distillation process. A per-client logit-quality reward would directly optimize for closing this gap.

#### Why It Should Work

- **Addresses RC2** (vicious cycle): A client with bad channel → corrupted logits → low logit-quality reward → lower selection probability. The vicious cycle is *inverted* into a virtuous cycle.
- **Addresses RC3** (diversity): Diverse logits increase aggregation quality → higher rewards for diverse clients
- **Builds on proven foundation**: Thompson sampling (APEX v2) is already the most noise-robust mechanism. LQTS makes it channel-aware without explicit channel modeling.
- **Novel aspect**: First Thompson sampler that uses logit-level contribution as reward. Existing bandit methods use loss or accuracy.

#### Algorithm

```
Hyperparameters:
  ema_alpha = 0.3           # reward EMA smoothing
  variance_floor_scale = 0.1 # posterior variance floor
  w_diversity = 0.25         # diversity weight in final score
  w_recency = 0.15           # fairness/recency weight

State (persisted in history["state"]):
  mu[c]                     # posterior mean per client (init 0.5)
  sigma2[c]                 # posterior variance per client (init 1.0)
  n[c]                      # observation count per client
  ema_reward[c]             # EMA of logit-quality reward

Each round:
  1. RETRIEVE LOGIT-QUALITY REWARDS FROM LAST ROUND
     # Computed by FD simulator hook (post-aggregation):
     # For each selected client c_i last round:
     #   reward_i = cosine_sim(logits_i, aggregated_logits) * data_weight_i
     #   Alternatively: leave-one-out contribution = ||agg_with_i - agg_without_i||
     rewards = history["state"].get("fd_logit_rewards", {})

  2. UPDATE THOMPSON POSTERIORS
     For each client c with a reward from last round:
       raw_reward = rewards[c]
       ema_reward[c] = ema_alpha * raw_reward + (1 - ema_alpha) * ema_reward[c]
       n[c] += 1
       mu[c] = ema_reward[c]  # EMA-smoothed mean
       sigma2[c] = max(variance_floor_scale / sqrt(n[c]), online_variance)

  3. THOMPSON SAMPLING
     For each client c:
       sample[c] = rng.normal(mu[c], sqrt(sigma2[c] / max(n[c], 1)))

  4. DIVERSITY-AUGMENTED GREEDY SELECTION
     Sort clients by sample[c] (descending)
     selected = []
     For i in 1..K:
       For each unselected c (in Thompson sample order):
         # Recency bonus
         gap = round_idx - c.last_selected_round
         recency = gap / (gap + max(N/K, 3))

         # Diversity: min cosine distance to already-selected
         if selected:
           div = min_cosine_distance(c.label_histogram, [s.label_histogram for s in selected])
         else:
           div = 1.0

         score = (1 - w_diversity - w_recency) * sample[c]
               + w_diversity * div
               + w_recency * recency
       Select c* = argmax(score)
       selected.append(c*)

  Return: (selected, scores, state_with_updated_posteriors)
```

#### FD Simulator Hook Required

Add to `fd_simulator.py` after line 730 (post client logit collection, pre aggregation):

```python
# Compute per-client logit contribution for LQTS/LEM
if client_logits and len(client_logits) > 1:
    fd_logit_stats = {}
    all_logits = torch.stack([client_logits[cid] for cid in ids])  # (K, N_pub, C)
    mean_logits = all_logits.mean(dim=0)  # (N_pub, C)
    for cid in ids:
        cl = client_logits[cid]
        # Cosine similarity to aggregated mean
        cos_sim = F.cosine_similarity(cl.flatten().unsqueeze(0),
                                       mean_logits.flatten().unsqueeze(0)).item()
        # Logit entropy
        probs = F.softmax(cl, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
        # Entropy variance across samples
        ent_per_sample = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        entropy_var = ent_per_sample.var().item()
        fd_logit_stats[cid] = {
            "cosine_to_mean": cos_sim,
            "entropy_mean": entropy,
            "entropy_var": entropy_var,
        }
    self.history["state"]["fd_logit_stats"] = fd_logit_stats
    self.history["state"]["fd_logit_rewards"] = {
        cid: stats["cosine_to_mean"] * (self.clients[cid].data_size / sum(self.clients[i].data_size for i in ids))
        for cid, stats in fd_logit_stats.items()
    }
```

#### File: `csfl_simulator/selection/fd_native/logit_quality_ts.py`
#### Registry key: `fd_native.logit_quality_ts`

---

### Method 3: Noise-Robust Fair Scheduler (NRFS) {#method-3-nrfs}

**Key Idea.** Exploit the fairness-accuracy correlation by design. Use a **deterministic round-robin backbone** that guarantees near-equal participation, but perturb the schedule based on channel conditions. Clients with temporarily-bad channels are **deferred** (not excluded) by 1–2 rounds. A hard fairness override ensures no client is starved beyond a maximum deferral count.

#### Motivation

The most striking finding from our experiments is that **fairness drives accuracy in FD** (the opposite of FL). The top-3 methods by Gini coefficient (MAML 0.036, FedAvg 0.060, APEX v2 0.229) are the top-3 by accuracy. This means the optimal FD selector should distribute participation as evenly as possible — i.e., a round-robin.

But pure round-robin ignores channel conditions, wasting communication on clients with terrible channels. NRFS is **"round-robin + channel-aware deferral"** — the simplest possible method that could work, grounded directly in the experimental evidence.

Since `simulate_round_env()` applies a random walk to channel quality (±0.05 per round), a client with bad channel this round likely recovers within 1–2 rounds. Deferral is therefore low-cost.

#### Why It Should Work

- **Addresses RC3** (diversity starvation): Round-robin backbone guarantees all clients participate equally, maximizing logit diversity
- **Addresses RC4** (K<N beats K=N): Selects exactly K<N clients, but rotates through all N over ceil(N/K) rounds
- **Addresses RC5** (logit fragility): Defers worst-channel clients to avoid wasting communication on corrupted logits
- **Addresses RC2** (vicious cycle): No loss-based signal at all — selection is purely schedule + channel-based
- **Novel aspect**: No existing FL method combines scheduled fairness with channel deferral. This is the first method designed from the fairness-accuracy correlation.
- **Simplicity**: No ML training, no posteriors, no learned parameters. A lightweight heuristic grounded in theory.

#### Algorithm

```
Hyperparameters:
  max_defer = 3              # maximum consecutive deferrals before forced participation
  channel_percentile = 20    # defer bottom X% of channels

State (persisted in history["state"]):
  deficit[c]                 # participation deficit (higher = more overdue)
  defer_count[c]             # consecutive deferrals

Each round:
  1. UPDATE DEFICITS
     For each client c:
       deficit[c] += 1.0 / K   # expected selection rate per round

  2. DETERMINE CHANNEL THRESHOLD
     channel_values = [c.channel_quality for c in clients]
     threshold = percentile(channel_values, channel_percentile)  # bottom 20% cutoff

  3. BUILD PRIORITY QUEUE
     For each client c:
       priority = deficit[c]   # simple: most overdue = highest priority

  4. GREEDY SELECTION WITH DEFERRAL
     Sort clients by priority (descending)
     selected = []
     For c in sorted_clients:
       if len(selected) == K: break

       if c.channel_quality < threshold AND defer_count[c] < max_defer:
         # Defer this client — they'll get higher priority next round
         defer_count[c] += 1
         continue
       else:
         # Select (either good channel, or fairness override after max_defer)
         selected.append(c)
         deficit[c] -= 1.0    # reduce deficit
         defer_count[c] = 0   # reset deferral count

  5. BACKFILL (if fewer than K selected due to mass deferral)
     If len(selected) < K:
       # Fill remaining slots from deferred clients by priority
       remaining = [c for c in sorted_clients if c not in selected]
       selected.extend(remaining[:K - len(selected)])

  Return: (selected_ids, [deficit[c] for c in selected], state)
```

#### File: `csfl_simulator/selection/fd_native/noise_robust_fair.py`
#### Registry key: `fd_native.noise_robust_fair`

---

### Method 4: Logit Entropy Maximization (LEM) {#method-4-lem}

**Key Idea.** Select clients whose logit predictions on the public dataset have the highest **informative entropy** — i.e., the most uncertain but structured predictions. These clients contribute the most useful gradients during server distillation. Crucially, distinguish between *genuine model uncertainty* (high entropy with high variance across samples) and *noise-corrupted logits* (near-uniform entropy with low variance).

#### Motivation

In FD, each client generates logits by running inference on the shared public dataset. These logits encode the client's "knowledge" about class boundaries. A client that is highly uncertain but in a structured way (e.g., confidently knows some classes but not others) produces the most informative logits for aggregation. A client whose logits are near-uniform (`entropy ≈ log(C)`) provides no useful signal — either its model hasn't learned, or its logits were destroyed by channel noise.

**Key insight**: Noise-corrupted logits and genuinely uninformative logits have a signature: `entropy ≈ log(C)` with `variance across samples ≈ 0`. Genuine uncertainty has `entropy < log(C)` with `variance > 0` (different samples produce different prediction confidence levels).

#### Why It Should Work

- **Addresses RC2** (vicious cycle): Selection signal is logit entropy, not client loss. High-loss clients with corrupted logits will have near-uniform entropy (detectable noise) and be penalized.
- **Addresses RC5** (logit fragility): Explicit noise filter detects corrupted logits by their statistical signature
- **Addresses RC3** (diversity): High-entropy clients with structured uncertainty are precisely those with partial class knowledge — selecting them maximizes class coverage in the logit pool
- **Novel aspect**: First client selection method that uses the actual logit outputs as the primary selection signal. All existing methods use loss, gradient, or system metrics.

#### Algorithm

```
Hyperparameters:
  noise_penalty = 0.5        # penalty for suspected noise-corrupted logits
  entropy_ema_alpha = 0.3    # EMA smoothing
  w_fairness = 0.15          # fairness weight
  max_entropy_ratio = 0.95   # entropy/log(C) threshold for noise detection
  min_entropy_var = 0.01     # variance threshold for noise detection

State (persisted in history["state"]):
  entropy_ema[c]             # EMA of mean logit entropy per client
  entropy_var_ema[c]         # EMA of logit entropy variance per client
  last_participation[c]      # round of last participation

Each round:
  1. RETRIEVE LOGIT STATS FROM LAST ROUND
     # Computed by FD simulator hook (see LQTS section)
     stats = history["state"].get("fd_logit_stats", {})
     For each client c that participated last round:
       entropy_ema[c] = alpha * stats[c]["entropy_mean"] + (1-alpha) * entropy_ema[c]
       entropy_var_ema[c] = alpha * stats[c]["entropy_var"] + (1-alpha) * entropy_var_ema[c]

  2. NOISE DETECTION
     max_entropy = log(num_classes)  # e.g., log(10) ≈ 2.303
     For each client c:
       ratio = entropy_ema[c] / max_entropy
       is_noisy = (ratio > max_entropy_ratio) AND (entropy_var_ema[c] < min_entropy_var)

  3. INFORMATION SCORING
     For each client c:
       if is_noisy[c]:
         info_score = entropy_ema[c] * (1 - noise_penalty)
       else:
         # Reward high entropy with high variance (genuine uncertainty)
         info_score = entropy_ema[c] * (1 + entropy_var_ema[c])

       # Fairness bonus
       gap = round_idx - last_participation.get(c, -1)
       fairness = gap / (gap + max(N/K, 3))

       score = (1 - w_fairness) * normalize(info_score) + w_fairness * fairness

  4. SELECT TOP-K
     Sort by score, select top K
     For unseen clients (no entropy data yet): assign median score + exploration bonus

  Return: (selected_ids, scores, state)
```

#### File: `csfl_simulator/selection/fd_native/logit_entropy_max.py`
#### Registry key: `fd_native.logit_entropy_max`

---

### Method Comparison Summary

| Method | Selection Signal | Channel Awareness | Fairness | ML Component | Complexity |
|--------|-----------------|-------------------|----------|-------------|------------|
| SNRD | Label histogram + channel quality | Explicit (adaptive weight) | Recency bonus | None | Low |
| LQTS | Logit quality (cosine to aggregated) | Implicit (via reward) | Thompson exploration | Thompson sampling | Medium |
| NRFS | Participation deficit | Deferral filter | Structural (round-robin) | None | Low |
| LEM | Logit entropy + noise filter | Implicit (noise detection) | Recency bonus | None | Low |
| *APEX v2 (baseline)* | *Composite (loss, grad, diversity)* | *None* | *Thompson exploration* | *Thompson + MLP* | *High* |
| *FedAvg (baseline)* | *None (random)* | *None* | *Inherent* | *None* | *None* |

---

## 3. Metrics for Holistic Evaluation {#3-metrics}

### Currently Tracked Metrics (per round)

| Category | Metric | Key | Description |
|----------|--------|-----|-------------|
| **Core Performance** | Testing Accuracy | `accuracy` | Global test accuracy (client-averaged for FD) |
| | Loss | `loss` | Cross-entropy loss on test set |
| | F1 Score | `f1` | Macro-averaged F1 |
| | Precision | `precision` | Macro-averaged precision |
| | Recall | `recall` | Macro-averaged recall |
| **FD Distillation** | KL Divergence | `kl_divergence_avg` | Average KL divergence during distillation (lower = better knowledge transfer) |
| | Distillation Loss | `distillation_loss_avg` | Server-side distillation loss |
| | Dynamic Steps | `dynamic_steps_kr` | Current $K_r$ value (training steps per round) |
| **Communication** | Logit Communication | `logit_comm_kb` | Per-round logit payload (KB) |
| | FL Equivalent Comm | `fl_equiv_comm_mb` | What this would cost in FL (MB) |
| | Comm Reduction Ratio | `comm_reduction_ratio` | FD/FL communication ratio |
| | Cumulative Comm | `cum_comm` | Total communication so far (MB) |
| **Channel** | Effective Noise Var | `effective_noise_var` | $\sigma_\omega^2$ from Eq. 23d |
| | Good Channel Count | `num_good_channel` | Selected clients with good channel |
| | Bad Channel Count | `num_bad_channel` | Selected clients with bad channel |
| **Fairness** | Fairness Gini | `fairness_gini` | Gini coefficient of participation counts |
| | Fairness Variance | `fairness_var` | Variance of participation counts |
| | Client Accuracy Avg | `client_accuracy_avg` | Mean accuracy across individual clients |
| | Client Accuracy Std | `client_accuracy_std` | Std dev of client accuracies (equity of learning) |
| **Timing** | Selection Time | `selection_time` | Time spent on client selection (s) |
| | Compute Time | `compute_time` | Time for training + distillation (s) |
| | Round Time | `round_time` | Total round time (s) |
| | Wall Clock | `wall_clock` | Cumulative wall time (s) |
| **Composite** | Composite Score | `composite` | Weighted combo (acc + time + fairness) |
| | Reward | `reward` | Per-round composite delta |
| | Server Accuracy | `server_accuracy` | Server model's test accuracy |

### NEW Metrics to Add (Required for This Study)

| Category | Metric | Key | Description | Why Needed |
|----------|--------|-----|-------------|------------|
| **Logit Quality** | Logit Entropy Avg | `logit_entropy_avg` | Mean entropy of selected clients' logits over public dataset | Measures information richness of logit pool; LEM's primary signal |
| **Logit Quality** | Logit Cosine Diversity | `logit_cosine_diversity` | Mean pairwise cosine distance among selected clients' logits | Measures how different clients' predictions are; diversity metric for logit space |
| **Logit Quality** | Logit Entropy Variance | `logit_entropy_var` | Variance of per-sample entropy across public dataset (averaged over clients) | Distinguishes genuine uncertainty from noise (key for noise detection) |
| **Per-Class** | Per-Class Accuracy Std | `per_class_accuracy_std` | Std dev of per-class accuracy (averaged across clients) | Measures whether model learns all classes equally or just majority classes |
| **Fairness** | Participation Gini | `participation_gini` | Gini coefficient computed on actual participation counts (not just variance) | More interpretable fairness measure than variance; directly correlates with FD accuracy |
| **Selection** | Channel Quality Selected Avg | `channel_quality_selected_avg` | Mean channel quality of selected clients | Measures whether selection biases toward good/bad channels |
| **Selection** | Label Coverage Ratio | `label_coverage_ratio` | Fraction of classes represented by at least one selected client per round | Measures class diversity in the logit pool |
| **Convergence** | Server-Client Accuracy Gap | `server_client_gap` | `server_accuracy - client_accuracy_avg` | Measures bidirectional distillation effectiveness; large gap = one-sided knowledge flow |

### Metric Groups for Paper Figures

| Figure Type | Metrics to Plot |
|-------------|----------------|
| **Main convergence** | `accuracy`, `server_accuracy` |
| **Distillation quality** | `kl_divergence_avg`, `distillation_loss_avg`, `logit_entropy_avg` |
| **Channel impact** | `effective_noise_var`, `num_good_channel`, `channel_quality_selected_avg` |
| **Fairness analysis** | `fairness_gini`, `client_accuracy_std`, `participation_gini` |
| **Communication** | `cum_comm`, `comm_reduction_ratio` |
| **Logit diversity** | `logit_cosine_diversity`, `label_coverage_ratio` |
| **Convergence quality** | `server_client_gap`, `per_class_accuracy_std` |

---

## 4. Experiment Suite {#4-experiments}

### Prerequisites (Code Changes Before Running)

1. **Implement MobileNetV2-FD and ShuffleNetV2-FD** in `csfl_simulator/core/models.py`
2. **Implement 4 FD-native selectors** in `csfl_simulator/selection/fd_native/`
3. **Add FD simulator logit stats hook** in `csfl_simulator/core/fd_simulator.py`
4. **Add new metrics** (logit_entropy_avg, logit_cosine_diversity, etc.) to FD simulator
5. **Register methods** in `presets/methods.yaml`
6. **Update plot script** with new method names/colors in `scripts/plot_fd_experiments.py`

### Paper-Matched Base Configuration

From Mu et al. Table III and Section VI:

| Parameter | Paper Value | Our Config |
|-----------|------------|------------|
| N (total clients) | 15 | **50** (scaled up) |
| K (per round) | 15 (full) | **15** (partial — our contribution) |
| R (rounds) | 200 | **300** (scaled up for convergence) |
| Models (CIFAR-10) | ResNet18, MobileNetV2, ShuffleNetV2 | Same |
| Models (MNIST) | CNN_1, CNN_2, CNN_3 | FD-CNN1, FD-CNN2, FD-CNN3 |
| Public dataset | STL-10 (2000) / FMNIST (2000) | Same |
| Batch size (train) | 128 | 128 |
| Batch size (distill) | 500 | 500 |
| Learning rate | 0.001 (Adam) | 0.001 (Adam) |
| Epochs | 2 | 2 |
| UL SNR | -8 dB | -8 dB |
| DL SNR | -20 dB | -20 dB |
| N_BS antennas | 64 | 64 |
| Quantization | 8-bit | 8-bit |
| Alpha | 0.5 | 0.5 |
| Dynamic steps | Yes (base=5, period=25) | Yes |
| Temperature | 1.0 | 1.0 |

### Method List for Experiments

**Baselines (FL-adapted):**
- `heuristic.random` — Random selection (FedAvg)
- `system_aware.fedcs` — FedCS (system-aware, FL champion)
- `system_aware.oort` — Oort (UCB-based exploration)
- `heuristic.label_coverage` — LabelCoverage (diversity, FL champion)
- `ml.maml_select` — MAML (meta-learned policy)
- `ml.apex_v2` — APEX v2 (Thompson + diversity)

**Novel FD-native:**
- `fd_native.snr_diversity` — SNRD (channel-adaptive diversity)
- `fd_native.logit_quality_ts` — LQTS (logit-quality Thompson sampling)
- `fd_native.noise_robust_fair` — NRFS (fair scheduling + channel deferral)
- `fd_native.logit_entropy_max` — LEM (logit entropy maximization)

---

### Experiment 1: Main Method Comparison — CIFAR-10 (The Headline Result)

**Goal:** Demonstrate FD-native methods outperform FL-adapted methods on CIFAR-10/STL-10 with paper-matched models.

**PowerShell (one line):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_main --methods "heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.maml_select,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --distillation-epochs 2 --temperature 1.0 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python -m csfl_simulator plot --run fd_cifar10_main --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini --format eps --width 7.16 --height 5.0; python scripts/plot_fd_experiments.py --run fd_cifar10_main --metrics accuracy,server_accuracy,logit_entropy_avg,logit_cosine_diversity --format eps --bar
```

**Expected output:** Accuracy convergence curves (10 methods), final accuracy bar chart, 4-panel multi-metric figure.

---

### Experiment 2: FL vs FD Ranking Inversion Proof

**Goal:** Run the same methods in FL mode to demonstrate ranking inversion. This is the "motivation" figure.

**PowerShell:**
```powershell
python -m csfl_simulator compare --paradigm fl --name fl_cifar10_baseline --methods "heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.maml_select,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model LightCNN --total-clients 50 --clients-per-round 15 --rounds 300 --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --run fl_cifar10_baseline --metrics accuracy --format eps; python scripts/plot_fd_experiments.py --run fd_cifar10_main --metrics accuracy --format eps
```

**Analysis:** Compute Spearman rank correlation between FL and FD final accuracy rankings.

---

### Experiment 3: Noise Sensitivity Sweep

**Goal:** Show FD-native methods degrade gracefully as DL SNR worsens, while FL methods collapse.

**PowerShell (4 noise levels + error-free):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_noise_errfree --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_noise_dl0 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db 0 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_noise_dl10 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -10 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_noise_dl20 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_noise_dl30 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -30 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot (multi-panel):**
```powershell
python scripts/plot_fd_experiments.py --runs "fd_cifar10_noise_errfree,fd_cifar10_noise_dl0,fd_cifar10_noise_dl10,fd_cifar10_noise_dl20,fd_cifar10_noise_dl30" --panel-metric accuracy --panel-labels "Error-Free,DL 0dB,DL -10dB,DL -20dB,DL -30dB" --format eps --out-dir paper/figures
```

---

### Experiment 4: Non-IID Heterogeneity Sweep (Alpha)

**Goal:** Show robustness across data heterogeneity levels.

**PowerShell (6 alpha values):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha01 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.1 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha03 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.3 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha05 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha1 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 1.0 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha5 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 5.0 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_alpha10 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 10.0 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --runs "fd_cifar10_alpha01,fd_cifar10_alpha03,fd_cifar10_alpha05,fd_cifar10_alpha1,fd_cifar10_alpha5,fd_cifar10_alpha10" --panel-metric accuracy --panel-labels "a=0.1,a=0.3,a=0.5,a=1.0,a=5.0,a=10.0" --format eps --out-dir paper/figures
```

---

### Experiment 5: K Sweep (Selection Ratio)

**Goal:** Validate K<N advantage and find optimal K for FD-native methods.

**PowerShell (5 K values):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_K5 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair,fd_native.logit_quality_ts" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 5 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_K10 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair,fd_native.logit_quality_ts" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_K15 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair,fd_native.logit_quality_ts" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 25 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_K20 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair,fd_native.logit_quality_ts" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 20 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_K30 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair,fd_native.logit_quality_ts" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 50 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --runs "fd_cifar10_K5,fd_cifar10_K10,fd_cifar10_K15,fd_cifar10_K20,fd_cifar10_K30" --panel-metric accuracy --panel-labels "K=5,K=10,K=15,K=20,K=30" --format eps --out-dir paper/figures
```

---

### Experiment 6: Scaling to N=100

**Goal:** Show methods scale to larger client populations.

**PowerShell:**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_N100 --methods "heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 100 --clients-per-round 30 --rounds 400 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python -m csfl_simulator plot --run fd_cifar10_N100 --metrics accuracy,kl_divergence_avg,fairness_gini --format eps
```

---

### Experiment 7: Group-Based FD (FedTSKD-G)

**Goal:** Test whether FedTSKD-G grouping helps or hurts FD-native methods.

**PowerShell (without grouping — reuse Exp 1, with grouping below):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_group --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --group-based --channel-threshold 0.5 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python -m csfl_simulator plot --run fd_cifar10_group --metrics accuracy,kl_divergence_avg --format eps
```

---

### Experiment 8: MNIST/FMNIST Cross-Dataset Validation

**Goal:** Verify results generalize to a simpler task with different model architectures.

**PowerShell:**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_mnist_main --methods "heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.maml_select,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset MNIST --public-dataset FMNIST --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --total-clients 30 --clients-per-round 10 --rounds 200 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --distillation-epochs 2 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python -m csfl_simulator plot --run fd_mnist_main --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini --format eps --width 7.16 --height 5.0; python scripts/plot_fd_experiments.py --run fd_mnist_main --metrics accuracy --format eps --bar
```

---

### Experiment 9: Communication Efficiency (FD vs FL)

**Goal:** Produce accuracy-vs-communication plot demonstrating FD maintains massive savings while FD-native methods close the accuracy gap.

**PowerShell (FL baseline for comparison):**
```powershell
python -m csfl_simulator compare --paradigm fl --name fl_cifar10_comm --methods "heuristic.random,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model LightCNN --total-clients 50 --clients-per-round 15 --rounds 300 --no-fast-mode --seed 42
```

**Plot (combined with Exp 1 FD results):**
```powershell
python scripts/plot_fd_experiments.py --run fd_cifar10_main --metrics accuracy,cum_comm,comm_reduction_ratio --format eps; python scripts/plot_fd_experiments.py --run fl_cifar10_comm --metrics accuracy,cum_comm --format eps
```

---

### Experiment 10: Antenna Count Sweep

**Goal:** Match paper's Fig. 11 — show how N_BS affects FD accuracy with client selection.

**PowerShell (3 antenna configs):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_ant32 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 32 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_ant64 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_ant128 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 128 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --runs "fd_cifar10_ant32,fd_cifar10_ant64,fd_cifar10_ant128" --panel-metric accuracy --panel-labels "32 BS Ant,64 BS Ant,128 BS Ant" --format eps --out-dir paper/figures
```

---

### Experiment 11: Ablation Studies

**Goal:** Validate each component of SNRD and LQTS through ablation.

**SNRD Ablations:**
- `fd_native.snrd_ablation_fixed_w` — Fixed w_channel=0.5 (no noise adaptation)
- `fd_native.snrd_ablation_no_channel` — w_channel=0 (diversity only, like LabelCov)
- `fd_native.snrd_ablation_no_diversity` — w_diversity=0 (channel only, like FedCS)
- `fd_native.snrd_ablation_no_fairness` — w_fairness=0 (no recency bonus)

**LQTS Ablations:**
- `fd_native.lqts_ablation_global_reward` — Use global accuracy delta as reward (like APEX v2)
- `fd_native.lqts_ablation_no_diversity` — No diversity bonus in greedy selection
- `fd_native.lqts_ablation_no_recency` — No fairness/recency term

**PowerShell:**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_ablation_snrd --methods "fd_native.snr_diversity,fd_native.snrd_ablation_fixed_w,fd_native.snrd_ablation_no_channel,fd_native.snrd_ablation_no_diversity,fd_native.snrd_ablation_no_fairness" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_ablation_lqts --methods "fd_native.logit_quality_ts,fd_native.lqts_ablation_global_reward,fd_native.lqts_ablation_no_diversity,fd_native.lqts_ablation_no_recency" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --run fd_cifar10_ablation_snrd --metrics accuracy --format eps --bar; python scripts/plot_fd_experiments.py --run fd_cifar10_ablation_lqts --metrics accuracy --format eps --bar
```

---

### Experiment 12: Multi-Seed Statistical Significance

**Goal:** Run the main comparison with 5 seeds to report mean ± std and run statistical tests.

**PowerShell (5 seeds for main CIFAR-10 experiment):**
```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_seed0 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 0
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_seed1 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 1
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_seed2 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 2
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_seed100 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 100
```

```powershell
python -m csfl_simulator compare --paradigm fd --name fd_cifar10_seed42 --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" --dataset CIFAR-10 --public-dataset STL-10 --public-dataset-size 2000 --partition dirichlet --dirichlet-alpha 0.5 --model-heterogeneous --model-pool "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD" --total-clients 50 --clients-per-round 15 --rounds 300 --channel-noise --ul-snr-db -8 --dl-snr-db -20 --dynamic-steps --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --no-fast-mode --seed 42
```

---

### Experiment Summary Table

| # | Name | Variables | Methods | Rounds | Key Output |
|---|------|-----------|---------|--------|------------|
| 1 | Main comparison | Baseline config | 10 | 300 | Accuracy convergence, bar chart |
| 2 | FL vs FD inversion | Paradigm (FL vs FD) | 10 | 300 | Spearman rho, rank shift table |
| 3 | Noise sweep | DL SNR: err-free, 0, -10, -20, -30 | 7 | 300 | Accuracy vs SNR, degradation % |
| 4 | Alpha sweep | Alpha: 0.1, 0.3, 0.5, 1, 5, 10 | 7 | 300 | Accuracy vs alpha |
| 5 | K sweep | K: 5, 10, 15, 25, 50 | 6 | 300 | Accuracy vs K, optimal K |
| 6 | Scaling N=100 | N=100, K=30 | 7 | 400 | Scalability validation |
| 7 | Group-based | With/without FedTSKD-G | 6 | 300 | Grouping delta per method |
| 8 | MNIST cross-val | Dataset=MNIST | 10 | 200 | Cross-dataset validation |
| 9 | Comm efficiency | FL vs FD paradigm | 2 | 300 | Accuracy vs cum_comm |
| 10 | Antenna sweep | N_BS: 32, 64, 128 | 4 | 300 | Accuracy vs antenna count |
| 11 | Ablation | SNRD/LQTS components | 5+4 | 300 | Component contribution |
| 12 | Multi-seed | Seed: 0, 1, 2, 42, 100 | 6 | 300 | Mean ± std, significance |

**Total experiment runs:** ~55 compare runs (each with 4-10 methods internally).
**Estimated compute (with AMP, single GPU):** ~200-400 GPU-hours on RTX 3090/4090.

---

## 5. Paper Narrative {#5-narrative}

### Proposed Title

**"Client Selection for Federated Distillation in Massive MIMO Networks: Why FL Methods Fail and How to Fix Them"**

or

**"FD-Native Client Selection: Channel-Aware, Diversity-Driven Strategies for Federated Distillation over Imperfect mMIMO Links"**

### Section Structure

| Section | Experiments Used | Key Claim |
|---------|-----------------|-----------|
| I. Introduction | — | FD is communication-efficient but client selection is unexplored |
| II. System Model | — | mMIMO FD framework (builds on Mu et al.) |
| III. The FL-to-FD Transfer Problem | Exp 1, 2 | FL selection methods fail in FD: ranking inversion (rho ≈ 0) |
| IV. Root Cause Analysis | Exp 3, Exp 10 | Channel noise creates vicious cycle; per-user SNR degrades with N |
| V. Proposed FD-Native Methods | — | SNRD, LQTS, NRFS, LEM algorithms |
| VI. Convergence Analysis | — | Theoretical bounds for FD with client selection |
| VII. Simulation Results | Exp 1, 3-12 | FD-native methods achieve 15-25% higher accuracy |
| VII-A. Main comparison | Exp 1, 8 | Headline table: 10 methods × 2 datasets |
| VII-B. Noise robustness | Exp 3, 10 | Degradation curves |
| VII-C. Heterogeneity | Exp 4 | Alpha sweep |
| VII-D. Scaling | Exp 5, 6 | K sweep + N=50 |
| VII-E. Grouping synergy | Exp 7 | FedTSKD-G interaction |
| VII-F. Fairness-accuracy | Exp 1, 8 | Pareto front |
| VII-G. Communication | Exp 9 | FD vs FL efficiency maintained |
| VII-H. Ablation | Exp 11 | Component contribution |
| VIII. Conclusion | — | FD requires FD-native selection; SNRD/LQTS recommended |

### Key Figures for Paper (IEEE 2-column format)

| Fig | Type | Size | Content |
|-----|------|------|---------|
| 1 | System diagram | Double-col | mMIMO FD architecture with client selection |
| 2 | Rank shift | Double-col | FL vs FD rankings side-by-side (2 panels) |
| 3 | Convergence | Double-col | 10-method accuracy curves (300 rounds) |
| 4 | Noise sweep | Double-col | 4-panel: accuracy at err-free, -10, -20, -30 dB |
| 5 | Alpha sweep | Double-col | 3-panel: accuracy at alpha=0.1, 0.5, 10 |
| 6 | K sweep | Single-col | Accuracy vs K line plot |
| 7 | Fairness-accuracy | Single-col | Scatter with Pareto front |
| 8 | Ablation | Single-col | Bar chart (SNRD + LQTS components) |
| 9 | Communication | Single-col | Accuracy vs cumulative comm (FD + FL) |

### Key Tables

| Table | Content |
|-------|---------|
| I | Comparison with prior FD works (extends Mu et al. Table I) |
| II | Model architectures (CNN1/2/3, ResNet18, MobileNetV2, ShuffleNetV2) |
| III | Main results: 10 methods × {accuracy, KL div, noise var, Gini, comm} |
| IV | Noise degradation %: 7 methods × 5 noise levels |
| V | Cross-dataset results (CIFAR-10 + MNIST) |
| VI | Ablation results (SNRD + LQTS variants) |

---

## Appendix: Registry Entries for methods.yaml

```yaml
  # --- FD-Native Methods ---
  - key: fd_native.snr_diversity
    module: csfl_simulator.selection.fd_native.snr_diversity
    display_name: "SNRD"
    origin: proposed
    params:
      w_fairness: 0.15
      noise_threshold: 1.0
      channel_ema_alpha: 0.3
    type: fd_native
    trainable: false

  - key: fd_native.logit_quality_ts
    module: csfl_simulator.selection.fd_native.logit_quality_ts
    display_name: "LQTS"
    origin: proposed
    params:
      ema_alpha: 0.3
      variance_floor_scale: 0.1
      w_diversity: 0.25
      w_recency: 0.15
    type: fd_native
    trainable: true

  - key: fd_native.noise_robust_fair
    module: csfl_simulator.selection.fd_native.noise_robust_fair
    display_name: "NRFS"
    origin: proposed
    params:
      max_defer: 3
      channel_percentile: 20
    type: fd_native
    trainable: false

  - key: fd_native.logit_entropy_max
    module: csfl_simulator.selection.fd_native.logit_entropy_max
    display_name: "LEM"
    origin: proposed
    params:
      noise_penalty: 0.5
      entropy_ema_alpha: 0.3
      w_fairness: 0.15
      max_entropy_ratio: 0.95
      min_entropy_var: 0.01
    type: fd_native
    trainable: false

  # --- SNRD Ablation Variants ---
  - key: fd_native.snrd_ablation_fixed_w
    module: csfl_simulator.selection.fd_native.snr_diversity
    display_name: "SNRD (fixed w)"
    origin: ablation
    params:
      w_fairness: 0.15
      fixed_w_channel: 0.5
    type: fd_native
    trainable: false

  - key: fd_native.snrd_ablation_no_channel
    module: csfl_simulator.selection.fd_native.snr_diversity
    display_name: "SNRD (no channel)"
    origin: ablation
    params:
      w_fairness: 0.15
      fixed_w_channel: 0.0
    type: fd_native
    trainable: false

  - key: fd_native.snrd_ablation_no_diversity
    module: csfl_simulator.selection.fd_native.snr_diversity
    display_name: "SNRD (no diversity)"
    origin: ablation
    params:
      w_fairness: 0.15
      fixed_w_channel: 0.85
    type: fd_native
    trainable: false

  - key: fd_native.snrd_ablation_no_fairness
    module: csfl_simulator.selection.fd_native.snr_diversity
    display_name: "SNRD (no fairness)"
    origin: ablation
    params:
      w_fairness: 0.0
      noise_threshold: 1.0
    type: fd_native
    trainable: false

  # --- LQTS Ablation Variants ---
  - key: fd_native.lqts_ablation_global_reward
    module: csfl_simulator.selection.fd_native.logit_quality_ts
    display_name: "LQTS (global reward)"
    origin: ablation
    params:
      use_global_reward: true
      w_diversity: 0.25
      w_recency: 0.15
    type: fd_native
    trainable: true

  - key: fd_native.lqts_ablation_no_diversity
    module: csfl_simulator.selection.fd_native.logit_quality_ts
    display_name: "LQTS (no diversity)"
    origin: ablation
    params:
      w_diversity: 0.0
      w_recency: 0.15
    type: fd_native
    trainable: true

  - key: fd_native.lqts_ablation_no_recency
    module: csfl_simulator.selection.fd_native.logit_quality_ts
    display_name: "LQTS (no recency)"
    origin: ablation
    params:
      w_diversity: 0.25
      w_recency: 0.0
    type: fd_native
    trainable: true
```

---

## Appendix: Updated SHORT_NAMES for plot_fd_experiments.py

```python
SHORT_NAMES = {
    "baseline.fedavg":              "FedAvg",
    "system_aware.fedcs":           "FedCS",
    "system_aware.tifl":            "TiFL",
    "ml.fedcor":                    "FedCor",
    "ml.maml_select":               "MAML",
    "ml.apex_v2":                   "APEX v2",
    "system_aware.oort":            "Oort",
    "heuristic.label_coverage":     "LabelCov",
    "heuristic.random":             "Random",
    # FD-Native
    "fd_native.snr_diversity":      "SNRD",
    "fd_native.logit_quality_ts":   "LQTS",
    "fd_native.noise_robust_fair":  "NRFS",
    "fd_native.logit_entropy_max":  "LEM",
    # Ablations
    "fd_native.snrd_ablation_fixed_w":     "SNRD-fixW",
    "fd_native.snrd_ablation_no_channel":  "SNRD-noCh",
    "fd_native.snrd_ablation_no_diversity":"SNRD-noDiv",
    "fd_native.snrd_ablation_no_fairness": "SNRD-noFair",
    "fd_native.lqts_ablation_global_reward":"LQTS-global",
    "fd_native.lqts_ablation_no_diversity": "LQTS-noDiv",
    "fd_native.lqts_ablation_no_recency":   "LQTS-noRec",
}
```

Additional `METRIC_LABELS` entries:
```python
METRIC_LABELS.update({
    "logit_entropy_avg":           "Avg. Logit Entropy",
    "logit_cosine_diversity":      "Logit Cosine Diversity",
    "logit_entropy_var":           "Logit Entropy Variance",
    "per_class_accuracy_std":      "Per-Class Acc. Std. Dev.",
    "participation_gini":          "Participation Gini",
    "channel_quality_selected_avg":"Avg. Selected Channel Quality",
    "label_coverage_ratio":        "Label Coverage Ratio",
    "server_client_gap":           "Server-Client Acc. Gap",
    "server_accuracy":             "Server Testing Accuracy",
})
```
