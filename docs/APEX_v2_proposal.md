# APEX v2: Diagnosis and Redesign Proposal

**From empirical failure analysis to principled fixes**

---

## 1. Diagnosis: What's Actually Going Wrong

We ran APEX across 7 experimental settings (3 heterogeneity levels, 3 datasets, IID, 100-client scale) and found **four distinct failure modes**, each traceable to a specific design decision in the algorithm.

### 1.1 Failure Mode 1: Recency-Induced Round-Robin (100-client collapse)

**Symptom**: At N=100, K=10, APEX degrades by 13.89pp (worst of all methods). Oort, which locks onto 10 clients forever, wins by +7.06pp.

**Root cause**: The recency formula `gap / (gap + 5.0)` saturates too quickly relative to the pool size.

At N=100, K=10, the natural revisit interval is ~10 rounds. After just 5 rounds without selection, a client's recency score is already 0.50 (half of maximum). After 10 rounds, it's 0.67. After 20 rounds, it's 0.80. The problem: **90 of 100 clients always have recency scores between 0.67 and 0.91** -- a narrow band that offers almost no differentiation.

Meanwhile, the 10 just-selected clients get recency ~0.17. The 0.50+ gap between recently-selected and not-recently-selected clients is large enough to **override Thompson sampling signals**, which are noisy in early training (only ~5 observations per client over 50 rounds when N=100).

Result: APEX degenerates into near-perfect round-robin. Every 5-round window selects 35-50 unique clients out of 50 total selections. This is the opposite of what smart selection should do.

**The constant 5.0 is the culprit.** It was tuned for N=50, K=10 (revisit interval 5), where `gap / (gap + 5)` provides a smooth gradient. At N=100, K=10, it should be ~10. At N=1000, K=50, it should be ~20. The formula needs to scale with N/K.

### 1.2 Failure Mode 2: Phase Detector Instability (alpha=0.1 oscillations)

**Symptom**: At extreme non-IID (alpha=0.1), APEX reaches peak 47.44% but crashes to 41.10% final. The last 15 rounds show >10pp oscillations every 1-2 rounds (e.g., R32: -12.90pp, R33: +12.45pp, R34: -9.95pp, R35: +12.68pp).

**Root cause**: A destructive feedback loop between the phase detector and Thompson posteriors.

The loop works like this:
1. Phase = "critical" -> diverse cohort selected -> good aggregation -> accuracy jumps, loss drops
2. Loss drop triggers phase -> "transition" or "exploitation" -> Thompson-heavy selection
3. Thompson posteriors (with only ~5 obs/client) are overconfident -> selects concentrated clients
4. At alpha=0.1, concentrated clients have near-single-class data -> model overfits -> accuracy crashes, loss spikes
5. Loss spike triggers phase -> "critical" again -> cycle repeats

The phase detector has **zero hysteresis**. It can flip from "critical" to "exploitation" and back in consecutive rounds. At alpha=0.1, the loss landscape is volatile enough to trigger this oscillation continuously.

At the code level: `_detect_phase()` (line 110-142) computes rate and CV from a 5-round window and makes an instantaneous classification. There is no memory of the previous phase, no minimum dwell time, no smoothing.

### 1.3 Failure Mode 3: Exploration Waste on Easy Tasks (alpha=0.6, Fashion-MNIST)

**Symptom**: At alpha=0.6, FedCS beats APEX by 4.13pp. At Fashion-MNIST, Oort beats APEX by 0.80pp. FedCS leads APEX in **46 of 50 rounds** at alpha=0.6.

**Root cause**: APEX explores when there's nothing to explore.

At alpha=0.6, client data distributions are only mildly skewed. The "right" strategy is pure exploitation -- just pick clients that train fast and have enough data. FedCS does exactly this (deadline-aware selection of fast clients) and wins.

But APEX's phase detector classifies early rounds as "critical" (loss dropping fast with normal fluctuation triggers both `rate > 0.05` AND `cv > 0.10`), setting w_div=0.60. **Sixty percent of the score is diversity** -- in a setting where maximizing diversity means actively selecting outlier clients with unusual distributions.

The diversity proxy becomes counterproductive: at alpha=0.6, most clients have representative data. The proxy's `_min_cosine_distance` pushes selection toward the few clients with atypical histograms -- exactly the clients whose gradients contribute most to client drift.

Even after the phase detector shifts to "transition" (w_div=0.30), the diversity weight is still high enough to distort selection. The 5-round moving average gap to FedCS remains a stable -2 to -4pp throughout training -- this is systematic miscalibration, not a transient exploration cost.

### 1.4 Failure Mode 4: Thompson Posterior Overconfidence (ablation insight)

**Symptom**: In the ablation study, removing Thompson Sampling (no_ts) actually **beats full APEX in 60% of rounds**. This is the component that hurts per-round accuracy most frequently.

**Root cause**: The posterior variance estimate collapses too quickly, and the reward signal is too noisy.

The Thompson posterior update (lines 218-236) uses Welford's online algorithm:
```python
new_var = ((n_i - 1) * old_var + delta * (credit - new_mu)) / max(n_i, 1)
st["ts_sigma2"][cid] = max(new_var, 1e-8)
```

The variance floor is `1e-8` -- essentially zero. After just 3-4 observations, the variance estimate can collapse to near-zero, making the Gaussian Thompson sample `N(mu, sqrt(1e-8/n))` effectively deterministic. At this point, Thompson "sampling" becomes a lookup table of past means. This defeats the entire purpose of Thompson sampling, which requires well-calibrated uncertainty.

The reward signal compounds this: `credit = composite_improvement / K`. The composite improvement is a single-round delta of a noisy metric. It can go negative. It's divided equally among K clients regardless of their individual contributions. This signal-to-noise ratio is too low for fast posterior calibration.

---

## 2. The Deeper Pattern: What These Four Failures Have in Common

All four failures share a single architectural flaw: **APEX has no self-awareness of its own uncertainty.**

- The recency bonus doesn't know how many clients exist
- The phase detector doesn't know if its classification is confident
- The diversity proxy doesn't know if diversity matters in this setting
- The Thompson posteriors don't know they're undersampled

APEX v1 was designed as a **static pipeline**: features go in, scores come out, weights are looked up from a table. The phase detector adds some dynamism, but it's reactive (responds to loss trajectory) rather than introspective (responds to its own prediction quality).

The core insight for v2: **Every component needs a confidence estimate, and the algorithm should modulate its own aggressiveness based on how well it understands the current state.**

---

## 3. Proposed Fixes (Ordered by Impact)

### 3.1 Fix 1: Adaptive Recency Scaling (fixes scalability collapse)

**Problem**: `gap / (gap + 5.0)` has a hardcoded constant that fails at N/K != 5.

**Fix**: Replace `5.0` with `C_rec = max(N/K, 3.0)`:

```python
# Current (line 340):
recency = gap / (gap + 5.0)

# Proposed:
C_rec = max(len(clients) / K, 3.0)
recency = gap / (gap + C_rec)
```

At N=50, K=10: C_rec=5.0 (unchanged). At N=100, K=10: C_rec=10.0. At N=1000, K=50: C_rec=20.0.

**Why this works**: The recency half-life now matches the expected revisit interval. A client that hasn't been selected for exactly one full cycle gets recency=0.50, regardless of pool size. This prevents the recency term from overwhelming Thompson signals at large N.

**Expected impact**: Largest single fix. Should close most of the 7.06pp gap to Oort at 100 clients by preventing forced round-robin.

### 3.2 Fix 2: Phase Detector Hysteresis (fixes alpha=0.1 oscillations)

**Problem**: Phase can flip between "critical" and "exploitation" in consecutive rounds, creating destructive boom-bust cycles.

**Fix**: Add minimum dwell time and smooth transition:

```python
def _detect_phase(loss_history, W, tau_critical, tau_unstable, tau_exploit,
                  prev_phase="critical", phase_age=0, min_dwell=3):
    """Phase detection with hysteresis."""
    if len(loss_history) < W:
        return "critical", 0

    # ... existing rate/cv computation ...

    # Raw classification (same as before)
    if rate > tau_critical and cv > tau_unstable:
        raw_phase = "critical"
    elif rate > tau_exploit:
        raw_phase = "transition"
    else:
        raw_phase = "exploitation"

    # Hysteresis: only transition if we've been in current phase for min_dwell rounds
    # AND the new phase is "adjacent" (no jumping critical <-> exploitation)
    PHASE_ORDER = {"critical": 0, "transition": 1, "exploitation": 2}
    if raw_phase != prev_phase:
        if phase_age < min_dwell:
            return prev_phase, phase_age + 1  # Stay in current phase
        if abs(PHASE_ORDER[raw_phase] - PHASE_ORDER[prev_phase]) > 1:
            return "transition", 0  # Force transition as intermediate
        return raw_phase, 0  # Allow transition
    return prev_phase, phase_age + 1
```

**Key changes**:
1. **Minimum dwell time** (`min_dwell=3`): Must stay in a phase for at least 3 rounds before transitioning. This prevents the 1-round flip-flop that causes oscillations.
2. **No jumping**: Cannot go directly from "critical" to "exploitation" or vice versa. Must pass through "transition". This prevents the immediate shift from diversity-heavy to exploitation-heavy that causes the alpha=0.1 crash.
3. **State tracking**: Phase detector now remembers its previous phase and how long it's been there.

**Expected impact**: Should eliminate the >10pp oscillations at alpha=0.1 and turn the 6.34pp peak-to-final drop into a monotonic (or near-monotonic) convergence.

### 3.3 Fix 3: Heterogeneity-Aware Diversity Scaling (fixes alpha=0.6 overexploration)

**Problem**: Diversity weight is phase-dependent but not heterogeneity-dependent. At alpha=0.6, w_div=0.60 in critical phase is actively harmful.

**Fix**: Compute an online heterogeneity estimate from client label histograms and scale diversity weights accordingly:

```python
def _estimate_heterogeneity(clients):
    """Estimate data heterogeneity from label histogram entropy.
    Returns a value in [0, 1] where 0=IID, 1=extreme non-IID."""
    hists = []
    for c in clients:
        if isinstance(c.label_histogram, dict) and c.label_histogram:
            h = np.array([float(v) for v in c.label_histogram.values()])
            h = h / (h.sum() + 1e-12)
            hists.append(h)
    if len(hists) < 2:
        return 0.5  # Unknown, use moderate assumption

    # Average pairwise Jensen-Shannon divergence
    # High JSD = high heterogeneity = diversity is valuable
    from scipy.spatial.distance import jensenshannon
    n_pairs = min(100, len(hists) * (len(hists) - 1) // 2)
    jsds = []
    for i in range(len(hists)):
        for j in range(i+1, len(hists)):
            jsds.append(jensenshannon(hists[i], hists[j]))
            if len(jsds) >= n_pairs:
                break
        if len(jsds) >= n_pairs:
            break

    avg_jsd = np.mean(jsds)
    # JSD ranges from 0 (identical) to ~0.83 (non-overlapping for 10 classes)
    # Normalize to [0, 1]
    return min(avg_jsd / 0.6, 1.0)
```

Then scale the diversity weights:

```python
het = _estimate_heterogeneity(clients)  # Compute once at round 0, cache

# Scale diversity weight by heterogeneity
w_ts_raw, w_div_raw, w_rec_raw = phase_weights[phase]
w_div_scaled = w_div_raw * het  # At alpha=0.6 (low het), this shrinks diversity
# Redistribute the removed diversity weight to Thompson
w_ts_scaled = w_ts_raw + (w_div_raw - w_div_scaled)
w_rec_scaled = w_rec_raw
```

**Why this works**: At alpha=0.6 (mild heterogeneity), JSD between client distributions is low -> het ~0.3 -> diversity weight drops from 0.60 to 0.18 in critical phase. At alpha=0.1 (extreme), JSD is high -> het ~0.9 -> diversity weight stays at 0.54. The algorithm self-calibrates to the heterogeneity regime.

**Expected impact**: Should close the 4.13pp gap to FedCS at alpha=0.6 and the 0.80pp gap to Oort on Fashion-MNIST.

### 3.4 Fix 4: Thompson Posterior Regularization (fixes overconfidence)

**Problem**: Variance floor of 1e-8 allows posteriors to become deterministic after 3-4 observations. Reward signal `composite_improvement/K` is too noisy.

**Fix A**: Add a minimum variance that decays slowly:

```python
# Current (line 230):
st["ts_sigma2"][cid] = max(new_var, 1e-8)

# Proposed:
# Minimum variance decays as 1/sqrt(n) -- slower than the Welford estimate (1/n)
# This keeps posteriors uncertain longer, enabling meaningful exploration
sigma2_floor = 0.1 / math.sqrt(max(n_i, 1))
st["ts_sigma2"][cid] = max(new_var, sigma2_floor)
```

At n_i=2: floor=0.071 (substantial uncertainty). At n_i=10: floor=0.032. At n_i=50: floor=0.014. This ensures Thompson sampling produces meaningfully different samples even after many observations, maintaining exploration where needed.

**Fix B**: Smooth the reward signal with EMA:

```python
# Current: credit = last_reward / K (noisy single-round delta)

# Proposed: maintain an EMA of rewards per client
ema_alpha = 0.3  # blend factor for new observation
old_ema = float(st["ts_reward_ema"].get(cid, credit))
smoothed_credit = ema_alpha * credit + (1 - ema_alpha) * old_ema
st["ts_reward_ema"][cid] = smoothed_credit

# Use smoothed_credit for posterior update instead of raw credit
```

This filters out the single-round noise in the reward signal, giving the posteriors more stable training data.

**Expected impact**: Should reduce the frequency of rounds where no_ts outperforms full APEX from 60% to <40%. More importantly, should prevent the exploitation trap at alpha=0.1 where overconfident posteriors drive concentrated selection.

### 3.5 Fix 5: Confidence-Aware Score Blending (the architectural fix)

**Problem**: The blend weight gamma=0.3 between contextual and Thompson is fixed regardless of how informative the posterior actually is.

**Fix**: Make gamma adaptive per client based on posterior confidence:

```python
# Current (line 332):
blended = (1.0 - gamma) * ctx + gamma * ts_sample

# Proposed:
# Clients with more observations and tighter posteriors get higher gamma
# (trust the posterior more). New clients get lower gamma (trust context more).
n_i = st["ts_n"].get(cid, 0)
var_i = st["ts_sigma2"].get(cid, 1.0)
confidence = 1.0 - 1.0 / (1.0 + n_i)  # 0 at n=0, 0.5 at n=1, 0.9 at n=9
effective_gamma = gamma * confidence

blended = (1.0 - effective_gamma) * ctx + effective_gamma * ts_sample
```

**Why this works**: New clients (n=0, confidence=0) are scored purely by contextual features (loss, grad norm, speed, data size) -- the most reliable signals when we know nothing about a client. Well-observed clients (n=10, confidence=0.91) get strong Thompson influence, leveraging the posterior. This prevents the cold-start randomness that hurts early convergence.

**Expected impact**: Should improve cold-start accuracy (currently the lowest of all methods at round 0) and reduce the early exploration cost that causes APEX to lag in rounds 0-10.

---

## 4. Interaction Analysis: Why These Fixes Work Together

The five fixes are not independent. They interact synergistically:

```
Fix 1 (Recency scaling)    ──> Prevents round-robin at large N
                                 └─> Thompson gets enough repeat observations
                                      └─> Fix 4 (posterior reg.) works better
                                           └─> Fix 5 (adaptive gamma) has
                                                calibrated posteriors to use

Fix 2 (Phase hysteresis)   ──> Prevents phase oscillation
                                 └─> Stable phase = stable diversity weight
                                      └─> Fix 3 (het-aware diversity) not
                                           overridden by erratic phase changes

Fix 3 (Het-aware diversity)──> Right diversity level for the setting
                                 └─> Phase detector sees smoother loss
                                      └─> Fix 2 triggers less often
                                           (fewer erratic transitions to dampen)
```

The key insight: **Fixes 1-3 stabilize the algorithm's behavior**, creating the stable conditions under which **Fixes 4-5 can improve Thompson sampling quality**. Without the stabilization, better posteriors just mean the algorithm exploits more confidently in the wrong direction.

---

## 5. What Changes, What Stays

### Stays the same:
- **Three-component architecture** (Thompson + diversity + recency) -- the components are right, the calibration is wrong
- **Phase detection concept** -- detecting training phases is valuable, the detector just needs hysteresis
- **Label-histogram diversity proxy** -- cheap and effective, just needs to know when to activate
- **Greedy selection with diversity** -- the O(K^2*L) greedy loop is sound
- **Zero trainable parameters** -- all fixes use closed-form computations
- **O(N*L + K^2*L) complexity** -- heterogeneity estimation is O(N^2) but can be cached (computed once)

### Changes:
- **Recency constant**: 5.0 -> N/K (one character change, biggest impact)
- **Phase detector**: stateless -> stateful with hysteresis (adds 2 state variables)
- **Diversity weight**: phase-only -> phase * heterogeneity (adds one scaling factor)
- **Thompson variance**: floor 1e-8 -> 0.1/sqrt(n) (one line change)
- **Thompson reward**: raw -> EMA-smoothed (adds one state variable per client)
- **Gamma**: fixed 0.3 -> confidence-scaled 0.3 * f(n_i) (one line change)

### New state per client:
- `reward_ema`: 1 float (Fix 4B)

### New global state:
- `prev_phase`: 1 string (Fix 2)
- `phase_age`: 1 int (Fix 2)
- `heterogeneity`: 1 float, cached (Fix 3)

Total new state: 1 float/client + 3 global values. Negligible.

---

## 6. Expected Performance After Fixes

Based on the root cause analysis, here are conservative predictions:

| Setting | APEX v1 | Expected v2 | Reasoning |
|---------|---------|-------------|-----------|
| CIFAR-10, a=0.3, N=50 | 45.37% | ~46-48% | Fix 4+5 improve Thompson quality; fix 2 reduces late dips |
| CIFAR-10, a=0.1, N=50 | 41.10% (final) / 47.44% (peak) | ~45-47% final | Fix 2 eliminates oscillations; final should approach peak |
| CIFAR-10, a=0.6, N=50 | 50.59% | ~53-54% | Fix 3 reduces diversity to near-zero; approaches FedCS performance |
| CIFAR-10, a=0.3, N=100 | 36.16% | ~40-43% | Fix 1 eliminates round-robin; should approach Oort level |
| MNIST, a=0.3 | 96.88% | ~97% | Already near ceiling; minor improvement |
| Fashion-MNIST, a=0.3 | 78.38% | ~79-80% | Fix 3 reduces unnecessary diversity |
| CIFAR-10, IID | 47.73% | ~48% | Fix 3 sets diversity to ~0; approaches FedAvg |

### The two biggest wins:
1. **100-client setting**: +4 to +7pp from Fix 1 alone
2. **Alpha=0.1 stability**: Eliminating oscillations should raise final accuracy by 4-6pp (closer to peak)

---

## 7. Implementation Priority

| Priority | Fix | LOC Changed | Risk | Expected Gain |
|----------|-----|-------------|------|---------------|
| **P0** | Fix 1: Recency scaling | ~3 lines | Very low | +4-7pp at N=100 |
| **P0** | Fix 2: Phase hysteresis | ~20 lines | Low | +4-6pp at alpha=0.1 |
| **P1** | Fix 3: Het-aware diversity | ~30 lines | Medium | +3-4pp at alpha=0.6 |
| **P1** | Fix 4: Posterior regularization | ~5 lines | Low | Reduces oscillation everywhere |
| **P2** | Fix 5: Adaptive gamma | ~5 lines | Low | Improves cold start, minor gains |

**Recommended approach**: Implement Fix 1 and Fix 2 first. These are low-risk, high-impact, and can be validated immediately with the existing experiment matrix. Then add Fix 3+4 together. Fix 5 is the cherry on top.

---

## 8. New Experiments to Run

After implementing fixes:

```powershell
# Validate Fix 1: Scalability
python -m csfl_simulator compare --name apexv2_scale100 \
    --methods "baseline.fedavg,system_aware.oort,ml.apex,ml.apex_v2" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN --total-clients 100 --clients-per-round 10 \
    --rounds 50 --no-fast-mode --track-grad-norm --device cuda --seed 42

# Validate Fix 2: Stability under extreme non-IID
python -m csfl_simulator compare --name apexv2_extreme \
    --methods "baseline.fedavg,system_aware.oort,ml.apex,ml.apex_v2" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 \
    --model LightCNN --total-clients 50 --clients-per-round 10 \
    --rounds 50 --no-fast-mode --track-grad-norm --device cuda --seed 42

# Validate Fix 3: Mild heterogeneity
python -m csfl_simulator compare --name apexv2_mild \
    --methods "baseline.fedavg,system_aware.fedcs,ml.apex,ml.apex_v2" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 \
    --model LightCNN --total-clients 50 --clients-per-round 10 \
    --rounds 50 --no-fast-mode --track-grad-norm --device cuda --seed 42

# Full benchmark: v1 vs v2 head-to-head
python -m csfl_simulator compare --name apexv2_benchmark \
    --methods "baseline.fedavg,heuristic.delta,system_aware.fedcs,system_aware.oort,ml.ucb_grad,ml.apex,ml.apex_v2" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN --total-clients 50 --clients-per-round 10 \
    --rounds 50 --no-fast-mode --track-grad-norm --device cuda --seed 42

# New ablation: which v2 fix contributes most
python -m csfl_simulator compare --name apexv2_ablation \
    --methods "ml.apex_v2,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN --total-clients 50 --clients-per-round 10 \
    --rounds 50 --no-fast-mode --track-grad-norm --device cuda --seed 42
```

---

## 9. What This Means for the Paper

### If v2 fixes work as expected:

The paper narrative shifts from "APEX is best at moderate non-IID" to **"APEX is robust across heterogeneity regimes and scales."** This is a much stronger claim.

The ablation section becomes richer: instead of just showing component importance, we show **how self-aware calibration transforms each component from sometimes-harmful to consistently-helpful.**

The limitations section shrinks significantly. The current three honest weaknesses (scalability, alpha=0.6, oscillations) become the motivation for the v2 design, and the paper presents the final algorithm as the result of principled empirical refinement.

### If some fixes don't work:

That's also valuable data. If, for example, Fix 3 (heterogeneity-aware diversity) doesn't close the gap at alpha=0.6, that tells us the diversity proxy itself is the problem at low heterogeneity, not just the weight. This would motivate a different kind of fix (e.g., a "diversity relevance" gating function that can fully disable the proxy).

Either way, the diagnostic framework (4 failure modes, root causes, targeted fixes) is a contribution in itself -- it demonstrates the kind of systematic analysis that advances the field beyond "throw a new architecture at the benchmark."

---

## 10. Summary

| Problem | Root Cause | Fix | Lines Changed |
|---------|-----------|-----|---------------|
| 100-client collapse | Recency constant hardcoded to 5.0 | Scale with N/K | 3 |
| Alpha=0.1 oscillations | Phase detector has zero hysteresis | Add min dwell + no-skip rule | 20 |
| Alpha=0.6 overexploration | Diversity weight ignores heterogeneity level | Scale by JSD estimate | 30 |
| Thompson overconfidence | Variance floor 1e-8, noisy rewards | Floor 0.1/sqrt(n) + EMA rewards | 10 |
| Cold start weakness | Fixed gamma ignores observation count | Confidence-scaled gamma | 5 |

Total: ~68 lines of changes for an algorithm that should be robust across all tested regimes. Zero new trainable parameters. Same asymptotic complexity. The architecture was right; the calibration was wrong.
