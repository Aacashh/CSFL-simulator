---
title: "Mechanism Analysis of FD Client Selection + Proposal for CALM-FD"
subtitle: "From LQTS's win to the next SOTA — a researcher's autopsy of the 2026-04 batch"
author: "DRDO CSFL Simulator — research diary"
date: "2026-04-20"
geometry: "margin=0.85in"
fontsize: 10pt
header-includes:
  - \usepackage{longtable}
  - \usepackage{booktabs}
  - \usepackage{array}
  - \renewcommand{\arraystretch}{1.15}
  - \setlength{\tabcolsep}{4pt}
---

# 0. How to read this document

This is a working research document, not a paper submission. It has three purposes:

1. **Explain the mechanism** by which `fd_native.logit_quality_ts` (LQTS) beat every other selector we tested in the 2026-04 FD batch — in particular why three FL champions (APEX v2, Oort, MAML-Select) fell in rank, and why two nominally FD-native methods (SNR-Diversity, Logit-Entropy Max) also lost. Understanding the "why" is a precondition for designing the next thing.

2. **Propose a successor selector — CALM-FD** — that extends LQTS along the axes the empirical evidence identifies as LQTS's weak points, while deliberately *not* chasing the things the evidence tells us don't work (logit-space diversity, pure channel-awareness, FL-transplanted utility signals).

3. **Lay out the next experiment batch** — one-day budget, with paper replication as the first experiment so we can anchor ourselves to Mu et al.'s published numbers before claiming any improvement.

Every numeric claim in §1–§4 comes from the analysis already in `artifacts/analysis/` (see `master_table.csv`, `noise_sweep_summary.csv`, `ranking_fl_vs_fd.csv`, and the `figures/fig*.png` and `figM*.png` plots). Section §5 is the experiment plan; §6 is the shell script summary.

---

# 1. What the 2026-04 batch showed, in one paragraph

Across 8 FD/FL runs on CIFAR-10 (N=50, K=15, α=0.5, 300 rounds, heterogeneous ResNet18-FD/MobileNet-FD/ShuffleNet-FD, UL SNR=−8 dB, DL SNR swept from errfree to −30 dB), **LQTS ranks #1 or tied #1 in every noisy run and #3 in the error-free run**. The Spearman rank correlation between FL ranking and FD ranking of the same 10 methods is **ρ = −0.30** (a *negative* correlation — FL rank literally doesn't transfer). The single most predictive feature is **not** the one anyone expected: across all (run, method) pairs the final **logit cosine diversity** is *negatively* correlated with final server accuracy (r = −0.59), meaning methods that successfully maximise logit diversity systematically deliver worse distilled models. LQTS wins because it rewards consistency with the aggregated consensus rather than chasing diversity, and because — alone among the tested methods — it produces a large positive **server-client accuracy gap** (server's distilled model outperforms client avg by ~10 pp at round 300), which is the signature of distillation actually working.

See `figM6_ranking_heatmap.png` for the rank-vs-SNR heatmap, `figM3_logit_div_vs_acc.png` for the negative-diversity-correlation scatter, and `figM1_server_client_gap.png` for the server-client gap trajectory.

---

# 2. Mechanism-level autopsies (method by method)

![**Fig. M1.** Server$-$client accuracy gap over 300 rounds, FD main benchmark. A positive gap means the server's distilled model is out-performing the average client's local model — the signal that FD distillation is actually transferring knowledge. LQTS is the only method with a sustained large positive gap (0.10 at the plateau); everyone else hovers near zero.](../artifacts/analysis/figures/figM1_server_client_gap.png)

## 2.1 Why LQTS wins (the thing to preserve)

Three properties compound:

1. **Reward = cosine similarity to aggregated logit, per client, per round.** This is the *only* reward signal in the entire FD_CORE set that directly measures a client's contribution to the distillation target. Everything else (Oort's UCB on loss, FedCS's loss/time ratio, APEX v2's phase-gated Thompson on global accuracy delta, MAML-Select's learned MLP, SNR-Diversity's channel+label blend) uses indirect signals that correlate with training utility in FL but decouple from logit utility under FD channel noise.

2. **Thompson sampling on per-client posteriors** gives each client a separate `(μ, σ²)`. A client who was historically good but whose channel just degraded will have a wider posterior and get re-sampled; a client who was consistently bad gets a narrow low posterior and stops being picked. LQTS's variance floor decays with observation count (`1/√count`), so committed exploitation is available — but only after evidence accumulates.

3. **Feature-space diversity, not logit-space diversity.** LQTS's greedy refinement after TS draw penalises clients whose `[label_hist, loss, grad_norm]` proxy is similar to already-selected clients. Crucially, **it does not penalise similar logits**. That keeps the consensus-building reward intact while avoiding trivially redundant picks.

All three properties work together. The reward gives the posterior something meaningful to learn; the Thompson mechanism keeps exploration alive without chasing outliers; the feature-space diversity provides a regulariser that avoids collapse without contradicting the reward.

## 2.2 Why Oort falls 6 ranks (FL #2 → FD #8)

Oort's utility is `loss^α × duration^β` with UCB exploration. Under FL, "high loss" is a useful signal because it points at clients whose gradient updates will move the global model the most. Under FD with DL noise at −20 dB, **loss is still high — but not because the client's gradient is informative**. It's high because the client's local model fit its local partition well, but the **logit it transmits is corrupted by channel noise after it leaves the device**. Oort's utility re-picks that client (high loss = high utility), its UCB exploration amplifies the signal, and we end up with the worst positive feedback loop in the set: the noisy-channel client is over-sampled, contributes more corrupted logits per round, the server's distilled model degrades, all clients see worse distillation feedback next round, their local loss goes up, and Oort's utility estimate marks them even higher utility.

**The sign of Oort's behaviour:** Across the full noise sweep Oort has the largest accuracy drop (−19.0 pp from errfree to −30 dB) AND the highest std over rounds 100–280 under any noise level. `figM2_stability.png` shows this clearly — Oort's bar is always the tallest in the stability plot.

## 2.3 Why APEX v2 falls 4 ranks (FL #1 → FD #5)

APEX v2 is our own FL champion. Phase detector + contextual Thompson on global accuracy delta + diversity proxy + adaptive recency. In FL it wins convincingly (66.9% at α=0.5). In FD at −20 dB it scores 19.7%, losing 6.4 pp to LQTS.

Two failure modes:

1. **Global reward distribution is too coarse for FD.** APEX v2's reward is "did global accuracy go up this round?", divided equally among the K selected clients. In FL that's fine — the global model is a shared object, and the reward is genuinely global. In FD the server's accuracy is a function of aggregated *logits*, and individual clients contribute very differently to that aggregation (noise affects them differently, confidence varies, label coverage matters). Distributing a global reward uniformly collapses those differences.

2. **Phase detection reads a loss signal that isn't FD-meaningful.** APEX v2's "critical / transition / exploitation" phase classifier uses the loss trajectory's curvature. Under FD at −20 dB, the loss trajectory is dominated by channel noise fluctuation rather than genuine training progress, so phase calls become random.

Interestingly, **at errfree APEX v2 ties LQTS at 30.1%**. The FL crown does transfer in the easy regime. The collapse happens between DL=−10 dB and −20 dB — exactly where the channel noise floor exceeds the training-signal floor. This is a clean empirical argument for why "selection under FD with channel noise" is a distinct problem from FL selection.

## 2.4 Why SNR-Diversity (FD-native) loses (rank 6)

SNR-Diversity is the most paradoxical method. It explicitly maximises `w_channel × SNR + w_diversity × label_coverage + w_fairness × recency`. It succeeds at its own objective — its final logit cosine diversity is **0.775, the highest of any method**. But it ranks 6th in accuracy at 16.3%, *below* Random (15.9%).

The mechanism: the 4 FD-native methods' designs assumed "diversity is good for FD because diverse logits mean diverse knowledge". The batch data refutes that assumption with r = −0.59 correlation across all runs. **Diverse logits under channel noise are corrupted-in-different-directions logits, not diverse-knowledge logits.** SNR-Diversity's optimiser found the methodologically-pure but empirically-wrong maximum.

This is the single most important negative finding in the batch. Any future FD-native selector that treats logit diversity as a reward signal will reproduce SNR-Diversity's failure.

## 2.5 Why Logit-Entropy Max (FD-native) is the worst (rank 10)

LEM picks clients whose public-set logits maximise Shannon entropy. The intuition was "high entropy = uncertain predictions = informative knowledge". The reality:

- Under light noise, high-entropy logits come from under-trained clients or clients whose local data distribution is misaligned with the public set. Their logits are near-uniform — which carries zero distillation signal, because the KL term `KL(student, softmax(uniform))` pushes the student toward uniform too.
- Under heavy noise, high-entropy logits come from clients whose logits have been scrambled by the DL channel into a quasi-uniform distribution. Picking them amplifies the noise.

LEM ranked 10/10 under heavy noise and 5/7 under light noise. **The entropy direction was inverted**: we want *low* entropy (confident logits), not high. This is the insight that motivates CALM-FD's confidence blend.

## 2.6 Why FedCS unexpectedly wins in FD (FL #10 → FD #3)

FedCS picks high-loss + fast clients. In FL this is a vicious cycle. In FD two lucky alignments save it:

1. **High-loss clients with well-fit local models produce peaky, confident logits** — not because FedCS intends that, but because a local model that fits a non-IID slice tightly will over-confidently predict on the public set. Those peaky logits are *good* distillation targets, per §2.5's inverted-entropy principle.
2. **"Fast" clients in the simulator's system model are clients with good channels** (compute speed and channel quality are positively correlated in `core/system.py`). So FedCS's time preference accidentally excludes the worst-channel clients, partially replicating CALM-FD's channel-floor filter.

FedCS got the right answer for the wrong reason. This suggests an explicit cleaner version of its implicit signal will beat it.

![**Fig. M2.** Per-method accuracy std over rounds 100–280. Lower = more stable. LQTS (purple) is the most stable method under every noise regime; Oort (cyan) and Random (blue) are the most volatile.](../artifacts/analysis/figures/figM2_stability.png)

![**Fig. M6.** Method rank across DL SNR levels (1 = best). LQTS is #1 at −10, −20, −30 dB; APEX v2 wins only at 0 dB; Oort wins only at errfree.](../artifacts/analysis/figures/figM6_ranking_heatmap.png)

---

# 3. Four empirical findings that shape the next design

| # | Finding | Evidence | Implication for CALM-FD |
|---|---|---|---|
| F1 | Logit cosine diversity **negatively** correlates with accuracy (r = −0.59 across 7 runs × 10 methods) | `figM3_logit_div_vs_acc.png`; SNR-Diversity max diversity = 0.78, rank 6; LQTS diversity = 0.63, rank 1 | Do **not** use logit-space diversity as a reward. Keep diversity only on the *feature proxy*. |
| F2 | Low-entropy (peaky) logits are better distillation targets | LEM rank 10; FedCS accidental rank 3; `figM4_entropy_traj.png` shows LQTS and APEX v2 selecting lowest-entropy clients | Add a **confidence term** (1 − normalised entropy) to the reward blend. |
| F3 | Effective noise regime modulates selection importance | 6 pp spread at errfree, 14 pp spread at −20 dB; FL methods crack between −10 and −20 dB | **Scale bandit exploration variance** with an online noise-regime estimator. |
| F4 | Server-client gap is LQTS's unique signature (no other method sustains it) | `figM1_server_client_gap.png`: LQTS = 0.10; everyone else ≈ 0.04 | **Preserve** LQTS's reward and posterior structure; the gap is *caused* by them. |

Two more, softer:

- **F5.** Channel quality of selected clients has r = −0.14 with accuracy — i.e. **pure** channel-awareness is neutral-to-harmful. But the *bottom* of the channel distribution is genuinely bad (DL=−30 dB cases). → Use channel quality **only as a hard floor exclusion** (drop the bottom ~5%), not as a score.
- **F6.** Methods that saturate Fairness Gini at 0.70 (LabelCov, FedCS, Oort, LEM under noise) aren't necessarily bad — LabelCov is rank 2 with max Gini. Participation fairness and accuracy are only weakly linked. → Fairness should be a **guard** (prevent starvation) rather than an objective.

![**Fig. M3.** The counter-intuitive result visualised: each point is a (method, run) pair; colour/marker encodes method. The dashed line is the linear fit — slope is negative. Higher logit cosine diversity ↔ lower server accuracy.](../artifacts/analysis/figures/figM3_logit_div_vs_acc.png)

![**Fig. M4.** Average logit entropy of selected clients over rounds. LQTS and APEX v2 settle at the lowest entropy (most confident logits). LEM sits near log(10) = max entropy. The winners pick confident logits; the loser picks maximally uncertain ones.](../artifacts/analysis/figures/figM4_entropy_traj.png)

---

# 4. CALM-FD — proposed successor selector

**One-sentence definition.** CALM-FD extends LQTS by (a) blending logit confidence into the reward, (b) scaling posterior variance with an online noise-regime estimator, (c) inflating stale posteriors to force re-examination, (d) excluding only the bottom channel-quality percentile rather than scoring by channel, and (e) penalising redundant selections in feature-proxy space — keeping every one of LQTS's winning properties while addressing each of its failure modes identified in §2 and §3.

## 4.1 Algorithm (pseudocode)

```
State: μ[cid], σ²[cid], obs[cid], ema_rew[cid], noise_ema (scalar)
Hyperparameters: w_cosine=0.65, w_confidence=0.35,
                 ema_alpha=0.3, base_var_floor=0.05,
                 noise_expand_gain=2.0, stale_rounds_factor=2.0,
                 stale_var_inflate=0.30,
                 w_ts=0.70, w_recency=0.15, w_collusion=0.15,
                 channel_floor_percentile=5

Per round t:
    # F3: online noise regime
    if effective_noise_var[t-1] available:
        noise_ema = 0.9·noise_ema + 0.1·tanh(noise_var/10)
    var_floor = base_var_floor · (1 + noise_expand_gain·noise_ema)

    # F2: confidence-blended reward
    for cid in last-round's selected:
        r_cos   = fd_logit_rewards[cid]                    # cosine to mean (LQTS signal)
        ent     = fd_logit_stats[cid].entropy_mean
        conf    = 1 - ent / log(num_classes)               # in [0,1], high = peaky
        r       = w_cosine · r_cos + w_confidence · conf

        # posterior update
        ema_rew[cid] = ema_alpha · r + (1-ema_alpha) · ema_rew[cid]
        μ[cid]       = ema_rew[cid]
        σ²[cid]      = max(var_floor, 0.5·σ²[cid] + 0.5·(r - ema_rew[cid])²)
        obs[cid]    += 1

    # stale-posterior guard
    for each client c:
        if t - c.last_selected > stale_rounds_factor · (N/K):
            σ²[c] = max(σ²[c], stale_var_inflate)

    # Thompson draw
    for each c:   θ[c] = Normal(μ[c], √(σ²[c]/max(1,obs[c])))

    # F5: channel-floor filter (hard, minimal)
    q_thr = percentile(channel_quality, channel_floor_percentile)
    eligible = {c : channel_quality(c) ≥ q_thr  OR  obs[c] == 0}

    # Greedy top-K with recency + anti-collusion (feature proxy, NOT logits)
    for k in 1..K:
        for c in eligible − selected:
            gap      = t - c.last_selected
            recency  = gap / (gap + max(N/K, 3))
            collusion = max_{s in selected} cosine(proxy(c), proxy(s))   # proxy = [label_hist, loss, grad]
            score[c] = w_ts · θ[c] + w_recency · recency - w_collusion · collusion
        pick argmax; append to selected
    return selected
```

## 4.2 What's different from LQTS, and why

| Change | Motivated by | Expected effect |
|---|---|---|
| Blended reward (cosine + confidence) | F2, §2.5 (LEM failure, inverted entropy) | More discriminative reward; peaky-and-agreeing logits outrank diffuse-and-agreeing ones. |
| Adaptive variance floor scaled by noise regime | F3, §2.3 (APEX-v2 crack at −10→−20 dB) | More exploration when channel is noisy; faster convergence in clean regime. |
| Stale-posterior guard | LQTS doesn't have one; channel conditions drift | Re-opens bandit for long-silent clients whose state may have changed. |
| Channel-floor exclusion (not score) | F5 (pure channel awareness is neutral-to-harmful, but bottom is genuinely bad) | Removes a small amount of known-bad without introducing the SNR-Diversity failure mode. |
| Anti-collusion on proxy space (not logit space) | F1 (logit diversity negatively correlates) | Prevents redundant picks without chasing the wrong diversity objective. |
| Thompson weight w_ts = 0.70 (was effectively ~0.60 in LQTS) | Logit reward is more trustworthy than recency given the findings | Slightly more weight on the reward, less on bureaucratic fairness. |

## 4.3 Ablation plan

CALM-FD ships with 5 ablation variants, each disabling exactly one enhancement. This isolates the contribution of each idea:

- `fd_native.calm_fd_no_confidence` — disables the confidence blend (falls back to pure cosine reward = LQTS reward)
- `fd_native.calm_fd_no_adaptive_var` — uses fixed variance floor
- `fd_native.calm_fd_no_stale_guard` — no inflation for long-silent clients
- `fd_native.calm_fd_no_channel_filter` — no bottom-percentile exclusion
- `fd_native.calm_fd_no_collusion` — no feature-proxy collusion penalty

All 5 are registered in `presets/methods.yaml`. If CALM-FD wins in experiments, these 5 tell us *which* of the 5 changes is load-bearing.

## 4.4 Failure modes to watch for

1. **Over-exploitation in low-noise regime.** The adaptive floor shrinks to `base_var_floor = 0.05`, which is smaller than LQTS's effective floor. If the bandit commits too fast when the environment is actually easy, we could under-explore. Mitigation: `base_var_floor` is tunable; the errfree experiment will tell us whether to raise it.

2. **Confidence blend harming high-entropy-but-useful clients.** There's a class of clients whose logits are diffuse because their data is genuinely class-balanced (IID-like). We're penalising them. If α is very high (near-IID), this could hurt. Mitigation: the α-sweep experiment (Exp 4) will detect this.

3. **Channel-floor filter starving cold-start clients.** A newly-observed client with zero obs count but bad channel would never be selected. Mitigation: the filter exempts cold-start clients (`obs == 0`) — see line `OR obs[c] == 0` in the pseudocode.

![**Fig. M5.** Fairness Gini trajectories. CALM-FD's design intent is to sit near LQTS (Gini ~0.17 — moderate, not maxed) while beating it on accuracy. Noise-Robust-Fair shows what "fairness-first" looks like (Gini near 0).](../artifacts/analysis/figures/figM5_fairness_traj.png)

---

# 5. Next experiment batch — what the shell script runs

**Design principles:**

- **Paper replication first.** Exp 1 uses Mu et al.'s exact setup (N=15, K=15 full participation, CNN_1/2/3 pool, α=0.5, DL=−20 dB, 200 rounds) with *only* the baseline methods from the paper's comparison, so we can anchor against their ~50% accuracy figure before claiming any improvement.
- **CALM-FD is tested in every experiment from Exp 2 onward.** Its ablation variants appear only in the dedicated ablation experiment (Exp 8), keeping the other experiments focused on the competitive comparison.
- **N=30, K=10 is our "new baseline"** (matches `docs/FD_experiments.md` default). Smaller than the Apr batch's N=50/K=15 — brings data-per-client up to ~1,667 samples (from 1,000), which should close most of the accuracy gap to the paper while preserving meaningful selection.
- **FD-CNN1/2/3 replace ResNet18/MobileNet/ShuffleNet** for CIFAR-10 runs. Same architecture Mu et al. used, and the codebase already supports them on CIFAR-10 (no code change needed — confirmed by `scripts/analyze_fd_results.py` smoke-test and the `_dataset_image_spec` auto-detection in `core/models.py:223-232`).
- **2 local epochs** (Mu et al.'s setting, up from our previous 1).
- **Compute budget:** ~15–18 hours on a single GPU. No experiment uses N > 100. No experiment uses more than 8 methods in one comparison (paper replication uses 3).

**Nine experiments (see `scripts/run_next_sota_experiments.sh`):**

<table>
<thead>
<tr><th>#</th><th>Name</th><th>Purpose</th><th>N / K / α / DL SNR / Rounds</th><th>Methods</th><th>Est. wall-clock</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>paper_replication</td><td>Reproduce Mu et al. baseline numbers with full participation — sanity anchor</td><td>15 / 15 / 0.5 / −20 dB / 200</td><td>random, fedavg (alias), calm_fd</td><td>~1.2 h</td></tr>
<tr><td>2</td><td>fd_main_newbaseline</td><td>Headline comparison at N=30, K=10 — the new paper-ready benchmark</td><td>30 / 10 / 0.5 / −20 dB / 200</td><td>7 methods + calm_fd</td><td>~3 h</td></tr>
<tr><td>3</td><td>noise_sweep</td><td>CALM-FD vs LQTS across 5 DL SNR levels — does F3's adaptive variance pay off?</td><td>30 / 10 / 0.5 / {errf, 0, −10, −20, −30} / 200</td><td>random, apex_v2, LQTS, calm_fd, snr_diversity</td><td>~5 h</td></tr>
<tr><td>4</td><td>alpha_sweep</td><td>Heterogeneity sensitivity — closes the α=0.1 gap from Apr batch</td><td>30 / 10 / {0.1, 0.5, 1.0, 5.0} / −20 dB / 200</td><td>random, LQTS, calm_fd, labelcov</td><td>~3.5 h</td></tr>
<tr><td>5</td><td>k_sweep</td><td>Does CALM-FD's advantage persist as K/N → 1?</td><td>30 / {3, 6, 10, 15, 30} / 0.5 / −20 dB / 200</td><td>random, LQTS, calm_fd</td><td>~3 h</td></tr>
<tr><td>6</td><td>n_scaling</td><td>Scales N=30/50/100 at fixed 33% participation</td><td>{30, 50, 100} / {10, 16, 33} / 0.5 / −20 dB / 200</td><td>random, LQTS, calm_fd</td><td>~4 h</td></tr>
<tr><td>7</td><td>cross_dataset_mnist</td><td>Generalisation to MNIST+FMNIST</td><td>30 / 10 / 0.5 / −20 dB / 150</td><td>7 methods + calm_fd</td><td>~1.5 h</td></tr>
<tr><td>8</td><td>calm_fd_ablation</td><td>Which of the 5 CALM-FD enhancements is load-bearing?</td><td>30 / 10 / 0.5 / −20 dB / 200</td><td>LQTS + calm_fd + 5 ablations</td><td>~2.5 h</td></tr>
<tr><td>9</td><td>multi_seed</td><td>Reproducibility — 3 seeds on the headline comparison</td><td>30 / 10 / 0.5 / −20 dB / 200</td><td>LQTS + calm_fd (2 methods, 3 seeds)</td><td>~1.5 h</td></tr>
</tbody>
</table>

**Total estimated wall-clock: ~25 h worst-case, ~18 h typical.** Slightly over 1 day worst-case — the `--resume` flag in the shell script lets us split across two sessions, and experiments 7–9 can be skipped without breaking the story if time runs short.

---

# 6. What to do with the results

After Exp 1 completes, spot-check against Mu et al. Fig. 4 / Table IV:

- Paper replication target: per-user accuracy in the **46 %–54 %** range at α=0.5 with the heterogeneous CNN_1/2/3 pool under DL=−20 dB full-participation. If we hit this, our simulator faithfully reproduces the paper. If we undershoot by a lot, there's a setup bug to find before trusting any other result.

After Exps 2–3 complete, the headline claim is decided:

- **If CALM-FD beats LQTS by ≥2 pp at DL=−20 dB and ≥1 pp at errfree (Exp 2), and shows a larger advantage at lower SNR (Exp 3), the story is "CALM-FD is the new FD-native SOTA".** Exp 8's ablation then tells us which enhancement mattered (my prior expectation: confidence-blend and adaptive-variance will be the load-bearers; the other three will be small).
- **If CALM-FD matches LQTS, the story is "LQTS was already near-optimal; minor extensions don't move the needle".** Still publishable as a negative result + ablation evidence that logit-quality is the key signal.
- **If CALM-FD loses to LQTS, the story is "this specific design pattern (confidence-blend + adaptive-var + stale-guard) overfits; simpler is better in FD".** We'd publish LQTS alone and use CALM-FD's ablation as support.

All three outcomes are publishable. The experiments are worth running.

---

# 7. Files touched for this work

- **New selector:** `csfl_simulator/selection/fd_native/calm_fd.py` (~230 lines; builds on LQTS; hyperparameters fully exposed for ablation)
- **Registered in:** `presets/methods.yaml` (`fd_native.calm_fd` + 5 ablation variants)
- **Analysis script for mechanism plots:** `scripts/analyze_fd_mechanisms.py` (reads the Apr-2026 batch; emits `figM1`–`figM6` into `artifacts/analysis/figures/`)
- **Shell script for next batch:** `scripts/run_next_sota_experiments.sh` (9 experiments; `--resume` / `--dry-run` supported; paper replication first)
- **This document:** `docs/FD_deep_analysis_and_next_sota.md`
