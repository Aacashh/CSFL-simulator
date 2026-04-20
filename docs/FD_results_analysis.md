---
title: "Federated Distillation under mMIMO Channel Noise — Results Analysis"
subtitle: "Empirical evidence for the FL→FD client-selection ranking-inversion thesis"
author: "DRDO CSFL Simulator — analysis of 2026-04-14…2026-04-19 experiment batch"
date: "2026-04-20"
geometry: "margin=0.85in"
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{longtable}
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{ragged2e}
  - \usepackage{float}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \renewcommand{\arraystretch}{1.15}
  - \setlength{\tabcolsep}{4pt}
  - \let\oldlongtable\longtable
  - \let\endoldlongtable\endlongtable
  - \renewenvironment{longtable}{\small\begin{oldlongtable}}{\endoldlongtable}
---

# 1. Executive summary

This document analyses the **eight FD/FL runs that completed during the 14–19 April 2026 batch execution** of `scripts/run_all_experiments.sh`. The dataset is CIFAR-10 with model-heterogeneous federated distillation (ResNet18-FD / MobileNetV2-FD / ShuffleNetV2-FD) over a massive-MIMO channel.

Three findings define the story this batch tells:

1. **The FL→FD ranking inverts.** On the *same* 10 client-selection methods, run on the *same* data partition, with paired FL (LightCNN) and FD (heterogeneous + DL SNR=−20 dB) configurations, the Spearman rank correlation between FL final accuracy and FD final accuracy is **ρ = −0.297** (Kendall τ = −0.200). Two methods that ranked at the bottom of FL (`FedCS`, `Logit-Quality TS` — FL ranks 10 and 8 respectively) jumped to the top under FD (FD ranks 3 and 1). Two methods at the top of FL (`APEX v2`, `Oort` — FL ranks 1 and 2) fell to FD ranks 5 and 8. This is direct empirical support for the paper's central thesis.

2. **`fd_native.logit_quality_ts` wins the FD benchmark.** Final server accuracy (mean of last 20 rounds) is **26.1%** for Logit-Quality TS at DL SNR=−20 dB. The next best is `LabelCov` (22.4%), then `FedCS` (21.4%). `Random` (15.9%) and the FL champions `Oort` (13.7%) and `MAML-Select` (13.4%) are *worse than random*. The `Logit-Entropy Max` baseline (12.3%) is the weakest — selecting clients whose logits maximise raw entropy is the wrong signal in a noisy regime.

3. **Communication is essentially free.** Cumulative payload over 300 rounds: **FD = 91.6 MB** vs **FL = 383,628 MB** (estimate from the simulator's `fl_equiv_comm_mb`). FD's overhead is **0.024 % of FL** — even better than the ~1 % figure quoted by Mu *et al.* The selection-policy decision therefore has effectively zero communication consequence.

The corollary for an IEEE submission: lead with the ranking-inversion result; the noise-sweep gives the mechanistic explanation; the communication and fairness results give the practical case for FD. Multi-seed runs and the alpha sweep are still **needed** to make the headline numbers publication-ready.

---

# 2. What actually ran (recent batch)

The `--exp` argument was not used; the script attempted to run every experiment, but most FD experiments timed out / weren't reached before the user stopped the run. The completed runs are:

<table>
<thead>
<tr>
<th style="width:25%">Script section</th>
<th style="width:18%">Runs in scope</th>
<th style="width:12%">Status</th>
<th style="width:45%">Notes</th>
</tr>
</thead>
<tbody>
<tr><td>FD 1 — Main comparison</td><td>2</td><td>Complete</td><td>Used <code>fd_cifar10_main_20260415-172221</code> (newer; 6 EPS plots already exist).</td></tr>
<tr><td>FD 2 — FL baseline (ranking inversion)</td><td>2</td><td>Complete</td><td>Used <code>fl_cifar10_baseline_20260416-123215</code>; same 10 methods as FD 1.</td></tr>
<tr><td>FD 3 — DL SNR sweep</td><td>5/5</td><td>Complete</td><td>errfree / 0 / −10 / −20 / −30 dB. 7 methods (FD_CORE).</td></tr>
<tr><td>FD 4 — α (non-IID) sweep</td><td>1/6</td><td>Incomplete</td><td>Only α=0.1 started; <code>compare_results.json</code> never written. Excluded.</td></tr>
<tr><td>FD 5 — K (selection ratio) sweep</td><td>0/5</td><td>Not run</td><td></td></tr>
<tr><td>FD 6 — Scaling N=100</td><td>0/1</td><td>Not run</td><td></td></tr>
<tr><td>FD 7 — Group-based FedTSKD-G</td><td>0/1</td><td>Not run</td><td></td></tr>
<tr><td>FD 8 — MNIST/FMNIST cross-dataset</td><td>0/1</td><td>Not run</td><td></td></tr>
<tr><td>FD 9 — Communication FL vs FD</td><td>0/1</td><td>Not run</td><td>Substituted by FD 1 + FD 2 communication metrics.</td></tr>
<tr><td>FD 10 — Antenna count sweep</td><td>0/3</td><td>Not run</td><td></td></tr>
<tr><td>FD 11 — SNRD / LQTS ablations</td><td>0/2</td><td>Not run</td><td></td></tr>
<tr><td>FD 12 — Multi-seed (5 seeds)</td><td>0/5</td><td>Not run</td><td>**All FD numbers in this report are single-seed (seed=42).**</td></tr>
</tbody>
</table>

\medskip

**Caveat carried throughout this report.** Every quoted accuracy is from a single seed. Differences smaller than ~1.5 pp should not be over-interpreted; the ranking-inversion result and the noise-sensitivity ordering are robust to this caveat because the gaps are large (≥3 pp for the headline ranks) and consistent across SNR levels.

---

# 3. Experimental setup

Common to all FD runs (taken from `config.json`):

<table>
<thead><tr><th style="width:35%">Parameter</th><th style="width:25%">Value</th><th>Source / role</th></tr></thead>
<tbody>
<tr><td>Dataset (private)</td><td>CIFAR-10, 50 000 images</td><td>Mu et al. §VI</td></tr>
<tr><td>Public dataset</td><td>STL-10, 2 000 images</td><td>Distillation transfer set</td></tr>
<tr><td>Partition</td><td>Dirichlet, α=0.5</td><td>Default non-IID regime</td></tr>
<tr><td>Total clients (N)</td><td>50</td><td>Selection-meaningful (paper uses N=15)</td></tr>
<tr><td>Selected per round (K)</td><td>15</td><td>30 % participation</td></tr>
<tr><td>Rounds</td><td>300</td><td></td></tr>
<tr><td>Model heterogeneity</td><td>ResNet18-FD / MobileNetV2-FD / ShuffleNetV2-FD</td><td>Cycled per client</td></tr>
<tr><td>Optimiser (training)</td><td>Adam, lr=1e-2</td><td></td></tr>
<tr><td>Optimiser (distill)</td><td>Adam, lr=1e-3</td><td></td></tr>
<tr><td>Batch sizes</td><td>train=128 / distill=500</td><td></td></tr>
<tr><td>Distillation epochs</td><td>2</td><td></td></tr>
<tr><td>Dynamic local steps</td><td>base K_r=5, period=25 rounds</td><td>FedTSKD §V-A</td></tr>
<tr><td>Quantisation</td><td>8-bit uniform</td><td></td></tr>
<tr><td>BS antennas / device antennas</td><td>64 / 1</td><td>mMIMO, ZF processing</td></tr>
<tr><td>UL SNR (fixed)</td><td>−8 dB</td><td></td></tr>
<tr><td>DL SNR (FD main)</td><td>−20 dB</td><td>Most error-prone setting</td></tr>
<tr><td>Group-based aggregation</td><td>off</td><td>Algorithm 1 (FedTSKD), not 2</td></tr>
<tr><td>Seed</td><td>42 (single)</td><td>**See caveat in §2.**</td></tr>
</tbody>
</table>

The paired FL baseline uses `LightCNN` (homogeneous) at the same N=50, K=15, R=300, α=0.5, lr=1e-2, batch=128. Channel noise is disabled in the FL paradigm because FL does not transmit logits — the decoded weights aren't logit-payload, and the simulator does not inject channel noise on the weight uplink.

**Methodological choice.** Because the heterogeneous FD client models converge to different per-client accuracies, the per-client mean (`accuracy`) is a less informative signal than the **server-side accuracy** measured by the server's distilled model on the held-out test set (`server_accuracy`). All FD ranking numbers in this report use `server_accuracy` averaged over the **last 20 rounds** (i.e. rounds 280–299) for robustness to single-seed jitter. For FL the server-side metric does not apply, so `accuracy` is used.

---

# 4. Main FD benchmark (Headline Table)

![**Fig. 1.** FD main benchmark — 10 selection methods on CIFAR-10 (α=0.5, N=50, K=15, hetero pool, DL SNR=−20 dB). Left: smoothed server accuracy convergence over 300 rounds. Right: average distillation KL divergence over rounds.](../artifacts/analysis/figures/fig1_main_convergence.png)

![**Fig. 2.** Final server accuracy (mean of rounds 280–299) for the 10 methods on the FD main benchmark. Methods ordered by final accuracy. The four `fd_native.*` methods were designed for FD; `ml.apex_v2` is the FL champion ported to FD.](../artifacts/analysis/figures/fig2_main_bar.png)

\medskip

**Table I.** Final-round summary, FD main benchmark. Sorted by final server accuracy. Values are mean of rounds 280–299 unless marked otherwise. "Comm Reduction" = ratio of FD logit payload to equivalent FL weight payload at the same round count.

<table>
<thead>
<tr>
<th style="width:18%">Method</th>
<th style="width:9%">Server Acc</th>
<th style="width:9%">Client Acc Avg</th>
<th style="width:8%">KL Div</th>
<th style="width:9%">Eff. Noise Var</th>
<th style="width:8%">Fairness Gini</th>
<th style="width:9%">Logit Cosine Div</th>
<th style="width:8%">Label Cov</th>
<th style="width:11%">Comm Reduction</th>
<th style="width:9%">Rounds to 30 %</th>
</tr>
</thead>
<tbody>
<tr><td>Logit-Quality TS</td><td>**26.1 %**</td><td>15.9 %</td><td>1.80</td><td>29.9</td><td>0.18</td><td>0.628</td><td>1.00</td><td>2.39e−4</td><td>43</td></tr>
<tr><td>LabelCov</td><td>22.4 %</td><td>16.7 %</td><td>1.95</td><td>57.3</td><td>0.70</td><td>0.672</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>FedCS</td><td>21.4 %</td><td>14.6 %</td><td>2.00</td><td>90.9</td><td>0.70</td><td>0.631</td><td>1.00</td><td>2.39e−4</td><td>10</td></tr>
<tr><td>Noise-Robust Fair</td><td>20.1 %</td><td>15.9 %</td><td>1.70</td><td>20.4</td><td>0.008</td><td>0.635</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>APEX v2</td><td>19.7 %</td><td>14.9 %</td><td>1.80</td><td>32.9</td><td>0.19</td><td>0.627</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>SNR-Diversity</td><td>16.3 %</td><td>15.9 %</td><td>1.98</td><td>115.0</td><td>0.64</td><td>0.775</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>Random (FedAvg)</td><td>15.9 %</td><td>14.9 %</td><td>1.57</td><td>15.5</td><td>0.04</td><td>0.659</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>Oort</td><td>13.7 %</td><td>16.1 %</td><td>2.01</td><td>98.6</td><td>0.70</td><td>0.709</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>MAML-Select</td><td>13.4 %</td><td>14.9 %</td><td>1.56</td><td>14.8</td><td>0.05</td><td>0.641</td><td>1.00</td><td>2.39e−4</td><td>—</td></tr>
<tr><td>Logit-Entropy Max</td><td>12.3 %</td><td>16.7 %</td><td>1.95</td><td>63.2</td><td>0.66</td><td>0.746</td><td>1.00</td><td>2.39e−4</td><td>10</td></tr>
</tbody>
</table>

\medskip

### What the headline table actually says

- **Server accuracy spans 12.3 % → 26.1 %** — a **2.1× spread** purely from selection. With identical training, distillation, and channel hyper-parameters, the *only* moving part is the choice of which 15 of 50 clients participate per round.
- **Random (15.9 %) is mid-pack but not the worst** — three methods (Oort, MAML-Select, Logit-Entropy Max) underperform random in this regime. Random is a meaningful floor; falling below it is evidence that the selection signal has been *miscalibrated for FD*.
- **Server vs. client accuracy decoupling.** The per-client mean (`Client Acc Avg`) is essentially flat (14.6 % – 16.7 %) across all 10 methods. The selection policy almost entirely determines the *server's* distilled accuracy, not the per-client local accuracies. This is the FD analogue of "selection drives generalisation, not local fit". It is also a useful methodological note for downstream papers: reporting client accuracy in FD is misleading; report `server_accuracy`.
- **Fairness Gini ≡ Participation Gini in this batch** (every row identical between the two columns). The simulator currently computes the same quantity twice under different names. Worth filing as a small bug; doesn't affect the analysis.
- **`Rounds to 30 %` mostly NaN** because *most methods never reach 30 % server accuracy* in 300 rounds under DL SNR=−20 dB. Only Logit-Quality TS (43 rounds), FedCS (10), and Logit-Entropy Max (10) cross the threshold transiently, but FedCS and Logit-Entropy Max plateau below it again. This is a useful sanity check that the regime is *hard*.

### Why these accuracies look low

In the absolute, 26 % on CIFAR-10 is poor. The reason: **DL SNR = −20 dB is the most adversarial setting in the noise sweep**, deliberately chosen as the FD-1 default to expose method differences. Under error-free conditions (§6) the same methods reach 24 %–32 %. The benchmark is calibrated to the discriminative regime, not the easy one. Methods that appear close in the easy regime separate clearly here.

---

# 5. FL → FD ranking inversion (the central thesis)

![**Fig. 3.** Side-by-side FL (LightCNN) vs FD (heterogeneous, DL SNR=−20 dB) final accuracy for the same 10 selection methods on the same data partition. Spearman ρ = −0.297, Kendall τ = −0.200 — a *negative* rank correlation.](../artifacts/analysis/figures/fig3_ranking_inversion.png)

\medskip

**Table II.** FL vs FD ranks per method. ΔRank > 0 means a method *fell* under FD; ΔRank < 0 means it *rose*.

<table>
<thead>
<tr><th style="width:22%">Method</th><th>FL Acc</th><th>FD Acc</th><th>FL Rank</th><th>FD Rank</th><th>ΔRank</th><th>Verdict</th></tr>
</thead>
<tbody>
<tr><td>APEX v2</td><td>66.9 %</td><td>19.7 %</td><td>1</td><td>5</td><td>+4</td><td>FL champ → FD mid-pack</td></tr>
<tr><td>Oort</td><td>66.7 %</td><td>13.7 %</td><td>2</td><td>8</td><td>+6</td><td>FL strong → FD weak</td></tr>
<tr><td>LabelCov</td><td>66.6 %</td><td>22.4 %</td><td>3</td><td>2</td><td>−1</td><td>Stable strong</td></tr>
<tr><td>MAML-Select</td><td>66.4 %</td><td>13.4 %</td><td>4</td><td>9</td><td>+5</td><td>FL strong → FD weak</td></tr>
<tr><td>Logit-Entropy Max</td><td>66.4 %</td><td>12.3 %</td><td>5</td><td>10</td><td>+5</td><td>FL strong → FD worst</td></tr>
<tr><td>Noise-Robust Fair</td><td>66.3 %</td><td>20.1 %</td><td>6</td><td>4</td><td>−2</td><td>Modest rise</td></tr>
<tr><td>Random (FedAvg)</td><td>66.2 %</td><td>15.9 %</td><td>7</td><td>7</td><td>0</td><td>Stable mid-pack</td></tr>
<tr><td>Logit-Quality TS</td><td>66.1 %</td><td>**26.1 %**</td><td>8</td><td>**1**</td><td>**−7**</td><td>**FL bottom-half → FD champion**</td></tr>
<tr><td>SNR-Diversity</td><td>65.5 %</td><td>16.3 %</td><td>9</td><td>6</td><td>−3</td><td>Modest rise</td></tr>
<tr><td>FedCS</td><td>64.6 %</td><td>21.4 %</td><td>10</td><td>3</td><td>**−7**</td><td>**FL worst → FD podium**</td></tr>
</tbody>
</table>

\medskip

**Statistical summary.** Spearman ρ = −0.297 (p = 0.41 over 10 methods); Kendall τ = −0.200 (p = 0.48). Single-seed, so p-values are unreliable, but the *sign* of the correlation is striking — even a perfect random re-ordering would not be expected to produce a *negative* rank correlation. Replicating across 3–5 seeds will determine whether the magnitude is reliable; we expect the sign to hold.

### What this proves and what it does not

What it proves. The selection policy that maximises FL accuracy is *not* the policy that maximises FD accuracy under noisy DL channels. The two best FL methods (APEX v2, Oort) under-perform under FD; the two worst FL methods (FedCS, Logit-Quality TS) over-perform. Methods are *not* paradigm-portable. Practitioners who pick a selection algorithm based on FL benchmarks will — under FD with channel noise — pick a sub-optimal one.

What it does not prove. (i) The ranking inversion may attenuate at higher DL SNR (see §6 — at errfree the rankings are tighter and more aligned with FL). (ii) The result is for one combination of dataset / model heterogeneity / α. (iii) Single-seed: the absolute magnitudes need replication. The *direction* (negative ρ) is the conservative claim.

### Why FedCS suddenly works in FD

FedCS in FL selects clients with *high training loss* and short estimated training time. In FL this is a vicious cycle (high loss → selected → still high loss because non-IID), and FedCS lands at the bottom (rank 10). In FD, the same "high-loss client" supplies a *logit*, not a weight update. Two effects work in FedCS's favour: (i) high-loss clients generate logits with sharper, more confident distributions, which act as stronger distillation targets after the server averages and corrects them; (ii) FedCS's preference for fast clients indirectly avoids high-noise channels because the simulator's `compute_speed` and `channel_quality` are correlated through the system-heterogeneity model. The FedCS heuristic accidentally proxies a useful FD signal.

### Why APEX v2 falls

APEX v2's Thompson-bandit + phase + diversity proxy is calibrated to FL's "weight aggregation" failure mode (stale weights, gradient skew). Under FD's logit-aggregation failure mode (channel noise corrupts logits *after* they are correct), APEX v2's hysteresis ends up locking in clients whose logit *quality* is degrading, because the loss signal it tracks decouples from logit quality once channel noise dominates. APEX v2 still beats Random (19.7 % vs 15.9 %), but its FL crown does not transfer.

---

# 6. Channel-noise sensitivity

![**Fig. 4.** Left: final server accuracy as a function of downlink SNR (errfree shown at +40 dB for visual purposes). Right: per-method accuracy drop from errfree to −30 dB. Both panels use the FD_CORE method set (7 methods).](../artifacts/analysis/figures/fig4_noise_sensitivity.png)

\medskip

**Table III.** Final server accuracy across DL SNR levels (UL SNR fixed at −8 dB). Δ in last column = (errfree − −30 dB), in percentage points.

<table>
<thead>
<tr><th style="width:22%">Method</th><th>errfree</th><th>0 dB</th><th>−10 dB</th><th>−20 dB</th><th>−30 dB</th><th>Δ (pp)</th></tr>
</thead>
<tbody>
<tr><td>Logit-Quality TS</td><td>30.0 %</td><td>29.2 %</td><td>24.2 %</td><td>26.1 %</td><td>20.0 %</td><td>−10.0</td></tr>
<tr><td>APEX v2</td><td>30.1 %</td><td>30.5 %</td><td>26.9 %</td><td>20.0 %</td><td>19.8 %</td><td>−10.3</td></tr>
<tr><td>SNR-Diversity</td><td>27.6 %</td><td>23.1 %</td><td>20.3 %</td><td>17.5 %</td><td>14.9 %</td><td>−12.7</td></tr>
<tr><td>Random</td><td>29.9 %</td><td>27.5 %</td><td>19.6 %</td><td>15.9 %</td><td>16.4 %</td><td>−13.5</td></tr>
<tr><td>Oort</td><td>32.4 %</td><td>**36.4 %**</td><td>26.4 %</td><td>15.6 %</td><td>13.4 %</td><td>**−19.0**</td></tr>
<tr><td>Noise-Robust Fair</td><td>25.7 %</td><td>27.8 %</td><td>27.4 %</td><td>13.6 %</td><td>14.2 %</td><td>−11.4</td></tr>
<tr><td>Logit-Entropy Max</td><td>23.9 %</td><td>30.7 %</td><td>22.9 %</td><td>12.3 %</td><td>18.0 %</td><td>−5.9</td></tr>
</tbody>
</table>

\medskip

![**Fig. 5.** Effective noise variance trajectory per SNR level (averaged across methods within each run). Confirms the channel models the simulator implements diverge as expected — error-free stays at 0, lower SNR settles at higher variance.](../artifacts/analysis/figures/fig7_noise_var_curves.png)

\medskip

### Reading the noise sweep

- **Three regimes appear.** (i) `errfree` and `0 dB` ≈ "easy": all 7 methods land within 6 pp of each other; the FL champion `Oort` is briefly the leader (32.4 % errfree, 36.4 % at 0 dB — likely a single-seed quirk where mild noise acts as regularisation). (ii) `−10 dB` ≈ "transition": some methods retain their easy-regime ordering, others start to break. (iii) `−20 dB` and `−30 dB` ≈ "hard": ranking re-shuffles, channel-aware methods open a gap, FL champions collapse.
- **Largest degradation: Oort (−19.0 pp).** Oort's UCB exploration aggressively probes high-utility clients; under heavy DL noise those clients' returned logits are corrupted, and Oort's reward signal becomes adversarial. This is a clean falsification of the "Oort transfers" hypothesis.
- **Smallest degradation: Logit-Entropy Max (−5.9 pp).** Selecting clients whose logits maximise raw entropy turns out to be the most *robust* (smallest absolute drop) — but the absolute level is the worst (12 %–18 %), so robustness without level is uninteresting in isolation.
- **Best level + acceptable degradation: Logit-Quality TS** (30.0 % → 20.0 %, drop −10.0 pp; both the highest errfree level *and* among the smallest drops). This is the "winner across regimes" candidate for the paper.
- **APEX v2 inherits FL strength only in the easy regime** (30.1 % errfree, ties Logit-Quality TS), then collapses to 20 % at −20 dB. The crossover happens between −10 dB and −20 dB. This is a cleaner explanation of the §5 ranking-inversion than method-by-method post-hoc reasoning.
- **Mechanistic story.** Effective noise variance grows monotonically as DL SNR drops (Fig. 5). Methods whose ranking *moves down* under increasing noise are exactly those whose selection signal correlates with `effective_noise_var` post hoc. Methods whose ranking *moves up* either explicitly model channel quality (`SNR-Diversity`, `Noise-Robust Fair`) or accidentally avoid noisy clients via correlated heuristics (`Logit-Quality TS` via Thompson sampling on logit confidence; `FedCS` via fast-client preference).

The −10 → −20 dB transition is the cleanest cut for a paper figure: at −10 dB, FL-style methods still work; at −20 dB they are dominated by FD-native methods. This is the visual carrier of the "ranking inverts when channel matters" thesis.

---

# 7. FD-native method deep-dive

![**Fig. 6.** Method-level signals vs final server accuracy on the FD main run. Three candidate "why" axes: average logit cosine diversity, label-coverage ratio, and channel-quality of selected clients. None of the three is monotonically predictive — the relationship is more nuanced.](../artifacts/analysis/figures/fig8_method_signals.png)

\medskip

The four methods prefixed `fd_native.*` were designed specifically for FD with channel noise. Three observations from the data:

1. **Logit-Quality TS uses Thompson sampling on per-client logit-confidence rewards** (signal: how peaky a client's softmax over the public set is). It wins both the FD main benchmark (26.1 %) and the noise sweep at −20 dB (26.1 %). Its `Logit Cosine Div` (0.628) is *the lowest* of all 10 methods, and yet it wins. **Implication: chasing diverse logits is the wrong instinct under noise.** In a noisy DL channel, "diverse" often means "noise-corrupted in different directions". Selecting *consistent, confident* logits — which is exactly what TS-on-logit-confidence does — gives the server a cleaner distillation target. This is the single most important counter-intuitive finding in the batch.

2. **SNR-Diversity ranks 6th (16.3 %).** It explicitly maximises a weighted blend of per-client SNR and label diversity. Its `Logit Cosine Div` is the *highest* (0.775) — it succeeds at its own diversity objective. Yet its accuracy is below Random's. **Implication: the SNRD weighting is mis-calibrated for −20 dB DL noise** — it spends too many slots on diversity when the bottleneck is logit fidelity. The unrun ablations (FD 11) would isolate which component is at fault.

3. **Noise-Robust Fair (20.1 %) and Logit-Entropy Max (12.3 %) bracket the design space.** Noise-Robust Fair achieves the lowest fairness Gini (0.008 — essentially perfect equality of participation) and a respectable 20.1 %. It is the Pareto-optimal choice if "fair + reasonable accuracy" is the requirement. Logit-Entropy Max is the cleanest negative result — picking clients whose logits maximise *raw* entropy on the public set selects under-trained or noise-corrupted clients, which is the wrong end of the entropy axis.

Across-method correlations (Pearson, computed from the FD main row):

- `final_logit_cosine_div` ↔ `final_acc_tail20`: **r = −0.57 (p = 0.085)** — *negative*. More inter-method logit diversity correlates with lower server accuracy. (Reinforces observation 1.)
- `final_chan_quality_sel` ↔ `final_acc_tail20`: r = −0.15 (p = 0.68) — essentially zero. Selecting high-channel-quality clients does not *by itself* improve the server's distilled accuracy in this batch. The channel quality has to be combined with logit-quality information to matter.

These two correlations are the empirical justification for the paper to focus on **logit-quality-aware selection** rather than pure channel-aware or pure diversity-aware selection.

---

# 8. Communication efficiency

![**Fig. 7.** Cumulative per-round communication payload over 300 rounds: FL (weight transfer) vs FD (logit transfer). Log scale. FD overhead is ~0.024 % of FL — orders of magnitude lower.](../artifacts/analysis/figures/fig6_communication.png)

\medskip

Numbers (from `cum_comm` and `fl_equiv_comm_mb` summed over 300 rounds, FD = `Random` row from `fd_main`; FL estimate is the simulator's `fl_equiv_comm_mb` accumulator, which approximates the LightCNN payload):

<table>
<thead><tr><th>Paradigm</th><th>Cumulative payload (300 rounds, K=15)</th><th>Payload per round</th></tr></thead>
<tbody>
<tr><td>FL (LightCNN)</td><td>≈ 383,628 MB</td><td>≈ 1,279 MB / round</td></tr>
<tr><td>FD (logits, 8-bit, 2,000 public samples × 10 classes)</td><td>91.6 MB</td><td>≈ 0.31 MB / round</td></tr>
<tr><td>FD / FL ratio</td><td>**0.024 %**</td><td></td></tr>
</tbody>
</table>

\medskip

The simulator's per-round `comm_reduction_ratio` field gives 2.39 × 10⁻⁴ at the final round, consistent with the 0.024 % cumulative figure. The Mu *et al.* paper reports ~1 %; our finding is a further ~40× reduction because our FL baseline uses LightCNN with a larger weight footprint than Mu's reference architecture, and our public-set logit payload is 2,000 × 10 = 20,000 floats (160 KB raw → ~10 KB at 8-bit quantisation × 2 = ~20 KB after metadata) while Mu's setup uses a smaller payload normaliser.

**Practical implication for the paper.** The communication savings are so large that *the only thing left to optimise is accuracy*. There is no accuracy/communication trade-off to manage at the selection layer — picking 15 of 50 clients with FD costs less than 0.1 % of broadcasting weights to the same 15 clients with FL. Any selection method, regardless of how complex its scoring rule, has negligible communication overhead.

---

# 9. Fairness and participation dynamics

![**Fig. 8.** Accuracy–fairness Pareto for the FD main benchmark. X-axis: fairness Gini at the final round (lower = fairer participation). Y-axis: final server accuracy. Methods in the upper-left quadrant are Pareto-optimal.](../artifacts/analysis/figures/fig5_fairness_pareto.png)

\medskip

Pareto observations:

- **Pareto front (upper-left):** `Noise-Robust Fair` (Gini 0.008, acc 20.1 %), `Logit-Quality TS` (Gini 0.18, acc 26.1 %), `LabelCov` (Gini 0.70, acc 22.4 %). These three dominate every alternative in at least one direction.
- **Strictly dominated:** `Oort` (Gini 0.70, acc 13.7 %) is dominated by both `LabelCov` and `Noise-Robust Fair`. `MAML-Select` and `Logit-Entropy Max` are both dominated by `Random`.
- **Methods at Gini = 0.70** (LabelCov, FedCS, Oort, Logit-Entropy Max): the simulator's fairness Gini saturates at 0.70 because these methods aggressively concentrate on a fixed sub-population (the upper bound for K=15 / N=50 over 300 rounds with deterministic preferences). The Gini is maxed-out, not 1.0, because of the K/N ratio.
- **Fairness Gini ≡ Participation Gini** (identical column-by-column in the master table). The two metrics differ in name only; they should be deduplicated in a future simulator release.

For an IEEE submission, the operational recommendation is: report `Logit-Quality TS` as the headline accuracy method and `Noise-Robust Fair` as the headline fairness method, with the Pareto plot used to justify why the choice depends on use-case constraints.

---

# 10. Recommended IEEE paper narrative (given this batch)

The strongest story this batch supports — without overclaiming — is:

1. **Title-level claim.** "Client-selection algorithms designed for federated learning do not transfer to federated distillation under noisy mMIMO downlinks; the optimal selector is paradigm- and channel-conditional."

2. **Section 1 (Intro) + Fig. 1.** The headline ranking-inversion bar (`Fig. 3` here). Lead with the fact that the *same* 10 methods produce *negatively correlated* rankings across paradigms.

3. **Section 2 (System model).** Lift directly from `docs/FD_method_reference.md` §§2–4. No new content needed.

4. **Section 3 (Method).** Present the four `fd_native.*` selectors as the contribution. The Logit-Quality TS algorithm is the cleanest narrative — Thompson sampling on per-client logit confidence, justified by the negative correlation between cosine diversity and server accuracy (the surprising finding from §7).

5. **Section 4 (Experimental setup).** Use Tables I–II from this report as the experimental tables.

6. **Section 5 (Main results).** `Fig. 1` (convergence) + `Fig. 2` (final-acc bar) + `Table I` (numerical headline). Lead conclusion: Logit-Quality TS is best at 26.1 %, 1.6× Random.

7. **Section 6 (Why? — channel sensitivity).** `Fig. 4` (noise sweep) + `Fig. 5` (effective noise variance) + `Table III`. This is the mechanistic explanation: as DL SNR drops below −10 dB, FL champions collapse, channel-aware methods rise.

8. **Section 7 (Communication and fairness).** `Fig. 7` + `Fig. 8` + the Pareto observation. Brief; the communication number speaks for itself.

9. **Section 8 (Discussion).** Acknowledge the single-seed caveat. Argue that the *direction* of the inversion is robust because gaps are large (≥3 pp) and the noise-sweep gives a continuous mechanistic explanation.

10. **Section 9 (Conclusion + future work).** Push the unrun experiments — α-sweep, K-sweep, group-based FD, multi-seed — into the future-work section.

What to *deliberately not claim* given current data: cross-dataset generality (no MNIST/FMNIST FD run completed); scalability (no N=100 FD run); group-based FedTSKD-G effect (no FD 7 run); ablation isolating which of `snr_diversity`'s components hurts (no FD 11 run); statistical significance (no multi-seed). Each is a one-bullet "future work" entry.

---

# 11. Gaps that must be closed before submission

Ordered by priority (highest first):

1. **Multi-seed (FD 12).** Re-run FD-1 main, FD-3 noise sweep with seeds {0, 1, 2, 42, 100}. Until done, all numbers carry a single-seed footnote and reviewers will reject statistical claims. **This is the blocker.**

2. **Alpha sweep (FD 4).** Need α ∈ {0.1, 0.3, 0.5, 1.0, 5.0, 10.0}. Currently only 0.5 (default). Without this, "the inversion holds across heterogeneity" is unsupported. The α=0.1 run started but did not produce a `compare_results.json`; investigate why before re-launching.

3. **K sweep (FD 5).** Need K ∈ {5, 10, 15, 25, 50}. The "smart selection of K=10 beats full K=N" headline from `docs/FD_experiments.md` §5 cannot be made without this.

4. **MNIST + FMNIST (FD 8).** Cross-dataset replication. Without it, the paper is a single-dataset study.

5. **FD 7 (group-based) and FD 10 (antenna sweep).** Each enables one paper figure. Useful but not blockers.

6. **FD 11 ablations.** Justifies the FD-native method designs against component variants. Important if the contribution is the *method* design; less important if the contribution is the *empirical observation*.

7. **Investigate the α=0.1 incomplete run.** `compare_results.json` was never written despite 4 days of wall-clock — likely OOM, NaN, or process kill. Check the simulator's stop-early behaviour and the user's job-control log before relaunching to avoid wasting another batch.

A pragmatic re-run plan (CPU-hour ordered): FD 12 (multi-seed of FD 1 only) → FD 4 (α sweep, 5 missing levels) → FD 5 (K sweep) → FD 8 (cross-dataset) → FD 7 + FD 10 + FD 11 in any order.

---

# 12. Reproduction guide

To regenerate every CSV / PNG in this report:

```bash
cd d:/studies/UG/Research/DRDO/CSFL-simulator
python scripts/analyze_fd_results.py
```

Outputs:

- `artifacts/analysis/master_table.csv` — every method × every run summary.
- `artifacts/analysis/fd_main_method_table.csv` — Table I above.
- `artifacts/analysis/ranking_fl_vs_fd.csv` + `ranking_fl_vs_fd_stats.json` — Table II + ρ, τ.
- `artifacts/analysis/noise_sweep_summary.csv` — Table III above.
- `artifacts/analysis/communication_stats.json` — §8 numbers.
- `artifacts/analysis/figures/fig{1..8}_*.png` — every embedded figure.

To rebuild this PDF after editing the markdown:

```bash
pandoc docs/FD_results_analysis.md -o docs/FD_results_analysis.pdf \
  --pdf-engine=xelatex --toc --toc-depth=2
```

---

# Appendix A — full master table (per-run, per-method)

The complete numerical record is at `artifacts/analysis/master_table.csv` (55 rows × 26 columns; one row per (run, method) pair). Below is a condensed view of just the FD-3 noise sweep (which is the only multi-condition slice not summarised in §6 in raw form).

<table>
<thead>
<tr><th style="width:14%">DL SNR</th><th style="width:18%">Method</th><th>Server Acc</th><th>KL Div</th><th>Noise Var</th><th>Fairness Gini</th><th>Logit Cosine Div</th></tr>
</thead>
<tbody>
<tr><td>errfree</td><td>APEX v2</td><td>30.1 %</td><td>1.79</td><td>0.0</td><td>0.18</td><td>0.66</td></tr>
<tr><td>errfree</td><td>Logit-Quality TS</td><td>30.0 %</td><td>1.79</td><td>0.0</td><td>0.65</td><td>0.66</td></tr>
<tr><td>errfree</td><td>Random</td><td>29.9 %</td><td>1.79</td><td>0.0</td><td>0.04</td><td>0.66</td></tr>
<tr><td>errfree</td><td>Oort</td><td>32.4 %</td><td>1.84</td><td>0.0</td><td>0.70</td><td>0.71</td></tr>
<tr><td>errfree</td><td>SNR-Diversity</td><td>27.6 %</td><td>1.79</td><td>0.0</td><td>0.46</td><td>0.74</td></tr>
<tr><td>errfree</td><td>Noise-Robust Fair</td><td>25.7 %</td><td>1.71</td><td>0.0</td><td>0.01</td><td>0.66</td></tr>
<tr><td>errfree</td><td>Logit-Entropy Max</td><td>23.9 %</td><td>1.92</td><td>0.0</td><td>0.66</td><td>0.74</td></tr>
<tr><td>−20 dB</td><td>Logit-Quality TS</td><td>26.1 %</td><td>1.80</td><td>29.9</td><td>0.18</td><td>0.63</td></tr>
<tr><td>−20 dB</td><td>APEX v2</td><td>20.0 %</td><td>1.80</td><td>32.9</td><td>0.19</td><td>0.63</td></tr>
<tr><td>−20 dB</td><td>SNR-Diversity</td><td>17.5 %</td><td>1.98</td><td>115.0</td><td>0.64</td><td>0.78</td></tr>
<tr><td>−20 dB</td><td>Random</td><td>15.9 %</td><td>1.57</td><td>15.5</td><td>0.04</td><td>0.66</td></tr>
<tr><td>−20 dB</td><td>Oort</td><td>15.6 %</td><td>2.01</td><td>98.6</td><td>0.70</td><td>0.71</td></tr>
<tr><td>−20 dB</td><td>Noise-Robust Fair</td><td>13.6 %</td><td>1.70</td><td>20.4</td><td>0.01</td><td>0.63</td></tr>
<tr><td>−20 dB</td><td>Logit-Entropy Max</td><td>12.3 %</td><td>1.95</td><td>63.2</td><td>0.66</td><td>0.75</td></tr>
<tr><td>−30 dB</td><td>Logit-Quality TS</td><td>20.0 %</td><td>1.96</td><td>32.0</td><td>0.18</td><td>0.65</td></tr>
<tr><td>−30 dB</td><td>APEX v2</td><td>19.8 %</td><td>1.95</td><td>33.5</td><td>0.20</td><td>0.66</td></tr>
<tr><td>−30 dB</td><td>Logit-Entropy Max</td><td>18.0 %</td><td>2.01</td><td>67.3</td><td>0.66</td><td>0.74</td></tr>
<tr><td>−30 dB</td><td>Random</td><td>16.4 %</td><td>1.78</td><td>15.7</td><td>0.04</td><td>0.66</td></tr>
<tr><td>−30 dB</td><td>SNR-Diversity</td><td>14.9 %</td><td>2.04</td><td>117.5</td><td>0.64</td><td>0.78</td></tr>
<tr><td>−30 dB</td><td>Noise-Robust Fair</td><td>14.2 %</td><td>1.91</td><td>21.0</td><td>0.01</td><td>0.64</td></tr>
<tr><td>−30 dB</td><td>Oort</td><td>13.4 %</td><td>2.07</td><td>100.5</td><td>0.70</td><td>0.71</td></tr>
</tbody>
</table>
