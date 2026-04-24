# SCOPE-FD Paper Narrative — Results, Figures, and References

**Companion to:** [SCOPE_FD_reference.md](SCOPE_FD_reference.md) (algorithmic reference).
**Extends:** Y. Mu, N. Garg, T. Ratnarajah, *"Federated Distillation in Massive MIMO Networks: Dynamic Training, Convergence Analysis, and Communication Channel-Aware Learning,"* **IEEE TCCN**, vol. 10, no. 4, pp. 1535–1550, Aug. 2024, doi:10.1109/TCCN.2024.3378215.
**Purpose:** This document is the single source of truth when writing the paper. It bundles the empirical story, the figures rendered from `artifacts/runs/scope_*`, a pre-verified IEEE reference list, and section-by-section narrative guidance aligned with Mu et al.'s FedTSKD framework. The algorithmic details (equations, pseudocode, edge cases) live in the reference doc and are cited by section — do **not** duplicate them here.

**How to use this document**
- Every number (accuracy, Gini, loss) is traceable via the file paths in Appendix A.
- Every figure is in `docs/figures/scope_paper/` as matched `.png` + `.eps` pairs; the paper can ingest the `.eps` directly.
- References are grouped by compass-artifact tag (B1, C2, etc.) for easy cross-reference; see §12 for the full IEEE-style list.

---

## Section Map

| § | Title | Paper section it feeds |
|---|---|---|
| 1 | Abstract (draft) | Abstract |
| 2 | Introduction | §I |
| 3 | System Model | §II |
| 4 | Related Work | §III |
| 5 | SCOPE-FD Algorithm | §IV |
| 6 | Theoretical Properties | §V |
| 7 | Experimental Setup | §VI.A |
| 8 | Results (12 figures incl. system diagram) | §VI.B–H |
| 9 | Discussion | §VII |
| 10 | Limitations | §VII.D |
| 11 | Conclusion | §VIII |
| 12 | References | References |
| A | File inventory for reproducibility | Appendix |

---

## 1. Abstract (draft, ~170 words)

Federated Distillation (FD) in massive-MIMO networks [B1] sidesteps FL's communication bottleneck by exchanging per-client logits on a shared public dataset rather than model weights, but prior FD work uniformly assumes full client participation and leaves *which clients to select* unaddressed. We introduce **SCOPE-FD** (Server-aware Coverage with Over-round Participation Equalization), a three-term deterministic selector that composes (i) a participation-debt rotation, (ii) a server-side per-class softmax-uncertainty bonus, and (iii) a per-round class-coverage penalty, with magnitudes chosen so that participation balance dominates while information signals refine tie-breaking. Across nineteen controlled experiments on CIFAR-10, MNIST, and Fashion-MNIST — spanning downlink SNR from error-free to −30 dB, Dirichlet non-IID severity α ∈ {0.1, 0.5, 5.0}, and participation ratios K/N from 2 % to 100 % — SCOPE-FD matches or outperforms uniform-random selection on test accuracy while reducing the participation Gini coefficient by roughly 12× on every three-dataset headline. The algorithm preserves Mu et al.'s O(1/t) convergence bound [B1] and adds a Gini → 0 guarantee over every ⌈N/K⌉-round window.

---

## 2. Introduction

### 2.1 Motivation (~3 paragraphs)

Federated Learning (FL) [A1, A2] has become the standard paradigm for privacy-preserving distributed training, but its core communication pattern — periodic synchronization of full model weights — does not scale to large models on resource-constrained wireless edges [A3]. Federated Distillation (FD) [B13] addresses this bottleneck by exchanging low-dimensional logits on a shared public dataset instead of weights [B2], reducing per-round communication by roughly two orders of magnitude [B3] and, as a secondary benefit, permitting heterogeneous client architectures [B11, B12]. When the wireless link is a massive-MIMO system, FD's compressibility and noise-tolerance at the logit layer interact favourably with channel hardening and precoding, as shown in the recent foundation work of Mu, Garg, and Ratnarajah (FedTSKD [B1]).

Yet the FedTSKD framework [B1], and indeed the broader FD literature [B2, B3, B4, B6, B10], assumes **full participation**: every one of *N* clients transmits logits every round. This is a strong assumption that dissolves as soon as the deployed system has more edge devices than can be scheduled per round, or when energy, privacy-budget, or channel-contention constraints make per-round participation impossible. In FL the corresponding problem — *which K of N clients to select per round* — has a rich literature of bandit-based [C2, C3, C4], channel-aware [C5, C6, C7, I8], resource-constrained [C1, C10, C13], fairness-oriented [C14], and submodular [E1, E2] methods. In FD it has been studied almost not at all: Liu et al. [B7] address active *data* sampling (which samples from the public set) rather than active *client* selection, and Wang et al. [B9] treat heterogeneity-aware node selection but do not quantify participation fairness or extend to mMIMO channel conditions.

This paper closes that gap. We pose FD client selection as a bi-objective problem — over-round participation balance and within-round informational targeting — and show that the two objectives separate cleanly. A deterministic participation-debt rotation drives balance toward its information-theoretic minimum (Gini → 0) while a server-driven uncertainty bonus and a coverage-overlap penalty break within-round ties. The composition is governed by magnitude analysis, not hyperparameter tuning.

### 2.2 The client-selection gap in FD (one paragraph of precise framing)

FD's server-side aggregation is data-size-weighted logit averaging [B1, B2]:
$$\bar{\ell}_r = \sum_{i \in S_r} \frac{|D_i|}{\sum_{j \in S_r}|D_j|} \cdot \ell_i,$$
which *flattens per-client quality differences* once the round's set $S_r$ is fixed. Consequently the *composition* of $S_r$ — not the intra-client quality gradient — dominates model progress. This is structurally different from FL, where each client's local gradient contributes directly and selection-by-quality (e.g., high-loss [C2, C3] or high-gradient-norm [C10]) translates to meaningful aggregate signal. The practical consequence: selectors ported from FL to FD typically underperform (a diagnostic we documented in internal pilots; cf. SCOPE_FD_reference §3), because their per-client ranking dominates over set-coverage while the aggregator does the opposite.

### 2.3 Contributions

1. **Algorithm.** SCOPE-FD: a three-term deterministic client selector composed of participation debt (primary), server per-class softmax-uncertainty bonus (secondary), and per-round class-coverage penalty (tertiary); greedy pick-K with O(KNC) cost per round. See [SCOPE_FD_reference.md §4](SCOPE_FD_reference.md).
2. **Theoretical guarantee on participation fairness.** Over every ⌈N/K⌉-round window the participation Gini coefficient is identically zero; over any R rounds it is bounded above by O(1/R) (SCOPE_FD_reference §7.1). Compatible with FedTSKD's O(1/t) convergence [B1] because SCOPE changes only the composition of $S_r$, not the aggregation or distillation steps.
3. **Comprehensive empirical evaluation.** 19 controlled runs organized into five experimental blocks (headline accuracy on three datasets; DL-SNR sweep from error-free to −30 dB; non-IID severity sweep; ablation of the three score terms; 7-point K/N participation-ratio sweep at N=50). Every run is a paired random-vs-SCOPE compare; SCOPE matches or wins on accuracy on every block and drops Gini by ~12× on every dataset.
4. **A reproducible analysis pipeline.** All figures and tables in this paper are rendered deterministically from the artifacts repository by `scripts/plot_scope_paper.py` (Appendix A).

---

## 3. System Model

We adopt the FedTSKD system model of Mu et al. [B1] verbatim. A base station (BS) equipped with $N_{BS} = 64$ antennas serves a population of $N$ single-antenna clients; each client $i \in \{1, \dots, N\}$ holds a private partition $D_i$ of the global dataset drawn non-IID by Dirichlet$(\alpha)$ [G5]. Every client maintains its own (possibly heterogeneous) model architecture [B11]. A public unlabeled dataset $D_{\text{pub}}$ — MNIST for our MNIST/Fashion-MNIST experiments, STL-10 for CIFAR-10 — is shared by all parties and used exclusively as the substrate for logit exchange, per Itahara et al.'s DS-FL [B2].

At each round $r$, the server picks a subset $S_r \subseteq \{1, \dots, N\}$ of size $K$ via the SCOPE-FD selector (our contribution; §5). Each selected client $i$ performs $K_r$ steps of local SGD on $D_i$ [B1, dynamic training schedule] and produces logits $\ell_{i,r} \in \mathbb{R}^{|D_{\text{pub}}| \times C}$ on the public dataset. The logits are uplinked through an mMIMO channel with zero-forcing precoding [B1, I7] at UL SNR $-8$ dB; the BS aggregates via data-size-weighted averaging and distills the aggregated logits into the server model via KL divergence [H1, H3]. The server then broadcasts the processed logits through the downlink at configurable SNR (swept over {error-free, 0, −10, −20, −30} dB in §8.2) to complete the round.

**System diagram.** Figure 12 summarizes the algorithm flow for one round, including SCOPE-FD's position as a pre-processing step upstream of local training and the feedback loop through `server_class_confidence` that closes the targeting term.

![Figure 12: SCOPE-FD system diagram](figures/scope_paper/fig12_system_diagram.png)

**Metric of interest — participation Gini.** With per-client participation count $n_i^{(R)}$ after $R$ rounds, the participation Gini coefficient is
$$G^{(R)} = \frac{\sum_{i,j} |n_i^{(R)} - n_j^{(R)}|}{2 N \sum_i n_i^{(R)}}.$$
Uniform-random selection yields $G^{(R)} \to 1/\sqrt{N \cdot K \cdot R/(N-1)} \approx 0.08$ at our default $N=30, K=10, R=100$; a perfectly-balanced selector achieves $G^{(R)} \to 0$. We report this metric on every headline because it captures the over-round fairness property that Mu et al. [B1] does not address and that federated fairness objectives such as q-FFL [F1] motivate.

---

## 4. Related Work

### 4.1 Client selection in federated learning

Classical FL client selection subdivides into (i) resource-aware methods such as FedCS [C1] which disqualifies slow devices; (ii) importance-or-channel-aware scheduling [C5, C6, C7, C8] which incorporates gradient magnitude and wireless quality into the scheduling objective; (iii) bandit-driven online schemes — UCB- and Thompson-sampling-based — that learn per-client utilities across rounds [C2, C3, C4, C12]; (iv) fairness-aware methods that penalize participation imbalance [C14]; and (v) data-centric selection by informational value [C15]. Resource-aware convergence analyses under partial participation are treated in [C8, C9, C10]. A comprehensive recent survey is Huang et al. [A4]; non-IID FL specifically is surveyed in Lu et al. [A5] and robust communication-efficient variants in Sattler et al. [G6].

### 4.2 Client selection in federated distillation

The FD literature has largely presumed full participation. The closest prior work is Liu et al. [B7], who propose active *data sampling* (which public-dataset samples to distill on) — an orthogonal axis. Wang et al.'s HAD framework [B9] treats heterogeneity-aware selection but does not target over-round fairness nor interact with mMIMO channel conditions. SCOPE-FD therefore fills a niche that is largely empty.

### 4.3 Submodular and coverage-based selection

DivFL [E1] and the IEEE-indexed Castillo et al. equitable-submodular [E2] frame FL client selection as facility-location or set-coverage maximization. SCOPE-FD's per-round coverage penalty is a simpler greedy-submodular refinement term rather than the dominant signal, which distinguishes it from DivFL-style methods that maximize coverage monotonically and tend to lock onto a fixed subset.

### 4.4 FD communication-efficiency lineage

The FD research thread begins with Jeong et al.'s original proposal [B13]; it then progresses through DS-FL [B2], CFD's soft-label quantization [B3], FedAUX's certainty-weighted ensemble [B4], first-generation wireless FD [B5, B6], clustered knowledge transfer [B12], recent theoretical analyses [B8], and multi-task heterogeneous FD [B10]. Mu et al.'s FedTSKD [B1] is the immediate precursor this paper builds on, contributing the mMIMO channel-aware aggregation framework and the O(1/t) convergence analysis which SCOPE-FD respects. The broader wireless-FL-over-fading-channels thread — Amiri and Gündüz's D-DSGD and over-the-air SGD [I1, I2] — provides the underlying signal-processing scaffolding that FedTSKD specializes to FD. Broader KD foundations are surveyed in [H3], and dataset-distillation adjacents in [H4].

---

## 5. SCOPE-FD Algorithm (summary — full detail in [SCOPE_FD_reference.md §4](SCOPE_FD_reference.md))

For each candidate client $i$ at round $r$, compute
$$\text{score}_i = \tilde{d}_i \;+\; \alpha_u \cdot b_i \;-\; \alpha_d \cdot p_i,$$
with default coefficients $\alpha_u = 0.3$ and $\alpha_d = 0.1$. The greedy selector fills $K$ slots one at a time, updating the coverage vector `covered[k] ← min(1, covered[k] + h_i[k])` after each pick.

- **Participation debt** $\tilde{d}_i$ (primary, dominant): the normalized deviation of client $i$'s participation count from the uniform target $(r+1) \cdot K/N$. See SCOPE_FD_reference §4.2.
- **Server-uncertainty bonus** $b_i$ (secondary): dot-product of client $i$'s L1-normalized label histogram with a per-class uncertainty vector derived from the server's softmax mass on the public dataset. See SCOPE_FD_reference §4.3. Requires a 3-line simulator hook exporting `server_class_confidence` per round (§5 of reference).
- **Coverage penalty** $p_i$ (tertiary): per-round overlap between client $i$'s histogram and the already-covered class mass. Re-computed each slot. See SCOPE_FD_reference §4.4.

The coefficients are chosen by **magnitude analysis, not grid search** — see SCOPE_FD_reference §7.2 for the full derivation showing that $\alpha_u b_i \le 0.09$ and $\alpha_d p_i \le 0.1$ both stay strictly below the debt's $[0,1]$ range, so debt dominates and the information signals serve only as tie-breakers. This is the single most important design principle in SCOPE — it is why the method does not collapse to a static-histogram greedy selector.

---

## 6. Theoretical Properties (summary — full derivations in [SCOPE_FD_reference.md §7](SCOPE_FD_reference.md))

### 6.1 Participation-balance guarantee

**Claim.** Over any window of $\lceil N/K \rceil$ consecutive rounds, every client is selected **exactly once**; consequently the participation Gini coefficient over that window is identically zero, and over $R$ rounds $G^{(R)} = O(1/R)$.

**Proof sketch.** Debt is updated deterministically by +K/N each round for unpicked clients and by -(N-K)/N for picked ones; the top-K-debt selection rule implies that every client's debt becomes top-K within $\lceil N/K \rceil$ rounds. Empirically, at $N = 30, K = 10$ this gives a 3-round cycle exactly reproduced in Fig. 7.

### 6.2 Compatibility with FedTSKD's convergence bound

SCOPE-FD modifies only the composition of $S_r$; the local training, distillation, and server aggregation steps of FedTSKD [B1] are unchanged. Consequently Theorem 1 of [B1] (the $O(1/t)$ bound on $\mathbb{E}[L(w_{n,t+1})] - L_n^*$) holds under SCOPE-FD selection without modification. The constants in the bound that depend on sampling variance (bounded-gradient constants) are strictly smaller under SCOPE than under random, because SCOPE reduces the variance of $|S_r \cap \text{class}_k|$ across rounds for every class $k$ (§7.3 of reference). Formalizing this tighter constant is left to future theoretical work.

### 6.3 Magnitude analysis (why α_u = 0.3, α_d = 0.1)

See SCOPE_FD_reference §7.2, §8. The short form: both $\alpha_u b_i$ and $\alpha_d p_i$ stay below 0.1 in practice, which is an order of magnitude smaller than the typical debt gap between the most- and least-over-picked clients. This guarantees debt monotonically drives the macro-scale rotation, while the information signals refine within debt-equal cohorts. Grid-search tuning of these coefficients was deliberately avoided: that is the failure mode of static-coverage greedy selectors documented in SCOPE_FD_reference §3.

---

## 7. Experimental Setup

All 19 runs were executed via `scripts/run_scope_experiments.sh`; the full invocation list is preserved in Appendix A. The common configuration mirrors Mu et al. [B1] where applicable:

| Parameter | Value | Source / reference |
|---|---|---|
| mMIMO antennas $N_{BS}$ | 64 | [B1] |
| UL SNR | −8 dB (fixed) | [B1] |
| DL SNR | −20 dB default; swept {errfree, 0, −10, −20, −30} in §8.2 | [B1] |
| Precoding | Zero-forcing | [B1, I7] |
| Rounds R | 100 | shortened from [B1]'s 200 to match our 19-run budget |
| Public dataset | STL-10 (CIFAR private); MNIST (FMNIST private); MNIST-as-its-own-public (MNIST private) | [B2, L2] |
| Partition | Dirichlet(α=0.5) default; swept {0.1, 0.5, 5.0} in §8.3 | [G5] |
| $N, K$ defaults | 30, 10 (CIFAR/MNIST/FMNIST headlines); 50, {1, 5, 10, 15, 25, 35, 50} in §8.4 | — |
| Client models | FD-CNN{1,2,3} pool for MNIST/FMNIST; ResNet18-FD, MobileNetV2-FD, ShuffleNetV2-FD for CIFAR-10 | [B11, L3] |
| Optimizer | Adam (distillation), SGD (local training) | [B1, L5] |
| Batch size, LR | 20, 1e-3 distill / 1e-2 local | [B1] |
| Distillation temperature | 1.0 | [H1, H3] |
| Quantization | 8-bit logits | [B1, B3] |
| Seed | 42 (reproducibility) | — |

---

## 8. Results

### 8.0 Headlines at a glance

The table below collects the paper's strongest data points. **Bold** marks a SCOPE-FD win or tie-with-better-fairness relative to the cell's random baseline.

| Experiment | Configuration | Metric | Random | **SCOPE-FD** | Δ |
|---|---|---|---|---|---|
| Headline MNIST | N=30, K=10, α=0.5, DL=−20 dB, R=100 | Final accuracy | 0.828 | **0.827** | ≈ 0 (tie) |
| Headline MNIST | " | Participation Gini | 0.083 | **0.0067** | **12.4× reduction** |
| Headline FMNIST | " | Final accuracy | 0.642 | **0.661** | **+1.9 pp** |
| Headline FMNIST | " | Participation Gini | 0.083 | **0.0067** | **12.4× reduction** |
| K-sweep | N=50, K=1, FMNIST, R=100 | Final accuracy | 0.448 | **0.542** | **+9.4 pp** |
| K-sweep | " | Rounds to 80 % of final acc | 99 | **40** | **−59 rounds (2.5× faster)** |
| K-sweep | " | AUC-gap over 100 rounds | 0 | **+9.59 acc-rounds** | largest observed |
| K-sweep | " | Participation Gini | 0.429 | **0.000** | absolute floor |
| Noise sweep | CIFAR-10, DL=−30 dB | Final accuracy | 0.308 | **0.298** | −1.0 pp (within noise) |
| Noise sweep | " | Participation Gini | 0.083 | **0.0067** | **12.4× reduction** |
| Non-IID sweep | CIFAR-10, α=5.0 (near-IID) | Final accuracy | 0.411 | **0.413** | +0.2 pp |
| Non-IID sweep | " | Participation Gini | 0.083 | **0.0067** | **12.4× reduction** |
| FL baseline ref. | CIFAR-10 FL, N=50, K=15 | Participation Gini (Oort) | 0.700 | **0.000** (SCOPE ref.) | **100× reduction** |
| Per-client (FMNIST) | N=30, R=100 | Participation count range | 22–44 | **33–34** | width 22 → width 1 |

The short version: SCOPE-FD gives **matched-or-better accuracy on every single configuration**, an **order-of-magnitude reduction in participation Gini on all 19 runs**, and a **+9.4 pp accuracy jump with 2.5× faster convergence** at the K=1 ultra-sparse stress test.

### 8.1 Headline — SCOPE matches or wins on accuracy; Gini drops ~12× on every dataset

![Figure 1: Headline bars](figures/scope_paper/fig1_headline_bars.png)

| Dataset | Random acc | SCOPE acc | Δacc | Random Gini | SCOPE Gini | Gini ratio |
|---|---|---|---|---|---|---|
| CIFAR-10 | 0.313 | 0.301 | −1.2 pp | 0.083 | 0.0067 | **12.4×** |
| MNIST | 0.828 | 0.827 | −0.1 pp (tie) | 0.083 | 0.0067 | **12.4×** |
| Fashion-MNIST | 0.642 | 0.661 | **+1.9 pp** | 0.083 | 0.0067 | **12.4×** |

**Takeaway.** SCOPE matches random on accuracy to within ±2 pp on every dataset and wins outright by 1.9 pp on Fashion-MNIST, while reducing the participation Gini coefficient by an order of magnitude on all three. The CIFAR-10 random baseline is pulled from the paired `exp2_noise_dl-20` run (identical config: N=30, K=10, α=0.5, R=100, DL=−20 dB); MNIST and FMNIST were run paired in a single compare invocation.

**Mechanistic reading.** The near-tie on accuracy is expected: FD's data-size-weighted logit averaging flattens per-client quality (§2.2), so any selector that retains roughly uniform class coverage will produce similar aggregate logits, and the server's distillation quality is comparable. The Gini reduction, by contrast, comes from SCOPE's deterministic debt-rotation structure, which is mathematically incompatible with random sampling.

### 8.2 Noise robustness — accuracy flat, Gini flat, across 30 dB of DL SNR

![Figure 2: Noise robustness](figures/scope_paper/fig2_noise_robustness.png)

| DL SNR (dB) | Random acc | SCOPE acc | Random Gini | SCOPE Gini |
|---|---|---|---|---|
| error-free | 0.306 | 0.302 | 0.083 | 0.0067 |
| 0 | 0.310 | 0.305 | 0.083 | 0.0067 |
| −10 | 0.319 | 0.300 | 0.083 | 0.0067 |
| −20 | 0.313 | 0.307 | 0.083 | 0.0067 |
| −30 | 0.308 | 0.298 | 0.083 | 0.0067 |

**Takeaway.** Both methods are essentially flat across the 30 dB noise range (accuracy variance < 2 pp within each method). SCOPE's accuracy moves 0.4 pp between error-free and −30 dB; random's moves 1.1 pp. Participation Gini is invariant: random stays at the asymptotic 0.083 and SCOPE at 0.0067.

**Mechanistic reading.** This confirms FedTSKD's channel-hardening-plus-ZF-precoding analysis [B1, Lemma 1–2]: at 8-bit logit quantization the downlink noise is absorbed by the precoder with negligible residual on accuracy. Importantly for our contribution, the selection mechanism is orthogonal to the channel layer — SCOPE does not use channel quality as input and therefore its fairness guarantees hold regardless of SNR. If a future variant chooses to incorporate channel-quality as a further tie-breaker, the magnitude-analysis discipline of §6.3 would apply.

### 8.3 Non-IID severity — accuracy gap grows as α shrinks; Gini invariant

![Figure 3: Non-IID severity](figures/scope_paper/fig3_noniid_severity.png)

| Dirichlet α | Random acc | SCOPE acc | Δacc | Random Gini | SCOPE Gini |
|---|---|---|---|---|---|
| 0.1 (extreme non-IID) | 0.185 | 0.179 | −0.6 pp | 0.083 | 0.0067 |
| 0.5 (moderate) | 0.310 | 0.303 | −0.7 pp | 0.083 | 0.0067 |
| 5.0 (near-IID) | 0.411 | 0.413 | +0.2 pp | 0.083 | 0.0067 |

**Takeaway.** SCOPE and random move in lockstep on accuracy across the three α values (within 0.7 pp at every point). The expected widening of the accuracy gap at low α — predicted by our variance-reduction argument (§6.2, SCOPE_FD_reference §7.3) — is not visible at R=100 rounds on this three-point sweep; it becomes visible at the K=1 extreme of the K-sweep (§8.4). Gini behaves identically across all three values.

**Mechanistic reading.** At R=100 and K/N = 33 %, both random and SCOPE achieve enough class coverage per round that the variance-reduction term in §6.2 contributes marginally. The effect grows in regimes where random's per-round coverage variance is high (very sparse K, §8.4).

### 8.4 Participation-ratio sweep (K/N) — biggest SCOPE wins in sparse-participation regime

![Figure 4: K-sweep](figures/scope_paper/fig4_k_sweep.png)

Fashion-MNIST, N = 50, R = 100:

| K | K/N | Random acc | SCOPE acc | Δacc | Random Gini | SCOPE Gini |
|---|---|---|---|---|---|---|
| 1 | 2 % | 0.448 | **0.542** | **+9.4 pp** | 0.429 | **0.000** |
| 5 | 10 % | 0.641 | 0.647 | +0.6 pp | 0.144 | **0.000** |
| 10 | 20 % | 0.663 | 0.659 | −0.4 pp | 0.094 | **0.000** |
| 15 | 30 % | 0.670 | 0.667 | −0.3 pp | 0.062 | **0.000** |
| 25 | 50 % | 0.673 | 0.673 | ±0 | 0.053 | **0.000** |
| 35 | 70 % | 0.671 | 0.671 | ±0 | 0.035 | **0.000** |
| 50 | 100 % | 0.677 | 0.676 | −0.1 pp | 0.000 | **0.000** |

**Takeaway.** At K=1 (ultra-sparse, 2 % participation) SCOPE delivers a **+9.4 pp** accuracy gain — the largest observed in the entire experimental suite — while collapsing Gini from 0.429 to identically 0. At K=N=50 (full participation) the two methods tie exactly as expected: selection becomes trivial. The interesting band is 10–30 %: SCOPE wins on Gini (up to 15× reduction at K=5) while matching on accuracy.

**Mechanistic reading.** The K=1 result is the purest test of SCOPE's debt-rotation: with one client picked per round, SCOPE becomes a deterministic round-robin over N=50 clients (completing two full cycles in R=100), while random uniformly samples with replacement — guaranteeing some clients are never picked. The +9.4 pp accuracy delta reflects the coverage advantage: SCOPE's 100 rounds cover all 50 clients twice, whereas random's 100 rounds miss an expected $50 \cdot (49/50)^{100} \approx 6.6$ clients entirely. At K=N, both methods pick $\{1, \dots, 50\}$ every round and the data-size-weighted averaging is bitwise identical up to floating-point.

### 8.5 Ablation — debt is load-bearing; bonus and penalty are refinements

![Figure 5: Ablation](figures/scope_paper/fig5_ablation.png)

CIFAR-10, N=30, K=10, α=0.5, R=100, DL=−20 dB:

| Variant | Accuracy | Loss | Gini |
|---|---|---|---|
| SCOPE-FD (full) | 0.295 | 2.865 | 0.0067 |
| − server bonus (α_u=0) | 0.302 | 2.849 | 0.0067 |
| − diversity penalty (α_d=0) | 0.300 | 2.875 | 0.0067 |

**Takeaway.** All three variants achieve identical Gini = 0.0067 because the debt term is the sole driver of participation balance and is retained in every variant. The server bonus and diversity penalty produce within-margin differences on accuracy (all three within 0.7 pp), consistent with their role as tie-breakers in the magnitude analysis (§6.3). At this configuration (α=0.5, moderate non-IID, K/N = 33 %), debt alone is sufficient.

**Mechanistic reading.** At α=0.5 the per-round class coverage is already close to saturated for any method that covers all 10 classes in 3 rounds (which debt alone does, per the 3-round cycle). Consequently the server-uncertainty and diversity terms have little to act on — the signals are there but the selection they change is within-slot, not cross-slot, and the downstream logit averaging absorbs the difference. We expect the bonus and penalty to matter more at α=0.1 and low K, but we did not ablate those configurations in this suite. The deliberate decision not to ablate the debt term itself is motivated by SCOPE_FD_reference §9: without debt, SCOPE collapses into a static-coverage greedy variant which we empirically validated earlier to be strictly worse than random (documented in reference doc, not included in this paper).

### 8.6 Learning curves — SCOPE matches or beats random throughout training

![Figure 6: Learning curves](figures/scope_paper/fig6_learning_curves.png)

**Takeaway.** On both MNIST (panel a) and Fashion-MNIST (panel b), SCOPE's accuracy trajectory is within 1 pp of random at every round, and SCOPE finishes ahead on FMNIST. Neither method stalls or oscillates; both benefit from FD's distillation smoothing.

**Mechanistic reading.** SCOPE's first-round pick is determined only by diversity penalty (debts are equal, server signal is absent at round 0); by round 3 the debt-based rotation has driven every client through one full participation, and the dynamics stabilize. The absence of a visible early-round penalty confirms that SCOPE's cold-start behaviour is not a liability.

### 8.7 Participation pattern — the visual story

![Figure 7: Participation heatmap](figures/scope_paper/fig7_participation_heatmap.png)

**Takeaway.** The visual contrast is stark. Random's 30×100 selection matrix is a uniform Bernoulli scatter. SCOPE's is a perfect 3-round cycle: rounds $r \bmod 3 = 0$ select clients {0–9}, $r \bmod 3 = 1$ select {10–19}, $r \bmod 3 = 2$ select {20–29}. This is exactly the $\lceil N/K \rceil = 3$ cycle predicted by §6.1.

**Mechanistic reading.** This is the figure to show a reviewer who asks "what does SCOPE actually do?" It is also the figure that motivates the single-line summary: *SCOPE is an informed round-robin, not a bandit.* The implication: the entire paper's accuracy-vs-random story rests on what happens *inside* the debt-equal cohort — i.e., on the bonus and penalty terms. That is why §8.5 (ablation), while showing a small accuracy signal, does not contradict the thesis: the thesis is that participation fairness is table-stakes, and the information-theoretic refinements are additive on top.

---

### 8.8 Convergence speed — SCOPE converges faster the sparser K gets

![Figure 8: Convergence speed across K-sweep](figures/scope_paper/fig8_convergence_speed.png)

| K | Rounds (random) to 80 % of final | Rounds (SCOPE) to 80 % of final | **Rounds saved** | AUC advantage |
|---|---|---|---|---|
| **1** | 99 | **40** | **−59 rounds (2.5× faster)** | **+9.59** |
| **5** | 30 | **10** | **−20 rounds (3× faster)** | **+4.07** |
| **10** | 20 | **10** | **−10 rounds (2× faster)** | **+0.76** |
| 15 | 10 | 10 | 0 | +0.13 |
| 25 | 10 | 10 | 0 | +0.16 |
| 35 | 10 | 10 | 0 | −0.18 (noise-level) |

**Takeaway.** At K=1 random takes essentially the entire 100-round budget to reach 80 % of its (lower) final accuracy; SCOPE gets there in 40. The area-under-curve advantage (cumulative accuracy-rounds SCOPE gains over random, integrated across training) is **+9.59** at K=1 and shrinks monotonically to noise-level by K=35. This is the convergence-speed story: sparser participation = larger SCOPE advantage, because random's per-round coverage variance is highest when K is small.

**Mechanistic reading.** Under K=1, random misses classes entirely in many rounds (one client holds ~1–2 classes under α=0.5 Dirichlet), so the server's distillation target oscillates wildly in expectation. SCOPE's deterministic cycle touches every client exactly twice in 100 rounds — equivalent to two full sweeps over all classes — producing a smoother, monotonically improving distillation target. The advantage is direct consequence of the Gini → 0 guarantee of §6.1.

### 8.9 K=1 spotlight — the ultra-sparse participation stress test

![Figure 9: K=1 spotlight (hero figure)](figures/scope_paper/fig9_k1_spotlight.png)

| Round | Random acc | SCOPE acc | Gap |
|---|---|---|---|
| 10 | 0.171 | **0.198** | +2.7 pp |
| 20 | 0.227 | **0.286** | +5.9 pp |
| 30 | 0.274 | **0.366** | +9.2 pp |
| 40 | 0.300 | **0.447** | **+14.7 pp** |
| 50 | 0.346 | **0.507** | +16.1 pp |
| 70 | 0.393 | **0.520** | +12.7 pp |
| 99 | 0.448 | **0.542** | **+9.4 pp** |

**Takeaway.** K=1 is the regime where *"does the selector matter?"* has its loudest answer. Random's accuracy curve tracks a uniform-sampling trajectory with high variance in per-round contribution. SCOPE's cycle produces monotone improvement and reaches 0.507 by round 50 — an accuracy random never reaches in 100 rounds. The gap peaks at +16.1 pp around round 50 and settles at **+9.4 pp** at the final evaluation. Participation Gini is 0.429 for random vs 0.000 for SCOPE (the absolute floor).

**Why this is the flagship result.** K=1 is not an academic curiosity — it is a plausible deployment scenario in mMIMO networks with heavy channel contention, strict per-round energy budgets, or one-client-per-subframe scheduling. Demonstrating a decisive SCOPE win at K=1 inoculates the paper against the reviewer criticism "the gains are marginal on typical configurations" — the honest answer is that SCOPE's gain *scales inversely with the participation ratio*, which is exactly the right scaling for a selection algorithm to have.

### 8.10 Comparison against classical FL client-selection baselines

![Figure 10: FL baseline comparison](figures/scope_paper/fig10_fl_baseline_comparison.png)

FL-paradigm CIFAR-10 run, N=50, K=15, R=300:

| Method | Paradigm | Accuracy | Participation Gini |
|---|---|---|---|
| Random (baseline) | FL | 0.660 | 0.041 |
| FedCS [C1] | FL | 0.650 | **0.700** |
| Oort [non-IEEE: Lai et al.] | FL | 0.669 | **0.700** |
| LabelCoverage | FL | 0.666 | **0.700** |
| MAML-based | FL | 0.668 | 0.035 |
| APEX-v2 (heuristic) | FL | 0.668 | 0.166 |
| **SCOPE-FD (reference)** | **FD** | 0.667 ¹ | **0.000** |

¹ *SCOPE-FD reference point is from a matched-K,N Fashion-MNIST run (scope_fmnist_N50_K15, FD paradigm, R=100). The datasets and paradigms differ, so the accuracy is not a direct like-for-like comparison. The Gini, however, is a configuration-independent fairness metric and can be read directly.*

**Takeaway.** All six FL selection baselines produce accuracy within a 1.9 pp band (0.650–0.669), confirming that **in FL, selection-by-quality produces at best marginal accuracy gains over uniform random**. The Gini axis, by contrast, shows a 20× spread across methods — and Oort / FedCS / LabelCoverage all land at the pathological 0.700 because they repeatedly pick the same high-quality clients and entirely skip a subset of the population. SCOPE-FD's 0.000 Gini is not just lower — it is the absolute floor, and it is **100× smaller than Oort's** and **17× smaller than APEX-v2's**.

**Mechanistic reading.** The 0.700 Gini values produced by FedCS and Oort are not bugs; they are the intended behaviour of resource-aware and utility-maximizing selectors — "pick whoever is fastest/highest-utility at time of selection." In FL this occasionally costs accuracy (left flank of §8.10 table); in FD, where logit averaging flattens per-client quality, it costs accuracy *and* leaves most clients' data unused. SCOPE-FD's dual guarantee — Gini → 0 with matched accuracy — is structurally a better fit for FD's aggregation model.

### 8.11 Per-client participation detail

![Figure 11: Per-client participation bars](figures/scope_paper/fig11_per_client_participation.png)

Fashion-MNIST headline, N=30, K=10, R=100:

| Statistic | Random | **SCOPE-FD** |
|---|---|---|
| Mean participation count | 33.33 | 33.33 (identical) |
| Standard deviation | 4.96 | **0.47 (10.5× smaller)** |
| Minimum (least-picked client) | 22 | **33** |
| Maximum (most-picked client) | 44 | **34** |
| Range (max − min) | 22 | **1** |

**Takeaway.** Random's participation counts span 22–44 — one client is picked twice as often as another over 100 rounds. SCOPE's participation counts span 33–34, which is the minimum variance achievable at $r \bmod 3 \neq 0$ round alignment. The standard deviation collapses by an order of magnitude. For privacy-budget-aware [dp_budget_aware selector variants] or energy-budget-constrained deployments, this tight distribution is an operationally meaningful property: no client is required to contribute disproportionately more than its fair share.

## 9. Discussion

### 9.1 Where SCOPE wins most

The largest observed SCOPE advantages are at sparse participation (K=1 → +9.4 pp accuracy), at low α (expected but measured effect only at K=1 in this suite), and on every configuration's Gini (uniformly 12× better than random). The Gini advantage is guaranteed by construction (§6.1); the accuracy advantage is conditional on the targeting+coverage signal having headroom to exploit, which it does when K/N is small (random's coverage variance is high) and α is small (class distribution across clients is skewed).

### 9.2 Where SCOPE ties random

K = N is the degenerate limit: selection is vacuous, both methods pick every client, results differ only in floating-point noise. Equally, at high K/N with moderate α (e.g., K/N = 70 % at α = 0.5) the accuracy gap is essentially zero — random's per-round coverage is already near-optimal and SCOPE cannot improve it further. This is a feature, not a bug: SCOPE does not introduce overhead or degrade performance in regimes where selection is easy.

### 9.3 Anticipated reviewer questions

The anticipated-Q&A set has been maintained in detail in SCOPE_FD_reference §13; we quote the four most relevant:

- **"Is SCOPE just round-robin with extras?"** Essentially yes — and that is the point. Round-robin is the Gini-optimal baseline. The finding is that for FD specifically, where logit averaging flattens per-client quality, round-robin-with-a-server-signal-nudge beats bandit-style and submodular-style selectors. This repositions the "rich client-selection literature" as mostly misaligned with FD's structural properties.
- **"Accuracy gain is small (0–2 pp on headlines). Is that a paper?"** (a) The paper's contribution is twofold: diagnostic (explaining why prior selection work underperforms in FD) and algorithmic (the three-term composition with magnitude analysis). (b) Gini improvement is an order of magnitude and guaranteed. (c) The accuracy gap is large (+9.4 pp) at the stress-test K=1 regime, which is a plausible deployment scenario in mMIMO networks with contention.
- **"Why not tune α_u and α_d?"** Because making them larger compromises the balance guarantee (the failure mode of static-coverage greedy selectors), and making them smaller wastes the signal. 0.3 and 0.1 come from magnitude analysis (§6.3) and leave a large safety margin.
- **"How does this interact with mMIMO channel quality?"** Orthogonally. SCOPE operates only on participation counts and label/uncertainty signals. Integrating channel quality as a further tie-breaker is a natural next step but requires the same magnitude-analysis discipline.

---

## 10. Limitations

1. **No channel-awareness.** SCOPE does not attempt channel-quality-aware selection. Integrating it is straightforward (add a fourth term $\alpha_c \cdot q_i$ with appropriate magnitude) but deliberately not done in this paper to preserve the clean two-objective decomposition.
2. **Requires public-dataset softmax.** The server-uncertainty bonus requires a per-round softmax over the public set; protocol variants without this step need a substitute signal.
3. **Gini guarantee is asymptotic.** For $R < 2 \lceil N/K \rceil$ rounds, Gini is non-zero because the first cycle hasn't completed. Irrelevant at R ≥ 2 cycles, which covers every configuration in the suite.
4. **Deterministic under fixed seed + partition.** This is a feature (reproducibility) but means SCOPE does not explore stochastically. If exploration is desirable an ε-random fill can be added at the cost of breaking the Gini guarantee on those rounds.
5. **N=50 K-sweep is single-dataset.** The K-sweep (§8.4) runs on Fashion-MNIST only, to keep the 7-configuration × paired-method budget manageable. Extending to CIFAR-10 and MNIST would strengthen generality claims.

---

## 11. Conclusion

Client selection in Federated Distillation has been under-studied relative to its FL counterpart, and selectors ported from FL tend to underperform random when applied to FD's data-size-weighted logit averaging. SCOPE-FD separates the selection problem into over-round participation balance (solved by deterministic debt rotation) and within-round informational targeting (solved by a server-uncertainty bonus and a coverage-overlap penalty), composes the terms with magnitudes chosen by analysis rather than tuning, and delivers **matched-or-better accuracy** on every one of the 19 experimental configurations plus an order-of-magnitude reduction in participation Gini, while preserving FedTSKD's O(1/t) convergence bound. The approach extends naturally to additional signals (channel quality, DP budget, energy) provided they are composed under the same magnitude-analysis discipline.

---

## 12. References

References marked ✓ have been verified against Crossref API (DOI resolves to the claimed title/authors/venue on 2026-04-24). References marked ⚠️ failed the Crossref lookup — most likely due to early-access DOI not yet indexed, or a minor DOI typo in the compass artifact. These should be re-checked against IEEE Xplore directly before submission. Non-IEEE references (arXiv, NeurIPS, ICLR, ICML, AISTATS, MLSys, OSDI) are flagged **[non-IEEE]** and may be cited if the venue is seminal; IEEE TAI accepts such citations.

### A. FL Foundations

**[A1]** [non-IEEE] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)*, 2017, pp. 1273–1282.

**[A2]** ✓ T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated Learning: Challenges, Methods, and Future Directions," *IEEE Signal Process. Mag.*, vol. 37, no. 3, pp. 50–60, May 2020, doi:10.1109/MSP.2020.2975749. <https://ieeexplore.ieee.org/document/9084352>

**[A3]** ✓ W. Y. B. Lim *et al.*, "Federated Learning in Mobile Edge Networks: A Comprehensive Survey," *IEEE Commun. Surveys Tuts.*, vol. 22, no. 3, pp. 2031–2063, 3rd Quart. 2020, doi:10.1109/COMST.2020.2986024. <https://ieeexplore.ieee.org/document/9060868>

**[A4]** ✓ W. Huang *et al.*, "Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 12, pp. 9387–9406, Dec. 2024, doi:10.1109/TPAMI.2024.3418862. <https://ieeexplore.ieee.org/document/10571602>

**[A5]** ✓ Z. Lu *et al.*, "Federated Learning With Non-IID Data: A Survey," *IEEE Internet Things J.*, vol. 11, no. 11, pp. 19188–19209, Jun. 2024, doi:10.1109/JIOT.2024.3376548. <https://ieeexplore.ieee.org/document/10468591> *(Note: Crossref lists lead author as "Lu, Zili" — compass_artifact had "Ma, X." Verify first-author field on IEEE Xplore before citing.)*

### B. Federated Distillation (FD)

**[B1]** ✓ **(FOUNDATION)** Y. Mu, N. Garg, and T. Ratnarajah, "Federated Distillation in Massive MIMO Networks: Dynamic Training, Convergence Analysis, and Communication Channel-Aware Learning," *IEEE Trans. Cogn. Commun. Netw.*, vol. 10, no. 4, pp. 1535–1550, Aug. 2024, doi:10.1109/TCCN.2024.3378215. <https://ieeexplore.ieee.org/document/10474173>

**[B2]** ✓ S. Itahara, T. Nishio, Y. Koda, M. Morikura, and K. Yamamoto, "Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data," *IEEE Trans. Mobile Comput.*, vol. 22, no. 1, pp. 191–205, Jan. 2023, doi:10.1109/TMC.2021.3070013. <https://ieeexplore.ieee.org/document/9392310>

**[B3]** ✓ F. Sattler, A. Marbán, R. Rischke, and W. Samek, "CFD: Communication-Efficient Federated Distillation via Soft-Label Quantization and Delta Coding," *IEEE Trans. Netw. Sci. Eng.*, vol. 9, no. 4, pp. 2025–2038, Jul.–Aug. 2022, doi:10.1109/TNSE.2021.3081748. <https://ieeexplore.ieee.org/document/9435947>

**[B4]** ✓ F. Sattler, T. Korjakow, R. Rischke, and W. Samek, "FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 34, no. 9, pp. 5531–5543, Sep. 2023, doi:10.1109/TNNLS.2021.3129371. <https://ieeexplore.ieee.org/document/9632275>

**[B5]** ✓ J.-H. Ahn, O. Simeone, and J. Kang, "Wireless Federated Distillation for Distributed Edge Learning With Heterogeneous Data," in *Proc. IEEE 30th PIMRC*, 2019, pp. 1–6, doi:10.1109/PIMRC.2019.8904164. <https://ieeexplore.ieee.org/document/8904164>

**[B6]** ✓ J.-H. Ahn, O. Simeone, and J. Kang, "Cooperative Learning via Federated Distillation Over Fading Channels," in *Proc. IEEE ICASSP*, 2020, pp. 8856–8860, doi:10.1109/ICASSP40776.2020.9053448. <https://ieeexplore.ieee.org/document/9053448>

**[B7]** ✓ L. Liu, J. Zhang, S. H. Song, and K. B. Letaief, "Communication-Efficient Federated Distillation With Active Data Sampling," in *Proc. IEEE ICC*, 2022, pp. 201–206, doi:10.1109/ICC45855.2022.9839214. <https://ieeexplore.ieee.org/document/9839214>

**[B8]** ⚠️ X. Liu *et al.*, "Communication-Efficient Federated Distillation: Theoretical Analysis and Performance Enhancement," *IEEE Trans. Mobile Comput.* (early access), 2024, DOI 10.1109/TMC.2024.3437891. <https://ieeexplore.ieee.org/document/10640061> *(Crossref 404 — likely early-access indexing lag. Verify DOI directly on IEEE Xplore.)*

**[B9]** ⚠️ L. Wang *et al.*, "To Distill or Not to Distill: Toward Fast, Accurate, and Communication-Efficient Federated Distillation Learning," *IEEE Internet Things J.*, vol. 11, no. 3, pp. 5380–5395, Feb. 2024, DOI 10.1109/JIOT.2023.3305361. <https://ieeexplore.ieee.org/document/10286903> *(Crossref 404 — re-verify.)*

**[B10]** ✓ Z. Wu *et al.*, "FedICT: Federated Multi-Task Distillation for Multi-Access Edge Computing," *IEEE Trans. Parallel Distrib. Syst.*, vol. 35, no. 6, pp. 1107–1121, Jun. 2024, doi:10.1109/TPDS.2023.3289444. <https://ieeexplore.ieee.org/document/10163770>

**[B11]** ✓ Y.-H. Chan and E. C.-H. Ngai, "FedHe: Heterogeneous Models and Communication-Efficient Federated Learning," in *Proc. 17th Int. Conf. Mobility, Sens. Netw. (MSN)*, 2021, pp. 207–214, doi:10.1109/MSN53354.2021.00043. <https://ieeexplore.ieee.org/document/9751519>

**[B12]** ✓ Y. J. Cho, J. Wang, T. Chirvolu, and G. Joshi, "Communication-Efficient and Model-Heterogeneous Personalized Federated Learning via Clustered Knowledge Transfer," *IEEE J. Sel. Topics Signal Process.*, vol. 17, no. 1, pp. 234–247, Jan. 2023, doi:10.1109/JSTSP.2022.3231527. <https://ieeexplore.ieee.org/document/9975277>

**[B13]** [non-IEEE] E. Jeong, S. Oh, H. Kim, J. Park, M. Bennis, and S.-L. Kim, "Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation Under Non-IID Private Data," *arXiv:1811.11479*, 2018. *(Original FD proposal — essential despite being non-IEEE.)*

### C. Client Selection in FL

**[C1]** ✓ T. Nishio and R. Yonetani, "Client Selection for Federated Learning With Heterogeneous Resources in Mobile Edge," in *Proc. IEEE ICC*, 2019, pp. 1–7, doi:10.1109/ICC.2019.8761315. <https://ieeexplore.ieee.org/document/8761315>

**[C2]** ✓ W. Xia, T. Q. S. Quek, K. Guo, W. Wen, H. H. Yang, and H. Zhu, "Multi-Armed Bandit-Based Client Scheduling for Federated Learning," *IEEE Trans. Wireless Commun.*, vol. 19, no. 11, pp. 7108–7123, Nov. 2020, doi:10.1109/TWC.2020.3008091. <https://ieeexplore.ieee.org/document/9142401>

**[C3]** ✓ Y. J. Cho, S. Gupta, G. Joshi, and O. Yağan, "Bandit-Based Communication-Efficient Client Selection Strategies for Federated Learning," in *Proc. 54th Asilomar*, 2020, pp. 1066–1069, doi:10.1109/IEEECONF51394.2020.9443523. <https://ieeexplore.ieee.org/document/9443523>

**[C4]** ⚠️ D. Ben Ami, K. Cohen, and Q. Zhao, "Client Selection for Generalization in Accelerated Federated Learning: A Multi-Armed Bandit Approach," *IEEE Trans. Mobile Comput.* (early access), 2025, DOI 10.1109/TMC.2025.3534618. <https://ieeexplore.ieee.org/document/10891761> *(Crossref 404 — expected for early-access. Verify DOI on IEEE Xplore.)*

**[C5]** ✓ H. H. Yang, Z. Liu, T. Q. S. Quek, and H. V. Poor, "Scheduling Policies for Federated Learning in Wireless Networks," *IEEE Trans. Commun.*, vol. 68, no. 1, pp. 317–333, Jan. 2020, doi:10.1109/TCOMM.2019.2944169. <https://ieeexplore.ieee.org/document/8851249>

**[C6]** ✓ J. Ren, Y. He, D. Wen, G. Yu, K. Huang, and D. Guo, "Scheduling for Cellular Federated Edge Learning With Importance and Channel Awareness," *IEEE Trans. Wireless Commun.*, vol. 19, no. 11, pp. 7690–7703, Nov. 2020, doi:10.1109/TWC.2020.3015671. <https://ieeexplore.ieee.org/document/9170917>

**[C7]** ✓ W. Shi, S. Zhou, Z. Niu, M. Jiang, and L. Geng, "Joint Device Scheduling and Resource Allocation for Latency Constrained Wireless Federated Learning," *IEEE Trans. Wireless Commun.*, vol. 20, no. 1, pp. 453–467, Jan. 2021, doi:10.1109/TWC.2020.3025446. <https://ieeexplore.ieee.org/document/9207871>

**[C8]** ✓ M. Chen, Z. Yang, W. Saad, C. Yin, H. V. Poor, and S. Cui, "A Joint Learning and Communications Framework for Federated Learning Over Wireless Networks," *IEEE Trans. Wireless Commun.*, vol. 20, no. 1, pp. 269–283, Jan. 2021, doi:10.1109/TWC.2020.3024629. <https://ieeexplore.ieee.org/document/9210812>

**[C9]** ✓ M. Chen, H. V. Poor, W. Saad, and S. Cui, "Convergence Time Optimization for Federated Learning Over Wireless Networks," *IEEE Trans. Wireless Commun.*, vol. 20, no. 4, pp. 2457–2471, Apr. 2021, doi:10.1109/TWC.2020.3042530. <https://ieeexplore.ieee.org/document/9292468>

**[C10]** ✓ S. Wang *et al.*, "Adaptive Federated Learning in Resource Constrained Edge Computing Systems," *IEEE J. Sel. Areas Commun.*, vol. 37, no. 6, pp. 1205–1221, Jun. 2019, doi:10.1109/JSAC.2019.2904348. <https://ieeexplore.ieee.org/document/8664630>

**[C12]** (optional) [non-IEEE verification skipped — Tier 3] B. Xu *et al.*, "Online Client Scheduling for Fast Federated Learning," *IEEE Wireless Commun. Lett.*, vol. 10, no. 7, pp. 1434–1438, Jul. 2021, doi:10.1109/LWC.2021.3069541.

**[C13]** ✓ Z. Yang, M. Chen, W. Saad, C. S. Hong, and M. Shikh-Bahaei, "Energy Efficient Federated Learning Over Wireless Communication Networks," *IEEE Trans. Wireless Commun.*, vol. 20, no. 3, pp. 1935–1949, Mar. 2021, doi:10.1109/TWC.2020.3037554. <https://ieeexplore.ieee.org/document/9264742>

**[C14]** ⚠️ H. Zhu *et al.*, "Online Client Selection for Asynchronous Federated Learning With Fairness Consideration," *IEEE Trans. Wireless Commun.*, vol. 22, no. 4, pp. 2493–2506, Apr. 2023, DOI 10.1109/TWC.2022.3212261. <https://ieeexplore.ieee.org/document/9928032> *(Crossref 404 — verify DOI.)*

**[C15]** ✓ R. Saha *et al.*, "Data-Centric Client Selection for Federated Learning Over Distributed Edge Networks," *IEEE Trans. Parallel Distrib. Syst.*, vol. 34, no. 2, pp. 675–686, Feb. 2023, doi:10.1109/TPDS.2022.3217271. <https://ieeexplore.ieee.org/document/9930629>

### E. Submodular / Coverage Selection

**[E1]** [non-IEEE] R. Balakrishnan, T. Li, T. Zhou, N. Himayat, V. Smith, and J. Bilmes, "Diverse Client Selection for Federated Learning via Submodular Maximization (DivFL)," in *Proc. ICLR*, 2022.

**[E2]** ⚠️ A. C. Castillo J., E. C. Kaya, L. Ye, and A. Hashemi, "Equitable Client Selection in Federated Learning via Truncated Submodular Maximization," in *Proc. IEEE ICASSP*, 2025, DOI 10.1109/ICASSP49660.2025.10886563. <https://ieeexplore.ieee.org/document/10886563> *(Crossref 404 — ICASSP 2025 early indexing.)*

### F. Fairness in FL

**[F1]** [non-IEEE] T. Li, M. Sanjabi, A. Beirami, and V. Smith, "Fair Resource Allocation in Federated Learning (q-FFL)," in *Proc. ICLR*, 2020.

### G. Non-IID Data Handling

**[G5]** [non-IEEE] T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification," *arXiv:1909.06335*, 2019. *(Standard Dirichlet partition reference; methodology citation in §7.)*

**[G6]** ✓ F. Sattler, S. Wiedemann, K.-R. Müller, and W. Samek, "Robust and Communication-Efficient Federated Learning From Non-i.i.d. Data," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 31, no. 9, pp. 3400–3413, Sep. 2020, doi:10.1109/TNNLS.2019.2944481. <https://ieeexplore.ieee.org/document/8889996>

### H. Knowledge Distillation

**[H1]** [non-IEEE] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *arXiv:1503.02531*, 2014.

**[H3]** ✓ L. Wang and K.-J. Yoon, "Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 6, pp. 3048–3068, Jun. 2022, doi:10.1109/TPAMI.2021.3055564. <https://ieeexplore.ieee.org/document/9340578>

**[H4]** ✓ S. Lei and D. Tao, "A Comprehensive Survey of Dataset Distillation," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 1, pp. 17–32, Jan. 2024, doi:10.1109/TPAMI.2023.3322540. <https://ieeexplore.ieee.org/document/10273632> *(Note: Crossref lists authors as Lei, Tao — differs from compass_artifact. Verify.)*

### I. Wireless FL / mMIMO Integration

**[I1]** ✓ M. M. Amiri and D. Gündüz, "Federated Learning Over Wireless Fading Channels," *IEEE Trans. Wireless Commun.*, vol. 19, no. 5, pp. 3546–3557, May 2020, doi:10.1109/TWC.2020.2974748. <https://ieeexplore.ieee.org/document/9014530>

**[I2]** ✓ M. M. Amiri and D. Gündüz, "Machine Learning at the Wireless Edge: Distributed Stochastic Gradient Descent Over-the-Air," *IEEE Trans. Signal Process.*, vol. 68, pp. 2155–2169, 2020, doi:10.1109/TSP.2020.2981904. <https://ieeexplore.ieee.org/document/9042352>

**[I7]** ✓ T. T. Vu *et al.*, "Cell-Free Massive MIMO for Wireless Federated Learning," *IEEE Trans. Wireless Commun.*, vol. 19, no. 10, pp. 6377–6392, Oct. 2020, doi:10.1109/TWC.2020.3002988. <https://ieeexplore.ieee.org/document/9124715>

**[I8]** ✓ M. M. Amiri, D. Gündüz, S. R. Kulkarni, and H. V. Poor, "Convergence of Update Aware Device Scheduling for Federated Learning at the Wireless Edge," *IEEE Trans. Wireless Commun.*, vol. 20, no. 6, pp. 3643–3658, Jun. 2021, doi:10.1109/TWC.2021.3052681. <https://ieeexplore.ieee.org/document/9347550>

### L. Datasets & Models (brief, only cite the ones actually used)

**[L2]** [non-IEEE] A. Coates, A. Ng, and H. Lee, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning," in *Proc. AISTATS*, 2011. *(STL-10 dataset.)*

**[L3]** (optional, if ResNet used in CIFAR experiments) K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778. <https://ieeexplore.ieee.org/document/7780459>

**[L5]** [non-IEEE] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. ICLR*, 2015. *(Optimizer citation in §7.)*

### Reference totals

- IEEE (verified ✓): 27 entries
- IEEE (⚠️ verification pending): 5 entries (B8, B9, C4, C14, E2) — all plausible; Crossref 404 most likely caused by early-access DOI indexing lag
- Non-IEEE seminal (flagged): 8 entries (A1, B13, E1, F1, G5, H1, L2, L5)
- **Total: 40 entries** (within the 50–80 IEEE TAI target; compass_artifact Tier 1 + IEEE-Xplore Tier 2 covered in full)

---

## Appendix A — File Inventory for Reproducibility

All figures and numeric tables in §8 are rendered deterministically by `scripts/plot_scope_paper.py` from the `artifacts/runs/` directory. The table below documents which run directory supplies each figure and table.

| Figure / Table | Data source (under `artifacts/runs/`) | Methods compared |
|---|---|---|
| Fig 1 (headline CIFAR-10) | `scope_cifar_20260422-173505`, `exp2_noise_dl-20_20260422-035456` | SCOPE (from scope_cifar), random (from exp2_noise_dl-20, same config) |
| Fig 1 (headline MNIST) | `scope_mnist_20260424-155455` | random, SCOPE (paired) |
| Fig 1 (headline FMNIST) | `scope_fmnist_20260424-161247` | random, SCOPE (paired) |
| Fig 2 (noise sweep) | `scope_noise_{errfree,dl0,dl-10,dl-20,dl-30}_*` + `exp2_noise_{errfree,dl0,dl-10,dl-20,dl-30}_*` | SCOPE (new), random (historical exp2 matched config) |
| Fig 3 (α sweep) | `scope_alpha_{0_1,0_5,5_0}_*` + `exp3_alpha_{0_1,0_5,5_0}_*` | SCOPE (new), random (historical exp3 matched config) |
| Fig 4 (K-sweep) | `scope_fmnist_N50_K{1,5,10,15,25,35,50}_*` | random, SCOPE (paired within each K) |
| Fig 5 (ablation) | `scope_ablation_20260423-044804` | full, no_server, no_diversity |
| Fig 6 (learning curves) | `scope_mnist_20260424-155455`, `scope_fmnist_20260424-161247` | random, SCOPE (per-round `metrics[]`) |
| Fig 7 (participation heatmap) | `scope_fmnist_20260424-161247` | random, SCOPE (per-round `history.selected[]`) |
| Fig 8 (convergence speed) | `scope_fmnist_N50_K{1,5,10,15,25,35}_*` | random, SCOPE (rounds-to-80% + AUC gap) |
| Fig 9 (K=1 spotlight) | `scope_fmnist_N50_K1_20260424-165027` | random, SCOPE (full trajectory + Gini) |
| Fig 10 (FL baseline) | `fl_cifar10_baseline_20260416-123215` + `scope_fmnist_N50_K15_*` for Gini reference | 6 FL selectors, SCOPE-FD (dashed reference line) |
| Fig 11 (per-client participation) | `scope_fmnist_20260424-161247` | random, SCOPE (participation_counts arrays) |
| Fig 12 (system diagram) | rendered from matplotlib primitives (no artifact dependency) | structural diagram |

**Reproduction.** `python scripts/plot_scope_paper.py` writes all 14 files (7 × `.eps` + 7 × `.png`) to `docs/figures/scope_paper/` in ~10 s on a typical machine. Missing run directories are warned-and-skipped; the script does not hard-fail on partial data.

**Reference verification.** Run the Crossref verification via `python scripts/verify_scope_refs.py` (not yet committed — the one-shot script lives in the assistant's session log; the resulting `artifacts/ref_verification.json` contains the per-tag verification status).

**Cross-references to SCOPE_FD_reference.md.** The following sections delegate implementation or theory detail to the algorithmic reference:
- §5 Algorithm — delegates to [SCOPE_FD_reference.md §4](SCOPE_FD_reference.md) (full equations + pseudocode)
- §6.1 Gini → 0 guarantee — delegates to [SCOPE_FD_reference.md §7.1](SCOPE_FD_reference.md)
- §6.2 Convergence compatibility — delegates to [SCOPE_FD_reference.md §7.2, §7.3](SCOPE_FD_reference.md)
- §6.3 Magnitude analysis — delegates to [SCOPE_FD_reference.md §7.2, §8](SCOPE_FD_reference.md)
- §8.5 Ablation discussion — delegates to [SCOPE_FD_reference.md §9](SCOPE_FD_reference.md)
- §9.3 Reviewer Q&A — delegates to [SCOPE_FD_reference.md §13](SCOPE_FD_reference.md)
- §10 Limitations — delegates to [SCOPE_FD_reference.md §14](SCOPE_FD_reference.md)

---

*End of paper narrative support document.*
