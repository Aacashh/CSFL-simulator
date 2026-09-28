# APEX — Major Revision Plan (IEEE TAI, Round 1)

**Status:** plan only. No revision experiment has been run yet. Everything below
comes from the round-1 reviews ([APEX_R1_reviews.md](APEX_R1_reviews.md)) and a
read of the code that produced the submitted numbers:
`csfl_simulator/selection/ml/apex_v2.py`, `csfl_simulator/core/simulator.py`,
the baseline selectors under `csfl_simulator/selection/`, `presets/methods.yaml`,
`scripts/run_apex_v2_experiments.sh` and
[APEX_v2_experiment_analysis.md](APEX_v2_experiment_analysis.md).

The submitted manuscript is in
`csfl_simulator/Paper Corrections/APEX_R1/submitted/main_v3.tex`. §13 audits it
line by line against the code and the reviews. Equation numbers follow the
source and match the reviewers': Eq. 3 local update, Eq. 5 convergence bound,
Eqs. 6–7 Γ and σ², Eq. 12 contextual utility, Eq. 16 reward, Eq. 21 het,
Eq. 24 final score. Tables I–V and Fig. 2 also match.

Reviewer labels: R1.1–R1.14 follow Reviewer 1's numbering; R2.1–R2.6 are
Reviewer 2's paragraphs in order; AE is the Associate Editor.

---

## 0. Bottom line

Three findings in the code matter more than any single comment. All three must
be fixed before any new number goes into the paper.

1. **Four of the seven baselines are defective implementations.** Oort and
   FedCS pick the same 10 of 50 clients for all 200 rounds. TiFL picks 12 and
   PoC 17. Their Gini values (0.800 / 0.800 / 0.796) match the submitted
   Table II exactly. The paper's gaps against these methods (+4 to +14 pp) and
   its fairness story ("system-aware methods have Gini ≥ 0.80") are mostly
   artifacts of the defects. Reviewer 2 (R2.5) has already noticed that the
   numbers look wrong.
2. **The selector's reward is computed from the test set.** Eq. 16's reward is
   the change in `0.6·acc + 0.2·time + 0.1·fair + 0.1·dp`, where `acc` is the
   test-set accuracy evaluated every round. It is not ΔAcc as the paper says,
   and it uses test data during training. R1.8 names the test-set dependence.
   Oort's implementation reads the same signal too, though its score never uses
   it.
3. **The Thompson-sampling component does not learn.** Rewards are ~10⁻³ per
   round, split by K, then passed through an EMA. The posterior means stay near
   10⁻⁴, while the variance floor makes each sample's noise about 2·10⁻². In a
   planted-signal test, APEX picked the 10 high-value clients 22.5% of the time
   with the reward on and 21.8% with it off (random selection gives 20%). The
   "contextual Thompson sampling" is acting as random jitter.

Together with the Table II/V discrepancy (§1, A7), these mean every table must
be regenerated. The paper is still publishable if the revision does four things.

- **(a) Fix the evaluation.** Use faithful baselines, no test-set leakage,
  5–10 seeds with paired tests, and one canonical results store behind every
  table and every number in the text.
- **(b) Revise APEX so each component has one job.** Each job needs a proof or
  a measurement behind it and an ablation showing it matters. Components that
  fail are deleted (§2, §10).
- **(c) Replace the theory with statements that are actually true.** Two small,
  rigorous propositions that apply to what APEX computes (a label-skew gradient
  bound and submodularity of the per-round objective). The regret bound becomes
  an explicitly heuristic remark (§3).
- **(d) Answer every comment point by point,** each with a pointer to new
  evidence (§8).

The contribution to aim for, if the data supports it: a training-free selector
that needs no gradient uploads and no server test data. It would carry a
per-round (1−1/e) guarantee on a surrogate that provably bounds cohort gradient
bias under label skew. It would tie the best baseline in benign regimes and
improve accuracy and worst-class recall where selection matters (extreme label
skew, low participation, drift), at a measured cost of milliseconds per round.
Whether it *wins* in those regimes is an empirical question. §10 says what to do
if it doesn't.

---

## 1. Code audit — what the submitted numbers actually measured

Reproduce A1–A3, A6, A8 and A10 with `python scripts/apex_selection_audit.py`.
It runs selection only, with no training, and takes about a minute on a CPU.

| ID | Finding | Evidence | Consequence | Comments |
|---|---|---|---|---|
| A1 | **Oort and FedCS lock onto the first cohort forever.** Never-selected clients keep `last_loss = 0`, so their score is 0. Oort's UCB bonus applies only to clients already explored (`n_i > 0`) and it has no ε-exploration. FedCS with `time_budget=None` falls back to top-K by stale loss. | `system_aware/oort.py`, `system_aware/fedcs.py`; audit: 10/50 clients ever selected, Gini 0.800 | Table II/III/IV rows for these methods are invalid | R2.5, R1.6 |
| A2 | **TiFL takes the lowest client IDs in each tier, every round.** Tier lists are rebuilt and `pop(0)`'d each call. The published TiFL samples uniformly *within* the chosen tier. | `system_aware/tifl.py`; audit: 12/50 clients, Gini 0.796 | invalid | R2.5 |
| A3 | **PoC isn't Power-of-Choice.** Candidates are drawn uniformly instead of ∝ data size. It ranks by *stale* training loss (0 for unseen clients) mixed with 0.3·speed + 0.2·recency. The paper's algorithm queries each candidate's *current* loss on the global model. | `system_aware/poc.py`; audit: 17/50 clients | invalid | R2.5 |
| A4 | **"FedCor" is a histogram-cosine heuristic, not the GP-based method.** | `ml/fedcor_approx.py` docstring: "FedCor-inspired … (approximation)" | Either port it or rename it | R2.5, R1.6 |
| A5 | **Reward = Δ composite score on the *test* set.** APEX's posterior is updated from it. Oort's implementation also reads it into a utility table but never uses that table in its score. | `core/simulator.py`: `eval_model(self.model, self.test_loader, …)`, then composite, then `last_reward` | Test-set leakage; paper's Eq. 16 doesn't match the code | R1.8, R1.11 |
| A6 | **The Thompson posterior can't move selection.** Credit ≈ 10⁻³/K. The variance floor 0.1/√n gives sample std ≈ 0.02 at n = 40. The blend is 0.7·context + 0.3·sample, with context in [0, 1]. For n < 2 it falls back to Beta(1, 1) ≈ U(0, 1). | audit: good-client share 0.225 (signal on) vs 0.218 (signal off) | The TS "learning" claim is unsupported; the TS term is noise injection | R1.8, R1.11, R1.4 |
| A7 | **Table II and Table V "full APEX" are two separate executions of one configuration.** The main benchmark ran as job `main_cifar10_a03_s*`, the ablation as `ablation_s*`. GPU execution isn't bit-reproducible: client parallelism auto-sizes CUDA streams from free VRAM (`parallel_clients=-1`), and `torch.use_deterministic_algorithms(True, warn_only=True)` only warns, with no `CUBLAS_WORKSPACE_CONFIG`. **Decisive:** at N=50, K=10, `apex_v2_no_adaptive_recency` is *code-identical* to full APEX, because max(N/K, 3) = 5.0 equals the fixed constant it replaces. It ran in the *same* job, with the same seeds, as the full model, yet Table V reports 0.7115 vs 0.6942. Het scaling multiplies the diversity weight by 0.986 at α=0.3 (see A10), so "w/o het scaling" is a near-replicate too, at 0.7143. Four near-replicates of one algorithm span 0.694–0.714. | `scripts/run_apex_v2_experiments.sh` (Exp 1 vs Exp 4); `presets/methods.yaml`; `apex_v2.py` (`C_rec`); `core/parallel.py` | The Table II/V gap **and** the Table V "improvements" are within the 3-seed replicate noise (~2 pp) | R1.2, R1.4 |
| A8 | **The "3.9 ms" overhead holds only at N=50.** The paper's O(N·K²·L) (Table I, §V-D) is correct, but it grows fast in practice. Measured (CPU, K = N/10): 0.9 ms at N=50, 27 ms at N=200, 376 ms at N=500, 3.1 s at N=1000. The abstract and Discussion present 3.9 ms as the method's overhead without a scale qualifier. | audit | A runtime table across N and methods is needed; the headline number must carry its N | R2.4 |
| A9 | **The phase detector's input depends on who was selected.** It uses the mean training loss of the last cohort. Choosing high-loss or diverse clients moves the signal, which moves the phase, which changes the cohort: a feedback loop. This is a plausible cause of the α=0.1 oscillation that hysteresis was added to suppress. | `apex_v2.py` step 3 | Needs a selection-independent signal | R1.4 (hysteresis) |
| A10 | **The heterogeneity scalar saturates.** It's a sampled mean of √JSD divided by an ad hoc 0.6 (the true maximum is √ln2 ≈ 0.833), computed once. Measured: 1.000 at α=0.1, 0.986 at α=0.3, 0.807 at α=0.6, 0.079 IID. **So het scaling is effectively off for α ≤ 0.3.** | `apex_v2.py::_estimate_heterogeneity`; check in §2.3(b) | Explains why ablating it at α=0.3 changes nothing systematic | R1.4, R1.13 |
| A11 | **The diversity vector concatenates [normalised loss, normalised grad-norm, histogram].** Any theory about histograms says nothing about the other two coordinates. Max–min cosine greedy has no approximation guarantee. | `apex_v2.py::_build_proxy` | Theory/implementation gap | R1.10 |
| A12 | Smaller reporting issues: final-round single-point accuracy with 3 seeds; MMR-Diverse has *higher* macro-F1 than APEX at the main setting (0.7005 vs 0.6948); the IID mean is lifted by one seed (s123 = 81.0%; without it APEX trails FedAvg); APEX's IID Gini is 0.60; the analysis doc lists N=200 with 2 of 3 seeds complete. Check what the paper actually reported. | [APEX_v2_experiment_analysis.md](APEX_v2_experiment_analysis.md) §3, §8, §5 | Must be stated honestly | R1.1, R1.9, R1.14 |

The audit output this plan relies on (seeded, reproducible):

```
== Lock-in check (N=50, K=10, T=200) ==
baseline.fedavg          unique clients ever selected= 50/50  Gini=0.073
system_aware.oort        unique clients ever selected= 10/50  Gini=0.800
system_aware.fedcs       unique clients ever selected= 10/50  Gini=0.800
system_aware.tifl        unique clients ever selected= 12/50  Gini=0.796
system_aware.poc         unique clients ever selected= 17/50  Gini=0.730
ml.apex_v2               unique clients ever selected= 50/50  Gini=0.156

== APEX v2 Thompson signal check: 10 planted 'good' clients raise cohort reward ==
reward signal=ON : share of good clients in rounds 100-200 = 0.225 (random = 0.200); posterior mean |mu| ~ 8.27e-05
reward signal=OFF: share of good clients in rounds 100-200 = 0.218 (random = 0.200); posterior mean |mu| ~ 7.22e-05
TS sample std at n=40 with variance floor 0.1/sqrt(n): 0.020  (contextual score range is [0,1])

== APEX v2 selection time vs N (K = N/10) ==
N=   50 K=   5: steady      0.9 ms
N=  200 K=  20: steady     26.9 ms
N=  500 K=  50: steady    376.4 ms
N= 1000 K= 100: steady   3089.9 ms
```

**The submitted run data isn't available, and the plan doesn't need it.**
Every table is regenerated anyway. The explanation for R1.2 rests on the code
alone: the code-identical variant, the ≈×1 het scaling, and the
non-deterministic execution path. The letter should state exactly that: *"the
original run directories were not retained; the explanation follows from the
code, and the revised protocol makes the discrepancy structurally
impossible"*. E0c becomes optional; do it only if the data turns up.

---

## 2. Revised APEX architecture

### 2.1 Rules the revision follows

1. Every component does one job, and the job is measurable (§6, E9/E10).
2. Every component can be switched off, and is kept only if the survival rule
   in §10 says so.
3. No selector, APEX or baseline, ever sees test-set data.
4. APEX stays a *selection* method. Aggregation remains FedAvg with weights
   n_i/n_S, so any gain is attributable to selection.
5. Every hyperparameter is fixed on development seeds and then frozen (§5.1).

### 2.2 Component map

| Component | Submitted (v2) | Revised | Why | Comments |
|---|---|---|---|---|
| Progress signal (reward and phase input) | Δ composite score on the **test set**, every round | Loss on a server **validation split** (2% of *train*, carved out before partitioning, identical for all methods). **Server-data-free variant:** a class-reweighted cohort loss (§2.3c) | No leakage; works without server data | R1.8, A5, A9 |
| Credit assignment | Equal split r/\|S\| | **First-order attribution** of the validation-loss decrease to each client update (§2.3d). Data-weighted equal split is the zeroth-order special case and the fallback under secure aggregation | Separates helpful from harmful updates; attributions sum exactly to the observed improvement to first order | R1.8 |
| Reward scale | Raw (~10⁻³), with context in [0, 1] → TS inert | Credits standardised by a running scale; context score z-scored across clients each round | Makes the posterior able to move selection | A6 |
| Posterior | Welford mean/variance + EMA (α_e) + floor 0.1/√n + Beta fallback + blend γ·n/(n+1) | Gaussian posterior **with the contextual score as prior mean** (strength κ) and **discounted** sufficient statistics (factor δ) | v2's confidence rule n/(n+1) is the Bayes shrinkage weight for κ=1 (v2 also capped it at γ = 0.3). Discounting bounds the effective sample size, replacing the variance floor and EMA with a standard non-stationary-bandit device. Five knobs become two | R1.4 (Fixes 4, 5), R1.11 |
| Diversity | Greedy max–min cosine on [loss, grad-norm, histogram] | Greedy **facility location** on 1 − TV(p_i, p_j) over label histograms only | Monotone submodular → (1−1/e) guarantee (Prop. 2). Connects to DivFL's gradient objective via Prop. 1 | R1.10, A11 |
| Heterogeneity index | Sampled √JSD / 0.6, cached at round 1 (≈1 for α ≤ 0.3) | **H = I(Z;Y)/H(Y)**, exact, O(N·L), recomputed each round (Z = client index of a random sample) | Prop. 1(iv): the gradient-dissimilarity term in non-convex FedAvg bounds is ≤ 2G²·I(Z;Y). Measured: 0.65 / 0.38 / 0.24 / 0.00 for α = 0.1 / 0.3 / 0.6 / IID | R1.4, R1.13, A10 |
| Histograms | Cached at round 1 | **Refreshed** whenever a client participates (L integers sent with its update); optional Laplace noise | Handles drift | R1.13 |
| Phase detector | 3 phases + hysteresis (dwell 3, no skipping), input = cohort training loss | Same logic and hysteresis, input = the selection-independent progress signal | Breaks the feedback loop (A9). Hysteresis was the one component the old ablation supported | R1.4 |
| Recency | a/(a + N/K) | Unchanged | Modular term, so the guarantee is untouched | — |
| Phase weights | 3 triples (9 numbers) | β_r constant; β_d(φ) ∈ {critical, transition, exploit}; β_q = 1 − β_d − β_r (4 numbers) | Fewer hand-set values | R1.4, sensitivity |
| Complexity | Claimed O(NL + K²L), actual O(NK²L) | O(K·N·M) greedy, with M = N exact or M ≤ 256 sampled demand points for large N, plus O(K·N·L) similarity refresh and O(K·d) credit; vectorised NumPy | Measured, not asserted (E8) | R2.4, A8 |

### 2.3 Component specifications

**(a) Server-side state and what clients send.** At each participation, client
i sends its model update Δ_i (already required), its label histogram h_i
(L integers), and its training loss and gradient norm (already sent). In the
server-data-free variant it also sends the per-class mean loss of the
*received* global model on its local data (L floats, one inference pass; PoC
needs a similar pass). The server keeps, per client: p_i = h_i/n_i, n_i, rounds
since last selection a_i, context features x_i, and discounted statistics
(N_i, Y_i).

**(b) Heterogeneity index.**
H = I(Z;Y)/H(Y) = Σ_i (n_i/n)·KL(p_i ‖ p̄) / H(p̄) ∈ [0, 1], where
p̄ = Σ_i n_i p_i / n. H = 0 when all clients share the global label
distribution. H = 1 when every client holds a single class. It's exact, costs
O(N·L), and is recomputed every round, so refreshed histograms (R1.13) move it
automatically. The diversity weight becomes H·β_d(φ); the removed mass
(1 − H)·β_d goes to β_q, as in v2. Values from the audit script (50 Dirichlet
clients, 10 classes, mean of 3 draws):

| α | 0.1 | 0.3 | 0.6 | 1.0 | IID |
|---|---|---|---|---|---|
| Submitted scalar (√JSD/0.6, capped) | 1.000 | 0.986 | 0.807 | 0.687 | 0.079 |
| H = I(Z;Y)/H(Y) | 0.652 | 0.376 | 0.237 | 0.157 | 0.002 |

H is much less saturated, so it actually changes behaviour across α. Whether a
linear map H·β_d is the right strength is checked on dev seeds, with a
sensitivity panel in E12.

**(c) Progress signal.**
- *Validation variant (default):* L_V(w_t) is computed on the held-out split V
  after each aggregation, one forward pass over |V| ≈ 1000 samples.
- *Server-data-free variant:* each selected client reports class-mean losses
  ℓ_{i,c}(w_t) of the received model. The server pools them per class across
  the cohort with class-count weights, giving ℓ̂_c. It then forms
  L̂(w_t) = Σ_c p̄(c)·ℓ̂_c; classes the cohort doesn't cover carry their last
  estimate forward. Under assumption A1 (§3.1), E[ℓ_{i,c}(w)] = ℓ_c(w) for every
  client, so L̂ estimates the global loss F(w_t) *whichever clients were
  selected*. It is selection-debiased by construction, from the same model that
  justifies the diversity term. The reward for cohort S_t arrives one round
  late, which bandits handle without modification.

Both variants appear in the main table (E1) and the reward study (E6). Pick the
default at gate G2 (§10); if they tie, lead with the data-free variant because
it answers R1.8's practicality concern directly.

**(d) Credit assignment.**
Define ρ_t = L(w_t) − L(w_{t+1}), the loss decrease produced by round t's
cohort. The first-order attribution is:

  c_i(t) = −(n_i/n_S)·⟨∇L_V(w_t), Δ_i⟩,  with  Σ_{i∈S_t} c_i(t) = −⟨∇L_V(w_t), Δ̄_t⟩ ≈ ρ_t.

This costs one server backward pass on V plus K inner products of dimension d.
- *Data-free variant:* replace −∇L_V by the aggregate direction and scale so
  the shares sum exactly to ρ_t:
  c_i = ρ_t·(n_i/n_S)·⟨Δ_i, Δ̄⟩/‖Δ̄‖². Since Σ_i (n_i/n_S)⟨Δ_i, Δ̄⟩ = ‖Δ̄‖²,
  this is an exact decomposition (the efficiency property of attribution).
- *Fallback:* when individual updates are hidden (secure aggregation),
  c_i = (n_i/n_S)·ρ_t. This is a data-weighted version of the submitted equal
  split.
- *Standardisation:* y_i(t) = c_i(t)/s_t, where s_t is the RMS of credits over
  the last 20 rounds. Loss decrements shrink by orders of magnitude during
  training; without rescaling, early rounds dominate the posterior forever.
- *What to tell R1.8 about equal splits:* under an additive-contribution model
  with co-selection independent of client identity,
  E[ρ_t/K | i ∈ S_t] = θ_i/K + const, which is rank-preserving. So equal
  splitting isn't wrong, but it's noisy (signal scaled by 1/K), and its bias
  becomes identity-dependent as soon as selection is non-random, as it is under
  APEX. That is the motivation for (d).
- *Risk to watch:* minority-class clients may have updates less aligned with
  the aggregate and receive lower credit, working against the diversity term.
  E6 compares the attribution rules, and E10 logs credit by class coverage.

**(e) Posterior** (per client, on the standardised scale).
- Prior: θ_i ~ N(m_i(t), 1/κ), where m_i(t) is the z-score across clients of
  the contextual score λᵀx_i(t). The features x_i are normalised loss,
  grad-norm, speed and data size; λ = (λ_ℓ, λ_g, λ_s, λ_d).
- Each round, for every client: N_i ← δ·N_i + 1[i∈S_t],
  Y_i ← δ·Y_i + y_i·1[i∈S_t].
- Posterior: μ_i = (κ·m_i + Y_i)/(κ + N_i), v_i = 1/(κ + N_i).
- Sample: θ̃_i ~ N(μ_i, v_i).

With δ = 1 and κ = 1, the weight on data is N_i/(N_i + 1), which is v2's
"confidence-aware γ" rule (Fix 5) without v2's extra cap at γ = 0.3. With δ < 1, N_i ≤ 1/(1−δ), so v_i stays above
1/(κ + 1/(1−δ)). This replaces the ad hoc variance floor and EMA (Fix 4) with
discounted Bayesian updating, the standard device for non-stationary bandits.
The hyperparameters are κ and δ, replacing γ, the confidence rule, α_e, c_f and
the Beta fallback.

**(f) Diversity.**
FL_t(S) = Σ_{k=1}^{N} ω_k · max_{i∈S} s_{ki}, with s_{ki} = 1 − TV(p_k, p_i) and
ω_k = n_k/n. The greedy gain of candidate i is
Σ_k ω_k·[s_{ki} − cur_k]₊, where cur_k is demand point k's current best
similarity. Rows of s are recomputed only for clients whose histogram changed.
For N > 256, use a fixed random subset of M = 256 demand points (still an FL
function, so still submodular).

Also log TV(p̄_S, p̄), the cohort-mixture mismatch in Prop. 1(iii), as a
diagnostic every round for every method. It isn't submodular, so it's measured
rather than optimised.

**(g) Recency.** r_i = a_i/(a_i + N/K), unchanged.

**(h) Phase detector.** v2's rule: over window W, relative improvement rate
and coefficient of variation against thresholds τ_c, τ_u, τ_e, then hysteresis
(minimum dwell, no critical↔exploitation jump). The input becomes the signal
from (c). The ablation must include a **fixed time-based schedule** with the
same average weights. If that schedule matches the detector, the "phase-aware"
claim has to be weakened to "scheduled", and the paper says so.

**(i) Per-round objective and selection.**

  F_t(S) = Σ_{i∈S} [β_q(φ)·q_i + β_r·r_i] + H·β_d(φ)·FL_t(S),  q_i = θ̃_i − min_j θ̃_j ≥ 0.

Plus the (1 − H)·β_d(φ) mass moved into β_q. Shifting q by a constant doesn't
change the maximiser when |S| = K. Select greedily for K steps (lazy greedy for
speed). With a time budget the constraint is a knapsack: use cost-benefit
greedy plus a best-singleton check (a (1−1/e)/2 guarantee).

**(j) Cost per round.**

| Step | Cost |
|---|---|
| Histogram refresh | O(K·L) |
| Similarity rows | O(K·N·L) |
| H | O(N·L) |
| Posterior | O(N) |
| Greedy | O(K·N·M) |
| Credit | O(K·d) |
| Validation pass (validation variant only) | one server forward+backward on V |

Target under 10 ms of selection time at N = 1000, K = 100 on one CPU core,
vectorised. E8 measures it; the paper reports the measurement, not the target.

### 2.4 Pseudocode (for the paper's Algorithm 1)

```
Input: N clients, K, T; hyperparameters (W, τ_c, τ_u, τ_e, dwell, κ, δ, λ, β_d(·), β_r)
for t = 0 … T−1:
    refresh p_i for clients that reported last round; update s rows; H ← I(Z;Y)/H(Y)
    φ ← PhaseDetector(progress-signal history)                      # with hysteresis
    for each client: decay (N_i, Y_i) by δ; m_i ← z-score(λᵀx_i); sample θ̃_i ~ N(μ_i, v_i)
    S_t ← Greedy_K( Σ_{i∈S}[β_q(φ,H) q_i + β_r r_i] + H β_d(φ) FL_t(S) )
    clients in S_t train locally; server aggregates with FedAvg
    progress ρ_t from validation loss (or L̂); credits c_i → y_i; update (N_i, Y_i) for i ∈ S_t
```

### 2.5 Hyperparameters

| | Submitted | Revised |
|---|---|---|
| Phase detector | W, τ_c, τ_u, τ_e, δ_min (5) | same (5; rename the dwell time to D_min, since δ is now the discount) |
| Blend / posterior | γ, α_e, c_f, confidence rule, Beta prior | κ, δ (2) |
| Context weights | w_l, w_g, w_s, w_q (4) | λ_ℓ, λ_g, λ_s, λ_d (4, on the simplex → 3 free) |
| Phase weights | 3 triples (9) | β_d × 3 + β_r (4) |
| Other | recency rule, het normaliser 0.6 | \|V\| (validation variant only) |
| **Total hand-set** | ~20 | ~14 |

All values are frozen on development seeds and are identical across every
dataset and setting. The paper then says "no gradient-trained parameters; 14
fixed hyperparameters, identical across all experiments; sensitivity in
Fig. X", instead of "zero trainable parameters".

---

## 3. Theory — what is proven, what isn't

### 3.1 Assumptions (stated in the System Model section)

- **A1 (label skew).** Client i's data distribution is D_i(x, y) = p_i(y)·Q(x|y),
  with Q shared by all clients. Dirichlet label partitions of a common dataset
  satisfy this by construction (in expectation over the partition draw).
  Feature-skew data (for example writer-partitioned handwriting) doesn't.
- **A2 (bounded class gradients).** Let ℓ_c(w) = E_{x~Q(·|c)} ℓ(w; x, c).
  Then ‖∇ℓ_c(w)‖ ≤ G for all c and w.
- **A3 (convergence context only).** F is L-smooth; stochastic gradients have
  variance ≤ σ². These are the standard non-convex FedAvg assumptions.

### 3.2 Proposition 1 — label-skew gradient geometry (full proof in appendix)

Under A1–A2, for all w:

1. ∇F_i(w) = Σ_c p_i(c)·∇ℓ_c(w).
2. ‖∇F_i(w) − ∇F_j(w)‖ ≤ 2G·TV(p_i, p_j).
3. For any cohort S aggregated with FedAvg weights,
   ‖Σ_{i∈S} (n_i/n_S)·∇F_i(w) − ∇F(w)‖ ≤ 2G·TV(p̄_S, p̄).
4. Σ_i (n_i/n)·‖∇F_i(w) − ∇F(w)‖² ≤ 2G²·I(Z;Y) = 2G²·H(Y)·H.

*Proof sketch.*
- (1) is linearity of expectation over y, then over x|y.
- (2): the difference equals Σ_c (p_i(c) − p_j(c))·∇ℓ_c. Apply the triangle
  inequality and A2, then use Σ_c |p_i(c) − p_j(c)| = 2·TV.
- (3): by (1), the FedAvg-weighted cohort gradient equals Σ_c p̄_S(c)·∇ℓ_c;
  repeat the argument of (2).
- (4): apply the (2)-style bound against p̄, then Pinsker (TV² ≤ KL/2), then
  I(Z;Y) = Σ_i (n_i/n)·KL(p_i ‖ p̄).

*What it answers:*
- (2) is the missing link in R1.10: label-histogram distance *upper-bounds*
  gradient dissimilarity. It's TV, not cosine, which is why the revised
  diversity term uses TV.
- (3) bounds the per-round bias of the selected cohort's gradient.
- (4) justifies H: it bounds the gradient-dissimilarity constant (σ_G²) that
  appears in non-convex partial-participation FedAvg bounds.

### 3.3 Corollary 1 — the DivFL bridge

Assign each client k to its most similar selected client σ(k) ∈ S. Then:

  Σ_k ω_k·‖∇F_k − ∇F_{σ(k)}‖ ≤ 2G·(1 − FL(S)).

The left side is DivFL's gradient-approximation objective. DivFL minimises it
with gradients; APEX maximises FL(S) with histograms only. State the direction
carefully: APEX maximises, within (1−1/e), an objective whose *complement
upper-bounds* DivFL's error. It does not attain DivFL's guarantee on gradients,
and the bound holds only under A1. This is exactly the connection R1.10 says is
missing.

### 3.4 Proposition 2 — the per-round selection guarantee

For any phase φ, weights β ≥ 0, H ∈ [0, 1] and any realisation of the samples
θ̃, F_t is monotone submodular. The greedy cohort therefore satisfies
F_t(S_g) ≥ (1 − 1/e)·max_{|S|=K} F_t(S) (Nemhauser, Wolsey & Fisher, 1978).

*Proof:* a non-negative modular term plus a non-negative multiple of a
facility-location function (monotone submodular) is monotone submodular.

State plainly that the submitted max–min greedy had no such guarantee, which is
part of why it was replaced.

### 3.5 Convergence context (replaces the strongly convex Eq. 5)

Replace Eq. 5 with a **non-convex** partial-participation FedAvg bound under A3
and a bounded-dissimilarity assumption (e.g. Yang, Fang & Liu, ICLR 2021). In
that bound the selection-relevant quantities are the dissimilarity σ_G², which
Prop. 1(4) bounds by 2G²·I(Z;Y), and, for biased selection, a participation or
selection-skew term (Cho et al., AISTATS 2022; Wang & Ji, NeurIPS 2022). Prop.
1(3) bounds that term per round via TV(p̄_S, p̄), which E9 measures.

Add one explicit sentence for R1.12: *the bound is design motivation; ResNet-18
with ReLU and BatchNorm is not globally L-smooth, and no claim is made that the
bound holds for the trained models.* Define every symbol immediately after the
equation (R2.1).

### 3.6 Regret — now a remark, not a proposition

For the bandit module alone, under a stationary top-K **semi-bandit** model
(each selected client's standardised credit is a noisy observation of a fixed
θ_i), combinatorial Thompson sampling has O(Σ log T / Δ) problem-dependent
regret (Wang & Chen, ICML 2018). That replaces the "K independent pulls"
decomposition. Full APEX doesn't inherit it, for three reasons, and the paper
lists them:

1. θ_i drifts as the model trains. Discounting is used *because* of this;
   discounted and sliding-window guarantees (Garivier & Moulines, 2011) are for
   piecewise-stationary means and UCB-style rules.
2. Credits are coupled through aggregation.
3. Selection maximises TS scores plus diversity and recency, not TS alone.

Label it "Remark (design reference, not a guarantee)", as R1.11 asks. Back it
with E10's diagnostics.

### 3.7 Where the theory stops, and the empirical checks that cover the gap

| Gap | Check |
|---|---|
| A1 holds only in expectation and at population level; real within-class features differ across clients | E9: Spearman correlation between TV(p_i, p_j) and measured ‖∇F_i − ∇F_j‖ at rounds {1, 25, 50, 100, 200} for α ∈ {0.1, 0.3, 0.6}; fitted G; share of pairs violating the bound |
| Prop. 1(3) is a bound, not an achieved reduction | E9: measured cohort bias ‖g_S − ∇F‖ and TV(p̄_S, p̄) for every method |
| Feature skew breaks A1 | E4 optional feature-skew dataset; report honestly whatever happens |
| Does the bandit learn anything? | E10: correlation of posterior means with held-out credits; counterfactual share of selections that change if θ̃ is replaced by the prior mean; E5 "TS → matched-variance jitter" variant |

---

## 4. Baselines — faithful, tuned equally, consistent everywhere

Every baseline gets a fidelity note in its docstring and a row in an appendix
table: the published algorithm's essentials, our implementation, the
hyperparameters taken from the paper, and any deviation.

| Method | Published algorithm (essentials) | Current code | Revision |
|---|---|---|---|
| FedAvg (McMahan et al., 2017) | Uniform random K of N **without replacement every round**; weights n_i/n_S | Correct (`baseline/fedavg.py`) | Keep; state it explicitly for R2.5 |
| PoC (Cho, Wang & Joshi, AISTATS 2022) | Sample d candidates without replacement ∝ data fraction; each computes its **current** local loss on w_t; pick the top-K by loss | A3 | Re-implement pow-d with fresh loss queries; count the candidates' forward passes as client overhead; tune d ∈ {2K, 3K, 5K} |
| Oort (Lai et al., OSDI 2021) | Statistical utility \|B_i\|·√(mean loss²) + staleness bonus; system penalty (T/t_i)^α for slow clients; ε-greedy exploration (0.9 → 0.2, ×0.98) of unexplored clients; cut-off sampling among the top; pacer | A1 | Port from FedScale's official implementation; tune ε-decay and T |
| FedCS (Nishio & Yonetani, ICC 2019) | Random resource-request pool, then greedy admission of clients that fit the round deadline (maximising count) | A1 | Faithful; with a non-binding deadline it reduces to random selection, so say so; its natural regime is E11 |
| TiFL (Chai et al., HPDC 2020) | Latency tiers; choose a tier (static or adaptive credits), then sample **uniformly within the tier** | A2 | Faithful static + adaptive; adaptive tier accuracy uses the validation split, never the test set |
| FedCor (Tang et al., CVPR 2022) | GP over client loss changes with a learned correlation kernel; warm-up; greedy on posterior expected loss reduction | A4 heuristic | Port the official code. If that's infeasible, rename the row "correlation-aware heuristic (FedCor-inspired)" |
| MMR-Diverse (Carbonara, Drioli & Foresti, ICASSP 2024) | Loss-sorted candidate pool, re-ranked by MMR over cosine similarity of client gradient proxies | `heuristic/mmr_diverse.py` scores 0.4·loss + 0.3·grad-norm + 0.15·speed + 0.15·channel, damps by 1/(1+participation), uses label-histogram embeddings; fidelity to the ICASSP paper unchecked | Check against the paper and align, or document every deviation. Keep it, because it's the strongest baseline |
| **DivFL (Balakrishnan et al., ICLR 2022)** — new | Greedy facility location on client gradients/updates (stale-update practical variant) | Only an FD port exists (`fd_native/divfl_fd.py`) | Add as the gradient-based counterpart APEX's theory compares with; report its gradient-upload cost |
| **FedAEB** — new (R2.6) | The learning-based selector already discussed in the manuscript's related work | Missing | Implement from the paper; report trainable parameters and training time next to APEX's zero |
| Fed-CBS — optional, recommended | Class-imbalance-reducing selection from label statistics | Missing | The closest histogram-based competitor; a next-round reviewer is likely to ask for it |
| CriticalFL — optional | Critical-period-aware participation (cited as the phase idea's precedent) | In-simulator reproduction exists: `experiments/maml_select/criticalfl.py` | Add if compute allows; it tests the phase claim directly |

**Tuning protocol (fairness).** Every method, APEX included, uses its published
defaults plus a grid of at most 4 values on its single most important knob.
Tune on development seeds at S1 and at α=0.1, then freeze. The appendix lists
every grid.

**Fidelity unit tests** (`tests/`, CPU-only, run in CI):
- Oort visits ≥ 90% of clients within its exploration horizon.
- PoC samples candidates ∝ n_i (χ² test) and queries fresh losses.
- TiFL samples uniformly within tiers.
- No selector's inputs contain test-set metrics (assert on the history keys).
- The submodularity of APEX's F_t is checked numerically on random instances.
- Same seed twice on CPU gives an identical selection sequence.

---

## 5. Experimental protocol

### 5.1 Seeds and freezing

- **Development seeds** {1001, 1002, 1003}: all design decisions, tuning and
  pilots.
- **Evaluation seeds**: 0–9 at S1, 0–4 elsewhere.
- Before the evaluation sweep, commit `configs/apex_revision_frozen.yaml`
  (every hyperparameter of APEX and every baseline). Cite its commit hash in
  the appendix. This rules out tuning on the reported seeds, and says so.

### 5.2 Pairing, determinism, one canonical store

- **Pairing.** For each seed, the partition, validation split, model
  initialisation, client system profiles and data order are identical for all
  methods. The shared-simulator design already pairs partition and init;
  data-loader generators still need seeding per (client, round).
- **Determinism.** Set `parallel_clients=0` (or a fixed stream count),
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`
  without `warn_only`, cuDNN benchmark off, and AMP off.
- **Run metadata.** Every run records the git commit, config hash, GPU model,
  driver, CUDA/cuDNN/torch versions and host.
- **Canonical store.** One result file per cell:
  `results/apex_revision/<setting>/<method>/<seed>.json`. Every table, figure
  and number in the text is generated from it. The ablation's "full APEX" row
  *is* the main table's APEX cells, the same files, so R1.2 can't recur.
- **Replicate noise.** E0b measures it; report it in the appendix as the
  resolution below which differences aren't interpreted.

### 5.3 Data

A stratified 2% validation split is carved from the training set *before*
partitioning (CIFAR-10: 1000 images). It's removed from client data for all
methods. The test set is used only for reporting.

### 5.4 Metrics

- **Primary: final accuracy** = mean test accuracy over the **last 10 rounds**.
  This replaces single-round final accuracy, which is noisy. Peak accuracy is
  secondary and flagged as optimistic, because it is selected on test data.
- **Class-level:** macro-F1 over the same window (the code already uses
  macro), **worst-class recall**, and the standard deviation of per-class
  recall. These answer R1.9 and are the metrics where coverage-aware selection
  should matter.
- **Speed:** rounds to reach 90% and 95% of FedAvg's final accuracy in the same
  setting (relative targets, so thresholds can't be cherry-picked); area under
  the accuracy curve.
- **Participation:** Gini, Jain's index, number of never-selected clients.
- **Mechanism:** TV(p̄_{S_t}, p̄) per round; H; the phase trace.
- **Cost:** selection time per round (mean, median, p95); client-side extra
  compute and communication.

### 5.5 Statistics

- **Within a setting:** a paired t-test of APEX against each baseline over
  seeds, with Holm correction across baselines. Report the mean paired
  difference with a 95% CI. At S1 (n=10) also report Wilcoxon signed-rank.
  With n = 5, Wilcoxon can't get below p = 0.0625, so use t-tests with
  bootstrap CIs there.
- **Across settings:** a Friedman test on per-setting mean ranks over an
  *enumerated* list of distinct settings (each (dataset, α, N, K) exactly
  once), plus a Nemenyi critical-difference diagram. Report a
  win/tie/loss table: significantly better, tied, or significantly worse than
  each baseline.
- **Language rules** (enforced by the claim audit, §9):
  - "Outperforms" requires a Holm-adjusted p < 0.05 *and* a CI that excludes 0.
  - Otherwise write "comparable" or "statistically tied".
  - "Ranked first" means "highest mean" and is only said with the test result.
  - Never say "significantly" without a test.

---

## 6. Experiment matrix

**Setting codes:**
- S1 = CIFAR-10, ResNet-18, Dirichlet α=0.3, N=50, K=10, T=200.
- "All" = the 11-method set: FedAvg, PoC, Oort, FedCS, TiFL, FedCor, MMR,
  DivFL, FedAEB, APEX (validation), APEX (data-free), with Fed-CBS if added.

| ID | Purpose | Settings | Methods | Seeds | ≈Runs | Answers | Priority |
|---|---|---|---|---|---|---|---|
| E0a | Baseline fidelity tests + selection audit | CPU only | all selectors | — | 0 | R2.5, R1.6 | P0 |
| E0b | Replicate-noise floor | S1, each (method, seed) run 3 times | APEX, FedAvg | 3 | 18 | R1.2 | P0 |
| E0c | Forensics on the submitted runs, *optional* (the data isn't available) | — | — | — | 0 | R1.2 | — |
| E1 | Main benchmark | S1 | All | 10 | 110 | R1.1–3, R1.9, R2.3, R2.5–6 | P0 |
| E2 | Heterogeneity | α ∈ {0.1, 0.6}, IID, 2-shards-per-client; N=50, K=10 | All | 5 | 220 | R1.5 | P0 (α=0.1), P1 |
| E3 | Scale (K stated per row) | α=0.3: (N,K) = (100,10), (200,20), (500,50); plus a low-participation stress test at α=0.1, (200,10) | All | 5 | 220 | R1.7 | P1 |
| E4 | Datasets | CIFAR-100 (ResNet-18), FMNIST (CNN); MNIST in the supplement; optional feature-skew set | All | 5 | 110 (+55) | R1.14, R1.10 boundary | P1 |
| E5 | Ablation (at the harder settings too) | S1; α=0.1 N=50; α=0.3 (200,20) | ~12 variants: full; −TS (prior mean only); TS → matched-variance jitter; −diversity; −recency; −phases (constant weights); fixed-schedule phases; −hysteresis; −H scaling; −histogram refresh; submitted v2 design with validation reward | 5 (10 at S1) | ~200 | R1.4, R1.5, R1.11 | P0 (S1, α=0.1), P1 |
| E6 | Reward and credit | S1, α=0.1 | {validation, data-free} × {data-weighted equal split, first-order attribution} | 5 | 40 | R1.8 | P0 |
| E7 | Label drift | S1 with drift at t=100 (50% of clients' label proportions re-drawn) | APEX static-histogram vs refresh, FedAvg, MMR, best baseline | 5 | 25 | R1.13 | P1 |
| E8 | Runtime on identical hardware | in-loop from E1–E3 + microbenchmark N ∈ {50 … 10⁴}, K ∈ {10, 10%·N}, synthetic states, 20 repetitions | All | — | 0 GPU | R2.4 | P0 |
| E9 | Proxy fidelity and bias diagnostics | α ∈ {0.1, 0.3, 0.6}, checkpoints {1, 25, 50, 100, 200} | APEX, FedAvg, DivFL, MMR | 1 | ~12 | R1.10, R1.12 | P1 |
| E10 | Bandit diagnostics | logged from E1 and E5 | APEX | — | 0 | R1.11 | P1 |
| E11 | System heterogeneity: tiered latency, deadline at the 70th percentile, stragglers dropped; accuracy vs wall-clock | S1 | All | 5 | 55 | R2.5, R1.6 | P2 |
| E12 | Sensitivity (one-at-a-time around the frozen defaults) | S1 | κ, δ, λ_ℓ, β_d(critical), τ_c, dwell, W | 3 | 42 | "20 hyperparameters" pre-emption | P2 |
| E13 | Histogram privacy (Laplace mechanism) | S1, α=0.1; ε ∈ {0.5, 1, 2, ∞} | APEX | 3 | 24 | Impact statement | P2 |

**Compute.** About 1,100 runs in total; the P0 subset (E0, E1, E2 at α=0.1, E5
at S1 and α=0.1, E6, E8) is about 380. At roughly 10 GPU-minutes per 200-round
ResNet-18 CIFAR run, that's about 180 GPU-hours in total and about 65 for P0.
Calibrate with one timed run on the cluster before scheduling. Deterministic
mode is slower. The FMNIST CNN runs are much cheaper.

---

## 7. Code layout

Follow the MAML-Select pattern (`csfl_simulator/experiments/maml_select/`):
an additive suite with shared-core changes kept minimal. `apex_v2.py` stays
untouched as the record of the submitted version.

```
csfl_simulator/experiments/apex_revision/
  README.md          how to reproduce every table and figure
  configs.yaml       settings, method sets, dev/eval seeds; frozen hyperparameters (§5.1)
  apex.py            revised selector (§2)
  baselines/         poc.py oort.py fedcs.py tifl.py fedcor.py divfl.py fedaeb.py [fedcbs.py]
  simulator.py       InstrumentedFLSimulator: validation split, per-class losses, histogram
                     refresh + drift injection, deadline model, per-client update statistics
                     for credit, diagnostics (E9/E10), determinism flags, run metadata
  run_suite.py       (setting, method, seed) → canonical store; resumable; manifest
  stats.py           paired t / Wilcoxon, Holm, bootstrap CI, Friedman + CD, W/T/L
  make_tables.py     LaTeX tables: mean ± std, significance markers, K per row
  make_figures.py
  numbers.py         writes numbers.tex (\newcommand macros) for every value quoted in the text
  microbench.py      E8
  diagnostics.py     E9, E10
  tests/             fidelity, determinism, submodularity, no-test-leak
```

Interface note: selectors currently see only `ClientInfo` and `history`. The
instrumented simulator exposes per-client scalars the credit rule needs
(n_i, ⟨Δ_i, Δ̄⟩, ⟨Δ_i, ∇L_V⟩, ‖Δ_i‖), the progress signal and per-class losses
through `history["round_feedback"]`. Tensors never pass through the selector
API, and `last_reward` is no longer derived from the test set.

---

## 8. Response matrix

"New" marks results that don't exist yet; the letter is written after the runs.
The stance column is the planned position.

| # | Comment | Action | Evidence | Manuscript change | Stance |
|---|---|---|---|---|---|
| AE | Reconcile discrepancies; strengthen statistics and baseline/runtime comparisons; temper theory; notation | §§1–9 | all | all | Summary paragraph listing the four workstreams, with pointers |
| R1.1 | +0.2 pp isn't "significant superiority" | 10 seeds at S1, paired tests + Holm, CIs, last-10-round metric; language rules | E1 | Abstract, Intro, §VI rewritten; "significantly" only with tests | Agree fully. Report ties as ties |
| R1.2 | Table II ≠ Table V | Replicate-noise floor; canonical store; determinism | E0b, A7 | Tables regenerated from one store; the ablation reuses the main-table runs | Explain honestly: two separate executions of the same configuration on non-deterministic GPU kernels. Evidence: a code-identical ablation variant differed by 1.7 pp. Now structurally impossible |
| R1.3 | Fig. 2(b) text vs curves | Correct the sentence; every figure description checked against data | claim audit | §VI | Agree; corrected |
| R1.4 | Ablation undercuts the components; no hard-setting ablation | Ablation at 3 settings incl. α=0.1 and N=200, 5–10 seeds, paired tests; core components ablated too (TS, diversity, phases, fixed schedule); survival rule (§10); components that fail are removed | E5, E10 | New ablation table + text using R1's own precise phrasing for the old result | Agree with the reviewer's restatement. Show the old differences were within noise (A7: an identical variant moved 1.7 pp; het scaling was ≈×1 at α=0.3). Report the new ablation, and which components were removed |
| R1.5 | α=0.3 isn't the most benign | Fix the text | — | §VI | Agree; α=0.6 is the most benign non-IID level |
| R1.6 | "Eight baselines"; inconsistent baseline sets | Correct the count; full baseline set in every core experiment; FedCS/TiFL regime explained + E11 | E1–E4, E11 | §VI setup, all tables | Agree; now consistent |
| R1.7 | K at N=200 | K=20 was used (run script). Tables list K per row; scale study redesigned at fixed 10% participation + a low-participation stress test | E3 | Table IV caption and columns | Clarify and fix |
| R1.8 | Equal split is coarse; test-set reward is impractical | First-order attribution; equal split as ablation and SecAgg fallback; rank-consistency argument; validation-split and server-data-free rewards; no selector touches the test set | E6, E10 | New Eqs. replacing Eq. 16; §IV; limitations | Agree on both. Acknowledge that the submitted reward used test-set evaluation and has been eliminated |
| R1.9 | F1 not discussed | Macro-F1 discussion, including where APEX isn't highest (MMR had higher F1 in the submitted S1); worst-class recall | E1–E4 | §VI paragraph + column | Agree; honest discussion |
| R1.10 | Histogram proxy ↔ gradient diversity not proven | Prop. 1, Corollary 1, Prop. 2 with proofs; assumptions explicit; empirical fidelity incl. where A1 fails | E9, E4 | New §V.A + appendix proofs | Agree the old link was informal. The new statements are rigorous and scoped |
| R1.11 | Regret bound doesn't apply | Demote to a remark; cite the CTS semi-bandit result; list the violated assumptions; diagnostics | E10 | §V.B | Agree; presented as a heuristic reference |
| R1.12 | Convexity gap | Non-convex bound as motivation; explicit gap sentence | — | §III/§V | Agree |
| R1.13 | Static histograms | Refresh on participation; per-round H; drift experiment; tempered impact statement | E7 | §IV, Impact Statement | Agree with the reframing; mechanism added and evaluated |
| R1.14 | "8 of 9" | Enumerated distinct settings; Friedman + CD; W/T/L with significance; phrase removed | E1–E4 | §VI + new figure | Agree; replaced with a formal multi-setting analysis |
| R2.1 | Variables before definitions | Notation table; definitions immediately after each equation | — | New Table (Notation); §III | Agree |
| R2.2 | w overloaded | Rename scalar weights (§9) | — | Throughout | Agree |
| R2.3 | Std in Tables III/IV | Mean ± std (n) in every table | all | all tables | Agree |
| R2.4 | Measured selection time for all methods | Runtime table + scaling plot on one machine; corrected Big-O | E8 | Table I revised + new table/figure | Agree. Report that APEX's Big-O was wrong (A8) and corrected |
| R2.5 | FedAvg sampling; why PoC/Oort are below published results | Confirm FedAvg is uniform without replacement every round; audit found implementation deviations (A1–A4); faithful re-implementations tuned equally; residual gaps explained | E0a, E1, E11 | §VI setup + appendix fidelity table | Thank the reviewer: the comment led to an audit that found the defects. State them plainly, show corrected results, and explain the remaining gaps (see draft below) |
| R2.6 | Compare with FedAEB | Implemented and compared; parameter count and training cost reported | E1–E4 | Tables + §II | Agree |

### Draft paragraphs for the three hardest replies

Finalise wording and numbers after the runs.

**R1.2.** *"We thank the reviewer for spotting this. Table II and Table V
reported two separate executions of the same configuration and seeds: the
main-benchmark job and the ablation job. Our GPU pipeline was not
bit-reproducible (auto-sized multi-stream client training and non-deterministic
cuBLAS kernels), so same-seed trajectories diverged. The size of this effect is
visible in the submitted Table V itself. At N=50 and K=10, the 'w/o adaptive
recency' variant is computationally identical to full APEX, because the
adaptive constant max(N/K, 3) equals the fixed value 5 it replaces. It ran in
the same job with the same seeds, yet it reported 0.712 against 0.694. In the revision, every table is generated from a
single store of runs: the ablation's full-APEX row is literally the
main-benchmark runs. Execution is deterministic, each run records its code
revision and hardware, and we report the measured replicate noise (Appendix X,
±[E0b] pp)."*

**R1.4.** Adopt the reviewer's corrected statement verbatim for the old result.
Then:
- Explain that with 3 seeds the differences were within replicate noise
  (evidence as above, plus het scaling being ≈×1 at α=0.3).
- Present the new ablation at α=0.1, α=0.3 and N=200 with 5–10 seeds and
  paired tests.
- State the survival rule and its outcome, e.g. *"posterior regularisation and
  confidence-scaled blending are replaced by a single discounted prior-centred
  posterior; component X did not meet the rule and was removed."*
- Removing components is a strength here, not a concession.

**R2.5.** *"FedAvg samples K=10 of N clients uniformly at random without
replacement in every round (Section VI-A). Prompted by the reviewer's question,
we audited all baseline implementations against their original papers and found
deviations that explain the degradation. Our Oort and FedCS selectors assigned
zero utility to never-selected clients and had no exploration, so after round 0
they selected the same 10 clients in every round (Gini = 0.80 = 1 − K/N).
TiFL selected the lowest-index clients within each tier, and PoC ranked by
stale losses instead of querying current ones. We re-implemented all baselines
following the original algorithms: Oort ported from its official FedScale
implementation, FedCor from its official code. We tuned each with the same
budget as APEX and regenerated every result (Appendix Y lists each algorithm,
its hyperparameters and any remaining deviation). The corrected gaps are
[numbers]. Where PoC/Oort still trail FedAvg, the reason is [loss-biased
selection skew under label imbalance, as analysed by Cho et al.; Oort's system
utility optimises time-to-accuracy, which is irrelevant without deadlines]. We
now also report a system-heterogeneous setting with round deadlines (E11),
where their design goal applies."*

---

## 9. Manuscript changes

**Notation table** (new; R2.1, R2.2):

| Symbol | Meaning | Replaces / note |
|---|---|---|
| w_t, w_i^t | global / local model parameters | *only* model parameters use w |
| N, K, T, E | clients, cohort size, rounds, local epochs | |
| S_t, n_i, n_S, n | cohort; sample counts | |
| M, 𝒴 | number of classes, label set | avoids clashing L with the smoothness constant |
| p_i, p̄, p̄_S | client, global and cohort label distributions | |
| TV, KL, H(·), I(Z;Y) | distances, entropy, mutual information | |
| H | heterogeneity index I(Z;Y)/H(Y) | replaces the JSD scalar |
| λ_ℓ, λ_g, λ_s, λ_d | context weights | were w_l, w_g, w_s, w_q (R2.2) |
| β_q, β_d(φ), β_r | phase-dependent term weights | were phase-weight triples |
| κ, δ | prior strength, discount | replace γ, α_e, c_f |
| μ_i, v_i, θ̃_i | posterior mean, variance, sample | v_i instead of σ² (reserved for SGD variance) |
| α | Dirichlet concentration only | EMA α_e and Beta (α, β) are gone |
| φ, W, τ_c, τ_u, τ_e | phase, window, thresholds | |
| G, L, σ², σ_G² | bounded class gradient, smoothness, SGD variance, dissimilarity | Γ from Eq. 5 removed with it |

**Text and table fixes:**
- "eight baselines" → the correct count (R1.6).
- The Fig. 2(b) sentence (R1.3).
- The "α=0.3 most benign" sentence (R1.5).
- Table IV caption and a K column (R1.7).
- Remove "8 out of 9" (R1.14).
- Remove "significantly outperforms" (R1.1).
- "Zero trainable parameters" → "no gradient-trained parameters; 14 fixed
  hyperparameters".
- Eq. 16 → the reward and credit equations of §2.3(c–d).
- Table I Big-O corrected for every method.
- Temper the Impact Statement (R1.13).
- Add a code and data availability statement.
- Section ordering: Conclusion last.

**Claim audit.** Every number in the text comes from `numbers.tex` macros
generated from the canonical store. A script flags any literal number in the
`.tex` not backed by a macro or a table cell. SCOPE-FD used the same approach
(`Scope_FD_Revision_Package/CLAIM_AUDIT.md`). It closes the whole class of
errors behind R1.2, R1.3, R1.5, R1.6, R1.7 and R1.14.

**Structure:**
- §III: system model states A1–A3 and what clients send.
- §IV: revised algorithm and Algorithm 1.
- §V: Prop. 1, Corollary 1, Prop. 2 and the remark.
- §VI: protocol (seeds, statistics, fidelity, determinism) then results.
- §VII: limitations (A1 scope, SecAgg fallback, histogram privacy).
- Appendix: proofs, baseline fidelity, hyperparameters, sensitivity, per-seed
  results.

---

## 10. Decision gates and contingencies

- **G1 — after the E0 fixes (dev seeds, S1 and α=0.1).** Run APEX v2 with the
  validation reward against the faithful baselines. This measures how much of
  the old advantage survives. Expect it to shrink; that's the reason for the
  revised architecture.
- **G2 — pilot of the revised APEX (dev seeds, S1, α=0.1, (200,10)).**
  - *Freeze if:* revised APEX ≥ v2-with-validation-reward on average, and it
    isn't significantly below FedAvg in any dev setting.
  - *Otherwise:* allow at most two documented iterations.
  - *Then fall back:* keep v2's selection logic with only the mandatory fixes
    (no test data, faithful baselines, statistics, corrected complexity), and
    present the theory as motivation only.
  - Choose the reward variant here too.
- **G3 — component survival rule (after E5).** A component stays only if both
  hold across the three ablation settings:
  1. Removing it significantly hurts at least one targeted setting (accuracy,
     worst-class recall or cross-seed variance, Holm-adjusted).
  2. Removing it doesn't significantly help in any setting.

  Otherwise it's removed and the letter says so. Write the rule into the paper
  before looking at E5.
- **If APEX only ties everywhere,** the contribution rests on:
  - no server data;
  - provable per-round guarantees;
  - worst-class recall and robustness;
  - overhead;
  - fairness against *faithful* baselines.

  That's thin for TAI. E2's shards, E3's low-participation stress test and E7's
  drift are where selection has room to matter. If APEX doesn't separate there
  either, reconsider the journal or the framing before resubmitting.
- **Never:** drop a setting because APEX loses, report peak accuracy as the
  headline, or re-tune on evaluation seeds.

---

## 11. Timeline (6 weeks; compress to about 3 with P0 only)

| Week | Work |
|---|---|
| 1 | E0 (fidelity tests, forensics, replicate noise); determinism + canonical store; validation split; faithful PoC/Oort/TiFL/FedCS; port FedCor; start DivFL and FedAEB |
| 2 | Revised APEX (§2) + unit tests; G1 and G2 pilots on dev seeds; freeze the config (commit hash) |
| 3–4 | P0 then P1 sweeps on the cluster; E8 microbenchmark; write the theory section and appendix proofs in parallel |
| 5 | Statistics, tables, figures, claim audit; G3; E9/E10 diagnostics |
| 6 | Manuscript rewrite, response letter (verbatim comments from [APEX_R1_reviews.md](APEX_R1_reviews.md), pointer to every change), final claim audit |

---

## 12. Needed from you

Received: the manuscript (now in `csfl_simulator/Paper Corrections/APEX_R1/submitted/`)
and the FedAEB reference: Zheng, Sun & Ni, IEEE TVT 73(6), 2024. FedAEB is a
Soft Actor-Critic agent for joint client selection and resource allocation;
implement its selection policy and report its trainable parameters. The
submitted run data isn't available, and none of the plan depends on it.

Still open:

1. The revision deadline and the GPU budget, to decide how much of P1/P2 to run.
2. Whether a deterministic 200-round ResNet-18 run fits the per-run time estimate
   (one timed run on the cluster settles it).

---

## 13. Manuscript audit (`main_v3.tex` against the code and the reviews)

Findings from reading the submitted text against `apex_v2.py`,
`core/simulator.py` and the figure data. "Paper vs code" rows must be fixed
even if a reviewer never looks at the code, because the paper describes an
algorithm that wasn't run.

### 13.1 Statements that don't match the code

| # | Where | Paper says | Code does | Fix |
|---|---|---|---|---|
| M1 | Eq. 16, §IV-B | r_i = ΔAcc/\|S_t\|, "global test accuracy" | Δ of 0.6·acc + 0.2·time + 0.1·fair + 0.1·dp, all on the test set | Replaced by §2.3(c–d); nothing on the test set |
| M2 | Eq. 21 | Mean pairwise JSD over all N(N−1)/2 pairs, normalised to [0, 1] | Mean **√JSD** over ≤200 evenly spaced pairs, divided by 0.6, capped at 1 | Replaced by H = I(Z;Y)/H(Y) (§2.3b) |
| M3 | §IV-C text after Eq. 22 | het ≈ 0.3 at α=0.6 (diversity weight 0.60 → 0.18); het ≈ 0.9 at α=0.1 | het = 0.81 at α=0.6 (→ 0.48) and 1.00 at α=0.1 (audit script) | The numbers in the text are wrong; state H's measured values from the store |
| M4 | §IV-A | Window holds the "global average loss"; statistics over older/newer *halves* of W | Mean training loss of the *last selected cohort*; compares the last W rounds with the W before them | Revised detector uses a selection-independent signal (§2.3h); describe exactly what is computed |
| M5 | §IV-B | "Posterior mean and variance" | Welford running mean/variance of EMA-smoothed credits (not a Bayesian posterior); Beta fallback updated with min(credit, 1) | Revised posterior is a real conjugate update (§2.3e) |
| M6 | §VI-B | "APEX leads … as the Thompson posteriors become well-calibrated" | Posteriors can't move selection at this reward scale (A6) | Delete; replace with E10 diagnostics |
| M7 | §VI-D | "These results confirm that heterogeneity-aware scaling enables robust performance" at α=0.1 | Scaling is ×1.00 at α=0.1, i.e. inactive | Delete; H-scaling is evaluated by the E5 ablation |
| M8 | §VI-G | "Indirect evidence … at α=0.1, where heterogeneity-aware scaling activates" | Same as M7 | Delete; the new ablation runs at α=0.1 directly |

### 13.2 Claims the evidence doesn't support

| # | Where | Claim | Problem | Fix |
|---|---|---|---|---|
| M9 | §VI-B | APEX and FedAvg reach the lowest final loss | Fig. 2(b): APEX ≈ 0.82, MMR ≈ 0.84, FedAvg ≈ 0.85 at round 200 (R1.3) | Generated sentence from data |
| M10 | §VI-C, Fig. 3 caption | APEX "uniquely occupies the Pareto frontier" | (acc, Gini): APEX (.707, .19), MMR (.705, .16), FedAvg (.695, .08). None dominates another, so all three are on the frontier | State the frontier set; drop "uniquely" |
| M11 | §VI-B | Highest peak accuracy "with the lowest variance … highly reproducible" | Peak is selected on test data; 3 seeds; the ablation shows ~2 pp replicate spread | Peak becomes secondary; reproducibility is measured (E0b) |
| M12 | Abstract, §I, §VI-C, §VII | "4× better participation fairness than system-aware baselines" | Comparison is against the defective baselines (A1–A3) | Recompute against faithful baselines; likely changes |
| M13 | Abstract, §VI-D, §VII | "+14.4 pp above Oort at α=0.1" | Same | Recompute |
| M14 | Abstract, Impact, §VI-H, §VII | "Ranks first … in 8 out of 9 settings" | Count undefined; overlapping settings (R1.14) | Friedman + CD over enumerated settings |
| M15 | §VI-G | α=0.3 "the most benign" | α=0.6 is (R1.5) | Delete |
| M16 | §VI-G | "Removing phase hysteresis … identifying it as the dominant stability mechanism" | −0.09 pp with 3 seeds; std .039 vs .015 is one bad seed | Re-test at 3 settings (E5) |
| M17 | §VI-H | "A limitation is elevated seed variance under near-IID conditions" | No IID result appears in the paper | Report the IID row (E2) or remove |
| M18 | §V-A Remark 1(a) | Loss-biased selection "is the same mechanism that gives PoC its 3× speedup" | PoC queries fresh losses; APEX uses stale ones weighted 0.4 | Drop the analogy; Prop. 1 replaces Remark 1 |
| M19 | §V-A Remark 1(b) | "Consistent with the analysis in [zhang2026fednkrf]" | FedNK-RF is federated kernel learning; it doesn't analyse client selection | Remove the citation |
| M20 | §V-B | O(√(NT log T)) *Bayesian* regret citing Agrawal & Goyal (2012) | That paper gives frequentist bounds for Bernoulli bandits; the Bayesian bound is Russo & Van Roy. The bound's premise fails (R1.11) | Remark per §3.6 |
| M21 | §V-C | "Without hysteresis … accuracy drops exceeding 10 pp per round at α=0.1, as observed in our experiments" | No figure or table shows it | Show the phase trace and accuracy (E5/E10) or delete |
| M22 | Related Work, Oort | "An exploration bonus is used to avoid selecting the same clients repeatedly" | True of Oort, but not of the implementation used, which is why it locked in | Consistent once the baseline is faithful |

### 13.3 Theory and citations

| # | Where | Problem | Fix |
|---|---|---|---|
| M23 | Eq. 5, §I, §III-B | Cited to `li2020federated`, which is **FedProx** (Li, Sahu et al., MLSys 2020). The strongly convex bound with Γ and σ² is from Li, Huang, Yang, Wang & Zhang, "On the Convergence of FedAvg on Non-IID Data", ICLR 2020 | Replace with the non-convex bound (§3.5); fix the citation wherever FedProx is cited as the FedAvg bound |
| M24 | Eqs. 5–7 | σ² (Eq. 7) is defined as gradient *dissimilarity*. In the cited bound, σ² is stochastic-gradient variance and heterogeneity is Γ, so dissimilarity is counted twice and σ²/K is misattributed to partial participation | Rewritten in §3.5 with σ² (SGD noise) and σ_G² (dissimilarity) kept separate |
| M25 | Eq. 8, §III-C | Objective E‖w_T − w*‖² assumes a unique optimum (strong convexity) | Non-convex objective: min_t E‖∇F(w_t)‖², or final accuracy, stated as the empirical target |
| M26 | §III-C, constraint 3 | "Per-client selection overhead O(1) in d" | Still true for clients under the revised design. The server's O(K·d) credit computation must be stated separately |

### 13.4 Notation collisions (R2.1, R2.2)

| Symbol | Uses in the submitted text | Resolution |
|---|---|---|
| **w** | Model parameters (Eqs. 2–4); scalar weights w_l, w_g, w_s, w_q (Eq. 12); w_ts, w_div, w_rec (Eqs. 22, 24) | λ for context weights; β for term weights; w only for models |
| **σ²** | Gradient dissimilarity (Eq. 7); posterior variance σ²_i (Eq. 13); σ²_floor (Eq. 14); σ_rw (§IV-A) | σ² = SGD variance; σ_G² dissimilarity; v_i posterior variance; s_W window std |
| **α** | Dirichlet concentration; EMA α_e (Eq. 15); Beta parameters α_i (§IV-E) | α only for Dirichlet; the other two disappear in the revision |
| **B** | Batch size (§III-A); constant in Eq. 5 | Constants become c_1, c_2 |
| **L** | Local loss L_n (Eq. 1); number of classes L (Table I, §V-D) | M classes; L smoothness |
| **γ / Γ** | Blend weight γ (Eqs. 17–18); client drift Γ (Eq. 6) | γ disappears |
| **ρ** | Relative improvement rate (Eq. 9) | Keep; the plan's loss decrease is written ΔL_t in the paper |
| **Δ** | Rounds since selection Δ_i (Eq. 23) | a_i; Δ_i becomes the model update |
| **δ** | Dwell δ_min | D_min; δ is the discount |
| **n** | Client index n (§III); observation count n_i (§IV-B) | Clients i, j, k throughout; N_i discounted count |
| **i** | Sample index (Eq. 1) and client index (§IV) | Samples (x, y) ∈ D_i; i for clients |
| — | "defined below" before Γ, σ² (Eq. 5) | Define immediately after each equation; add a notation table |

### 13.5 Presentation

- Abstract: "eight baseline methods" (seven), "zero trainable parameters",
  "3.9 ms" without N, "4×", "14.4 pp". All are rewritten from the store.
- Table II: F1 bold goes to MMR (correct) but the text never discusses it
  (R1.9). Say "macro-F1". Replace "Final Acc." with the last-10-round mean.
- Table III: no std (R2.3). Every method's Gini is identical at α = 0.1, 0.3
  and 0.6 (APEX .19/.19/.19, PoC .53/.53/.53). For random or locked-in
  selectors that is expected, but for APEX it means its participation doesn't
  respond to heterogeneity at all. Check it in the new runs, and say what it
  means if it persists.
- Table IV: no std; caption says K=10 but N=200 used K=20 (R1.7); duplicated
  C-10 and N=50 columns (R1.14 overlap).
- Table V: the "Std" column duplicates the ± values.
- Fig. 2 caption: shading "for APEX" only; shade every method or none.
- Fig. 1 (infographic): must be redrawn for the revised components.
- §VI-A: add the validation split, seeds (dev vs eval), determinism,
  baseline tuning and the fidelity appendix.
- Related Work: add Fed-CBS and the non-convex FedAvg analysis; keep FedAEB
  (now compared, R2.6).
- §VII/§VIII: merge Future Work into Conclusion, or keep but trim; either is
  fine for TAI.

---

## 14. Section-by-section rewrite map

| Section | Keep | Rewrite | New |
|---|---|---|---|
| Title | ✓ | — | — |
| Abstract | Framing sentence | Every number, from the store; "ranking first" becomes a statistically qualified statement | "No gradient uploads, no server test data" |
| Impact Statement | Opening | Remove "8 out of 9", "4×"; temper the healthcare/IoT claim (R1.13) | One sentence on drift support |
| I. Introduction | Motivation | Fix the Eq. 5 citation (M23); contribution list rewritten to match what's proven | Contribution: Prop. 1 and Prop. 2 |
| II. Related Work | Structure | MMR paragraph (M10-consistent); Oort paragraph | Fed-CBS, non-convex FedAvg analysis; Table I with a measured-cost column pointer |
| III. System Model | A–B setup (Eqs. 1–4) | III-B convergence bound (M23–M25); III-C problem statement | A1–A3; validation split; what clients send; notation table |
| IV. APEX | Phase detector idea, recency, greedy | IV-B posterior and reward (M1, M5); IV-C diversity and het (M2, M3); Algorithm 1 | Credit assignment subsection; histogram refresh; privacy paragraph with the E13 result |
| V. Theory | — | Remark 1 → Prop. 1 + Corollary 1; Prop. 2 → Remark; complexity with measured costs | Prop. 2 (submodularity); "Where the theory stops" paragraph; appendix proofs |
| VI. Experiments | Datasets, models | Everything else, from the store | Protocol subsection; faithful baselines; ablation at 3 settings; runtime table (R2.4); drift; E9/E10 diagnostics; F1 and worst-class discussion |
| VII–VIII | — | From results | Limitations paragraph: A1 scope, SecAgg fallback, histogram privacy |
| Appendix (new) | — | — | Proofs; baseline fidelity table; hyperparameters and grids; sensitivity; per-seed results; replicate noise |

---

## Appendix — citations to verify before they go in the paper

Each of these is used above from memory. Check the venue, year, theorem number
and exact statement against the source.

- Cho, Wang & Joshi, "Towards Understanding Biased Client Selection in
  Federated Learning", AISTATS 2022 (PoC; the selection-skew analysis).
- Lai et al., "Oort: Efficient Federated Learning via Guided Participant
  Selection", OSDI 2021; FedScale code.
- Nishio & Yonetani, "Client Selection for Federated Learning with
  Heterogeneous Resources in Mobile Edge", ICC 2019 (FedCS).
- Chai et al., "TiFL: A Tier-based Federated Learning System", HPDC 2020.
- Tang et al., "FedCor: Correlation-Based Active Client Selection Strategy for
  Heterogeneous Federated Learning", CVPR 2022.
- Balakrishnan et al., "Diverse Client Selection for Federated Learning via
  Submodular Maximization", ICLR 2022 (DivFL).
- Zhang et al., "Fed-CBS: A Heterogeneity-Aware Client Sampling Mechanism for
  Federated Learning via Class-Imbalance Reduction", ICML 2023.
- Yang, Fang & Liu, "Achieving Linear Speedup with Partial Worker Participation
  in Non-IID Federated Learning", ICLR 2021 (non-convex bound; check its exact
  form).
- Wang & Ji, "A Unified Analysis of Federated Learning with Arbitrary Client
  Participation", NeurIPS 2022.
- Wang & Chen, "Thompson Sampling for Combinatorial Semi-Bandits", ICML 2018.
- Garivier & Moulines, "On Upper-Confidence Bound Policies for Switching Bandit
  Problems", ALT 2011.
- Nemhauser, Wolsey & Fisher, "An Analysis of Approximations for Maximizing
  Submodular Set Functions", Math. Programming 1978.
- Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking…", SIGIR 1998.
- Russo & Van Roy (2014/2018) for the original O(√(NT log T)) Bayesian regret
  reference in the submitted §V.B.
- Yan et al., CriticalFL, KDD 2023.
- Li, Huang, Yang, Wang & Zhang, "On the Convergence of FedAvg on Non-IID
  Data", ICLR 2020. The submitted paper cites FedProx (`li2020federated`) for
  this bound (M23).
- Carbonara, Drioli & Foresti, "Diversity-Aware Client Selection via Maximal
  Marginal Relevance for Federated Learning", ICASSP 2024 (MMR-Diverse).
- Zheng, Sun & Ni, "FedAEB", IEEE TVT 73(6), 2024.
