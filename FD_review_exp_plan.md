# SCOPE-FD — Major Revision Action Plan (TAI-2026-May-A-00956)

**Status:** Planning stage. No execution code has been written yet — this document
specifies *what* to build and *in what order*, grounded in a read of the actual
reviews (`csfl_simulator/experiments/SCOPE-FD Reviews/reviews.txt`) and the actual
codebase (`csfl_simulator/selection/fd_native/scope_fd.py`, `core/fd_simulator.py`,
`core/channel.py`, `core/dp.py`, `core/parallel.py`, `presets/methods.yaml`,
`docs/SCOPE_FD_reference.md`, `docs/SCOPE_FD_paper_narrative.md`).

**Companion docs:** [SCOPE_FD_reference.md](SCOPE_FD_reference.md) (algorithm),
[SCOPE_FD_paper_narrative.md](SCOPE_FD_paper_narrative.md) (current results/figures).
This plan only adds what's needed to answer the reviewers; it does not repeat
algorithmic detail already documented there.

---

## 0. Codebase audit — what already exists vs. what must be built

This matters because it changes the cost estimate of nearly every item below.

| Capability | Status | Evidence |
|---|---|---|
| Debt-only ablation (`α_u=α_d=0`) | **Missing — 1-line fix** | `presets/methods.yaml` has `scope_fd_no_server` (debt+diversity) and `scope_fd_no_diversity` (debt+uncertainty) but no `scope_fd_debt_only` variant with both zeroed |
| Debt + under-prediction only | **Exists** | `scope_fd_no_diversity` (`disable_diversity_penalty=True`) — this *is* Reviewer 2's "debt plus under-prediction" ablation |
| Debt + coverage only | **Exists** | `scope_fd_no_server` (`disable_server_signal=True`) — this *is* Reviewer 2's "debt plus coverage" ablation |
| Literature FL-selection baselines ported to FD (reviewer's "DevFL" example) | **Missing** | Real published FL client-selection methods need to be ported into the FD pipeline — see §1.2/BL-1 for a researched shortlist (DivFL, SubTrunc, UnionFL) with real citations and portability notes |
| mMIMO channel noise (UL/DL, ZF/MMSE, quantization) | **Exists** | `core/channel.py`, wired into `fd_simulator.py`, driven by `--channel-noise`, `--ul-snr-db`, `--dl-snr-db` |
| Energy accounting per client | **Exists but unused by selection** | `ClientInfo.energy_rate`, `estimated_energy` (`core/client.py`, `core/system.py`), `--energy-budget` flag — never read by `scope_fd.py`'s score |
| DP mechanism | **Exists, wrong target** | `core/dp.py` does Gaussian noise on **gradients** + an epsilon budget counter. Nothing perturbs the **label histogram** that reviewers are worried about |
| Multi-seed loop / mean±std / CI plotting | **Missing** | `--seed` is a single int everywhere; `run_scope_experiments.sh` hardcodes `seed 42`; no aggregation script exists |
| Async FD / client dropout | **Missing** | `fd_simulator.py` always waits for all K selected clients; no staleness buffer, no non-response injection |
| Non-image dataset support | **Adjacent asset exists** | `csfl_simulator/experiments/audio_fsdd/` runs FSDD audio (log-spectrograms) but only in **FL** paradigm, built for the *MAML-Select* paper, not FD |
| Dataset catalogue | **Broader than the paper uses** | `core/datasets.py` already supports MNIST, Fashion-MNIST, KMNIST, EMNIST/digits, CIFAR-10, CIFAR-100 — the paper only *reports* FMNIST/MNIST (+CIFAR-10 as "informational") |
| AMP / mixed precision | **Exists, on by default** | `--use-amp` (default `True`) |
| GPU-aware client parallelism | **Exists** | `core/parallel.py` `ParallelTrainer._auto_detect_parallel()` sizes concurrent client replicas from free VRAM (60% headroom heuristic, capped at 8 streams) |
| DataLoader tuning | **Exists** | `core/datasets.py::make_loader` already sets `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor=2` |
| Memory hygiene helpers | **Exists** | `core/utils.py::cleanup_memory`, `check_memory_critical` |
| K∤N generalization | **Untested but not structurally broken** | The debt formula `target = (round_idx+1)*K/n` is continuous-valued, not mod-based, so it should degrade gracefully when `K∤N` — this is an experiment, not a code fix |

**Takeaway:** roughly half of the reviewers' asks are *experiment-design* problems
(run the sweep, add seeds) rather than *missing-feature* problems. The genuinely
new engineering is: (1) a multi-seed/statistics harness, (2) histogram-level
privacy mechanisms, (3) dropout/staleness injection, (4) a channel/energy-aware
4th scoring term, (5) real literature client-selection baselines ported into
the FD pipeline, (6) porting the FD pipeline to a non-image dataset.

Reviewer 1 explicitly asked for *literature* methods (DevFL — almost certainly
a typo for **DivFL**, Balakrishnan et al., ICLR 2022, already cited as [E1] in
the narrative doc's related work) to be ported into the FD pipeline; see §1.2/BL-1
for the full researched shortlist and portability assessment.

---

## 1. Reviewer Requirement Extraction Matrix

Every sentence from the reviews, grouped by theme, with source IDs so nothing
gets lost. Where reviewers duplicate a request, the source list shows all of them
so it's answered once, not three times independently.

### 1.1 Convergence / Theory

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **TH-1** | R1 ("analytical contribution of V-B is minimal... can easily be derived"), R2.3 ("needs more care... provide a clearer theorem... or state more modestly"), Attached-Doc §1 ("clarify whether assumptions... remain satisfied") | The paper claims FedTSKD's O(1/t) bound holds "without modification" under partial participation, but reviewers agree this is asserted, not derived | Two-track fix: (a) **attempt** a real partial-participation theorem that ties the bound's variance-dependent constant explicitly to the participation-Gini (formalizing the current hand-wave in §6.2 of the narrative doc); (b) **regardless of (a)'s success**, rewrite the claim to R2's explicit suggested fallback — "SCOPE-FD is compatible with the FedTSKD pipeline under the assumptions of [13]" — and add an assumption-by-assumption satisfaction discussion (which of [B1]'s Theorem-1 sampling assumptions SCOPE's *deterministic, without-replacement* rotation does or doesn't satisfy). This is pure theory/writing work — no GPU experiments — but it *is* informed by the empirical Gini/variance data already in hand and the new K∤N and non-aligned-window data from GN-6/Attached §2 below |

### 1.2 Baselines & Comparisons

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **BL-1** | R1 ("baseline is too simple... other client selection mechanisms... DevFL... submodular diversity maximization... similar to SCOPE's coverage penalty") | Random alone is not a fair baseline; need real *published* FL-selection methods, ported into the FD pipeline | Literature search (below) turned up one direct match to the reviewer's own example plus two more recent, more portable candidates. Recommendation: commit to **DivFL** (the reviewer's named example) and **SubTrunc**; treat **UnionFL** as a cheap bonus row; treat **AdaFL** and **FedSTS** as stretch-only given weaker portability. All are ported as new selector files under `selection/fd_native/`, run at the same headline/K-sweep configs as SCOPE-FD and random |
| **BL-2** | R3.1 ("explain clearly why traditional FL client selection methods cannot be directly applied to FD") | Needs to be a *demonstrated* fact, not asserted prose | Port one FL-style selection policy (e.g., a loss/utility-greedy selector resembling FedCS/Oort) to run **unmodified inside the FD pipeline** at the SCOPE headline config, and show it either produces pathological Gini or no accuracy benefit — giving Fig-quality evidence for the "FD's data-size-weighted logit averaging flattens per-client quality" argument already in §2.2 of the narrative doc |

#### BL-1 literature search — candidate baselines and portability

FD clients only ever produce two things a selector can see: a **static label
histogram** (partition-time) and, per round, a **logit vector on the shared
public dataset**. Any FL baseline whose selection signal is fundamentally a
**raw model gradient/update** cannot be ported faithfully without inventing a
substitute signal — which risks the reviewer later objecting that the "ported"
baseline isn't really the cited method anymore. This is the deciding factor
below, not just recency or venue prestige.

| Method | Citation | Venue / status | Core signal | Portability to FD | Verdict |
|---|---|---|---|---|---|
| **DivFL** | R. Balakrishnan et al., "Diverse Client Selection for Federated Learning via Submodular Maximization" | ICLR 2022 (non-IEEE, already cited [E1] in narrative's related work — IEEE TAI accepts seminal non-IEEE venues) | Submodular facility-location maximization over **gradient space** | Needs a substitute representation. **Recommended FD-native adaptation:** replace the gradient vector with each client's **public-set logit vector** (the same quantity FD already exchanges) as the facility-location representation — this stays true to DivFL's actual algorithm (submodular greedy over a similarity/dissimilarity matrix), just swapping the embedding source | **Commit.** This is literally the reviewer's own named example |
| **SubTrunc** | A. C. Castillo Jiménez, E. C. Kaya, L. Ye, A. Hashemi, "Equitable Client Selection in Federated Learning via Truncated Submodular Maximization" | **IEEE ICASSP 2025** (already cited [E2] in narrative — real IEEE venue, most recent of the candidates) | Truncated submodular maximization diversified by **client loss** | **High.** "Client loss" is directly computable in FD as each client's loss on the shared public dataset — no gradient needed at all. This is the easiest and most literature-faithful port available | **Commit — primary new baseline** |
| **UnionFL** | Same authors, extended version: arXiv:2408.13683, "Submodular Maximization Approaches for Equitable Client Selection in Federated Learning" | **arXiv preprint only as of this writing — not yet confirmed in an IEEE-indexed venue; verify against IEEE Xplore before citing in the manuscript**, same authors as SubTrunc | Submodular selection diversified by **historical selection data** (i.e., past participation) | High — historical-participation signal maps directly onto FD's existing `participation_count` field, no gradient needed | **Optional bonus row** if E2b's infra is already built for SubTrunc — conceptually close to SCOPE's own debt term, so it doubles as a "does literature already solve fairness the way we do?" comparison, not just an accuracy contest |
| **AdaFL** | Q. Li, X. Li, L. Zhou, X. Yan, "AdaFL: Adaptive Client Selection and Dynamic Contribution Evaluation for Efficient Federated Learning" | **IEEE ICASSP 2024** (IEEE Xplore doc 10447356) | Dynamically varies **how many** clients are selected per round, plus a contribution-evaluation term | Medium-low — its headline mechanism (adaptive K) is a different problem than SCOPE-FD's fixed-K setting, and porting it changes what's being compared; the contribution-evaluation half would need a gradient substitute | **Stretch only** — not worth the scope creep unless a reviewer specifically follow-up asks about variable-K selection |
| **FedSTS** | D. Gao et al., "FedSTS: A Stratified Client Selection Framework for Consistently Fast Federated Learning" | IEEE journal, 2024 (IEEE Xplore doc 10689614 — **exact journal/volume/issue unverified, confirm via Crossref before citing**) | Stratifies clients by similarity of **compressed ("Information-Squeezed") gradients** | Low — explicitly gradient-based; a faithful port would require defining an analogous "compressed public-set logit" stratification signal, which is enough of a re-design that it stops being a faithful reproduction of FedSTS | **Stretch only / likely skip** — flag as a method we considered and deliberately excluded, with the reason stated, rather than silently omitting it |

**Recommendation given the time budget:** commit to **DivFL** (adapted to use
public-set logits, since it's the reviewer's own named example and skipping it
would look evasive) and **SubTrunc** (the most faithfully portable, and already
IEEE ICASSP 2025 — the most recent and citable of the set) as the two new
baseline selectors. Add **UnionFL** only if it comes essentially free once
SubTrunc's submodular-greedy scaffolding exists (same authors, similar
implementation shape). Do not pursue AdaFL or FedSTS unless a second revision
round specifically demands them — both require inventing a substitute signal
that a careful reviewer could reasonably call out as no longer the cited method.

### 1.3 Ablation Studies

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **AB-1** | R2.1 ("the most important missing comparison is a debt-only selector... please also report ablations for debt plus under-prediction only and debt plus coverage only"), Attached-Doc §4 (same four-way ablation, worded independently) | Unclear whether SCOPE's gains come from the under-prediction/coverage terms or simply from balanced rotation | **Register `scope_fd_debt_only`** (`alpha_uncertainty=0, alpha_diversity=0`) — the only missing variant; the other three (debt+underpred, debt+coverage, full) already exist as registered methods. Run the complete 4-way ablation table at every headline/K-sweep/channel-sweep config, multi-seed |
| **AB-2** | R1.4 ("coefficients... are not ablated. The selection process... needs to be described"), R3.2 ("how sensitive is the proposed algorithm to α_u and α_d?") | No empirical grid search backs the magnitude-analysis argument in §7.2/§8 of the reference doc | Run an `α_u × α_d` grid sweep (e.g., `α_u ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8}`, `α_d ∈ {0, 0.05, 0.1, 0.2, 0.4}`) on 1–2 representative configs; render as an accuracy-heatmap + Gini-heatmap; use it to *empirically validate* (or correct) the magnitude-analysis claim, and write up the selection rationale as a proper subsection rather than a footnote |

### 1.4 Reproducibility / Multiple Seeds

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **RP-1** | R1.3 ("useful to show confidence intervals on the graphs") | Single-seed plots have no uncertainty bands | Build the multi-seed harness (§3.3 below); regenerate every figure with shaded CI bands |
| **RP-2** | R2.4 ("single reported seed... 'statistically tied' and '3x faster' should be supported by multi-seed results and error bars... complemented with final accuracy tables, rounds to a fixed absolute accuracy, and sensitivity to α_u, α_d, Dirichlet α, and K/N") | Most detailed reproducibility ask; bundles seeds + new metrics + sensitivity sweeps | Multi-seed harness (3–5 seeds) on every headline config; add two new metrics beside "rounds to 80% of final": (a) plain final-accuracy table, (b) rounds-to-fixed-absolute-accuracy-threshold (e.g., 60/70/80% absolute, not relative-to-final); wire the α_u/α_d/Dirichlet-α/K-N sweeps (AB-2, GN-2/3) into the same statistics pipeline |
| **RP-3** | Attached-Doc §3 ("reporting results over multiple independent runs together with... mean/standard deviation would substantially improve... particularly for convergence-speed comparisons and the sparse participation setting") | Same ask, emphasizes convergence-speed claims and K=1 specifically | Covered by RP-2's harness; ensure K=1 spotlight (§8.3 of narrative) and the channel-robustness sweep (§8.6) are prioritized first since those carry the paper's two strongest claims |

### 1.5 Privacy

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **PR-1** | R2.2 ("Eqs. (12) and (14) require each client to reveal its label histogram... this should be treated as a central limitation. Please either evaluate the Laplace-noise or server-side surrogate variants mentioned in Section IV-D, or narrow the privacy-related claims") | Histogram-sharing is a real privacy leak; paper's own IV-D promises variants it never evaluates | Implement **both** variants explicitly promised in IV-D: (a) Laplace-mechanism DP on the histogram before it reaches the selector, swept over ε ∈ {0.1, 0.5, 1, 2, 5, ∞}; (b) a **server-side surrogate** histogram estimated indirectly from each client's submitted public-set logits (no direct histogram disclosure) as a `scope_fd_surrogate_hist` variant. Report accuracy/Gini degradation curves vs. ε and vs. surrogate-only, so the "narrow the claims" fallback becomes unnecessary |
| **PR-2** | Attached-Doc §5 ("a slightly more detailed discussion of the practical implications of sharing these statistics") | Wants deeper prose, not just an experiment | Fold PR-1's results into a proper limitations paragraph: what an honest adversary learns from a label histogram vs. raw data, and how much the Laplace/surrogate variants close that gap at what accuracy cost |

### 1.6 System / Wireless (mMIMO) Modeling

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **SW-1** | R2.5 ("the selection score... does not include any channel or energy information. The SNR sweep... does not show that SCOPE-FD exploits MIMO properties. Please either add channel/energy-aware experiments or clearly position SCOPE-FD as a fairness/data-coverage selector running on top of an mMIMO communication substrate") | Reviewer offers an explicit either/or; current SNR sweep only proves robustness, not exploitation | Do **both**, since they're cheap together: (a) add an optional 4th scoring term `α_c · q_i` reading the already-existing `ClientInfo.channel_quality`/`energy_rate`/`estimated_energy` fields, magnitude-capped the same way as the other two terms (`scope_fd_channel_aware` variant), and evaluate it under an energy-budget-constrained scenario (reuse `--energy-budget`) where channel-awareness should visibly help; (b) regardless of (a)'s effect size, soften the framing in Intro/Discussion to explicitly say SCOPE is a fairness/coverage selector layered on an mMIMO substrate it does not itself exploit, unless (a) shows a real, defensible gain |
| **SW-2** | R1 ("Experiment scale is too small to represent Massive-MIMO systems") | N=30/50 is small for "massive" | Scale-up sweep: N ∈ {50, 100, 200, 500} at matched K/N ratios (5%/10%/20%), reusing the existing `parallel.py` VRAM-aware client-batching so wall-clock stays bounded |

### 1.7 Generality / Scale (a category the user's brief didn't name explicitly but the reviews clearly need)

| ID | Source(s) | Claim / Question | Proposed Resolution |
|---|---|---|---|
| **GN-1** | R1 ("only two datasets... very similar characteristics... applicability... to domains other than images is questionable") | Narrow domain coverage | Repurpose the existing `audio_fsdd` pipeline (built for MAML-Select, FL paradigm) into an **FD-paradigm** run (`--paradigm fd`) with the existing `AudioCNN` + FSDD log-spectrograms, comparing SCOPE-FD vs. random vs. debt-only vs. DivFL. This is the fastest credible non-image result given existing infra. A true non-grid (tabular) dataset is a stretch goal if time allows |
| **GN-2** | R1 ("Only a single value of Dirichlet α (0.5) is tested") | No heterogeneity sweep | α ∈ {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0, iid}, headline N/K, multi-seed |
| **GN-3** | R3 ("How does SCOPE-FD perform under more severe non-IID settings?") | Same request, different reviewer | Covered by GN-2's extreme-α tail (0.01–0.1) |
| **GN-4** | R3 ("since the proposed method relies on a public dataset, how sensitive is its performance to the quality or distribution of that public dataset?") | Public-set robustness untested | Sweep public-dataset identity (MNIST/FMNIST/EMNIST cross-pairs, some already exist), size (2000 → 500 → 100), injected label noise, and a deliberately mismatched-domain public set (e.g., CIFAR public for an FMNIST-private run) |
| **GN-5** | R3.1 | (duplicate of BL-2 — cross-referenced, not re-solved) | See BL-2 |
| **GN-6** | Attached-Doc §2 ("additional experiments using different values of N and R, configurations where K does not divide N, or evaluation windows that do not exactly coincide with complete participation cycles") | Fairness guarantee only demonstrated at clean K|N configs | K∤N configs (e.g., N=47,K=6; N=53,K=7); vary N and R independently; compute a **rolling-window Gini** (window not aligned to ⌈N/K⌉) instead of only the full-cycle Gini currently reported |
| **GN-7** | R3 ("How would the method perform under asynchronous federated learning or client dropout scenarios?") | No dropout/async support exists at all | Two separate, scoped experiments: (a) **client dropout** — inject a per-round non-response probability `p_drop` after selection, server aggregates over whoever actually returned logits; (b) **bounded-staleness async** — reuse the existing per-client latency heterogeneity (`energy_rate`/`estimated_duration`) to let clients return logits 1–2 rounds late, aggregated with a staleness-discounted weight. Scoped deliberately smaller than a full asynchronous-FL engine rebuild — the manuscript is not an async-FL paper, so a bounded-staleness variant is honest and sufficient |

### 1.8 Writing / Presentation (no experiments — tracked so nothing is dropped)

| ID | Source(s) | Fix |
|---|---|---|
| **WR-1** | R3 ("Improve the resolution of Figure 1") | Regenerate at higher DPI/vector format; verify the bottleneck is render settings and not the underlying data resolution |
| **WR-2** | R2.6 | Define "K\|N" as a divisibility condition in the abstract; fix "[13].FMNIST" spacing; use lowercase "for" in the title if title-case is applied consistently elsewhere |

---

## 2. High-Performance Experimental Setup & Code Architecture Plan

Target: a single NVIDIA GPU, 16 GB VRAM, running the full sweep suite above
(roughly 15–20 experiment families × 3–5 seeds × several configs each) without
babysitting.

### 2.1 Memory Management

The models in play (FD-CNN1/2/3, small ResNet/MobileNet/ShuffleNet for CIFAR,
`AudioCNN` for GN-1) are all small by 16GB standards — the existing
`ParallelTrainer._auto_detect_parallel()` heuristic (60% of free VRAM ÷
`model_mem × 4`, capped at 8 concurrent client replicas) already has generous
headroom for these. Concretely:

- **Mixed precision** is already default-on (`--use-amp`, default `True`). Keep it on for every new experiment; it's free throughput and free memory headroom.
- **Gradient accumulation** is not currently needed for the CNN-scale models used today, but should be added as a *contingency path* for two specific new experiments that could push memory: (a) the CIFAR-100/ResNet scale-up under SW-2's larger-N configs, if `--model-pool` includes anything beyond the current light pool; (b) the audio pipeline if spectrogram batch sizes grow. Concretely: expose a `--grad-accum-steps` flag in the client local-training loop that splits a logical batch into `N` micro-batches, only stepping the optimizer every `N`th micro-batch — gated behind a VRAM check (`check_memory_critical`) so it only activates when needed, rather than unconditionally slowing every run.
- **Aggressive garbage collection**: `cleanup_memory()` already exists and should be called (a) after every completed run in the multi-seed loop, before the next seed starts (release model replicas + optimizer state + CUDA cache), and (b) inside the async/dropout experiments specifically, since those may hold onto stale client state across "late" rounds longer than the synchronous path.
- **VRAM guard rail**: wrap every experiment invocation in the orchestrator with a `check_memory_critical(threshold_percent=90)` pre-flight check; if it trips, halve the requested `--parallel-clients` for that run rather than crashing mid-sweep.

### 2.2 Parallelization & Concurrency

- **Data loading**: `core/datasets.py::make_loader` already sets `pin_memory`, `persistent_workers`, and `prefetch_factor=2`. Default `num_workers` should be raised from whatever the CLI default is today to `min(4, os.cpu_count())` for all new sweep configs — public-dataset inference (needed every round for the server-uncertainty term) is a second forward pass per round and benefits from prefetch just as much as local training does.
- **Client-level parallelism**: keep using `ParallelTrainer`'s existing CUDA-stream-per-client-replica design unmodified — it already saturates the GPU within a round. Nothing to build here; just make sure every new experiment class (dropout, async, channel-aware) still routes through it rather than falling back to a sequential loop.
- **Seed-level concurrency (new decision)**: do **not** run multiple seeds truly concurrently inside one process — each seed would multiply the `ParallelTrainer` replica pool by however many seeds run at once, which stacks additively on top of the existing per-round client parallelism and risks exceeding 16 GB precisely when the models grow (SW-2 scale-up, GN-1 audio). Instead:
  - Default: **sequential seed loop**, one OS process per (config, seed), because these models are cheap enough that seed-level wall-clock, not VRAM, is the actual constraint.
  - Opt-in: allow **2 concurrent seed processes** (not more) when a pre-flight `nvidia-smi`/`check_memory_critical` probe shows more than ~50% VRAM still free after the first process's `ParallelTrainer` has allocated its replicas — gated automatically by the orchestrator, not a manual flag the user has to tune per run.
- **GPU saturation monitoring**: the orchestrator should log `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` on a background timer for the duration of the sweep, so idle gaps (e.g., between-round Python/CPU-bound bookkeeping in `fd_simulator.py`) are visible and can be targeted for further overlap if they turn out to dominate wall-clock.

### 2.3 Reproducibility Framework (the highest-leverage new piece of infrastructure)

This is the single build that unblocks RP-1, RP-2, RP-3, and materially strengthens
AB-1/AB-2/GN-2/GN-6 — worth building once, well, rather than ad hoc per experiment.

**Design:**

```
scripts/run_scope_revision_suite.py      # new: Python orchestrator (not bash)
configs/scope_revision_sweeps.yaml       # new: declarative sweep spec
csfl_simulator/experiments/scope_fd/
    aggregate_results.py                 # new: seed -> mean/std/CI, sig. tests
    plot_with_ci.py                      # new: reuses plot_scope_paper.py style,
                                          #      adds shaded CI bands / error bars
```

- **Sweep spec (`configs/scope_revision_sweeps.yaml`)**: one YAML file, one entry per
  experiment *family* (not per run), each entry expands to a Cartesian product of
  `{method, dataset, N, K, dirichlet_alpha, dl_snr_db, seed, ...}` — this is what
  keeps hyperparameter sweeps (α_u/α_d grid, Dirichlet-α sweep, N/K scale-up,
  public-dataset sensitivity, dropout probability, ε for the Laplace mechanism)
  free of code duplication: they're all "just another axis in the grid," not a
  new script each.
- **Orchestrator (`run_scope_revision_suite.py`)**: reads the YAML, expands the grid,
  and for each `(config, seed)` pair:
  1. computes a deterministic run-directory name from a hash of the config + seed,
  2. **skips** it if a completed `compare_results.json` already exists there (resume/idempotent — critical for a multi-day sweep on one GPU),
  3. runs the pre-flight VRAM check (§2.1),
  4. invokes the existing `csfl_simulator/__main__.py run`/`compare` CLI as a subprocess (reuses everything already built — no need to re-architect the simulator itself),
  5. logs GPU utilization in the background (§2.2),
  6. calls `cleanup_memory()` after the subprocess exits.
- **Aggregation (`aggregate_results.py`)**: given a run-directory glob (all seeds for
  one config), computes per-metric mean, std, and a 95% CI (bootstrap or
  t-distribution given typically 3–5 seeds), plus a paired significance test
  (Wilcoxon signed-rank across matched seeds) for every SCOPE-vs-baseline
  comparison — this is what turns "statistically tied" / "3× faster" from
  assertions into supported claims (RP-2). Emits both a machine-readable JSON and
  a LaTeX-ready table fragment (matching the paper's existing table style).
- **Plotting (`plot_with_ci.py`)**: thin wrapper around the existing
  `scripts/plot_scope_paper.py` rendering conventions (same `.eps`+`.png` pairing,
  same style presets) but reading the aggregated mean/std instead of a single
  run, with shaded CI bands on learning curves and error bars on bar charts —
  directly answers RP-1 and regenerates Fig. 1/4/6/8/9/13 from the narrative doc.

### 2.4 Ablation & Hyperparameter Sweep Configuration Structure

All sweeps (α_u/α_d grid, Dirichlet-α, N/K/K∤N, DL-SNR × K, dropout probability,
Laplace ε, public-dataset variants) are just additional axes in the same YAML
schema described in §2.3 — no new argparse surface is needed beyond what
`csfl_simulator/__main__.py` already exposes (`--scope-au`, `--scope-ad`,
`--dirichlet-alpha`, `--dl-snr-db`, etc. all already exist per the existing CLI).
The only two flags this plan requires *adding* to the CLI are:
- `--dropout-prob` (client non-response probability per round; consumed by the new dropout injection point in `fd_simulator.py`),
- `--staleness-window` (max rounds a client's logits may lag before being discarded; consumed by the new bounded-staleness aggregation path).

Everything else (channel-aware 4th term's `alpha_channel`, Laplace `epsilon`,
surrogate-histogram toggle) is a per-method `params:` entry in
`presets/methods.yaml`, exactly like the existing `scope_fd_no_server`/
`scope_fd_no_diversity` registrations — zero new plumbing required.

---

## 3. Detailed Experiment List

| # | Experiment | Addresses | New Code? | Relative Cost | Seeds |
|---|---|---|---|---|---|
| E1 | 4-way ablation (debt-only / +under-pred / +coverage / full) at headline + K-sweep + channel-sweep configs | AB-1 | 1-line registration (`scope_fd_debt_only`) | Cheap | 3–5 |
| E2a | DivFL ported to FD (Balakrishnan et al., ICLR 2022; facility-location submodular greedy over per-client **public-set logit vectors**, substituting for the original gradient-space representation) | BL-1 | New selector file `fd_native/divfl_fd.py` | Cheap–Medium | 3–5 |
| E2b | SubTrunc ported to FD (Castillo et al., IEEE ICASSP 2025; truncated submodular maximization diversified by **client loss on the public dataset**) | BL-1 | New selector file `fd_native/subtrunc_fd.py` | Cheap (loss is already computed each round) | 3–5 |
| E2c | UnionFL ported to FD (same authors, arXiv:2408.13683 — **verify IEEE-indexed publication before citing**; diversified by historical participation) — bonus row only if E2b's scaffolding makes this near-free | BL-1 | Reuses E2b's submodular scaffolding + existing `participation_count` field | Cheap (if pursued at all) | 3–5 |
| E3 | α_u × α_d grid sweep + heatmaps | AB-2 | None (YAML sweep only) | Cheap | 3 |
| E4 | Multi-seed pass over all existing headline/K-sweep/K=1-spotlight/channel-sweep configs | RP-1, RP-2, RP-3 | Orchestrator + aggregator (§2.3) | Medium (mostly wall-clock, reruns existing cheap configs) | 3–5 |
| E5 | New metrics: final-accuracy table + rounds-to-fixed-absolute-accuracy | RP-2 | Small addition to `aggregate_results.py` | Cheap | (reuses E4 data) |
| E6 | Extended Dirichlet-α sweep {0.01…5.0, iid} | GN-2, GN-3, R1 | None | Cheap–Medium | 5 |
| E7 | N/K scale-up {50,100,200,500} + explicit K∤N configs + rolling-window Gini | SW-2, GN-6 | Rolling-window Gini metric (small addition to `metrics.py`) | Medium | 5 |
| E8 | Client dropout injection, `p_drop ∈ {0,0.1,0.2,0.3}` | GN-7 (dropout half) | New hook in `fd_simulator.py` client loop + `--dropout-prob` flag | Medium | 5 |
| E9 | Bounded-staleness async variant | GN-7 (async half) | New staleness buffer + `--staleness-window` flag | Medium–High | 5 |
| E10 | Laplace-noise histogram DP sweep (ε) + server-side surrogate variant | PR-1, PR-2 | `laplace_noise_histogram()` in `dp.py` + `scope_fd_surrogate_hist` selector variant | Medium | 5 |
| E11 | Channel/energy-aware 4th term (`scope_fd_channel_aware`) under existing SNR sweep + new energy-budget scenario | SW-1 | Extend `scope_fd.py` with optional `alpha_channel` term | Cheap–Medium | 5 |
| E12 | Public-dataset sensitivity (identity, size, label noise, domain mismatch) | GN-4 | None (uses existing `--public-dataset*` flags) | Medium | 3–5 |
| E13 | Non-image domain: audio_fsdd ported to FD paradigm | GN-1 | Port existing audio pipeline to `--paradigm fd`; run SCOPE/random/debt-only/DivFL/SubTrunc | Medium–High | 3–5 |
| E14 | FL-selector-in-FD demonstration | BL-2, GN-5 | Port one FL selection policy to call inside the FD client loop | Medium | 3 |
| E16 | Figure 1 regeneration at higher resolution | WR-1 | None — rendering-settings fix in `plot_scope_paper.py` invocation | Trivial | — |
| E17 | Editorial fixes (K\|N definition, spacing, title case) | WR-2 | Manuscript text only | Trivial | — |

(E15, the theory item, has no GPU cost and is tracked separately in §1.1/TH-1 —
it runs in parallel with the whole suite above, not after it.)

---

## 4. Order of Operations

This plan has exactly two broad phases. Everything experimental happens in
**Phase A**; nothing in **Phase B** starts until Phase A has produced real
numbers.

> **Hard gate — read this before touching the manuscript.**
> **Do not edit the manuscript draft (any `.tex`, the Response-to-Reviewers
> letter, or the narrative doc's results sections) until Phase A's experiments
> have actually completed and been aggregated.** Every table, figure, and claim
> in Phase B depends on data that does not exist yet. Drafting prose around
> anticipated results is precisely the "assert now, verify later" pattern that
> got this manuscript a Major Revision in the first place (single-seed
> "statistically tied" / "3× faster" claims). The one narrow exception is
> **A4/TH-1**: the convergence-theory derivation can be worked out on paper or
> in a scratch note during Phase A, since it has no GPU dependency — but even
> that text does **not** get inserted into the manuscript until Phase B opens.
> If a subphase in Phase A stalls or is descoped, the corresponding claim in
> Phase B gets softened or dropped — it does not get written optimistically
> and patched later.

### Phase A — Experimentation

#### A0 — Infrastructure (build once, use everywhere)
1. Register `scope_fd_debt_only` in `presets/methods.yaml` (trivial, unblocks E1 immediately).
2. Build the multi-seed orchestrator + aggregator + CI-plotting stack (§2.3) — everything downstream depends on this for statistical credibility.
3. Add the rounds-to-fixed-absolute-accuracy metric (E5) alongside the existing rounds-to-80%-of-final metric.
4. Implement the Laplace-histogram mechanism and surrogate-histogram selector (E10 code).
5. Implement the dropout injection point and bounded-staleness buffer (E8/E9 code).
6. Implement the channel/energy-aware 4th term (E11 code).
7. Implement the ported literature baselines — DivFL (E2a) and SubTrunc (E2b) committed; UnionFL (E2c) only if it comes free off SubTrunc's scaffolding.
8. Port one FL selector to run inside the FD loop (E14 code).

#### A1 — Cheap, low-risk reruns (small CNNs, N≤50, existing configs)
9. E1 (4-way ablation), E4 (multi-seed pass on existing headline/K-sweep/K=1/channel-sweep configs), E5 (derived metrics from E4's data), E3 (α_u/α_d grid), E6 (Dirichlet-α sweep), E16 (Figure 1 fix).

#### A2 — Medium-cost new scenarios
10. E7 (N/K scale-up + K∤N + rolling-window Gini), E8 (dropout), E9 (bounded-staleness async), E10 (privacy sweep), E11 (channel/energy-aware variant + energy-budget scenario), E12 (public-dataset sensitivity), E2a/E2b/(E2c) (DivFL/SubTrunc/UnionFL baseline runs), E14 (FL-selector-in-FD demo runs).

#### A3 — Higher-cost / new-domain
11. E13 (audio_fsdd ported to FD paradigm) — the most infrastructurally novel item, scheduled last among experiments so A0's harness is battle-tested on cheaper configs first.

#### A4 — Theory (runs in parallel with A1–A3, no GPU dependency)
12. TH-1: attempt the partial-participation convergence theorem; regardless of outcome, work out the assumption-satisfaction discussion and the softened claim language R2.3 explicitly suggests as a fallback, informed by A2's K∤N and non-aligned-window Gini data (E7/GN-6) once available. This produces a standalone analysis note — it is **not** inserted into the manuscript yet; that happens in B1.

**Phase A exit criteria** (all must hold before Phase B opens): every experiment in the Detailed Experiment List (§3) has either completed with aggregated mean±std/CI output, or has been explicitly descoped with a written reason; A4's theory note exists in at least draft form.

---

### Phase B — Manuscript Draft — **locked until Phase A exit criteria are met**

#### B1 — Manuscript integration
13. Rewrite Table (ablation, 4-way), add baseline table (random / DivFL / SubTrunc / [UnionFL] / debt-only / SCOPE-FD), add mean±std and CI bands to every figure, add sensitivity tables (α_u, α_d, Dirichlet-α, K/N), add privacy limitations paragraph backed by E10, rewrite mMIMO positioning paragraph per SW-1's resolution, add dropout/async subsection, add non-image-domain subsection, add K∤N/rolling-window fairness-generality subsection, insert A4/TH-1's convergence rewrite, apply E17's editorial fixes.

#### B2 — QA / honesty pass
14. Re-verify every quantitative claim in the rewritten manuscript against the regenerated mean±std tables (no claim should still cite a single-seed number). Re-check that CIFAR-10/-100 framing stays honest (informational, not headline, consistent with the project's existing discipline of not overselling ties as wins — see the narrative doc's own §9.2 "where SCOPE ties random and why that's fine"). Confirm the channel-aware claim in SW-1 is only as strong as E11's actual measured effect size. Final PDF compile.

---

## Appendix — Cross-reference back to the reviewers

Every numbered reviewer point maps to at least one row above:

- **Reviewer 1** → TH-1, GN-1, RP-1, AB-2, BL-1, GN-2/GN-3/SW-2
- **Reviewer 2** → AB-1, PR-1, TH-1, RP-2, SW-1, WR-2
- **Reviewer 3** → BL-2/GN-5, AB-2, WR-1, GN-3, GN-7, GN-4
- **Attached Review Document** → TH-1, GN-6, RP-3, AB-1, PR-2

No sentence from `reviews.txt` is unaddressed in this matrix.
