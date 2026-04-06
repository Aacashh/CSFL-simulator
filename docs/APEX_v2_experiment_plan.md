# APEX v2: IEEE TAI Experiment Plan

Experiment matrix designed to address every P0/P1 issue from the critical review.
All commands use `--no-fast-mode` (real training), `--track-grad-norm`, and `--device cuda`.

**Key changes from v1 experiments:**
- **200 rounds** (was 50) — addresses reviewer concern about premature evaluation
- **ResNet18** on CIFAR-10/100 (was LightCNN) — stronger model for realistic accuracy
- **3 seeds** per experiment (42, 123, 456) — statistical rigor with mean +/- std
- **PoC + MMR-Diverse** added as baselines — the two missing comparisons flagged as P0
- **Ablation at main benchmark settings** (N=50, not N=100) — fixes the contradictory ablation
- **CIFAR-100** added — broader dataset diversity
- **Scalability to N=500** — beyond the trivial N=100 test

---

## Experiment 1: Main Benchmark (Table I) — CIFAR-10, alpha=0.3

The core comparison table. 8 methods, 3 seeds, 200 rounds, ResNet18.

```bash
# Seed 42
python -m csfl_simulator compare --name main_cifar10_a03_s42 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.tifl,system_aware.poc,heuristic.mmr_diverse,ml.fedcor,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

# Seed 123
python -m csfl_simulator compare --name main_cifar10_a03_s123 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.tifl,system_aware.poc,heuristic.mmr_diverse,ml.fedcor,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

# Seed 456
python -m csfl_simulator compare --name main_cifar10_a03_s456 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.tifl,system_aware.poc,heuristic.mmr_diverse,ml.fedcor,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Table I (main results), Figure 2 (convergence curves)
**Addresses:** P0 #1 (low accuracy), P0 #4 (50 rounds), P0 #5 (missing baselines), P1 #9 (single seed)

---

## Experiment 2: Heterogeneity Robustness — alpha sweep {0.1, 0.3, 0.6}

Tests APEX v2's stability across non-IID regimes. alpha=0.3 is shared with Exp 1 (reuse those runs).

```bash
# alpha=0.1 (extreme non-IID) — 3 seeds
python -m csfl_simulator compare --name het_a01_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name het_a01_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name het_a01_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456

# alpha=0.6 (mild non-IID) — 3 seeds
python -m csfl_simulator compare --name het_a06_s42 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name het_a06_s123 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name het_a06_s456 --methods "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Table II (heterogeneity robustness), Figure 3 (accuracy vs alpha)
**Addresses:** P0 #1 (convergence), Fix 2 (oscillation at alpha=0.1), Fix 3 (overexploration at alpha=0.6)

---

## Experiment 3: Scalability — N in {100, 200, 500}

Tests whether APEX v2's adaptive recency fix (Fix 1) actually resolves the N=100 collapse and scales further.

```bash
# N=100, K=10
python -m csfl_simulator compare --name scale_n100_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name scale_n100_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name scale_n100_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456

# N=200, K=20
python -m csfl_simulator compare --name scale_n200_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 200 --clients-per-round 20 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name scale_n200_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 200 --clients-per-round 20 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name scale_n200_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 200 --clients-per-round 20 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456

# N=500, K=50
python -m csfl_simulator compare --name scale_n500_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 500 --clients-per-round 50 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name scale_n500_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 500 --clients-per-round 50 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name scale_n500_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 500 --clients-per-round 50 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Table III (scalability), Figure 4 (accuracy vs N)
**Addresses:** P2 #12 (scalability only to N=100), Fix 1 (recency scaling)

---

## Experiment 4: Ablation Study — at main benchmark settings

Critical fix: ablation runs at the **same** N=50, K=10 as the main benchmark, not N=100.

```bash
# Seed 42
python -m csfl_simulator compare --name ablation_s42 --methods "ml.apex_v2,ml.apex_v2_no_adaptive_recency,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg,ml.apex_v2_no_adaptive_gamma" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

# Seed 123
python -m csfl_simulator compare --name ablation_s123 --methods "ml.apex_v2,ml.apex_v2_no_adaptive_recency,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg,ml.apex_v2_no_adaptive_gamma" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

# Seed 456
python -m csfl_simulator compare --name ablation_s456 --methods "ml.apex_v2,ml.apex_v2_no_adaptive_recency,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg,ml.apex_v2_no_adaptive_gamma" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Table IV (ablation), Figure 5 (component contribution bar chart)
**Addresses:** P0 #2 (ablation contradicts claims — v2 fixes should show each component is consistently positive)

---

## Experiment 5: Dataset Diversity

Adds CIFAR-100 (harder task) and validates on MNIST/Fashion-MNIST.

```bash
# CIFAR-100 with ResNet18 — 3 seeds
python -m csfl_simulator compare --name cifar100_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-100 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name cifar100_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-100 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name cifar100_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-100 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456

# Fashion-MNIST with CNN-MNIST — 3 seeds
python -m csfl_simulator compare --name fmnist_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset Fashion-MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name fmnist_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset Fashion-MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name fmnist_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset Fashion-MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456

# MNIST with CNN-MNIST — 3 seeds (ceiling test)
python -m csfl_simulator compare --name mnist_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name mnist_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name mnist_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset MNIST --partition dirichlet --dirichlet-alpha 0.3 --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Table V (cross-dataset generalization)
**Addresses:** P2 #13 (limited dataset diversity)

---

## Experiment 6: IID Baseline

Sanity check: APEX v2 should not hurt performance on IID data.

```bash
python -m csfl_simulator compare --name iid_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition iid --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name iid_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition iid --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name iid_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2" --dataset CIFAR-10 --partition iid --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Row in Table II (heterogeneity robustness, alpha=inf equivalent)
**Addresses:** Fix 3 validation (het-aware diversity should set w_div near 0 on IID)

---

## Experiment 7: v1 vs v2 Head-to-Head

Direct comparison of APEX v1 and v2 to quantify the improvement from the five fixes.

```bash
python -m csfl_simulator compare --name v1v2_a03_s42 --methods "ml.apex,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name v1v2_a01_s42 --methods "ml.apex,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name v1v2_a06_s42 --methods "ml.apex,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name v1v2_n100_s42 --methods "ml.apex,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42
```

**Paper artifact:** Figure 6 (v1 vs v2 convergence curves across settings)
**Addresses:** Validates all five fixes empirically

---

## Experiment 8: Convergence Speed — LightCNN comparison

Run the main benchmark with LightCNN too, to show model-independence and compare with v1 results.

```bash
python -m csfl_simulator compare --name lightcnn_a03_s42 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 42

python -m csfl_simulator compare --name lightcnn_a03_s123 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 123

python -m csfl_simulator compare --name lightcnn_a03_s456 --methods "baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 200 --no-fast-mode --track-grad-norm --device cuda --seed 456
```

**Paper artifact:** Supplementary table showing model-independence
**Addresses:** Validates that improvements aren't model-specific

---

## Plotting Commands

After experiments complete, generate IEEE-ready figures:

```bash
# Main benchmark convergence curves
python -m csfl_simulator plot --run main_cifar10_a03_s42 --metrics accuracy,loss,fairness_gini --format eps

# Heterogeneity comparison
python -m csfl_simulator plot --run het_a01_s42 --metrics accuracy,loss --format eps
python -m csfl_simulator plot --run het_a06_s42 --metrics accuracy,loss --format eps

# Scalability
python -m csfl_simulator plot --run scale_n100_s42 --metrics accuracy --format eps
python -m csfl_simulator plot --run scale_n200_s42 --metrics accuracy --format eps
python -m csfl_simulator plot --run scale_n500_s42 --metrics accuracy --format eps

# Ablation
python -m csfl_simulator plot --run ablation_s42 --metrics accuracy,fairness_gini --format eps

# Cross-dataset
python -m csfl_simulator plot --run cifar100_s42 --metrics accuracy --format eps
python -m csfl_simulator plot --run fmnist_s42 --metrics accuracy --format eps

# v1 vs v2
python -m csfl_simulator plot --run v1v2_a03_s42 --metrics accuracy,loss --format eps
python -m csfl_simulator plot --run v1v2_n100_s42 --metrics accuracy --format eps
```

---

## Experiment-to-Paper Mapping

| Paper Section | Experiment | Review Issue Addressed |
|---------------|------------|----------------------|
| Table I: Main Results | Exp 1 (3 seeds) | P0 #1 (accuracy), P0 #4 (rounds), P0 #5 (baselines), P1 #9 (seeds) |
| Table II: Heterogeneity | Exp 2 + Exp 6 | Fix 2 (oscillation), Fix 3 (overexploration) |
| Table III: Scalability | Exp 3 | P2 #12 (scalability), Fix 1 (recency) |
| Table IV: Ablation | Exp 4 (3 seeds) | P0 #2 (ablation contradicts claims) |
| Table V: Cross-Dataset | Exp 5 | P2 #13 (dataset diversity) |
| Figure 2: Convergence | Exp 1 | P0 #4 (show plateau, not early dynamics) |
| Figure 3: Alpha Sweep | Exp 2 | Robustness narrative |
| Figure 4: Scalability | Exp 3 | N=50 to N=500 trend |
| Figure 5: Ablation | Exp 4 | Each component is net-positive |
| Figure 6: v1 vs v2 | Exp 7 | Principled refinement narrative |

---

## Total Experiment Count

| Experiment | Runs | Est. GPU-hours (each ~15-40 min at 200 rounds) |
|------------|------|------------------------------------------------|
| Exp 1: Main benchmark | 3 | ~2-3h |
| Exp 2: Heterogeneity | 6 | ~3-5h |
| Exp 3: Scalability | 9 | ~6-12h (N=500 is slow) |
| Exp 4: Ablation | 3 | ~2-3h |
| Exp 5: Cross-dataset | 9 | ~4-6h |
| Exp 6: IID | 3 | ~1-2h |
| Exp 7: v1 vs v2 | 4 | ~1-2h |
| Exp 8: LightCNN | 3 | ~1-2h |
| **Total** | **40** | **~20-35h** |

---

## Review Issues NOT Addressed by Experiments (need code/paper changes)

| Issue | Action Required |
|-------|----------------|
| P0 #3: Theoretical claims without proofs | Write formal proof or downgrade to "discussion" |
| P1 #7: Composite score is non-standard | Report accuracy/Gini/convergence separately; drop DP term |
| P1 #8: Privacy gap (label histograms) | Add noisy histogram experiment (requires code change to inject Laplace noise into histograms before APEX sees them) |
| P2 #6: "Zero parameters" misleading | Reframe as "no backpropagation overhead" in paper text |
| P2 #10: Reward design (equal credit) | Discuss credit assignment in paper; optionally implement per-client reward ablation |
| P2 #11: Phase thresholds dataset-dependent | Run sensitivity sweep over tau_c, tau_u, tau_e (requires parameterizing these in methods.yaml) |
| P3 #14: Section ordering | Fix in LaTeX |
| P3 #19: Reproducibility | Add code URL, full hyperparameter table, optimizer details |
