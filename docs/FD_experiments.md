# Client Selection for Federated Distillation --- Experiment Protocol

**Thesis:** Client selection methods designed for FL transfer to FD but with shifted performance rankings. Channel-aware and diversity-aware methods gain disproportionate advantage because FD's failure mode (corrupted logits via channel noise) is qualitatively different from FL's (stale weights). Smart selection of K < N clients can even beat full participation.

**Reference system model:** Mu et al., "Federated Distillation in Massive MIMO Networks," IEEE TCCN, vol. 10, no. 4, Aug 2024.

---

## Methods Under Evaluation (8)

| ID | Key | Origin | Core Signal | FD Hypothesis |
|----|-----|--------|-------------|---------------|
| M1 | `baseline.fedavg` | McMahan 2017 | Random | Baseline minimum bar |
| M2 | `system_aware.fedcs` | Nishio 2019 | loss / time | Expected to fail: selects high-loss clients whose logits may be channel-corrupted |
| M3 | `system_aware.tifl` | Chai 2020 | Tier round-robin | Structural diversity, no quality signal |
| M4 | `ml.fedcor` | Tang 2022 | Label cosine similarity | Penalises correlated logits — directly addresses FD diversity need |
| M5 | `ml.maml_select` | Custom | Learned MLP (6 features) | Adapts but no explicit channel awareness |
| M6 | `ml.apex_v2` | Custom | Thompson + phase + diversity proxy | Detects anomalous logits; hysteresis prevents chasing noise — expected winner |
| M7 | `system_aware.oort` | Lai 2021 | UCB + loss/time | UCB exploration prevents fixating on noisy clients |
| M8 | `heuristic.label_coverage` | Custom | Greedy label IDF coverage | Guarantees per-class logit representation — critical for FD aggregation |

**8-method string** (used in every command):
```
"baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage"
```

---

## Paper-Aligned Parameters (Section VI of Mu et al.)

| Parameter | CIFAR-10 + STL-10 | MNIST + FMNIST |
|-----------|-------------------|----------------|
| Private data | 50 000 (CIFAR-10) | 3 000 (MNIST) |
| Public data | STL-10, 2 000 imgs | FMNIST, 2 000 imgs |
| Partition | Dirichlet alpha=0.5 | Dirichlet alpha=0.5 |
| Optimiser | Adam eta=0.001 | Adam eta=0.001 |
| Batch (train / distill) | 128 / 500 | 20 / 500 |
| Epochs (both phases) | 2 | 2 |
| Quantisation | 8-bit uniform | 8-bit uniform |
| N_BS / N_D | 64 / 1 | 64 / 1 |
| UL SNR / DL SNR | -8 dB / -20 dB | -8 dB / -20 dB |
| Model heterogeneity | FD-CNN1 / FD-CNN2 / FD-CNN3 | FD-CNN1/2/3 |
| Dynamic steps | base=5, period=25 | base=5, period=25 |

**Client selection adaptation:** The paper uses full participation (K=N=15). We set **N=30, K=10** (33%) as the default to make selection meaningful, and sweep K in Experiment 5.

---

## Method-Colour-Marker Mapping (consistent across ALL figures)

| Method | Colour | Hex | Marker | Dash |
|--------|--------|-----|--------|------|
| FedAvg | Blue | #1f77b4 | o | solid |
| FedCS | Red | #d62728 | s | dashed |
| TiFL | Green | #2ca02c | ^ | dotted |
| FedCor | Purple | #9467bd | D | dashdot |
| MAML | Brown | #8c564b | v | short dash |
| APEX v2 | Pink | #e377c2 | * | dashdotdot |
| Oort | Cyan | #17becf | P | long dash |
| LabelCov | Orange | #ff7f0e | X | dash-dot-dot |

---

## Experiment 1 --- FL Baseline Ranking (CIFAR-10)

**Story role:** Establishes FL performance rankings. The reader's mental anchor --- every FD experiment is compared back to this.

**Run:**
```powershell
python -m csfl_simulator compare --name exp01_fl_baseline --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model LightCNN --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --lr 0.01 --no-fast-mode --seed 42
```

**Plot:**
```powershell
python scripts/plot_fd_experiments.py --run exp01_fl_baseline --metrics accuracy,loss,f1,fairness_gini --format eps --smooth 10 --bar --out-dir paper/figures/exp01
```

**Figures:** Fig. 1 (FL accuracy + loss convergence, 8 methods), Table I (FL final accuracy ranking).

**Expected:** Oort and APEX v2 top FL performers; FedCS reasonable; Label Coverage mid-tier.

---

## Experiment 2 --- FD Main Comparison (CIFAR-10 + STL-10) [HEADLINE]

**Story role:** The central experiment. Reveals the new FD ranking. Direct comparison with Exp 1 demonstrates the ranking shift thesis.

**Run:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp02_fd_main --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp02_fd_main --metrics server_accuracy,accuracy,kl_divergence_avg,effective_noise_var,client_accuracy_std --format eps --smooth 10 --bar --out-dir paper/figures/exp02
```
```powershell
python scripts/plot_fd_experiments.py --run exp02_fd_main --metrics loss,distillation_loss_avg,comm_reduction_ratio,f1 --format eps --smooth 10 --out-dir paper/figures/exp02_supp
```

**Figures:** Fig. 2 (4-panel: accuracy, KL div, noise var, client std), Fig. 3 (final accuracy bar), Table II (accuracy, rounds_to_80pct, comm_reduction).

**Expected:** Ranking shifts --- FedCS drops (vicious cycle: high loss -> selected -> corrupted logits -> higher loss). Label Coverage and FedCor rise. APEX v2 expected top due to diversity proxy + phase hysteresis.

---

## Experiment 3 --- Ranking Shift Cross-Validation (MNIST + FMNIST)

**Story role:** Proves the ranking shift on a second dataset. Also serves as the required FL-alongside-FD comparison.

**FL run:**
```powershell
python -m csfl_simulator compare --name exp03_fl_mnist --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset MNIST --partition dirichlet --dirichlet-alpha 0.5 --model CNN-MNIST --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 20 --lr 0.01 --no-fast-mode --seed 42
```

**FD run:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp03_fd_mnist --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset MNIST --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --public-dataset FMNIST --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 20 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --track-grad-norm --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp03_fl_mnist --metrics accuracy,loss --format eps --smooth 10 --bar --out-dir paper/figures/exp03/fl
```
```powershell
python scripts/plot_fd_experiments.py --run exp03_fd_mnist --metrics accuracy,loss,comm_reduction_ratio --format eps --smooth 10 --bar --out-dir paper/figures/exp03/fd
```

**Figures:** Fig. 4 (FL vs FD accuracy side-by-side on MNIST), Table III (Spearman rank correlation FL vs FD).

**Expected:** Same ranking shift pattern as CIFAR-10. Communication reduction confirms ~1% overhead.

---

## Experiment 4 --- Channel Noise Sensitivity (DL SNR sweep)

**Story role:** Proves that the ranking shift is CAUSED by channel noise degrading logits. As noise increases, the gap between channel-aware and channel-unaware methods widens. This is the mechanistic explanation.

**Error-free (no channel noise):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp04_errfree --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**DL SNR = 0 dB:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp04_dl0db --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db 0 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**DL SNR = -20 dB (most error-prone):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp04_dl-20db --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp04_errfree --metrics accuracy,effective_noise_var --format eps --smooth 10 --bar --out-dir paper/figures/exp04/errfree
```
```powershell
python scripts/plot_fd_experiments.py --run exp04_dl0db --metrics accuracy,effective_noise_var --format eps --smooth 10 --bar --out-dir paper/figures/exp04/dl0db
```
```powershell
python scripts/plot_fd_experiments.py --run exp04_dl-20db --metrics accuracy,effective_noise_var --format eps --smooth 10 --bar --out-dir paper/figures/exp04/dl-20db
```

**Figures:** Fig. 5 (3-panel accuracy, one per noise level), Fig. 6 (accuracy degradation grouped bars), Table IV (delta accuracy error-free minus -20 dB).

**Expected:** Error-free: rankings close to FL. -20 dB: full ranking shift, diversity-aware methods dominate. FedCS drops the most. This proves the causal mechanism.

---

## Experiment 5 --- Participation Rate Scaling (K sweep)

**Story role:** Shows smart selection of K=10 can beat full participation K=30. The paper's Fig. 9 shows accuracy saturates at N>15 due to per-user SNR degradation. Our headline practical result.

**K=5:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp05_k5 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 5 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**K=10:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp05_k10 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**K=15:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp05_k15 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 15 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**K=30 (full participation):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp05_k30 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 30 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp05_k5 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp05/k5
```
```powershell
python scripts/plot_fd_experiments.py --run exp05_k10 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp05/k10
```
```powershell
python scripts/plot_fd_experiments.py --run exp05_k15 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp05/k15
```
```powershell
python scripts/plot_fd_experiments.py --run exp05_k30 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp05/k30
```

**Figures:** Fig. 7 (final accuracy vs K/N ratio, one line per method).

**Expected:** K=30 all methods equivalent (no selection). APEX v2 at K=10 exceeds FedAvg at K=30 --- the headline result.

---

## Experiment 6 --- Non-IID Sensitivity (alpha sweep)

**Story role:** Shows the advantage of diversity-aware selectors is proportional to data heterogeneity.

**alpha=0.1 (extreme non-IID):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp06_a01 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**alpha=0.5 (default):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp06_a05 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**alpha=1.0:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp06_a1 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 1.0 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**alpha=10.0 (near IID):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp06_a10 --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 10.0 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp06_a01 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp06/a01
```
```powershell
python scripts/plot_fd_experiments.py --run exp06_a05 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp06/a05
```
```powershell
python scripts/plot_fd_experiments.py --run exp06_a1 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp06/a1
```
```powershell
python scripts/plot_fd_experiments.py --run exp06_a10 --metrics accuracy --format eps --smooth 10 --bar --out-dir paper/figures/exp06/a10
```

**Figures:** Fig. 8 (4-panel accuracy, one per alpha), Table V (final accuracy across alpha).

**Expected:** alpha=0.1: LabelCov and FedCor dominate. alpha=10: all methods converge. Selection advantage proportional to heterogeneity.

---

## Experiment 7 --- Group-Based FD (FedTSKD-G)

**Story role:** Shows whether channel-aware aggregation (Algorithm 2) complements or substitutes for smart selection. If it complements, selection + grouping gives the best result.

**Without grouping:**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp07_nogroup --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**With grouping (FedTSKD-G):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp07_group --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --group-based --channel-threshold 0.5 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp07_nogroup --metrics accuracy,kl_divergence_avg --format eps --smooth 10 --bar --out-dir paper/figures/exp07/nogroup
```
```powershell
python scripts/plot_fd_experiments.py --run exp07_group --metrics accuracy,kl_divergence_avg --format eps --smooth 10 --bar --out-dir paper/figures/exp07/group
```

**Figures:** Fig. 9 (side-by-side accuracy with/without grouping), Table VI (grouping benefit per method).

**Expected:** Grouping helps channel-unaware methods MORE. APEX v2 already partially mitigates the problem, so it gains less from grouping. Best overall: APEX v2 + grouping.

---

## Experiment 8 --- Model Heterogeneity Impact

**Story role:** Isolates the effect of model heterogeneity (Table IV insight). Methods with quality signals can avoid over-sampling weak-model clients.

**Homogeneous (all FD-CNN1):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp08_homo --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --track-grad-norm --no-fast-mode --seed 42
```

**Heterogeneous (FD-CNN1/FD-CNN2/FD-CNN3):**
```powershell
python -m csfl_simulator compare --paradigm fd --name exp08_hetero --methods "baseline.fedavg,system_aware.fedcs,system_aware.tifl,ml.fedcor,ml.maml_select,ml.apex_v2,system_aware.oort,heuristic.label_coverage" --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 --model FD-CNN1 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" --track-grad-norm --public-dataset STL-10 --public-dataset-size 2000 --total-clients 30 --clients-per-round 10 --rounds 200 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --fd-optimizer adam --channel-noise --ul-snr-db -8 --dl-snr-db -20 --n-bs-antennas 64 --quantization-bits 8 --no-fast-mode --seed 42
```

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp08_homo --metrics accuracy,client_accuracy_std --format eps --smooth 10 --bar --out-dir paper/figures/exp08/homo
```
```powershell
python scripts/plot_fd_experiments.py --run exp08_hetero --metrics accuracy,client_accuracy_std --format eps --smooth 10 --bar --out-dir paper/figures/exp08/hetero
```

**Figures:** Fig. 10 (homo vs hetero side-by-side), Table VII (degradation per method).

**Expected:** Heterogeneity degrades all methods 3-8%. Methods with loss/quality signals (APEX v2, MAML, Oort) degrade less.

---

## Experiment 9 --- Communication Efficiency (reuses Exp 1 + 2)

**Story role:** Quantifies FD vs FL communication advantage. No new runs needed.

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp02_fd_main --metrics cum_comm,logit_comm_kb,comm_reduction_ratio --format eps --smooth 10 --out-dir paper/figures/exp09
```
```powershell
python scripts/plot_fd_experiments.py --run exp01_fl_baseline --metrics cum_comm --format eps --smooth 10 --out-dir paper/figures/exp09/fl
```

**Figures:** Fig. 11 (cumulative comm FL vs FD), Table VIII (per-round and total communication).

**Expected:** FD overhead is ~1% of FL. Even random FD beats smart FL on communication.

---

## Experiment 10 --- Fairness and Selection Dynamics (reuses Exp 2)

**Story role:** Shows fairness implications. Some selectors achieve accuracy by always picking the same clients.

**Plots:**
```powershell
python scripts/plot_fd_experiments.py --run exp02_fd_main --metrics fairness_gini,client_accuracy_std,num_good_channel --format eps --smooth 10 --out-dir paper/figures/exp10
```

**Figures:** Fig. 12 (fairness Gini + client std + channel selection), Table IX (accuracy-fairness Pareto).

**Expected:** FedCS/Oort less fair (greedy). LabelCov/TiFL more fair (structural diversity). APEX v2 Pareto-optimal: high accuracy + moderate fairness.

---

## Run Summary

| Exp | Name | Paradigm | Runs |
|-----|------|----------|------|
| 1 | exp01_fl_baseline | FL | 1 |
| 2 | exp02_fd_main | FD | 1 |
| 3 | exp03_fl_mnist, exp03_fd_mnist | FL + FD | 2 |
| 4 | exp04_errfree/dl0db/dl-20db | FD | 3 |
| 5 | exp05_k5/k10/k15/k30 | FD | 4 |
| 6 | exp06_a01/a05/a1/a10 | FD | 4 |
| 7 | exp07_nogroup/group | FD | 2 |
| 8 | exp08_homo/hetero | FD | 2 |
| 9 | (reuses 1+2) | -- | 0 |
| 10 | (reuses 2) | -- | 0 |
| **Total** | | | **19** |

**Statistical rigour:** Repeat with seeds 42, 123, 7 (57 total). Report mean +/- std.

**Estimated time:** ~36-72 hours on a single GPU (RTX 3060+).

---

## Paper Figure Map

| Figure | Exp | Content |
|--------|-----|---------|
| Fig. 1 | 1 | FL accuracy + loss (8 methods) |
| Fig. 2 | 2 | FD 4-panel: accuracy, KL div, noise var, client std |
| Fig. 3 | 2 | FD final accuracy bar chart |
| Fig. 4 | 3 | FL vs FD ranking shift (MNIST) |
| Fig. 5 | 4 | Channel noise 3-panel accuracy |
| Fig. 6 | 4 | Accuracy degradation grouped bars |
| Fig. 7 | 5 | Final accuracy vs K/N ratio |
| Fig. 8 | 6 | Non-IID alpha 4-panel |
| Fig. 9 | 7 | With/without FedTSKD-G grouping |
| Fig. 10 | 8 | Homo vs hetero models |
| Fig. 11 | 9 | Communication FL vs FD |
| Fig. 12 | 10 | Fairness + selection dynamics |

| Table | Exp | Content |
|-------|-----|---------|
| I | 1 | FL final accuracy ranking |
| II | 2 | FD accuracy, rounds_to_80pct, comm_reduction |
| III | 3 | Spearman rank correlation FL vs FD |
| IV | 4 | Accuracy delta by noise level |
| V | 6 | Accuracy across alpha |
| VI | 7 | Grouping benefit per method |
| VII | 8 | Heterogeneity degradation |
| VIII | 9 | Communication totals |
| IX | 10 | Accuracy-fairness Pareto |
