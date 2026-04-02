# CSFL Simulator

Client Selection for Federated Learning simulator — a research platform for comparing client selection algorithms under realistic heterogeneity (data, system, privacy). Supports both **Federated Learning (FL)** and **Federated Distillation (FD)** paradigms.

## Quick Start

```bash
# List all 37+ selection methods
python -m csfl_simulator list-methods

# Run a single simulation with a name for easy retrieval
python -m csfl_simulator run --name baseline_mnist --method heuristic.random --dataset MNIST --rounds 10

# Compare methods on the SAME data partition (fair comparison)
python -m csfl_simulator compare --name noniid_study \
    --methods "heuristic.random,system_aware.oort,ml.bandit.epsilon_greedy" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 20

# Generate IEEE-ready EPS plots from any run
python -m csfl_simulator plot --run noniid_study --metrics accuracy,loss,f1 --format eps

# List all saved runs
python -m csfl_simulator list-runs

# Streamlit UI
streamlit run csfl_simulator/app/main.py

# Batch matrix run (all methods x grid of params)
python scripts/run_matrix.py --dataset CIFAR10 --rounds 30 --fast

# --- Federated Distillation (FD) ---

# Basic FD run (logits exchanged instead of weights)
python -m csfl_simulator run --paradigm fd --method heuristic.random --dataset MNIST --rounds 50

# FD with model heterogeneity (different client architectures)
python -m csfl_simulator run --paradigm fd --method system_aware.oort \
    --dataset CIFAR-10 --model-heterogeneous --model-pool "FD-CNN1,FD-CNN2,FD-CNN3" \
    --public-dataset STL-10 --rounds 100 --total-clients 15 --clients-per-round 15

# FD with mMIMO channel noise
python -m csfl_simulator run --paradigm fd --method heuristic.random \
    --dataset MNIST --channel-noise --ul-snr-db -8 --dl-snr-db -20

# FD with group-based scheme (FedTSKD-G)
python -m csfl_simulator run --paradigm fd --method heuristic.random \
    --dataset CIFAR-10 --channel-noise --group-based --rounds 200

# Compare selection methods under FD
python -m csfl_simulator compare --paradigm fd \
    --methods "heuristic.random,system_aware.oort,ml.apex_v2" \
    --dataset CIFAR-10 --public-dataset STL-10 --rounds 50

# Plot FD-specific metrics
python -m csfl_simulator plot --run fd_experiment \
    --metrics accuracy,kl_divergence_avg,comm_reduction_ratio
```

## Project Structure

```
csfl_simulator/
├── __main__.py              # CLI entry point (run/compare/plot/list-methods/list-runs)
├── core/
│   ├── simulator.py         # FLSimulator + SimConfig — the main engine
│   ├── fd_simulator.py      # FDSimulator — Federated Distillation engine (FedTSKD/FedTSKD-G)
│   ├── fd_aggregation.py    # Logit averaging + group-based aggregation
│   ├── channel.py           # MIMOChannel — mMIMO uplink/downlink noise simulation
│   ├── datasets.py          # Dataset loading (MNIST, Fashion-MNIST, CIFAR-10/100, STL-10)
│   ├── partition.py         # iid, dirichlet, label-shard + size distribution
│   ├── client.py            # ClientInfo dataclass (all per-client state)
│   ├── models.py            # CNN-MNIST, CNNMnistFedAvg, LightCIFAR, ResNet18, FD-CNN1/2/3
│   ├── aggregation.py       # FedAvg weighted aggregation
│   ├── metrics.py           # eval_model(), eval_fd_clients()
│   ├── system.py            # System heterogeneity simulation (speed, channel, energy)
│   ├── parallel.py          # CUDA parallel client training via streams
│   ├── dp.py                # Differential privacy (clipping, noise, accounting)
│   └── utils.py             # Seed, device, run dirs, memory cleanup
├── selection/
│   ├── interface.py         # Canonical function signature
│   ├── registry.py          # MethodRegistry — dynamic YAML-based method loader
│   ├── common.py            # Shared utilities (normalize, recency, label_entropy)
│   ├── baseline/            # FedAvg (uniform random)
│   ├── heuristic/           # 10+ heuristic selectors
│   ├── system_aware/        # Oort, FedCS, TiFL, TriBudget, etc.
│   ├── ml/                  # Bandits, GNN, GAT, Transformer, MAML, etc.
│   └── wrappers/            # Selector wrappers (quota safety)
├── app/
│   ├── main.py              # Streamlit UI (~1500 lines)
│   ├── state.py             # Session state / snapshot management
│   ├── export.py            # Jupyter notebook export
│   └── components/          # Plot components
├── presets/
│   ├── methods.yaml         # All registered selection methods
│   ├── datasets.yaml        # Dataset configs
│   └── models.yaml          # Model-dataset compatibility
└── scripts/
    ├── run_matrix.py         # Batch experiment runner
    ├── export_runs_to_csv.py # Export results
    └── download_data.py      # Pre-download datasets
```

## Key Concepts

### SimConfig
All simulation parameters live in `SimConfig` (dataclass in `core/simulator.py`). Every CLI flag maps 1:1 to a field. Key fields:
- `name`: Optional run name (creates `artifacts/runs/<name>_<timestamp>/` folder)
- `dataset`: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- `partition`: iid, dirichlet, label-shard
- `model`: CNN-MNIST, CNN-MNIST (FedAvg), LightCNN, ResNet18
- `total_clients`, `clients_per_round`, `rounds`, `local_epochs`
- `size_distribution`: uniform, lognormal, power_law (data quantity heterogeneity)
- `time_budget`, `energy_budget`, `bytes_budget` (system-aware constraints)
- `dp_sigma`, `dp_epsilon_per_round`, `dp_clip_norm` (differential privacy)
- `parallel_clients`: 0=sequential, -1=auto, N=fixed parallelism
- `fast_mode`: True breaks training after 2 batches (for quick iteration)

### Selection Method Interface
Every selection method is a Python module with this function:
```python
def select_clients(round_idx: int, K: int, clients: List[ClientInfo],
                   history: Dict, rng, time_budget=None, device=None,
                   **kwargs) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    # Returns: (selected_client_ids, per_client_scores, state_update_dict)
```

### Adding a New Method
1. Create a module in `selection/heuristic/`, `selection/system_aware/`, or `selection/ml/`
2. Implement `select_clients()` matching the interface above
3. Register in `presets/methods.yaml`:
```yaml
  - key: heuristic.my_method
    module: csfl_simulator.selection.heuristic.my_method
    display_name: "My Method"
    origin: custom
    params: {}
    type: heuristic
    trainable: false
```
4. The method is immediately available in CLI and Streamlit UI.

### ClientInfo Fields (available to selectors)
`id`, `data_size`, `last_loss`, `grad_norm`, `compute_speed`, `channel_quality`, `estimated_duration`, `energy_rate`, `estimated_energy`, `estimated_bytes`, `tier`, `dp_epsilon_remaining`, `dp_epsilon_used`, `participation_count`, `last_selected_round`, `label_histogram`, `utility_estimate`

### FLSimulator Usage (Programmatic)
```python
from csfl_simulator.core.simulator import FLSimulator, SimConfig

cfg = SimConfig(dataset="CIFAR-10", partition="dirichlet", dirichlet_alpha=0.3,
                model="LightCNN", total_clients=50, clients_per_round=10, rounds=20)
sim = FLSimulator(cfg)
sim.setup()  # Partition data, init model (call once)

# Compare methods on the SAME partition
result_a = sim.run("heuristic.random")     # resets model + state internally
result_b = sim.run("system_aware.oort")    # same partition, same initial model

# result["metrics"] is a list of dicts per round with:
#   accuracy, loss, f1, wall_clock, cum_tflops, cum_comm, fairness_gini, composite, ...
```

## CLI Reference

### Named Runs
Use `--name` to give runs human-readable names. Results are saved to `artifacts/runs/<name>_<timestamp>/`:
```bash
python -m csfl_simulator run --name oort_cifar10_alpha03 --method system_aware.oort --dataset CIFAR-10 --rounds 20
python -m csfl_simulator compare --name method_shootout --methods "heuristic.random,system_aware.oort" --dataset MNIST
```
Later access results by name prefix:
```bash
python -m csfl_simulator plot --run oort_cifar10_alpha03 --format eps
python -m csfl_simulator list-runs
```

### Full run example
All `SimConfig` fields are available as CLI flags (use `--` prefix, replace `_` with `-`):
```bash
python -m csfl_simulator run \
    --name my_experiment \
    --method system_aware.oort \
    --dataset CIFAR-10 \
    --partition dirichlet --dirichlet-alpha 0.3 \
    --model LightCNN \
    --total-clients 100 --clients-per-round 10 \
    --rounds 50 \
    --local-epochs 2 \
    --batch-size 64 --lr 0.01 \
    --time-budget 10.0 \
    --seed 42 \
    --no-fast-mode \
    --device auto
```

### Generating IEEE plots (EPS/PDF)
```bash
# Generate EPS plots from a named run (IEEE single-column width by default)
python -m csfl_simulator plot --run my_experiment --metrics accuracy,loss,f1 --format eps

# PDF format, custom size for IEEE double-column
python -m csfl_simulator plot --run my_experiment --format pdf --width 7.0 --height 3.5

# PNG for quick preview
python -m csfl_simulator plot --run my_experiment --format png --dpi 150

# Plot from a compare run — generates per-method overlays
python -m csfl_simulator plot --run method_shootout --metrics accuracy,f1,fairness_gini --format eps

# Custom output directory
python -m csfl_simulator plot --run my_experiment --format eps --out-dir ./paper/figures/
```
Plot defaults: EPS format, 300 DPI, 3.5x2.6 inches (IEEE single-column), serif font, classic style.
Generates individual metric plots + a multi-panel figure when multiple metrics are requested.

### Listing runs
```bash
python -m csfl_simulator list-runs  # Shows name, method, dataset, accuracy, path
```

## Method Categories

| Category | Keys (prefix) | Examples |
|----------|--------------|----------|
| Baseline | `baseline.` | `baseline.fedavg` |
| Heuristic | `heuristic.` | `random`, `topk_loss`, `gradient_norm`, `fairness_adjusted`, `label_coverage`, `mmr_diverse`, `dp_budget_aware`, `round_robin`, `cluster_balanced`, `proportional_data` |
| System-Aware | `system_aware.` | `fedcs`, `oort`, `tifl`, `oort_plus`, `poc`, `tribudget`, `fedcs_energy`, `oort_energy` |
| ML-Based | `ml.` | `bandit.epsilon_greedy`, `bandit.linucb`, `bandit.rff_linucb`, `rl_gnn`, `gat`, `gt_ppcs`, `neural_linear_ucb`, `deepset_ranker`, `rankformer_tiny`, `meta_ranker`, `maml_select`, `fedcor`, `dp_eig`, `gnn_dpp`, `pareto_rl`, `ucb_grad`, `fedcluster_plus` |

## Model-Dataset Compatibility

| Model | Compatible Datasets |
|-------|-------------------|
| CNN-MNIST | MNIST, Fashion-MNIST |
| CNN-MNIST (FedAvg) | MNIST |
| LightCNN | CIFAR-10, CIFAR-100 |
| ResNet18 | CIFAR-10, CIFAR-100 |
| FD-CNN1 (~545K params) | MNIST, Fashion-MNIST (FD heterogeneity) |
| FD-CNN2 (~102K params) | MNIST, Fashion-MNIST (FD heterogeneity) |
| FD-CNN3 (~68K params) | MNIST, Fashion-MNIST (FD heterogeneity) |
| ResNet18-FD (~11.2M params) | CIFAR-10, CIFAR-100 (FD heterogeneity) |
| MobileNetV2-FD (~2.2M params) | CIFAR-10, CIFAR-100 (FD heterogeneity) |
| ShuffleNetV2-FD (~350K params) | CIFAR-10, CIFAR-100 (FD heterogeneity) |

## Output Metrics (per round)

Core: `accuracy`, `loss`, `f1`, `precision`, `recall`
Timing: `selection_time`, `compute_time`, `round_time`, `wall_clock`
Efficiency: `cum_tflops`, `cum_comm` (MB), `training_tflops`, `total_tflops`
Fairness: `fairness_var`, `fairness_gini`
Privacy: `dp_used_avg`
Composite: `composite` (weighted combo of acc/time/fairness/dp)
Convergence (final round only): `time_to_50/80/90pct_final`, `auc_acc_time_norm`, `acc_gain_per_hour`

### FD-Specific Metrics (paradigm=fd only)
Distillation: `kl_divergence_avg`, `distillation_loss_avg`
Communication: `logit_comm_kb`, `fl_equiv_comm_mb`, `comm_reduction_ratio`
Channel: `effective_noise_var`, `num_good_channel`, `num_bad_channel`
Training: `dynamic_steps_kr`
Heterogeneity: `client_accuracy_avg`, `client_accuracy_std`

## Artifacts

Results are saved to `artifacts/runs/<name>_<timestamp>/` containing:
- `config.json` — simulation configuration
- `metrics.json` — per-round metrics
- `results.json` — full results (single run) or `compare_results.json` (comparison)
- `plots/` — generated plots (created by `plot` command)

Datasets download to `data/`.

## Federated Distillation (FD) Mode

FD is a parallel training paradigm enabled with `--paradigm fd`. Based on "Federated Distillation in Massive MIMO Networks" (Mu et al., IEEE TCCN 2024).

### Key Differences from FL
- **Exchange logits** (soft predictions) instead of model weights (~1% communication overhead)
- **Model heterogeneity**: different clients can have different architectures
- **Public dataset**: shared unlabeled dataset for logit generation
- **KL divergence distillation**: clients learn from aggregated logits via KL loss
- **Dynamic training steps**: K_r decreases over rounds (FedTSKD)
- **Channel-aware groups**: FedTSKD-G splits clients by channel quality

### FD CLI Flags
```
--paradigm fd                   # Enable FD mode
--public-dataset STL-10         # Public dataset (same, STL-10, FMNIST)
--public-dataset-size 2000      # Number of public samples
--distillation-epochs 2         # Distillation steps per round
--distillation-batch-size 500   # Distillation batch size
--temperature 1.0               # KL divergence temperature
--distillation-lr 0.001         # Adam learning rate for distillation
--dynamic-steps / --no-dynamic-steps  # Dynamic training steps
--dynamic-steps-base 5          # Initial step multiplier
--dynamic-steps-period 25       # Rounds per decrease
--model-heterogeneous           # Enable per-client architecture variation
--model-pool "FD-CNN1,FD-CNN2,FD-CNN3"  # Model architectures to cycle
--channel-noise                 # Enable mMIMO channel noise
--n-bs-antennas 64              # Base station antennas
--ul-snr-db -8.0                # Uplink SNR (dB)
--dl-snr-db -20.0               # Downlink SNR (dB)
--quantization-bits 8           # Logit quantization
--group-based                   # Enable FedTSKD-G
--channel-threshold 0.5         # Good/bad channel threshold
--fd-optimizer adam              # Optimizer (adam or sgd)
```

### FD Round Structure (Algorithm 1: FedTSKD)
1. Client selection (same interface as FL — all methods work)
2. Local distillation: clients learn from previously received logits via KL divergence
3. Local training: K_r SGD/Adam steps on private data (K_r decreases over rounds)
4. Inference: clients predict on public dataset, producing logits
5. Uplink: logits transmitted to server (with optional mMIMO noise)
6. Server aggregation: weighted logit averaging
7. Server distillation: server model learns from aggregated logits
8. Downlink: server broadcasts processed logits to clients (with optional noise)

### FDSimulator Usage (Programmatic)
```python
from csfl_simulator.core.fd_simulator import FDSimulator
from csfl_simulator.core.simulator import SimConfig

cfg = SimConfig(
    paradigm="fd",
    dataset="CIFAR-10", partition="dirichlet", dirichlet_alpha=0.3,
    model="FD-CNN1", model_heterogeneous=True,
    model_pool="FD-CNN1,FD-CNN2,FD-CNN3",
    public_dataset="STL-10", public_dataset_size=2000,
    total_clients=15, clients_per_round=15, rounds=200,
    channel_noise=True, ul_snr_db=-8.0, dl_snr_db=-20.0,
)
sim = FDSimulator(cfg)
sim.setup()
result = sim.run("heuristic.random")
# result["metrics"] includes FD-specific: kl_divergence_avg, logit_comm_kb, comm_reduction_ratio, ...
```

## Development Notes

- Windows environment (cp1252 encoding) — avoid emoji in print() to stdout/stderr
- `thop` package is optional — without it, FLOPs metrics will be 0
- `torch-geometric` is optional — only needed for GNN-based selectors (rl_gnn, gat, gt_ppcs)
- Memory management is aggressive (cleanup every round, emergency cleanup at 70% RAM)
- `fast_mode=True` (default) breaks training after 2 batches — disable for real experiments
