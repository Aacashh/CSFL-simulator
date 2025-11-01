# CSFL-simulator

A Streamlit-based, playground-style simulator for Client Selection in Federated Learning (CSFL).

Highlights
- Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- Partitioning: IID, Dirichlet (alpha), Label-shard
- Models: CNN-MNIST (light), LightCNN (CIFAR), ResNet18 (CIFAR)
- Selection methods (presets):
  - Heuristic: Random, Proportional-to-Data, Top-K Loss, Gradient-Norm, Fairness-Adjusted, Cluster-Balanced, Round-Robin, MMR-Diverse, Label-Coverage, DP-Budget-Aware
  - System-aware: FedCS (deadline-aware), TiFL (tiers), Oort-style (utility/time + exploration), Power-of-Choice (two-stage), Oort-Plus (fairness+recency)
  - ML-based: RL-GNN (trainable), Graph Transformer (GT-PPCS), GAT-based, Bandits (epsilon-greedy, LinUCB, RFF-LinUCB), Meta Ranker (SGDRegressor), NeuralLinear-UCB (tiny MLP + Bayesian head), DeepSets Ranker, RankFormer-Tiny
- **⚡ CUDA Parallelization**: 3-5x speedup with parallel client training using CUDA streams (see [PARALLELIZATION.md](PARALLELIZATION.md))
- Realism knobs: heterogeneity, network/channel, mobility, time budget, dropouts, DP noise and epsilon
- Export to .ipynb: generate a self-contained notebook capturing the simulation code, selection code, and parameters
- Compare methods side-by-side with interactive visualizations

Status
- The simulator now exposes system/time and DP controls, computes a composite reward (accuracy/time/fairness/DP), and includes three new lightweight NN-based selectors.

Quick start (pip)
1) Create and activate a virtual environment

```bash
cd CSFL-simulator
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
```

2) Install dependencies

- First install torch/torchvision appropriate for your CUDA (or CPU) from https://pytorch.org
- Then install the package in editable mode

```bash
pip install -e .
```

3) (Optional) Download datasets locally

By default, datasets are stored under `data/`. You can pre-download all supported datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100):

```bash
python scripts/download_data.py --all
```

Or choose specific ones:

```bash
python scripts/download_data.py --datasets mnist fashion-mnist
```

Note: The app will auto-download datasets on first use if they are missing.

4) (Optional) Install torch-geometric and companions for GNN presets

Refer to the official wheels (replace TORCH/CUDA tags):

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-geometric
```

5) Launch the app

**Mac users:** Use the safe startup script to avoid cache issues:
```bash
./run_mac.sh
```

**All platforms:**
```bash
streamlit run csfl_simulator/app/main.py
```

**Note:** If you get channel mismatch errors after updating code, see `QUICK_FIX.txt` or run `./clean_cache.sh`

6) Smoke test
- Dataset: MNIST, IID
- Model: CNN-MNIST
- Rounds: 3, Clients: 10, K per round: 3
- Method: Random

Export to Notebook
- Use the Export tab to produce a notebook under artifacts/exports/<run_id>.ipynb

Performance Optimization
- **CUDA Parallelization**: Enable parallel client training for 3-5x speedup on GPU
  - In the UI: Sidebar → Advanced → "Parallel clients" (try `-1` for auto-detect)
  - Maintains exact reproducibility with deterministic seeding
  - Automatically manages GPU memory
  - See [PARALLELIZATION.md](PARALLELIZATION.md) for full guide
  - Test with: `python test_parallelization.py`

Notes
- GPU is auto-detected; you can manually switch CPU/GPU in the sidebar.
- Some presets (RL-GNN, GAT) require torch-geometric.
- Several advanced selectors use scikit-learn (installed via pip install -e .): RFF-LinUCB and Meta Ranker.
- Composite metric (optimization target): acc/time/fairness/DP weights can be configured in the Advanced sidebar.
- Pretrained weights: hooks are present; you can place weights under artifacts/checkpoints/pretrained/.
- See csfl_simulator/selection/README.md for details on method behavior and parameters.
