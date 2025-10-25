# CSFL-simulator

A Streamlit-based, playground-style simulator for Client Selection in Federated Learning (CSFL).

Highlights
- Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- Partitioning: IID, Dirichlet (alpha), Label-shard
- Models: CNN-MNIST (light), LightCNN (CIFAR), ResNet18 (CIFAR)
- Selection methods (presets):
  - Heuristic: Random, Proportional-to-Data, Top-K Loss, Gradient-Norm, Fairness-Adjusted, Cluster-Balanced, Round-Robin, MMR-Diverse, Label-Coverage, DP-Budget-Aware
  - System-aware: FedCS (deadline-aware), TiFL (tiers), Oort-style (utility/time + exploration), Power-of-Choice (two-stage), Oort-Plus (fairness+recency)
  - ML-based: RL-GNN (trainable), Graph Transformer (GT-PPCS), GAT-based, Bandits (epsilon-greedy, LinUCB, RFF-LinUCB), Meta Ranker (SGDRegressor)
- Realism knobs: heterogeneity, network/channel, mobility, time budget, dropouts, DP noise and epsilon
- Export to .ipynb: generate a self-contained notebook capturing the simulation code, selection code, and parameters
- Compare methods side-by-side with interactive visualizations

Status
- This is an initial scaffold wired to run a basic simulation loop (Random preset) and display metrics.
- Subsequent updates will port and integrate your GRL-client-selection, GT-PPCS, and CIFAR-GAT methods fully and add all plots.

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

3) (Optional) Install torch-geometric and companions for GNN presets

Refer to the official wheels (replace TORCH/CUDA tags):

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-geometric
```

4) Launch the app

```bash
streamlit run csfl_simulator/app/main.py
```

5) Smoke test
- Dataset: MNIST, IID
- Model: CNN-MNIST
- Rounds: 3, Clients: 10, K per round: 3
- Method: Random

Export to Notebook
- Use the Export tab to produce a notebook under artifacts/exports/<run_id>.ipynb

Notes
- GPU is auto-detected; you can manually switch CPU/GPU in the sidebar.
- Some presets (RL-GNN, GAT) require torch-geometric.
- Several advanced selectors use scikit-learn (installed via pip install -e .): RFF-LinUCB and Meta Ranker.
- Pretrained weights: hooks are present; you can place weights under artifacts/checkpoints/pretrained/.
- See csfl_simulator/selection/README.md for details on method behavior and parameters.

References
- GRL-client-selection.ipynb (your latest work)
- GT-PPCS.py
- cifar10-gat.py
- Related research PDFs in this directory
