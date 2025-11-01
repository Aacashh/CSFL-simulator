# Parameter Tooltips Guide

This guide documents all the helpful tooltips added to the CSFL Simulator interface. Each parameter now has a small "?" icon that users can hover over to see explanations.

## Main Parameters

### 📊 Dataset & Distribution

**Dataset**
- What it does: Selects the dataset for federated learning
- Tooltip: "The dataset to use for federated learning. MNIST/Fashion-MNIST are simpler (28x28 grayscale), CIFAR is more complex (32x32 RGB)."
- Options: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100

**Partition**
- What it does: Controls how data is distributed across clients
- Tooltip: "How data is distributed across clients:
  • IID: Uniformly random (all clients have similar data)
  • Dirichlet: Non-IID controlled by alpha parameter (realistic heterogeneity)
  • Label-shard: Each client gets data from only a few classes"
- Options: iid, dirichlet, label-shard

**Dirichlet Alpha**
- What it does: Controls data heterogeneity level
- Tooltip: "Controls data heterogeneity in Dirichlet partition. Lower values (0.1) = more non-IID (each client specializes), higher values (2.0) = closer to IID."
- Range: 0.05 - 2.0

**Label Shards per Client**
- What it does: Number of classes each client has
- Tooltip: "For label-shard partition: number of different classes each client has data from. Lower = more heterogeneous."
- Range: 1 - 10

### 🧠 Model & Training

**Model**
- What it does: Neural network architecture to train
- Tooltip: "Neural network architecture:
  • CNN-MNIST: Lightweight for MNIST/Fashion-MNIST
  • LightCNN: For CIFAR datasets
  • ResNet18: Deeper model for more complex tasks"
- Options: CNN-MNIST, LightCNN, ResNet18

**Total Clients**
- What it does: Number of devices in the FL system
- Tooltip: "Total number of participating clients (devices) in the federated learning system."
- Range: 2 - 1000

**Clients per Round (K)**
- What it does: How many clients train each round
- Tooltip: "Number of clients selected to participate in each training round. Smaller K = faster but less diverse, larger K = slower but more comprehensive."
- Range: 1 - 100

**Rounds**
- What it does: Number of communication cycles
- Tooltip: "Number of federated learning rounds (communication cycles). Each round: clients train locally → server aggregates → repeat."
- Range: 1 - 200

**Local Epochs**
- What it does: Training iterations per client per round
- Tooltip: "Number of training epochs each selected client performs on their local data before sending updates to the server."
- Range: 1 - 10

**Batch Size**
- What it does: Samples per training batch
- Tooltip: "Number of samples per training batch on each client. Larger = faster but needs more memory."
- Range: 8 - 512

**Learning Rate**
- What it does: Optimization step size
- Tooltip: "Step size for gradient descent optimization. Typical values: 0.001-0.01. Too high = unstable, too low = slow convergence."
- Range: 0.0001 - 1.0

### ⚙️ System Configuration

**Device**
- What it does: Hardware selection for training
- Tooltip: "Hardware to run training on:
  • auto: Automatically detect GPU if available
  • cpu: Use CPU only
  • cuda: Force GPU (NVIDIA)"
- Options: auto, cpu, cuda

**Seed**
- What it does: Random number generator seed
- Tooltip: "Random seed for reproducibility. Same seed = same results. Useful for comparing different methods fairly."
- Range: 0 - 10,000

**Fast Mode**
- What it does: Quick testing mode
- Tooltip: "When enabled, uses fewer batches per epoch for faster testing. Disable for full training runs."
- Type: Checkbox

**Pretrained**
- What it does: Load existing model weights
- Tooltip: "Load pre-trained model weights if available. Useful for transfer learning or continuing from a checkpoint."
- Type: Checkbox

## Advanced Parameters (System & Privacy)

### ⏱️ System Constraints

**Round Time Budget**
- What it does: Maximum time per round
- Tooltip: "Maximum time allowed per round in seconds. Set to 0 for unlimited. Simulates real-world time constraints where some clients may not finish in time."
- Range: 0.0 - 1,000,000 seconds

### 🔒 Differential Privacy

**DP Gaussian Noise Sigma**
- What it does: Privacy noise level
- Tooltip: "Standard deviation of Gaussian noise added to model updates for Differential Privacy. Higher = more privacy but less accuracy. 0 = no DP noise."
- Range: 0.0 - 10.0

**DP Epsilon per Selection**
- What it does: Privacy budget consumption
- Tooltip: "Privacy budget (epsilon) consumed per client selection. Lower epsilon = stronger privacy guarantee. Used with DP-aware selection methods."
- Range: 0.0 - 100.0

**DP Gradient Clip Norm**
- What it does: Gradient clipping for DP
- Tooltip: "Clips gradient norm before adding DP noise to bound sensitivity. Required for formal DP guarantees. 0 = no clipping. Typical values: 1.0-5.0."
- Range: 0.0 - 10.0

### ⚡ CUDA Parallelization

**Parallel Clients**
- What it does: Parallel training configuration
- Tooltip: "Number of clients to train in parallel using CUDA streams:
  • 0: Sequential training (no parallelization)
  • -1: Auto-detect optimal based on GPU memory
  • 2-8: Fixed number of parallel clients
  Parallel training can give 2-5x speedup on GPU with minimal memory overhead."
- Options: 0, -1, 2, 3, 4, 6, 8

### 🎯 Composite Reward Weights

**w_acc (Accuracy Weight)**
- What it does: Accuracy importance in optimization
- Tooltip: "Weight for accuracy in composite reward. Higher = prioritize model accuracy."
- Range: 0.0 - 1.0

**w_time (Time Weight)**
- What it does: Training speed importance
- Tooltip: "Weight for training time in composite reward. Higher = prioritize faster training rounds."
- Range: 0.0 - 1.0

**w_fair (Fairness Weight)**
- What it does: Participation equality importance
- Tooltip: "Weight for fairness in composite reward. Higher = ensure more equal participation across clients."
- Range: 0.0 - 1.0

**w_dp (Privacy Weight)**
- What it does: Privacy preservation importance
- Tooltip: "Weight for differential privacy in composite reward. Higher = prioritize privacy-preserving selections."
- Range: 0.0 - 1.0

## Selection Methods

**Selection Method**
- What it does: Algorithm for choosing clients
- Tooltip: "Algorithm for selecting which K clients participate each round. Options include:
  • Random: Baseline random selection
  • Heuristic: Data size, loss, gradient-based
  • System-aware: Consider device capabilities, deadlines
  • ML-based: Neural network, RL, bandit approaches"
- Options: Various methods from registry

**Methods for Comparison**
- What it does: Pre-select methods for batch comparison
- Tooltip: "Pre-select multiple methods to compare side-by-side. You can run them all together in the Compare tab."
- Type: Multi-select

**Repeats per Method**
- What it does: Statistical reliability through repetition
- Tooltip: "Number of times to repeat each method with different random seeds. Higher repeats = more reliable results with error bars. Useful for statistical significance."
- Range: 1 - 10

## How to Use Tooltips

1. **Desktop**: Hover your mouse over the "?" icon next to any parameter
2. **Mobile/Tablet**: Tap the "?" icon to see the tooltip
3. **Keyboard Navigation**: Tab to the "?" icon and press Enter/Space

## Benefits

✅ **New Users**: Quick understanding of what each parameter does
✅ **Researchers**: Technical details about algorithms and privacy
✅ **Practitioners**: Practical guidance on typical values
✅ **Students**: Educational context about federated learning concepts

## Future Enhancements

Potential improvements for tooltips:
- Add links to relevant papers/documentation
- Include visual examples or diagrams
- Add "Learn More" buttons for detailed explanations
- Context-sensitive tooltips that adapt based on other parameters

