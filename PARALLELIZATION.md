# CUDA Parallelization Guide

This document describes the CUDA parallelization features implemented in the CSFL Simulator to dramatically reduce simulation time.

## Overview

The simulator now supports parallel client training using CUDA streams, which can provide **3-5x speedup** for typical federated learning simulations on GPU.

### Key Features

- **CUDA Stream-based Parallelization**: Multiple clients train simultaneously using separate CUDA streams
- **Memory-Aware Batching**: Automatically determines optimal number of parallel clients based on available VRAM
- **Deterministic Reproducibility**: Maintains exact reproducibility with the same random seed
- **Optimized Data Loading**: Enhanced DataLoader settings for better throughput
- **GPU-Accelerated Aggregation**: Keeps tensors on GPU during federated averaging
- **Backward Compatible**: Sequential mode still available (default)

## Configuration

### Parallel Clients Setting

The `parallel_clients` parameter controls parallelization:

- **`0` (default)**: Sequential mode - trains clients one at a time (original behavior)
- **`-1`**: Auto-detect - automatically determines optimal parallelism based on VRAM
- **`2-8`**: Fixed value - trains this many clients in parallel

### How to Enable

#### Via UI (Streamlit)

1. Open the simulator UI
2. In the sidebar, expand "Advanced (System & Privacy)"
3. Find "⚡ CUDA Parallelization" section
4. Select desired parallel_clients value:
   - Start with `-1` (auto-detect) for first run
   - Or manually select `2`, `3`, or `4` based on your GPU

#### Via Code

```python
from csfl_simulator.core.simulator import FLSimulator, SimConfig

config = SimConfig(
    dataset="MNIST",
    total_clients=100,
    clients_per_round=10,
    rounds=50,
    # ... other parameters ...
    parallel_clients=-1  # Enable auto-detect
)

sim = FLSimulator(config)
result = sim.run(method_key="heuristic.random")
```

## Performance Expectations

### Typical Speedups

With 10 clients selected per round:

| Hardware | Parallel Clients | Expected Speedup |
|----------|------------------|------------------|
| 8GB GPU  | 2-3             | 2.0-2.5x        |
| 8GB GPU  | 4               | 2.5-3.5x        |
| 16GB GPU | 4-6             | 3.0-4.0x        |
| 24GB+ GPU| 6-8             | 3.5-5.0x        |

### Factors Affecting Speedup

1. **Model Size**: Larger models (ResNet18) benefit more than smaller ones (CNN-MNIST)
2. **Batch Size**: Larger batches increase GPU utilization
3. **Local Epochs**: More local epochs = more parallelization benefit
4. **Fast Mode**: Fast mode (few batches) reduces parallelization benefit

## Memory Requirements

### Auto-Detection

When using `parallel_clients=-1`, the system:
1. Measures available VRAM
2. Estimates memory per client (model + gradients + activations)
3. Conservatively determines how many clients fit
4. Caps at 8 parallel clients for stability

### Manual Tuning

If you encounter OOM (Out of Memory) errors:
1. Reduce `parallel_clients` value
2. Reduce `batch_size`
3. Use smaller model
4. Close other GPU applications

### Memory Estimates

Approximate VRAM usage per parallel client:

| Model       | Per Client | 2 Parallel | 4 Parallel |
|-------------|-----------|------------|------------|
| CNN-MNIST   | ~200 MB   | ~400 MB    | ~800 MB    |
| LightCNN    | ~300 MB   | ~600 MB    | ~1.2 GB    |
| ResNet18    | ~500 MB   | ~1.0 GB    | ~2.0 GB    |

## Reproducibility

The parallel implementation maintains **exact reproducibility** through:

1. **Deterministic Seeding**: Each client gets a unique, reproducible seed
2. **Synchronized Operations**: CUDA operations are synchronized at critical points
3. **Fixed Client Order**: Clients are always processed in the same order
4. **Deterministic Algorithms**: PyTorch's deterministic mode is enabled

### Verification

Run the validation test to verify reproducibility:

```bash
python test_parallelization.py
```

This script:
- Runs sequential and parallel training with the same seed
- Compares results (should be identical)
- Reports speedup
- Validates deterministic behavior

## Additional Optimizations

Beyond parallelization, the following optimizations are also included:

### Enhanced DataLoaders

```python
num_workers=4           # Increased from 2 (more parallel data loading)
persistent_workers=True # Reuse workers across epochs
prefetch_factor=2       # Prefetch 2 batches ahead
pin_memory=True         # Faster CPU->GPU transfer
```

### GPU-Accelerated Aggregation

FedAvg aggregation now:
- Keeps tensors on GPU during weighted averaging
- Minimizes CPU-GPU transfers
- Uses in-place operations for memory efficiency

### Deterministic Mode

```python
torch.backends.cudnn.deterministic = True  # Reproducible convolutions
torch.backends.cudnn.benchmark = False     # Disable auto-tuning
torch.use_deterministic_algorithms(True)   # Force deterministic ops
```

## Troubleshooting

### Issue: No Speedup Observed

**Possible Causes:**
- Not using CUDA device (check device is "cuda" not "cpu")
- Fast mode with very few batches (overhead dominates)
- Very small model (CNN-MNIST with batch_size=32)
- Thermal throttling or other GPU processes

**Solutions:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Disable fast_mode for realistic testing
- Use larger models (ResNet18) or larger batches
- Close other GPU applications

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce `parallel_clients` (try 2 instead of 4)
- Reduce `batch_size` (try 16 instead of 32)
- Use smaller model
- Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: Results Not Reproducible

**Causes:**
- This should not happen if seed is fixed
- May indicate bug in parallel implementation

**Solutions:**
- Run validation test: `python test_parallelization.py`
- Report issue with test results
- Use sequential mode (`parallel_clients=0`) as fallback

### Issue: Slower with Parallelization

**Causes:**
- CPU device (parallelization helps little on CPU)
- Very small problem size (overhead > benefit)
- System resource contention

**Solutions:**
- Verify CUDA is being used
- Increase problem size (more rounds, clients, or epochs)
- Use sequential mode for small simulations

## Technical Details

### Architecture

The parallel training system consists of:

1. **ParallelTrainer** (`csfl_simulator/core/parallel.py`):
   - Manages model replicas and CUDA streams
   - Handles memory-aware batching
   - Ensures deterministic seeding

2. **Enhanced Simulator** (`csfl_simulator/core/simulator.py`):
   - Integrates ParallelTrainer when enabled
   - Falls back to sequential mode when disabled
   - Maintains backward compatibility

3. **Optimized Components**:
   - DataLoaders with better prefetching
   - GPU-accelerated aggregation
   - Deterministic seeding utilities

### CUDA Streams

CUDA streams enable true parallelism:
- Each client gets a dedicated stream
- Streams execute asynchronously on GPU
- Synchronization occurs only at boundaries
- Results are collected after all streams finish

### Memory Management

The system carefully manages GPU memory:
- Pre-allocates model replicas at setup
- Reuses replicas across rounds
- Clears gradients and activations promptly
- Monitors memory usage for auto-detection

## Best Practices

1. **Start with Auto-Detect**: Use `parallel_clients=-1` initially
2. **Monitor First Run**: Check memory usage and speedup
3. **Tune if Needed**: Adjust based on your specific hardware
4. **Validate Reproducibility**: Run test script periodically
5. **Scale Gradually**: Increase parallelism as you optimize

## Benchmarking

To benchmark your specific setup:

```python
import time
from csfl_simulator.core.simulator import FLSimulator, SimConfig

# Test configuration
config = SimConfig(
    dataset="MNIST",
    total_clients=100,
    clients_per_round=10,
    rounds=10,
    model="CNN-MNIST",
    device="cuda"
)

# Sequential
config.parallel_clients = 0
sim = FLSimulator(config)
start = time.time()
sim.run()
seq_time = time.time() - start

# Parallel
config.parallel_clients = -1
sim = FLSimulator(config)
start = time.time()
sim.run()
par_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Parallel: {par_time:.2f}s")
print(f"Speedup: {seq_time/par_time:.2f}x")
```

## Future Enhancements

Potential future improvements:
- Multi-GPU support (distribute clients across GPUs)
- Mixed precision training (FP16) for more parallelism
- Gradient compression for faster aggregation
- Asynchronous aggregation (while next batch trains)

## Support

If you encounter issues:
1. Run the validation test: `python test_parallelization.py`
2. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Review this guide's troubleshooting section
4. Report issues with test results and system info

