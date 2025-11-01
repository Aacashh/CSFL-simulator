"""
Validation script to test CUDA parallelization implementation.

This script:
1. Runs a small simulation with sequential training (parallel_clients=0)
2. Runs the same simulation with parallel training (parallel_clients=2)
3. Compares results to ensure they are identical (reproducibility check)
4. Measures and reports speedup

Usage:
    python test_parallelization.py
"""
import time
import torch
import numpy as np
from csfl_simulator.core.simulator import FLSimulator, SimConfig


def run_test():
    print("=" * 80)
    print("CUDA Parallelization Validation Test")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Parallelization will have limited benefit on CPU.")
        print("This test will still run but won't show significant speedup.\n")
    else:
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ CUDA available: {device_name} ({total_mem:.1f} GB)")
    
    # Base configuration
    base_config = {
        "dataset": "MNIST",
        "partition": "iid",
        "total_clients": 20,
        "clients_per_round": 5,
        "rounds": 3,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.01,
        "model": "CNN-MNIST",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "fast_mode": True,
    }
    
    print("\nConfiguration:")
    print(f"  - Dataset: {base_config['dataset']}")
    print(f"  - Total clients: {base_config['total_clients']}")
    print(f"  - Clients per round: {base_config['clients_per_round']}")
    print(f"  - Rounds: {base_config['rounds']}")
    print(f"  - Device: {base_config['device']}")
    
    # Test 1: Sequential training
    print("\n" + "-" * 80)
    print("Test 1: Sequential Training (parallel_clients=0)")
    print("-" * 80)
    
    cfg_seq = SimConfig(**base_config, parallel_clients=0)
    sim_seq = FLSimulator(cfg_seq)
    
    start_time = time.time()
    result_seq = sim_seq.run(method_key="heuristic.random")
    seq_time = time.time() - start_time
    
    seq_metrics = result_seq["metrics"]
    seq_final_acc = seq_metrics[-1]["accuracy"] if seq_metrics else 0.0
    
    print(f"✓ Sequential training completed in {seq_time:.2f}s")
    print(f"  Final accuracy: {seq_final_acc:.4f}")
    
    # Test 2: Parallel training
    print("\n" + "-" * 80)
    print("Test 2: Parallel Training (parallel_clients=2)")
    print("-" * 80)
    
    cfg_par = SimConfig(**base_config, parallel_clients=2)
    sim_par = FLSimulator(cfg_par)
    
    start_time = time.time()
    result_par = sim_par.run(method_key="heuristic.random")
    par_time = time.time() - start_time
    
    par_metrics = result_par["metrics"]
    par_final_acc = par_metrics[-1]["accuracy"] if par_metrics else 0.0
    
    print(f"✓ Parallel training completed in {par_time:.2f}s")
    print(f"  Final accuracy: {par_final_acc:.4f}")
    
    # Test 3: Auto-detect parallel
    if torch.cuda.is_available():
        print("\n" + "-" * 80)
        print("Test 3: Auto-detect Parallel (parallel_clients=-1)")
        print("-" * 80)
        
        cfg_auto = SimConfig(**base_config, parallel_clients=-1)
        sim_auto = FLSimulator(cfg_auto)
        
        start_time = time.time()
        result_auto = sim_auto.run(method_key="heuristic.random")
        auto_time = time.time() - start_time
        
        auto_metrics = result_auto["metrics"]
        auto_final_acc = auto_metrics[-1]["accuracy"] if auto_metrics else 0.0
        
        print(f"✓ Auto-detect parallel training completed in {auto_time:.2f}s")
        print(f"  Final accuracy: {auto_final_acc:.4f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Check reproducibility (accuracies should be very close)
    acc_diff = abs(seq_final_acc - par_final_acc)
    reproducible = acc_diff < 0.001  # Allow tiny floating point differences
    
    print(f"\n1. Reproducibility Check:")
    print(f"   Sequential accuracy: {seq_final_acc:.6f}")
    print(f"   Parallel accuracy:   {par_final_acc:.6f}")
    print(f"   Difference:          {acc_diff:.6f}")
    
    if reproducible:
        print("   ✓ PASSED: Results are reproducible!")
    else:
        print("   ✗ FAILED: Results differ significantly!")
        print("   This may indicate an issue with deterministic seeding.")
    
    print(f"\n2. Performance:")
    print(f"   Sequential time: {seq_time:.2f}s")
    print(f"   Parallel time:   {par_time:.2f}s")
    
    if par_time > 0:
        speedup = seq_time / par_time
        print(f"   Speedup:         {speedup:.2f}x")
        
        if speedup > 1.5:
            print(f"   ✓ Excellent speedup! ({speedup:.2f}x faster)")
        elif speedup > 1.1:
            print(f"   ✓ Good speedup! ({speedup:.2f}x faster)")
        elif speedup > 0.9:
            print(f"   ≈ Similar performance (overhead may dominate with small models)")
        else:
            print(f"   ⚠ Parallel is slower - check CUDA availability and configuration")
    
    # Memory info
    if torch.cuda.is_available():
        print(f"\n3. GPU Memory:")
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Max allocated: {max_allocated:.2f} GB")
    
    print("\n" + "=" * 80)
    
    if reproducible:
        print("✓ ALL TESTS PASSED!")
        if torch.cuda.is_available() and speedup > 1.1:
            print(f"  Parallelization is working correctly with {speedup:.2f}x speedup.")
        else:
            print("  Reproducibility is maintained.")
    else:
        print("✗ TESTS FAILED: Reproducibility issue detected.")
        print("  Please report this issue.")
    
    print("=" * 80)
    
    return reproducible


if __name__ == "__main__":
    try:
        success = run_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

