
import logging
import sys
from csfl_simulator.core.simulator import FLSimulator, SimConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_simulation():
    print("Starting simulation reproduction...")
    
    # Configuration based on user report
    config = SimConfig(
        dataset="cifar100",
        model="resnet18",
        partition="label_shard",
        shards_per_client=10, # "10 labels for each client"
        total_clients=10,
        clients_per_round=5,
        rounds=2, # Short run to check for crash
        local_epochs=1,
        batch_size=32,
        parallel_clients=2, # Enable parallel to trigger potential issues
        device="cuda", # Force CUDA if available
        smoke_test_mode=True  # Fail fast
    )
    
    try:
        sim = FLSimulator(config)
        print("Simulator initialized.")
        results = sim.run()
        print("Simulation completed successfully.")
    except Exception as e:
        print(f"Simulation FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simulation()
