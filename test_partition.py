
import numpy as np
from csfl_simulator.core.partition import label_shard_partition

def test_label_shard():
    # Create dummy labels: 100 classes, 500 samples each
    labels = []
    for i in range(100):
        labels.extend([i] * 500)
    
    num_clients = 10
    shards_per_client = 10
    
    # This should ideally give each client 10 distinct labels if it was "Non-IID"
    # But based on code reading, it might be random.
    mapping = label_shard_partition(labels, num_clients, shards_per_client)
    
    print(f"Total clients: {len(mapping)}")
    for cid, idxs in mapping.items():
        client_labels = [labels[i] for i in idxs]
        unique_labels = np.unique(client_labels)
        print(f"Client {cid}: {len(idxs)} samples, {len(unique_labels)} unique labels")
        if len(unique_labels) > shards_per_client * 2: # heuristic check
             print(f"  -> Seems IID (Random) - Expected ~{shards_per_client} labels, got {len(unique_labels)}")
        else:
             print(f"  -> Seems Non-IID")

if __name__ == "__main__":
    test_label_shard()
