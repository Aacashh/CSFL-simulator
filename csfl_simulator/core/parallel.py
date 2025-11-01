"""
Parallel training engine for CUDA-accelerated client training.

This module provides CUDA-based parallelization for federated learning client training,
using CUDA streams and memory-aware batching while maintaining deterministic behavior.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class ParallelTrainer:
    """
    Manages parallel client training using CUDA streams and model replicas.
    
    Features:
    - Memory-aware batching based on available VRAM
    - Per-client CUDA streams for true parallelism
    - Deterministic seeding and synchronization
    - Automatic fallback to sequential mode if needed
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        device: str,
        criterion: nn.Module,
        lr: float,
        max_parallel: int = -1,
        track_grad_norm: bool = False
    ):
        """
        Args:
            global_model: The global model to replicate
            device: Device string (e.g., 'cuda', 'cpu')
            criterion: Loss criterion
            lr: Learning rate
            max_parallel: Maximum parallel clients (-1 for auto-detect, 0 for sequential)
            track_grad_norm: Whether to track gradient norms
        """
        self.global_model = global_model
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.track_grad_norm = track_grad_norm
        self.is_cuda = device.startswith('cuda')
        
        # Determine optimal parallelism
        if max_parallel == -1 and self.is_cuda:
            self.max_parallel = self._auto_detect_parallel()
        elif max_parallel == 0:
            self.max_parallel = 1  # Sequential mode
        else:
            self.max_parallel = max(1, max_parallel)
        
        # Pre-allocate model replicas and streams
        self.model_replicas: List[nn.Module] = []
        self.streams: List[torch.cuda.Stream] = []
        self._init_replicas()
    
    def _auto_detect_parallel(self) -> int:
        """Auto-detect optimal number of parallel clients based on VRAM."""
        if not self.is_cuda:
            return 1
        
        try:
            # Get available memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            
            # Estimate model size
            model_mem = sum(p.numel() * p.element_size() for p in self.global_model.parameters())
            # Account for gradients, optimizer state, and activations (rough 4x multiplier)
            estimated_per_client = model_mem * 4
            
            # Conservative estimate: use 60% of free memory
            usable_mem = int(free_mem * 0.6)
            max_clients = max(1, usable_mem // estimated_per_client)
            
            # Cap at reasonable values
            return min(max_clients, 8)
        except Exception:
            # Fallback to conservative value
            return 2
    
    def _init_replicas(self):
        """Initialize model replicas and CUDA streams."""
        self.model_replicas = []
        self.streams = []
        
        for i in range(self.max_parallel):
            # Create model replica
            replica = copy.deepcopy(self.global_model).to(self.device)
            self.model_replicas.append(replica)
            
            # Create CUDA stream if on CUDA
            if self.is_cuda:
                stream = torch.cuda.Stream()
                self.streams.append(stream)
            else:
                self.streams.append(None)
    
    def train_clients_parallel(
        self,
        client_ids: List[int],
        client_loaders: Dict[int, Any],
        local_epochs: int,
        fast_mode: bool = False,
        seed_offset: int = 0
    ) -> List[Tuple[Dict[str, torch.Tensor], int, float, float]]:
        """
        Train multiple clients in parallel batches.
        
        Args:
            client_ids: List of client IDs to train
            client_loaders: Dict mapping client ID to DataLoader
            local_epochs: Number of local training epochs
            fast_mode: If True, break after 2 batches
            seed_offset: Seed offset for reproducibility (typically round number)
        
        Returns:
            List of (state_dict, data_size, last_loss, grad_norm) for each client
        """
        results = []
        
        # Process clients in batches for parallelism
        for batch_start in range(0, len(client_ids), self.max_parallel):
            batch_end = min(batch_start + self.max_parallel, len(client_ids))
            batch_ids = client_ids[batch_start:batch_end]
            
            # Train this batch in parallel
            batch_results = self._train_batch(
                batch_ids,
                client_loaders,
                local_epochs,
                fast_mode,
                seed_offset
            )
            results.extend(batch_results)
        
        return results
    
    def _train_batch(
        self,
        client_ids: List[int],
        client_loaders: Dict[int, Any],
        local_epochs: int,
        fast_mode: bool,
        seed_offset: int
    ) -> List[Tuple[Dict[str, torch.Tensor], int, float, float]]:
        """Train a batch of clients in parallel using CUDA streams."""
        batch_results = []
        
        if self.is_cuda and len(client_ids) > 1:
            # Parallel training with CUDA streams
            batch_results = self._train_batch_cuda_streams(
                client_ids, client_loaders, local_epochs, fast_mode, seed_offset
            )
        else:
            # Sequential training (CPU or single client)
            for idx, cid in enumerate(client_ids):
                result = self._train_single_client(
                    cid, client_loaders[cid], local_epochs, fast_mode, seed_offset, idx
                )
                batch_results.append(result)
        
        return batch_results
    
    def _train_batch_cuda_streams(
        self,
        client_ids: List[int],
        client_loaders: Dict[int, Any],
        local_epochs: int,
        fast_mode: bool,
        seed_offset: int
    ) -> List[Tuple[Dict[str, torch.Tensor], int, float, float]]:
        """Train clients in parallel using CUDA streams."""
        # Launch training on each stream
        futures = []
        for idx, cid in enumerate(client_ids):
            stream = self.streams[idx]
            with torch.cuda.stream(stream):
                # Set deterministic seed for this client
                client_seed = seed_offset * 10000 + cid
                torch.manual_seed(client_seed)
                if self.is_cuda:
                    torch.cuda.manual_seed_all(client_seed)
                
                # Train on this stream
                result = self._train_single_client_on_stream(
                    cid, client_loaders[cid], local_epochs, fast_mode, idx
                )
                futures.append(result)
        
        # Synchronize all streams
        if self.is_cuda:
            torch.cuda.synchronize()
        
        return futures
    
    def _train_single_client(
        self,
        cid: int,
        loader: Any,
        local_epochs: int,
        fast_mode: bool,
        seed_offset: int,
        replica_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """Train a single client (used for sequential mode or as base method)."""
        # Set deterministic seed
        client_seed = seed_offset * 10000 + cid
        torch.manual_seed(client_seed)
        if self.is_cuda:
            torch.cuda.manual_seed_all(client_seed)
        
        return self._train_single_client_on_stream(cid, loader, local_epochs, fast_mode, replica_idx)
    
    def _train_single_client_on_stream(
        self,
        cid: int,
        loader: Any,
        local_epochs: int,
        fast_mode: bool,
        replica_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """Core training logic for a single client on a specific model replica."""
        model = self.model_replicas[replica_idx]
        
        # Reset to global model state
        model.load_state_dict(self.global_model.state_dict())
        model.train()
        
        # Create optimizer
        opt = optim.SGD(model.parameters(), lr=self.lr)
        
        last_loss = 0.0
        last_grad_norm = 0.0
        
        # Training loop
        for e in range(local_epochs):
            for bi, (x, y) in enumerate(loader):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad()
                out = model(x)
                loss = self.criterion(out, y)
                loss.backward()
                
                # Track gradient norm if requested
                if self.track_grad_norm:
                    try:
                        last_param = next(reversed(list(model.parameters())))
                        if last_param.grad is not None:
                            last_grad_norm = float(last_param.grad.norm().item())
                    except Exception:
                        pass
                
                opt.step()
                last_loss = float(loss.item())
                
                if fast_mode and bi > 1:
                    break
        
        # Export weights (clone to avoid sharing memory)
        with torch.no_grad():
            state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        
        data_size = len(loader.dataset)
        
        return state_dict, data_size, last_loss, last_grad_norm
    
    def update_global_model(self, new_state_dict: Dict[str, torch.Tensor]):
        """Update the global model and sync all replicas."""
        self.global_model.load_state_dict(new_state_dict)
        # Replicas will be synced at the start of next training batch
    
    def cleanup(self):
        """Clean up resources to free GPU memory."""
        # Delete model replicas
        for replica in self.model_replicas:
            del replica
        self.model_replicas.clear()
        
        # Clear CUDA streams
        if self.is_cuda:
            for stream in self.streams:
                if stream is not None:
                    # Synchronize before deleting
                    stream.synchronize()
            self.streams.clear()
        
        # Force GPU cleanup
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def create_trainer(
    model: nn.Module,
    device: str,
    criterion: nn.Module,
    lr: float,
    parallel_clients: int = 0,
    track_grad_norm: bool = False
) -> Optional[ParallelTrainer]:
    """
    Factory function to create a ParallelTrainer if parallelization is enabled.
    
    Args:
        model: Global model
        device: Device string
        criterion: Loss criterion
        lr: Learning rate
        parallel_clients: Number of parallel clients (0=sequential, -1=auto)
        track_grad_norm: Whether to track gradient norms
    
    Returns:
        ParallelTrainer instance if parallel_clients != 0, else None
    """
    if parallel_clients == 0:
        return None
    
    return ParallelTrainer(
        global_model=model,
        device=device,
        criterion=criterion,
        lr=lr,
        max_parallel=parallel_clients,
        track_grad_norm=track_grad_norm
    )

