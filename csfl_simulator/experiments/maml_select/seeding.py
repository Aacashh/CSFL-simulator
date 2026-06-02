"""Robust random seeding module for reproducible FL experiments.

Provides:
- Deterministic seed setting across all RNG backends (Python, NumPy, PyTorch)
- Seed state logging to JSON for audit trails
- ``fork_rng()`` context manager that isolates selector/meta-learner RNG from
  the FL training RNG, preventing cross-contamination between components.
- Initial model weight hash for verification of identical initialisation.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class SeedRecord:
    """Audit record of the seeding configuration used for a run."""

    seed: int
    python_rng_state_hash: str
    numpy_rng_state_hash: str
    torch_rng_state_hash: str
    cuda_available: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    model_weight_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    performance_mode: bool = False,
) -> None:
    """Set all RNG seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, forces cuDNN deterministic mode.
        performance_mode: If True, enables cuDNN benchmark mode (faster but
            not bit-for-bit reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    if performance_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    elif deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def _hash_bytes(data: bytes) -> str:
    """Return a short SHA256 hex digest."""
    return hashlib.sha256(data).hexdigest()[:16]


def _rng_state_hash_python() -> str:
    state = random.getstate()
    return _hash_bytes(str(state).encode())


def _rng_state_hash_numpy() -> str:
    state = np.random.get_state()
    return _hash_bytes(str(state[1][:10].tolist()).encode())


def _rng_state_hash_torch() -> str:
    state = torch.random.get_rng_state()
    return _hash_bytes(state.numpy().tobytes()[:256])


def model_weight_hash(model: torch.nn.Module) -> str:
    """Compute a hash of model initial weights for verification."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode())
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def capture_seed_record(
    seed: int,
    model: Optional[torch.nn.Module] = None,
) -> SeedRecord:
    """Capture current RNG state as an audit record."""
    record = SeedRecord(
        seed=seed,
        python_rng_state_hash=_rng_state_hash_python(),
        numpy_rng_state_hash=_rng_state_hash_numpy(),
        torch_rng_state_hash=_rng_state_hash_torch(),
        cuda_available=torch.cuda.is_available(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )
    if model is not None:
        record.model_weight_hash = model_weight_hash(model)
    return record


def save_seed_record(
    record: SeedRecord,
    output_dir: Path,
    filename: str = "seed_record.json",
) -> Path:
    """Save a seed record to a JSON file."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(record.to_dict(), f, indent=2)
    return path


@contextmanager
def fork_rng(seed_offset: int = 0, devices: Optional[list] = None):
    """Context manager that forks all RNG states.

    Use this to isolate the selector/meta-learner RNG from the FL training
    RNG so that changes in one component do not affect the other's randomness.

    Example::

        with fork_rng(seed_offset=round_idx):
            # Selector operations use isolated RNG
            selected = selector.select(...)
        # Original RNG state is restored here

    Args:
        seed_offset: Optional offset added to current torch seed inside fork.
        devices: CUDA device indices to fork. Defaults to all available.
    """
    # Save Python random state
    py_state = random.getstate()
    # Save NumPy state
    np_state = np.random.get_state()

    # Fork torch RNG (including CUDA)
    cuda_devices = devices or []
    if not cuda_devices and torch.cuda.is_available():
        cuda_devices = list(range(torch.cuda.device_count()))

    with torch.random.fork_rng(devices=cuda_devices):
        if seed_offset != 0:
            current_seed = torch.initial_seed()
            torch.manual_seed(current_seed + seed_offset)
        try:
            yield
        finally:
            pass

    # Restore Python and NumPy states
    random.setstate(py_state)
    np.random.set_state(np_state)
