import os
import time
import random
import json
import gc
from pathlib import Path
from typing import Tuple, Any, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # .../CSFL-simulator
DATA_ROOT = ROOT / "data"
ART_ROOT = ROOT / "artifacts"


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (ART_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (ART_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ART_ROOT / "exports").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables strict deterministic mode (slower but reproducible)
                      If False, allows optimizations that may vary slightly across runs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Strict deterministic mode for exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms where possible
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Fallback for older PyTorch versions
            pass
    else:
        # Performance mode: allow non-deterministic optimizations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def autodetect_device(prefer_gpu: bool = True) -> str:
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def new_run_dir(prefix: str = "run") -> Tuple[Path, str]:
    ensure_dirs()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{prefix}_{ts}_{os.getpid()}"
    out = ART_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out, run_id


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def checkpoint_path(method: str, run_id: str, name: str) -> Path:
    p = ART_ROOT / "checkpoints" / method / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def cleanup_memory(force_cuda_empty: bool = True, verbose: bool = False):
    """
    Aggressively clean up memory to prevent system hangs during long simulations.
    
    Args:
        force_cuda_empty: If True, empties CUDA cache (recommended)
        verbose: If True, prints memory stats before/after cleanup
    """
    # Get memory info before
    info_before = None
    if verbose:
        try:
            import psutil
            mem_before = psutil.virtual_memory()
            info_before = {
                'ram_gb': mem_before.used / 1024**3,
                'ram_percent': mem_before.percent
            }
            if torch.cuda.is_available():
                info_before['gpu_gb'] = torch.cuda.memory_allocated() / 1024**3
            print(f"[Memory] Before cleanup: RAM {info_before['ram_percent']:.1f}% ({info_before['ram_gb']:.2f} GB)", end="")
            if 'gpu_gb' in info_before:
                print(f", GPU {info_before['gpu_gb']:.2f} GB", end="")
            print()
        except Exception:
            pass
    
    # CRITICAL: Run garbage collection MULTIPLE times
    # This is necessary because Python's GC may not catch all circular references in one pass
    for _ in range(3):
        gc.collect()
    
    # CUDA memory cleanup
    if torch.cuda.is_available():
        # Synchronize first to ensure all operations complete
        torch.cuda.synchronize()
        if force_cuda_empty:
            # Empty cache multiple times for thorough cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    # Final garbage collection pass
    gc.collect()
    
    # Report results
    if verbose:
        try:
            import psutil
            mem_after = psutil.virtual_memory()
            ram_freed = info_before['ram_gb'] - (mem_after.used / 1024**3)
            print(f"[Memory] After cleanup: RAM {mem_after.percent:.1f}% ({mem_after.used / 1024**3:.2f} GB) - freed {ram_freed:.2f} GB", end="")
            if torch.cuda.is_available() and 'gpu_gb' in info_before:
                gpu_after = torch.cuda.memory_allocated() / 1024**3
                gpu_freed = info_before['gpu_gb'] - gpu_after
                print(f", GPU {gpu_after:.2f} GB - freed {gpu_freed:.2f} GB", end="")
            print()
        except Exception:
            pass


def get_memory_info() -> dict:
    """Get current memory usage information."""
    info = {
        "timestamp": time.time(),
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_available_gb": 0.0,
        "ram_percent": 0.0,
    }
    
    try:
        import psutil
        # CPU and RAM
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info["ram_used_gb"] = mem.used / 1024**3
        info["ram_available_gb"] = mem.available / 1024**3
        info["ram_percent"] = mem.percent
    except ImportError:
        pass
    
    # GPU memory if available
    if torch.cuda.is_available():
        try:
            info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            free_mem, total_mem = torch.cuda.mem_get_info()
            info["gpu_free_gb"] = free_mem / 1024**3
            info["gpu_total_gb"] = total_mem / 1024**3
            info["gpu_used_gb"] = info["gpu_total_gb"] - info["gpu_free_gb"]
            info["gpu_percent"] = (info["gpu_used_gb"] / info["gpu_total_gb"]) * 100 if info["gpu_total_gb"] > 0 else 0
        except Exception:
            pass
    
    return info


def check_memory_critical(threshold_percent: float = 90.0) -> tuple[bool, str]:
    """
    Check if memory usage is critical and return warning message.
    
    Args:
        threshold_percent: Memory usage percentage to consider critical
        
    Returns:
        (is_critical, warning_message)
    """
    info = get_memory_info()
    warnings = []
    
    # Check RAM
    if info.get("ram_percent", 0) > threshold_percent:
        warnings.append(f"RAM at {info['ram_percent']:.1f}%")
    
    # Check GPU
    if info.get("gpu_percent", 0) > threshold_percent:
        warnings.append(f"GPU at {info['gpu_percent']:.1f}%")
    
    is_critical = len(warnings) > 0
    message = ", ".join(warnings) if warnings else "Memory OK"
    
    return is_critical, message
