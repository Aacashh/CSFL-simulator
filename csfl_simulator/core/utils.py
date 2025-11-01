import os
import time
import random
import json
from pathlib import Path
from typing import Tuple, Any

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
