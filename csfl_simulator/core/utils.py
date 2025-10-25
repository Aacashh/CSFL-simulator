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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
