from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import time
import numpy as np

from csfl_simulator.core.utils import ROOT


SNAP_DIR = (ROOT / "artifacts" / "checkpoints").resolve()
SNAP_DIR.mkdir(parents=True, exist_ok=True)


def _now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _split_arrays(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Extract large numeric series into an array dict for NPZ; return (meta, arrays)."""
    arrays: Dict[str, np.ndarray] = {}
    meta = json.loads(json.dumps(payload))  # deep copy via JSON

    # Run payload
    if "metrics" in meta and isinstance(meta["metrics"], list):
        # Convert metrics list to per-key arrays for common metrics
        try:
            rounds = [int(m.get("round", i)) for i, m in enumerate(payload["metrics"])]
            meta.setdefault("series", {})
            meta["series"]["rounds"] = "__npz__:rounds"
            arrays["rounds"] = np.asarray(rounds, dtype=np.int32)
            for key in ["accuracy", "f1", "precision", "recall", "loss", "composite", "fairness_var", "round_time"]:
                vals = []
                for m in payload["metrics"]:
                    try:
                        vals.append(float(m.get(key, 0.0) or 0.0))
                    except Exception:
                        vals.append(0.0)
                meta["series"][key] = f"__npz__:{key}"
                arrays[key] = np.asarray(vals, dtype=float)
        except Exception:
            pass
        # History (selected) can be large; keep in JSON as list of lists for readability

    # Compare payload
    if "metric_to_series" in meta and isinstance(meta["metric_to_series"], dict):
        mts: Dict[str, Dict[str, List[float]]] = payload["metric_to_series"]
        meta_mts: Dict[str, Dict[str, str]] = {}
        for metric, series_map in mts.items():
            meta_mts[metric] = {}
            for name, ys in series_map.items():
                key = f"mts::{metric}::{name}"
                meta_mts[metric][name] = f"__npz__:{key}"
                arrays[key] = np.asarray(list(map(float, ys)), dtype=float)
        meta["metric_to_series"] = meta_mts
    if "selection_counts" in meta and isinstance(meta["selection_counts"], dict):
        sc: Dict[str, List[int]] = payload["selection_counts"]
        meta_sc: Dict[str, str] = {}
        for name, ys in sc.items():
            key = f"selcounts::{name}"
            meta_sc[name] = f"__npz__:{key}"
            arrays[key] = np.asarray(list(map(int, ys)), dtype=np.int32)
        meta["selection_counts"] = meta_sc

    return meta, arrays


def _merge_arrays(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    def resolve(value):
        if isinstance(value, str) and value.startswith("__npz__:"):
            key = value.split(":", 1)[1]
            return arrays.get(key)
        return value

    out = json.loads(json.dumps(meta))
    # series
    if isinstance(out.get("series"), dict):
        for k, v in list(out["series"].items()):
            out["series"][k] = resolve(v)
    # metric_to_series
    if isinstance(out.get("metric_to_series"), dict):
        for metric, mp in list(out["metric_to_series"].items()):
            for name, v in list(mp.items()):
                out["metric_to_series"][metric][name] = resolve(v)
    # selection_counts
    if isinstance(out.get("selection_counts"), dict):
        for name, v in list(out["selection_counts"].items()):
            out["selection_counts"][name] = resolve(v)
    return out


def save_snapshot(name: Optional[str], payload: Dict[str, Any]) -> Path:
    """Save a run or comparison payload. When name is None, save as 'latest'.
    Returns path to the JSON metadata file.
    """
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_str()
    base = ("latest" if not name else name.strip().replace(" ", "_").replace("/", "_").lower())
    json_path = SNAP_DIR / f"{base}.json"
    npz_path = SNAP_DIR / f"{base}.npz"
    # annotate type
    payload = {**payload}
    if "metric_to_series" in payload:
        payload.setdefault("type", "compare")
    else:
        payload.setdefault("type", "run")
    payload.setdefault("saved_at", ts)
    # split arrays to NPZ
    meta, arrays = _split_arrays(payload)
    # write files
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if arrays:
        np.savez(npz_path, **arrays)
    else:
        # ensure an empty npz exists for consistency
        np.savez(npz_path, _empty=np.asarray([], dtype=np.int8))
    return json_path


def load_snapshot(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    npz_path = path.with_suffix(".npz")
    arrays: Dict[str, np.ndarray] = {}
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                for k in npz.files:
                    arrays[k] = npz[k]
        except Exception:
            arrays = {}
    return _merge_arrays(meta, arrays)


def list_snapshots() -> List[Path]:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(SNAP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


