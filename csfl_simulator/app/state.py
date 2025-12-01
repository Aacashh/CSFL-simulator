from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Literal
import json
import time
import os
import numpy as np

from csfl_simulator.core.utils import ROOT


SNAP_DIR = (ROOT / "artifacts" / "checkpoints").resolve()
SNAP_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 2
AUTOSAVE_KEEP = 5


def _now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _split_arrays_document(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Split arrays out of a full snapshot document (v2 format)."""
    arrays: Dict[str, np.ndarray] = {}
    meta = json.loads(json.dumps(doc))
    data = meta.get("data", {})

    # Run data arrays
    if isinstance(data.get("metrics"), list):
        try:
            rounds = [int(m.get("round", i)) for i, m in enumerate(data["metrics"])]
            data.setdefault("series", {})
            data["series"]["rounds"] = "__npz__:rounds"
            arrays["rounds"] = np.asarray(rounds, dtype=np.int32)
            for key in ["accuracy", "f1", "precision", "recall", "loss", "composite", "fairness_var", "round_time"]:
                vals = []
                for m in data["metrics"]:
                    try:
                        vals.append(float(m.get(key, 0.0) or 0.0))
                    except Exception:
                        vals.append(0.0)
                data["series"][key] = f"__npz__:{key}"
                arrays[key] = np.asarray(vals, dtype=float)
        except Exception:
            pass

    # Compare data arrays
    if isinstance(data.get("metric_to_series"), dict):
        mts: Dict[str, Dict[str, List[float]]] = doc["data"]["metric_to_series"]  # type: ignore
        meta_mts: Dict[str, Dict[str, str]] = {}
        for metric, series_map in mts.items():
            meta_mts[metric] = {}
            for name, ys in series_map.items():
                key = f"mts::{metric}::{name}"
                meta_mts[metric][name] = f"__npz__:{key}"
                arrays[key] = np.asarray(list(map(float, ys)), dtype=float)
        data["metric_to_series"] = meta_mts
    if isinstance(data.get("selection_counts"), dict):
        sc: Dict[str, List[int]] = doc["data"]["selection_counts"]  # type: ignore
        meta_sc: Dict[str, str] = {}
        for name, ys in sc.items():
            key = f"selcounts::{name}"
            meta_sc[name] = f"__npz__:{key}"
            arrays[key] = np.asarray(list(map(int, ys)), dtype=np.int32)
        data["selection_counts"] = meta_sc

    meta["data"] = data
    return meta, arrays


def _merge_arrays_document(meta_doc: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    def resolve(value):
        if isinstance(value, str) and value.startswith("__npz__:"):
            key = value.split(":", 1)[1]
            return arrays.get(key)
        return value

    out = json.loads(json.dumps(meta_doc))
    data = out.get("data", {})
    # series
    if isinstance(data.get("series"), dict):
        for k, v in list(data["series"].items()):
            data["series"][k] = resolve(v)
    # metric_to_series
    if isinstance(data.get("metric_to_series"), dict):
        for metric, mp in list(data["metric_to_series"].items()):
            for name, v in list(mp.items()):
                data["metric_to_series"][metric][name] = resolve(v)
    # selection_counts
    if isinstance(data.get("selection_counts"), dict):
        for name, v in list(data["selection_counts"].items()):
            data["selection_counts"][name] = resolve(v)
    out["data"] = data
    return out


# --------- Validation and Migration ---------

def _default_run_ui() -> Dict[str, Any]:
    return {
        "show_accuracy": True,
        "show_loss": True,
        "show_time": True,
        "show_fair": True,
        "show_composite": True,
    }


def _default_compare_ui() -> Dict[str, Any]:
    return {
        "chart_style": "Interactive (Plotly)",
        "plotly_template": "plotly_white",
        "mpl_style": "classic",
        "methods_filter": None,
        "metrics_filter": None,
        "smoothing": 0,
        "y_scale": "linear",
        "legend_position": "right",
        "legend_cols": 1,
        "line_width": 2.0,
        "show_combined": True,
    }


def validate_run_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data or {})
    data.setdefault("run_id", "")
    data.setdefault("config", {})
    data.setdefault("device", "")
    data.setdefault("metrics", [])
    data.setdefault("history", {"selected": []})
    data.setdefault("participation_counts", [])
    data.setdefault("method", "")
    data.setdefault("stopped_early", False)
    data.setdefault("saved_at", _now_str())
    return data


def validate_compare_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data or {})
    mts = data.get("metric_to_series") or {}
    if not isinstance(mts, dict):
        mts = {}
    data["metric_to_series"] = mts
    methods = data.get("methods") or list(mts.get("Accuracy", {}).keys())
    data["methods"] = methods
    sc = data.get("selection_counts") or {}
    if not isinstance(sc, dict):
        sc = {}
    data["selection_counts"] = sc
    data.setdefault("metric_objects", {})
    data.setdefault("saved_at", _now_str())
    return data


def _migrate_v1_to_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    kind = payload.get("type") or payload.get("kind") or ("compare" if "metric_to_series" in payload else "run")
    doc = {"kind": kind, "schema_version": SCHEMA_VERSION, "data": {}, "ui": None, "saved_at": payload.get("saved_at", _now_str())}
    if kind == "run":
        data = {
            "run_id": payload.get("run_id", ""),
            "config": payload.get("config", {}),
            "device": payload.get("device", ""),
            "metrics": payload.get("metrics", []),
            "history": payload.get("history", {"selected": payload.get("selected", [])}) or {"selected": payload.get("selected", [])},
            "participation_counts": payload.get("participation_counts", []),
            "method": payload.get("method", ""),
            "stopped_early": payload.get("stopped_early", False),
        }
        doc["data"] = validate_run_v2(data)
    else:
        data = {
            "metric_to_series": payload.get("metric_to_series", {}),
            "selection_counts": payload.get("selection_counts", {}),
            "metric_objects": payload.get("metric_objects", {}),
            "methods": list((payload.get("metric_to_series", {}) or {}).get("Accuracy", {}).keys()),
        }
        doc["data"] = validate_compare_v2(data)
    return doc


# --------- Atomic Save/Load (V2) ---------

def _atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _save_document(kind: Literal['run','compare'], data: Dict[str, Any], ui: Optional[Dict[str, Any]], name: Optional[str], autosave: bool) -> Path:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_str()
    base = None
    if name:
        base = name.strip().replace(" ", "_").replace("/", "_").lower()
    else:
        base = f"{kind}_auto_{ts}"
    json_path = SNAP_DIR / f"{base}.json"
    npz_path = SNAP_DIR / f"{base}.npz"

    doc = {
        "kind": kind,
        "schema_version": SCHEMA_VERSION,
        "saved_at": ts,
        "data": data,
        "ui": ui or ({} if kind == 'run' else {}),
    }
    # Split arrays
    meta, arrays = _split_arrays_document(doc)
    _atomic_write(json_path, json.dumps(meta, indent=2).encode("utf-8"))
    # npz atomic - note: np.savez automatically adds .npz extension
    tmp_npz = Path(str(npz_path) + ".tmp")
    np.savez(tmp_npz, **(arrays or {"_empty": np.asarray([], dtype=np.int8)}))
    # np.savez created tmp_npz + ".npz", so we need to move that
    tmp_npz_actual = Path(str(tmp_npz) + ".npz")
    os.replace(tmp_npz_actual, npz_path)

    # Update latest pointers
    if not name:
        latest_json = SNAP_DIR / f"latest_{kind}.json"
        latest_npz = SNAP_DIR / f"latest_{kind}.npz"
        try:
            _atomic_write(latest_json, json.dumps(meta, indent=2).encode("utf-8"))
            tmp_npz2 = Path(str(latest_npz) + ".tmp")
            np.savez(tmp_npz2, **(arrays or {"_empty": np.asarray([], dtype=np.int8)}))
            # np.savez created tmp_npz2 + ".npz", so we need to move that
            tmp_npz2_actual = Path(str(tmp_npz2) + ".npz")
            os.replace(tmp_npz2_actual, latest_npz)
        except Exception:
            pass
        # rotate autosaves: keep last N
        try:
            autos = sorted(SNAP_DIR.glob(f"{kind}_auto_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for stale in autos[AUTOSAVE_KEEP:]:
                npz = stale.with_suffix(".npz")
                try:
                    stale.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    npz.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    return json_path


def save_run(data: Dict[str, Any], ui: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> Path:
    return _save_document('run', validate_run_v2(data), ui or _default_run_ui(), name, autosave=(name is None))


def save_compare(data: Dict[str, Any], ui: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> Path:
    return _save_document('compare', validate_compare_v2(data), ui or _default_compare_ui(), name, autosave=(name is None))


def _load_document(path: Path) -> Dict[str, Any]:
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
    # v2 doc
    if "data" in meta and "schema_version" in meta:
        return _merge_arrays_document(meta, arrays)
    # v1 -> v2 migration
    migrated = _migrate_v1_to_v2(_merge_arrays_legacy(meta, arrays))
    return migrated


def load_run(path: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    doc = _load_document(Path(path))
    if doc.get("kind") != 'run':
        raise ValueError("Snapshot is not a run kind")
    data = validate_run_v2(doc.get("data", {}))
    ui = doc.get("ui") or _default_run_ui()
    return data, ui


def load_compare(path: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    doc = _load_document(Path(path))
    if doc.get("kind") != 'compare':
        raise ValueError("Snapshot is not a compare kind")
    data = validate_compare_v2(doc.get("data", {}))
    ui = doc.get("ui") or _default_compare_ui()
    return data, ui


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


def _merge_arrays_legacy(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # Legacy format used save_snapshot/load_snapshot with arrays at top-level
    def resolve(value):
        if isinstance(value, str) and value.startswith("__npz__:"):
            key = value.split(":", 1)[1]
            return arrays.get(key)
        return value
    out = json.loads(json.dumps(meta))
    if isinstance(out.get("series"), dict):
        for k, v in list(out["series"].items()):
            out["series"][k] = resolve(v)
    if isinstance(out.get("metric_to_series"), dict):
        for metric, mp in list(out["metric_to_series"].items()):
            for name, v in list(mp.items()):
                out["metric_to_series"][metric][name] = resolve(v)
    if isinstance(out.get("selection_counts"), dict):
        for name, v in list(out["selection_counts"].items()):
            out["selection_counts"][name] = resolve(v)
    return out


def load_snapshot(path: Path) -> Dict[str, Any]:
    """Legacy loader kept for backward compatibility; returns merged doc.
    Prefer load_run/load_compare for kind-specific parsing.
    """
    return _load_document(Path(path))


def list_snapshots(kind: Optional[Literal['run','compare']] = None) -> List[Path]:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    snaps = sorted(SNAP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if kind is None:
        return snaps
    out: List[Path] = []
    for p in snaps:
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            k = meta.get("kind") or meta.get("type")
            if k == kind:
                out.append(p)
        except Exception:
            continue
    return out


