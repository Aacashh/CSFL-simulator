from __future__ import annotations
from pathlib import Path
from typing import Tuple
import yaml

TEMPLATE = """
# Custom selection method template
from typing import List, Dict, Optional, Tuple
from csfl_simulator.core.client import ClientInfo


def select_clients(round_idx: int, K: int, clients: List[ClientInfo], history: Dict, rng,
                   time_budget=None, device=None) -> Tuple[List[int], Optional[List[float]], Optional[Dict]]:
    # Example: random selection
    ids = [c.id for c in clients]
    rng.shuffle(ids)
    return ids[:K], None, {}
"""


def default_template() -> str:
    return TEMPLATE.strip() + "\n"


def save_custom_method(project_root: Path, key: str, code: str) -> Tuple[Path, Path]:
    """Save custom method into selection/custom/<slug>.py and register in presets/methods.yaml.
    Returns (module_path, presets_file)
    """
    slug = key.replace(".", "_").replace("/", "_")
    mod_dir = project_root / "csfl_simulator" / "selection" / "custom"
    mod_dir.mkdir(parents=True, exist_ok=True)
    module_file = mod_dir / f"{slug}.py"
    module_file.write_text(code)

    presets = project_root / "presets" / "methods.yaml"
    data = yaml.safe_load(presets.read_text()) if presets.exists() else {"methods": []}
    entry = {
        "key": f"custom.{slug}",
        "module": f"csfl_simulator.selection.custom.{slug}",
        "display_name": key,
        "params": {},
        "type": "custom",
        "trainable": False,
    }
    # De-duplicate existing entry
    data_methods = data.get("methods", [])
    data_methods = [m for m in data_methods if m.get("key") != entry["key"]]
    data_methods.append(entry)
    data["methods"] = data_methods
    presets.write_text(yaml.safe_dump(data, sort_keys=False))
    return module_file, presets
