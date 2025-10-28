from __future__ import annotations
import importlib
from typing import Callable, Any, Tuple, Dict
from pathlib import Path
import yaml

PRESETS_PATH = Path(__file__).resolve().parents[2] / "presets" / "methods.yaml"


class MethodRegistry:
    def __init__(self):
        self.methods: Dict[str, str] = {}
        self.params: Dict[str, dict] = {}
        self.display_names: Dict[str, str] = {}
        self.origins: Dict[str, str] = {}
        self._label_to_key: Dict[str, str] = {}

    def list_methods(self):
        return list(self.methods.keys())

    def register(self, key: str, module_path: str, params: dict | None = None, display_name: str | None = None, origin: str | None = None):
        self.methods[key] = module_path
        if params is not None:
            self.params[key] = params
        if display_name is not None:
            self.display_names[key] = display_name
        if origin is not None:
            self.origins[key] = origin

    def load_presets(self):
        if PRESETS_PATH.exists():
            data = yaml.safe_load(PRESETS_PATH.read_text())
            for m in data.get("methods", []):
                self.register(
                    m["key"],
                    m["module"],
                    m.get("params", {}),
                    m.get("display_name", m["key"]),
                    m.get("origin", "unknown"),
                )
        # Build label->key mapping with origin tag for UI
        self._rebuild_labels()

    def _label_for(self, key: str, include_origin: bool = True) -> str:
        name = self.display_names.get(key, key)
        if include_origin:
            origin = self.origins.get(key, "unknown")
            return f"{name} ({origin})"
        return name

    def _rebuild_labels(self):
        self._label_to_key = {}
        for k in self.methods.keys():
            label = self._label_for(k, include_origin=True)
            # Ensure uniqueness: if collision, append key
            if label in self._label_to_key:
                label = f"{label} — {k}"
            self._label_to_key[label] = k

    def labels_map(self) -> Dict[str, str]:
        if not self._label_to_key:
            self._rebuild_labels()
        return dict(self._label_to_key)

    def key_from_label(self, label: str) -> str:
        if not self._label_to_key:
            self._rebuild_labels()
        if label not in self._label_to_key:
            # Fallback: maybe label is actually a key
            if label in self.methods:
                return label
            raise KeyError(f"Unknown selection method label: {label}")
        return self._label_to_key[label]

    def get(self, key: str) -> Callable:
        if key not in self.methods:
            raise KeyError(f"Unknown selection method key: {key}")
        module_path = self.methods[key]
        mod = importlib.import_module(module_path)
        if not hasattr(mod, "select_clients"):
            raise AttributeError(f"Module {module_path} missing select_clients function")
        return getattr(mod, "select_clients")

    def get_params(self, key: str) -> dict:
        return dict(self.params.get(key, {}))

    def invoke(self, key: str, *args: Any, **kwargs: Any) -> Tuple[list[int], list[float] | None, dict | None]:
        func = self.get(key)
        # Merge preset params, letting explicit kwargs override
        merged = self.get_params(key)
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        return func(*args, **merged)
