from __future__ import annotations
import importlib
from typing import Callable, Any, Tuple
from pathlib import Path
import yaml

PRESETS_PATH = Path(__file__).resolve().parents[2] / "presets" / "methods.yaml"


class MethodRegistry:
    def __init__(self):
        self.methods = {}
        self.params = {}

    def list_methods(self):
        return list(self.methods.keys())

    def register(self, key: str, module_path: str, params: dict | None = None):
        self.methods[key] = module_path
        if params is not None:
            self.params[key] = params

    def load_presets(self):
        if PRESETS_PATH.exists():
            data = yaml.safe_load(PRESETS_PATH.read_text())
            for m in data.get("methods", []):
                self.register(m["key"], m["module"], m.get("params", {}))

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
