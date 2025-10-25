from __future__ import annotations
import importlib
from typing import Callable
from pathlib import Path
import yaml

PRESETS_PATH = Path(__file__).resolve().parents[2] / "presets" / "methods.yaml"


class MethodRegistry:
    def __init__(self):
        self.methods = {}

    def list_methods(self):
        return list(self.methods.keys())

    def register(self, key: str, module_path: str):
        self.methods[key] = module_path

    def load_presets(self):
        if PRESETS_PATH.exists():
            data = yaml.safe_load(PRESETS_PATH.read_text())
            for m in data.get("methods", []):
                self.register(m["key"], m["module"])

    def get(self, key: str) -> Callable:
        if key not in self.methods:
            raise KeyError(f"Unknown selection method key: {key}")
        module_path = self.methods[key]
        mod = importlib.import_module(module_path)
        if not hasattr(mod, "select_clients"):
            raise AttributeError(f"Module {module_path} missing select_clients function")
        return getattr(mod, "select_clients")
