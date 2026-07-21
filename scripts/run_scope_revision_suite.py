#!/usr/bin/env python3
"""Resumable multi-seed orchestrator for the SCOPE-FD revision experiments."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs" / "scope_revision_sweeps.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "scope_revision"

NEGATED_FLAGS = {
    "use_amp": "--no-amp",
    "dynamic_steps": "--no-dynamic-steps",
    "performance_mode": "--no-performance-mode",
    "smoke_test_mode": "--no-smoke-test-mode",
}
IGNORED_CONFIG_KEYS = {"notes", "tags"}


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _expand_family(name: str, spec: Dict[str, Any], defaults: Dict[str, Any]):
    base = {**defaults, **spec.get("base", {})}
    axes = spec.get("axes", {})
    axis_names = list(axes)
    axis_values = [values if isinstance(values, list) else [values] for values in axes.values()]
    products = itertools.product(*axis_values) if axis_values else [()]
    cases = spec.get("cases") or [{}]
    seeds = spec.get("seeds", [11, 22, 33])
    methods = spec.get("methods", [])
    for case in cases:
        for product in products:
            config = {**base, **case, **dict(zip(axis_names, product))}
            for seed in seeds:
                yield {
                    "family": name,
                    "methods": methods,
                    "seed": int(seed),
                    "config": {**config, "seed": int(seed)},
                }


def expand_spec(data: Dict[str, Any], selected_families: set[str] | None = None):
    defaults = data.get("defaults", {})
    for name, family in data.get("families", {}).items():
        if selected_families and name not in selected_families:
            continue
        if not family.get("enabled", True) and not selected_families:
            continue
        yield from _expand_family(name, family, defaults)


def _config_args(config: Dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if key in IGNORED_CONFIG_KEYS or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif key in NEGATED_FLAGS:
                args.append(NEGATED_FLAGS[key])
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        args.extend([flag, str(value)])
    return args


def _is_complete(path: Path, methods: Iterable[str]) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", {})
        return all(method in results and results[method].get("metrics") for method in methods)
    except Exception:
        return False


class GPUMonitor:
    def __init__(self, log_path: Path, interval: float = 15.0):
        self.log_path = log_path
        self.interval = max(2.0, float(interval))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        def run():
            header_written = self.log_path.exists()
            while not self.stop_event.is_set():
                try:
                    proc = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        with self.log_path.open("a", encoding="utf-8") as handle:
                            if not header_written:
                                handle.write("timestamp,utilization_gpu_pct,memory_used_mib,memory_total_mib\n")
                                header_written = True
                            handle.write(proc.stdout.strip() + "\n")
                except (FileNotFoundError, subprocess.SubprocessError):
                    return
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _preflight_parallel_clients(config: Dict[str, Any]) -> Dict[str, Any]:
    adjusted = dict(config)
    try:
        from csfl_simulator.core.utils import check_memory_critical
        critical, message = check_memory_critical(threshold_percent=90.0)
    except Exception:
        critical, message = False, ""
    if critical:
        requested = int(adjusted.get("parallel_clients", -1))
        adjusted["parallel_clients"] = max(1, requested // 2) if requested > 1 else 1
        accumulation = max(1, int(adjusted.get("grad_accum_steps", 1)))
        adjusted["grad_accum_steps"] = max(2, accumulation * 2)
        print(
            f"[preflight] {message}; parallel_clients={adjusted['parallel_clients']}, "
            f"grad_accum_steps={adjusted['grad_accum_steps']}",
            file=sys.stderr,
        )
    return adjusted


def run_one(job: Dict[str, Any], output_root: Path, dry_run: bool, monitor_interval: float):
    identity = {
        "family": job["family"],
        "methods": job["methods"],
        "config": job["config"],
    }
    digest = _stable_hash(identity)
    run_dir = output_root / job["family"] / digest
    result_path = run_dir / "compare_results.json"
    if _is_complete(result_path, job["methods"]):
        return "skipped", run_dir

    config = _preflight_parallel_clients(job["config"])
    run_name = f"scope_rev_{job['family']}_{digest}"
    command = [
        sys.executable,
        "-m",
        "csfl_simulator",
        "compare",
        "--methods",
        ",".join(job["methods"]),
        "--name",
        run_name,
        "--output",
        str(result_path),
        *_config_args(config),
    ]
    if dry_run:
        print(subprocess.list2cmdline(command))
        return "dry-run", run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **identity,
        "resolved_config": config,
        "hash": digest,
        "command": command,
        "status": "running",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    monitor = GPUMonitor(run_dir / "gpu_usage.csv", monitor_interval)
    monitor.start()
    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    finally:
        monitor.stop()
        gc.collect()
        try:
            from csfl_simulator.core.utils import cleanup_memory
            cleanup_memory(force_cuda_empty=True)
        except Exception:
            pass

    manifest["returncode"] = proc.returncode
    manifest["status"] = "complete" if proc.returncode == 0 else "failed"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest["status"], run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--family", action="append", help="Run only this family; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-seeds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu-monitor-interval", type=float, default=15.0)
    args = parser.parse_args()

    data = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    selected = set(args.family) if args.family else None
    jobs = list(expand_spec(data, selected))
    print(f"Expanded {len(jobs)} runs from {args.spec}")

    workers = args.parallel_seeds
    if workers == 2:
        try:
            from csfl_simulator.core.utils import get_memory_info
            gpu = get_memory_info()
            free_fraction = gpu.get("gpu_free_gb", 0) / max(gpu.get("gpu_total_gb", 1), 1e-9)
            if free_fraction <= 0.5:
                print("[preflight] <=50% GPU memory free; falling back to sequential seeds")
                workers = 1
        except Exception:
            workers = 1

    results = []
    if workers == 1:
        for job in jobs:
            status, run_dir = run_one(
                job, args.output_root, args.dry_run, args.gpu_monitor_interval
            )
            print(f"[{status}] {run_dir}")
            results.append(status)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_one, job, args.output_root, args.dry_run, args.gpu_monitor_interval
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                status, run_dir = future.result()
                print(f"[{status}] {run_dir}")
                results.append(status)

    failures = sum(status == "failed" for status in results)
    if failures:
        raise SystemExit(f"{failures} run(s) failed; inspect per-run stderr.log")


if __name__ == "__main__":
    main()
