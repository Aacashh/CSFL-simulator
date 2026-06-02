"""Run the additive MAML-Select resubmission experiment matrix."""
from __future__ import annotations

import argparse
import json
import random
import re
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs.yaml"
DEFAULT_OUTPUT = HERE.parents[2] / "artifacts" / "maml_select_letter"
MAML_MODULE = "csfl_simulator.experiments.maml_select.selector"
CRITICALFL_MODULE = "csfl_simulator.experiments.maml_select.criticalfl"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _load(path: Path) -> Dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def _methods(experiment: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = experiment.get("method_variants")
    if variants:
        return [
            {"key": item["key"], "label": item.get("label", item["key"]), "params": item.get("params", {})}
            for item in variants
        ]
    keys = experiment.get("methods", config.get(experiment.get("methods_from", ""), []))
    return [{"key": key, "label": key, "params": {}} for key in keys]


def build_matrix(config: Dict[str, Any], profile: str, only: Iterable[str], seeds: List[int] | None) -> List[Dict[str, Any]]:
    requested = set(only)
    matrix: List[Dict[str, Any]] = []
    for experiment in config["experiments"]:
        if profile not in experiment["profiles"]:
            continue
        if requested and experiment["id"] not in requested:
            continue
        experiment_seeds = seeds if seeds else experiment["seeds"]
        for scenario_name in experiment["scenarios"]:
            scenario = dict(config["defaults"])
            scenario.update(config["scenarios"][scenario_name])
            for seed in experiment_seeds:
                methods = _methods(experiment, config)
                random.Random(f"{experiment['id']}:{scenario_name}:{seed}").shuffle(methods)
                for method in methods:
                    matrix.append(
                        {
                            "experiment_id": experiment["id"],
                            "scenario_name": scenario_name,
                            "scenario": scenario,
                            "seed": int(seed),
                            "method_key": method["key"],
                            "method_label": method["label"],
                            "method_params": method["params"],
                        }
                    )
    return matrix


def _register_research_methods(simulator: Any, config: Dict[str, Any], item: Dict[str, Any]) -> None:
    simulator.registry.register(
        "research.maml_select",
        MAML_MODULE,
        params=dict(config["maml_select"]),
        display_name="MAML-Select",
        origin="resubmission experiment suite",
    )
    simulator.registry.register(
        "research.criticalfl",
        CRITICALFL_MODULE,
        params=dict(config["criticalfl"]),
        display_name="CriticalFL cohort augmentation",
        origin="CriticalFL Algorithm 2 reproduction",
    )
    if item["method_key"].startswith("research.maml_select."):
        params = dict(config["maml_select"])
        params.update(item["method_params"])
        simulator.registry.register(
            item["method_key"],
            MAML_MODULE,
            params=params,
            display_name=item["method_label"],
            origin="MAML-Select experiment variant",
        )


def _sim_config(item: Dict[str, Any], device: str) -> Any:
    from csfl_simulator.core.simulator import SimConfig

    values = dict(item["scenario"])
    values.update(
        {
            "seed": item["seed"],
            "device": device,
            "name": _slug(f"maml_select_{item['experiment_id']}_{item['scenario_name']}_{item['method_key']}_s{item['seed']}"),
        }
    )
    return SimConfig(**values)


def _run_label(item: Dict[str, Any]) -> str:
    return _slug(f"{item['experiment_id']}__{item['scenario_name']}__{item['method_key']}__seed_{item['seed']}")


def run_one(item: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from csfl_simulator.experiments.maml_select.energy import CodeCarbonMeter
    from csfl_simulator.experiments.maml_select.simulator import InstrumentedFLSimulator

    run_label = _run_label(item)
    output_dir = args.output_dir / run_label
    result_path = output_dir / "result.json"
    if args.resume and result_path.exists():
        print(f"[skip] {run_label}")
        return {"run_label": run_label, "status": "skipped", "result_path": str(result_path)}

    print(f"[run ] {run_label}")
    simulator = InstrumentedFLSimulator(
        _sim_config(item, args.device),
        grid_carbon_g_per_kwh=args.grid_intensity,
        credit_batches=args.credit_batches,
    )
    _register_research_methods(simulator, config, item)
    simulator.setup()
    meter = CodeCarbonMeter(
        output_dir=output_dir,
        run_label=run_label,
        country_iso_code=args.country_iso_code,
        grid_intensity_g_per_kwh=args.grid_intensity,
        enabled=not args.no_hardware_meter,
        measure_power_secs=args.measure_power_secs,
    )
    meter.start()
    try:
        simulation = simulator.run(item["method_key"])
    finally:
        hardware_energy = meter.stop()
        simulator.cleanup()

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "experiment_id": item["experiment_id"],
        "scenario_name": item["scenario_name"],
        "method_key": item["method_key"],
        "method_label": item["method_label"],
        "method_params": item["method_params"],
        "seed": item["seed"],
        "hardware_energy": hardware_energy,
        "simulation": simulation,
    }
    _dump(payload, result_path)
    return {"run_label": run_label, "status": "completed", "result_path": str(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("quick", "core", "full"), default="core")
    parser.add_argument("--only", action="append", default=[], metavar="EXPERIMENT_ID")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--device", default="auto", help="Simulator device, for example cuda or cpu.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--country-iso-code", default="IND", help="Three-letter ISO code for CodeCarbon offline mode.")
    parser.add_argument("--grid-intensity", type=float, default=475.0, help="Declared grid intensity in gCO2eq/kWh.")
    parser.add_argument("--credit-batches", type=int, default=1)
    parser.add_argument("--measure-power-secs", type=int, default=1)
    parser.add_argument("--no-hardware-meter", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without loading data or training.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load(args.config)
    matrix = build_matrix(config, args.profile, args.only, args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "config": str(args.config),
        "country_iso_code": args.country_iso_code,
        "declared_grid_intensity_g_per_kwh": args.grid_intensity,
        "hardware_meter_enabled": not args.no_hardware_meter,
        "matrix": matrix,
    }
    _dump(manifest, args.output_dir / f"manifest_{args.profile}.json")
    print(f"Prepared {len(matrix)} run(s) for profile={args.profile}.")
    if args.dry_run:
        for item in matrix:
            print(f"  {_run_label(item)}")
        return

    statuses = []
    for item in matrix:
        try:
            statuses.append(run_one(item, config, args))
        except Exception as exc:
            error = {
                "run_label": _run_label(item),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            statuses.append(error)
            print(f"[fail] {error['run_label']}: {exc}")
            if args.fail_fast:
                break
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "statuses": statuses,
    }
    _dump(summary, args.output_dir / f"summary_{args.profile}.json")
    completed = sum(item["status"] == "completed" for item in statuses)
    skipped = sum(item["status"] == "skipped" for item in statuses)
    failed = sum(item["status"] == "failed" for item in statuses)
    print(f"Finished: completed={completed}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
