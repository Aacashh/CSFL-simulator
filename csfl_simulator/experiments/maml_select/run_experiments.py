"""Run the additive MAML-Select resubmission experiment matrix.

Extended for the revision to include seven integrated baselines/selectors
(FedAvg, FedCS, Oort, TiFL, FedCor approximation, CriticalFL reproduction,
FedGCS-style approximation) and MAML-Select, statistical significance tests,
and CIFAR-10 reconciliation CSV output. Exact external reproductions remain
separately identified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs.yaml"
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "maml_select"
DEFAULT_ANALYSIS = REPO_ROOT / "artifacts" / "maml_select" / "analysis"
MAML_MODULE = "csfl_simulator.experiments.maml_select.selector"
MAML_V2_MODULE = "csfl_simulator.experiments.maml_select.selector_v2"
CRITICALFL_MODULE = "csfl_simulator.experiments.maml_select.criticalfl"
FEDGCS_MODULE = "csfl_simulator.experiments.maml_select.fedgcs"
EXPERIMENT_ONLY_SCENARIO_FIELDS = {
    "report_accuracy_target", "stop_on_accuracy_target",
    "cifar10_augment", "lr_scheduler", "lr_warmup_rounds",
}
_METHOD_SHORT = {
    "baseline.fedavg": "fedavg",
    "system_aware.fedcs": "fedcs",
    "system_aware.oort": "oort",
    "system_aware.tifl": "tifl",
    "ml.fedcor": "fedcor",
    "research.criticalfl": "criticalfl",
    "research.fedgcs": "fedgcs",
    "research.maml_select": "maml_select",
    "research.maml_select_v2": "maml_select_v2",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _load(path: Path) -> Dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def _append_jsonl(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, allow_nan=True) + "\n")


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
    maml_params = dict(config["maml_select"])
    maml_v2_params = dict(config["maml_select_v2"])
    if item["method_key"] == "research.maml_select":
        maml_params.update(item.get("method_params", {}))
    if item["method_key"] == "research.maml_select_v2":
        maml_v2_params.update(item.get("method_params", {}))
    simulator.registry.register(
        "research.maml_select",
        MAML_MODULE,
        params=maml_params,
        display_name="MAML-Select",
        origin="resubmission experiment suite",
    )
    simulator.registry.register(
        "research.maml_select_v2",
        MAML_V2_MODULE,
        params=maml_v2_params,
        display_name="MAML-Select v2",
        origin="post-campaign experimental selector",
    )
    simulator.registry.register(
        "research.criticalfl",
        CRITICALFL_MODULE,
        params=dict(config["criticalfl"]),
        display_name="CriticalFL cohort augmentation",
        origin="CriticalFL Algorithm 2 reproduction",
    )
    simulator.registry.register(
        "research.fedgcs",
        FEDGCS_MODULE,
        params=dict(config.get("fedgcs", {})),
        display_name="FedGCS-style (approx.)",
        origin="disclosed in-simulator FedGCS-style approximation; not the official IJCAI code",
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
    if item["method_key"].startswith("research.maml_select_v2."):
        params = dict(config["maml_select_v2"])
        params.update(item["method_params"])
        simulator.registry.register(
            item["method_key"],
            MAML_V2_MODULE,
            params=params,
            display_name=item["method_label"],
            origin="MAML-Select v2 experiment variant",
        )


def _sim_config(item: Dict[str, Any], device: str) -> Any:
    from csfl_simulator.core.simulator import SimConfig

    values = dict(item["scenario"])
    for field in EXPERIMENT_ONLY_SCENARIO_FIELDS:
        values.pop(field, None)
    values.update(
        {
            "seed": item["seed"],
            "device": device,
            "name": _slug(f"maml_select_{item['experiment_id']}_{item['scenario_name']}_{item['method_key']}_s{item['seed']}"),
        }
    )
    return SimConfig(**values)


def _run_label(item: Dict[str, Any]) -> str:
    method_short = _METHOD_SHORT.get(item["method_key"], _slug(item["method_key"]))
    return f"{item['experiment_id']}_{item['scenario_name']}_{method_short}_s{item['seed']}"


def run_one(item: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from csfl_simulator.experiments.maml_select.energy import CodeCarbonMeter
    from csfl_simulator.experiments.maml_select.simulator import InstrumentedFLSimulator
    from csfl_simulator.experiments.maml_select.seeding import (
        set_global_seed, capture_seed_record, save_seed_record,
    )

    run_label = _run_label(item)
    output_dir = args.output_dir / run_label
    result_path = output_dir / "result.json"
    skip_marker = output_dir / ".skip_run"
    if args.resume and skip_marker.exists():
        reason = skip_marker.read_text(encoding="utf-8").strip()
        suffix = f": {reason}" if reason else ""
        print(f"[skip] {run_label}{suffix}")
        return {"run_label": run_label, "status": "skipped", "result_path": str(result_path)}
    if args.resume and result_path.exists():
        print(f"[skip] {run_label}")
        return {"run_label": run_label, "status": "skipped", "result_path": str(result_path)}

    print(f"[run ] {run_label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    round_metrics_path = output_dir / "round_metrics.jsonl"
    for stale_path in (
        output_dir / "progress.json",
        output_dir / f"codecarbon_{run_label}.csv",
    ):
        stale_path.unlink(missing_ok=True)
    round_metrics_path.write_text("")

    # Robust seeding. Default is strict deterministic (bit-for-bit); --performance-mode
    # switches to the CUDA fast path (cuDNN benchmark + TF32) while keeping seed-based
    # reproducibility of high-level results.
    set_global_seed(
        item["seed"],
        deterministic=not args.performance_mode,
        performance_mode=args.performance_mode,
    )

    def record_round(round_idx: int, payload: Dict[str, Any]) -> None:
        metric = dict(payload["metrics"])
        _append_jsonl(metric, round_metrics_path)
        if bool(metric.get("evaluated")) or int(round_idx) == 0:
            _dump(
                {
                    "run_label": run_label,
                    "round": int(round_idx),
                    "rounds": int(item["scenario"]["rounds"]),
                    "latest_metrics": metric,
                },
                output_dir / "progress.json",
            )

    simulator = InstrumentedFLSimulator(
        _sim_config(item, args.device),
        grid_carbon_g_per_kwh=args.grid_intensity,
        credit_batches=args.credit_batches,
        report_accuracy_target=item["scenario"].get("report_accuracy_target"),
        stop_on_accuracy_target=bool(item["scenario"].get("stop_on_accuracy_target", False)),
        cifar10_augment=bool(item["scenario"].get("cifar10_augment", False)),
        lr_scheduler=item["scenario"].get("lr_scheduler"),
        lr_warmup_rounds=int(item["scenario"].get("lr_warmup_rounds", 0)),
        local_optimizer=config["local_training"]["optimizer"],
        local_momentum=float(config["local_training"]["momentum"]),
        local_weight_decay=float(config["local_training"]["weight_decay"]),
        model_initialization=config["local_training"]["model_initialization"],
        scratch_root=output_dir / "_scratch",
    )
    _register_research_methods(simulator, config, item)
    simulator.setup()

    # Save seed record
    seed_record = capture_seed_record(item["seed"], simulator.model)
    save_seed_record(seed_record, output_dir)

    meter = CodeCarbonMeter(
        output_dir=output_dir,
        run_label=run_label,
        country_iso_code=args.country_iso_code,
        grid_intensity_g_per_kwh=args.grid_intensity,
        enabled=not args.no_hardware_meter,
        measure_power_secs=args.measure_power_secs,
        verified_hardware_telemetry=args.verified_hardware_telemetry,
    )
    meter.start()
    try:
        simulation = simulator.run(item["method_key"], on_progress=record_round)
    finally:
        hardware_energy = meter.stop()
        simulator.cleanup()

    payload = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "experiment_id": item["experiment_id"],
        "scenario_name": item["scenario_name"],
        "method_key": item["method_key"],
        "method_label": item["method_label"],
        "method_params": item["method_params"],
        "seed": item["seed"],
        "training_protocol": config["local_training"],
        "maml_select_protocol": config["maml_select"],
        "maml_select_v2_protocol": config.get("maml_select_v2", {}),
        "hardware_energy": hardware_energy,
        "simulation": simulation,
        "seed_record": seed_record.to_dict(),
    }
    _dump(payload, result_path)
    return {"run_label": run_label, "status": "completed", "result_path": str(result_path)}


def _compute_aggregate_stats(statuses: List[Dict], config: Dict, args: argparse.Namespace) -> None:
    """Post-run: compute mean±std and paired t-tests, export CSVs."""
    try:
        import pandas as pd
        from scipy.stats import ttest_rel
    except ImportError:
        print("[warn] pandas or scipy not available; skipping aggregate stats.")
        return

    rows = []
    for status in statuses:
        if status.get("status") != "completed":
            continue
        result_path = Path(status["result_path"])
        if not result_path.exists():
            continue
        try:
            with result_path.open() as f:
                payload = json.load(f)
            final = payload["simulation"]["metrics"][-1]
            sim_config = payload["simulation"]["config"]
            rows.append({
                "experiment_id": payload["experiment_id"],
                "scenario_name": payload["scenario_name"],
                "method_key": payload["method_key"],
                "method_label": payload.get("method_label", ""),
                "seed": int(payload["seed"]),
                "dataset": sim_config["dataset"],
                "model": sim_config["model"],
                "final_accuracy": float(final.get("accuracy", 0)),
                "cum_training_tflops": float(final.get("cum_training_tflops", 0)),
                "cum_time": float(final.get("cum_time", 0)),
                "cum_comm_mb": float(final.get("cum_comm_mb", 0)),
                "rounds_completed": int(payload["simulation"].get("rounds_completed", 0)),
                "fairness_gini": float(final.get("fairness_gini", 0)),
                "fairness_jain": float(final.get("fairness_jain", 0)),
            })
        except Exception as e:
            print(f"[warn] Could not parse {result_path}: {e}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    output_dir = args.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary table: mean ± std ---
    summary = df.groupby(["experiment_id", "scenario_name", "method_key"]).agg(
        accuracy_mean=("final_accuracy", "mean"),
        accuracy_std=("final_accuracy", "std"),
        tflops_mean=("cum_training_tflops", "mean"),
        tflops_std=("cum_training_tflops", "std"),
        rounds_mean=("rounds_completed", "mean"),
        rounds_std=("rounds_completed", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)

    # --- Statistical significance tests ---
    test_rows = []
    for (exp_id, scenario), group in df.groupby(["experiment_id", "scenario_name"]):
        maml = group[group["method_key"] == "research.maml_select"].set_index("seed")
        fedavg = group[group["method_key"] == "baseline.fedavg"].set_index("seed")
        for ref_key, ref_name, ref_data in [
            ("research.maml_select", "MAML-Select", maml),
            ("baseline.fedavg", "FedAvg", fedavg),
        ]:
            for method_key, baseline in group.groupby("method_key"):
                if method_key == ref_key:
                    continue
                baseline_indexed = baseline.set_index("seed")
                shared_seeds = sorted(set(ref_data.index) & set(baseline_indexed.index))
                if len(shared_seeds) < 2:
                    continue
                for metric in ["final_accuracy", "cum_training_tflops"]:
                    left = ref_data.loc[shared_seeds, metric].astype(float)
                    right = baseline_indexed.loc[shared_seeds, metric].astype(float)
                    valid = np.isfinite(left) & np.isfinite(right)
                    left, right = left[valid], right[valid]
                    if len(left) < 2:
                        continue
                    stat, pval = ttest_rel(left, right)
                    test_rows.append({
                        "experiment_id": exp_id,
                        "scenario_name": scenario,
                        "reference": ref_name,
                        "baseline": method_key,
                        "metric": metric,
                        "n_paired": int(len(left)),
                        "ref_mean": float(left.mean()),
                        "baseline_mean": float(right.mean()),
                        "t_statistic": float(stat),
                        "p_value": float(pval),
                        "significant_0p05": bool(pval < 0.05),
                    })
    if test_rows:
        tests_df = pd.DataFrame(test_rows)
        tests_df.to_csv(output_dir / "significance_tests.csv", index=False)

    # --- CIFAR-10 reconciliation CSV ---
    cifar_rows = df[df["dataset"] == "CIFAR-10"].copy()
    if not cifar_rows.empty:
        cifar_rows.to_csv(output_dir / "cifar10_reconciled_results.csv", index=False)
        print(f"[info] CIFAR-10 reconciled results saved ({len(cifar_rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("quick", "pilot", "pilot_lambda", "core", "energy", "scaling", "full", "cifar100", "cifar100_v2", "maml_v2", "maml_v2_cpu", "audio_fsdd", "review_hardening", "arch_ablation", "lambda_anchor"),
        default="core",
    )
    parser.add_argument("--only", action="append", default=[], metavar="EXPERIMENT_ID")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--method-key",
        action="append",
        default=[],
        help="Run only the specified method key. May be repeated.",
    )
    parser.add_argument("--device", default="auto", help="Simulator device, for example cuda or cpu.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--country-iso-code", default="IND", help="Three-letter ISO code for CodeCarbon offline mode.")
    parser.add_argument("--grid-intensity", type=float, default=475.0, help="Declared grid intensity in gCO2eq/kWh.")
    parser.add_argument("--credit-batches", type=int, default=1)
    parser.add_argument("--measure-power-secs", type=int, default=1)
    parser.add_argument("--no-hardware-meter", action="store_true")
    parser.add_argument(
        "--verified-hardware-telemetry",
        action="store_true",
        help="Mark CodeCarbon energy as measured only after telemetry is verified on the host.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--start-at-run-label",
        default=None,
        help="Drop earlier matrix entries and begin at this exact run label.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without loading data or training.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="CUDA fast path: enable cuDNN benchmark autotuning + TF32 (the single largest "
             "GPU-throughput win). Seed-based reproducibility of high-level results is preserved, "
             "but results are not bit-for-bit deterministic. No effect on CPU; AMP mixed precision "
             "is already on by default for CUDA. Recommended on NVIDIA GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load(args.config)
    matrix = build_matrix(config, args.profile, args.only, args.seeds)
    if args.method_key:
        requested_methods = set(args.method_key)
        matrix = [item for item in matrix if item["method_key"] in requested_methods]
    if args.start_at_run_label:
        labels = [_run_label(item) for item in matrix]
        try:
            start_index = labels.index(args.start_at_run_label)
        except ValueError as exc:
            available = "\n  ".join(labels)
            raise SystemExit(
                f"Run label {args.start_at_run_label!r} was not found in the matrix.\n"
                f"Available labels:\n  {available}"
            ) from exc
        matrix = matrix[start_index:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "config": str(args.config),
        "country_iso_code": args.country_iso_code,
        "declared_grid_intensity_g_per_kwh": args.grid_intensity,
        "hardware_meter_enabled": not args.no_hardware_meter,
        "verified_hardware_telemetry": args.verified_hardware_telemetry,
        "training_protocol": config["local_training"],
        "maml_select_protocol": config["maml_select"],
        "maml_select_v2_protocol": config.get("maml_select_v2", {}),
        "matrix": matrix,
    }
    _dump(manifest, args.output_dir / f"manifest_{args.profile}.json")
    print(f"Prepared {len(matrix)} run(s) for profile={args.profile}.")
    if args.dry_run:
        methods_seen = set()
        for item in matrix:
            methods_seen.add(item["method_key"])
            print(f"  {_run_label(item)}")
        print(f"\nMethods in matrix ({len(methods_seen)}): {sorted(methods_seen)}")
        return

    statuses = []
    total = len(matrix)
    for idx, item in enumerate(matrix):
        print(f"\n{'='*60}")
        print(f"  Run {idx+1}/{total}")
        print(f"{'='*60}")
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

    # Post-run aggregate statistics
    _compute_aggregate_stats(statuses, config, args)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "statuses": statuses,
    }
    _dump(summary, args.output_dir / f"summary_{args.profile}.json")
    completed = sum(item["status"] == "completed" for item in statuses)
    skipped = sum(item["status"] == "skipped" for item in statuses)
    failed = sum(item["status"] == "failed" for item in statuses)
    print(f"\nFinished: completed={completed}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
