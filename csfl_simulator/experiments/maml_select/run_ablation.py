"""Feature ablation study for MAML-Select state vector.

Runs separate trials where one feature is masked out from the 6-dimensional
state vector s_{i,t} = (loss, grad_norm, latency, battery, frequency, staleness)
to quantify each feature's individual contribution to the meta-policy.

Usage:
    python -m csfl_simulator.experiments.maml_select.run_ablation
    python -m csfl_simulator.experiments.maml_select.run_ablation --device cpu --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs.yaml"
DEFAULT_OUTPUT = HERE.parents[2] / "artifacts" / "maml_select_letter" / "ablation"
MAML_MODULE = "csfl_simulator.experiments.maml_select.selector"

ABLATION_VARIANTS = [
    {"label": "All Features", "disabled_features": []},
    {"label": "w/o Loss", "disabled_features": ["loss"]},
    {"label": "w/o Gradient Norm", "disabled_features": ["grad_norm"]},
    {"label": "w/o Latency", "disabled_features": ["latency"]},
    {"label": "w/o Battery", "disabled_features": ["battery"]},
    {"label": "w/o Frequency", "disabled_features": ["frequency"]},
    {"label": "w/o Staleness", "disabled_features": ["staleness"]},
]

DEFAULT_SEEDS = [42, 123, 2026]


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open() as f:
        return yaml.safe_load(f)


def run_ablation(args: argparse.Namespace) -> None:
    from csfl_simulator.experiments.maml_select.simulator import InstrumentedFLSimulator
    from csfl_simulator.experiments.maml_select.seeding import set_global_seed
    from csfl_simulator.core.simulator import SimConfig

    config = _load_config(args.config)
    scenario = dict(config["defaults"])
    scenario.update(config["scenarios"].get(args.scenario, {}))
    seeds = args.seeds or DEFAULT_SEEDS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_runs = len(ABLATION_VARIANTS) * len(seeds)
    run_idx = 0

    for variant in ABLATION_VARIANTS:
        for seed in seeds:
            run_idx += 1
            disabled = variant["disabled_features"]
            label = variant["label"]
            print(f"\n[{run_idx}/{total_runs}] {label}, Seed={seed}")

            set_global_seed(seed, deterministic=True)

            sim_values = dict(scenario)
            for key in ("report_accuracy_target", "stop_on_accuracy_target",
                        "cifar10_augment", "lr_scheduler", "lr_warmup_rounds"):
                sim_values.pop(key, None)
            sim_values.update({
                "seed": seed,
                "device": args.device,
                "name": f"ablation_{label.replace(' ', '_').replace('/', '_')}_s{seed}",
            })

            try:
                sim = InstrumentedFLSimulator(
                    SimConfig(**sim_values),
                    report_accuracy_target=scenario.get("report_accuracy_target"),
                )
                maml_params = dict(config["maml_select"])
                maml_params["disabled_features"] = disabled
                sim.registry.register(
                    "research.maml_select",
                    MAML_MODULE,
                    params=maml_params,
                    display_name=label,
                    origin="feature ablation",
                )
                sim.setup()
                result = sim.run("research.maml_select")
                final = result["metrics"][-1]
                row = {
                    "variant": label,
                    "disabled_features": ",".join(disabled) if disabled else "none",
                    "seed": seed,
                    "final_accuracy": float(final.get("accuracy", 0)),
                    "cum_training_tflops": float(final.get("cum_training_tflops", 0)),
                    "cum_time": float(final.get("cum_time", 0)),
                    "fairness_gini": float(final.get("fairness_gini", 0)),
                    "rounds_completed": int(result.get("rounds_completed", 0)),
                }
                results.append(row)
                print(f"  -> Accuracy: {row['final_accuracy']:.4f}")
                sim.cleanup()
            except Exception as e:
                print(f"  [FAIL] {e}")
                traceback.print_exc()
                results.append({
                    "variant": label,
                    "disabled_features": ",".join(disabled) if disabled else "none",
                    "seed": seed,
                    "error": str(e),
                })

    # Save raw results
    with (output_dir / "ablation_results.json").open("w") as f:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "results": results}, f, indent=2)

    # Save CSV
    try:
        import pandas as pd
        df = pd.DataFrame([r for r in results if "error" not in r])
        df.to_csv(output_dir / "ablation_results.csv", index=False)

        if not df.empty:
            summary = df.groupby("variant").agg(
                accuracy_mean=("final_accuracy", "mean"),
                accuracy_std=("final_accuracy", "std"),
                tflops_mean=("cum_training_tflops", "mean"),
            ).reset_index()
            summary.to_csv(output_dir / "ablation_summary.csv", index=False)
            print("\n--- Ablation Summary ---")
            print(summary.to_string(index=False))
    except ImportError:
        print("[warn] pandas not available; only JSON output saved.")

    print(f"\nResults saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario", default="fashion_main")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        from csfl_simulator.core.utils import autodetect_device
        args.device = autodetect_device()
    run_ablation(args)


if __name__ == "__main__":
    main()
