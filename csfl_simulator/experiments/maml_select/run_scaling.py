"""Selection overhead scaling benchmark.

Benchmarks the wall-clock time of client selection for all 8 algorithms as the
client pool scales: N = 100, 250, 500, 1000.  Specifically highlights
MAML-Select's O(N·|φ|) linear scaling vs FedCor's O(N³) cubic scaling.

Usage:
    python -m csfl_simulator.experiments.maml_select.run_scaling
    python -m csfl_simulator.experiments.maml_select.run_scaling --rounds 10 --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs.yaml"
DEFAULT_OUTPUT = HERE.parents[2] / "artifacts" / "maml_select_letter" / "scaling"

MAML_MODULE = "csfl_simulator.experiments.maml_select.selector"
CRITICALFL_MODULE = "csfl_simulator.experiments.maml_select.criticalfl"
FEDGCS_MODULE = "csfl_simulator.experiments.maml_select.fedgcs"

CLIENT_POOL_SIZES = [100, 250, 500, 1000]

ALL_METHODS = [
    "baseline.fedavg",
    "system_aware.fedcs",
    "system_aware.oort",
    "system_aware.tifl",
    "ml.fedcor",
    "research.criticalfl",
    "research.fedgcs",
    "research.maml_select",
]


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open() as f:
        return yaml.safe_load(f)


def run_scaling(args: argparse.Namespace) -> None:
    from csfl_simulator.experiments.maml_select.simulator import InstrumentedFLSimulator
    from csfl_simulator.experiments.maml_select.seeding import set_global_seed
    from csfl_simulator.core.simulator import SimConfig

    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    rounds = args.rounds
    results: List[Dict[str, Any]] = []

    total_runs = len(CLIENT_POOL_SIZES) * len(ALL_METHODS)
    run_idx = 0

    for N in CLIENT_POOL_SIZES:
        K = max(1, N // 10)  # Keep K/N ratio consistent at 10%

        for method_key in ALL_METHODS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] N={N}, K={K}, Method={method_key}")

            set_global_seed(seed, deterministic=True)

            sim_values = dict(config["defaults"])
            sim_values.update({
                "total_clients": N,
                "clients_per_round": K,
                "rounds": rounds,
                "local_epochs": 1,  # Minimal training for overhead measurement
                "eval_every": rounds,  # Only eval at end
                "seed": seed,
                "device": args.device,
                "name": f"scaling_N{N}_{method_key.replace('.', '_')}",
            })

            try:
                sim = InstrumentedFLSimulator(SimConfig(**sim_values))
                # Register research methods
                sim.registry.register(
                    "research.maml_select", MAML_MODULE,
                    params=dict(config["maml_select"]),
                    display_name="MAML-Select", origin="scaling benchmark",
                )
                sim.registry.register(
                    "research.criticalfl", CRITICALFL_MODULE,
                    params=dict(config["criticalfl"]),
                    display_name="CriticalFL", origin="scaling benchmark",
                )
                sim.registry.register(
                    "research.fedgcs", FEDGCS_MODULE,
                    params=dict(config.get("fedgcs", {})),
                    display_name="FedGCS", origin="scaling benchmark",
                )
                sim.setup()

                # Run and collect per-round selection times
                result = sim.run(method_key)
                selection_times = []
                for metric in result["metrics"]:
                    if int(metric.get("round", -1)) >= 0:
                        sel_time = float(metric.get("selection_overhead_seconds", 0))
                        if sel_time > 0:
                            selection_times.append(sel_time)

                mean_sel = float(np.mean(selection_times)) if selection_times else 0.0
                std_sel = float(np.std(selection_times)) if selection_times else 0.0
                total_sel = float(np.sum(selection_times)) if selection_times else 0.0

                row = {
                    "method": method_key,
                    "N": N,
                    "K": K,
                    "rounds": rounds,
                    "mean_selection_seconds": mean_sel,
                    "std_selection_seconds": std_sel,
                    "total_selection_seconds": total_sel,
                    "samples": len(selection_times),
                }
                results.append(row)
                print(f"  -> Mean selection time: {mean_sel*1000:.2f} ms ± {std_sel*1000:.2f} ms")
                sim.cleanup()

            except Exception as e:
                print(f"  [FAIL] {e}")
                traceback.print_exc()
                results.append({
                    "method": method_key,
                    "N": N,
                    "K": K,
                    "rounds": rounds,
                    "error": str(e),
                })

    # Save results
    with (output_dir / "scaling_results.json").open("w") as f:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "rounds_per_run": rounds,
            "results": results,
        }, f, indent=2)

    try:
        import pandas as pd
        df = pd.DataFrame([r for r in results if "error" not in r])
        df.to_csv(output_dir / "scaling_results.csv", index=False)

        if not df.empty:
            # Pivot table for readability
            pivot = df.pivot_table(
                index="method",
                columns="N",
                values="mean_selection_seconds",
                aggfunc="first",
            )
            pivot.to_csv(output_dir / "scaling_pivot.csv")
            print("\n--- Scaling Results (mean selection seconds) ---")
            print(pivot.to_string())
    except ImportError:
        print("[warn] pandas not available; only JSON output saved.")

    print(f"\nResults saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds per run (short for overhead measurement)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        from csfl_simulator.core.utils import autodetect_device
        args.device = autodetect_device()
    run_scaling(args)


if __name__ == "__main__":
    main()
