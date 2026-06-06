#!/usr/bin/env bash
set -euo pipefail

# Complete the remaining reviewer-evidence experiments after skipping
# MAML-Select v2 and avoiding unnecessary reruns of slow baselines. The
# intended flow is: finish CriticalFL seed 42, reuse the existing CriticalFL
# seed 123 round-150 trace, then run FMNIST sensitivity/ablation and scaling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-mps}"
COUNTRY_ISO_CODE="${COUNTRY_ISO_CODE:-IND}"
GRID_INTENSITY="${GRID_INTENSITY:-475}"
SCALING_ROUNDS="${SCALING_ROUNDS:-10}"

RUNS_CIFAR100="${RUNS_CIFAR100:-${REPO_ROOT}/runs/maml_select_cifar100}"
ARTIFACTS_CIFAR100="${ARTIFACTS_CIFAR100:-${REPO_ROOT}/artifacts/maml_select_cifar100}"
RUNS_MAIN="${RUNS_MAIN:-${REPO_ROOT}/runs/maml_select}"
ARTIFACTS_MAIN="${ARTIFACTS_MAIN:-${REPO_ROOT}/artifacts/maml_select}"
LOG_DIR="${RUNS_MAIN}/logs"

mkdir -p "${RUNS_CIFAR100}/logs" "${ARTIFACTS_CIFAR100}/analysis" "${ARTIFACTS_CIFAR100}/plots"
mkdir -p "${RUNS_MAIN}/sensitivity" "${RUNS_MAIN}/ablation" "${RUNS_MAIN}/scaling"
mkdir -p "${ARTIFACTS_MAIN}/analysis" "${ARTIFACTS_MAIN}/plots" "${LOG_DIR}"

cd "${REPO_ROOT}"
export CSFL_KEEP_BN=1

echo "Using device=${DEVICE}. The experiment runner will initialize the backend."

PID_FILE="${RUNS_CIFAR100}/logs/cifar100_mps_200.pid"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(tr -cd '0-9' < "${PID_FILE}")"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null && [[ "${ALLOW_WHILE_CIFAR100_ACTIVE:-0}" != "1" ]]; then
    echo "The CIFAR-100 campaign still appears active (PID ${PID})."
    echo "This launcher is designed for after that run completes."
    echo "Re-run this script after the v2 campaign finishes."
    echo "Override only if you are certain: ALLOW_WHILE_CIFAR100_ACTIVE=1 bash $0"
    exit 2
  fi
fi

HW_ARGS=()
if [[ "${NO_HARDWARE_METER:-0}" == "1" ]]; then
  HW_ARGS+=(--no-hardware-meter)
fi

EXTRA_ARGS=()
if [[ -n "${MAML_SELECT_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${MAML_SELECT_EXTRA_ARGS}"
fi

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/maml_select_cifar100_150.XXXXXX")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - "${SCRIPT_DIR}/configs.yaml" "${TMP_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
with source.open() as handle:
    config = yaml.safe_load(handle)

# The CIFAR-100 paper table will be read at row round=150. The simulator uses
# zero-based round indices, so 151 rounds makes round 150 the final result.
config["defaults"]["rounds"] = 151
config["defaults"]["eval_every"] = 10

with target.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

run_experiment() {
  "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments "$@"
}

mark_skipped_cifar100_run() {
  local run_label="$1"
  local reason="$2"
  mkdir -p "${RUNS_CIFAR100}/${run_label}"
  printf '%s\n' "${reason}" > "${RUNS_CIFAR100}/${run_label}/.skip_run"
}

materialize_criticalfl_s123_round150() {
  "${PYTHON_BIN}" - "${RUNS_CIFAR100}" <<'PY'
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
run_label = "cifar100_benchmarks_cifar100_main_criticalfl_s123"
run_dir = root / run_label
jsonl_path = run_dir / "round_metrics.jsonl"
result_path = run_dir / "result.json"

if result_path.exists():
    print(f"[skip] {run_label}: result.json already exists")
    raise SystemExit(0)

if not jsonl_path.exists():
    raise SystemExit(f"Cannot materialize {run_label}; missing {jsonl_path}")

rows = []
with jsonl_path.open() as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        round_idx = int(row.get("round", -1))
        if 0 <= round_idx <= 150:
            rows.append(row)

if not rows or int(rows[-1].get("round", -1)) != 150:
    raise SystemExit(f"Cannot materialize {run_label}; round 150 is not available")

template_path = root / "cifar100_benchmarks_cifar100_main_fedavg_s123" / "result.json"
if not template_path.exists():
    template_path = root / "cifar100_benchmarks_cifar100_main_maml_select_s123" / "result.json"
if not template_path.exists():
    raise SystemExit("Cannot materialize CriticalFL seed 123; no CIFAR-100 seed-123 result template found")

with template_path.open() as handle:
    template = json.load(handle)

config = dict(template["simulation"]["config"])
config.update(
    {
        "dataset": "CIFAR-100",
        "model": "ResNet18",
        "rounds": 151,
        "seed": 123,
        "name": "maml_select_cifar100_benchmarks_cifar100_main_research.criticalfl_s123",
    }
)

selected_history = [list(map(int, row.get("selected_clients", []))) for row in rows]
counts = [0 for _ in range(int(config.get("total_clients", 100)))]
for ids in selected_history:
    for cid in ids:
        if 0 <= cid < len(counts):
            counts[cid] += 1

final = rows[-1]
base_accuracy = float(rows[0].get("accuracy", final.get("accuracy", 0.0)))
final_accuracy = float(final.get("accuracy", base_accuracy))
improvement = max(0.0, final_accuracy - base_accuracy)
threshold = base_accuracy + 0.8 * improvement
time_to_80pct = math.nan
for row in rows:
    if float(row.get("accuracy", 0.0)) >= threshold:
        time_to_80pct = float(row.get("cum_time", math.nan))
        break
final["time_to_80pct_final"] = time_to_80pct
final["mean_cohort_size"] = float(sum(len(ids) for ids in selected_history) / max(1, len(selected_history)))

seed_record_path = run_dir / "seed_record.json"
if seed_record_path.exists():
    with seed_record_path.open() as handle:
        seed_record = json.load(handle)
else:
    seed_record = dict(template.get("seed_record", {}))
    seed_record["seed"] = 123

simulation = {
    "run_id": template["simulation"].get("run_id", run_label),
    "run_dir": str(run_dir),
    "metrics": rows,
    "config": config,
    "device": template["simulation"].get("device", "mps"),
    "stopped_early": False,
    "stop_on_accuracy_target": False,
    "report_accuracy_target": None,
    "rounds_completed": len(rows),
    "method": "research.criticalfl",
    "history": {"selected": selected_history},
    "participation_counts": counts,
    "modelled_energy_assumptions": template["simulation"].get("modelled_energy_assumptions", {}),
    "training_protocol": template["simulation"].get("training_protocol", {}),
}

payload = {
    "schema_version": 2,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_label": run_label,
    "experiment_id": "cifar100_benchmarks",
    "scenario_name": "cifar100_main",
    "method_key": "research.criticalfl",
    "method_label": "research.criticalfl",
    "method_params": {},
    "seed": 123,
    "training_protocol": template.get("training_protocol", {}),
    "maml_select_protocol": template.get("maml_select_protocol", {}),
    "maml_select_v2_protocol": template.get("maml_select_v2_protocol", {}),
    "hardware_energy": {
        "status": "not_measured",
        "note": "Materialized from existing round_metrics.jsonl through round 150; no rerun was performed.",
    },
    "simulation": simulation,
    "seed_record": seed_record,
}

result_path.parent.mkdir(parents=True, exist_ok=True)
with result_path.open("w") as handle:
    json.dump(payload, handle, indent=2, allow_nan=True)
print(f"[made] {result_path}")
PY
}

echo
echo "============================================================"
echo "  Stage 1: Targeted CIFAR-100 150-round formal result files"
echo "============================================================"
mark_skipped_cifar100_run \
  "cifar100_benchmarks_cifar100_main_tifl_s123" \
  "Skipped by revision decision; use prior TiFL evidence only if needed."
mark_skipped_cifar100_run \
  "cifar100_benchmarks_cifar100_main_fedcor_s123" \
  "Skipped by revision decision; use prior FedCor evidence only if needed."
materialize_criticalfl_s123_round150

for spec in \
  "42 research.criticalfl"
do
  read -r seed method_key <<< "${spec}"
  run_experiment \
    --config "${TMP_CONFIG}" \
    --profile cifar100 \
    --device "${DEVICE}" \
    --output-dir "${RUNS_CIFAR100}" \
    --analysis-dir "${ARTIFACTS_CIFAR100}/analysis" \
    --country-iso-code "${COUNTRY_ISO_CODE}" \
    --grid-intensity "${GRID_INTENSITY}" \
    --seed "${seed}" \
    --method-key "${method_key}" \
    --resume \
    ${HW_ARGS[@]+"${HW_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done

echo
echo "============================================================"
echo "  Stage 2: Fashion-MNIST lambda sensitivity"
echo "============================================================"
run_experiment \
  --profile core \
  --device "${DEVICE}" \
  --output-dir "${RUNS_MAIN}" \
  --analysis-dir "${ARTIFACTS_MAIN}/analysis" \
  --country-iso-code "${COUNTRY_ISO_CODE}" \
  --grid-intensity "${GRID_INTENSITY}" \
  --only lambda_sensitivity \
  --resume \
  ${HW_ARGS[@]+"${HW_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "============================================================"
echo "  Stage 3: Fashion-MNIST feature ablation"
echo "============================================================"
run_experiment \
  --profile core \
  --device "${DEVICE}" \
  --output-dir "${RUNS_MAIN}" \
  --analysis-dir "${ARTIFACTS_MAIN}/analysis" \
  --country-iso-code "${COUNTRY_ISO_CODE}" \
  --grid-intensity "${GRID_INTENSITY}" \
  --only feature_ablation \
  --resume \
  ${HW_ARGS[@]+"${HW_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "============================================================"
echo "  Stage 4: Lightweight scaling overhead"
echo "============================================================"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_scaling \
  --device "${DEVICE}" \
  --rounds "${SCALING_ROUNDS}" \
  --output-dir "${RUNS_MAIN}/scaling"

echo
echo "============================================================"
echo "  Stage 5: Regenerate analysis tables and EPS plots"
echo "============================================================"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.analyze_results \
  --results-dir "${RUNS_CIFAR100}" \
  --output-dir "${ARTIFACTS_CIFAR100}/analysis"

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.generate_plots \
  --results-dir "${RUNS_CIFAR100}" \
  --output-dir "${ARTIFACTS_CIFAR100}/plots" \
  --scaling-dir "${RUNS_MAIN}/scaling"

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.analyze_results \
  --results-dir "${RUNS_MAIN}" \
  --output-dir "${ARTIFACTS_MAIN}/analysis"

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.generate_plots \
  --results-dir "${RUNS_MAIN}" \
  --output-dir "${ARTIFACTS_MAIN}/plots" \
  --scaling-dir "${RUNS_MAIN}/scaling"

"${PYTHON_BIN}" - <<'PY'
import csv
import json
from pathlib import Path

repo = Path.cwd()
root = repo / "runs" / "maml_select_cifar100"
out = repo / "artifacts" / "maml_select_cifar100" / "analysis" / "cifar100_round150_matrix_status.csv"

targets = {
    42: [
        ("baseline.fedavg", "fedavg"),
        ("system_aware.fedcs", "fedcs"),
        ("system_aware.oort", "oort"),
        ("system_aware.tifl", "tifl"),
        ("ml.fedcor", "fedcor"),
        ("research.criticalfl", "criticalfl"),
        ("research.fedgcs", "fedgcs"),
        ("research.maml_select", "maml_select"),
    ],
    123: [
        ("baseline.fedavg", "fedavg"),
        ("system_aware.fedcs", "fedcs"),
        ("system_aware.oort", "oort"),
        ("system_aware.tifl", "tifl"),
        ("ml.fedcor", "fedcor"),
        ("research.criticalfl", "criticalfl"),
        ("research.fedgcs", "fedgcs"),
        ("research.maml_select", "maml_select"),
    ],
    2026: [
        ("baseline.fedavg", "fedavg"),
        ("research.maml_select", "maml_select"),
    ],
}

rows = []
for seed, methods in targets.items():
    for method_key, slug in methods:
        run_dir = root / f"cifar100_benchmarks_cifar100_main_{slug}_s{seed}"
        result_path = run_dir / "result.json"
        metric = None
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            metrics = payload.get("simulation", {}).get("metrics", [])
            by_round = {int(item.get("round", -999)): item for item in metrics if int(item.get("round", -999)) >= 0}
            metric = by_round.get(150) or (metrics[-1] if metrics else None)
        status = "ready" if metric and int(metric.get("round", -1)) >= 150 else "missing"
        rows.append({
            "seed": seed,
            "method_key": method_key,
            "status": status,
            "round": "" if metric is None else metric.get("round", ""),
            "accuracy": "" if metric is None else metric.get("accuracy", ""),
            "cum_modelled_energy_wh": "" if metric is None else metric.get("cum_modelled_energy_wh", ""),
            "cum_modelled_carbon_g": "" if metric is None else metric.get("cum_modelled_carbon_g", ""),
            "result_path": str(result_path),
        })

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote CIFAR-100 round-150 status matrix to {out}")
PY

echo
echo "Remaining reviewer-evidence launcher finished."
echo "CIFAR-100 analysis: ${ARTIFACTS_CIFAR100}/analysis"
echo "CIFAR-100 plots:    ${ARTIFACTS_CIFAR100}/plots"
echo "Reviewer analysis:  ${ARTIFACTS_MAIN}/analysis"
echo "Reviewer plots:     ${ARTIFACTS_MAIN}/plots"
echo "Scaling CSVs:       ${RUNS_MAIN}/scaling"
