#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_DIR="${REPO_ROOT}/runs/maml_select_cifar100"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts/maml_select_cifar100"
ANALYSIS_DIR="${ARTIFACTS_DIR}/analysis"
PLOTS_DIR="${ARTIFACTS_DIR}/plots"
LOG_DIR="${RUNS_DIR}/logs"
EXTRA_RUN_ARGS=()
if [[ -n "${MAML_SELECT_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_RUN_ARGS <<< "${MAML_SELECT_EXTRA_ARGS}"
fi

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}" "${PLOTS_DIR}"
cd "${REPO_ROOT}"
# Plain FL uses the standard ResNet18 BatchNorm layers. The shared model factory
# otherwise enables GroupNorm for the repository's separate distillation studies.
export CSFL_KEEP_BN=1

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable. Run this campaign from a normal macOS Terminal, outside the Codex sandbox.")
x = torch.ones(1, device="mps")
print(f"MPS ready: {x.device}")
PY

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile cifar100 \
  --device mps \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --country-iso-code IND \
  --grid-intensity 475 \
  --resume \
  "${EXTRA_RUN_ARGS[@]}"

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.analyze_results \
  --results-dir "${RUNS_DIR}" \
  --output-dir "${ANALYSIS_DIR}"

"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.generate_plots \
  --results-dir "${RUNS_DIR}" \
  --output-dir "${PLOTS_DIR}"
