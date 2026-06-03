#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_audio_fsdd}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
NO_HARDWARE_METER="${NO_HARDWARE_METER:-1}"
COUNTRY_ISO_CODE="${COUNTRY_ISO_CODE:-IND}"
GRID_INTENSITY="${GRID_INTENSITY:-475}"

CONFIG="${SCRIPT_DIR}/configs.yaml"
RUNS_DIR="${REPO_ROOT}/runs/audio_fsdd"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts/audio_fsdd"
ANALYSIS_DIR="${ARTIFACTS_DIR}/analysis"
PLOTS_DIR="${ARTIFACTS_DIR}/plots"
LOG_DIR="${RUNS_DIR}/logs"
LOG_FILE="${LOG_DIR}/audio_fsdd_100r.log"

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}" "${PLOTS_DIR}"
cd "${REPO_ROOT}"

echo "Audio FSDD 100-round pipeline"
echo "Repo:       ${REPO_ROOT}"
echo "Config:     ${CONFIG}"
echo "Runs:       ${RUNS_DIR}"
echo "Artifacts:  ${ARTIFACTS_DIR}"
echo "Device:     ${DEVICE}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r csfl_simulator/experiments/maml_select/requirements.txt

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
PY

echo "Preparing FSDD dataset cache..."
python - <<'PY'
from csfl_simulator.core.datasets import get_full_data
train, test = get_full_data("FSDD")
print(f"FSDD ready: train={len(train)} test={len(test)} classes={len(train.classes)}")
PY

METER_ARGS=()
if [[ "${NO_HARDWARE_METER}" == "1" ]]; then
  METER_ARGS+=(--no-hardware-meter)
fi

{
  echo "Started at $(date)"
  python -m csfl_simulator.experiments.maml_select.run_experiments \
    --config "${CONFIG}" \
    --profile audio_fsdd \
    --device "${DEVICE}" \
    --output-dir "${RUNS_DIR}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --country-iso-code "${COUNTRY_ISO_CODE}" \
    --grid-intensity "${GRID_INTENSITY}" \
    --resume \
    "${METER_ARGS[@]}"

  python -m csfl_simulator.experiments.maml_select.analyze_results \
    --results-dir "${RUNS_DIR}" \
    --output-dir "${ANALYSIS_DIR}"

  python -m csfl_simulator.experiments.audio_fsdd.plot_audio_fsdd \
    --results-dir "${RUNS_DIR}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --plots-dir "${PLOTS_DIR}"

  echo "Finished at $(date)"
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Audio FSDD pipeline complete."
echo "Logs:      ${LOG_FILE}"
echo "Results:   ${RUNS_DIR}"
echo "Analysis:  ${ANALYSIS_DIR}"
echo "Plots:     ${PLOTS_DIR}"
