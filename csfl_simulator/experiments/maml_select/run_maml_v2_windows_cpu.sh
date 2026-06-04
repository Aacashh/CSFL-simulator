#!/usr/bin/env bash
set -euo pipefail

# Windows CPU launcher for the isolated MAML-Select v2 evidence run.
# Intended usage from the repository root in Git Bash:
#   bash csfl_simulator/experiments/maml_select/run_maml_v2_windows_cpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-maml-v2-cpu}"

case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/Scripts/python.exe}" ;;
  *) PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}" ;;
esac

RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/maml_select_v2_windows_cpu}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/maml_select_v2_windows_cpu}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${ARTIFACTS_DIR}/analysis}"
LOG_DIR="${LOG_DIR:-${RUNS_DIR}/logs}"
COUNTRY_ISO="${COUNTRY_ISO:-IND}"
GRID_INTENSITY="${GRID_INTENSITY:-475}"
MAML_V2_SEEDS="${MAML_V2_SEEDS:-42 123 2026}"
ENABLE_CODECARBON="${ENABLE_CODECARBON:-0}"
INSTALL_CPU_TORCH="${INSTALL_CPU_TORCH:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
CSFL_CPU_THREADS="${CSFL_CPU_THREADS:-}"

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}"
LOG_FILE="${LOG_DIR}/maml_v2_windows_cpu_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

find_python() {
  if command -v py >/dev/null 2>&1; then
    printf 'py -3'
  elif command -v python >/dev/null 2>&1; then
    printf 'python'
  elif command -v python3 >/dev/null 2>&1; then
    printf 'python3'
  else
    return 1
  fi
}

create_venv() {
  if [[ -x "${PYTHON_BIN}" ]]; then
    return
  fi
  local base_python
  base_python="$(find_python)" || {
    log "ERROR: Python 3 was not found. Install Python 3.10+ and reopen Git Bash."
    exit 1
  }
  log "Creating virtual environment: ${VENV_DIR}"
  # shellcheck disable=SC2086
  ${base_python} -m venv "${VENV_DIR}"
}

detect_threads() {
  if [[ -n "${CSFL_CPU_THREADS}" ]]; then
    printf '%s' "${CSFL_CPU_THREADS}"
    return
  fi
  "${PYTHON_BIN}" - <<'PY'
import os
logical = os.cpu_count() or 4
# For PyTorch CPU convs on laptops, all logical threads can oversubscribe badly.
# Use a bounded physical-core-ish default; override with CSFL_CPU_THREADS=N.
print(max(2, min(8, logical // 2 if logical > 4 else logical)))
PY
}

cd "${REPO_ROOT}"
log "Repository: ${REPO_ROOT}"
log "Runs:       ${RUNS_DIR}"
log "Artifacts:  ${ARTIFACTS_DIR}"
log "Log:        ${LOG_FILE}"

create_venv

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  log "Installing minimal CPU experiment dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  if [[ "${INSTALL_CPU_TORCH}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  "${PYTHON_BIN}" -m pip install -e . --no-deps
  "${PYTHON_BIN}" -m pip install numpy scipy scikit-learn pandas pyyaml thop codecarbon matplotlib tqdm
else
  log "Skipping dependency installation because SKIP_INSTALL=1"
fi

CSFL_CPU_THREADS="$(detect_threads)"
export CSFL_CPU_THREADS
export OMP_NUM_THREADS="${CSFL_CPU_THREADS}"
export MKL_NUM_THREADS="${CSFL_CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CSFL_CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CSFL_CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CSFL_CPU_THREADS}"
export KMP_BLOCKTIME=0
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=""

if [[ "${CSFL_USE_GROUPNORM:-0}" == "1" ]]; then
  unset CSFL_KEEP_BN
  log "Normalization: GroupNorm replacement enabled for experimental testing"
else
  export CSFL_KEEP_BN=1
  log "Normalization: preserving BatchNorm layers for paper-faithful ResNet18"
fi

log "CPU threads: ${CSFL_CPU_THREADS}"
"${PYTHON_BIN}" - <<'PY' | tee -a "${LOG_FILE}"
import os
import torch
threads = int(os.environ.get("CSFL_CPU_THREADS", "4"))
torch.set_num_threads(threads)
torch.set_num_interop_threads(max(1, min(2, threads // 2)))
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torch num threads: {torch.get_num_threads()}")
print(f"torch interop threads: {torch.get_num_interop_threads()}")
PY

log "Downloading/checking Fashion-MNIST and CIFAR-10"
"${PYTHON_BIN}" "scripts/download_data.py" --datasets fashion-mnist cifar10 || \
  log "Dataset pre-download failed; torchvision will retry during the run."

SEED_ARGS=()
for seed in ${MAML_V2_SEEDS}; do
  SEED_ARGS+=(--seed "${seed}")
done

METER_ARGS=(--no-hardware-meter)
if [[ "${ENABLE_CODECARBON}" == "1" ]]; then
  METER_ARGS=()
  log "CodeCarbon hardware tracker enabled. Modelled per-round carbon is logged either way."
else
  log "CodeCarbon hardware tracker disabled for CPU speed. Modelled energy/carbon remains logged per round."
fi

EXTRA_ARGS=()
if [[ -n "${MAML_V2_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${MAML_V2_EXTRA_ARGS}"
fi

log "Dry-run: validating MAML-Select v2 CPU matrix"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile maml_v2_cpu \
  --device cpu \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  "${SEED_ARGS[@]}" \
  --dry-run | tee -a "${LOG_FILE}"

log "Starting MAML-Select v2 CPU run for Fashion-MNIST and CIFAR-10"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile maml_v2_cpu \
  --device cpu \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --country-iso-code "${COUNTRY_ISO}" \
  --grid-intensity "${GRID_INTENSITY}" \
  --resume \
  "${SEED_ARGS[@]}" \
  "${METER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

log "Running analysis tables"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.analyze_results \
  --results-dir "${RUNS_DIR}" \
  --output-dir "${ANALYSIS_DIR}" 2>&1 | tee -a "${LOG_FILE}"

log "Complete. Runtime logs: ${RUNS_DIR}"
log "Analysis tables: ${ANALYSIS_DIR}"
