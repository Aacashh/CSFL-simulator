#!/usr/bin/env bash
set -euo pipefail

# Targeted review-hardening experiments:
#   1) CIFAR-10 lambda sensitivity: lambda in {0.1, 0.5, 1.0, 5.0}
#   2) CIFAR-10 inner-step ablation: inner_steps in {1, 2, 5}
#
# Total default workload: 7 variants x 3 seeds = 21 CIFAR-10 MAML-Select runs.
# Each run uses 100 diagnostic rounds, leaving the main 200-round benchmarks intact.
# The run is resumable and isolated from the main campaign.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Pick an interpreter that actually has torch. On macOS the bare `python3` is
# often the python.org framework build with no torch, while the GPU-capable
# PyTorch lives in a conda env. Honour an explicit PYTHON_BIN, otherwise probe
# common locations and use the first one that can `import torch`.
_pick_python() {
  local candidates=()
  [[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("${CONDA_PREFIX}/bin/python")
  candidates+=("python" "python3")
  local c
  for c in "${candidates[@]}"; do
    if command -v "${c}" >/dev/null 2>&1 && "${c}" -c "import torch" >/dev/null 2>&1; then
      command -v "${c}"
      return 0
    fi
  done
  # Last resort: return python3 so the device check below reports the problem.
  echo "python3"
}
PYTHON_BIN="${PYTHON_BIN:-$(_pick_python)}"
DEVICE="${DEVICE:-auto}"
ALLOW_CPU="${ALLOW_CPU:-0}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/maml_select_review_hardening}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/maml_select/review_hardening}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${RUNS_DIR}/logs" "${ARTIFACTS_DIR}/analysis"

echo "============================================================"
echo "  MAML-Select Review-Hardening Experiments"
echo "============================================================"
echo "Repo:      ${REPO_ROOT}"
echo "Python:    ${PYTHON_BIN}"
echo "Device:    ${DEVICE}"
echo "Runs:      ${RUNS_DIR}"
echo "Artifacts: ${ARTIFACTS_DIR}"
echo ""
echo "Workload: CIFAR-10 only, 21 resumable 100-round diagnostic runs"
echo "  - lambda sensitivity: 4 variants x seeds 42/123/2026"
echo "  - inner-step ablation: 3 variants x seeds 42/123/2026"
echo ""

echo "[0/4] Device check"
RESOLVED_DEVICE="$(
  DEVICE_REQUEST="${DEVICE}" "${PYTHON_BIN}" -c "import os, sys, torch; req=os.environ.get('DEVICE_REQUEST','auto'); resolved=req; 
if req == 'auto':
    resolved = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
if resolved == 'cuda' and not torch.cuda.is_available():
    print('cuda unavailable', file=sys.stderr); sys.exit(2)
if resolved == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    print('mps unavailable', file=sys.stderr); sys.exit(3)
print(resolved)"
)"
echo "Resolved device: ${RESOLVED_DEVICE}"
if [[ "${RESOLVED_DEVICE}" == "cpu" && "${ALLOW_CPU}" != "1" ]]; then
  echo "[error] Refusing to start 21 CIFAR-10 runs on CPU."
  echo "        Use DEVICE=cuda or DEVICE=mps on a GPU-enabled environment."
  echo "        To override intentionally, rerun with ALLOW_CPU=1."
  exit 4
fi

echo "[1/4] Downloading CIFAR-10 if needed"
"${PYTHON_BIN}" scripts/download_data.py --datasets cifar10 || \
  echo "[warn] Dataset pre-download failed; torchvision will retry during the first run."

echo "[2/4] Dry-run matrix check"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile review_hardening \
  --device "${DEVICE}" \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --dry-run

echo "[3/4] Running targeted experiments"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile review_hardening \
  --device "${DEVICE}" \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --no-hardware-meter \
  --resume

echo "[4/4] Summarizing outputs"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.summarize_review_hardening \
  --runs-root "${RUNS_DIR}" \
  --output-dir "${ARTIFACTS_DIR}"

echo ""
echo "Done."
echo "Results:   ${RUNS_DIR}"
echo "Analysis:  ${ARTIFACTS_DIR}/analysis"
echo "Tables:    ${ARTIFACTS_DIR}/tables"
echo "Plots:     ${ARTIFACTS_DIR}/plots"
