#!/usr/bin/env bash
set -euo pipefail

# Lambda no-penalty anchor (lambda=0): the pure-accuracy baseline for the lambda sweep.
#   CIFAR-10 (cifar10_review_100, 100 rounds) + Fashion-MNIST (fashion_main, 200 rounds),
#   seeds {42, 123, 2026} = 6 runs. Dedicated runs/ tree, resumable, isolated from the
#   completed 21-run lambda/inner-step campaign. Built to run on an NVIDIA GPU (CUDA).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Pick an interpreter that actually has torch (bare python3 on macOS often lacks it).
_pick_python() {
  local candidates=()
  [[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("${CONDA_PREFIX}/bin/python")
  candidates+=("python" "python3")
  local c
  for c in "${candidates[@]}"; do
    if command -v "${c}" >/dev/null 2>&1 && "${c}" -c "import torch" >/dev/null 2>&1; then
      command -v "${c}"; return 0
    fi
  done
  echo "python3"
}
PYTHON_BIN="${PYTHON_BIN:-$(_pick_python)}"
DEVICE="${DEVICE:-auto}"
ALLOW_CPU="${ALLOW_CPU:-0}"
# CUDA fast path: cuDNN benchmark autotune + TF32 (default on). PERFORMANCE_MODE=0 for strict determinism.
PERFORMANCE_MODE="${PERFORMANCE_MODE:-1}"
PERF_FLAG=""
[[ "${PERFORMANCE_MODE}" == "1" ]] && PERF_FLAG="--performance-mode"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/maml_select_lambda_anchor}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/maml_select/lambda_anchor}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${RUNS_DIR}/logs" "${ARTIFACTS_DIR}/analysis"

echo "============================================================"
echo "  MAML-Select Lambda Anchor (lambda=0 pure-accuracy baseline)"
echo "============================================================"
echo "Repo:      ${REPO_ROOT}"
echo "Python:    ${PYTHON_BIN}"
echo "Device:    ${DEVICE}"
echo "Runs:      ${RUNS_DIR}"
echo ""
echo "Workload: 6 resumable runs"
echo "  - CIFAR-10  (100 rounds) lambda=0 x seeds 42/123/2026"
echo "  - Fashion-MNIST (200 rounds) lambda=0 x seeds 42/123/2026"
echo ""

echo "[0/4] Device check"
RESOLVED_DEVICE="$(
  DEVICE_REQUEST="${DEVICE}" "${PYTHON_BIN}" -c "import os, sys, torch; req=os.environ.get('DEVICE_REQUEST','auto'); resolved=req
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
  echo "[error] Refusing to start on CPU. Use DEVICE=cuda on the GPU laptop, or ALLOW_CPU=1 to override."
  exit 4
fi

echo "[1/4] Ensuring datasets are present"
"${PYTHON_BIN}" scripts/download_data.py --datasets cifar10 fashion-mnist || \
  echo "[warn] Dataset pre-download failed; torchvision will retry during the first run."

echo "[2/4] Dry-run matrix check"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile lambda_anchor \
  --device "${DEVICE}" \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --dry-run

echo "[3/4] Running lambda anchor (performance_mode=${PERFORMANCE_MODE})"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile lambda_anchor \
  --device "${DEVICE}" \
  ${PERF_FLAG} \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --no-hardware-meter \
  --resume

echo "[4/4] Done. Pull ${RUNS_DIR} back to the Mac and merge into the lambda table with:"
echo "      python -m csfl_simulator.experiments.maml_select.summarize_review_hardening \\"
echo "        --extra-runs-root ${RUNS_DIR}"
echo ""
echo "Results: ${RUNS_DIR}"
