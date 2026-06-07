#!/usr/bin/env bash
set -euo pipefail

# Selector-architecture (meta-policy capacity) ablation.
#   hidden_dim in {16, 32, 64, 128}  x  seeds {42, 123, 2026}  = 12 Fashion-MNIST runs.
# Answers the reviewer note that the 6-64-64-1 MLP width is never ablated.
# Fashion-MNIST/LightCNN (fashion_main, 200 rounds) for direct comparability with the
# existing lambda_sensitivity / feature_ablation Fashion tables. Resumable and isolated
# from the main + review-hardening campaigns (separate runs/ and artifacts/ trees).

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
# CUDA fast path: cuDNN benchmark autotune + TF32 (default on; set PERFORMANCE_MODE=0 for
# strict bit-for-bit determinism). No-op on CPU/MPS; AMP is already on by default for CUDA.
PERFORMANCE_MODE="${PERFORMANCE_MODE:-1}"
PERF_FLAG=""
[[ "${PERFORMANCE_MODE}" == "1" ]] && PERF_FLAG="--performance-mode"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/maml_select_arch_ablation}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/maml_select/arch_ablation}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${RUNS_DIR}/logs" "${ARTIFACTS_DIR}/analysis"

echo "============================================================"
echo "  MAML-Select Architecture (Selector-Capacity) Ablation"
echo "============================================================"
echo "Repo:      ${REPO_ROOT}"
echo "Python:    ${PYTHON_BIN}"
echo "Device:    ${DEVICE}"
echo "Runs:      ${RUNS_DIR}"
echo "Artifacts: ${ARTIFACTS_DIR}"
echo ""
echo "Workload: Fashion-MNIST, 12 resumable 200-round runs"
echo "  - hidden_dim in {16, 32, 64, 128} x seeds 42/123/2026"
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
  echo "[error] Refusing to start 12 runs on CPU. Use DEVICE=mps, or ALLOW_CPU=1 to override."
  exit 4
fi

echo "[1/4] Ensuring Fashion-MNIST is present"
"${PYTHON_BIN}" scripts/download_data.py --datasets fashion-mnist || \
  echo "[warn] Dataset pre-download failed; torchvision will retry during the first run."

echo "[2/4] Dry-run matrix check"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile arch_ablation \
  --device "${DEVICE}" \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --dry-run

echo "[3/4] Running architecture ablation (performance_mode=${PERFORMANCE_MODE})"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile arch_ablation \
  --device "${DEVICE}" \
  ${PERF_FLAG} \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis" \
  --no-hardware-meter \
  --resume

echo "[4/4] Summarizing outputs"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.summarize_arch_ablation \
  --runs-root "${RUNS_DIR}" \
  --output-dir "${ARTIFACTS_DIR}" || \
  echo "[warn] Summary step failed; result.json files are intact under ${RUNS_DIR}."

echo ""
echo "Done."
echo "Results:   ${RUNS_DIR}"
echo "Analysis:  ${ARTIFACTS_DIR}/analysis"
echo "Tables:    ${ARTIFACTS_DIR}/tables"
