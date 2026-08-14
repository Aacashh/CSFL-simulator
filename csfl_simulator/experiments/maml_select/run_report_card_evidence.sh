#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  MAML-Select, R2 report-card evidence
# =============================================================================
#
# Four stages, ordered so the blocking items finish first. Every stage is
# resumable, so an interrupted run picks up where it stopped and a finished
# stage costs nothing to re-enter.
#
#   A  no-adaptation control        12 runs   inner_steps 0 and 1,
#                                             CIFAR-10 100r and CIFAR-100 150r,
#                                             3 seeds                  BLOCKING
#   B  alpha = 0.1                  18 runs   FedAvg, FedGCS, MAML-Select on
#                                             Fashion-MNIST and CIFAR-10, 3 seeds
#   C  benchmarks and ablation      69 runs   Fashion-MNIST and CIFAR-10 with all
#                                             8 methods, plus the 7 feature
#                                             ablations                BLOCKING
#   D  CIFAR-100 benchmarks         24 runs   optional, only needed if you want
#                                             the shard counts logged there too
#
# Stage A answers "does the inner step do anything".
# Stage B answers "does it hold under severe non-IID".
# Stage C is what produces the round-latency and time-to-target columns for the
#   benchmark table, and it makes the benchmark row and the full-state ablation
#   row come from one campaign so they reconcile exactly.
# Stage D is optional. The CIFAR-100 numbers already in the paper reproduce from
#   the existing runs, and the mean shard size there is recoverable from the
#   TFLOPs column without re-running anything.
#
# -----------------------------------------------------------------------------
# Usage
#
#   bash csfl_simulator/experiments/maml_select/run_report_card_evidence.sh
#
#   STAGES=AB  bash .../run_report_card_evidence.sh     # only stages A and B
#   DEVICE=cuda bash .../run_report_card_evidence.sh    # force a device
#   ANALYZE_ONLY=1 bash .../run_report_card_evidence.sh # skip runs, print tables
#
# Environment
#   PYTHON_BIN   interpreter that has torch. Probed if unset.
#   DEVICE       auto | cuda | mps | cpu.  Default auto.
#   ALLOW_CPU    set to 1 to permit a CPU run. Refused by default, because
#                stage C is 69 ResNet18 and LightCNN runs.
#   STAGES       subset of ABCD. Default ABC.
#   RUNS_DIR     where run directories are written.
#   SEEDS        override the seed list, e.g. SEEDS="42 123 2026 7 99".
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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
  echo "python3"
}

PYTHON_BIN="${PYTHON_BIN:-$(_pick_python)}"
DEVICE="${DEVICE:-auto}"
ALLOW_CPU="${ALLOW_CPU:-0}"
STAGES="${STAGES:-ABC}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/report_card}"
MAIN_RUNS_DIR="${MAIN_RUNS_DIR:-${REPO_ROOT}/runs/report_card_main}"
C100_RUNS_DIR="${C100_RUNS_DIR:-${REPO_ROOT}/runs/report_card_cifar100}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/maml_select/report_card}"

SEED_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then
  for s in ${SEEDS}; do SEED_ARGS+=(--seed "${s}"); done
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${RUNS_DIR}/logs" "${MAIN_RUNS_DIR}/logs" "${C100_RUNS_DIR}/logs" \
         "${ARTIFACTS_DIR}/analysis"

echo "============================================================"
echo "  MAML-Select, R2 report-card evidence"
echo "============================================================"
echo "Repo:       ${REPO_ROOT}"
echo "Python:     ${PYTHON_BIN}"
echo "Device:     ${DEVICE}"
echo "Stages:     ${STAGES}"
echo "Runs:       ${RUNS_DIR}"
echo "Main runs:  ${MAIN_RUNS_DIR}"
echo "Artifacts:  ${ARTIFACTS_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
if [[ "${ANALYZE_ONLY}" != "1" ]]; then
  echo "[preflight 1/3] Device check"
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
  echo "                 resolved device: ${RESOLVED_DEVICE}"
  if [[ "${RESOLVED_DEVICE}" == "cpu" && "${ALLOW_CPU}" != "1" ]]; then
    echo "[error] Refusing to start this campaign on CPU."
    echo "        Stage C alone is 69 runs, most of them ResNet18 for 200 rounds."
    echo "        Use DEVICE=cuda, or rerun with ALLOW_CPU=1 if you mean it."
    exit 4
  fi

  echo "[preflight 2/3] The inner_steps=0 path actually disables adaptation"
  "${PYTHON_BIN}" - <<'PYCHECK'
import numpy as np, torch
from csfl_simulator.experiments.maml_select import selector as S

model = S._seeded_policy(2026, "cpu", 64)
before = {k: v.detach().clone() for k, v in model.named_parameters()}
x = torch.tensor(np.random.default_rng(0).normal(size=(8, 6)).astype("float32"))
y = torch.tensor(np.random.default_rng(1).normal(size=(8,)).astype("float32"))

zero = S._adapt(model, x, y, 0.01, 0)
one = S._adapt(model, x, y, 0.01, 1)

same = all(torch.equal(zero[k], before[k]) for k in before)
moved = any(not torch.equal(one[k], before[k]) for k in before)
assert same, "inner_steps=0 changed the parameters, the control is not a control"
assert moved, "inner_steps=1 did not change the parameters, the adaptation is dead"
print("                 ok: 0 steps leaves phi untouched, 1 step moves it")
PYCHECK

  echo "[preflight 3/3] Datasets"
  "${PYTHON_BIN}" scripts/download_data.py --datasets fashion-mnist cifar10 cifar100 || \
    echo "[warn] pre-download failed; torchvision will retry on first use."
fi

run_profile() {
  local profile="$1" outdir="$2" label="$3"
  echo ""
  echo "------------------------------------------------------------"
  echo "  ${label}"
  echo "------------------------------------------------------------"
  "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile "${profile}" \
    --device "${DEVICE}" \
    --output-dir "${outdir}" \
    --analysis-dir "${ARTIFACTS_DIR}/analysis" \
    "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}" \
    --dry-run
  "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile "${profile}" \
    --device "${DEVICE}" \
    --output-dir "${outdir}" \
    --analysis-dir "${ARTIFACTS_DIR}/analysis" \
    "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}" \
    --no-hardware-meter \
    --resume
}

if [[ "${ANALYZE_ONLY}" != "1" ]]; then
  # Stages A and B share the report_card profile. Running it once covers both,
  # and --only narrows it when a single stage is requested.
  if [[ "${STAGES}" == *A* && "${STAGES}" == *B* ]]; then
    run_profile report_card "${RUNS_DIR}" "Stages A and B: control and alpha=0.1, 30 runs"
  elif [[ "${STAGES}" == *A* ]]; then
    echo ""
    echo "  Stage A: no-adaptation control, 12 runs"
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
      --profile report_card --only no_adaptation_control \
      --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
      --analysis-dir "${ARTIFACTS_DIR}/analysis" \
      "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}" --no-hardware-meter --resume
  elif [[ "${STAGES}" == *B* ]]; then
    echo ""
    echo "  Stage B: alpha=0.1, 18 runs"
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
      --profile report_card --only heterogeneity_alpha_0p1 \
      --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
      --analysis-dir "${ARTIFACTS_DIR}/analysis" \
      "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}" --no-hardware-meter --resume
  fi

  if [[ "${STAGES}" == *C* ]]; then
    run_profile report_card_main "${MAIN_RUNS_DIR}" \
      "Stage C: benchmarks and feature ablation, 69 runs"
  fi

  if [[ "${STAGES}" == *D* ]]; then
    run_profile cifar100 "${C100_RUNS_DIR}" \
      "Stage D: CIFAR-100 benchmarks, 24 runs, optional"
  fi
fi

# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  Building the table rows"
echo "------------------------------------------------------------"
ROOTS=("${RUNS_DIR}" "${MAIN_RUNS_DIR}" "${C100_RUNS_DIR}"
       "${REPO_ROOT}/runs/maml_select_cifar100"
       "${REPO_ROOT}/runs/maml_select_review_hardening"
       "${REPO_ROOT}/runs/maml_select")

mkdir -p "${ARTIFACTS_DIR}"
"${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.report_card_tables \
  --runs-root "${ROOTS[@]}" \
  | tee "${ARTIFACTS_DIR}/report_card_tables.txt"

echo ""
echo "============================================================"
echo "Done."
echo "  runs:   ${RUNS_DIR}, ${MAIN_RUNS_DIR}"
echo "  tables: ${ARTIFACTS_DIR}/report_card_tables.txt"
echo ""
echo "Send back ${ARTIFACTS_DIR}/report_card_tables.txt, and the run"
echo "directories if you want the numbers re-derived from the logs."
echo "============================================================"
