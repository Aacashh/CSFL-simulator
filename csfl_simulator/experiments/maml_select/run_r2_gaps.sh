#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
#  MAML-Select, the two gaps left in the R2 evidence
# =============================================================================
#
# The alpha = 0.1 study came back with 18 runs. Twelve failed, and those twelve
# were the no-adaptation control. This script runs them again, and runs the
# benchmark sweep that Table II still depends on.
#
#   A  no-adaptation control     12 runs
#      inner_steps 0 against inner_steps 1, CIFAR-10 at 100 rounds and
#      CIFAR-100 at 150 rounds, three seeds. The 0-step arm never touches the
#      support set, so its outer step is a plain Adam step on the query loss.
#      That is online regression with the same network, which is the control a
#      referee needs in order to believe the adaptation step earns its place.
#
#   B  benchmark sweep          48 runs
#      All eight methods on Fashion-MNIST and CIFAR-10, three seeds. Two things
#      depend on it. Table II currently reports 90.11 +- 0.47 for MAML-Select on
#      Fashion-MNIST while every run of that configuration on the analysis
#      machine gives 90.23 +- 0.55, so the table and the ablation disagree about
#      one experiment. And the round-latency column can only be filled for a
#      dataset where every method has been run.
#
# Stage A is the one to protect if time is short. Run it alone with STAGES=A.
#
# -----------------------------------------------------------------------------
# Usage
#
#   bash csfl_simulator/experiments/maml_select/run_r2_gaps.sh
#
#   STAGES=A    bash .../run_r2_gaps.sh    # the control only, 12 runs
#   STAGES=B    bash .../run_r2_gaps.sh    # the sweep only, 48 runs
#   DEVICE=cuda bash .../run_r2_gaps.sh    # force a device
#   DRY_RUN=1   bash .../run_r2_gaps.sh    # list the runs and stop
#
# Both stages resume. Re-entering a finished stage costs nothing, and an
# interrupted one picks up where it stopped.
#
# Every run writes a log. If something fails, send back the whole logs directory
# rather than the summary, because last time the failures arrived with no
# record of why.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
STAGES="${STAGES:-AB}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/MAML-R2-Gaps}"
LOG_DIR="${OUT_ROOT}/logs"
ARTIFACTS="${OUT_ROOT}/artifacts"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Put this checkout ahead of anything installed. Git Bash hands a Windows
# python.exe a POSIX path it cannot read, so convert when cygpath exists and use
# the platform's own separator.
_NATIVE_ROOT="${REPO_ROOT}"
_PATH_SEP=":"
if command -v cygpath >/dev/null 2>&1; then
  _NATIVE_ROOT="$(cygpath -w "${REPO_ROOT}")"
  _PATH_SEP=";"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${_NATIVE_ROOT}${_PATH_SEP}${PYTHONPATH}"
else
  export PYTHONPATH="${_NATIVE_ROOT}"
fi

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" "${ARTIFACTS}/analysis"

echo "============================================================"
echo "  MAML-Select, R2 evidence gaps"
echo "============================================================"
echo "Repo:      ${REPO_ROOT}"
echo "Python:    ${PYTHON_BIN}"
echo "Device:    ${DEVICE}"
echo "Stages:    ${STAGES}"
echo "Output:    ${OUT_ROOT}"
echo "Logs:      ${LOG_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Preflight. Three checks, each of which has broken a campaign before.
# -----------------------------------------------------------------------------
echo "[preflight 1/3] the package imports"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

for name in ("csfl_simulator.core.client",
             "csfl_simulator.experiments.maml_select.selector",
             "csfl_simulator.experiments.maml_select.simulator",
             "csfl_simulator.experiments.maml_select.run_experiments"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print("  cannot import %s: %s" % (name, exc))
        sys.exit(1)
print("  ok")
PY

echo "[preflight 2/3] the selector accepts zero inner steps"
"${PYTHON_BIN}" - <<'PY'
import inspect
import sys

from csfl_simulator.experiments.maml_select import selector

src = inspect.getsource(selector)
if "max(1, int(inner_steps))" in src:
    print("  the selector still clamps inner_steps to at least 1,")
    print("  so the 0-step arm would silently run as a 1-step arm")
    sys.exit(1)
print("  ok")
PY

echo "[preflight 3/3] device"
RESOLVED_DEVICE="$(
  DEVICE_REQUEST="${DEVICE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

req = os.environ.get("DEVICE_REQUEST", "auto")
resolved = req
if req == "auto":
    if torch.cuda.is_available():
        resolved = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
if resolved == "cuda" and not torch.cuda.is_available():
    print("cuda was asked for and is not available", file=sys.stderr)
    sys.exit(2)
print(resolved)
PY
)"
echo "  running on ${RESOLVED_DEVICE}"
echo ""

DRY_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_ARGS=(--dry-run)
fi

run_stage () {
  local tag="$1" profile="$2" experiment="$3" expected="$4"
  local log="${LOG_DIR}/${tag}_${STAMP}.log"

  echo "------------------------------------------------------------"
  echo "  stage ${tag}: ${experiment}, ${expected} runs"
  echo "  log: ${log}"
  echo "------------------------------------------------------------"

  if "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
      --profile "${profile}" \
      --only "${experiment}" \
      --device "${RESOLVED_DEVICE}" \
      --output-dir "${OUT_ROOT}/${tag}" \
      --analysis-dir "${ARTIFACTS}/analysis" \
      --resume \
      "${DRY_ARGS[@]}" 2>&1 | tee "${log}"; then
    echo "  stage ${tag} finished"
  else
    echo ""
    echo "  STAGE ${tag} FAILED. The log is at ${log}."
    echo "  Send that file back. Do not re-run before reading it, because a"
    echo "  second attempt overwrites nothing but wastes the same hours."
    return 1
  fi
  echo ""
}

STATUS=0
if [[ "${STAGES}" == *A* ]]; then
  run_stage A report_card no_adaptation_control 12 || STATUS=1
fi
if [[ "${STAGES}" == *B* && "${STATUS}" -eq 0 ]]; then
  run_stage B report_card_main main_benchmarks 48 || STATUS=1
fi

# -----------------------------------------------------------------------------
# What came back
# -----------------------------------------------------------------------------
echo "============================================================"
echo "  Summary"
echo "============================================================"
COMPLETED="$(find "${OUT_ROOT}" -name result.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  result.json files under ${OUT_ROOT}: ${COMPLETED}"
echo "  logs: ${LOG_DIR}"
echo ""
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "  Dry run. Nothing was executed and nothing was written."
elif [[ "${STATUS}" -eq 0 ]]; then
  echo "  Send back the whole of:"
  echo "      ${OUT_ROOT}"
  echo "  It holds the run outputs, the analysis CSVs and every log."
else
  echo "  At least one stage did not finish. Send back ${LOG_DIR}."
fi

exit "${STATUS}"
