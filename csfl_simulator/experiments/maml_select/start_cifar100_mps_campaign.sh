#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_DIR="${REPO_ROOT}/runs/maml_select_cifar100"
LOG_DIR="${RUNS_DIR}/logs"
LOG_FILE="${LOG_DIR}/cifar100_mps_200.log"
PID_FILE="${LOG_DIR}/cifar100_mps_200.pid"
LABEL="org.mamlselect.cifar100-mps-200"
CAFFEINATE_LABEL="org.mamlselect.cifar100-mps-200-caffeinate"
CAMPAIGN_SCRIPT="${SCRIPT_DIR}/run_cifar100_mps_campaign.sh"

mkdir -p "${LOG_DIR}"
if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf 'CIFAR-100 campaign is already running with PID %s\n' "${EXISTING_PID}"
    exit 0
  fi
fi
if launchctl list "${LABEL}" >/dev/null 2>&1; then
  printf 'CIFAR-100 campaign is already supervised as %s\n' "${LABEL}"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable. Launch from a normal macOS Terminal, outside the Codex sandbox.")
print("MPS device verified.")
PY

if [[ -s "${LOG_FILE}" ]]; then
  mv "${LOG_FILE}" "${LOG_FILE%.log}_$(date +%Y%m%d-%H%M%S).log"
fi

printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v PYTHON_BIN_QUOTED '%q' "${PYTHON_BIN}"
printf -v CAMPAIGN_SCRIPT_QUOTED '%q' "${CAMPAIGN_SCRIPT}"
printf -v LOG_FILE_QUOTED '%q' "${LOG_FILE}"
printf -v EXTRA_ARGS_QUOTED '%q' "${MAML_SELECT_EXTRA_ARGS:-}"

launchctl submit -l "${LABEL}" -- /bin/zsh -lc \
  "cd ${REPO_ROOT_QUOTED} && exec env MPLCONFIGDIR=/tmp/maml-select-matplotlib PYTHONUNBUFFERED=1 PYTHON_BIN=${PYTHON_BIN_QUOTED} MAML_SELECT_EXTRA_ARGS=${EXTRA_ARGS_QUOTED} bash ${CAMPAIGN_SCRIPT_QUOTED} >>${LOG_FILE_QUOTED} 2>&1"
sleep 1
JOB="$(launchctl list "${LABEL}")"
PID="$(printf '%s\n' "${JOB}" | awk -F'= ' '/"PID"/ {gsub(/;/, "", $2); print $2}')"
printf '%s\n' "${PID}" >"${PID_FILE}"
if [[ -n "${PID}" ]] && ! launchctl list "${CAFFEINATE_LABEL}" >/dev/null 2>&1; then
  launchctl submit -l "${CAFFEINATE_LABEL}" -- /usr/bin/caffeinate -dimsu -w "${PID}"
fi
printf 'Started CIFAR-100 MPS campaign with PID %s under %s\n' "${PID:-pending}" "${LABEL}"
printf 'Log: %s\n' "${LOG_FILE}"
