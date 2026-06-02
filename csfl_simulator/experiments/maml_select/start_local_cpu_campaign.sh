#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_DIR="${REPO_ROOT}/runs/maml_select"
ANALYSIS_DIR="${REPO_ROOT}/artifacts/maml_select/analysis"
LOG_DIR="${RUNS_DIR}/logs"
LOG_FILE="${LOG_DIR}/full_cpu_200.log"
PID_FILE="${LOG_DIR}/full_cpu_200.pid"
LABEL="org.mamlselect.full-cpu-200"
CAFFEINATE_LABEL="org.mamlselect.full-cpu-200-caffeinate"

mkdir -p "${LOG_DIR}"
if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf 'Campaign is already running with PID %s\n' "${EXISTING_PID}"
    exit 0
  fi
fi
if launchctl list "${LABEL}" >/dev/null 2>&1; then
  printf 'Campaign is already supervised as %s\n' "${LABEL}"
  exit 0
fi

COMMAND=(
  env MPLCONFIGDIR=/tmp/maml-select-matplotlib PYTHONUNBUFFERED=1
  "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
  --profile full \
  --device cpu \
  --output-dir "${RUNS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --country-iso-code IND \
  --grid-intensity 475 \
  --resume \
  "$@"
)
printf -v COMMAND_QUOTED '%q ' "${COMMAND[@]}"
printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v LOG_FILE_QUOTED '%q' "${LOG_FILE}"

launchctl submit -l "${LABEL}" -- /bin/zsh -lc \
  "cd ${REPO_ROOT_QUOTED} && exec ${COMMAND_QUOTED} >>${LOG_FILE_QUOTED} 2>&1"
sleep 1
JOB="$(launchctl list "${LABEL}")"
PID="$(printf '%s\n' "${JOB}" | awk -F'= ' '/"PID"/ {gsub(/;/, "", $2); print $2}')"
printf '%s\n' "${PID}" >"${PID_FILE}"
if [[ -n "${PID}" ]] && ! launchctl list "${CAFFEINATE_LABEL}" >/dev/null 2>&1; then
  launchctl submit -l "${CAFFEINATE_LABEL}" -- /usr/bin/caffeinate -dimsu -w "${PID}"
fi
printf 'Started full CPU campaign with PID %s under %s\n' "${PID:-pending}" "${LABEL}"
printf 'Log: %s\n' "${LOG_FILE}"
