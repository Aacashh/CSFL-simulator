#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/maml_select_letter"
LOG_DIR="${ARTIFACT_DIR}/logs"
LOG_FILE="${LOG_DIR}/full_cpu_200.log"
PID_FILE="${LOG_DIR}/full_cpu_200.pid"
LABEL="org.mamlselect.full-cpu-200"
CAFFEINATE_LABEL="org.mamlselect.full-cpu-200-caffeinate"

if JOB="$(launchctl list "${LABEL}" 2>/dev/null)"; then
  PID="$(printf '%s\n' "${JOB}" | awk -F'= ' '/"PID"/ {gsub(/;/, "", $2); print $2}')"
  printf 'Campaign status: supervised (PID %s, label %s)\n' "${PID:-pending}" "${LABEL}"
elif [[ -f "${PID_FILE}" ]] && PID="$(cat "${PID_FILE}")" && kill -0 "${PID}" 2>/dev/null; then
  printf 'Campaign status: running (PID %s)\n' "${PID}"
else
  printf 'Campaign status: not started\n'
fi
if launchctl list "${CAFFEINATE_LABEL}" >/dev/null 2>&1; then
  printf 'macOS sleep guard: active\n'
else
  printf 'macOS sleep guard: inactive\n'
fi

COMPLETED="$(find "${ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name result.json -print \
  | grep -Ec '/(main_benchmarks|lambda_sensitivity|feature_ablation|hardware_energy_to_target|heterogeneity|scaling|larger_benchmark)__' || true)"
printf 'Completed full-profile results: %s / 174\n' "${COMPLETED}"

LATEST_ROUND_LOG="$(find "${ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name round_metrics.jsonl -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_ROUND_LOG}" ]]; then
  ROUND_RECORDS="$(wc -l <"${LATEST_ROUND_LOG}" | tr -d ' ')"
  printf 'Active run round records: %s / 200 (%s)\n' "${ROUND_RECORDS}" "${LATEST_ROUND_LOG}"
fi

LATEST_PROGRESS="$(find "${ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name progress.json -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_PROGRESS}" ]]; then
  printf '\nLatest round checkpoint:\n'
  cat "${LATEST_PROGRESS}"
fi

if [[ -f "${LOG_FILE}" ]]; then
  printf '\nRecent campaign log:\n'
  tail -n 20 "${LOG_FILE}"
fi
