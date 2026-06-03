#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS_DIR="${REPO_ROOT}/runs/maml_select_cifar100"
LOG_DIR="${RUNS_DIR}/logs"
LOG_FILE="${LOG_DIR}/cifar100_mps_200.log"
PID_FILE="${LOG_DIR}/cifar100_mps_200.pid"
LABEL="org.mamlselect.cifar100-mps-200"
CAFFEINATE_LABEL="org.mamlselect.cifar100-mps-200-caffeinate"

if JOB="$(launchctl list "${LABEL}" 2>/dev/null)"; then
  PID="$(printf '%s\n' "${JOB}" | awk -F'= ' '/"PID"/ {gsub(/;/, "", $2); print $2}')"
  printf 'CIFAR-100 campaign: supervised (PID %s)\n' "${PID:-pending}"
elif [[ -f "${PID_FILE}" ]] && PID="$(cat "${PID_FILE}")" && kill -0 "${PID}" 2>/dev/null; then
  printf 'CIFAR-100 campaign: running (PID %s)\n' "${PID}"
else
  printf 'CIFAR-100 campaign: not started\n'
fi
if launchctl list "${CAFFEINATE_LABEL}" >/dev/null 2>&1; then
  printf 'macOS sleep guard: active\n'
else
  printf 'macOS sleep guard: inactive\n'
fi

COMPLETED="$(find "${RUNS_DIR}" -mindepth 2 -maxdepth 2 -name result.json -print 2>/dev/null | wc -l | tr -d ' ')"
printf 'Completed CIFAR-100 results: %s / 24\n' "${COMPLETED}"

LATEST_ROUND_METRICS="$(find "${RUNS_DIR}" -mindepth 2 -maxdepth 2 -name round_metrics.jsonl -size +0 -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_ROUND_METRICS}" ]]; then
  printf '\nLatest completed training round:\n'
  tail -n 1 "${LATEST_ROUND_METRICS}"
fi

LATEST_PROGRESS="$(find "${RUNS_DIR}" -mindepth 2 -maxdepth 2 -name progress.json -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_PROGRESS}" ]]; then
  printf '\nLatest evaluated checkpoint (refreshed every 10 rounds):\n'
  cat "${LATEST_PROGRESS}"
fi

if [[ -f "${LOG_FILE}" ]]; then
  printf '\nRecent campaign log:\n'
  tail -n 24 "${LOG_FILE}"
fi

printf '\nThis status command exits after printing; the supervised campaign continues in the background.\n'
