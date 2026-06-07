#!/usr/bin/env bash
# Live terminal dashboard for the MAML-Select review-hardening GPU campaign.
# Usage:  bash watch_progress.sh
# Ctrl-C to stop watching (the campaign keeps running in the background).
set -u

RUNS="${RUNS:-/Users/advaitpathak/CSFL-simulator/CSFL-simulator/runs/maml_select_review_hardening}"
CAMP_LOG="${CAMP_LOG:-${RUNS}/logs/campaign_full.log}"
TOTAL="${TOTAL:-21}"
ROUNDS="${ROUNDS:-100}"

while true; do
  clear
  done_n="$(find "${RUNS}" -name result.json 2>/dev/null | wc -l | tr -d ' ')"
  device="$(grep -aE 'Resolved device' "${CAMP_LOG}" 2>/dev/null | tail -1 | sed 's/.*: //')"
  # Active run = the most recently appended round_metrics.jsonl.
  active="$(find "${RUNS}" -name round_metrics.jsonl 2>/dev/null -print0 | xargs -0 ls -t 2>/dev/null | head -1)"

  echo "===================================================================="
  echo "  MAML-Select Review-Hardening  —  live GPU progress"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')    device: ${device:-resolving...}"
  echo "===================================================================="
  echo "  Completed runs : ${done_n}/${TOTAL}"
  if [ -n "${active}" ]; then
    rdir="$(dirname "${active}")"
    label="$(basename "${rdir}")"
    rounds_done="$(wc -l < "${active}" 2>/dev/null | tr -d ' ')"
    last="$(tail -1 "${active}" 2>/dev/null)"
    acc="$(printf '%s' "${last}" | sed -n 's/.*"accuracy": *\([0-9.]*\).*/\1/p')"
    echo "  Active run     : ${label}"
    echo "  Round          : ${rounds_done}/${ROUNDS}    latest acc=${acc:-n/a}"
  else
    echo "  Active run     : (starting up / loading CIFAR-10 ...)"
  fi
  echo "--------------------------------------------------------------------"
  echo "  Recent campaign log:"
  tail -4 "${CAMP_LOG}" 2>/dev/null | sed 's/^/    /'
  echo "--------------------------------------------------------------------"
  echo "  Ctrl-C to stop watching (campaign keeps running in background)."
  if [ "${done_n}" -ge "${TOTAL}" ]; then echo "  ALL ${TOTAL} RUNS COMPLETE."; break; fi
  sleep 5
done
