#!/usr/bin/env bash
set -uo pipefail

# Waits for the CIFAR-10 review-hardening campaign (21 runs) to finish, then launches
# the Fashion-MNIST architecture ablation on the GPU. Queuing (rather than running
# concurrently) keeps the single Apple GPU dedicated to one campaign at a time, so the
# in-flight CIFAR timing/overhead measurements stay clean and the campaign is not slowed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CIFAR_RUNS="${CIFAR_RUNS:-${REPO_ROOT}/runs/maml_select_review_hardening}"
ARCH_SCRIPT="${SCRIPT_DIR}/run_arch_ablation.sh"
PYTHON_BIN="${PYTHON_BIN:-/Users/advaitpathak/anaconda3/envs/csfl-env/bin/python}"
TARGET="${TARGET:-21}"

LOGDIR="${REPO_ROOT}/runs/maml_select_arch_ablation/logs"
mkdir -p "${LOGDIR}"
SUP_LOG="${LOGDIR}/supervisor.log"
ARCH_LOG="${LOGDIR}/arch_campaign.log"

echo "$(date '+%F %T')  supervisor started; waiting for ${TARGET}/${TARGET} CIFAR runs" >> "${SUP_LOG}"
while :; do
  n="$(find "${CIFAR_RUNS}" -name result.json 2>/dev/null | wc -l | tr -d ' ')"
  echo "$(date '+%F %T')  CIFAR complete=${n}/${TARGET}" >> "${SUP_LOG}"
  if [ "${n:-0}" -ge "${TARGET}" ]; then break; fi
  sleep 300
done

echo "$(date '+%F %T')  CIFAR campaign complete; launching architecture ablation" >> "${SUP_LOG}"
PYTHON_BIN="${PYTHON_BIN}" DEVICE=auto ALLOW_CPU=0 PYTHONUNBUFFERED=1 \
  bash "${ARCH_SCRIPT}" >> "${ARCH_LOG}" 2>&1
status=$?
echo "$(date '+%F %T')  architecture ablation finished (exit ${status})" >> "${SUP_LOG}"
