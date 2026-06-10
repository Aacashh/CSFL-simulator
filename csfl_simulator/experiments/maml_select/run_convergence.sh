#!/usr/bin/env bash
#
# Selector-convergence verification (Statements 1 and 2).
# Runs MAML-Select on each dataset for the full protocol (one seed) and records,
# per round, the inner-step support descent, the query objective, the
# meta-gradient norm, and the meta-update magnitude. Each dataset writes its own
# log via the MAML_SELECT_CONV_LOG environment variable (read by the selector).
#
# Run on the CUDA machine (Git Bash / WSL):
#     bash run_convergence.sh
# Optional overrides:
#     DEVICE=cuda PYTHON=python bash run_convergence.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"   # .../CSFL-simulator
OUT="${REPO_ROOT}/csfl_simulator/runs/maml_select_convergence"
mkdir -p "${OUT}"

PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"

cd "${REPO_ROOT}"

run() {
  local exp="$1" tag="$2"
  local log="${OUT}/selector_convergence_${tag}.jsonl"
  rm -f "${log}"
  echo ""
  echo "=================================================================="
  echo " Convergence run: ${tag}   (device=${DEVICE})"
  echo " Selector log -> ${log}"
  echo "=================================================================="
  MAML_SELECT_CONV_LOG="${log}" \
    "${PY}" -m csfl_simulator.experiments.maml_select.run_experiments \
      --profile convergence --only "${exp}" --seed 42 \
      --device "${DEVICE}" --output-dir "${OUT}"
}

run conv_fashion  fashion
run conv_cifar10  cifar10
run conv_cifar100 cifar100

echo ""
echo "All convergence runs complete."
echo "Per-round selector logs:   ${OUT}/selector_convergence_*.jsonl"
echo "Per-round model metrics:   ${OUT}/conv_*/round_metrics.jsonl"
echo "Next (on this machine):    python -m csfl_simulator.experiments.maml_select.build_convergence_figure --runs-dir ${OUT}"
