#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD Major Revision — Phase A experiment runner.
#
# Thin wrapper around the already-implemented Phase A0 infrastructure
# (scripts/run_scope_revision_suite.py + configs/scope_revision_sweeps.yaml +
# csfl_simulator/experiments/scope_fd/{aggregate_results,plot_with_ci}.py).
# See Scope_FD_Revision.md for the full plan this suite executes.
#
# WHAT THIS SCRIPT DOES
#   1. Runs the fast unit tests (tests/test_scope_revision.py) so a broken
#      selector/metric never eats a multi-hour GPU sweep.
#   2. Runs every experiment family declared in configs/scope_revision_sweeps.yaml
#      through the resumable Python orchestrator, in the file's declared order
#      (cheap ablation/coefficient sweeps first, the audio_fsdd port last, as
#      Scope_FD_Revision.md's Phase A1 -> A2 -> A3 ordering intends).
#   3. Aggregates every completed run into mean/std/95% CI + a LaTeX table
#      fragment, so no result is ever reported from a single seed.
#
#   Plotting individual paper figures (scripts/.../plot_with_ci.py) is left as
#   a follow-up step once specific figures are selected — it needs a
#   metric/kind/group choice per figure, which is a manuscript-integration
#   decision, not something this "run everything" script should guess at.
#
# USAGE
#   bash scripts/run_scope_revision_suite.sh                  # run everything
#   bash scripts/run_scope_revision_suite.sh --family E1,E2   # only some families
#   bash scripts/run_scope_revision_suite.sh --dry-run        # print commands only
#   bash scripts/run_scope_revision_suite.sh --resume         # rerun; already-
#                                                              # complete runs are
#                                                              # skipped automatically
#   bash scripts/run_scope_revision_suite.sh --skip-tests     # skip the pytest gate
#
# Family names match the keys under `families:` in
# configs/scope_revision_sweeps.yaml (ablation_headline, ablation_k_sweep,
# ablation_channel_sweep, coefficient_grid, literature_baselines,
# dirichlet_severity, iid_sanity, scale_and_nondivisible, dropout,
# bounded_staleness, histogram_privacy, channel_energy,
# public_dataset_sensitivity, audio_fsdd).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPEC="configs/scope_revision_sweeps.yaml"
OUTPUT_ROOT="artifacts/scope_revision"
AGGREGATE_DIR="artifacts/scope_revision/aggregated"
REFERENCE_METHOD="fd_native.scope_fd"
PARALLEL_SEEDS=1
GPU_MONITOR_INTERVAL=15
DRY_RUN=false
SKIP_TESTS=false
FAMILY_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --family)               FAMILY_ARG="$2"; shift 2 ;;
        --spec)                 SPEC="$2"; shift 2 ;;
        --output-root)          OUTPUT_ROOT="$2"; shift 2 ;;
        --parallel-seeds)       PARALLEL_SEEDS="$2"; shift 2 ;;
        --gpu-monitor-interval) GPU_MONITOR_INTERVAL="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=true; shift ;;
        --resume)               shift ;;  # default behavior: the orchestrator
                                           # already skips completed runs
        --skip-tests)           SKIP_TESTS=true; shift ;;
        -h|--help)
            sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

AGGREGATE_DIR="${OUTPUT_ROOT}/aggregated"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================================="
echo " SCOPE-FD revision suite"
echo "   spec:            $SPEC"
echo "   output root:      $OUTPUT_ROOT"
echo "   parallel seeds:    $PARALLEL_SEEDS"
echo "   families:          ${FAMILY_ARG:-<all>}"
echo "   dry run:           $DRY_RUN"
echo "   log file:          $LOG_FILE"
echo "=============================================================="
echo

if [[ "$SKIP_TESTS" == false ]]; then
    echo "[1/3] Running unit tests (tests/test_scope_revision.py)..."
    python3 -m pytest tests/test_scope_revision.py -q
    echo
else
    echo "[1/3] Skipping unit tests (--skip-tests passed)."
    echo
fi

echo "[2/3] Running experiment sweep..."
ORCHESTRATOR_ARGS=(
    --spec "$SPEC"
    --output-root "$OUTPUT_ROOT"
    --parallel-seeds "$PARALLEL_SEEDS"
    --gpu-monitor-interval "$GPU_MONITOR_INTERVAL"
)
if [[ -n "$FAMILY_ARG" ]]; then
    IFS=',' read -ra FAMILIES <<< "$FAMILY_ARG"
    for fam in "${FAMILIES[@]}"; do
        ORCHESTRATOR_ARGS+=(--family "$fam")
    done
fi
if [[ "$DRY_RUN" == true ]]; then
    ORCHESTRATOR_ARGS+=(--dry-run)
fi

python3 scripts/run_scope_revision_suite.py "${ORCHESTRATOR_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
echo

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run only — skipping aggregation (no results were produced)."
    exit 0
fi

echo "[3/3] Aggregating results (mean / std / 95% CI + paired significance)..."
python3 csfl_simulator/experiments/scope_fd/aggregate_results.py \
    "$OUTPUT_ROOT" \
    --output-dir "$AGGREGATE_DIR" \
    --reference-method "$REFERENCE_METHOD"

echo
echo "=============================================================="
echo " Done."
echo "   Aggregated JSON:  ${AGGREGATE_DIR}/aggregated_results.json"
echo "   LaTeX table:      ${AGGREGATE_DIR}/summary_table.tex"
echo "   Run log:          $LOG_FILE"
echo
echo " Next: per-figure plots (curves/bars/heatmap with CI bands) via"
echo "   python3 csfl_simulator/experiments/scope_fd/plot_with_ci.py \\"
echo "     ${AGGREGATE_DIR}/aggregated_results.json --kind curves --metric accuracy \\"
echo "     --group <group_key> --output <path>"
echo "=============================================================="
