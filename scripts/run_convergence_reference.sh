#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD — convergence full-participation reference (X2, CLAIM_AUDIT.md G2)
#
# WHY THIS EXISTS
# ---------------
#   Three reviewer replies are still pending, and all three are the same
#   question from three different reviewers: whether the FedTSKD O(1/t)
#   convergence bound survives replacing full participation with SCOPE-FD's
#   deterministic partial participation (R1 comment 1, R2 comment 3, comment 1
#   of the attached review letter). Nothing in the 278-run revision campaign
#   answers this, because no run has ever gone above K/N=33% (K=10, N=30) --
#   there is no full-participation (K=N) trajectory to compare against, so the
#   existing per-round log-log fit (slope -0.67, R^2=0.42) has nothing to be
#   measured relative to.
#
#   This is the ONE experiment CLAIM_AUDIT.md's own gap analysis (G2) flags as
#   load-bearing for that specific claim -- not a general hardening sweep. It
#   is intentionally scoped to the minimum that lets the reply state a real,
#   measured relationship instead of asserting one:
#
#     K = 20   an intermediate point on the way to full participation
#     K = 30   = N, i.e. full participation. Every selector is identical here
#               by construction (the debt term cannot differentiate clients
#               when all of them are selected every round), so this run also
#               doubles as a sanity check that scope_fd and random converge to
#               the same trajectory once K=N, which is what Proposition 1
#               predicts and what a reviewer will look for first.
#
#   5 seeds each (not 3), to match the RP-2 headline-configuration protocol,
#   since this feeds a reviewer-facing claim rather than an exploratory sweep.
#   10 total run invocations (5 seeds x 2 values of K; each invocation
#   compares random and scope_fd together in one process via the existing
#   `compare` CLI, so this is not 20 separate training runs).
#
#   This is a thin wrapper around the SAME orchestrator every other family in
#   configs/scope_revision_sweeps.yaml already uses
#   (scripts/run_scope_revision_suite.py) -- no new plumbing, no new selector
#   code, nothing that hasn't already been exercised by the rest of the
#   campaign. The family itself is `convergence_full_participation` in that
#   YAML file.
#
# WHAT THIS SCRIPT DOES
#   1. Runs the fast unit tests (tests/test_scope_revision.py) so a broken
#      selector/metric never eats GPU time.
#   2. Runs the convergence_full_participation family through the resumable
#      orchestrator (already-complete runs are skipped automatically, so this
#      script is safe to re-run after a preemption or a crash on shared HPC
#      nodes).
#   3. Aggregates the 10 runs into mean/std/95% CI + a paired significance
#      test against the reference method, same as every other family.
#
#   This script intentionally has NO scheduler directives (#SBATCH / #PBS
#   etc.), because the account, partition, GPU type, and walltime are specific
#   to your allocation and guessing at them would be worse than leaving it out.
#   Wrap it in your own sbatch/qsub submission, or run it directly on an
#   interactive GPU node:
#
#     sbatch --gpus=1 --time=08:00:00 --wrap "bash scripts/run_convergence_reference.sh"
#
# USAGE
#   bash scripts/run_convergence_reference.sh                # run it
#   bash scripts/run_convergence_reference.sh --dry-run       # print commands only
#   bash scripts/run_convergence_reference.sh --skip-tests    # skip the pytest gate
#   bash scripts/run_convergence_reference.sh --parallel-seeds 2  # only if you
#       have confirmed >50% free VRAM after one seed's ParallelTrainer has
#       allocated its replicas -- see Scope_FD_Revision.md Sec 2.2
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="runs_scope_revised"
PARALLEL_SEEDS=1
GPU_MONITOR_INTERVAL=15
DRY_RUN=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-root)          OUTPUT_ROOT="$2"; shift 2 ;;
        --parallel-seeds)       PARALLEL_SEEDS="$2"; shift 2 ;;
        --gpu-monitor-interval) GPU_MONITOR_INTERVAL="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=true; shift ;;
        --skip-tests)           SKIP_TESTS=true; shift ;;
        -h|--help)
            sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

AGGREGATE_DIR="${OUTPUT_ROOT}/aggregated"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/convergence_reference_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================================="
echo " SCOPE-FD convergence full-participation reference (X2)"
echo "   family:            convergence_full_participation"
echo "   output root:       $OUTPUT_ROOT"
echo "   parallel seeds:    $PARALLEL_SEEDS"
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

echo "[2/3] Running the convergence_full_participation family (10 runs: 5 seeds x K in {20, 30})..."
ORCHESTRATOR_ARGS=(
    --family convergence_full_participation
    --output-root "$OUTPUT_ROOT"
    --parallel-seeds "$PARALLEL_SEEDS"
    --gpu-monitor-interval "$GPU_MONITOR_INTERVAL"
)
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
    --reference-method fd_native.scope_fd

echo
echo "=============================================================="
echo " Done."
echo "   Aggregated JSON:  ${AGGREGATE_DIR}/aggregated_results.json"
echo "   LaTeX table:      ${AGGREGATE_DIR}/summary_table.tex"
echo "   Run log:          $LOG_FILE"
echo
echo " Next (once this completes):"
echo "   1. Pull the per-round accuracy/loss trajectories for K=20, K=30 out of"
echo "      ${OUTPUT_ROOT}/convergence_full_participation/*/compare_results.json"
echo "      and re-run the log-log server-loss fit (same method CLAIM_AUDIT.md"
echo "      already used to get slope=-0.67, R^2=0.42) now WITH a true K=N"
echo "      reference trajectory to compare against."
echo "   2. That comparison is what lets the three PENDING replies in"
echo "      response_to_reviewers.tex state a measured relationship instead of"
echo "      an assertion -- whichever way it comes out. Ask me to draft the"
echo "      assumption-by-assumption discussion and the three reply texts"
echo "      once you have this aggregated JSON back."
echo "=============================================================="
