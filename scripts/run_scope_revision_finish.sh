#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD revision — FINISH-LINE runner (time-boxed, ~2 days GPU).
#
# Runs only the work still outstanding after the 227/429 runs already in
# runs_scope_revised/, using the pruned configs/scope_revision_finish.yaml.
# Every pruning decision and its supporting measurement is documented at the
# top of that YAML.
#
# Families run in PRIORITY ORDER, most reviewer-critical first, so that if this
# is killed part-way (as the last campaign was) the most important evidence is
# already on disk. Resume is automatic: completed runs are skipped by hash, so
# re-running this script after an interruption picks up where it stopped.
#
#   TIER 1  (~28h)  every reviewer question that currently has ZERO data
#     1. dirichlet_severity  alpha=0.01     R1 / R3   severe non-IID     ~3.7h
#     2. histogram_privacy   eps sweep      R2.2      privacy            ~5.1h
#     3. channel_energy      SNR x budget   R2.5      channel/energy     ~7.2h
#     4. dropout             p in .1/.2/.3  R3        client dropout     ~5.4h
#     5. bounded_staleness   w in 1,2       R3        async              ~3.6h
#     6. audio_fsdd                         R1        non-image domain   ~3.0h
#
#   TIER 2  (~36h)  requested, but expensive
#     7. dataset_generality  MNIST,EMNIST   R1.3/R2.4 multi-seed CIs     ~6.1h
#     8. cifar10_multiseed                  R2.4      CIFAR + error bars  ~10h
#     9. public_dataset_sensitivity (OFAT)  R3        public-set robust ~12.6h
#    10. scale_and_nondivisible N=500,K=25  R1        mMIMO scale        ~7.1h
#
#   TIER 3  (~2.6h) optional strengthener, not a direct reviewer ask
#    11. audio_fsdd_k_sweep                 sparse-K result off-image
#
# Estimates come from measured per-method cost on the completed runs
# (~14.7 min/method-run median, scaling roughly as 3.5 + 1.75*K minutes).
# Audio timings are extrapolated from image runs and may differ.
#
# USAGE
#   bash scripts/run_scope_revision_finish.sh                # everything
#   bash scripts/run_scope_revision_finish.sh --tier1        # stop after tier 1
#   bash scripts/run_scope_revision_finish.sh --no-optional  # skip tier 3
#   bash scripts/run_scope_revision_finish.sh --dry-run
#   bash scripts/run_scope_revision_finish.sh --aggregate-only
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPEC="configs/scope_revision_finish.yaml"
OUTPUT_ROOT="runs_scope_revised"
REFERENCE_METHOD="fd_native.scope_fd"
GPU_MONITOR_INTERVAL=30
DRY_RUN=false
SKIP_TESTS=false
AGGREGATE_ONLY=false
MAX_TIER=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --spec)            SPEC="$2"; shift 2 ;;
        --output-root)     OUTPUT_ROOT="$2"; shift 2 ;;
        --tier1)           MAX_TIER=1; shift ;;
        --tier2)           MAX_TIER=2; shift ;;
        --no-optional)     MAX_TIER=2; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --skip-tests)      SKIP_TESTS=true; shift ;;
        --aggregate-only)  AGGREGATE_ONLY=true; shift ;;
        -h|--help)         sed -n '2,48p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

AGGREGATE_DIR="${OUTPUT_ROOT}/aggregated"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/finish_$(date +%Y%m%d_%H%M%S).log"

TIER1=(dirichlet_severity histogram_privacy channel_energy dropout bounded_staleness audio_fsdd)
TIER2=(dataset_generality cifar10_multiseed public_dataset_sensitivity scale_and_nondivisible)
TIER3=(audio_fsdd_k_sweep)

FAMILIES=("${TIER1[@]}")
[[ $MAX_TIER -ge 2 ]] && FAMILIES+=("${TIER2[@]}")
[[ $MAX_TIER -ge 3 ]] && FAMILIES+=("${TIER3[@]}")

echo "=============================================================="
echo " SCOPE-FD revision — finish-line run"
echo "   spec:          $SPEC"
echo "   output root:   $OUTPUT_ROOT"
echo "   max tier:      $MAX_TIER"
echo "   families:      ${#FAMILIES[@]}"
echo "   dry run:       $DRY_RUN"
echo "   log:           $LOG_FILE"
echo "=============================================================="
echo

aggregate() {
    echo
    echo "Aggregating ALL results in ${OUTPUT_ROOT} (old + new) ..."
    python3 csfl_simulator/experiments/scope_fd/aggregate_results.py \
        "$OUTPUT_ROOT" \
        --output-dir "$AGGREGATE_DIR" \
        --reference-method "$REFERENCE_METHOD"
    echo
    echo "  Aggregated JSON: ${AGGREGATE_DIR}/aggregated_results.json"
    echo "  LaTeX table:     ${AGGREGATE_DIR}/summary_table.tex"
}

if [[ "$AGGREGATE_ONLY" == true ]]; then
    aggregate
    exit 0
fi

if [[ "$SKIP_TESTS" == false ]]; then
    echo "[gate] Running unit tests ..."
    python3 -m pytest tests/test_scope_revision.py -q
    echo

    # audio_fsdd has never completed a run, and it is the only non-image
    # evidence in the campaign (R1). Validate the dataset up front -- a broken
    # download should surface now, not ~28h in when its slot comes up.
    # Non-fatal: every other family is independent of FSDD.
    echo "[gate] Verifying FSDD audio dataset (downloads on first use) ..."
    if python3 - <<'PY'
from csfl_simulator.core.datasets import get_dataset
tr = get_dataset("FSDD", train=True, download=True)
te = get_dataset("FSDD", train=False, download=True)
x, y = tr[0]
assert tuple(x.shape) == (1, 64, 64), f"unexpected shape {tuple(x.shape)}"
print(f"  FSDD OK — train={len(tr)} test={len(te)} shape={tuple(x.shape)}")
PY
    then :; else
        echo
        echo "  !! FSDD VERIFICATION FAILED — audio_fsdd / audio_fsdd_k_sweep will fail."
        echo "  !! Every other family is unaffected and will still run."
        echo
    fi
    echo
fi

SUITE_START=$(date +%s)
COMPLETED_FAMILIES=()

for i in "${!FAMILIES[@]}"; do
    fam="${FAMILIES[$i]}"
    n=$((i + 1))
    echo "=============================================================="
    echo " [family $n/${#FAMILIES[@]}] $fam"
    echo " started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================="
    FAM_START=$(date +%s)

    ARGS=(--spec "$SPEC" --output-root "$OUTPUT_ROOT" --family "$fam"
          --gpu-monitor-interval "$GPU_MONITOR_INTERVAL")
    [[ "$DRY_RUN" == true ]] && ARGS+=(--dry-run)

    # Keep going if one family fails; a single broken family must not cost us
    # the remaining ones. Failures are reported in the summary below.
    if python3 scripts/run_scope_revision_suite.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        status="ok"
    else
        status="FAILED"
    fi

    FAM_MIN=$(( ($(date +%s) - FAM_START) / 60 ))
    COMPLETED_FAMILIES+=("$fam:$status:${FAM_MIN}min")
    echo
    echo ">>> $fam finished [$status] in ${FAM_MIN} min"
    echo ">>> elapsed so far: $(( ($(date +%s) - SUITE_START) / 3600 ))h"
    echo
done

echo "=============================================================="
echo " Per-family summary"
for entry in "${COMPLETED_FAMILIES[@]}"; do echo "   $entry"; done
echo " Total elapsed: $(( ($(date +%s) - SUITE_START) / 3600 ))h"
echo "=============================================================="

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run — skipping aggregation."
    exit 0
fi

aggregate

cat <<'EOF'

==============================================================
 Reviewer coverage after this run
   R1  non-image domain ............ audio_fsdd
   R1  Dirichlet sweep / severe ..... dirichlet_severity (now incl. 0.01)
   R1  mMIMO scale ................. scale_and_nondivisible (to N=500)
   R1  confidence intervals ........ every family, 3-5 seeds
   R2.2 privacy (Laplace/surrogate). histogram_privacy
   R2.4 multi-seed + abs-accuracy .. aggregate_results.py
   R2.5 channel/energy-aware ....... channel_energy
   R3  severe non-IID .............. dirichlet_severity
   R3  dropout / async ............. dropout, bounded_staleness
   R3  public-set sensitivity ...... public_dataset_sensitivity

 Needs NO further GPU time:
   Attached-doc "different values of R" — every run stores all 101 per-round
   rows (incl. fairness_gini, rolling_window_gini), so R in {25,50,75,100} is
   a post-hoc slice of data already on disk.

 Still NOT experimental (theory / text work):
   R1, R2.3, Attached-1  partial-participation convergence theorem
   R2.6  editorial fixes (K|N definition, spacing, title case)
   R3    Figure 1 resolution (re-render)
==============================================================
EOF
