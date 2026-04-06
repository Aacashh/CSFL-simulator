#!/usr/bin/env bash
# =============================================================================
# APEX v2 — IEEE TAI Experiment Suite
# Runs all experiments from docs/APEX_v2_experiment_plan.md iteratively.
#
# Usage:
#   bash scripts/run_apex_v2_experiments.sh              # Run everything
#   bash scripts/run_apex_v2_experiments.sh --fast        # Fast mode (debug)
#   bash scripts/run_apex_v2_experiments.sh --exp 4       # Run only Experiment 4
#   bash scripts/run_apex_v2_experiments.sh --seed 42     # Run only seed 42
#   bash scripts/run_apex_v2_experiments.sh --exp 1 --seed 42  # Exp 1, seed 42 only
#   bash scripts/run_apex_v2_experiments.sh --resume      # Skip runs whose output dir exists
#   bash scripts/run_apex_v2_experiments.sh --dry-run     # Print commands without executing
# =============================================================================
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
FAST_FLAG="--no-fast-mode"
RUN_ONLY=""
SEED_ONLY=""
RESUME=false
DRY_RUN=false
DEVICE="cuda"

# ---- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)    FAST_FLAG="--fast-mode"; shift ;;
        --exp)     RUN_ONLY="$2"; shift 2 ;;
        --seed)    SEED_ONLY="$2"; shift 2 ;;
        --resume)  RESUME=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device)  DEVICE="$2"; shift 2 ;;
        --cpu)     DEVICE="cpu"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Shared config -----------------------------------------------------------
BASE="--no-fast-mode --track-grad-norm --device ${DEVICE}"
if [[ "$FAST_FLAG" == "--fast-mode" ]]; then
    BASE="--fast-mode --track-grad-norm --device ${DEVICE}"
fi

SEEDS=(42 123 456)

# ---- Method sets -------------------------------------------------------------
MAIN_METHODS="baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.tifl,system_aware.poc,heuristic.mmr_diverse,ml.fedcor,ml.apex_v2"
CORE_METHODS="baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2"
SCALE_METHODS="baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2"
ABLATION_METHODS="ml.apex_v2,ml.apex_v2_no_adaptive_recency,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg,ml.apex_v2_no_adaptive_gamma"
V1V2_METHODS="ml.apex,ml.apex_v2"

# ---- Helpers -----------------------------------------------------------------
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=""

log() {
    echo ""
    echo "========== [$(date '+%Y-%m-%d %H:%M:%S')] $1 =========="
    echo ""
}

should_run_exp() { [[ -z "$RUN_ONLY" ]] || [[ "$RUN_ONLY" == "$1" ]]; }
should_run_seed() { [[ -z "$SEED_ONLY" ]] || [[ "$SEED_ONLY" == "$1" ]]; }

# Check if a run name already has output (for --resume)
run_exists() {
    local name="$1"
    # Look for any directory matching the name prefix in artifacts/runs/
    local matches
    matches=$(find artifacts/runs -maxdepth 1 -type d -name "${name}_*" 2>/dev/null | head -1)
    [[ -n "$matches" ]]
}

# Run a single experiment. Args: name, then all remaining args to pass.
run_one() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))

    if $RESUME && run_exists "$name"; then
        echo "  [SKIP] ${name} — output already exists (--resume)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY] python -m csfl_simulator compare --name ${name} $*"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local start_ts
    start_ts=$(date +%s)
    echo "  [RUN] ${name}"

    if python -m csfl_simulator compare --name "${name}" "$@"; then
        local end_ts elapsed
        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        echo "  [OK]  ${name} — ${elapsed}s"
        PASSED=$((PASSED + 1))
    else
        local end_ts elapsed
        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        echo "  [FAIL] ${name} — ${elapsed}s"
        FAILED=$((FAILED + 1))
        FAILURES="${FAILURES}\n  - ${name}"
    fi
}

# Run a compare experiment across all (or filtered) seeds.
# Args: exp_number name_prefix methods [extra_args...]
run_seeded() {
    local exp="$1" prefix="$2" methods="$3"
    shift 3
    for s in "${SEEDS[@]}"; do
        should_run_seed "$s" || continue
        run_one "${prefix}_s${s}" \
            --methods "${methods}" \
            ${BASE} \
            --seed "$s" \
            "$@"
    done
}

# =============================================================================
GLOBAL_START=$(date +%s)
log "APEX v2 IEEE TAI Experiment Suite"
echo "  Device:  ${DEVICE}"
echo "  Mode:    $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:  ${RESUME}"
echo "  Dry-run: ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Exp:     ${RUN_ONLY} only"
[[ -n "$SEED_ONLY" ]] && echo "  Seed:    ${SEED_ONLY} only"

# =============================================================================
# Experiment 1: Main Benchmark — CIFAR-10, alpha=0.3, N=50, K=10, 200 rounds
# Paper: Table I, Figure 2
# Addresses: P0 #1 (accuracy), P0 #4 (rounds), P0 #5 (baselines), P1 #9 (seeds)
# =============================================================================
if should_run_exp 1; then
    log "EXP 1: Main Benchmark (CIFAR-10, alpha=0.3, ResNet18, 200 rounds)"
    run_seeded 1 "main_cifar10_a03" "${MAIN_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 2: Heterogeneity Robustness — alpha sweep {0.1, 0.6}
# (alpha=0.3 is covered by Exp 1)
# Paper: Table II, Figure 3
# Addresses: Fix 2 (oscillation at alpha=0.1), Fix 3 (overexploration at alpha=0.6)
# =============================================================================
if should_run_exp 2; then
    log "EXP 2a: Extreme non-IID (alpha=0.1)"
    run_seeded 2 "het_a01" "${CORE_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 2b: Mild non-IID (alpha=0.6)"
    run_seeded 2 "het_a06" "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 3: Scalability — N in {100, 200, 500}
# Paper: Table III, Figure 4
# Addresses: P2 #12 (scalability), Fix 1 (adaptive recency)
# =============================================================================
if should_run_exp 3; then
    log "EXP 3a: Scalability N=100, K=10"
    run_seeded 3 "scale_n100" "${SCALE_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200

    log "EXP 3b: Scalability N=200, K=20"
    run_seeded 3 "scale_n200" "${SCALE_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 200 --clients-per-round 20 --rounds 200

    log "EXP 3c: Scalability N=500, K=50"
    run_seeded 3 "scale_n500" "${SCALE_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 500 --clients-per-round 50 --rounds 200
fi

# =============================================================================
# Experiment 4: Ablation Study — at main benchmark settings (N=50, NOT N=100)
# Paper: Table IV, Figure 5
# Addresses: P0 #2 (ablation contradicts claims)
# =============================================================================
if should_run_exp 4; then
    log "EXP 4: Ablation (CIFAR-10, alpha=0.3, N=50, K=10)"
    run_seeded 4 "ablation" "${ABLATION_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 5: Dataset Diversity — CIFAR-100, Fashion-MNIST, MNIST
# Paper: Table V
# Addresses: P2 #13 (limited dataset diversity)
# =============================================================================
if should_run_exp 5; then
    log "EXP 5a: CIFAR-100 (ResNet18)"
    run_seeded 5 "cifar100" "${CORE_METHODS}" \
        --dataset CIFAR-100 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 5b: Fashion-MNIST (CNN-MNIST)"
    run_seeded 5 "fmnist" "${CORE_METHODS}" \
        --dataset Fashion-MNIST --partition dirichlet --dirichlet-alpha 0.3 \
        --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 5c: MNIST (CNN-MNIST)"
    run_seeded 5 "mnist" "${SCALE_METHODS}" \
        --dataset MNIST --partition dirichlet --dirichlet-alpha 0.3 \
        --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 6: IID Sanity Check
# Validates Fix 3 (het-aware diversity sets w_div near 0 on IID)
# =============================================================================
if should_run_exp 6; then
    log "EXP 6: IID Baseline (CIFAR-10)"
    run_seeded 6 "iid" "${SCALE_METHODS}" \
        --dataset CIFAR-10 --partition iid \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 7: v1 vs v2 Head-to-Head
# Direct comparison across settings to quantify the five fixes
# =============================================================================
if should_run_exp 7; then
    log "EXP 7: v1 vs v2 (alpha=0.3, N=50)"
    run_one "v1v2_a03_s42" --methods "${V1V2_METHODS}" ${BASE} --seed 42 \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 7: v1 vs v2 (alpha=0.1, N=50)"
    run_one "v1v2_a01_s42" --methods "${V1V2_METHODS}" ${BASE} --seed 42 \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 7: v1 vs v2 (alpha=0.6, N=50)"
    run_one "v1v2_a06_s42" --methods "${V1V2_METHODS}" ${BASE} --seed 42 \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "EXP 7: v1 vs v2 (alpha=0.3, N=100)"
    run_one "v1v2_n100_s42" --methods "${V1V2_METHODS}" ${BASE} --seed 42 \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Experiment 8: LightCNN comparison (model-independence check)
# =============================================================================
if should_run_exp 8; then
    log "EXP 8: LightCNN (CIFAR-10, alpha=0.3)"
    run_seeded 8 "lightcnn_a03" "${CORE_METHODS}" \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# Generate plots for all completed runs
# =============================================================================
if should_run_exp plots; then
    log "Generating IEEE-ready EPS plots"

    for run_name in \
        main_cifar10_a03_s42 het_a01_s42 het_a06_s42 \
        scale_n100_s42 scale_n200_s42 scale_n500_s42 \
        ablation_s42 cifar100_s42 fmnist_s42 \
        v1v2_a03_s42 v1v2_n100_s42 iid_s42 lightcnn_a03_s42; do
        echo "  Plotting ${run_name}..."
        python -m csfl_simulator plot --run "${run_name}" \
            --metrics accuracy,loss,fairness_gini --format eps 2>/dev/null || \
            echo "  [WARN] Plot failed for ${run_name} (run may not exist yet)"
    done
fi

# =============================================================================
# Summary
# =============================================================================
GLOBAL_END=$(date +%s)
GLOBAL_ELAPSED=$(( GLOBAL_END - GLOBAL_START ))
HOURS=$(( GLOBAL_ELAPSED / 3600 ))
MINS=$(( (GLOBAL_ELAPSED % 3600) / 60 ))

log "DONE"
echo "  Total:   ${TOTAL}"
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "  Time:    ${HOURS}h ${MINS}m"

if [[ -n "$FAILURES" ]]; then
    echo ""
    echo "  Failed runs:"
    echo -e "${FAILURES}"
fi

echo ""
