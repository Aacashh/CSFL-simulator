#!/usr/bin/env bash
# =============================================================================
# Combined Experiment Suite — APEX v2 (FL) + FD Client Selection
# Runs ALL experiments unattended. Designed for HPC batch jobs.
#
# Usage:
#   bash scripts/run_all_experiments.sh                    # Run everything
#   bash scripts/run_all_experiments.sh --fast             # Fast mode (debug)
#   bash scripts/run_all_experiments.sh --exp apex:1       # APEX Exp 1 only
#   bash scripts/run_all_experiments.sh --exp fd:3         # FD Exp 3 only
#   bash scripts/run_all_experiments.sh --exp apex         # All APEX experiments
#   bash scripts/run_all_experiments.sh --exp fd           # All FD experiments
#   bash scripts/run_all_experiments.sh --seed 42          # Single seed only
#   bash scripts/run_all_experiments.sh --resume           # Skip existing runs
#   bash scripts/run_all_experiments.sh --dry-run          # Print without executing
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
BASE_FL="--track-grad-norm --device ${DEVICE} ${FAST_FLAG}"
BASE_FD="--paradigm fd --public-dataset-size 2000 --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --distillation-epochs 2 --temperature 1.0 --fd-optimizer adam --n-bs-antennas 64 --quantization-bits 8 --device ${DEVICE} ${FAST_FLAG}"

SEEDS=(42 123 456)

# ---- APEX method sets --------------------------------------------------------
APEX_MAIN="baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.tifl,system_aware.poc,heuristic.mmr_diverse,ml.fedcor,ml.apex_v2"
APEX_CORE="baseline.fedavg,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2"
APEX_SCALE="baseline.fedavg,system_aware.oort,system_aware.poc,ml.apex_v2"
APEX_ABLATION="ml.apex_v2,ml.apex_v2_no_adaptive_recency,ml.apex_v2_no_hysteresis,ml.apex_v2_no_het_scaling,ml.apex_v2_no_posterior_reg,ml.apex_v2_no_adaptive_gamma"
APEX_V1V2="ml.apex,ml.apex_v2"

# ---- FD method sets ---------------------------------------------------------
FD_ALL="heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.maml_select,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
FD_CORE="heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
FD_COMPACT="heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair"
FD_SEED="heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
CIFAR_MODELS="ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

# ---- Counters ----------------------------------------------------------------
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=""

# ---- Helpers -----------------------------------------------------------------
log() {
    echo ""
    echo "========== [$(date '+%Y-%m-%d %H:%M:%S')] $1 =========="
    echo ""
}

# Filter logic: --exp apex => all APEX, --exp fd:3 => FD exp 3, --exp apex:1 => APEX exp 1
# Supports comma-separated: --exp "apex:4,apex:6,apex:7"
should_run() {
    local suite="$1" exp="$2"  # e.g. suite=apex exp=1
    [[ -z "$RUN_ONLY" ]] && return 0
    IFS=',' read -ra FILTERS <<< "$RUN_ONLY"
    for f in "${FILTERS[@]}"; do
        [[ "$f" == "${suite}:${exp}" ]] && return 0
        [[ "$f" == "${suite}" ]] && return 0
        [[ "$f" =~ ^[0-9]+$ ]] && [[ "$suite" == "apex" ]] && [[ "$f" == "$exp" ]] && return 0
    done
    return 1
}

should_run_seed() { [[ -z "$SEED_ONLY" ]] || [[ "$SEED_ONLY" == "$1" ]]; }

run_exists() {
    local name="$1"
    local matches
    matches=$(find artifacts/runs -maxdepth 1 -type d -name "${name}_*" 2>/dev/null | head -1)
    [[ -n "$matches" ]]
}

run_one() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))

    if $RESUME && run_exists "$name"; then
        echo "  [SKIP] ${name} -- output exists (--resume)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY] python -m csfl_simulator compare --name ${name} $*"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local start_ts end_ts elapsed
    start_ts=$(date +%s)
    echo "  [RUN] ${name}"

    if python -m csfl_simulator compare --name "${name}" "$@"; then
        end_ts=$(date +%s); elapsed=$(( end_ts - start_ts ))
        echo "  [OK]  ${name} -- ${elapsed}s"
        PASSED=$((PASSED + 1))
    else
        end_ts=$(date +%s); elapsed=$(( end_ts - start_ts ))
        echo "  [FAIL] ${name} -- ${elapsed}s"
        FAILED=$((FAILED + 1))
        FAILURES="${FAILURES}\n  - ${name}"
    fi
}

# run_one but for single-method `run` commands (FD exp 9)
run_one_single() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))

    if $RESUME && run_exists "$name"; then
        echo "  [SKIP] ${name} -- output exists (--resume)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY] python -m csfl_simulator $*"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local start_ts end_ts elapsed
    start_ts=$(date +%s)
    echo "  [RUN] ${name}"

    if python -m csfl_simulator "$@"; then
        end_ts=$(date +%s); elapsed=$(( end_ts - start_ts ))
        echo "  [OK]  ${name} -- ${elapsed}s"
        PASSED=$((PASSED + 1))
    else
        end_ts=$(date +%s); elapsed=$(( end_ts - start_ts ))
        echo "  [FAIL] ${name} -- ${elapsed}s"
        FAILED=$((FAILED + 1))
        FAILURES="${FAILURES}\n  - ${name}"
    fi
}

run_seeded() {
    local prefix="$1" methods="$2"
    shift 2
    for s in "${SEEDS[@]}"; do
        should_run_seed "$s" || continue
        run_one "${prefix}_s${s}" \
            --methods "${methods}" \
            --seed "$s" \
            "$@"
    done
}

plot_safe() {
    local run_name="$1"; shift
    if $DRY_RUN; then
        echo "  [DRY] plot ${run_name}"
        return 0
    fi
    python -m csfl_simulator plot --run "${run_name}" "$@" 2>/dev/null || \
        echo "  [WARN] Plot skipped: ${run_name}"
}

plot_fd_safe() {
    local run_name="$1"; shift
    if $DRY_RUN; then
        echo "  [DRY] plot_fd ${run_name}"
        return 0
    fi
    python scripts/plot_fd_experiments.py --run "${run_name}" "$@" 2>/dev/null || \
        echo "  [WARN] FD plot skipped: ${run_name}"
}

# =============================================================================
GLOBAL_START=$(date +%s)
log "Combined Experiment Suite (APEX v2 FL + FD)"
echo "  Device:  ${DEVICE}"
echo "  Mode:    $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:  ${RESUME}"
echo "  Dry-run: ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:  ${RUN_ONLY}"
[[ -n "$SEED_ONLY" ]] && echo "  Seed:    ${SEED_ONLY} only"

# #############################################################################
#                        PART A: APEX v2 (FL) EXPERIMENTS
# #############################################################################

# =============================================================================
# APEX 1: Main Benchmark -- CIFAR-10, alpha=0.3, N=50, K=10, 200 rounds
# Paper: Table I, Figure 2
# =============================================================================
if should_run apex 1; then
    log "APEX 1: Main Benchmark (CIFAR-10, alpha=0.3, ResNet18, 200 rounds)"
    run_seeded "main_cifar10_a03" "${APEX_MAIN}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX 2: Heterogeneity Robustness -- alpha={0.1, 0.6}
# Paper: Table II, Figure 3
# =============================================================================
if should_run apex 2; then
    log "APEX 2a: Extreme non-IID (alpha=0.1)"
    run_seeded "het_a01" "${APEX_CORE}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "APEX 2b: Mild non-IID (alpha=0.6)"
    run_seeded "het_a06" "baseline.fedavg,system_aware.fedcs,system_aware.oort,system_aware.poc,heuristic.mmr_diverse,ml.apex_v2" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX 3: Scalability -- N={100, 200, 500}
# Paper: Table III, Figure 4
# =============================================================================
if should_run apex 3; then
    log "APEX 3a: Scalability N=100, K=10"
    run_seeded "scale_n100" "${APEX_SCALE}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200

    log "APEX 3b: Scalability N=200, K=20"
    run_seeded "scale_n200" "${APEX_SCALE}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 200 --clients-per-round 20 --rounds 200

    # log "APEX 3c: Scalability N=500, K=50"
    # run_seeded "scale_n500" "${APEX_SCALE}" \
    #     ${BASE_FL} \
    #     --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
    #     --model ResNet18 --total-clients 500 --clients-per-round 50 --rounds 200
fi

# =============================================================================
# APEX 4: Ablation Study -- at main benchmark settings (N=50)
# Paper: Table IV, Figure 5
# =============================================================================
if should_run apex 4; then
    log "APEX 4: Ablation (CIFAR-10, alpha=0.3, N=50, K=10)"
    run_seeded "ablation" "${APEX_ABLATION}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX 5: Dataset Diversity -- CIFAR-100, Fashion-MNIST, MNIST
# Paper: Table V
# =============================================================================
if should_run apex 5; then
    log "APEX 5a: CIFAR-100 (ResNet18)"
    run_seeded "cifar100" "${APEX_CORE}" \
        ${BASE_FL} \
        --dataset CIFAR-100 --partition dirichlet --dirichlet-alpha 0.3 \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

    log "APEX 5b: Fashion-MNIST (CNN-MNIST)"
    run_seeded "fmnist" "${APEX_CORE}" \
        ${BASE_FL} \
        --dataset Fashion-MNIST --partition dirichlet --dirichlet-alpha 0.3 \
        --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200

    log "APEX 5c: MNIST (CNN-MNIST)"
    run_seeded "mnist" "${APEX_SCALE}" \
        ${BASE_FL} \
        --dataset MNIST --partition dirichlet --dirichlet-alpha 0.3 \
        --model CNN-MNIST --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX 6: IID Sanity Check
# =============================================================================
if should_run apex 6; then
    log "APEX 6: IID Baseline (CIFAR-10)"
    run_seeded "iid" "${APEX_SCALE}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition iid \
        --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX 7: v1 vs v2 Head-to-Head
# =============================================================================
if should_run apex 7; then
#     log "APEX 7: v1 vs v2 (alpha=0.3, N=50)"
#     run_one "v1v2_a03_s42" --methods "${APEX_V1V2}" ${BASE_FL} --seed 42 \
#         --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
#         --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

#     log "APEX 7: v1 vs v2 (alpha=0.1, N=50)"
#     run_one "v1v2_a01_s42" --methods "${APEX_V1V2}" ${BASE_FL} --seed 42 \
#         --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.1 \
#         --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

#     log "APEX 7: v1 vs v2 (alpha=0.6, N=50)"
#     run_one "v1v2_a06_s42" --methods "${APEX_V1V2}" ${BASE_FL} --seed 42 \
#         --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.6 \
#         --model ResNet18 --total-clients 50 --clients-per-round 10 --rounds 200

#     log "APEX 7: v1 vs v2 (alpha=0.3, N=100)"
#     run_one "v1v2_n100_s42" --methods "${APEX_V1V2}" ${BASE_FL} --seed 42 \
#         --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
#         --model ResNet18 --total-clients 100 --clients-per-round 10 --rounds 200
    echo "  [SKIP] APEX 7 (v1 vs v2) — commented out"
fi

# =============================================================================
# APEX 8: LightCNN comparison (model-independence)
# =============================================================================
if should_run apex 8; then
    log "APEX 8: LightCNN (CIFAR-10, alpha=0.3)"
    run_seeded "lightcnn_a03" "${APEX_CORE}" \
        ${BASE_FL} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.3 \
        --model LightCNN --total-clients 50 --clients-per-round 10 --rounds 200
fi

# =============================================================================
# APEX plots
# =============================================================================
if should_run apex plots; then
    log "Generating APEX IEEE-ready EPS plots"
    for run_name in \
        main_cifar10_a03_s42 het_a01_s42 het_a06_s42 \
        scale_n100_s42 scale_n200_s42 scale_n500_s42 \
        ablation_s42 cifar100_s42 fmnist_s42 \
        v1v2_a03_s42 v1v2_n100_s42 iid_s42 lightcnn_a03_s42; do
        plot_safe "${run_name}" --metrics accuracy,loss,fairness_gini --format eps
    done
fi

# #############################################################################
#                        PART B: FD EXPERIMENTS
# #############################################################################

# =============================================================================
# FD 1: Main Method Comparison (CIFAR-10, N=50, K=15, R=300)
# =============================================================================
if should_run fd 1; then
    log "FD 1: Main method comparison (CIFAR-10, N=50, K=15, 300 rounds)"
    run_one "fd_cifar10_main" \
        --methods "${FD_ALL}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42

    plot_safe "fd_cifar10_main" \
        --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini \
        --format eps --width 7.16 --height 5.0
    plot_fd_safe "fd_cifar10_main" --metrics accuracy --format eps --bar
fi

# =============================================================================
# FD 2: FL vs FD Ranking Inversion
# =============================================================================
if should_run fd 2; then
    log "FD 2: FL baseline for ranking inversion (CIFAR-10)"
    run_one "fl_cifar10_baseline" \
        --methods "${FD_ALL}" \
        --paradigm fl ${FAST_FLAG} --device ${DEVICE} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 \
        --model LightCNN \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --seed 42

    plot_fd_safe "fl_cifar10_baseline" --metrics accuracy --format eps --bar
fi

# =============================================================================
# FD 3: Noise Sensitivity Sweep (5 DL SNR levels)
# =============================================================================
if should_run fd 3; then
    log "FD 3: Noise sensitivity sweep"
    for dl_snr in "errfree" "0" "-10" "-20" "-30"; do
        if [[ "$dl_snr" == "errfree" ]]; then
            noise_flag=""
            label="errfree"
        else
            noise_flag="--channel-noise --ul-snr-db -8 --dl-snr-db ${dl_snr}"
            label="dl${dl_snr}"
        fi
        log "  Noise sweep: DL SNR = ${label}"
        run_one "fd_cifar10_noise_${label}" \
            --methods "${FD_CORE}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 50 --clients-per-round 15 --rounds 300 \
            ${noise_flag} --seed 42
    done

    for label in errfree dl0 dl-10 dl-20 dl-30; do
        plot_fd_safe "fd_cifar10_noise_${label}" --metrics accuracy --format eps
    done
fi

# =============================================================================
# FD 4: Non-IID Sweep (Dirichlet alpha)
# =============================================================================
if should_run fd 4; then
    log "FD 4: Non-IID heterogeneity sweep"
    for alpha in 0.1 0.3 0.5 1.0 5.0 10.0; do
        alpha_label=$(echo "$alpha" | tr '.' '_')
        log "  Alpha = ${alpha}"
        run_one "fd_cifar10_alpha_${alpha_label}" \
            --methods "${FD_CORE}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha ${alpha} \
            --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 50 --clients-per-round 15 --rounds 300 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done

    for alpha_label in 0_1 0_3 0_5 1_0 5_0 10_0; do
        plot_fd_safe "fd_cifar10_alpha_${alpha_label}" --metrics accuracy --format eps
    done
fi

# =============================================================================
# FD 5: K Sweep (Selection Ratio)
# =============================================================================
if should_run fd 5; then
    log "FD 5: K sweep (selection ratio)"
    for K in 5 10 15 25 50; do
        log "  K = ${K}"
        run_one "fd_cifar10_K${K}" \
            --methods "${FD_CORE}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 50 --clients-per-round ${K} --rounds 300 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done

    for K in 5 10 15 25 50; do
        plot_fd_safe "fd_cifar10_K${K}" --metrics accuracy --format eps
    done
fi

# =============================================================================
# FD 6: Scaling to N=100
# =============================================================================
if should_run fd 6; then
    log "FD 6: Scaling test (N=100, K=30)"
    run_one "fd_cifar10_N100" \
        --methods "${FD_CORE}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 100 --clients-per-round 30 --rounds 400 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42

    plot_safe "fd_cifar10_N100" --metrics accuracy,kl_divergence_avg,fairness_gini --format eps
fi

# =============================================================================
# FD 7: Group-Based FD (FedTSKD-G)
# =============================================================================
if should_run fd 7; then
    log "FD 7: Group-based FD (FedTSKD-G)"
    run_one "fd_cifar10_group" \
        --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --group-based --channel-threshold 0.5 \
        --seed 42

    plot_safe "fd_cifar10_group" --metrics accuracy,kl_divergence_avg --format eps
fi

# =============================================================================
# FD 8: MNIST/FMNIST Cross-Dataset
# =============================================================================
if should_run fd 8; then
    log "FD 8: MNIST/FMNIST cross-dataset validation"
    run_one "fd_mnist_main" \
        --methods "${FD_ALL}" \
        ${BASE_FD} \
        --dataset MNIST --public-dataset FMNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 200 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42

    plot_safe "fd_mnist_main" \
        --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini \
        --format eps --width 7.16 --height 5.0
    plot_fd_safe "fd_mnist_main" --metrics accuracy --format eps --bar
fi

# =============================================================================
# FD 9: Communication Efficiency (FL vs FD)
# =============================================================================
if should_run fd 9; then
    log "FD 9: Communication efficiency comparison (FL vs FD)"
    run_one "fl_cifar10_comm" \
        --methods "heuristic.random,ml.apex_v2" \
        --paradigm fl ${FAST_FLAG} --device ${DEVICE} \
        --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 \
        --model LightCNN \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --seed 42

    plot_fd_safe "fl_cifar10_comm" --metrics accuracy,cum_comm --format eps
fi

# =============================================================================
# FD 10: Antenna Count Sweep
# =============================================================================
if should_run fd 10; then
    log "FD 10: Antenna count sweep"
    for ant in 32 64 128; do
        log "  N_BS = ${ant} antennas"
        run_one "fd_cifar10_ant${ant}" \
            --methods "${FD_COMPACT}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 50 --clients-per-round 15 --rounds 300 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --n-bs-antennas ${ant} \
            --seed 42
    done

    for ant in 32 64 128; do
        plot_fd_safe "fd_cifar10_ant${ant}" --metrics accuracy --format eps
    done
fi

# =============================================================================
# FD 11: Ablation Studies (SNRD + LQTS)
# =============================================================================
if should_run fd 11; then
    log "FD 11a: SNRD ablation"
    run_one "fd_cifar10_ablation_snrd" \
        --methods "fd_native.snr_diversity,fd_native.snrd_ablation_fixed_w,fd_native.snrd_ablation_no_channel,fd_native.snrd_ablation_no_diversity,fd_native.snrd_ablation_no_fairness" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42

    log "FD 11b: LQTS ablation"
    run_one "fd_cifar10_ablation_lqts" \
        --methods "fd_native.logit_quality_ts,fd_native.lqts_ablation_global_reward,fd_native.lqts_ablation_no_diversity,fd_native.lqts_ablation_no_recency" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42

    plot_fd_safe "fd_cifar10_ablation_snrd" --metrics accuracy --format eps --bar
    plot_fd_safe "fd_cifar10_ablation_lqts" --metrics accuracy --format eps --bar
fi

# =============================================================================
# FD 12: Multi-Seed Statistical Significance
# =============================================================================
if should_run fd 12; then
    log "FD 12: Multi-seed runs"
    for seed in 0 1 2 42 100; do
        log "  Seed = ${seed}"
        run_one "fd_cifar10_seed${seed}" \
            --methods "${FD_SEED}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 50 --clients-per-round 15 --rounds 300 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed ${seed}
    done

    for seed in 0 1 2 42 100; do
        plot_fd_safe "fd_cifar10_seed${seed}" --metrics accuracy --format eps
    done
fi

# =============================================================================
# FD plots
# =============================================================================
if should_run fd plots; then
    log "Generating FD plots"
    for run_name in \
        fd_cifar10_main fd_mnist_main fd_cifar10_group fd_cifar10_N100 \
        fd_cifar10_ablation_snrd fd_cifar10_ablation_lqts; do
        plot_safe "${run_name}" --metrics accuracy,kl_divergence_avg,fairness_gini --format eps
    done
fi

# #############################################################################
#                              SUMMARY
# #############################################################################
GLOBAL_END=$(date +%s)
GLOBAL_ELAPSED=$(( GLOBAL_END - GLOBAL_START ))
HOURS=$(( GLOBAL_ELAPSED / 3600 ))
MINS=$(( (GLOBAL_ELAPSED % 3600) / 60 ))
SECS=$(( GLOBAL_ELAPSED % 60 ))

log "ALL EXPERIMENTS COMPLETE"
echo "  Total:   ${TOTAL}"
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "  Time:    ${HOURS}h ${MINS}m ${SECS}s"

if [[ -n "$FAILURES" ]]; then
    echo ""
    echo "  Failed runs:"
    echo -e "${FAILURES}"
fi

echo ""
echo "  Results in: artifacts/runs/"
echo "  Run: python -m csfl_simulator list-runs"