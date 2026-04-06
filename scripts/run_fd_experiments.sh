#!/usr/bin/env bash
# =============================================================================
# FD Client Selection Experiment Suite
# Run all experiments iteratively. Each experiment is logged with timestamp.
# Usage: bash scripts/run_fd_experiments.sh [--fast] [--exp N]
#   --fast   : Use fast mode (2 batches per round) for debugging
#   --exp N  : Run only experiment N (1-12)
# =============================================================================
set -euo pipefail

FAST_FLAG="--no-fast-mode"
RUN_ONLY=""
for arg in "$@"; do
    case $arg in
        --fast) FAST_FLAG="--fast-mode" ;;
        --exp) shift; RUN_ONLY="$1" ;;
        [0-9]*) RUN_ONLY="$arg" ;;
    esac
done

# Common base config (paper-matched)
BASE_FD="--paradigm fd --public-dataset-size 2000 --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 --batch-size 128 --distillation-batch-size 500 --distillation-lr 0.001 --distillation-epochs 2 --temperature 1.0 --fd-optimizer adam --n-bs-antennas 64 --quantization-bits 8 ${FAST_FLAG}"

# Method sets
ALL_METHODS="heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.maml_select,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
CORE_METHODS="heuristic.random,system_aware.oort,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
COMPACT_METHODS="heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.noise_robust_fair"

CIFAR_MODELS="ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

log() { echo ""; echo "========== [$(date '+%Y-%m-%d %H:%M:%S')] $1 =========="; echo ""; }

should_run() { [ -z "$RUN_ONLY" ] || [ "$RUN_ONLY" = "$1" ]; }

# =============================================================================
# Experiment 1: Main Method Comparison (CIFAR-10, N=50, K=15, R=300)
# =============================================================================
if should_run 1; then
log "EXP 1: Main method comparison (CIFAR-10, N=50, K=15, 300 rounds)"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_cifar10_main \
    --methods "${ALL_METHODS}" \
    --dataset CIFAR-10 --public-dataset STL-10 \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --seed 42

python -m csfl_simulator plot --run fd_cifar10_main \
    --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini \
    --format eps --width 7.16 --height 5.0

python scripts/plot_fd_experiments.py --run fd_cifar10_main \
    --metrics accuracy --format eps --bar
fi

# =============================================================================
# Experiment 2: FL vs FD Ranking Inversion
# =============================================================================
if should_run 2; then
log "EXP 2: FL baseline for ranking inversion proof (CIFAR-10)"
python -m csfl_simulator compare \
    --paradigm fl --name fl_cifar10_baseline \
    --methods "${ALL_METHODS}" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 \
    --model LightCNN \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    ${FAST_FLAG} --seed 42

python scripts/plot_fd_experiments.py --run fl_cifar10_baseline \
    --metrics accuracy --format eps --bar
fi

# =============================================================================
# Experiment 3: Noise Sensitivity Sweep (5 DL SNR levels)
# =============================================================================
if should_run 3; then
log "EXP 3: Noise sensitivity sweep"
for dl_snr in "errfree" "0" "-10" "-20" "-30"; do
    if [ "$dl_snr" = "errfree" ]; then
        noise_flag=""
        label="errfree"
    else
        noise_flag="--channel-noise --ul-snr-db -8 --dl-snr-db ${dl_snr}"
        label="dl${dl_snr}"
    fi
    log "  Noise sweep: DL SNR = ${label}"
    python -m csfl_simulator compare ${BASE_FD} \
        --name "fd_cifar10_noise_${label}" \
        --methods "${CORE_METHODS}" \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        ${noise_flag} --seed 42
done

for label in errfree dl0 dl-10 dl-20 dl-30; do
    python scripts/plot_fd_experiments.py --run "fd_cifar10_noise_${label}" \
        --metrics accuracy --format eps
done
fi

# =============================================================================
# Experiment 4: Non-IID Sweep (Dirichlet alpha)
# =============================================================================
if should_run 4; then
log "EXP 4: Non-IID heterogeneity sweep"
for alpha in 0.1 0.3 0.5 1.0 5.0 10.0; do
    # Convert dots to underscores for filename
    alpha_label=$(echo "$alpha" | tr '.' '_')
    log "  Alpha = ${alpha}"
    python -m csfl_simulator compare ${BASE_FD} \
        --name "fd_cifar10_alpha_${alpha_label}" \
        --methods "${CORE_METHODS}" \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha ${alpha} \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
done

for alpha_label in 0_1 0_3 0_5 1_0 5_0 10_0; do
    python scripts/plot_fd_experiments.py --run "fd_cifar10_alpha_${alpha_label}" \
        --metrics accuracy --format eps
done
fi

# =============================================================================
# Experiment 5: K Sweep (Selection Ratio)
# =============================================================================
if should_run 5; then
log "EXP 5: K sweep (selection ratio)"
for K in 5 10 15 25 50; do
    log "  K = ${K}"
    python -m csfl_simulator compare ${BASE_FD} \
        --name "fd_cifar10_K${K}" \
        --methods "${CORE_METHODS}" \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round ${K} --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
done

for K in 5 10 15 25 50; do
    python scripts/plot_fd_experiments.py --run "fd_cifar10_K${K}" \
        --metrics accuracy --format eps
done
fi

# =============================================================================
# Experiment 6: Scaling to N=100
# =============================================================================
if should_run 6; then
log "EXP 6: Scaling test (N=100, K=30)"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_cifar10_N100 \
    --methods "${CORE_METHODS}" \
    --dataset CIFAR-10 --public-dataset STL-10 \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
    --total-clients 100 --clients-per-round 30 --rounds 400 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --seed 42

python -m csfl_simulator plot --run fd_cifar10_N100 \
    --metrics accuracy,kl_divergence_avg,fairness_gini --format eps
fi

# =============================================================================
# Experiment 7: Group-Based FD (FedTSKD-G)
# =============================================================================
if should_run 7; then
log "EXP 7: Group-based FD (FedTSKD-G)"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_cifar10_group \
    --methods "heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max" \
    --dataset CIFAR-10 --public-dataset STL-10 \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --group-based --channel-threshold 0.5 \
    --seed 42

python -m csfl_simulator plot --run fd_cifar10_group \
    --metrics accuracy,kl_divergence_avg --format eps
fi

# =============================================================================
# Experiment 8: MNIST/FMNIST Cross-Dataset Validation
# =============================================================================
if should_run 8; then
log "EXP 8: MNIST/FMNIST cross-dataset validation"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_mnist_main \
    --methods "${ALL_METHODS}" \
    --dataset MNIST --public-dataset FMNIST \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${MNIST_MODELS}" \
    --total-clients 50 --clients-per-round 15 --rounds 200 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --seed 42

python -m csfl_simulator plot --run fd_mnist_main \
    --metrics accuracy,kl_divergence_avg,effective_noise_var,fairness_gini \
    --format eps --width 7.16 --height 5.0

python scripts/plot_fd_experiments.py --run fd_mnist_main \
    --metrics accuracy --format eps --bar
fi

# =============================================================================
# Experiment 9: Communication Efficiency (FL baseline)
# =============================================================================
if should_run 9; then
log "EXP 9: Communication efficiency comparison (FL vs FD)"
python -m csfl_simulator compare \
    --paradigm fl --name fl_cifar10_comm \
    --methods "heuristic.random,ml.apex_v2" \
    --dataset CIFAR-10 --partition dirichlet --dirichlet-alpha 0.5 \
    --model LightCNN \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    ${FAST_FLAG} --seed 42

python scripts/plot_fd_experiments.py --run fl_cifar10_comm \
    --metrics accuracy,cum_comm --format eps
fi

# =============================================================================
# Experiment 10: Antenna Count Sweep
# =============================================================================
if should_run 10; then
log "EXP 10: Antenna count sweep"
for ant in 32 64 128; do
    log "  N_BS = ${ant} antennas"
    python -m csfl_simulator compare ${BASE_FD} \
        --name "fd_cifar10_ant${ant}" \
        --methods "${COMPACT_METHODS}" \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --n-bs-antennas ${ant} \
        --seed 42
done

for ant in 32 64 128; do
    python scripts/plot_fd_experiments.py --run "fd_cifar10_ant${ant}" \
        --metrics accuracy --format eps
done
fi

# =============================================================================
# Experiment 11: Ablation Studies
# =============================================================================
if should_run 11; then
log "EXP 11a: SNRD ablation"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_cifar10_ablation_snrd \
    --methods "fd_native.snr_diversity,fd_native.snrd_ablation_fixed_w,fd_native.snrd_ablation_no_channel,fd_native.snrd_ablation_no_diversity,fd_native.snrd_ablation_no_fairness" \
    --dataset CIFAR-10 --public-dataset STL-10 \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --seed 42

log "EXP 11b: LQTS ablation"
python -m csfl_simulator compare ${BASE_FD} \
    --name fd_cifar10_ablation_lqts \
    --methods "fd_native.logit_quality_ts,fd_native.lqts_ablation_global_reward,fd_native.lqts_ablation_no_diversity,fd_native.lqts_ablation_no_recency" \
    --dataset CIFAR-10 --public-dataset STL-10 \
    --partition dirichlet --dirichlet-alpha 0.5 \
    --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
    --total-clients 50 --clients-per-round 15 --rounds 300 \
    --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
    --seed 42

python scripts/plot_fd_experiments.py --run fd_cifar10_ablation_snrd \
    --metrics accuracy --format eps --bar
python scripts/plot_fd_experiments.py --run fd_cifar10_ablation_lqts \
    --metrics accuracy --format eps --bar
fi

# =============================================================================
# Experiment 12: Multi-Seed Statistical Significance
# =============================================================================
if should_run 12; then
log "EXP 12: Multi-seed runs for statistical significance"
SEED_METHODS="heuristic.random,ml.apex_v2,fd_native.snr_diversity,fd_native.logit_quality_ts,fd_native.noise_robust_fair,fd_native.logit_entropy_max"
for seed in 0 1 2 42 100; do
    log "  Seed = ${seed}"
    python -m csfl_simulator compare ${BASE_FD} \
        --name "fd_cifar10_seed${seed}" \
        --methods "${SEED_METHODS}" \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 50 --clients-per-round 15 --rounds 300 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed ${seed}
done

for seed in 0 1 2 42 100; do
    python scripts/plot_fd_experiments.py --run "fd_cifar10_seed${seed}" \
        --metrics accuracy --format eps
done
fi

# =============================================================================
log "ALL EXPERIMENTS COMPLETE"
echo "Results in: artifacts/runs/"
echo "Run 'python -m csfl_simulator list-runs' to see all results."
