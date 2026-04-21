#!/usr/bin/env bash
# =============================================================================
# Trimmed CALM-FD Experiment Suite (v2) — tightened for compute budget.
#
# Changes from scripts/run_next_sota_experiments.sh (per advisor feedback):
#   - Drops Exp 1 (paper replication) — already run, no need to repeat.
#   - Drops Exp 9 (multi-seed reproducibility) — single-seed only.
#   - Rounds 200 -> 100 across every experiment.
#   - SOTA comparison set trimmed to 4 FL methods + CALM:
#       random, FedCS, Oort, PoC, calm_fd
#     LQTS + APEX + label_coverage + snr_diversity dropped from sweeps.
#   - Alpha sweep trimmed 4 -> 3 values (low / mid / high-IID: 0.1, 0.5, 5.0).
#   - Baseline study stays N=30, K=10 (already matched advisor's ask).
#   - Ablation study keeps CALM's 5 knockout variants but runs 100 rounds now.
#
# Usage:
#   bash scripts/run_next_sota_experiments_v2.sh                    # Run everything
#   bash scripts/run_next_sota_experiments_v2.sh --fast             # Fast debug mode
#   bash scripts/run_next_sota_experiments_v2.sh --exp 1            # Single experiment
#   bash scripts/run_next_sota_experiments_v2.sh --exp "1,2,3"      # Multiple
#   bash scripts/run_next_sota_experiments_v2.sh --resume           # Skip existing
#   bash scripts/run_next_sota_experiments_v2.sh --dry-run          # Print plan only
#   bash scripts/run_next_sota_experiments_v2.sh --cpu              # Force CPU
# =============================================================================
set -euo pipefail

# ---- Defaults ----
FAST_FLAG="--no-fast-mode"
RUN_ONLY=""
RESUME=false
DRY_RUN=false
DEVICE="cuda"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)    FAST_FLAG="--fast-mode"; shift ;;
        --exp)     RUN_ONLY="$2"; shift 2 ;;
        --resume)  RESUME=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device)  DEVICE="$2"; shift 2 ;;
        --cpu)     DEVICE="cpu"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Shared config — paper-matched
# =============================================================================
CIFAR_MODELS="ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

# Speedup stack (CUDA-only where noted):
#   --parallel-clients -1  Auto CUDA-stream parallelism across clients in a round.
#                          Default is already -1 but we set it explicitly so the intent is visible.
#   --use-amp              Mixed precision. ~1.5-2x on Ampere+ for conv-heavy ResNet/MobileNet.
#   --channels-last        Memory format for CNNs. ~10-20% on Ampere+, no-op on MNIST.
#   --use-torch-compile    torch.compile wrap on client + server models. ~30-60s first-round
#                          compile cost; 20-40% sustained speedup thereafter. Over 100 rounds
#                          this pays back easily.
#   --performance-mode     cuDNN autotune (default on — not passed, just noted here).
PERF_FLAGS="--parallel-clients -1"
if [[ "$DEVICE" == "cuda" ]]; then
    PERF_FLAGS="${PERF_FLAGS} --use-amp --channels-last --use-torch-compile"
fi

# Base FD block — identical to v1 (paper-matched Mu et al. §VI), but with 100 rounds per caller.
BASE_FD="--paradigm fd \
         --local-epochs 2 \
         --public-dataset-size 2000 \
         --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 \
         --batch-size 128 --distillation-batch-size 500 \
         --distillation-lr 0.001 --distillation-epochs 2 --temperature 1.0 \
         --fd-optimizer adam \
         --n-bs-antennas 64 --quantization-bits 8 \
         --eval-every 10 \
         --profile \
         ${PERF_FLAGS} \
         --device ${DEVICE} ${FAST_FLAG}"

# ---- Method sets (v2) ----
# The one SOTA comparison set used by every experiment except the ablation.
# 4 FL SOTA baselines + CALM. LQTS, APEX, label-coverage, snr-diversity all dropped.
FD_SOTA="heuristic.random,system_aware.fedcs,system_aware.oort,system_aware.poc,fd_native.calm_fd"

# Ablation set: LQTS floor + CALM + 5 single-feature knockouts.
# LQTS is kept as the FD-native floor so we can see how much each CALM component adds over LQTS alone.
FD_ABLATION="fd_native.logit_quality_ts,fd_native.calm_fd,fd_native.calm_fd_no_confidence,fd_native.calm_fd_no_adaptive_var,fd_native.calm_fd_no_stale_guard,fd_native.calm_fd_no_channel_filter,fd_native.calm_fd_no_collusion"

# Round budget (advisor: 200 -> 100 everywhere).
ROUNDS=100

# ---- Counters & helpers ----
TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""
CUR_EXP_NUM=0

# Sub-run counts per experiment block (must match the bodies below).
exp_run_count() {
    case "$1" in
        1) echo 1 ;;   # Baseline headline
        2) echo 5 ;;   # Noise sweep: 5 SNR levels
        3) echo 3 ;;   # Alpha sweep: 3 values (low/mid/high-IID)
        4) echo 2 ;;   # FL-random vs FD-CALM communication comparison
        5) echo 1 ;;   # Cross-dataset MNIST
        6) echo 1 ;;   # CALM-FD ablation
        *) echo 0 ;;
    esac
}

log() { echo ""; echo "========== [$(date '+%H:%M:%S')] $1 =========="; echo ""; }

should_run() {
    local n="$1"
    [[ -z "$RUN_ONLY" ]] && return 0
    IFS=',' read -ra ARR <<< "$RUN_ONLY"
    for x in "${ARR[@]}"; do
        [[ "$x" == "$n" ]] && return 0
    done
    return 1
}

compute_total_planned() {
    local t=0
    for e in 1 2 3 4 5 6; do
        if should_run "$e"; then
            t=$(( t + $(exp_run_count "$e") ))
        fi
    done
    echo "$t"
}

fmt_hms() {
    local s="$1"
    printf "%dh %02dm %02ds" $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

progress_prefix() {
    local done_count=$(( PASSED + FAILED + SKIPPED ))
    local current=$(( done_count + 1 ))
    local pct=0
    if (( TOTAL_PLANNED > 0 )); then
        pct=$(( current * 100 / TOTAL_PLANNED ))
    fi
    local now=$(date +%s)
    local elapsed=$(( now - GLOBAL_START ))
    local eta_str=""
    if (( done_count > 0 && TOTAL_PLANNED > done_count )); then
        local per_run=$(( elapsed / done_count ))
        local remaining=$(( (TOTAL_PLANNED - done_count) * per_run ))
        eta_str=" | ETA~$(fmt_hms "$remaining")"
    fi
    printf "[%d/%d %d%% | %s | elapsed=%s%s]" \
        "$current" "$TOTAL_PLANNED" "$pct" "${CUR_EXP:-?}" "$(fmt_hms "$elapsed")" "$eta_str"
}

run_exists() {
    local name="$1"
    find artifacts/runs -maxdepth 1 -type d -name "${name}_*" 2>/dev/null | grep -q . && return 0 || return 1
}

run_one() {
    local name="$1"; shift
    TOTAL=$((TOTAL+1))
    local prefix; prefix="$(progress_prefix)"
    if $RESUME && run_exists "$name"; then
        echo "  ${prefix} [SKIP] ${name} (output exists)"
        SKIPPED=$((SKIPPED+1)); return 0
    fi
    if $DRY_RUN; then
        echo "  ${prefix} [DRY] python -m csfl_simulator compare --name ${name} $*"
        SKIPPED=$((SKIPPED+1)); return 0
    fi
    local t0=$(date +%s)
    echo "  ${prefix} [RUN] ${name}"
    if python -m csfl_simulator compare --name "${name}" "$@"; then
        local dt=$(( $(date +%s) - t0 ))
        echo "  ${prefix} [OK]   ${name} — $(fmt_hms "$dt")"; PASSED=$((PASSED+1))
    else
        local dt=$(( $(date +%s) - t0 ))
        echo "  ${prefix} [FAIL] ${name} — $(fmt_hms "$dt")"; FAILED=$((FAILED+1))
        FAILURES="${FAILURES}\n  - ${name}"
    fi
}

GLOBAL_START=$(date +%s)
TOTAL_PLANNED=$(compute_total_planned)

log "CALM-FD Trimmed Suite v2 (4 SOTA FL + CALM, 100 rounds, single seed)"
echo "  Device:     ${DEVICE}"
echo "  Mode:       $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:     ${RESUME}"
echo "  Dry-run:    ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:     ${RUN_ONLY}"
echo "  Planned:    ${TOTAL_PLANNED} compare invocations across selected experiments"
echo "  SOTA set:   ${FD_SOTA}"
echo "  Rounds:     ${ROUNDS}"

# =============================================================================
# EXP 1 — Baseline headline (N=30, K=10, alpha=0.5, DL=-20 dB)
# The paper-ready anchor comparing CALM against 4 SOTA FL baselines.
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1; CUR_EXP="E1/6-baseline"
    log "EXP 1/6: Baseline headline — N=30, K=10, alpha=0.5, DL=-20 dB"
    run_one "exp1_baseline_sota" \
        --methods "${FD_SOTA}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 2 — DL SNR sweep (5 levels)
# Does CALM's adaptive variance help more as noise increases?
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2
    log "EXP 2/6: Noise sweep — 5 DL SNR levels"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="E2/6-noise-${label}"
        log "  Noise sweep level: ${label}"
        run_one "exp2_noise_${label}" \
            --methods "${FD_SOTA}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
            ${flags} \
            --seed 42
    done
fi

# =============================================================================
# EXP 3 — Alpha (non-IID) sweep — 3 values
# Low (0.1) / Mid (0.5) / High-IID (5.0) per advisor.
# =============================================================================
if should_run 3; then
    CUR_EXP_NUM=3
    log "EXP 3/6: Alpha sweep — low / mid / high-IID"
    for alpha in 0.1 0.5 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="E3/6-alpha${alabel}"
        log "  alpha = ${alpha}"
        run_one "exp3_alpha_${alabel}" \
            --methods "${FD_SOTA}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha "${alpha}" \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done
fi

# =============================================================================
# EXP 4 — Paradigm comparison: FL+random vs FD+CALM (accuracy vs communication)
# Matched private data / partition / N / K / rounds; only paradigm + selector differ.
# Post-run, plot `accuracy` vs `cum_comm` (MB) across both to show FD's comm savings.
#   FL side: ResNet18 (homogeneous — FL can't do model heterogeneity), full weights.
#   FD side: ResNet18-FD pool (heterogeneous), logits only.
# Channel noise DISABLED on both sides here — the story is paradigm-level comm savings,
# so we want an apples-to-apples error-free channel. Channel-robustness is Exp 2's job.
# =============================================================================
if should_run 4; then
    CUR_EXP_NUM=4
    log "EXP 4/6: Paradigm comparison — FL(random) vs FD(CALM), accuracy-vs-comm (error-free channel)"

    CUR_EXP="E4/6-fl-random"
    log "  FL side: ResNet18, random selection, weight exchange"
    run_one "exp4_fl_random" \
        --methods "heuristic.random" \
        --paradigm fl \
        --local-epochs 2 \
        --batch-size 128 --lr 0.01 \
        --eval-every 10 --profile \
        ${PERF_FLAGS} --device ${DEVICE} ${FAST_FLAG} \
        --dataset CIFAR-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18 \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --seed 42

    CUR_EXP="E4/6-fd-calm"
    log "  FD side: heterogeneous pool, CALM selection, logit exchange"
    run_one "exp4_fd_calm" \
        --methods "fd_native.calm_fd" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --seed 42
fi

# =============================================================================
# EXP 5 — Cross-dataset (MNIST + FMNIST)
# =============================================================================
if should_run 5; then
    CUR_EXP_NUM=5; CUR_EXP="E5/6-mnist"
    log "EXP 5/6: Cross-dataset — MNIST (private) + FMNIST (public)"
    run_one "exp5_mnist" \
        --methods "${FD_SOTA}" \
        ${BASE_FD} \
        --dataset MNIST --public-dataset FMNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 6 — CALM-FD ablation
# Internal study — keeps LQTS floor + 5 CALM knockouts. Not a SOTA comparison.
# =============================================================================
if should_run 6; then
    CUR_EXP_NUM=6; CUR_EXP="E6/6-ablation"
    log "EXP 6/6: CALM-FD ablation — LQTS floor + 5 single-feature knockouts"
    run_one "exp6_calm_fd_ablation" \
        --methods "${FD_ABLATION}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# Summary
# =============================================================================
GLOBAL_END=$(date +%s)
DT=$(( GLOBAL_END - GLOBAL_START ))

log "TRIMMED SUITE v2 COMPLETE"
echo "  Planned:  ${TOTAL_PLANNED}"
echo "  Total:    ${TOTAL}"
echo "  Passed:   ${PASSED}"
echo "  Failed:   ${FAILED}"
echo "  Skipped:  ${SKIPPED}"
echo "  Wall:     $(fmt_hms "$DT")"

if [[ -n "$FAILURES" ]]; then
    echo ""
    echo "  Failed runs:"
    echo -e "${FAILURES}"
fi

echo ""
echo "  Results in: artifacts/runs/  (prefix: exp1_* ... exp6_*)"
echo "  Next:  python scripts/analyze_fd_results.py     # extend to pick up new runs"
echo "         python scripts/analyze_fd_mechanisms.py  # regenerate mechanism plots"
