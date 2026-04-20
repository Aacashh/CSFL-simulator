#!/usr/bin/env bash
# =============================================================================
# Next-SOTA FD Experiment Suite — validates CALM-FD against LQTS + baselines
#
# Key differences from scripts/run_all_experiments.sh:
#   - CIFAR-10 FD uses ResNet18/MobileNetV2/ShuffleNetV2 (paper Table IV CIFAR pool),
#     MNIST FD uses FD-CNN1/2/3 (paper Table IV MNIST pool)
#   - 2 local epochs (was 1) — matches Mu et al.
#   - New-baseline N=30, K=10 (was N=50, K=15) — brings data/client up to 1667
#   - Paper replication experiment runs FIRST as anchor
#   - CALM-FD + 5 ablation variants are tested
#
# Usage:
#   bash scripts/run_next_sota_experiments.sh                    # Run everything
#   bash scripts/run_next_sota_experiments.sh --fast             # Fast debug mode
#   bash scripts/run_next_sota_experiments.sh --exp 1            # Single experiment
#   bash scripts/run_next_sota_experiments.sh --exp "1,2,3"      # Multiple
#   bash scripts/run_next_sota_experiments.sh --resume           # Skip existing
#   bash scripts/run_next_sota_experiments.sh --dry-run          # Print plan only
#   bash scripts/run_next_sota_experiments.sh --cpu              # Force CPU
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
# Paper-matched heterogeneity pools (Mu et al. Table IV):
#   - CIFAR-10 + STL-10  → ResNet18 / MobileNetV2 / ShuffleNetV2 (the deep pool)
#   - MNIST   + FMNIST   → CNN_1 / CNN_2 / CNN_3 (the Table III small CNN pool)
# Using the wrong pool (e.g. CNN_1/2/3 on CIFAR) under-parameterises the task by ~10x
# and costs several pp of final server accuracy — the paper's Table IV heterogeneity
# results are explicitly reported for these pool-dataset combinations.
CIFAR_MODELS="ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

# Speed flags: only engaged on CUDA.
#   --use-amp            Mixed precision (GradScaler breaks on CPU, hence the gate).
#   --use-torch-compile  Wraps each client + the server model with torch.compile. Adds
#                        ~30-60 s of first-round compile time, then yields 20-40 %
#                        sustained speedup across every subsequent round.
#   --channels-last      channels_last memory format. 10-20 % faster on Ampere+ GPUs
#                        for conv-heavy CIFAR workloads (ResNet18 / MobileNetV2 /
#                        ShuffleNetV2).  No-op on MNIST single-channel inputs.
# profile is always on so the first few rounds' wall-clock is visible live; eval-every 10
# halves eval cost vs. the default 5 without hurting final-accuracy reports (final round
# is always evaluated).
PERF_FLAGS=""
if [[ "$DEVICE" == "cuda" ]]; then
    PERF_FLAGS="--use-amp --channels-last"
fi

# Base FD block — paper-matched (Mu et al. §VI): 2 local epochs, 2 distill epochs, Adam lr 0.001,
# 128 train batch / 500 distill batch, dynamic steps base=5 period=25, 8-bit quantisation, 64 BS antennas.
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

# Method sets
FD_FULL="heuristic.random,system_aware.fedcs,system_aware.oort,heuristic.label_coverage,ml.apex_v2,fd_native.logit_quality_ts,fd_native.snr_diversity,fd_native.calm_fd"
FD_CORE="heuristic.random,ml.apex_v2,fd_native.logit_quality_ts,fd_native.snr_diversity,fd_native.calm_fd"
FD_TRIO="heuristic.random,fd_native.logit_quality_ts,fd_native.calm_fd"
FD_ABLATION="fd_native.logit_quality_ts,fd_native.calm_fd,fd_native.calm_fd_no_confidence,fd_native.calm_fd_no_adaptive_var,fd_native.calm_fd_no_stale_guard,fd_native.calm_fd_no_channel_filter,fd_native.calm_fd_no_collusion"

# For Exp 1 paper replication we strictly use only what the paper tested, plus calm_fd as our new proposal
FD_PAPER_REPLICATION="heuristic.random,fd_native.logit_quality_ts,fd_native.calm_fd"

# ---- Counters & helpers ----
TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""      # Short tag for the current experiment, e.g. "E3[dl-20]"
CUR_EXP_NUM=0   # Which experiment block (1..9) we are currently inside

# Sub-run counts per experiment block (must stay in sync with the bodies below).
# Used to compute TOTAL_PLANNED up-front so every log line can show "i/N (pp%)".
exp_run_count() {
    case "$1" in
        1) echo 1 ;;   # Paper replication: 1 compare
        2) echo 1 ;;   # New-baseline headline: 1 compare
        3) echo 5 ;;   # Noise sweep: 5 SNR levels
        4) echo 4 ;;   # Alpha sweep: 4 alphas
        5) echo 5 ;;   # K sweep: 5 K values
        6) echo 3 ;;   # N scaling: 3 N/K pairs
        7) echo 1 ;;   # Cross-dataset MNIST: 1 compare
        8) echo 1 ;;   # CALM-FD ablation: 1 compare
        9) echo 3 ;;   # Multi-seed: 3 seeds
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

# Compute how many compare invocations will run given the current --exp filter.
# Called once at script start. Negligible cost.
compute_total_planned() {
    local t=0
    for e in 1 2 3 4 5 6 7 8 9; do
        if should_run "$e"; then
            t=$(( t + $(exp_run_count "$e") ))
        fi
    done
    echo "$t"
}

# Pretty-print seconds as "Xh Ym Zs".
fmt_hms() {
    local s="$1"
    printf "%dh %02dm %02ds" $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# Progress prefix: "[i/N ppp% | tag | elapsed=Xh Ym | ETA~Xh Ym]".
# Uses only arithmetic + date +%s — no external procs, no overhead per round.
progress_prefix() {
    local done_count=$(( PASSED + FAILED + SKIPPED ))
    local current=$(( done_count + 1 ))   # 1-based index for the run about to start
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

log "CALM-FD Next-SOTA Suite (paper-matched pools: CIFAR→ResNet/MobileNet/ShuffleNet, MNIST→CNN_1/2/3, 2 local epochs)"
echo "  Device:     ${DEVICE}"
echo "  Mode:       $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:     ${RESUME}"
echo "  Dry-run:    ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:     ${RUN_ONLY}"
echo "  Planned:    ${TOTAL_PLANNED} compare invocations across selected experiments"

# =============================================================================
# EXP 1 — Paper replication
# Goal: Reproduce Mu et al.'s Table IV number (~50% overall at alpha=0.5, DL=-20dB, full participation)
# Config exact to paper: N=K=15 full participation, CNN_1/2/3, alpha=0.5, 200 rounds.
# This is our sanity anchor — if we don't hit ~50%, something's broken in our setup.
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1; CUR_EXP="E1/9-paper-repl"
    log "EXP 1/9: Paper replication — N=K=15, full participation"
    run_one "exp1_paper_replication" \
        --methods "${FD_PAPER_REPLICATION}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 15 --clients-per-round 15 --rounds 200 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 2 — New baseline headline: N=30, K=10
# Goal: The new paper-ready benchmark. Compare CALM-FD against LQTS + 6 other methods.
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2; CUR_EXP="E2/9-newbaseline"
    log "EXP 2/9: New-baseline headline — N=30, K=10, alpha=0.5, DL=-20 dB"
    run_one "exp2_fd_main_newbaseline" \
        --methods "${FD_FULL}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds 200 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 3 — DL SNR sweep (5 levels)
# Goal: Does CALM-FD's adaptive variance help more as noise increases?
# =============================================================================
if should_run 3; then
    CUR_EXP_NUM=3
    log "EXP 3/9: Noise sweep — 5 DL SNR levels"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="E3/9-noise-${label}"
        log "  Noise sweep level: ${label}"
        run_one "exp3_noise_${label}" \
            --methods "${FD_CORE}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round 10 --rounds 200 \
            ${flags} \
            --seed 42
    done
fi

# =============================================================================
# EXP 4 — alpha sweep (non-IID sensitivity)
# Goal: Does CALM-FD still win at extreme non-IID (alpha=0.1) and near-IID (alpha=5)?
# =============================================================================
if should_run 4; then
    CUR_EXP_NUM=4
    log "EXP 4/9: Alpha (non-IID) sweep — 4 levels"
    for alpha in 0.1 0.5 1.0 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="E4/9-alpha${alabel}"
        log "  alpha = ${alpha}"
        run_one "exp4_alpha_${alabel}" \
            --methods "heuristic.random,heuristic.label_coverage,fd_native.logit_quality_ts,fd_native.calm_fd" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha "${alpha}" \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round 10 --rounds 200 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done
fi

# =============================================================================
# EXP 5 — K sweep (selection ratio)
# Goal: Does CALM-FD's advantage over LQTS persist as K/N -> 1 (full participation)?
# =============================================================================
if should_run 5; then
    CUR_EXP_NUM=5
    log "EXP 5/9: K sweep — 5 participation levels (N=30)"
    for K in 3 6 10 15 30; do
        CUR_EXP="E5/9-K${K}"
        log "  K = ${K} (K/N = $((K * 100 / 30))%)"
        run_one "exp5_K${K}" \
            --methods "${FD_TRIO}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round ${K} --rounds 200 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done
fi

# =============================================================================
# EXP 6 — N scaling (fixed ~33% participation)
# Goal: Scalability — does CALM-FD hold as N grows? No N > 100 (compute budget).
# =============================================================================
if should_run 6; then
    CUR_EXP_NUM=6
    log "EXP 6/9: N scaling — 30, 50, 100 at ~33% participation"
    for pair in "30:10" "50:16" "100:33"; do
        N="${pair%%:*}"
        K="${pair#*:}"
        CUR_EXP="E6/9-N${N}K${K}"
        log "  N=${N}, K=${K}"
        run_one "exp6_N${N}_K${K}" \
            --methods "${FD_TRIO}" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients ${N} --clients-per-round ${K} --rounds 200 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done
fi

# =============================================================================
# EXP 7 — Cross-dataset (MNIST + FMNIST)
# Goal: Does CALM-FD generalise beyond CIFAR-10? Mu et al. tested both.
# =============================================================================
if should_run 7; then
    CUR_EXP_NUM=7; CUR_EXP="E7/9-mnist"
    log "EXP 7/9: Cross-dataset — MNIST (private) + FMNIST (public)"
    run_one "exp7_mnist_main" \
        --methods "${FD_FULL}" \
        ${BASE_FD} \
        --dataset MNIST --public-dataset FMNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds 150 \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 8 — CALM-FD ablation
# Goal: Which of the 5 CALM-FD enhancements is load-bearing?
# =============================================================================
if should_run 8; then
    CUR_EXP_NUM=8; CUR_EXP="E8/9-ablation"
    log "EXP 8/9: CALM-FD ablation — 5 single-feature knockouts + LQTS baseline"
    run_one "exp8_calm_fd_ablation" \
        --methods "${FD_ABLATION}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds 200 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 9 — Multi-seed reproducibility
# Goal: Smallest possible multi-seed run (3 seeds, 2 methods) to quote mean +/- std
#       for the headline claim. Everything else is single-seed.
# =============================================================================
if should_run 9; then
    CUR_EXP_NUM=9
    log "EXP 9/9: Multi-seed reproducibility — LQTS vs CALM-FD, 3 seeds"
    for s in 42 123 7; do
        CUR_EXP="E9/9-seed${s}"
        log "  Seed ${s}"
        run_one "exp9_seed${s}" \
            --methods "fd_native.logit_quality_ts,fd_native.calm_fd" \
            ${BASE_FD} \
            --dataset CIFAR-10 --public-dataset STL-10 \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
            --total-clients 30 --clients-per-round 10 --rounds 200 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed ${s}
    done
fi

# =============================================================================
# Summary
# =============================================================================
GLOBAL_END=$(date +%s)
DT=$(( GLOBAL_END - GLOBAL_START ))

log "NEXT-SOTA SUITE COMPLETE"
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
echo "  Results in: artifacts/runs/  (prefix: exp1_* ... exp9_*)"
echo "  Next:  python scripts/analyze_fd_results.py     # extend to pick up new runs"
echo "         python scripts/analyze_fd_mechanisms.py  # regenerate mechanism plots"
