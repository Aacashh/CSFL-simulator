#!/usr/bin/env bash
# =============================================================================
# CALM-FD Trimmed Suite v2.1 — thermal-safe + method-trimmed + salvage-aware.
#
# Changes from scripts/run_next_sota_experiments_v2.sh:
#
#   (A) Thermal / VRAM safety
#       - Drops --use-torch-compile:  dynamo was hitting recompile_limit=8 on
#                                     grad_mode flips (train/eval/autocast),
#                                     producing method 1 round times that
#                                     bounced 463s -> 25s -> 189s, eating VRAM
#                                     for kernel variants with no sustained win.
#       - --parallel-clients 2        (was -1 auto): bounded peak VRAM, flat
#                                     GPU power draw = no thermal throttle.
#       - Keeps --use-amp and         both strictly reduce VRAM and speed up
#           --channels-last:          conv-heavy ResNet/MobileNet. No downside.
#
#   (B) Method trimming (per advisor + Exp 1 evidence)
#       Exp 1 (CIFAR headline):       5 methods — FD_SOTA          (already done)
#       Exp 2 (noise sweep):          2 methods — FD_SWEEP         (trimmed)
#       Exp 3 (alpha sweep):          2 methods — FD_SWEEP         (trimmed)
#       Exp 4 (FL vs FD comm):        1 method per side            (unchanged)
#       Exp 5 (MNIST headline):       5 methods — FD_SOTA          (unchanged)
#       Exp 6 (ablation):             7 methods — FD_ABLATION      (unchanged)
#
#       Rationale from Exp 1: FedCS / Oort / PoC all landed at ~0.19 accuracy
#       with Gini=0.67 (massive participation concentration). One phenomenon,
#       not three — FL-designed SA methods over-concentrate under FD. The
#       headline table (Exp 1 + Exp 5) already establishes this. In sweeps the
#       question being asked is narrower: "under what conditions does CALM
#       differentiate from its actual competitor?" — and Exp 1 showed that
#       competitor is RANDOM (0.307), not Oort (worst SA at 0.188). Adding any
#       SA method to sweep plots would draw a flat ~0.19 line that adds no
#       information and wastes ~5 hr of compute.
#
#   (C) Salvage-aware resume
#       run_exists() now checks for a completion marker file
#       (compare_results.json / results.json) rather than just the directory.
#       This means:
#         - Fully-done runs (Exp 1, any completed Exp 2 sub-runs) are skipped.
#         - A run killed mid-training has a directory but no results file ->
#           we correctly re-run it instead of silently skipping.
#
# Usage (recommended after killing in-progress Exp 2):
#   bash scripts/run_next_sota_experiments_v2_1.sh --resume
#
# All other flags (--exp, --dry-run, --fast, --cpu, --device) behave identically
# to v2.
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

# Thermal-safe speedup stack:
#   --parallel-clients 2   Fixed 2-way CUDA-stream parallelism. Explicit cap so
#                          peak VRAM is predictable; auto (-1) was producing
#                          unbounded concurrency on an already-saturated card.
#   --use-amp              Mixed precision: ~halves activation VRAM, speeds up
#                          conv workloads. Default-on, listed for visibility.
#   --channels-last        channels_last memory format for CNNs. Free speedup
#                          on Ampere+, no-op on MNIST single-channel inputs.
#
# torch.compile is intentionally OMITTED here — it was net-negative on this GPU
# (recompile thrash on grad_mode changes + VRAM cost for specialised variants).
PERF_FLAGS="--parallel-clients 2"
if [[ "$DEVICE" == "cuda" ]]; then
    PERF_FLAGS="${PERF_FLAGS} --use-amp --channels-last"
fi

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

# ---- Method sets (v2.1) ----
# Headline table — full SOTA field. Used by Exp 1 and Exp 5 only.
FD_SOTA="heuristic.random,system_aware.fedcs,system_aware.oort,system_aware.poc,fd_native.calm_fd"

# Sweep table — 2 methods. Random is the real adversary per Exp 1 (0.307 vs
# CALM 0.304); the SA methods pooled at ~0.19 in the headline, so adding one
# to sweeps adds a flat line. Used by Exp 2 (noise) and Exp 3 (alpha).
FD_SWEEP="heuristic.random,fd_native.calm_fd"

# Ablation — LQTS floor + 5 CALM knockouts. Used by Exp 6 only.
FD_ABLATION="fd_native.logit_quality_ts,fd_native.calm_fd,fd_native.calm_fd_no_confidence,fd_native.calm_fd_no_adaptive_var,fd_native.calm_fd_no_stale_guard,fd_native.calm_fd_no_channel_filter,fd_native.calm_fd_no_collusion"

ROUNDS=100

# ---- Counters & helpers ----
TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""
CUR_EXP_NUM=0

exp_run_count() {
    case "$1" in
        1) echo 1 ;;   # Baseline headline
        2) echo 5 ;;   # Noise sweep
        3) echo 3 ;;   # Alpha sweep
        4) echo 2 ;;   # FL vs FD comm comparison
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

# Salvage-aware: a run counts as "done" only if a completion marker exists.
# This protects us from mid-training kills, which leave a partially-populated
# directory on disk. Without this check --resume would silently skip the victim.
run_exists() {
    local name="$1"
    local d
    shopt -s nullglob
    local dirs=( artifacts/runs/${name}_*/ )
    shopt -u nullglob
    for d in "${dirs[@]}"; do
        [[ -d "$d" ]] || continue
        if [[ -f "${d}compare_results.json" ]] || [[ -f "${d}results.json" ]]; then
            return 0
        fi
    done
    return 1
}

run_one() {
    local name="$1"; shift
    TOTAL=$((TOTAL+1))
    local prefix; prefix="$(progress_prefix)"
    if $RESUME && run_exists "$name"; then
        echo "  ${prefix} [SKIP] ${name} (completion marker found)"
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

log "CALM-FD Trimmed Suite v2.1 (thermal-safe, no torch.compile, sweeps=3 methods)"
echo "  Device:        ${DEVICE}"
echo "  Mode:          $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:        ${RESUME}   (uses completion-marker check, not dir existence)"
echo "  Dry-run:       ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:        ${RUN_ONLY}"
echo "  Planned:       ${TOTAL_PLANNED} compare invocations across selected experiments"
echo "  Headline set:  ${FD_SOTA}"
echo "  Sweep set:     ${FD_SWEEP}"
echo "  Rounds:        ${ROUNDS}"

# =============================================================================
# EXP 1 — Baseline headline (5 methods)
# Likely already DONE from the v2 run. --resume will skip via compare_results.json.
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1; CUR_EXP="E1/6-baseline"
    log "EXP 1/6: Baseline headline — N=30, K=10, alpha=0.5, DL=-20 dB (5 methods)"
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
# EXP 2 — DL SNR sweep (5 levels, 3 methods)
# Trimmed: random / oort / calm_fd. Any already-complete exp2_noise_* sub-runs
# (with 5 methods) are salvaged via --resume — the 3 trimmed methods are a
# subset of the 5-method runs, so plots can use the intersection cleanly.
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2
    log "EXP 2/6: Noise sweep — 5 DL SNR levels (2-method sweep set: random vs CALM)"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="E2/6-noise-${label}"
        log "  Noise sweep level: ${label}"
        run_one "exp2_noise_${label}" \
            --methods "${FD_SWEEP}" \
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
# EXP 3 — Alpha (non-IID) sweep — 3 values (low / mid / high-IID), 3 methods
# =============================================================================
if should_run 3; then
    CUR_EXP_NUM=3
    log "EXP 3/6: Alpha sweep — low / mid / high-IID (2-method sweep set: random vs CALM)"
    for alpha in 0.1 0.5 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="E3/6-alpha${alabel}"
        log "  alpha = ${alpha}"
        run_one "exp3_alpha_${alabel}" \
            --methods "${FD_SWEEP}" \
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
# Matched private data / partition / N / K / rounds; only paradigm + selector
# differ. Channel noise DISABLED on both sides for apples-to-apples comm story.
# =============================================================================
if should_run 4; then
    CUR_EXP_NUM=4
    log "EXP 4/6: Paradigm comparison — FL(random) vs FD(CALM), accuracy-vs-comm (error-free)"

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
# EXP 5 — Cross-dataset headline: MNIST + FMNIST (5 methods — second headline)
# =============================================================================
if should_run 5; then
    CUR_EXP_NUM=5; CUR_EXP="E5/6-mnist"
    log "EXP 5/6: Cross-dataset headline — MNIST (private) + FMNIST (public), 5 methods"
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
# EXP 6 — CALM-FD ablation (LQTS floor + 5 single-feature knockouts)
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

log "TRIMMED SUITE v2.1 COMPLETE"
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
