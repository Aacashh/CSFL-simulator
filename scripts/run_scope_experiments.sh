#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD Experiment Suite.
#
# WHY THIS EXISTS (THE WHOLE STORY)
# ---------------------------------
#   Timeline of FD selection attempts in this project:
#     - CALM-FD       → tied random (0.304 vs 0.307 on the headline config).
#     - PRISM-FD      → LOST to random (0.271 vs 0.308). Post-mortem: greedy
#                       coverage on static label histograms deterministically
#                       locks onto the same ~15 clients, driving Gini from
#                       random's 0.083 to 0.536 and collapsing data diversity.
#     - SCOPE-FD      → new design: participation-debt is the PRIMARY ranking
#                       (guarantees Gini → 0 asymptotically, strictly better
#                       than random's 0.083). Server uncertainty and per-round
#                       diversity layered ON TOP at weights 0.3 and 0.1 so
#                       they can only nudge, never override balance.
#
# WHAT THIS SCRIPT DOES
# ---------------------
#   Validates SCOPE-FD against random across the same configs the earlier
#   v2.1 run covered, plus the MNIST config v2.1 never reached. Existing
#   random curves on disk are re-used; only the SCOPE curves need to be
#   generated for the CIFAR sweeps. Headline and MNIST are run as paired
#   compares (random + SCOPE) for self-contained paper tables.
#
# RANDOM DATA ALREADY ON DISK (from v2.0/v2.1 — not re-run)
#   - exp1_baseline_sota              (α=0.5, DL=-20, CIFAR)
#   - exp2_noise_{errfree,dl0,-10,-20,-30}   (5 noise levels, CIFAR)
#   - exp3_alpha_{0_1,0_5,5_0}        (3 α levels, CIFAR)
#   - exp4_fl_random                  (FL paradigm, for comm comparison)
#
#   ALL CIFAR random data is already on disk. We never re-run random on CIFAR
#   — scope_headline's random baseline comes from exp2_noise_dl-20 (identical
#   config, deterministic under seed 42).
#
# MISSING RANDOM DATA (this suite backfills — pairs both methods in one run)
#   - MNIST (private) + FMNIST (public): v2.1 killed before reaching its exp5.
#   - FMNIST (private) + MNIST (public): never in any prior suite — a second
#     small-dataset evaluation point for the paper.
#
# USAGE
#   bash scripts/run_scope_experiments.sh --resume
#   bash scripts/run_scope_experiments.sh --exp 1       # headline only
#   bash scripts/run_scope_experiments.sh --dry-run
# =============================================================================
set -euo pipefail

FAST_FLAG="--no-fast-mode"
RUN_ONLY=""
RESUME=false
DRY_RUN=false
DEVICE="cuda"

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

CIFAR_MODELS="ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

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

ROUNDS=100

# ---- Method sets ----
PAIR_SET="heuristic.random,fd_native.scope_fd"
SCOPE_ONLY="fd_native.scope_fd"
SCOPE_ABLATION="fd_native.scope_fd,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""
CUR_EXP_NUM=0

exp_run_count() {
    case "$1" in
        1) echo 1 ;;   # CIFAR headline (SCOPE alone; random from exp2_noise_dl-20)
        2) echo 5 ;;   # Noise sweep (SCOPE alone)
        3) echo 3 ;;   # Alpha sweep (SCOPE alone)
        4) echo 1 ;;   # MNIST headline (pair: random + SCOPE)
        5) echo 1 ;;   # FMNIST headline (pair: random + SCOPE)
        6) echo 1 ;;   # SCOPE ablation
        7) echo 7 ;;   # K-sweep at N=50 on Fashion-MNIST (7 K values)
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
    for e in 1 2 3 4 5 6 7; do
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

log "SCOPE-FD Suite (debt-balanced + server-aware + diversity)"
echo "  Device:         ${DEVICE}"
echo "  Mode:           $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:         ${RESUME}"
echo "  Dry-run:        ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:         ${RUN_ONLY}"
echo "  Planned:        ${TOTAL_PLANNED} compare invocations"
echo "  Headline/MNIST: ${PAIR_SET}"
echo "  CIFAR sweeps:   ${SCOPE_ONLY} (pair with existing random)"
echo "  Ablation:       ${SCOPE_ABLATION}"
echo "  Rounds:         ${ROUNDS}"

# =============================================================================
# EXP 1 — CIFAR headline: SCOPE alone (α=0.5, DL=-20 dB).
# Random baseline already on disk (exp2_noise_dl-20, seed 42 deterministic).
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1; CUR_EXP="S1/7-cifar"
    log "EXP 1/7: CIFAR headline — SCOPE alone (random from exp2_noise_dl-20)"
    run_one "scope_cifar" \
        --methods "${SCOPE_ONLY}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 2 — CIFAR noise sweep: SCOPE alone across 5 DL SNR levels.
# Random curves are already in exp2_noise_* on disk (same seed, deterministic).
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2
    log "EXP 2/7: Noise sweep — 5 DL SNR levels (SCOPE alone; pair with existing random)"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="S2/7-noise-${label}"
        log "  Noise sweep level: ${label}"
        run_one "scope_noise_${label}" \
            --methods "${SCOPE_ONLY}" \
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
# EXP 3 — CIFAR alpha sweep: SCOPE alone across 3 non-IID levels.
# Random curves already in exp3_alpha_* on disk.
# =============================================================================
if should_run 3; then
    CUR_EXP_NUM=3
    log "EXP 3/7: Alpha sweep — low / mid / high-IID (SCOPE alone; pair with existing random)"
    for alpha in 0.1 0.5 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="S3/7-alpha${alabel}"
        log "  alpha = ${alpha}"
        run_one "scope_alpha_${alabel}" \
            --methods "${SCOPE_ONLY}" \
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
# EXP 4 — MNIST(private) + FMNIST(public) headline: random vs SCOPE.
# Backfills the missing v2.1 exp5. No MNIST random baseline exists yet, so
# both methods run together.
# =============================================================================
if should_run 4; then
    CUR_EXP_NUM=4; CUR_EXP="S4/7-mnist"
    log "EXP 4/7: MNIST(private) + FMNIST(public) — random vs SCOPE"
    run_one "scope_mnist" \
        --methods "${PAIR_SET}" \
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
# EXP 5 — FMNIST(private) + MNIST(public) headline: random vs SCOPE.
# Second small-dataset evaluation. Swaps the private/public roles from Exp 4
# to show SCOPE is not overfit to one dataset pairing.
# =============================================================================
if should_run 5; then
    CUR_EXP_NUM=5; CUR_EXP="S5/7-fmnist"
    log "EXP 5/7: Fashion-MNIST(private) + MNIST(public) — random vs SCOPE"
    run_one "scope_fmnist" \
        --methods "${PAIR_SET}" \
        ${BASE_FD} \
        --dataset Fashion-MNIST --public-dataset MNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 6 — SCOPE ablation (at CIFAR headline config).
# Shows that disabling the server-signal OR the diversity penalty each cost
# something — debt alone is not sufficient.
# =============================================================================
if should_run 6; then
    CUR_EXP_NUM=6; CUR_EXP="S6/7-ablation"
    log "EXP 6/7: SCOPE ablation — full / no-server / no-diversity"
    run_one "scope_ablation" \
        --methods "${SCOPE_ABLATION}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 7 — K-sweep at N=50 on Fashion-MNIST (private) + MNIST (public).
# Probes how SCOPE's edge behaves across participation ratios from 2% (K=1)
# through 100% (K=N=50). Each run is a paired random + SCOPE compare — no
# N=50 baselines exist on disk, so both methods run together here.
# K=1 is the ultra-sparse stress test; K=50 is the degenerate full-participation
# case (selection is trivially identical for both methods — sanity reference).
# =============================================================================
if should_run 7; then
    CUR_EXP_NUM=7
    log "EXP 7/7: K-sweep — N=50 on Fashion-MNIST (7 K values: 1,5,10,15,25,35,50)"
    for K in 1 5 10 15 25 35 50; do
        CUR_EXP="S7/7-N50-K${K}"
        log "  K = ${K} ($((K*100/50))%)"
        run_one "scope_fmnist_N50_K${K}" \
            --methods "${PAIR_SET}" \
            ${BASE_FD} \
            --dataset Fashion-MNIST --public-dataset MNIST \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
            --total-clients 50 --clients-per-round ${K} --rounds ${ROUNDS} \
            --batch-size 20 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed 42
    done
fi

# =============================================================================
# Summary
# =============================================================================
GLOBAL_END=$(date +%s)
DT=$(( GLOBAL_END - GLOBAL_START ))

log "SCOPE SUITE COMPLETE"
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
echo "  Results in: artifacts/runs/  (prefix: scope_*)"
echo "  Pairing:    random baselines from exp2_*/exp3_* (CIFAR sweeps)."
