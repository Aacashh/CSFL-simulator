#!/usr/bin/env bash
# =============================================================================
# PRISM-FD Salvage Experiment Suite (CALM dropped).
#
# WHAT HAPPENED BEFORE:
#   The v2.1 sweep showed CALM-FD cannot meaningfully beat uniform random
#   under FD — across α=0.5/DL=-20 (headline), DL=-30 (extreme noise) and
#   α=0.1 (extreme non-IID) the two methods tie within noise.
#
#   Forensic reading of the FD pipeline explained why: CALM optimises per-
#   client Thompson rewards, but FD aggregates logits with a data-size-weighted
#   mean and distills the server on that mean. In FD the selected SET'S
#   collective label coverage matters more than any per-client quality score.
#   Random accidentally achieves Gini 0.08 (near-uniform). CALM is dropped
#   from this suite entirely — it provides no additional paper signal.
#
# WHAT THIS SCRIPT DOES:
#   Validates PRISM-FD — a new proposed method — against the random baseline
#   across the same configs v2.1 covered, PLUS the MNIST config v2.1 never
#   reached (exp5 was in v2.1's plan but never executed).
#
#   PRISM-FD exploits the one signal random literally cannot see: the server's
#   own per-class confidence on the public dataset (exposed via a 3-line
#   simulator hook). It greedily picks the K clients whose union of label
#   histograms best covers the classes the server is currently weakest at —
#   a submodular set objective with (1 - 1/e) greedy guarantee.
#
# WHAT v2.1 COMPLETED (random data is already on disk):
#   - CIFAR α=0.5/DL=-20:                 exp1_baseline_sota (5 methods)
#   - CIFAR noise sweep, 5 SNR levels:    exp2_noise_{errfree,dl0,dl-10,dl-20,dl-30}
#   - CIFAR alpha sweep, 3 levels:        exp3_alpha_{0_1,0_5,5_0}
#   - FL random errfree (for comm plot):  exp4_fl_random
#   - FD random errfree (for comm plot):  exp2_noise_errfree (free side-effect)
#
# WHAT v2.1 NEVER REACHED (random data is MISSING and we re-run it):
#   - MNIST FD run (exp5_mnist) — killed before v2.1 got there.
#
# SALVAGE STRATEGY:
#   Existing exp1_*/exp2_*/exp3_*/exp4_* directories are preserved. New runs
#   use the "prism_*" prefix. Random is deterministic under the fixed seed,
#   so for the CIFAR sweeps we only run PRISM alone — the random curves come
#   from the already-completed exp2_*/exp3_* dirs. The two places we re-run
#   random alongside PRISM are (a) the headline table (one clean compare for
#   the paper's Table I) and (b) the MNIST run (because no MNIST random exists
#   yet — this backfills the missing v2.1 exp5 row).
#
# USAGE:
#   bash scripts/run_prism_experiments.sh --resume
#   bash scripts/run_prism_experiments.sh --exp 1       # headline only
#   bash scripts/run_prism_experiments.sh --dry-run
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


# Thermal-safe perf stack (no torch.compile, bounded parallelism).
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
# Headline / MNIST: 2-way compare so the paper tables are self-contained.
PAIR_SET="heuristic.random,fd_native.prism_fd"
# CIFAR sweep runs: PRISM alone. Random curves come from existing exp2_*/exp3_*.
PRISM_ONLY="fd_native.prism_fd"
# Ablation: PRISM vs (no-server-signal variant) vs (sharp uncertainty).
PRISM_ABLATION="fd_native.prism_fd,fd_native.prism_fd_no_server,fd_native.prism_fd_sharp"

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""
CUR_EXP_NUM=0

exp_run_count() {
    case "$1" in
        1) echo 1 ;;   # CIFAR headline (α=0.5, DL=-20, random + PRISM)
        2) echo 5 ;;   # Noise sweep (PRISM alone, 5 levels)
        3) echo 3 ;;   # Alpha sweep (PRISM alone, 3 levels)
        4) echo 1 ;;   # MNIST headline (random + PRISM) — backfills missing v2.1 exp5
        5) echo 1 ;;   # PRISM ablation
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
    for e in 1 2 3 4 5; do
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

# Salvage-aware: only skip if a completion marker exists (not just directory).
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

log "PRISM-FD Salvage Suite"
echo "  Device:        ${DEVICE}"
echo "  Mode:          $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:        ${RESUME}"
echo "  Dry-run:       ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:        ${RUN_ONLY}"
echo "  Planned:       ${TOTAL_PLANNED} compare invocations"
echo "  Headline/MNIST: ${PAIR_SET}"
echo "  CIFAR sweeps:  ${PRISM_ONLY} (random from existing exp2_*/exp3_*)"
echo "  Ablation:      ${PRISM_ABLATION}"
echo "  Rounds:        ${ROUNDS}"

# =============================================================================
# EXP 1 — CIFAR headline: random vs PRISM (α=0.5, DL=-20 dB)
# This is the paper's main Table I. Single clean compare so random and PRISM
# share the exact same partition/init in one invocation. CALM dropped.
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1; CUR_EXP="P1/5-headline"
    log "EXP 1/5: CIFAR headline — N=30, K=10, α=0.5, DL=-20 dB (random vs PRISM)"
    run_one "prism_headline" \
        --methods "${PAIR_SET}" \
        ${BASE_FD} \
        --dataset CIFAR-10 --public-dataset STL-10 \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model ResNet18-FD --model-heterogeneous --model-pool "${CIFAR_MODELS}" \
        --total-clients 30 --clients-per-round 10 --rounds ${ROUNDS} \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 2 — Noise sweep: PRISM alone across 5 DL SNR levels.
# Pair with existing exp2_noise_{errfree,dl0,dl-10,dl-20,dl-30} random curves
# at plot time (same seed + partition => random is deterministic).
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2
    log "EXP 2/5: Noise sweep — 5 DL SNR levels (PRISM alone; pair with existing random)"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="P2/5-noise-${label}"
        log "  Noise sweep level: ${label}"
        run_one "prism_noise_${label}" \
            --methods "${PRISM_ONLY}" \
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
# EXP 3 — Alpha sweep: PRISM alone across 3 non-IID levels.
# Pair with existing exp3_alpha_{0_1,0_5,5_0} random curves.
# =============================================================================
if should_run 3; then
    CUR_EXP_NUM=3
    log "EXP 3/5: Alpha sweep — low / mid / high-IID (PRISM alone; pair with existing random)"
    for alpha in 0.1 0.5 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="P3/5-alpha${alabel}"
        log "  alpha = ${alpha}"
        run_one "prism_alpha_${alabel}" \
            --methods "${PRISM_ONLY}" \
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
# EXP 4 — MNIST headline: random vs PRISM. BACKFILLS THE MISSING v2.1 EXP 5.
# v2.1 was planned to run MNIST but was killed before reaching it, so no MNIST
# random baseline exists yet — both methods run together here.
# Uses the smaller FD-CNN1/2/3 heterogeneous pool; rounds are much faster than
# the CIFAR experiments.
# =============================================================================
if should_run 4; then
    CUR_EXP_NUM=4; CUR_EXP="P4/5-mnist"
    log "EXP 4/5: MNIST headline — random vs PRISM (backfills v2.1's missing exp5)"
    run_one "prism_mnist" \
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
# EXP 5 — PRISM ablation (at CIFAR headline config).
# Disables (a) the server-uncertainty signal entirely, and (b) replaces linear
# uncertainty with squared, to show which component of PRISM carries the gain.
# =============================================================================
if should_run 5; then
    CUR_EXP_NUM=5; CUR_EXP="P5/5-ablation"
    log "EXP 5/5: PRISM ablation — server-signal-off, sharp-γ=2, full"
    run_one "prism_ablation" \
        --methods "${PRISM_ABLATION}" \
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

log "PRISM SUITE COMPLETE"
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
echo "  Results in: artifacts/runs/  (prefix: prism_*)"
echo "  Pairing:    random/CALM baselines are in exp1_*, exp2_*, exp3_* from earlier suites."
