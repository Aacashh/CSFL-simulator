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
# =============================================================================
# NEW EXPERIMENTS (EXP 9–12) — ADDED IN RESPONSE TO REVIEWER-STYLE CRITIQUE
# =============================================================================
#
# Why these were added:
#   The original EXP 6 ablation runs at CIFAR, but the paper's Table II and
#   Table III are on FMNIST — so the existing ablation does NOT populate any
#   row in any paper table. The new EXP 9–12 run ablations at the SAME FMNIST
#   configs the paper actually reports, plus multi-seed coverage of the K=1
#   stress test.
#
#   The original ablation set had three variants:
#     - scope_fd            (full: αu=0.3, αd=0.1)
#     - scope_fd_no_server  (αu=0,   αd=0.1 — debt + diversity)
#     - scope_fd_no_diversity (αu=0.3, αd=0   — debt + uncertainty)
#   It was MISSING the most important fourth variant for the paper's narrative:
#     - scope_fd_debt_only  (αu=0,   αd=0   — pure round-robin)
#
#   Without scope_fd_debt_only, the paper's own claim — "round-robin equipped
#   with two principled tie-breaker terms is sufficient to dominate the random
#   baseline" — is unverified. A reviewer will absolutely ask "what does pure
#   round-robin do? Do the bonus and penalty contribute anything measurable?"
#
#   ⚠️  PREREQUISITE:  scope_fd_debt_only must be registered as a method in the
#       codebase before EXP 9, 10, and 12 will run. If it does not exist yet,
#       add a 3-line variant in csfl_simulator/methods/fd_native/__init__.py
#       (or wherever scope_fd_no_server is registered) that hard-codes
#       αu = αd = 0. The ablation blocks below are written so the codebase
#       change is the only thing needed; the script invocations work as-is.
#
# =============================================================================
# HOW TO INCORPORATE THE NEW RESULTS INTO THE PAPER
# =============================================================================
#
# Once EXP 9–12 finish, the deltas to paper-main_scope.tex are as follows.
# All edits are designed to fit within the existing page budget — no new
# figures are introduced; existing tables grow by a few rows / a footnote.
#
# 1) ABLATION → new compact Table IV (after Table III)
#    From scope_fmnist_N50_K5_ablation (EXP 9) and
#    scope_fmnist_N50_K1_ablation (EXP 10), build:
#
#        TABLE IV. ABLATION AT FMNIST N=50, R=100, DL SNR -20 dB.
#        | Variant                | K | Final acc. | Gini  | Rounds-to-80% |
#        | Random (baseline)      | 5 | 0.641      | 0.144 | 30            |
#        | SCOPE-FD debt-only     | 5 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (no server)   | 5 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (no coverage) | 5 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (full)        | 5 | 0.647      | 0.000 | 10            |
#        | Random (baseline)      | 1 | 0.448      | 0.429 | 99            |
#        | SCOPE-FD debt-only     | 1 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (no server)   | 1 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (no coverage) | 1 | <fill>     | 0.000 | <fill>        |
#        | SCOPE-FD (full)        | 1 | 0.542      | 0.000 | 40            |
#
#    Insert a single paragraph in Section VI-C interpreting the table:
#      - If full > debt-only on accuracy: confirms three-term composition matters.
#      - If full ≈ debt-only: paper should be reframed honestly as "round-robin
#        with optional information-side tie-breakers"; the Gini guarantee
#        still holds and is still the headline contribution.
#    Either outcome is publishable. Do NOT bury the result.
#
# 2) MULTI-SEED AT K=1 → footnote on the K=1 row of Table II
#    From scope_fmnist_N50_K1_seed* (EXP 11), compute mean ± std across the
#    4 seeds (42, 7, 123, 2024) for both Random and SCOPE-FD final accuracy.
#    Update the K=1 row of Table II to:
#        K=1 | Final acc. Random: 0.448 ± <std>
#            | Final acc. SCOPE: 0.542 ± <std>
#    Add a footnote to Table II: "K=1 row reports mean ± std across 4 seeds;
#    all other rows use seed 42 only, consistent with [13]."
#    This is the cheapest possible response to "single-seed at the headline
#    number" without committing to multi-seed everywhere.
#
# 3) (OPTIONAL) COEFFICIENT SENSITIVITY → one sentence in Section IV-E
#    From scope_fmnist_N50_K5_coef_* (EXP 12, conditional), report:
#        "Across (αu, αd) ∈ {(0.1,0.05), (0.2,0.1), (0.3,0.1), (0.4,0.2)},
#         all satisfying the dominance-margin constraint (16), the final-round
#         accuracy varied by less than X pp and the participation Gini stayed
#         at 0.000 in every configuration, confirming that the specific values
#         within the admissible region are not a tuning hyperparameter."
#    This converts the dominance-margin claim from rhetorical to empirical
#    in a single line of paper real estate.
#
# 4) SECTION VI-A SETUP UPDATE
#    Add one sentence: "For the K=1 stress regime, results are reported as
#    mean ± std across 4 seeds {42, 7, 123, 2024}; all other configurations
#    use seed 42, consistent with the simulation protocol of [13]."
#
# =============================================================================
#
# USAGE
#   bash scripts/run_scope_experiments.sh --resume
#   bash scripts/run_scope_experiments.sh --exp 9,10,11    # just the new ones
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

# NEW: 4-variant ablation set including the missing pure-round-robin variant.
# Requires fd_native.scope_fd_debt_only to be registered in the codebase
# (αu = αd = 0, identical pipeline to scope_fd otherwise).
SCOPE_ABLATION_FULL="fd_native.scope_fd,fd_native.scope_fd_debt_only,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"

# NEW: paired comparison that includes the random baseline alongside all four
# ablation variants — used for the FMNIST ablation runs so Table IV has the
# random row built in without needing a separate alignment step in post.
ABLATION_PAIR="heuristic.random,fd_native.scope_fd,fd_native.scope_fd_debt_only,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"

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
        6) echo 1 ;;   # SCOPE ablation (CIFAR — does NOT populate paper tables)
        7) echo 7 ;;   # K-sweep at N=50 on Fashion-MNIST (7 K values)
        8) echo 5 ;;   # FMNIST channel-sweep at N=50, K=5 (5 DL SNR levels)
        # --- NEW: paper-table-populating experiments ---
        9)  echo 1 ;;  # FMNIST K=5 ablation (4 SCOPE variants + random)
        10) echo 1 ;;  # FMNIST K=1 ablation (4 SCOPE variants + random)
        11) echo 3 ;;  # K=1 multi-seed (3 extra seeds, paired random+SCOPE)
        12) echo 4 ;;  # OPTIONAL: coefficient sensitivity at FMNIST K=5
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
    # Note: 12 is included in the loop but the EXP 12 block itself is
    # gated behind a guard variable (RUN_COEF_SWEEP) below; if that guard
    # is off, EXP 12 simply doesn't execute and the planned count is wrong
    # by 4 — harmless, only affects the progress display.
    for e in 1 2 3 4 5 6 7 8 9 10 11 12; do
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

# Toggle for the optional coefficient sensitivity sweep (EXP 12). Set to
# true only if your codebase exposes --scope-au and --scope-ad CLI flags;
# otherwise leave false and skip the sweep, citing only the dominance-margin
# constraint analytically.
RUN_COEF_SWEEP=false

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
    CUR_EXP_NUM=1; CUR_EXP="S1/12-cifar"
    log "EXP 1/12: CIFAR headline — SCOPE alone (random from exp2_noise_dl-20)"
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
    log "EXP 2/12: Noise sweep — 5 DL SNR levels (SCOPE alone; pair with existing random)"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="S2/12-noise-${label}"
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
    log "EXP 3/12: Alpha sweep — low / mid / high-IID (SCOPE alone; pair with existing random)"
    for alpha in 0.1 0.5 5.0; do
        alabel=$(echo "$alpha" | tr '.' '_')
        CUR_EXP="S3/12-alpha${alabel}"
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
    CUR_EXP_NUM=4; CUR_EXP="S4/12-mnist"
    log "EXP 4/12: MNIST(private) + FMNIST(public) — random vs SCOPE"
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
    CUR_EXP_NUM=5; CUR_EXP="S5/12-fmnist"
    log "EXP 5/12: Fashion-MNIST(private) + MNIST(public) — random vs SCOPE"
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
# NOTE: This block remains for completeness with prior runs. It does NOT
# populate any row in the paper's tables (which are FMNIST-based). For the
# paper-table-populating ablation, see EXP 9 and EXP 10 below.
# =============================================================================
if should_run 6; then
    CUR_EXP_NUM=6; CUR_EXP="S6/12-ablation-cifar"
    log "EXP 6/12: SCOPE ablation (CIFAR — supplementary, not in paper tables)"
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
    log "EXP 7/12: K-sweep — N=50 on Fashion-MNIST (7 K values: 1,5,10,15,25,35,50)"
    for K in 1 5 10 15 25 35 50; do
        CUR_EXP="S7/12-N50-K${K}"
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
# EXP 8 — FMNIST channel-noise sweep at N=50, K=5 (5 DL SNR levels).
# Selected because the K-sweep on Fashion-MNIST shows SCOPE's cleanest win band
# at K=5 (10% participation): +0.6 pp accuracy, +4.07 AUC advantage over 100
# rounds, 20 rounds faster to 80% of final accuracy, and Gini 0.000 vs random's
# 0.144. This block probes whether that advantage survives mMIMO channel noise
# from error-free down to -30 dB DL SNR — the same 5-level sweep used on CIFAR
# in Exp 2, now at a (dataset, K, N) combination where SCOPE has real accuracy
# headroom to hold onto. Each run is a paired random + SCOPE compare because
# no FMNIST-N50-K5 random baselines exist at arbitrary SNR levels on disk.
# =============================================================================
if should_run 8; then
    CUR_EXP_NUM=8
    log "EXP 8/12: FMNIST channel sweep — N=50, K=5, 5 DL SNR levels"
    for lvl in "errfree:" "dl0:--channel-noise --ul-snr-db -8 --dl-snr-db 0" \
               "dl-10:--channel-noise --ul-snr-db -8 --dl-snr-db -10" \
               "dl-20:--channel-noise --ul-snr-db -8 --dl-snr-db -20" \
               "dl-30:--channel-noise --ul-snr-db -8 --dl-snr-db -30"; do
        label="${lvl%%:*}"
        flags="${lvl#*:}"
        CUR_EXP="S8/12-fmnist-N50-K5-${label}"
        log "  FMNIST N=50 K=5 @ ${label}"
        run_one "scope_fmnist_N50_K5_noise_${label}" \
            --methods "${PAIR_SET}" \
            ${BASE_FD} \
            --dataset Fashion-MNIST --public-dataset MNIST \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
            --total-clients 50 --clients-per-round 5 --rounds ${ROUNDS} \
            --batch-size 20 \
            ${flags} \
            --seed 42
    done
fi

# =============================================================================
# EXP 9 — FMNIST K=5 ABLATION (NEW). Populates the K=5 block of Table IV.
# -----------------------------------------------------------------------------
# This is the paper's most important missing experiment: it directly tests
# whether the three-term composition is doing real work, or whether pure
# round-robin (debt-only) already captures all of SCOPE's empirical advantage.
#
# Configuration mirrors EXP 5/EXP 7 K=5 EXACTLY (same seed, same dataset,
# same channel, same partition) so the random and full-SCOPE columns of
# Table IV can be filled directly from scope_fmnist_N50_K5 and the three
# ablation columns from this run. The run pair-compares all five methods
# in a single invocation so per-round metrics are aligned by seed.
#
# PREREQUISITE: fd_native.scope_fd_debt_only must be registered. If it is
# not, this block will fail and you should either (a) register the variant
# or (b) drop scope_fd_debt_only from ABLATION_PAIR and run with the
# remaining 4 methods, accepting that the ablation won't answer the
# "is round-robin alone sufficient" question.
# =============================================================================
if should_run 9; then
    CUR_EXP_NUM=9; CUR_EXP="S9/12-fmnist-N50-K5-ablation"
    log "EXP 9/12: FMNIST N=50 K=5 ablation — 4 SCOPE variants + random"
    run_one "scope_fmnist_N50_K5_ablation" \
        --methods "${ABLATION_PAIR}" \
        ${BASE_FD} \
        --dataset Fashion-MNIST --public-dataset MNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 50 --clients-per-round 5 --rounds ${ROUNDS} \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 10 — FMNIST K=1 ABLATION (NEW). Populates the K=1 block of Table IV.
# -----------------------------------------------------------------------------
# Companion to EXP 9 at the ultra-sparse stress regime. K=1 is exactly where
# the original Table II shows SCOPE's 9.4 pp accuracy gain, so it is also
# where the ablation matters most:
#
#   - If full SCOPE >> debt-only at K=1, the bonus and penalty are doing
#     real work in the high-stress regime; the three-term framing is
#     vindicated and Table IV anchors the paper's central technical claim.
#
#   - If full SCOPE ≈ debt-only at K=1, the 9.4 pp gain is attributable to
#     round-robin alone, not to the three-term composition. This is still
#     a publishable result (round-robin is a known design that no one had
#     applied to FD partial participation), but the paper must reframe:
#       * Section IV-A: "the three-term selector" → "the debt-driven
#         selector with optional information-side tie-breakers"
#       * Section VI: report the ablation honestly; do not bury it.
#       * Abstract: drop "three terms" framing; lead with the participation
#         guarantee and the FD-aggregation-aware design rationale.
#     Either reframing is harmless to the paper's core fairness contribution
#     (Proposition 1 holds for any debt-dominated score, including αu=αd=0).
# =============================================================================
if should_run 10; then
    CUR_EXP_NUM=10; CUR_EXP="S10/12-fmnist-N50-K1-ablation"
    log "EXP 10/12: FMNIST N=50 K=1 ablation — 4 SCOPE variants + random"
    run_one "scope_fmnist_N50_K1_ablation" \
        --methods "${ABLATION_PAIR}" \
        ${BASE_FD} \
        --dataset Fashion-MNIST --public-dataset MNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 50 --clients-per-round 1 --rounds ${ROUNDS} \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# EXP 11 — K=1 MULTI-SEED (NEW). Adds error bars to the Table II K=1 row.
# -----------------------------------------------------------------------------
# The K=1 number (Random 0.448 vs SCOPE 0.542 for a 9.4 pp gap) was reported
# at seed 42 only. Reviewers will rightly note that K=1 is the most
# seed-sensitive cell in the entire paper (one-client-per-round on a Dirichlet
# partition has high realised-distribution variance round-to-round). Three
# additional seeds {7, 123, 2024} let the K=1 row of Table II be reported as
# "0.448 ± σ_R" vs "0.542 ± σ_S" with a 4-seed sample, which is enough to
# either confirm the headline gap is robust or honestly report that it isn't.
#
# Each invocation pair-compares random + full SCOPE (the same PAIR_SET used
# in EXP 7 K=1) so the seed-aligned final-round accuracies can be averaged
# directly in post-processing. Total cost: 3 paired runs.
#
# Single-seed protocol from [13] is preserved everywhere else; the table
# footnote should make this explicit (see "HOW TO INCORPORATE" at the top).
# =============================================================================
if should_run 11; then
    CUR_EXP_NUM=11
    log "EXP 11/12: FMNIST N=50 K=1 multi-seed — 3 extra seeds (paired random+SCOPE)"
    for SEED in 7 123 2024; do
        CUR_EXP="S11/12-fmnist-K1-seed${SEED}"
        log "  seed = ${SEED}"
        run_one "scope_fmnist_N50_K1_seed${SEED}" \
            --methods "${PAIR_SET}" \
            ${BASE_FD} \
            --dataset Fashion-MNIST --public-dataset MNIST \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
            --total-clients 50 --clients-per-round 1 --rounds ${ROUNDS} \
            --batch-size 20 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            --seed ${SEED}
    done
fi

# =============================================================================
# EXP 12 — COEFFICIENT SENSITIVITY (NEW, OPTIONAL). One-line paper claim.
# -----------------------------------------------------------------------------
# The paper claims that any (αu, αd) inside the dominance-margin region
# αu + αd < 1 delivers the same participation guarantee, and that the
# specific values (0.3, 0.1) are not a tuning hyperparameter. This block
# verifies that claim empirically with four (αu, αd) pairs spanning the
# admissible region:
#   (0.1, 0.05)   light bonus, light penalty
#   (0.2, 0.1)    moderate bonus, moderate penalty
#   (0.3, 0.1)    paper's reported choice
#   (0.4, 0.2)    near-margin (sum 0.6, still inside)
# All four should land within ~1 pp on final accuracy and 0.000 on Gini if
# the dominance-margin argument is correct. If they don't, the paper's
# Section IV-E claim is overstated and the values become a tuned hyperparameter,
# which the paper would need to disclose honestly.
#
# This block is GATED by RUN_COEF_SWEEP because it requires --scope-au and
# --scope-ad CLI flags in the simulator. If your codebase doesn't expose
# these flags, leave RUN_COEF_SWEEP=false and the block is skipped; the
# paper claim then stands on the analytical dominance-margin argument alone,
# which is acceptable but weaker than empirical confirmation.
#
# To enable: set RUN_COEF_SWEEP=true near the top of the script (next to the
# global toggles) AND ensure your simulator accepts --scope-au / --scope-ad.
# =============================================================================
if should_run 12 && $RUN_COEF_SWEEP; then
    CUR_EXP_NUM=12
    log "EXP 12/12: Coefficient sensitivity — 4 (αu, αd) pairs at FMNIST N=50, K=5"
    for pair in "0_10_0_05:--scope-au 0.10 --scope-ad 0.05" \
                "0_20_0_10:--scope-au 0.20 --scope-ad 0.10" \
                "0_30_0_10:--scope-au 0.30 --scope-ad 0.10" \
                "0_40_0_20:--scope-au 0.40 --scope-ad 0.20"; do
        label="${pair%%:*}"
        flags="${pair#*:}"
        CUR_EXP="S12/12-fmnist-K5-coef-${label}"
        log "  (αu, αd) labelled ${label}"
        run_one "scope_fmnist_N50_K5_coef_${label}" \
            --methods "${SCOPE_ONLY}" \
            ${BASE_FD} \
            --dataset Fashion-MNIST --public-dataset MNIST \
            --partition dirichlet --dirichlet-alpha 0.5 \
            --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
            --total-clients 50 --clients-per-round 5 --rounds ${ROUNDS} \
            --batch-size 20 \
            --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
            ${flags} \
            --seed 42
    done
elif should_run 12 && ! $RUN_COEF_SWEEP; then
    log "EXP 12/12: Coefficient sensitivity SKIPPED (RUN_COEF_SWEEP=false)"
    echo "  To enable: set RUN_COEF_SWEEP=true near the top of this script"
    echo "  AND ensure the simulator accepts --scope-au / --scope-ad flags."
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
echo ""
echo "  NEW (EXP 9–12) results to incorporate into the paper:"
echo "    - scope_fmnist_N50_K5_ablation     → Table IV K=5 block"
echo "    - scope_fmnist_N50_K1_ablation     → Table IV K=1 block"
echo "    - scope_fmnist_N50_K1_seed{7,123,2024} → Table II K=1 row error bars"
echo "    - scope_fmnist_N50_K5_coef_*       → Section IV-E one-sentence claim (if run)"
echo ""
echo "  See the 'HOW TO INCORPORATE' block at the top of this script for"
echo "  the exact paper edits each result enables."