#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD Pre-Submission Experiment Suite
#
# WHY THIS EXISTS
# ---------------
#   Companion to run_scope_experiments.sh, focused on the two final experiments
#   needed before IEEE submission. Both target the paper's two weakest
#   defensive points and were prioritised over polish-grade additions on
#   time-budget grounds:
#
#     1. The choice of (αu, αd) = (0.3, 0.1) in Section IV-E. The paper claims
#        this is NOT a tuning hyperparameter but a placement inside the
#        admissible region defined by the dominance-margin constraint αu+αd<1
#        in equation (16). A reviewer will challenge this. EXP 1 below sweeps
#        5 (αu, αd) pairs spanning pure round-robin, the paper default, the
#        interior, the near-boundary, and a deliberately VIOLATING point —
#        providing empirical confirmation of the analytical argument in a
#        single new compact table (Table IV).
#
#     2. The empirical results in Tables II/III are FMNIST-only. Reviewers
#        will rightly ask whether the result is FMNIST-specific. EXP 2 adds a
#        single KMNIST run at the practical sparse regime (K=5, K/N=10%) to
#        replicate the headline 3× convergence speedup on a second dataset
#        of comparable difficulty but a completely different domain (cursive
#        Japanese characters vs fashion items).
#
#   The earlier EXP 12 in run_scope_experiments.sh covered 4 (αu, αd) pairs
#   that were ALL safely inside the admissible region. EXP 1 in this script
#   supersedes that block by spanning the boundary AND a violating case,
#   which is what makes the empirical defense of (16) complete: a constraint
#   that never fails in any test is not a constraint, it is a coincidence.
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT INCLUDE
# ----------------------------------------------
#   Considered and dropped on time grounds:
#     - Multi-seed runs (paper protocol matches [13] in single-seed throughout)
#     - Per-round eval refresh on existing FMNIST configurations (cosmetic)
#     - Cumulative-participation visualisation data (numerical evidence in
#       Section VI-C is sufficient — std 4.96 vs 0.47 already reported)
#     - Dirichlet α sweep on the new dataset
#     - K=1 and K=15 on KMNIST (K=5 alone refutes the "FMNIST artifact"
#       objection; broader K coverage on a second dataset is overkill for a
#       single-table addition)
#
# WHERE THE OUTPUTS PLUG INTO THE PAPER
# -------------------------------------
#   EXP 1 → new compact Table IV (Section IV-E, ~¼ column).
#           5-row coefficient table:
#             Row 1: (0.0, 0.0)  — pure round-robin lower bound
#             Row 2: (0.1, 0.1)  — small interior point
#             Row 3: (0.3, 0.1)  — paper default
#             Row 4: (0.5, 0.4)  — near boundary, still admissible
#             Row 5: (0.7, 0.4)  — VIOLATES (16); Gini predicted nonzero
#           Expected story: rows 1–4 all give Gini = 0 with accuracy in a
#           narrow band → insensitivity inside the admissible region.
#           Row 5 produces Gini > 0 → constraint (16) is necessary, not
#           decorative. One sentence in IV-E converts the analytical claim
#           to an empirical one.
#
#   EXP 2 → 2–3 sentence KMNIST replication paragraph in Section VI-B, OR
#           a single-row addendum to Table II. No new figure required.
#           Expected story: SCOPE-FD reaches 80% of final accuracy in ~10
#           rounds vs ~30 for random, Gini 0.000 vs nonzero, accuracy tied.
#
#   To make room: trim the redundant "Classical FL client selectors..."
#   paragraph at the end of Section VI-C (overlaps with Section II), and
#   the K=N=50 omission paragraph at end of VI-B can become a footnote.
#
# PREREQUISITES
# -------------
#   1. Simulator must accept --scope-au and --scope-ad CLI flags. These
#      pass through to the SCOPE-FD method constructor and override the
#      defaults αu=0.3, αd=0.1. Same prerequisite as the optional EXP 12
#      in run_scope_experiments.sh — if that block was working in your
#      fork, no additional change is needed.
#
#      ⚠️  IMPORTANT: verify that the simulator's SCOPE-FD implementation
#         does NOT internally clamp/assert αu + αd < 1. Row 5 of the
#         ablation deliberately violates that constraint, and any internal
#         clamping would silently nullify the test. If clamping exists,
#         disable it for this run only.
#
#   2. Simulator must accept "KMNIST" as a --dataset value. KMNIST
#      (Kuzushiji-MNIST) is a drop-in replacement for MNIST: same
#      torchvision API (torchvision.datasets.KMNIST), same 28x28 grayscale
#      shape, same 10-class structure. If your loader registry doesn't
#      have it yet, add 3 lines mirroring the MNIST registration. The
#      FD-CNN1/2/3 model pool works without any modification.
#
# USAGE
#   bash scripts/run_scope_submission_experiments.sh
#   bash scripts/run_scope_submission_experiments.sh --resume
#   bash scripts/run_scope_submission_experiments.sh --exp 1     # coef only
#   bash scripts/run_scope_submission_experiments.sh --exp 2     # KMNIST only
#   bash scripts/run_scope_submission_experiments.sh --dry-run
# =============================================================================
set -euo pipefail

# Activate the project-local virtual environment at <repo>/venv.
# Resolve relative to this script so it works regardless of CWD.
# Override by exporting CSFL_VENV before invoking the script, e.g.
#   CSFL_VENV=~/myenv bash scripts/run_scope_submission_experiments.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CSFL_VENV="${CSFL_VENV:-${REPO_ROOT}/venv}"
# Linux/macOS layout has bin/activate; Windows (Git Bash) layout has Scripts/activate.
if [[ -f "${CSFL_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CSFL_VENV}/bin/activate"
elif [[ -f "${CSFL_VENV}/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CSFL_VENV}/Scripts/activate"
else
    echo "ERROR: venv not found at ${CSFL_VENV} (looked for bin/activate and Scripts/activate)" >&2
    echo "       Set CSFL_VENV to your virtual environment path, or create one at ${CSFL_VENV}." >&2
    exit 1
fi
echo "Activated venv: ${CSFL_VENV}"
echo "Python:         $(which python) ($(python --version 2>&1))"

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

# Same MNIST-grade architecture pool as the existing FMNIST runs. KMNIST is
# 28x28x1 with 10 classes, identical to MNIST/FMNIST tensor-wise, so the
# pool transfers without modification.
MNIST_MODELS="FD-CNN1,FD-CNN2,FD-CNN3"

PERF_FLAGS="--parallel-clients 2"
if [[ "$DEVICE" == "cuda" ]]; then
    PERF_FLAGS="${PERF_FLAGS} --use-amp --channels-last"
fi

# Identical FD substrate to run_scope_experiments.sh — every flag matches the
# paper's reported configuration so the new runs are seed-aligned with the
# existing scope_fmnist_N50_K5 baseline on disk.
BASE_FD="--paradigm fd \
         --local-epochs 2 \
         --public-dataset-size 2000 \
         --dynamic-steps --dynamic-steps-base 5 --dynamic-steps-period 25 \
         --batch-size 128 --distillation-batch-size 500 \
         --distillation-lr 0.001 --distillation-epochs 2 --temperature 1.0 \
         --fd-optimizer adam \
         --n-bs-antennas 64 --quantization-bits 8 \
         --eval-every 1 \
         --profile \
         ${PERF_FLAGS} \
         --device ${DEVICE} ${FAST_FLAG}"

ROUNDS=100

# ---- Method sets ----
# PAIR_SET    : random + SCOPE-FD in one invocation (KMNIST has no on-disk
#               baseline yet, so both run together for seed alignment).
# SCOPE_ONLY  : SCOPE-FD alone (random row of Table IV comes from the
#               existing scope_fmnist_N50_K5 run, which is identical config
#               under the same seed and therefore deterministic).
PAIR_SET="heuristic.random,fd_native.scope_fd"
SCOPE_ONLY="fd_native.scope_fd"

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=""
CUR_EXP=""
CUR_EXP_NUM=0

exp_run_count() {
    case "$1" in
        1) echo 5 ;;   # Coefficient ablation: 5 (αu, αd) pairs at FMNIST K=5
        2) echo 1 ;;   # KMNIST K=5 paired (random + SCOPE in one invocation)
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
    for e in 1 2; do
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

log "SCOPE-FD Pre-Submission Suite"
echo "  Device:         ${DEVICE}"
echo "  Mode:           $([ "$FAST_FLAG" = "--fast-mode" ] && echo "FAST (debug)" || echo "FULL")"
echo "  Resume:         ${RESUME}"
echo "  Dry-run:        ${DRY_RUN}"
[[ -n "$RUN_ONLY" ]] && echo "  Filter:         ${RUN_ONLY}"
echo "  Planned:        ${TOTAL_PLANNED} compare invocations"
echo "  Rounds:         ${ROUNDS}"
echo "  Seed:           42 (single-seed protocol from [13])"

# =============================================================================
# EXP 1 — COEFFICIENT ABLATION (5 ROWS) AT FMNIST N=50, K=5.
# -----------------------------------------------------------------------------
# Defends Section IV-E's claim that (αu, αd) is NOT a tuned hyperparameter
# but a placement inside the admissible region defined by the dominance-
# margin constraint αu + αd < 1 in equation (16).
#
# The 5 (αu, αd) pairs are chosen to span the full argument:
#
#   (0.0, 0.0)   — PURE ROUND-ROBIN. Lower bound: how much of SCOPE's gain
#                  comes from the debt term alone, without info-side
#                  tie-breakers? If full SCOPE ≈ pure round-robin, the
#                  paper's "three-term" framing is overstated; if full
#                  SCOPE > pure round-robin, the bonus and penalty are
#                  doing real work. Either result is publishable; the
#                  ablation row exists so the paper does not have to
#                  guess which one it is.
#   (0.1, 0.1)   — INTERIOR POINT. Sum 0.2, half the magnitude of default.
#                  Confirms results are insensitive to specific values
#                  inside the region (the heart of the dominance-margin
#                  argument: any admissible point is equivalent).
#   (0.3, 0.1)   — PAPER DEFAULT. Sum 0.4, well inside (16); reference
#                  row mirroring the paper's Table II/III SCOPE results.
#   (0.5, 0.4)   — NEAR-BOUNDARY. Sum 0.9, dominance margin of just 0.1.
#                  Confirms the guarantee holds all the way to the
#                  boundary, not just at conservative interior points.
#   (0.7, 0.4)   — VIOLATES (16). Sum 1.1; the dominance-margin condition
#                  is intentionally broken. PREDICTED OUTCOME: Gini becomes
#                  nonzero because the bonus and penalty can now override
#                  the debt term and re-pick the same high-utility clients
#                  across rounds. THIS row is what makes the empirical
#                  defense of (16) complete. If Gini stays at 0.000 on
#                  this row too, either the implementation is silently
#                  clamping αu + αd < 1 (verify and disable for this
#                  run) or the constraint is conservative on this dataset
#                  and a remark to that effect should appear in IV-E.
#
# Configuration mirrors EXP 7 K=5 / EXP 5 EXACTLY (FMNIST private, MNIST
# public, Dirichlet α=0.5, FD-CNN1/2/3, N=50, K=5, DL SNR -20 dB, seed 42).
# Random baseline for Table IV is REUSED from the on-disk
# scope_fmnist_N50_K5 run — same seed, same config, deterministic. Hence
# --methods SCOPE_ONLY (not paired): only the 5 SCOPE rows are new.
#
# ⚠️  PREREQUISITE: simulator exposes --scope-au and --scope-ad CLI flags
#     AND does not internally enforce αu + αd < 1. See script header.
# =============================================================================
if should_run 1; then
    CUR_EXP_NUM=1
    log "EXP 1/2: Coefficient ablation — 5 (αu, αd) pairs at FMNIST N=50, K=5"
    for pair in "0_00_0_00:--scope-au 0.00 --scope-ad 0.00" \
                "0_10_0_10:--scope-au 0.10 --scope-ad 0.10" \
                "0_30_0_10:--scope-au 0.30 --scope-ad 0.10" \
                "0_50_0_40:--scope-au 0.50 --scope-ad 0.40" \
                "0_70_0_40:--scope-au 0.70 --scope-ad 0.40"; do
        label="${pair%%:*}"
        flags="${pair#*:}"
        CUR_EXP="S1/2-fmnist-K5-coef-${label}"
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
fi

# =============================================================================
# EXP 2 — KMNIST AT K=5 (PRACTICAL SPARSE REGIME).
# -----------------------------------------------------------------------------
# Refutes the "FMNIST artifact" reviewer concern. KMNIST (Kuzushiji-MNIST)
# is a 28x28 grayscale 10-class dataset of cursive Japanese characters
# (torchvision.datasets.KMNIST). Selected over alternatives for three
# concrete reasons:
#
#   1. DROP-IN COMPATIBLE with the existing FD-CNN1/2/3 model pool — same
#      input shape, channel count, and class count as MNIST/FMNIST. Zero
#      architecture or model-pool changes.
#
#   2. GENUINELY HARDER than MNIST (~95% ceiling vs ~99%) but easier than
#      CIFAR — leaves headroom for the selection-axis signal to be
#      observable. This is the same reason the paper itself cited in
#      Section VI-A for preferring FMNIST over CIFAR-10 as the primary
#      benchmark, so the choice is internally consistent.
#
#   3. DIFFERENT DOMAIN from FMNIST — cursive Japanese characters vs
#      fashion items. Replication on KMNIST is much stronger evidence of
#      dataset-independence than another fashion- or digit-style benchmark
#      would be.
#
# Only K=5 (the practical sparse regime, K/N = 10%) is run, because this
# is where the paper's headline 3× convergence-speedup result lives. K=1
# and K=15 on KMNIST are deliberately omitted: if the trend replicates at
# K=5 reviewers will accept the dataset-independence argument; if it does
# not, adding K=1 and K=15 would not change that.
#
# Public dataset: MNIST. Same cross-pairing structure as the FMNIST→MNIST
# headline already in the paper, so any difference is attributable to the
# private-dataset content rather than to a public-dataset mismatch.
#
# This is a paired (random + SCOPE) compare — no KMNIST baselines exist
# on disk yet, so both methods run in one invocation to keep them
# seed-aligned for direct trajectory comparison.
#
# ⚠️  PREREQUISITE: simulator accepts "KMNIST" as a --dataset value.
#     See script header for the 3-line registry addition if needed.
# =============================================================================
if should_run 2; then
    CUR_EXP_NUM=2; CUR_EXP="S2/2-kmnist-N50-K5"
    log "EXP 2/2: KMNIST(private) + MNIST(public) — N=50, K=5, paired random+SCOPE"
    run_one "scope_kmnist_N50_K5" \
        --methods "${PAIR_SET}" \
        ${BASE_FD} \
        --dataset KMNIST --public-dataset MNIST \
        --partition dirichlet --dirichlet-alpha 0.5 \
        --model FD-CNN1 --model-heterogeneous --model-pool "${MNIST_MODELS}" \
        --total-clients 50 --clients-per-round 5 --rounds ${ROUNDS} \
        --batch-size 20 \
        --channel-noise --ul-snr-db -8 --dl-snr-db -20 \
        --seed 42
fi

# =============================================================================
# Summary
# =============================================================================
GLOBAL_END=$(date +%s)
DT=$(( GLOBAL_END - GLOBAL_START ))

log "PRE-SUBMISSION SUITE COMPLETE"
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
echo ""
echo "  Paper integration:"
echo "    - scope_fmnist_N50_K5_coef_*  → new compact Table IV in Section IV-E"
echo "        (random row reused from existing scope_fmnist_N50_K5 on disk)"
echo "    - scope_kmnist_N50_K5         → 2-3 sentence replication paragraph"
echo "        in Section VI-B, or single-row addendum to Table II"
echo ""
echo "  Reclaim ~½ column for these additions by trimming the redundant"
echo "  'Classical FL client selectors...' paragraph at end of VI-C and"
echo "  collapsing the K=N=50 omission paragraph at end of VI-B to a footnote."
