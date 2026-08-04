#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD — theorem-hardening campaign.
#
#   bash run_scope_hardening.sh                 sequential
#   bash run_scope_hardening.sh --jobs 2        two jobs at once, VRAM permitting
#   bash run_scope_hardening.sh --only X1,X3    just those families
#   bash run_scope_hardening.sh --dry-run
#
# These are not more coverage experiments. The reviewer questions are already
# answered by the main campaign. These five test the paper's own claims, and are
# ordered so that the two capable of changing what the paper may assert run
# first. Full reasoning is in
# csfl_simulator/Paper Corrections/Scope_FD_Revision_Package/CLAIM_AUDIT.md
#
#   X1  alpha_u + alpha_d swept across 1        ~3.7 h
#       Proposition 1 holds when alpha_u + alpha_d < 1, because the normalised
#       debt gap is 1/(1+eps) while the two information terms are bounded by
#       their weights. Every run so far sits at 0.4, deep inside the safe
#       region, so the precondition has never been exercised. This sweeps
#       across the threshold. If the Gini degrades exactly where the proof says
#       it must, Proposition 1 is sharp rather than merely sufficient.
#
#   X2  K in {20, 30} at N=30, i.e. up to full participation   ~6.7 h
#       The convergence claim is about replacing full participation with
#       partial participation, yet the highest ratio ever run is K/N = 33%.
#       There is currently nothing to compare against. At K = N every selector
#       picks every client, so one run per seed is the shared reference.
#
#   X3  four-way ablation at alpha in {0.05, 0.1, 0.3}, K = 3   ~5.1 h
#       The ablation exists only at alpha = 0.5, which is the setting where the
#       coverage term is least likely to matter, since a Dirichlet draw at 0.5
#       already gives each client several classes. Existing data shows the
#       complete score winning at K = 3 and losing at K = 1 and K >= 20, which
#       points to a useful regime of 1 < K <~ C that should widen as alpha
#       falls. This tests that directly.
#
#   X4  headline extended to R = 300                            ~3.7 h
#       Proposition 1 predicts the Gini decays as O(1/R). Measured 3.33% at
#       R = 25 falling to 1.33% at R = 100. At R = 300 the bound is 0.50%.
#
#   X5  FedCS and TiFL ported into FD beside Oort               ~1.2 h
#       The claim that FL selectors do not transfer currently rests on Oort
#       alone. Two more turns one data point into three.
#
# MEMORY. Every job is its own `python -m csfl_simulator` process, so all CUDA
# memory is returned to the driver when it exits. That is a stronger guarantee
# than any in-process cleanup. What this script adds is a wait for the driver to
# actually release it before the next job starts, because teardown lags process
# exit and back-to-back launches can otherwise fail on a fragmented heap.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Output root. Honours SCOPE_OUT, then falls back to whichever layout is
# actually present, since some checkouts keep the runs under runs/.
OUT="${SCOPE_OUT:-}"
if [[ -z "$OUT" ]]; then
    if   [[ -d "runs_scope_revised" ]];      then OUT="runs_scope_revised"
    elif [[ -d "runs/runs_scope_revised" ]]; then OUT="runs/runs_scope_revised"
    else OUT="runs_scope_revised"
    fi
fi
SEEDS="11 22 33"
JOBS=1
DRY_RUN=false
SKIP_TESTS=false
ONLY=""
VRAM_FREE_MB=3500          # wait until at least this much is free before launching

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)       JOBS="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        --output-root) OUT="$2"; shift 2 ;;
        -h|--help)    sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT/logs"
LOG="$OUT/logs/hardening_$(date +%Y%m%d_%H%M%S).log"

M4="fd_native.scope_fd,fd_native.scope_fd_debt_only,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"
M2="heuristic.random,fd_native.scope_fd"
M_FL="heuristic.random,fd_native.scope_fd,system_aware.oort,system_aware.fedcs,system_aware.tifl"

COMMON="--paradigm fd --partition dirichlet --dirichlet-alpha 0.5 \
--total-clients 30 --clients-per-round 5 --rounds 100 --local-epochs 1 \
--batch-size 64 --lr 0.001 --public-dataset same --public-dataset-size 2000 \
--distillation-epochs 2 --distillation-batch-size 500 --eval-every 5 \
--use-amp --performance-mode --parallel-clients -1 --num-workers 4"
P_IMG="--dataset Fashion-MNIST --model FD-CNN2 --model-heterogeneous --model-pool FD-CNN1,FD-CNN2,FD-CNN3"

JOBS_LIST=()
add() { JOBS_LIST+=("$1|$2|$3|$4"); }

# ---- X1  Proposition 1 precondition ----------------------------------------
# (au, ad) chosen so au+ad crosses 1 while keeping the 3:1 ratio of the default.
for pair in "0.30 0.10" "0.60 0.20" "0.75 0.25" "0.98 0.32" "1.35 0.45" "2.25 0.75"; do
    set -- $pair; au=$1; ad=$2
    sum=$(python3 -c "print(f'{$au+$ad:.2f}')")
    for s in $SEEDS; do
        add X1_prop1_boundary "sum${sum}_s$s" "fd_native.scope_fd" \
            "$P_IMG --scope-au $au --scope-ad $ad --seed $s"
    done
done

# ---- X2  towards full participation ----------------------------------------
# At K=N every selector admits every client, so one method is the shared
# reference; below that the comparison still needs random.
for s in $SEEDS; do
    add X2_full_participation "K20_s$s" "$M2" "$P_IMG --clients-per-round 20 --seed $s"
done
for s in $SEEDS; do
    add X2_full_participation "K30_full_s$s" "fd_native.scope_fd" \
        "$P_IMG --clients-per-round 30 --seed $s"
done

# ---- X3  do the information terms ever help? -------------------------------
for a in 0.05 0.1 0.3; do for s in $SEEDS; do
    add X3_ablation_heterogeneity "a${a}_K3_s$s" "$M4" \
        "$P_IMG --dirichlet-alpha $a --clients-per-round 3 --seed $s"
done; done

# ---- X4  long horizon ------------------------------------------------------
for s in $SEEDS; do
    add X4_long_horizon "R300_s$s" "$M2" "$P_IMG --rounds 300 --eval-every 10 --seed $s"
done

# ---- X5  more FL selectors inside FD ---------------------------------------
for s in $SEEDS; do
    add X5_fl_selectors "s$s" "$M_FL" "$P_IMG --seed $s"
done

# ---- filter ----------------------------------------------------------------
if [[ -n "$ONLY" ]]; then
    KEEP=()
    IFS=',' read -ra WANT <<< "$ONLY"
    for j in "${JOBS_LIST[@]}"; do
        for w in "${WANT[@]}"; do
            [[ "${j%%|*}" == ${w}_* ]] && { KEEP+=("$j"); break; }
        done
    done
    JOBS_LIST=("${KEEP[@]}")
fi
N_JOBS=${#JOBS_LIST[@]}

# ---- helpers ---------------------------------------------------------------
is_done() {
    [[ -f "$1" ]] || return 1
    python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    r=json.load(open(sys.argv[1])).get("results",{})
    sys.exit(0 if r and all(v.get("metrics") for v in r.values()) else 1)
except Exception: sys.exit(1)
PY
}

# CUDA teardown lags process exit, so wait for the driver to hand the memory
# back rather than launching straight into a fragmented heap.
wait_for_vram() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    for _ in $(seq 1 60); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        [[ -z "$free" ]] && return 0
        (( free >= VRAM_FREE_MB )) && return 0
        sleep 5
    done
    echo "    [vram] still below ${VRAM_FREE_MB} MiB free after 5 min, proceeding anyway"
}

run_one() {
    local job="$1" idx="$2"
    IFS='|' read -r fam tag methods args <<< "$job"
    local dir="$OUT/$fam/$tag" res="$OUT/$fam/$tag/compare_results.json"
    if is_done "$res"; then echo "[$idx/$N_JOBS] skip (done)  $fam/$tag"; return 0; fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "[$idx/$N_JOBS] $fam/$tag"
        echo "    python3 -m csfl_simulator compare --methods $methods --name h_${fam}_${tag} --output $res $COMMON $args"
        return 0
    fi
    mkdir -p "$dir"
    printf '{"family": "%s", "tag": "%s", "methods": "%s"}\n' "$fam" "$tag" "$methods" > "$dir/manifest.json"
    echo "[$idx/$N_JOBS] $fam/$tag  $(date '+%H:%M:%S')"
    local t0=$(date +%s)
    # shellcheck disable=SC2086
    python3 -m csfl_simulator compare --methods "$methods" \
        --name "h_${fam}_${tag}" --output "$res" \
        $COMMON $args > "$dir/stdout.log" 2>&1
    local rc=$? mins=$(( ($(date +%s)-t0)/60 ))
    if [[ $rc -eq 0 ]] && is_done "$res"; then echo "    ok  (${mins} min)"
    else echo "    FAILED rc=$rc (${mins} min), see $dir/stdout.log"; fi
}

# ---- gate ------------------------------------------------------------------
echo "=============================================================="
echo " SCOPE-FD hardening campaign"
echo "   jobs:     $N_JOBS      concurrency: $JOBS"
echo "   families: X1 boundary  X2 full-participation  X3 ablation-vs-alpha"
echo "             X4 long-horizon  X5 FL-selectors"
echo "   log:      $LOG"
echo "=============================================================="
if [[ "$SKIP_TESTS" == false && "$DRY_RUN" == false ]]; then
    python3 -m pytest tests/test_scope_revision.py -q || { echo "tests failed, aborting"; exit 1; }
fi

START=$(date +%s)
if [[ "$JOBS" -le 1 ]]; then
    i=0
    for job in "${JOBS_LIST[@]}"; do
        i=$((i+1)); [[ "$DRY_RUN" == false ]] && wait_for_vram
        run_one "$job" "$i"
    done 2>&1 | tee -a "$LOG"
else
    echo "[note] running $JOBS jobs at a time; each is its own process, so CUDA"
    echo "       memory is returned on exit. Output per job is in its own stdout.log."
    # The wait must live inside the same subshell as the background jobs. A
    # `wait` placed after the pipeline runs in the parent, which knows nothing
    # about them, and aggregation would then start on half-written results.
    {
        i=0
        for job in "${JOBS_LIST[@]}"; do
            i=$((i+1))
            while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 10; done
            [[ "$DRY_RUN" == false ]] && wait_for_vram
            run_one "$job" "$i" &
        done
        wait
    } 2>&1 | tee -a "$LOG"
fi

[[ "$DRY_RUN" == true ]] && { echo; echo "Dry run, nothing executed."; exit 0; }

echo
echo "Elapsed $(( ($(date +%s)-START)/3600 ))h. Aggregating ..."
python3 csfl_simulator/experiments/scope_fd/aggregate_results.py "$OUT" \
    --output-dir "$OUT/aggregated" --reference-method fd_native.scope_fd
cat <<'EOF'

==============================================================
 What to read out of these runs
   X1  Gini against alpha_u+alpha_d. Flat at 1.33% below 1 and
       rising above it confirms Proposition 1 is sharp.
   X2  SCOPE at K=20 against the K=N=30 reference. A closing gap
       supports the compatibility claim in Section V-B.
   X3  Complete score minus debt-only against alpha at K=3. A gap
       opening as alpha falls justifies the three-term design.
   X4  Gini at R=300 against the bound N/(4KR)=0.50%.
   X5  FedCS and TiFL beside Oort. Three collapses, not one.
==============================================================
EOF
