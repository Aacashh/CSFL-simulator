#!/usr/bin/env bash
# =============================================================================
# MAML-Select — second-round revision, single entry point.
#
#   nohup bash run_maml_revision2.sh --jobs 3 > MAML-Revision-2/console.log 2>&1 &
#
# Everything this round needs, in one run. Output goes to MAML-Revision-2/.
# Nothing outside that folder is written except the regenerated numbers file in
# the revision package.
#
# ---------------------------------------------------------------------------
# WHAT THIS ANSWERS, AND WHAT IT DOES NOT
#
# Four of the six reviewer comments are answered by the manuscript text and by
# the released reference implementation, not by new training. This script
# re-verifies those rather than re-running them, which takes seconds:
#
#   C1  main and supplementary describe different optimizers
#       -> method section rewritten from source. Verified by the code_release
#          tests, which pin the support and query lag D_sup(t) = D_query(t-1).
#   C2  latency cost defined inconsistently
#       -> normalized penalty. Verified by the scale-free cost test.
#   C3  per-client score not derived from the cohort objective
#       -> Proposition 1. Verified against brute-force Top-K enumeration.
#   C4  fairness and coverage must be re-evaluated given forced coverage
#       -> recomputed from the existing run logs with cold-start rounds
#          discarded. Stage 3 regenerates those numbers.
#
# Two need machine time:
#
#   C6  convergence claims assume a stationary objective
#       -> Stage 1 re-runs the three selector-convergence runs with per-round
#          drift logging, which turns the tracking term V_T of Corollary 2 from
#          an argued quantity into a measured one. This is the one result the
#          revision cannot ship without.
#   C5  CriticalFL coverage values contradict each other
#       -> already refuted from existing logs, but both current CriticalFL runs
#          stopped short of 200 rounds. Stage 2 adds a complete seed so the
#          refutation rests on a finished trajectory.
#
# Stage 2 also completes the CIFAR-100 grid for TiFL and FedCor, so those table
# rows carry a standard deviation over two finished seeds.
#
# ---------------------------------------------------------------------------
# SPEED
#
# The six runs are mutually independent. They write to separate directories,
# seed independently, and share only the read-only dataset cache. --jobs runs
# them concurrently, which is the only safe speed lever here.
#
#   parallel_clients must stay 0. simulator.py raises if it is not, because
#   MAML-Select captures per-client credit sequentially. Do not change it.
#   performance_mode is already true in the configs.yaml defaults.
#
# Measured on an M2 Pro over MPS one CIFAR-100 run takes about two hours, so
# --jobs 1 is roughly ten hours and --jobs 3 roughly four. Raise --jobs until
# GPU memory is the limit. 3 is a reasonable start on a 24 GB card.
#
# OPTIONS
#   --jobs N          concurrent runs, default 2
#   --device DEV      cuda (default), mps, or cpu
#   --only-drift      stage 1 only
#   --only-seeds      stage 2 only
#   --only-analysis   stage 3 only, no training
#   --fresh           ignore existing results and re-run everything
#   --dry-run         print the plan, run nothing
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"
export PYTHONUNBUFFERED=1

OUT="${ROOT}/MAML-Revision-2"
CONV_OUT="${OUT}/convergence"
BENCH_OUT="${OUT}/cifar100"
LOGDIR="${OUT}/logs"
PKG="${ROOT}/csfl_simulator/Paper Corrections/MAML_Select_Revision_Package"
PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOCK="${ROOT}/.run_maml_revision2.lock"

JOBS=2
DEVICE="cuda"
ONLY_DRIFT=false
ONLY_SEEDS=false
ONLY_ANALYSIS=false
FRESH=false
DRY_RUN=false
PIDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)          JOBS="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --only-drift)    ONLY_DRIFT=true; shift ;;
        --only-seeds)    ONLY_SEEDS=true; shift ;;
        --only-analysis) ONLY_ANALYSIS=true; shift ;;
        --fresh)         FRESH=true; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help)       sed -n '2,67p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${CONV_OUT}" "${BENCH_OUT}" "${LOGDIR}"

T0=$(date +%s)
banner() {
    echo
    echo "=============================================================="
    echo " $1"
    echo " $(date '+%Y-%m-%d %H:%M:%S')   elapsed $(( ($(date +%s)-T0)/60 )) min"
    echo "=============================================================="
}

echo "=============================================================="
echo " MAML-Select revision 2"
echo "   repo    ${ROOT}"
echo "   output  ${OUT}"
echo "   device  ${DEVICE}      concurrent runs  ${JOBS}"
echo "   started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ---- single instance --------------------------------------------------------
if [[ "$DRY_RUN" == false ]]; then
    if [[ -f "$LOCK" ]]; then
        OLD="$(cat "$LOCK" 2>/dev/null || true)"
        if [[ -n "$OLD" ]] && kill -0 "$OLD" 2>/dev/null; then
            echo "Already running (PID $OLD).  tail -f ${LOGDIR}/*.log"; exit 1
        fi
        echo "[lock] stale lock from PID ${OLD:-unknown}, taking over"
    fi
    echo $$ > "$LOCK"
    trap 'rm -f "$LOCK"' EXIT
fi

# ---- do not fight the SCOPE-FD campaign for the GPU -------------------------
if [[ -f "${ROOT}/.run_all.lock" && "${DEVICE}" != "cpu" && "$DRY_RUN" == false ]]; then
    SP="$(cat "${ROOT}/.run_all.lock" 2>/dev/null || true)"
    if [[ -n "$SP" ]] && kill -0 "$SP" 2>/dev/null; then
        echo; echo "!! The SCOPE-FD campaign is running (PID ${SP})."
        echo "   Two campaigns on one GPU will slow both. Wait, or use --device cpu."
        exit 1
    fi
fi

# =============================================================================
# STAGE 0   verification that needs no machine time
# =============================================================================
if [[ "$DRY_RUN" == false && "$ONLY_DRIFT" == false && "$ONLY_SEEDS" == false ]]; then
    banner "STAGE 0   claim tests and dataset check"

    echo "[test] released reference implementation, comments C1 C2 C3"
    ( cd "${PKG}/code_release" && "${PY}" -m pytest tests/ -q ) > "${LOGDIR}/S0_tests_${STAMP}.log" 2>&1
    if [[ $? -eq 0 ]]; then
        tail -2 "${LOGDIR}/S0_tests_${STAMP}.log"
        echo "  claim tests pass"
    else
        tail -15 "${LOGDIR}/S0_tests_${STAMP}.log"
        echo "  !! claim tests FAILED. C1, C2 and C3 rest on these. Fix before running."
        exit 1
    fi

    # A trust-store failure cost the SCOPE-FD campaign 12 runs mid-flight. Fail
    # in the first minute instead. Datasets are fetched serially here so the
    # concurrent stage below never races on a download.
    #     pip install -U certifi
    #     export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
    echo
    echo "[gate] datasets"
    "${PY}" - <<'PYEOF' || { echo "  !! dataset fetch failed, see the certifi note in this script"; exit 1; }
import sys
from csfl_simulator.core.datasets import get_dataset
bad = []
for name in ("Fashion-MNIST", "CIFAR-10", "CIFAR-100"):
    try:
        ds = get_dataset(name, train=True, download=True)
        print("  %s OK, %d train samples" % (name, len(ds)))
    except Exception as exc:
        bad.append(name)
        print("  %s FAILED: %s: %s" % (name, type(exc).__name__, str(exc)[:110]))
sys.exit(1 if bad else 0)
PYEOF
fi

# =============================================================================
# job list
# =============================================================================
JOBS_LIST=()
add() { JOBS_LIST+=("$1|$2|$3"); }   # label | conv-log or - | args

if [[ "$ONLY_SEEDS" == false && "$ONLY_ANALYSIS" == false ]]; then
    for pair in "fashion:conv_fashion" "cifar10:conv_cifar10" "cifar100:conv_cifar100"; do
        tag="${pair%%:*}"; exp="${pair##*:}"
        add "A_${tag}" "${CONV_OUT}/selector_convergence_${tag}.jsonl" \
            "--profile convergence --only ${exp} --seed 42 --output-dir ${CONV_OUT}"
    done
fi

if [[ "$ONLY_DRIFT" == false && "$ONLY_ANALYSIS" == false ]]; then
    for spec in "system_aware.tifl:2026" "ml.fedcor:2026" "research.criticalfl:2026"; do
        m="${spec%%:*}"; s="${spec##*:}"
        add "B_$(echo "$m" | tr '.' '_')_s${s}" "-" \
            "--profile cifar100 --only cifar100_benchmarks --method-key ${m} --seed ${s} --output-dir ${BENCH_OUT}"
    done
fi

# =============================================================================
# run the pool.  bash 3.2 has no `wait -n`, so poll for a free slot.
# =============================================================================
run_one() {
    local label="$1" conv="$2" args="$3"
    local log="${LOGDIR}/${label}_${STAMP}.log"
    local status="${LOGDIR}/.${label}.status"
    local resume=""
    rm -f "$status"
    [[ "$FRESH" == false ]] && resume="--resume"
    if [[ "$conv" != "-" ]]; then
        rm -f "$conv"
        MAML_SELECT_CONV_LOG="$conv" "${PY}" -m csfl_simulator.experiments.maml_select.run_experiments \
            $args --device "${DEVICE}" $resume > "$log" 2>&1
    else
        "${PY}" -m csfl_simulator.experiments.maml_select.run_experiments \
            $args --device "${DEVICE}" $resume > "$log" 2>&1
    fi
    echo $? > "$status"
}

if [[ ${#JOBS_LIST[@]} -gt 0 ]]; then
    banner "RUNS   ${#JOBS_LIST[@]} jobs, ${JOBS} at a time"
    for spec in "${JOBS_LIST[@]}"; do
        IFS='|' read -r label conv args <<< "$spec"
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [${label}]  ${args}"
            [[ "$conv" != "-" ]] && echo "        MAML_SELECT_CONV_LOG=${conv}"
            continue
        fi
        while :; do
            live=0; alive=()
            for p in ${PIDS[@]:-}; do
                if kill -0 "$p" 2>/dev/null; then alive+=("$p"); live=$((live+1)); fi
            done
            PIDS=(${alive[@]:-})
            [[ $live -lt $JOBS ]] && break
            sleep 10
        done
        echo "  [$(date '+%H:%M:%S')] start ${label}"
        run_one "$label" "$conv" "$args" &
        PIDS+=($!)
    done
    [[ "$DRY_RUN" == false ]] && wait
fi

if [[ "$DRY_RUN" == true ]]; then
    echo; echo "Dry run. ${#JOBS_LIST[@]} jobs planned, nothing executed."; exit 0
fi

# ---- per-job outcome --------------------------------------------------------
FAILED=0
if [[ ${#JOBS_LIST[@]} -gt 0 ]]; then
    echo
    echo "  job outcomes"
    for spec in "${JOBS_LIST[@]}"; do
        IFS='|' read -r label conv args <<< "$spec"
        st="$(cat "${LOGDIR}/.${label}.status" 2>/dev/null || echo '?')"
        if [[ "$st" == "0" ]]; then
            extra=""
            if [[ "$conv" != "-" ]]; then
                if grep -q drift_increment "$conv" 2>/dev/null; then
                    extra="   drift logged"
                else
                    extra="   !! no drift_increment, the selector.py patch is missing"
                    FAILED=$((FAILED+1))
                fi
            fi
            echo "    ok      ${label}${extra}"
        else
            echo "    FAILED  ${label}  rc=${st}   see ${LOGDIR}/${label}_${STAMP}.log"
            FAILED=$((FAILED+1))
        fi
    done
fi

# =============================================================================
# STAGE 3   regenerate every number the revision quotes
# =============================================================================
banner "STAGE 3   regenerate revision_numbers.json"
"${PY}" "${PKG}/analyze_revision.py" "${ROOT}" 2>&1 | tee "${LOGDIR}/S3_analysis_${STAMP}.log"
cp -f "${PKG}/revision_numbers.json" "${OUT}/revision_numbers.json" 2>/dev/null || true

"${PY}" - "$OUT" <<'PYEOF'
import datetime, json, os, sys
out = sys.argv[1]
p = os.path.join(out, "revision_numbers.json")
nums = json.load(open(p)) if os.path.exists(p) else {}
conv = {k: v for k, v in nums.get("selector_convergence", {}).items() if isinstance(v, dict)}
L = ["# Reviewer coverage, regenerated " + datetime.date.today().isoformat(), "",
     "| Comment | Answered by | State |", "|---|---|---|"]
def row(c, how, ok):
    L.append("| %s | %s | %s |" % (c, how, "ready" if ok else "INCOMPLETE"))
row("C1 differing optimizers", "method section rewritten, code_release tests", True)
row("C2 latency cost", "normalized penalty, scale-free cost test", True)
row("C3 score vs cohort objective", "Proposition 1, brute-force Top-K test", True)
row("C4 fairness and coverage", "recomputed with cold start discarded",
    bool(nums.get("fairness_without_cold_start")))
row("C5 CriticalFL coverage", "J <= Cov invariant plus a finished CriticalFL seed",
    bool(nums.get("benchmarks")))
has_drift = bool(conv) and all(d.get("drift_mean") is not None for d in conv.values())
row("C6 non-stationary objective", "measured V_T from drift logging", has_drift)
L += ["", "## Drift, the C6 evidence", ""]
if conv:
    L += ["| dataset | rounds | inner descent | mean drift | V_T |", "|---|---|---|---|---|"]
    for k in sorted(conv):
        v = conv[k]
        L.append("| %s | %s | %s | %s | %s |" % (
            k, v.get("rounds"), v.get("inner_descent_nonpositive"),
            v.get("drift_mean", "not measured"), v.get("drift_sum", "not measured")))
else:
    L.append("No convergence logs found. Stage 1 did not produce them.")
open(os.path.join(out, "COVERAGE.md"), "w").write("\n".join(L) + "\n")
print()
print("\n".join(L))
PYEOF

echo
echo "=============================================================="
echo " total wall clock  $(( ($(date +%s)-T0)/60 )) min"
echo " runs              ${OUT}"
echo " coverage report   ${OUT}/COVERAGE.md"
echo " numbers           ${PKG}/revision_numbers.json"
if [[ $FAILED -gt 0 ]]; then
    echo " !! ${FAILED} job(s) need attention, see above"
else
    echo " all jobs completed"
fi
echo "=============================================================="
echo
echo " Next: read V_T out of COVERAGE.md and put it into Sec. V-E of the"
echo " manuscript, replacing the paragraph that says V_T was not measured."
