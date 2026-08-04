#!/usr/bin/env bash
# =============================================================================
# SCOPE-FD — run everything, unattended.
#
#   nohup bash run_all.sh > run_all.log 2>&1 &
#
# Three phases, strictly in order, each waiting for the one before it:
#
#   1  datasets   fetch and verify all six. Nothing else starts unless every
#                 one of them loads with downloading disabled.
#   2  revision   the reviewer-coverage campaign (run_scope_revision.sh)
#   3  hardening  the claim-hardening campaign  (run_scope_hardening.sh)
#
# RESUMABLE AT EVERY LEVEL. Datasets already on disk are skipped, and both
# campaigns skip any run whose results are already complete. If this is killed
# at any point, by a crash, a reboot or a dropped session, launch the identical
# command again and it continues from where it stopped. Nothing is recomputed.
#
# SAFE TO LEAVE. Only one copy can run at a time, enforced by a lock file, so a
# second launch reports the running PID and exits rather than putting two
# campaigns on the same GPU.
#
# OPTIONS
#   --skip-hardening   stop after phase 2
#   --only-hardening   skip phase 2, useful once the revision campaign is done
#   --dry-run          print what each phase would do, run nothing
#   --jobs N           concurrency for phase 3 (default 1)
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

# Python buffers stdout when it is not a terminal, so under nohup the per-round
# progress would arrive in chunks. Stream it instead.
export PYTHONUNBUFFERED=1

SKIP_HARDENING=false
ONLY_HARDENING=false
DRY_RUN=false
HJOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-hardening) SKIP_HARDENING=true; shift ;;
        --only-hardening) ONLY_HARDENING=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --jobs)           HJOBS="$2"; shift 2 ;;
        -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOCK="$ROOT/.run_all.lock"
LOGDIR="$ROOT/logs_run_all"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- single instance --------------------------------------------------------
if [[ "$DRY_RUN" == false ]]; then
    if [[ -f "$LOCK" ]]; then
        OLD="$(cat "$LOCK" 2>/dev/null || true)"
        if [[ -n "$OLD" ]] && kill -0 "$OLD" 2>/dev/null; then
            echo "A run is already in progress (PID $OLD)."
            echo "  follow it :  tail -f $LOGDIR/*.log"
            echo "  stop it   :  kill $OLD"
            exit 1
        fi
        echo "[lock] stale lock from PID ${OLD:-unknown}, taking over"
    fi
    echo $$ > "$LOCK"
    trap 'rm -f "$LOCK"' EXIT
fi

banner() {
    echo
    echo "=============================================================="
    echo " $1"
    echo " $(date '+%Y-%m-%d %H:%M:%S')   elapsed $(( ($(date +%s)-T0)/60 )) min"
    echo "=============================================================="
}

T0=$(date +%s)
echo "=============================================================="
echo " SCOPE-FD full run"
echo "   repo:      $ROOT"
echo "   started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "   phases:    $([[ "$ONLY_HARDENING" == true ]] && echo "3 only" || { [[ "$SKIP_HARDENING" == true ]] && echo "1, 2" || echo "1, 2, 3"; })"
echo "   dry run:   $DRY_RUN"
echo "=============================================================="

DRY=""; [[ "$DRY_RUN" == true ]] && DRY="--dry-run"

# ---- phase 1: datasets ------------------------------------------------------
if [[ "$ONLY_HARDENING" == false ]]; then
    banner "PHASE 1 of 3   datasets"
    # --no-run is essential: fetch_datasets.sh would otherwise launch the
    # revision campaign itself, in the background, and phase 2 would then run
    # a second copy against the same output tree.
    if [[ "$DRY_RUN" == true ]]; then
        echo "  would run: bash scripts/fetch_datasets.sh --no-run"
    else
        bash scripts/fetch_datasets.sh --no-run 2>&1 | tee "$LOGDIR/1_datasets_$STAMP.log"
        rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo
            echo "!! Datasets are not ready (exit $rc). Nothing else was started,"
            echo "   because every campaign would fail on the missing data."
            echo "   Fix the download, then run this same command again."
            exit 1
        fi
    fi
fi

# ---- phase 2: reviewer-coverage campaign ------------------------------------
if [[ "$ONLY_HARDENING" == false ]]; then
    banner "PHASE 2 of 3   revision campaign"
    echo "  completed runs are skipped; only outstanding work executes"
    if [[ "$DRY_RUN" == true ]]; then
        bash run_scope_revision.sh --dry-run --skip-tests 2>&1 | tail -5
    else
        bash run_scope_revision.sh 2>&1 | tee "$LOGDIR/2_revision_$STAMP.log"
        echo ">>> phase 2 finished after $(( ($(date +%s)-T0)/3600 ))h"
    fi
fi

# ---- phase 3: claim-hardening campaign --------------------------------------
if [[ "$SKIP_HARDENING" == false ]]; then
    banner "PHASE 3 of 3   hardening campaign"
    if [[ "$DRY_RUN" == true ]]; then
        bash run_scope_hardening.sh --dry-run --skip-tests 2>&1 | tail -5
    else
        bash run_scope_hardening.sh --jobs "$HJOBS" 2>&1 | tee "$LOGDIR/3_hardening_$STAMP.log"
        echo ">>> phase 3 finished after $(( ($(date +%s)-T0)/3600 ))h"
    fi
fi

[[ "$DRY_RUN" == true ]] && { echo; echo "Dry run complete, nothing executed."; exit 0; }

# ---- summary ----------------------------------------------------------------
banner "DONE"
OUT="${SCOPE_OUT:-}"
if [[ -z "$OUT" ]]; then
    if   [[ -d "runs_scope_revised" ]];      then OUT="runs_scope_revised"
    elif [[ -d "runs/runs_scope_revised" ]]; then OUT="runs/runs_scope_revised"
    fi
fi
if [[ -n "$OUT" && -d "$OUT" ]]; then
    python3 - "$OUT" <<'PY'
import json, os, sys, glob
from collections import Counter
root = sys.argv[1]
ok = Counter(); bad = Counter()
for mf in glob.glob(os.path.join(root, "*", "*", "manifest.json")):
    fam = mf.split(os.sep)[-3]
    rf = mf.replace("manifest.json", "compare_results.json")
    good = False
    if os.path.exists(rf):
        try:
            r = json.load(open(rf)).get("results", {})
            good = bool(r) and all(v.get("metrics") for v in r.values())
        except Exception:
            good = False
    (ok if good else bad)[fam] += 1
print(f"  {'family':<34}{'complete':>9}{'incomplete':>12}")
for fam in sorted(set(ok) | set(bad)):
    print(f"  {fam:<34}{ok[fam]:>9}{bad[fam]:>12}")
print(f"\n  TOTAL complete: {sum(ok.values())}    incomplete: {sum(bad.values())}")
PY
fi
echo
echo "  total wall clock: $(( ($(date +%s)-T0)/3600 ))h"
echo "  logs: $LOGDIR/"
echo
echo "  Anything incomplete can be retried by running this same command again."
