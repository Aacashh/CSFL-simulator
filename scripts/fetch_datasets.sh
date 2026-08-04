#!/usr/bin/env bash
# =============================================================================
# Pre-fetch the datasets the SCOPE-FD revision campaign needs.
#
#   bash scripts/fetch_datasets.sh              fetch, verify, then launch the campaign
#   bash scripts/fetch_datasets.sh --no-run      fetch and verify only
#
# WHY THIS EXISTS
# Twelve runs in the previous campaign died with
#     urllib.error.URLError: [SSL: CERTIFICATE_VERIFY_FAILED]
# while torchvision tried to fetch CIFAR-10, STL-10 and EMNIST mid-run, and the
# failure only surfaced when each job reached its slot, hours in. This script
# fetches all three up front with curl, which can be told to skip certificate
# verification, and then verifies every archive against the MD5 that torchvision
# itself expects. Skipping verification on the transport while checking the hash
# of the payload is what keeps that safe: a corrupted or substituted file fails
# the hash and the script stops.
#
# Idempotent. Anything already extracted is left alone, so it is safe to re-run.
#
# Datasets and sizes:
#   FMNIST     ~30 MB    the primary private set
#   MNIST      ~10 MB    private set and cross-pair public set
#   CIFAR-10   ~163 MB   private set for the CIFAR runs
#   STL-10     ~2.5 GB   public set for the CIFAR runs
#   EMNIST     ~536 MB   private set for the cross-dataset runs
#   FSDD       ~10 MB    audio, fetched through the simulator's own loader
# =============================================================================
set -uo pipefail

RUN_AFTER=true
for a in "$@"; do
    case "$a" in
        --no-run) RUN_AFTER=false ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $a"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/data"
mkdir -p "$DATA"

FAILED=()

hr() { printf '%s\n' "-------------------------------------------------------------"; }

# md5 tool differs between Linux and macOS
md5of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
    else md5 -q "$1"; fi
}

# fetch <url> <dest> <expected-md5>
fetch() {
    local url="$1" dest="$2" want="$3"
    if [[ -f "$dest" ]] && [[ "$(md5of "$dest")" == "$want" ]]; then
        echo "  archive already present and verified"
        return 0
    fi
    # -C - resumes a partial file, which matters on a captive-portal network
    # that drops the connection every few hours. Re-running the script simply
    # continues from where it stopped.
    local attempt
    for attempt in 1 2 3; do
        echo "  downloading $(basename "$dest")  (attempt $attempt, resumable) ..."
        curl -C - -kL --fail --retry 5 --retry-delay 10 --retry-connrefused \
             --connect-timeout 30 -o "$dest" "$url"
        local rc=$?
        # 33 means the server refused a ranged request, so start over once
        if [[ $rc -eq 33 ]]; then rm -f "$dest"; continue; fi
        if [[ $rc -ne 0 ]]; then
            echo "  interrupted (curl rc=$rc). The partial file is kept."
            echo "  Re-run this script after logging back in and it will resume."
            return 1
        fi
        local got; got="$(md5of "$dest")"
        if [[ "$got" == "$want" ]]; then echo "  md5 OK"; return 0; fi
        echo "  !! MD5 mismatch (expected $want, got $got); discarding and retrying"
        rm -f "$dest"
    done
    return 1
}

# ------------------------------------------------- Fashion-MNIST and MNIST ---
# These are the primary datasets for almost every run. They were previously
# assumed to be on disk, which is true of a machine that has run before and
# false of a fresh checkout. torchvision fetches them from mirrors that have
# been reliable here, so the simulator's own loader is used.
hr; echo "Fashion-MNIST and MNIST  (~90 MB total)"
python3 - <<'PYEOF' || FAILED+=("Fashion-MNIST/MNIST")
from csfl_simulator.core.datasets import get_dataset
for name in ("Fashion-MNIST", "MNIST"):
    for train in (True, False):
        get_dataset(name, train=train, download=True)
    print(f"  {name} OK")
PYEOF

# ---------------------------------------------------------------- CIFAR-10 ---
hr; echo "CIFAR-10  (~163 MB)"
if [[ -d "$DATA/cifar-10-batches-py" ]]; then
    echo "  already extracted, skipping"
else
    if fetch "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
             "$DATA/cifar-10-python.tar.gz" "c58f30108f718f92721af3b95e74349a"; then
        tar -xzf "$DATA/cifar-10-python.tar.gz" -C "$DATA" && echo "  extracted"
    else
        FAILED+=("CIFAR-10")
    fi
fi

# ------------------------------------------------------------------ STL-10 ---
hr; echo "STL-10  (~2.5 GB)"
if [[ -d "$DATA/stl10_binary" ]]; then
    echo "  already extracted, skipping"
else
    if fetch "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz" \
             "$DATA/stl10_binary.tar.gz" "91f7769df0f17e558f3565bffb0c7dfb"; then
        tar -xzf "$DATA/stl10_binary.tar.gz" -C "$DATA" && echo "  extracted"
    else
        FAILED+=("STL-10")
    fi
fi

# ------------------------------------------------------------------ EMNIST ---
hr; echo "EMNIST  (~536 MB)"
RAW="$DATA/EMNIST/raw"
# torchvision opens all four of these, so all four must be present. Checking
# only one lets a partial extraction from an earlier attempt short-circuit the
# repair, which is exactly what happened before.
EMNIST_FILES=(emnist-digits-train-images-idx3-ubyte emnist-digits-train-labels-idx1-ubyte
              emnist-digits-test-images-idx3-ubyte  emnist-digits-test-labels-idx1-ubyte)
emnist_complete=true
for f in "${EMNIST_FILES[@]}"; do
    [[ -f "$RAW/$f" ]] || { emnist_complete=false; break; }
done
if [[ "$emnist_complete" == true ]]; then
    echo "  already extracted, skipping"
else
    [[ -d "$RAW" ]] && echo "  incomplete extraction detected, redoing it"
    mkdir -p "$RAW"
    OK=false
    # primary mirror, then the Google-hosted mirror torchvision also ships
    for url in "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip" \
               "https://storage.googleapis.com/emnist/gzip.zip"; do
        echo "  trying $url"
        if fetch "$url" "$RAW/gzip.zip" "58c8d27c78d21e728a6bc7b3cc06412e"; then
            OK=true; break
        fi
    done
    if [[ "$OK" == true ]]; then
        # torchvision wants emnist-digits-*-idx?-ubyte directly in raw/. The
        # archive nests them under gzip/, but mirrors differ, so find them
        # wherever they land rather than assuming the layout.
        (
            cd "$RAW" || exit 1
            unzip -oq gzip.zip || exit 1
            found=0
            while IFS= read -r f; do
                gunzip -c "$f" > "$(basename "${f%.gz}")" && found=$((found+1))
            done < <(find . -name 'emnist-digits-*.gz' -type f)
            [[ $found -gt 0 ]] || { echo "  !! no emnist-digits-*.gz inside the archive"; exit 1; }
            find . -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
        ) && echo "  extracted"
        # confirm the four files torchvision actually opens
        miss=0
        for f in "${EMNIST_FILES[@]}"; do
            [[ -f "$RAW/$f" ]] || { echo "  !! missing $f"; miss=1; }
        done
        [[ $miss -eq 0 ]] || FAILED+=("EMNIST")
    else
        FAILED+=("EMNIST")
    fi
fi

# -------------------------------------------------------------------- FSDD ---
hr; echo "FSDD  (~10 MB, via the simulator's own loader)"
python3 - <<'PY' || FAILED+=("FSDD")
from csfl_simulator.core.datasets import get_dataset
tr = get_dataset("FSDD", train=True, download=True)
print(f"  FSDD OK  train={len(tr)}")
PY

# ------------------------------------------------------------------ verify ---
hr; echo "Verifying every dataset loads with download disabled"
python3 - <<'PY'
import sys
from csfl_simulator.core.datasets import get_dataset
bad = []
for d in ("Fashion-MNIST", "MNIST", "CIFAR-10", "STL-10", "EMNIST", "FSDD"):
    try:
        tr = get_dataset(d, train=True, download=False)
        print(f"  {d:<14} OK   train={len(tr)}")
    except Exception as e:
        bad.append(d)
        print(f"  {d:<14} FAIL {type(e).__name__}: {str(e)[:90]}")
sys.exit(1 if bad else 0)
PY
VERIFY=$?

hr
if [[ ${#FAILED[@]} -ne 0 || $VERIFY -ne 0 ]]; then
    [[ ${#FAILED[@]} -gt 0 ]] && echo "Download problems: ${FAILED[*]}"
    [[ $VERIFY -ne 0 ]] && echo "At least one dataset does not load with download=False."
    echo
    echo "The campaign was NOT started. If a mirror is blocked from this network,"
    echo "fetch the archive on another machine, copy it to the path below, then"
    echo "re-run this script to verify and extract:"
    echo "    CIFAR-10  -> ${DATA}/cifar-10-python.tar.gz"
    echo "    STL-10    -> ${DATA}/stl10_binary.tar.gz"
    echo "    EMNIST    -> ${DATA}/EMNIST/raw/gzip.zip"
    exit 1
fi

echo "All datasets present and loadable."

if [[ "$RUN_AFTER" != true ]]; then
    echo "Launch the campaign when ready:"
    echo "    nohup bash run_scope_revision.sh > finish.log 2>&1 &"
    exit 0
fi

# Refuse to start a second campaign on the same GPU and the same output tree.
# The previous attempt left several background jobs behind, and two runs writing
# the same directories would contend for the GPU and interleave their results.
EXISTING="$(pgrep -f "bash run_scope_revision.sh" || true)"
if [[ -n "$EXISTING" ]]; then
    echo
    echo "!! A campaign already appears to be running (PID: $(echo $EXISTING | tr '\n' ' '))."
    echo "   Not starting another one. Inspect it with:"
    echo "       tail -f ${REPO_ROOT}/finish.log"
    echo "   Or stop it and re-run this script:"
    echo "       kill $(echo $EXISTING | tr '\n' ' ')"
    exit 1
fi

hr
LOGFILE="${REPO_ROOT}/finish.log"
echo "Starting the campaign, detached, logging to ${LOGFILE}"
cd "$REPO_ROOT"
nohup bash run_scope_revision.sh > "$LOGFILE" 2>&1 &
CAMPAIGN_PID=$!
echo "  PID ${CAMPAIGN_PID}"
echo

# Give the pre-flight gate time to run so a failure is visible before we exit,
# rather than the user discovering it hours later.
# The runner pipes its job loop through tee, which block-buffers off a terminal,
# so the first job line can lag. The gate's own output is unbuffered, so treat
# either signal as "we got past the part that fails".
echo "Waiting for the pre-flight gate ..."
STARTED=false
for _ in $(seq 1 36); do
    sleep 5
    if grep -qE "^\[[0-9]+/[0-9]+\]|FSDD OK" "$LOGFILE" 2>/dev/null; then
        STARTED=true
        echo "  gate passed"
        break
    fi
    if ! kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
        echo "  !! the campaign exited during the gate. Log follows:"
        hr; tail -30 "$LOGFILE"; hr
        exit 1
    fi
done
if [[ "$STARTED" != true ]]; then
    if kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
        echo "  no gate marker yet, but the process is alive. Continuing."
    else
        echo "  !! the campaign is no longer running. Log follows:"
        hr; tail -30 "$LOGFILE"; hr
        exit 1
    fi
fi

hr
tail -25 "$LOGFILE"
hr
echo "Campaign running as PID ${CAMPAIGN_PID}. It survives this terminal closing."
echo "  follow it   :  tail -f ${LOGFILE}"
echo "  count done  :  grep -c '>>> ok' ${LOGFILE}"
echo "  stop it     :  kill ${CAMPAIGN_PID}"
