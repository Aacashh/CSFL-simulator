#!/usr/bin/env bash
# =============================================================================
# Pre-fetch the datasets the SCOPE-FD revision campaign needs.
#
#   bash scripts/fetch_datasets.sh
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
#   CIFAR-10   ~163 MB   private set and public set for the CIFAR runs
#   STL-10     ~2.5 GB   public set for the CIFAR runs
#   EMNIST     ~536 MB   private set for the cross-dataset runs
#   FSDD       ~10 MB    audio, fetched through the simulator's own loader
# =============================================================================
set -uo pipefail

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
    echo "  downloading $(basename "$dest") ..."
    curl -kL --fail --retry 3 --retry-delay 5 -o "$dest" "$url" || return 1
    local got; got="$(md5of "$dest")"
    if [[ "$got" != "$want" ]]; then
        echo "  !! MD5 MISMATCH"
        echo "     expected $want"
        echo "     got      $got"
        echo "     deleting the bad file so a re-run starts clean"
        rm -f "$dest"
        return 1
    fi
    echo "  md5 OK"
}

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
hr; echo "STL-10  (~2.5 GB, this is the slow one)"
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
if [[ -f "$RAW/emnist-digits-train-images-idx3-ubyte" ]]; then
    echo "  already extracted, skipping"
else
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
        ( cd "$RAW" \
          && unzip -oq gzip.zip \
          && for f in gzip/*.gz; do gunzip -c "$f" > "$(basename "${f%.gz}")"; done \
          && rm -rf gzip ) && echo "  extracted"
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
if [[ ${#FAILED[@]} -eq 0 && $VERIFY -eq 0 ]]; then
    echo "All datasets present. The campaign can be launched:"
    echo "    nohup bash run_scope_revision.sh > finish.log 2>&1 &"
else
    [[ ${#FAILED[@]} -gt 0 ]] && echo "Download problems: ${FAILED[*]}"
    [[ $VERIFY -ne 0 ]] && echo "At least one dataset does not load with download=False."
    echo
    echo "If a mirror is blocked from this network, fetch the archive on another"
    echo "machine and copy it in, then re-run this script to extract and verify:"
    echo "    CIFAR-10  -> ${DATA}/cifar-10-python.tar.gz"
    echo "    STL-10    -> ${DATA}/stl10_binary.tar.gz"
    echo "    EMNIST    -> ${DATA}/EMNIST/raw/gzip.zip"
    exit 1
fi
