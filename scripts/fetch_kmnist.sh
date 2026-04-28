#!/usr/bin/env bash
# =============================================================================
# Pre-fetch KMNIST IDX files into <repo>/data/KMNIST/raw/.
#
# WHY THIS EXISTS
# ---------------
# torchvision's default KMNIST mirror (codh.rois.ac.jp) is frequently
# unreachable from HPC clusters and academic networks behind restrictive
# firewalls — urllib has no retry and times out after ~30s, taking down
# any run that needs KMNIST. This script:
#
#   1. Skips work if the 4 raw files are already present.
#   2. Tries torchvision's downloader (uses urllib).
#   3. Falls back to curl with longer timeouts + retries (different stack;
#      sometimes succeeds where urllib fails — different default proxy
#      handling, IPv4-first behavior, etc.).
#   4. On total failure, prints an actionable manual-recovery workflow.
#
# Idempotent: safe to run any number of times.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
RAW_DIR="${DATA_DIR}/KMNIST/raw"
MIRROR_URL="${KMNIST_MIRROR:-http://codh.rois.ac.jp/kmnist/dataset/kmnist}"

FILES=(
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
)

# ---- Step 1: pre-check ------------------------------------------------------
all_present=true
for f in "${FILES[@]}"; do
    [[ -f "${RAW_DIR}/${f}" ]] || { all_present=false; break; }
done
if $all_present; then
    echo "KMNIST: all 4 raw files present at ${RAW_DIR} — skipping fetch."
    exit 0
fi

mkdir -p "${RAW_DIR}"
echo "KMNIST: missing files in ${RAW_DIR}"

# ---- Step 2: torchvision download (urllib) ----------------------------------
echo "KMNIST: attempting torchvision download from ${MIRROR_URL} ..."
if python - <<PY
import sys
from torchvision import datasets
try:
    datasets.KMNIST(r"${DATA_DIR}", train=True,  download=True)
    datasets.KMNIST(r"${DATA_DIR}", train=False, download=True)
except Exception as e:
    print(f"torchvision: {e}", file=sys.stderr)
    sys.exit(1)
PY
then
    echo "KMNIST: torchvision download succeeded."
    exit 0
fi
echo "KMNIST: torchvision download failed; falling back to curl..."

# ---- Step 3: curl fallback --------------------------------------------------
curl_ok=true
for f in "${FILES[@]}"; do
    if [[ -f "${RAW_DIR}/${f}" ]]; then
        echo "  cached: ${f}"
        continue
    fi
    tmp="${RAW_DIR}/${f}.partial"
    if curl -fSL --connect-timeout 30 --max-time 600 \
            --retry 5 --retry-delay 10 --retry-connrefused \
            -o "${tmp}" "${MIRROR_URL}/${f}"; then
        mv "${tmp}" "${RAW_DIR}/${f}"
        echo "  fetched: ${f}"
    else
        rm -f "${tmp}"
        echo "  FAILED: ${f}" >&2
        curl_ok=false
        break
    fi
done

if $curl_ok; then
    # Re-validate with torchvision (this will MD5-check the gz files).
    if python - <<PY
from torchvision import datasets
datasets.KMNIST(r"${DATA_DIR}", train=True,  download=False)
datasets.KMNIST(r"${DATA_DIR}", train=False, download=False)
PY
    then
        echo "KMNIST: curl fetch + integrity check OK."
        exit 0
    else
        echo "KMNIST: curl-fetched files failed integrity check." >&2
    fi
fi

# ---- Step 4: print manual recovery instructions -----------------------------
cat >&2 <<EOF

==============================================================================
KMNIST automatic fetch FAILED.
The official mirror (codh.rois.ac.jp) appears unreachable from this host.

MANUAL FALLBACK — run these on a machine with working internet:

  python -c "from torchvision import datasets; \\
      datasets.KMNIST('./data', train=True,  download=True); \\
      datasets.KMNIST('./data', train=False, download=True)"

Then SCP the 4 .gz files to this host:

  scp ./data/KMNIST/raw/*.gz \\
      <user>@<this-host>:${RAW_DIR}/

Files required (gz form, NOT extracted):
  - train-images-idx3-ubyte.gz
  - train-labels-idx1-ubyte.gz
  - t10k-images-idx3-ubyte.gz
  - t10k-labels-idx1-ubyte.gz

Once placed, re-run:
  bash scripts/run_scope_submission_experiments.sh --resume --exp 2
==============================================================================
EOF
exit 1
