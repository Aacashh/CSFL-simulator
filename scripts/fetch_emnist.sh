#!/usr/bin/env bash
# =============================================================================
# Pre-fetch EMNIST raw archive into <repo>/data/EMNIST/raw/.
#
# WHY THIS EXISTS
# ---------------
# torchvision's EMNIST mirror has historically lived at biometrics.nist.gov
# (and rds.westclintech.com / cloudfront mirrors in newer torchvision releases).
# These are far more reliable from Indian academic networks than the Japanese
# codh.rois.ac.jp mirror that broke for KMNIST, but the same defensive pattern
# applies: pre-fetch with retries before launching a 100-round paired run, so
# a network blip fails in seconds instead of after model setup.
#
# Behaviour:
#   1. Skip work if the EMNIST gzip archive is already extracted on disk.
#   2. Try torchvision's downloader (uses urllib).
#   3. On total failure, print a manual-recovery workflow that mirrors the
#      KMNIST one — download-on-Windows, scp up, re-run with --resume --exp 2.
#
# Idempotent: safe to run any number of times.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
RAW_DIR="${DATA_DIR}/EMNIST/raw"

# Files torchvision unpacks from the EMNIST archive (digits split).
# We treat them as the integrity signal that "EMNIST is on disk".
EXPECTED_FILES=(
    emnist-digits-train-images-idx3-ubyte
    emnist-digits-train-labels-idx1-ubyte
    emnist-digits-test-images-idx3-ubyte
    emnist-digits-test-labels-idx1-ubyte
)

# ---- Step 1: pre-check ------------------------------------------------------
all_present=true
for f in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "${RAW_DIR}/${f}" && ! -f "${RAW_DIR}/${f}.gz" ]]; then
        all_present=false
        break
    fi
done
if $all_present; then
    echo "EMNIST: digits split already present at ${RAW_DIR} — skipping fetch."
    exit 0
fi

mkdir -p "${RAW_DIR}"
echo "EMNIST: missing files in ${RAW_DIR}"

# ---- Step 2: torchvision download (urllib) ----------------------------------
echo "EMNIST: attempting torchvision download (NIST mirror) ..."
if python - <<PY
import sys
from torchvision import datasets
try:
    datasets.EMNIST(r"${DATA_DIR}", split="digits", train=True,  download=True)
    datasets.EMNIST(r"${DATA_DIR}", split="digits", train=False, download=True)
except Exception as e:
    print(f"torchvision: {e}", file=sys.stderr)
    sys.exit(1)
PY
then
    echo "EMNIST: torchvision download succeeded."
    exit 0
fi

# ---- Step 3: print manual recovery instructions -----------------------------
cat >&2 <<EOF

==============================================================================
EMNIST automatic fetch FAILED.
The torchvision EMNIST mirror appears unreachable from this host.

MANUAL FALLBACK — run on a machine with working internet:

  python -c "from torchvision import datasets; \\
      datasets.EMNIST('./data', split='digits', train=True,  download=True); \\
      datasets.EMNIST('./data', split='digits', train=False, download=True)"

Then SCP the EMNIST raw directory up:

  scp -r ./data/EMNIST \\
      <user>@<this-host>:${DATA_DIR}/

After placing the archive, re-run:
  bash scripts/run_scope_submission_experiments.sh --resume --exp 2
==============================================================================
EOF
exit 1
