#!/usr/bin/env bash
# =============================================================================
# Prepare CSFL-simulator for air-gapped HPC deployment
#
# Downloads all datasets, verifies integrity, and optionally creates a
# portable tarball of the data/ directory for transfer.
#
# Usage (run on a machine WITH internet):
#   bash scripts/prepare_hpc.sh              # Download + verify
#   bash scripts/prepare_hpc.sh --pack       # Download + verify + create tarball
#   bash scripts/prepare_hpc.sh --verify     # Verify only (no download)
#   bash scripts/prepare_hpc.sh --check-code # Also scan code for runtime downloads
#
# On the HPC, after transferring:
#   tar xf csfl_data.tar.gz -C /path/to/CSFL-simulator/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
ART_DIR="${REPO_ROOT}/artifacts"

PACK=false
VERIFY_ONLY=false
CHECK_CODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pack)       PACK=true; shift ;;
        --verify)     VERIFY_ONLY=true; shift ;;
        --check-code) CHECK_CODE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo ""; echo "===== [$(date '+%H:%M:%S')] $1 ====="; }
ok()  { echo "  [OK]   $1"; }
fail() { echo "  [FAIL] $1"; ERRORS=$((ERRORS + 1)); }

ERRORS=0

# =============================================================================
# Step 1: Download all datasets
# =============================================================================
if ! $VERIFY_ONLY; then
    log "Downloading all datasets to ${DATA_DIR}"

    python "${SCRIPT_DIR}/download_data.py" --root "${DATA_DIR}" --all

    echo ""
    echo "Downloads complete."
fi

# =============================================================================
# Step 2: Verify dataset integrity
# =============================================================================
log "Verifying datasets"

# --- MNIST ---
MNIST_DIR="${DATA_DIR}/MNIST/raw"
if [[ -d "$MNIST_DIR" ]]; then
    count=$(find "$MNIST_DIR" -type f | wc -l)
    if [[ $count -ge 4 ]]; then ok "MNIST (${count} files)"; else fail "MNIST: expected >= 4 files, found ${count}"; fi
else
    fail "MNIST: directory ${MNIST_DIR} missing"
fi

# --- Fashion-MNIST ---
FMNIST_DIR="${DATA_DIR}/FashionMNIST/raw"
if [[ -d "$FMNIST_DIR" ]]; then
    count=$(find "$FMNIST_DIR" -type f | wc -l)
    if [[ $count -ge 4 ]]; then ok "Fashion-MNIST (${count} files)"; else fail "Fashion-MNIST: expected >= 4 files, found ${count}"; fi
else
    fail "Fashion-MNIST: directory ${FMNIST_DIR} missing"
fi

# --- CIFAR-10 ---
CIFAR10_DIR="${DATA_DIR}/cifar-10-batches-py"
if [[ -d "$CIFAR10_DIR" ]]; then
    # Should have data_batch_1..5, test_batch, batches.meta = 7 files
    count=$(find "$CIFAR10_DIR" -maxdepth 1 -type f | wc -l)
    if [[ $count -ge 7 ]]; then ok "CIFAR-10 (${count} files)"; else fail "CIFAR-10: expected >= 7 files, found ${count}"; fi
else
    fail "CIFAR-10: directory ${CIFAR10_DIR} missing"
fi

# --- CIFAR-100 ---
CIFAR100_DIR="${DATA_DIR}/cifar-100-python"
if [[ -d "$CIFAR100_DIR" ]]; then
    count=$(find "$CIFAR100_DIR" -maxdepth 1 -type f | wc -l)
    if [[ $count -ge 3 ]]; then ok "CIFAR-100 (${count} files)"; else fail "CIFAR-100: expected >= 3 files, found ${count}"; fi
else
    fail "CIFAR-100: directory ${CIFAR100_DIR} missing"
fi

# --- STL-10 (only needed for FD experiments, but download anyway) ---
STL10_DIR="${DATA_DIR}/stl10_binary"
if [[ -d "$STL10_DIR" ]]; then
    count=$(find "$STL10_DIR" -type f | wc -l)
    if [[ $count -ge 5 ]]; then ok "STL-10 (${count} files)"; else fail "STL-10: expected >= 5 files, found ${count}"; fi
else
    fail "STL-10: directory ${STL10_DIR} missing"
fi

# --- Quick load test via Python ---
log "Running Python load test (no download flag)"
python -c "
import sys
from torchvision import datasets, transforms
root = '${DATA_DIR}'
t = transforms.ToTensor()
errors = []
for name, cls, kwargs in [
    ('MNIST',         datasets.MNIST,        {'train': True}),
    ('Fashion-MNIST', datasets.FashionMNIST, {'train': True}),
    ('CIFAR-10',      datasets.CIFAR10,      {'train': True}),
    ('CIFAR-100',     datasets.CIFAR100,     {'train': True}),
    ('STL-10',        datasets.STL10,        {'split': 'train'}),
]:
    try:
        ds = cls(root, download=False, transform=t, **kwargs)
        print(f'  [OK]   {name}: {len(ds)} samples')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        errors.append(name)
if errors:
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))

# =============================================================================
# Step 3: Verify directory structure
# =============================================================================
log "Verifying artifact directories"
mkdir -p "${ART_DIR}/runs" "${ART_DIR}/checkpoints" "${ART_DIR}/exports"
ok "artifacts/runs/"
ok "artifacts/checkpoints/"
ok "artifacts/exports/"

# =============================================================================
# Step 4: Disk usage summary
# =============================================================================
log "Disk usage"
echo ""
du -sh "${DATA_DIR}"/MNIST 2>/dev/null || true
du -sh "${DATA_DIR}"/FashionMNIST 2>/dev/null || true
du -sh "${DATA_DIR}"/cifar-10-batches-py 2>/dev/null || true
du -sh "${DATA_DIR}"/cifar-100-python 2>/dev/null || true
du -sh "${DATA_DIR}"/stl10_binary 2>/dev/null || true
echo "-----"
du -sh "${DATA_DIR}" 2>/dev/null || true

# =============================================================================
# Step 5: (Optional) Scan code for hidden runtime downloads
# =============================================================================
if $CHECK_CODE; then
    log "Scanning code for runtime download calls"

    echo "  Checking for download=True in Python files..."
    grep -rn "download=True" "${REPO_ROOT}/csfl_simulator/" --include="*.py" | while read -r line; do
        echo "    $line"
    done

    echo ""
    echo "  Checking for pretrained/weights in model code..."
    grep -rn "pretrained\|weights=" "${REPO_ROOT}/csfl_simulator/core/models.py" | while read -r line; do
        echo "    $line"
    done

    echo ""
    echo "  Checking for urllib/requests/wget calls..."
    grep -rn "urllib\|requests\.get\|wget\|urlretrieve" "${REPO_ROOT}/csfl_simulator/" --include="*.py" | while read -r line; do
        echo "    $line"
    done || ok "No direct URL fetching found"
fi

# =============================================================================
# Step 6: (Optional) Pack into tarball for transfer
# =============================================================================
if $PACK; then
    log "Packing data for HPC transfer"
    TARBALL="${REPO_ROOT}/csfl_data.tar.gz"
    echo "  Creating ${TARBALL}..."
    tar czf "${TARBALL}" -C "${REPO_ROOT}" data/
    TAR_SIZE=$(du -sh "${TARBALL}" | cut -f1)
    ok "Tarball created: ${TARBALL} (${TAR_SIZE})"
    echo ""
    echo "  To deploy on HPC:"
    echo "    scp csfl_data.tar.gz hpc:/path/to/CSFL-simulator/"
    echo "    cd /path/to/CSFL-simulator && tar xf csfl_data.tar.gz"
fi

# =============================================================================
# Summary
# =============================================================================
log "Summary"
if [[ $ERRORS -eq 0 ]]; then
    echo "  All checks passed. Ready for HPC deployment."
    echo ""
    echo "  Transfer checklist:"
    echo "    1. Copy the entire CSFL-simulator/ directory (or use --pack for data-only tarball)"
    echo "    2. Ensure Python + pip packages are installed on HPC (torch, torchvision, numpy, etc.)"
    echo "    3. Run:  bash scripts/prepare_hpc.sh --verify   (on HPC, to confirm data is intact)"
    echo "    4. Run:  bash scripts/run_apex_v2_experiments.sh --exp 1"
else
    echo "  ${ERRORS} error(s) found. Fix issues above before deploying."
    exit 1
fi

echo ""
