#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# ONE-COMMAND SETUP & RUN for GPU PC
# ═══════════════════════════════════════════════════════════════════════════════
#
# After git pull on the GPU PC, just run:
#
#   chmod +x csfl_simulator/experiments/maml_select/setup_and_run.sh
#   bash csfl_simulator/experiments/maml_select/setup_and_run.sh
#
# This script will:
#   1. Create a Python virtual environment (if not exists)
#   2. Install the repo + all experiment dependencies
#   3. Download datasets (Fashion-MNIST, CIFAR-10, CIFAR-100)
#   4. Verify GPU availability
#   5. Launch the full experiment campaign inside tmux (or nohup)
#
# The entire campaign is resumable. If interrupted, just re-run this script.
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
CAMPAIGN_SCRIPT="${SCRIPT_DIR}/run_full_gpu_campaign.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $*${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $*"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
}

print_error() {
    echo -e "${RED}[✗]${NC} $*"
}

cd "${REPO_ROOT}"

print_header "MAML-Select Experiment Setup"
echo "  Repo:   ${REPO_ROOT}"
echo "  venv:   ${VENV_DIR}"
echo "  Script: ${CAMPAIGN_SCRIPT}"
echo ""

# ── Step 1: Python virtual environment ───────────────────────────────────────

print_header "Step 1/5: Python Virtual Environment"

if [[ -d "${VENV_DIR}" ]]; then
    print_step "Virtual environment already exists at ${VENV_DIR}"
else
    echo "  Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    print_step "Created virtual environment"
fi

# Activate
source "${VENV_DIR}/bin/activate"
print_step "Activated: $(python --version) at $(which python)"

# ── Step 2: Install dependencies ─────────────────────────────────────────────

print_header "Step 2/5: Installing Dependencies"

echo "  Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

echo "  Installing repo in editable mode..."
pip install -e . -q
print_step "Installed csfl_simulator"

echo "  Installing experiment requirements..."
pip install -r csfl_simulator/experiments/maml_select/requirements.txt -q
print_step "Installed experiment dependencies"

# Install PyTorch with CUDA if not already present
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    print_warn "PyTorch CUDA not available. Attempting to install..."
    echo "  Detecting CUDA version..."
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
        if [[ "${CUDA_MAJOR}" -ge 12 ]]; then
            echo "  Installing PyTorch for CUDA 12.x..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
        elif [[ "${CUDA_MAJOR}" -ge 11 ]]; then
            echo "  Installing PyTorch for CUDA 11.8..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
        else
            print_warn "CUDA ${CUDA_VER} detected. Installing default PyTorch..."
            pip install torch torchvision -q
        fi
    else
        print_warn "nvidia-smi not found. Installing CPU PyTorch..."
        pip install torch torchvision -q
    fi
}

# Verify PyTorch
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
print_step "PyTorch verified"

# ── Step 3: Download datasets ────────────────────────────────────────────────

print_header "Step 3/5: Downloading Datasets"

if [[ -f "scripts/download_data.py" ]]; then
    echo "  Downloading Fashion-MNIST, CIFAR-10, CIFAR-100..."
    python scripts/download_data.py --datasets fashion-mnist cifar10 cifar100 || {
        print_warn "download_data.py failed. Datasets will be downloaded on first run by torchvision."
    }
else
    echo "  No download script found. Downloading via torchvision..."
    python -c "
import torchvision
import torchvision.transforms as T
print('  Downloading Fashion-MNIST...')
torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
print('  Downloading CIFAR-10...')
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('  Downloading CIFAR-100...')
torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
"
fi
print_step "Datasets ready"

# ── Step 4: GPU verification ────────────────────────────────────────────────

print_header "Step 4/5: GPU Verification"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    print_step "NVIDIA GPU detected"
else
    print_warn "No NVIDIA GPU detected. Experiments will run on CPU (much slower)."
fi

# CodeCarbon check
python -c "
try:
    import codecarbon
    print(f'  CodeCarbon {codecarbon.__version__} — energy tracking enabled')
except ImportError:
    print('  [WARN] CodeCarbon not installed. Energy will be modeled only.')
"

# ── Step 5: Quick dry-run validation ─────────────────────────────────────────

print_header "Step 5/5: Quick Validation"

echo "  Running dry-run to verify experiment matrix..."
python -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile quick --device auto --dry-run
print_step "Dry-run passed"

# ── Launch Campaign ──────────────────────────────────────────────────────────

print_header "Ready to Launch Full Campaign"
echo ""
echo -e "  ${BOLD}Estimated total: ~230 simulation runs across 7 phases${NC}"
echo "  All runs are 200 rounds with 3 seeds (42, 123, 2026)"
echo "  The campaign is RESUMABLE — re-run if interrupted"
echo ""
echo "  Results will be saved to:"
echo "    ${REPO_ROOT}/runs/maml_select/"
echo ""

# Check if tmux is available
if command -v tmux &>/dev/null; then
    echo -e "  ${CYAN}tmux detected — campaign will run in a tmux session${NC}"
    echo ""
    read -p "  Start the campaign now? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "  To start manually later:"
        echo "    source ${VENV_DIR}/bin/activate"
        echo "    tmux new -s maml"
        echo "    bash ${CAMPAIGN_SCRIPT}"
        exit 0
    fi

    # Launch in tmux
    tmux new-session -d -s maml \
        "source ${VENV_DIR}/bin/activate && bash ${CAMPAIGN_SCRIPT} 2>&1 | tee ${REPO_ROOT}/runs/maml_select/logs/campaign_stdout.log; echo 'Campaign finished. Press Enter to close.'; read"
    
    echo ""
    print_step "Campaign launched in tmux session 'maml'"
    echo ""
    echo "  To monitor progress:"
    echo "    tmux attach -t maml"
    echo ""
    echo "  To detach (leave running): Ctrl+B then D"
    echo ""
    echo "  To check progress without attaching:"
    echo "    find ${REPO_ROOT}/runs/maml_select -name 'progress.json' -newer /tmp/.campaign_start 2>/dev/null | wc -l"
    echo ""
else
    echo -e "  ${YELLOW}tmux not found — using nohup instead${NC}"
    echo ""
    read -p "  Start the campaign now? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "  To start manually later:"
        echo "    source ${VENV_DIR}/bin/activate"
        echo "    nohup bash ${CAMPAIGN_SCRIPT} > campaign.log 2>&1 &"
        exit 0
    fi

    NOHUP_LOG="${REPO_ROOT}/runs/maml_select/logs/campaign_nohup.log"
    nohup bash -c "source ${VENV_DIR}/bin/activate && bash ${CAMPAIGN_SCRIPT}" > "${NOHUP_LOG}" 2>&1 &
    CAMPAIGN_PID=$!
    echo "${CAMPAIGN_PID}" > "${REPO_ROOT}/runs/maml_select/logs/campaign.pid"

    echo ""
    print_step "Campaign launched in background (PID: ${CAMPAIGN_PID})"
    echo ""
    echo "  To monitor progress:"
    echo "    tail -f ${NOHUP_LOG}"
    echo ""
    echo "  To check if still running:"
    echo "    ps aux | grep ${CAMPAIGN_PID}"
    echo ""
fi

echo "  ═══════════════════════════════════════════════════════════════"
echo "  After the campaign completes, find all results at:"
echo "    ${REPO_ROOT}/runs/maml_select/"
echo ""
echo "  Key files to collect:"
echo "    analysis/paired_significance_tests.csv   (artifacts/)"
echo "    analysis/main_summary.csv                  (artifacts/)"
echo "    analysis/energy_to_target_summary.csv       (artifacts/)"
echo "    plots/*.eps                                 (artifacts/)"
echo "    sensitivity/sensitivity_summary.csv          (runs/)"
echo "    ablation/ablation_summary.csv                (runs/)"
echo "    scaling/scaling_pivot.csv                    (runs/)"
echo "  ═══════════════════════════════════════════════════════════════"
