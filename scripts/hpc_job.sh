#!/bin/bash
#SBATCH --job-name=csfl-apex-fd
#SBATCH --output=logs/csfl_%j.out
#SBATCH --error=logs/csfl_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aakashsphy21@itbhu.ac.in

# =============================================================================
# CSFL Simulator — HPC Job Script
#
# Usage:
#   sbatch scripts/hpc_job.sh                    # Run all experiments
#   sbatch scripts/hpc_job.sh --exp apex         # APEX only
#   sbatch scripts/hpc_job.sh --exp fd           # FD only
#   sbatch scripts/hpc_job.sh --exp apex:1       # APEX Exp 1 only
#   sbatch scripts/hpc_job.sh --resume           # Skip completed runs
#   sbatch scripts/hpc_job.sh --fast             # Debug with fast mode
#
# Before first submission:
#   mkdir -p logs
#   bash scripts/prepare_hpc.sh --verify
# =============================================================================
set -euo pipefail

echo "=============================================="
echo "  CSFL Simulator — HPC Job"
echo "  Job ID:    ${SLURM_JOB_ID:-local}"
echo "  Node:      $(hostname)"
echo "  Started:   $(date)"
echo "=============================================="

# ---------- Environment Setup -------------------------------------------------

# Show what's available
echo ""
echo "--- Available CUDA modules ---"
module avail cuda 2>&1 | head -20
echo ""
echo "--- Available Python modules ---"
module avail python 2>&1 | head -20
echo ""
echo "--- Available GCC/compiler modules ---"
module avail gcc 2>&1 | head -10
echo ""

# Load required modules (adjust versions to match your cluster)
module purge
module load cuda/11.8       2>/dev/null || module load cuda          2>/dev/null || echo "WARN: No CUDA module found"
module load cudnn/8.6       2>/dev/null || module load cudnn         2>/dev/null || echo "WARN: No cuDNN module found"
module load gcc/11.3        2>/dev/null || module load gcc           2>/dev/null || echo "WARN: No GCC module found"
module load python/3.11     2>/dev/null || module load python/3.10   2>/dev/null || \
module load python/3.9      2>/dev/null || module load python        2>/dev/null || echo "WARN: No Python module found"

echo ""
echo "--- Loaded modules ---"
module list 2>&1
echo ""

# Activate virtual environment
source ~/csfl-env/bin/activate
echo "Python:  $(which python) ($(python --version 2>&1))"
echo "Pip:     $(pip --version 2>&1 | head -1)"
echo ""

# Verify torch + CUDA
python -c "
import torch
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.cuda.is_available()} (v{torch.version.cuda})')
if torch.cuda.is_available():
    print(f'GPU:       {torch.cuda.get_device_name(0)}')
    print(f'GPU Mem:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
echo ""

# Fix the deterministic CuBLAS warning (from the error you hit)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Prevent torch from trying to download anything
export TORCH_HOME="${HOME}/.cache/torch"
export HF_DATASETS_OFFLINE=1

# Performance tuning
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---------- Change to project directory ---------------------------------------
cd ~/CSFL-simulator
echo "Workdir: $(pwd)"
echo ""

# ---------- Verify data is available offline ----------------------------------
echo "--- Verifying datasets ---"
python -c "
from torchvision import datasets, transforms
import sys
root = 'data'
t = transforms.ToTensor()
ok = True
for name, cls, kw in [
    ('MNIST',         datasets.MNIST,        {'train': True}),
    ('Fashion-MNIST', datasets.FashionMNIST, {'train': True}),
    ('CIFAR-10',      datasets.CIFAR10,      {'train': True}),
    ('CIFAR-100',     datasets.CIFAR100,     {'train': True}),
    ('STL-10',        datasets.STL10,        {'split': 'train'}),
]:
    try:
        ds = cls(root, download=False, transform=t, **kw)
        print(f'  [OK]   {name}: {len(ds)} samples')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        ok = False
if not ok:
    print('ERROR: Missing datasets. Run prepare_hpc.sh on a machine with internet.')
    sys.exit(1)
"
echo ""

# ---------- Run experiments ---------------------------------------------------
echo "=============================================="
echo "  Starting experiments"
echo "  Args: $@"
echo "=============================================="
echo ""

bash scripts/run_all_experiments.sh --resume "$@"

# ---------- Done --------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Job complete: $(date)"
echo "  Results in: artifacts/runs/"
echo "=============================================="

# List results
python -m csfl_simulator list-runs 2>/dev/null || true
