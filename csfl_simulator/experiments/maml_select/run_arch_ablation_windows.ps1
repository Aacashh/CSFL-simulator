[CmdletBinding()]
param(
    [string] $Device = "auto",
    [string] $TorchIndexUrl = "",
    [string] $PythonExe = "",
    [switch] $AllowCpu,
    [switch] $NoPerformanceMode
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$VenvDir = Join-Path $RepoRoot ".venv"
if (-not $PythonExe) {
    $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
}
$RunsDir = Join-Path $RepoRoot "runs\maml_select_arch_ablation"
$ArtifactsDir = Join-Path $RepoRoot "artifacts\maml_select\arch_ablation"

function Write-Header {
    param([string] $Message)
    Write-Host ""
    Write-Host "==============================================================="
    Write-Host "  $Message"
    Write-Host "==============================================================="
}

function Invoke-StepPython {
    param([string[]] $Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Set-Location $RepoRoot

Write-Header "MAML-Select Architecture (Selector-Capacity) Ablation"
Write-Host "Repo:      $RepoRoot"
Write-Host "Python:    $PythonExe"
Write-Host "Device:    $Device"
Write-Host "Runs:      $RunsDir"
Write-Host "Artifacts: $ArtifactsDir"
Write-Host ""
Write-Host "Workload: Fashion-MNIST, 12 resumable 200-round runs"
Write-Host "  - hidden_dim in {16, 32, 64, 128} x seeds 42/123/2026"

$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "expandable_segments:True" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "4" }

Write-Header "Step 1/5: Python Environment"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating virtual environment at $VenvDir"
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py.exe -3 -m venv $VenvDir
    }
    elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        & python.exe -m venv $VenvDir
    }
    else {
        throw "Python 3 was not found. Install Python 3 and ensure py.exe or python.exe is on PATH."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
}

Write-Header "Step 2/5: Dependencies"
Invoke-StepPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q")
Invoke-StepPython @("-m", "pip", "install", "-e", ".", "-q")
Invoke-StepPython @("-m", "pip", "install", "-r", "csfl_simulator\experiments\maml_select\requirements.txt", "-q")
if ($TorchIndexUrl) {
    Write-Host "Installing PyTorch from requested index: $TorchIndexUrl"
    Invoke-StepPython @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", $TorchIndexUrl, "-q")
}

Write-Header "Step 3/5: Dataset and Device Check"
& $PythonExe "scripts\download_data.py" "--datasets" "fashion-mnist"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Dataset pre-download failed. Torchvision will retry during the first run."
}

$TorchCheck = @'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
'@
Invoke-StepPython @("-c", $TorchCheck)

$DeviceCheck = @"
import os
import sys
import torch
requested = "$Device"
resolved = requested
if requested == "auto":
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
if resolved == "cuda" and not torch.cuda.is_available():
    print("Requested CUDA, but CUDA is not available.", file=sys.stderr)
    sys.exit(2)
print(resolved)
"@
$ResolvedDevice = (& $PythonExe "-c" $DeviceCheck).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Device check failed."
}
Write-Host "Resolved device: $ResolvedDevice"
if (($ResolvedDevice -eq "cpu") -and (-not $AllowCpu)) {
    throw "Refusing to start 12 runs on CPU. Re-run with -Device cuda on the GPU laptop, or add -AllowCpu to override intentionally."
}

New-Item -ItemType Directory -Force -Path (Join-Path $RunsDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ArtifactsDir "analysis") | Out-Null

# CUDA fast path: cuDNN benchmark + TF32 (default on). Use -NoPerformanceMode for strict determinism.
$RunArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "arch_ablation",
    "--device", $Device,
    "--output-dir", $RunsDir,
    "--analysis-dir", (Join-Path $ArtifactsDir "analysis"),
    "--no-hardware-meter",
    "--resume"
)
if (-not $NoPerformanceMode) { $RunArgs += "--performance-mode" }

Write-Header "Step 4/5: Dry-Run Matrix Check"
Invoke-StepPython @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "arch_ablation",
    "--device", $Device,
    "--output-dir", $RunsDir,
    "--analysis-dir", (Join-Path $ArtifactsDir "analysis"),
    "--dry-run"
)

Write-Header "Step 5/5: Run Architecture Ablation (performance_mode=$([int](-not $NoPerformanceMode)))"
Invoke-StepPython $RunArgs

Write-Header "Summarizing Outputs"
Invoke-StepPython @(
    "-m", "csfl_simulator.experiments.maml_select.summarize_arch_ablation",
    "--runs-root", $RunsDir,
    "--output-dir", $ArtifactsDir
)

Write-Host ""
Write-Host "Done."
Write-Host "Results:  $RunsDir"
Write-Host "Analysis: $(Join-Path $ArtifactsDir 'analysis')"
Write-Host "Tables:   $(Join-Path $ArtifactsDir 'tables')"
