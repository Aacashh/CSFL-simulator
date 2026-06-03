[CmdletBinding()]
param(
    [switch] $VerifiedHardwareTelemetry,
    [switch] $Foreground,
    [string] $TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$VenvDir = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$CampaignScript = Join-Path $ScriptDir "run_full_windows_campaign.ps1"
$RunsDir = Join-Path $RepoRoot "runs\maml_select"
$LogDir = Join-Path $RunsDir "logs"
$PidFile = Join-Path $LogDir "campaign_windows.pid"

function Write-Header {
    param([string] $Message)
    Write-Host ""
    Write-Host "==============================================================="
    Write-Host "  $Message"
    Write-Host "==============================================================="
}

function Write-Step {
    param([string] $Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Invoke-VenvPython {
    param([string[]] $Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Set-Location $RepoRoot

Write-Header "MAML-Select Windows Experiment Setup"
Write-Host "  Repo:   $RepoRoot"
Write-Host "  venv:   $VenvDir"
Write-Host "  Script: $CampaignScript"

Write-Header "Step 1/5: Python Virtual Environment"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "  Creating virtual environment..."
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
        throw "Failed to create the Python virtual environment."
    }
}
Write-Step "Virtual environment ready: $PythonExe"

Write-Header "Step 2/5: Installing Dependencies"
Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q")
Invoke-VenvPython @("-m", "pip", "install", "-e", ".", "-q")
Invoke-VenvPython @("-m", "pip", "install", "-r", "csfl_simulator\experiments\maml_select\requirements.txt", "-q")
if ($TorchIndexUrl) {
    Write-Host "  Installing PyTorch from the requested package index..."
    Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", $TorchIndexUrl, "-q")
}
Write-Step "Repository and experiment dependencies installed"

Write-Header "Step 3/5: Downloading Datasets"
Write-Host "  Downloading Fashion-MNIST and CIFAR-10..."
& $PythonExe "scripts\download_data.py" "--datasets" "fashion-mnist" "cifar10"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Dataset pre-download failed. Torchvision will retry during the first run."
}
else {
    Write-Step "Datasets ready"
}

Write-Header "Step 4/5: GPU Verification"
$NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    & $NvidiaSmi.Source "--query-gpu=index,name,memory.total,driver_version" "--format=csv,noheader"
    Write-Step "NVIDIA GPU detected"
}
else {
    Write-Warning "nvidia-smi.exe was not found. Experiments will run on CPU and will be much slower."
}

$TorchCheck = @'
import torch
print(f"  PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
'@
Invoke-VenvPython @("-c", $TorchCheck)

$CodeCarbonCheck = @'
try:
    import codecarbon
    print(f"  CodeCarbon {codecarbon.__version__} - energy tracking enabled")
except ImportError:
    print("  [WARN] CodeCarbon not installed. Energy will be modeled only.")
'@
Invoke-VenvPython @("-c", $CodeCarbonCheck)

Write-Header "Step 5/5: Quick Validation"
Write-Host "  Running dry-run to verify the experiment matrix..."
Invoke-VenvPython @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "quick",
    "--device", "auto",
    "--dry-run"
)
Write-Step "Dry-run passed"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if ($VerifiedHardwareTelemetry) {
    $env:VERIFIED_HARDWARE_TELEMETRY = "1"
}
elseif (-not $env:VERIFIED_HARDWARE_TELEMETRY) {
    $env:VERIFIED_HARDWARE_TELEMETRY = "0"
}
$env:PYTHON_BIN = $PythonExe

Write-Header "Ready to Launch Full Campaign"
Write-Host ""
Write-Host "  Campaign: 236 unique 200-round matrix runs, 32 short overhead runs, and 3 quick checks"
Write-Host "  Repeated evidence uses seeds 42, 123, and 2026"
Write-Host "  The campaign is resumable. Re-run this setup command after an interruption."
Write-Host "  Results: $RunsDir"
Write-Host ""

if (Test-Path -LiteralPath $PidFile) {
    $ExistingProcessId = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($ExistingProcessId -and (Get-Process -Id $ExistingProcessId -ErrorAction SilentlyContinue)) {
        Write-Warning "A Windows campaign is already running with PID $ExistingProcessId."
        Write-Host "  Monitor it with:"
        Write-Host "    .\csfl_simulator\experiments\maml_select\show_windows_campaign.cmd"
        exit 0
    }
}

$Answer = Read-Host "  Start the campaign now? [Y/n]"
if ($Answer -match "^[Nn]") {
    Write-Host "  Launch later with:"
    Write-Host "    .\csfl_simulator\experiments\maml_select\setup_and_run_windows.cmd"
    exit 0
}

$PowerShellExe = (Get-Process -Id $PID).Path
if ($Foreground) {
    & $PowerShellExe "-NoProfile" "-ExecutionPolicy" "Bypass" "-File" $CampaignScript
    exit $LASTEXITCODE
}

$ArgumentList = "-NoProfile -ExecutionPolicy Bypass -File `"$CampaignScript`""
$CampaignProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $RepoRoot `
    -PassThru
Set-Content -LiteralPath $PidFile -Value $CampaignProcess.Id -Encoding ascii

Write-Step "Campaign launched in a separate PowerShell process (PID: $($CampaignProcess.Id))"
Write-Host ""
Write-Host "  Monitor progress with:"
Write-Host "    .\csfl_simulator\experiments\maml_select\show_windows_campaign.cmd"
Write-Host ""
Write-Host "  Follow the latest log with:"
Write-Host "    .\csfl_simulator\experiments\maml_select\show_windows_campaign.cmd -Follow"
