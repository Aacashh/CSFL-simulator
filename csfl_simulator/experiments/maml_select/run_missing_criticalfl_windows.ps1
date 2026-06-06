[CmdletBinding()]
param(
    [string] $Device = "cuda",
    [string] $TorchIndexUrl = "",
    [switch] $NoInstall,
    [switch] $AllowCpuFallback,
    [switch] $EnableHardwareMeter,
    [string] $CountryIso = "IND",
    [string] $GridIntensity = "475"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$VenvDir = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RunsDir = Join-Path $RepoRoot "runs\maml_select"
$ArtifactsDir = Join-Path $RepoRoot "artifacts\maml_select"
$AnalysisDir = Join-Path $ArtifactsDir "analysis"
$LogDir = Join-Path $RunsDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "missing_criticalfl_windows_$Timestamp.log"
$TargetConfig = Join-Path $LogDir "missing_criticalfl_config.yaml"
$PidFile = Join-Path $LogDir "missing_criticalfl_windows.pid"
$RunStart = Get-Date

$TargetRunLabels = @(
    "main_benchmarks_cifar10_main_criticalfl_s42",
    "main_benchmarks_cifar10_main_criticalfl_s123",
    "main_benchmarks_fashion_main_criticalfl_s123"
)

function Write-Log {
    param([string] $Message)
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $Line
    Add-Content -LiteralPath $LogFile -Value $Line -Encoding utf8
}

function Write-Section {
    param([string] $Message)
    Write-Log ""
    Write-Log "============================================================="
    Write-Log "  $Message"
    Write-Log "============================================================="
}

function Get-Elapsed {
    param([datetime] $Started)
    $Elapsed = (Get-Date) - $Started
    return "{0:00}:{1:00}:{2:00}" -f [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds
}

function Invoke-Python {
    param(
        [string] $Name,
        [string[]] $Arguments
    )
    $Started = Get-Date
    Write-Log "START: $Name"
    Write-Log "  cmd: $PythonExe $($Arguments -join ' ')"
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $LogFile -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }
    Write-Log "OK: $Name [$(Get-Elapsed $Started)]"
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $AnalysisDir | Out-Null
Set-Content -LiteralPath $PidFile -Value $PID -Encoding ascii

Write-Section "Targeted CriticalFL Windows/NVIDIA Runs"
Write-Log "Repo:      $RepoRoot"
Write-Log "Runs:      $RunsDir"
Write-Log "Analysis:  $AnalysisDir"
Write-Log "Device:    $Device"
Write-Log "Grid:      $GridIntensity gCO2eq/kWh ($CountryIso)"
Write-Log "Log:       $LogFile"
Write-Log "Targets:"
foreach ($RunLabel in $TargetRunLabels) {
    Write-Log "  - $RunLabel"
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }

if (-not $NoInstall) {
    Write-Section "Step 1/5: Python Environment"
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Log "Creating virtual environment at $VenvDir"
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
            throw "Failed to create Python virtual environment."
        }
    }

    Invoke-Python -Name "Upgrade packaging tools" -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q"
    )
    Invoke-Python -Name "Install repository package" -Arguments @(
        "-m", "pip", "install", "-e", ".", "-q"
    )
    Invoke-Python -Name "Install experiment dependencies" -Arguments @(
        "-m", "pip", "install", "-r", "csfl_simulator\experiments\maml_select\requirements.txt", "-q"
    )
    if ($TorchIndexUrl) {
        Invoke-Python -Name "Install CUDA-enabled PyTorch" -Arguments @(
            "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", $TorchIndexUrl, "-q"
        )
    }
}
elseif (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python.exe"
    Write-Log "NoInstall requested and .venv was not found; using python.exe from PATH."
}

Write-Section "Step 2/5: Dataset Preload"
& $PythonExe "scripts\download_data.py" "--datasets" "fashion-mnist" "cifar10" 2>&1 |
    Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) {
    Write-Log "[WARN] Dataset pre-download failed. Torchvision will retry during the first run."
}
else {
    Write-Log "Datasets ready."
}

Write-Section "Step 3/5: NVIDIA/CUDA Verification"
$NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    (& $NvidiaSmi.Source "--query-gpu=index,name,memory.total,driver_version" "--format=csv,noheader" 2>&1) |
        ForEach-Object { Write-Log "GPU: $_" }
}
else {
    Write-Log "[WARN] nvidia-smi.exe was not found."
}

$TorchCheck = @'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory GB: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f}")
'@
Invoke-Python -Name "PyTorch CUDA check" -Arguments @("-c", $TorchCheck)
$CudaAvailable = ((& $PythonExe "-c" "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null) | Out-String).Trim()
if (($Device -eq "cuda") -and ($CudaAvailable -ne "1")) {
    if ($AllowCpuFallback) {
        Write-Log "[WARN] CUDA is unavailable in PyTorch. Falling back to CPU because -AllowCpuFallback was set."
        $Device = "cpu"
    }
    else {
        throw "CUDA is unavailable in PyTorch. Re-run with: .\csfl_simulator\experiments\maml_select\run_missing_criticalfl_windows.cmd -TorchIndexUrl https://download.pytorch.org/whl/cu121"
    }
}

Write-Section "Step 4/5: Build Target Matrix"
$ConfigBuilder = @'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["experiments"] = [
    {
        "id": "main_benchmarks",
        "profiles": ["core"],
        "scenarios": ["cifar10_main"],
        "methods": ["research.criticalfl"],
        "seeds": [42, 123],
    },
    {
        "id": "main_benchmarks",
        "profiles": ["core"],
        "scenarios": ["fashion_main"],
        "methods": ["research.criticalfl"],
        "seeds": [123],
    },
]
cfg["scenarios"]["cifar10_main"]["rounds"] = 200
cfg["scenarios"]["fashion_main"]["rounds"] = 200

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
'@
Invoke-Python -Name "Write targeted CriticalFL config" -Arguments @(
    "-c", $ConfigBuilder,
    "csfl_simulator\experiments\maml_select\configs.yaml",
    $TargetConfig
)

Invoke-Python -Name "Dry-run targeted matrix" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--config", $TargetConfig,
    "--profile", "core",
    "--only", "main_benchmarks",
    "--method-key", "research.criticalfl",
    "--device", $Device,
    "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--dry-run"
)

Write-Section "Step 5/5: Run Missing CriticalFL Jobs"
$HardwareArgs = @()
if (-not $EnableHardwareMeter) {
    $HardwareArgs += "--no-hardware-meter"
}
$RunArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--config", $TargetConfig,
    "--profile", "core",
    "--only", "main_benchmarks",
    "--method-key", "research.criticalfl",
    "--device", $Device,
    "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso,
    "--grid-intensity", $GridIntensity,
    "--resume",
    "--fail-fast"
) + $HardwareArgs
Invoke-Python -Name "CriticalFL missing table runs" -Arguments $RunArgs

Write-Section "Targeted CriticalFL Runs Complete - $(Get-Elapsed $RunStart)"
foreach ($RunLabel in $TargetRunLabels) {
    $ResultPath = Join-Path (Join-Path $RunsDir $RunLabel) "result.json"
    if (Test-Path -LiteralPath $ResultPath) {
        Write-Log "DONE: $RunLabel"
    }
    else {
        Write-Log "MISSING: $RunLabel"
    }
}
Write-Log "Re-run this same command to resume any interrupted or missing job."
