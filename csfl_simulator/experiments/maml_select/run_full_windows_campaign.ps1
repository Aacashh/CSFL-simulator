[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python.exe" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }
$CountryIso = if ($env:COUNTRY_ISO) { $env:COUNTRY_ISO } else { "IND" }
$GridIntensity = if ($env:GRID_INTENSITY) { $env:GRID_INTENSITY } else { "475" }
$VerifiedHardwareTelemetry = if ($env:VERIFIED_HARDWARE_TELEMETRY) { $env:VERIFIED_HARDWARE_TELEMETRY } else { "0" }
$RunsDir = if ($env:RUNS_DIR) { $env:RUNS_DIR } else { Join-Path $RepoRoot "runs\maml_select" }
$ArtifactsDir = if ($env:ARTIFACTS_DIR) { $env:ARTIFACTS_DIR } else { Join-Path $RepoRoot "artifacts\maml_select" }
$AnalysisDir = Join-Path $ArtifactsDir "analysis"
$LogDir = Join-Path $RunsDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "campaign_windows_$Timestamp.log"
$FailedStepCount = 0
$CampaignStart = Get-Date

$TelemetryArgs = @()
if ($VerifiedHardwareTelemetry -eq "1") {
    $TelemetryArgs += "--verified-hardware-telemetry"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null

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

function Invoke-Step {
    param(
        [string] $Name,
        [string[]] $Arguments
    )
    $Started = Get-Date
    Write-Log "  START: $Name"
    Write-Log "    cmd: $PythonBin $($Arguments -join ' ')"
    & $PythonBin @Arguments 2>&1 | Tee-Object -FilePath $LogFile -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Log "  OK: $Name [$(Get-Elapsed $Started)]"
    }
    else {
        Write-Log "  FAILED: $Name (exit $ExitCode) [$(Get-Elapsed $Started)] - continuing"
        $script:FailedStepCount += 1
    }
}

Set-Location $RepoRoot

Write-Section "MAML-Select Windows Campaign - $Timestamp"
Write-Log "  Repo:       $RepoRoot"
Write-Log "  Runs:       $RunsDir"
Write-Log "  Artifacts:  $ArtifactsDir"
Write-Log "  Device:     $Device"
Write-Log "  Grid:       $GridIntensity gCO2eq/kWh ($CountryIso)"
Write-Log "  Verified hardware telemetry: $VerifiedHardwareTelemetry"

$NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    (& $NvidiaSmi.Source "--query-gpu=name,memory.total,driver_version" "--format=csv,noheader" 2>&1) |
        ForEach-Object { Write-Log "  GPU: $_" }
}
else {
    Write-Log "  [WARN] nvidia-smi.exe was not found. Using DEVICE=cpu."
    $Device = "cpu"
}

$TorchInfo = & $PythonBin "-c" "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Unable to import PyTorch with $PythonBin."
}
$TorchInfo | ForEach-Object { Write-Log "  $_" }
$CudaAvailable = ((& $PythonBin "-c" "import torch; print('1' if torch.cuda.is_available() else '0')") | Out-String).Trim()
if (($Device -eq "cuda") -and ($CudaAvailable -ne "1")) {
    Write-Log "  [WARN] CUDA is unavailable in PyTorch. Using DEVICE=cpu."
    $Device = "cpu"
}

Write-Section "PHASE 0: Quick Sanity Check"
Invoke-Step -Name "Dry-run experiment matrix" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "quick", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir, "--dry-run"
)
Invoke-Step -Name "Quick validation (12 rounds, 1 seed)" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "quick", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir, "--no-hardware-meter", "--resume"
)

Write-Section "PHASE 1: Core Benchmarks (Fashion-MNIST + CIFAR-10, 200 rounds, 3 seeds)"
$CoreArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "core", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity
) + $TelemetryArgs + @("--resume")
Invoke-Step -Name "Core benchmarks and CIFAR-10 reconciliation" -Arguments $CoreArgs

Write-Section "PHASE 2: Energy-to-Target (Fashion 70%, CIFAR-10 88%)"
$EnergyArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "energy", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity
) + $TelemetryArgs + @("--resume")
Invoke-Step -Name "Energy-to-target experiments" -Arguments $EnergyArgs

Write-Section "PHASE 3: Lambda Sensitivity"
Invoke-Step -Name "Lambda sensitivity experiment matrix" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "core", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity,
    "--only", "lambda_sensitivity", "--resume"
)

Write-Section "PHASE 4: Feature Ablation"
Invoke-Step -Name "Feature ablation experiment matrix" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "core", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity,
    "--only", "feature_ablation", "--resume"
)

Write-Section "PHASE 5: Extended Matrix (heterogeneity + scaling)"
$FullArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "full", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity
) + $TelemetryArgs + @("--resume")
Invoke-Step -Name "Full extended matrix" -Arguments $FullArgs

Write-Section "PHASE 6: Time Complexity (N=20, 40, 80, 100)"
Invoke-Step -Name "Scaling overhead benchmark (10 rounds per N)" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_scaling",
    "--device", $Device, "--rounds", "10", "--output-dir", (Join-Path $RunsDir "scaling")
)
Invoke-Step -Name "Scaling overhead matrix (200 rounds)" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--profile", "scaling", "--device", $Device, "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIso, "--grid-intensity", $GridIntensity,
    "--resume"
)

Write-Section "PHASE 7: Analysis and EPS Plots"
Invoke-Step -Name "Statistical analysis" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.analyze_results",
    "--results-dir", $RunsDir, "--output-dir", $AnalysisDir
)
Invoke-Step -Name "Publication plots (EPS only)" -Arguments @(
    "-m", "csfl_simulator.experiments.maml_select.generate_plots",
    "--results-dir", $RunsDir,
    "--output-dir", (Join-Path $ArtifactsDir "plots"),
    "--sensitivity-dir", (Join-Path $RunsDir "sensitivity"),
    "--ablation-dir", (Join-Path $RunsDir "ablation"),
    "--scaling-dir", (Join-Path $RunsDir "scaling")
)

if ($FailedStepCount -gt 0) {
    Write-Section "CAMPAIGN FINISHED WITH $FailedStepCount FAILED STEP(S)"
    Write-Log "Inspect $LogFile, then rerun setup_and_run_windows.cmd. Completed results will be resumed."
    exit 1
}

Write-Section "CAMPAIGN COMPLETE - $(Get-Elapsed $CampaignStart)"
Write-Log "  Runtime JSON logs: $RunsDir"
Write-Log "  Analysis tables:   $(Join-Path $ArtifactsDir 'analysis')"
Write-Log "  EPS plots:         $(Join-Path $ArtifactsDir 'plots')"
Write-Log "  Log:               $LogFile"
