param(
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "auto" }),
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "" }),
    [string]$CountryIsoCode = $(if ($env:COUNTRY_ISO_CODE) { $env:COUNTRY_ISO_CODE } else { "IND" }),
    [string]$GridIntensity = $(if ($env:GRID_INTENSITY) { $env:GRID_INTENSITY } else { "475" }),
    [string]$NoHardwareMeter = $(if ($env:NO_HARDWARE_METER) { $env:NO_HARDWARE_METER } else { "1" })
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$Requested)
    if ($Requested) {
        return $Requested
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }
    throw "Python was not found. Install Python 3.10+ and ensure python.exe or py.exe is on PATH."
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-Host ">" $Exe ($Arguments -join " ")
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $Exe $($Arguments -join ' ')"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$RepoRoot = $RepoRoot.Path
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $RepoRoot ".venv_audio_fsdd" }
$Config = Join-Path $ScriptDir "configs.yaml"
$RunsDir = Join-Path $RepoRoot "runs\audio_fsdd"
$ArtifactsDir = Join-Path $RepoRoot "artifacts\audio_fsdd"
$AnalysisDir = Join-Path $ArtifactsDir "analysis"
$PlotsDir = Join-Path $ArtifactsDir "plots"
$LogDir = Join-Path $RunsDir "logs"
$LogFile = Join-Path $LogDir "audio_fsdd_100r.log"

New-Item -ItemType Directory -Force -Path $LogDir, $AnalysisDir, $PlotsDir | Out-Null
Set-Location $RepoRoot

Write-Host "Audio FSDD 100-round pipeline"
Write-Host "Repo:      $RepoRoot"
Write-Host "Config:    $Config"
Write-Host "Runs:      $RunsDir"
Write-Host "Artifacts: $ArtifactsDir"
Write-Host "Device:    $Device"

$BasePython = Resolve-Python -Requested $PythonBin
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment: $VenvDir"
    if ((Split-Path -Leaf $BasePython) -eq "py.exe") {
        Invoke-Checked $BasePython @("-3", "-m", "venv", $VenvDir)
    } else {
        Invoke-Checked $BasePython @("-m", "venv", $VenvDir)
    }
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython"
}

Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", ".")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "csfl_simulator\experiments\maml_select\requirements.txt")

Invoke-Checked $VenvPython @("-c", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {hasattr(torch.backends, ""mps"") and torch.backends.mps.is_available()}')")

Write-Host "Preparing FSDD dataset cache..."
Invoke-Checked $VenvPython @("-c", "from csfl_simulator.core.datasets import get_full_data; train,test=get_full_data('FSDD'); print(f'FSDD ready: train={len(train)} test={len(test)} classes={len(train.classes)}')")

$MeterArgs = @()
if ($NoHardwareMeter -eq "1") {
    $MeterArgs += "--no-hardware-meter"
}

Write-Host "Started at $(Get-Date)" | Tee-Object -FilePath $LogFile

$RunArgs = @(
    "-m", "csfl_simulator.experiments.maml_select.run_experiments",
    "--config", $Config,
    "--profile", "audio_fsdd",
    "--device", $Device,
    "--output-dir", $RunsDir,
    "--analysis-dir", $AnalysisDir,
    "--country-iso-code", $CountryIsoCode,
    "--grid-intensity", $GridIntensity,
    "--resume"
) + $MeterArgs

& $VenvPython @RunArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) {
    throw "Audio FSDD experiment run failed with exit code $LASTEXITCODE"
}

& $VenvPython -m csfl_simulator.experiments.maml_select.analyze_results `
    --results-dir $RunsDir `
    --output-dir $AnalysisDir 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) {
    throw "Audio FSDD analysis failed with exit code $LASTEXITCODE"
}

& $VenvPython -m csfl_simulator.experiments.audio_fsdd.plot_audio_fsdd `
    --results-dir $RunsDir `
    --analysis-dir $AnalysisDir `
    --plots-dir $PlotsDir 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) {
    throw "Audio FSDD plotting failed with exit code $LASTEXITCODE"
}

Write-Host "Finished at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append
Write-Host ""
Write-Host "Audio FSDD pipeline complete."
Write-Host "Logs:     $LogFile"
Write-Host "Results:  $RunsDir"
Write-Host "Analysis: $AnalysisDir"
Write-Host "Plots:    $PlotsDir"
