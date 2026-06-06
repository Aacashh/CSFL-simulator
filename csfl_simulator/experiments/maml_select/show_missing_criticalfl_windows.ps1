[CmdletBinding()]
param(
    [switch] $Follow
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$RunsDir = Join-Path $RepoRoot "runs\maml_select"
$LogDir = Join-Path $RunsDir "logs"
$PidFile = Join-Path $LogDir "missing_criticalfl_windows.pid"
$TargetRunLabels = @(
    "main_benchmarks_cifar10_main_criticalfl_s42",
    "main_benchmarks_cifar10_main_criticalfl_s123",
    "main_benchmarks_fashion_main_criticalfl_s123"
)

if (Test-Path -LiteralPath $PidFile) {
    $RunPid = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($RunPid -and (Get-Process -Id $RunPid -ErrorAction SilentlyContinue)) {
        Write-Host "Missing CriticalFL status: running (PID $RunPid)"
    }
    else {
        Write-Host "Missing CriticalFL status: not running (stale PID file)"
    }
}
else {
    Write-Host "Missing CriticalFL status: not started"
}

Write-Host ""
foreach ($RunLabel in $TargetRunLabels) {
    $RunDir = Join-Path $RunsDir $RunLabel
    $ResultPath = Join-Path $RunDir "result.json"
    $ProgressPath = Join-Path $RunDir "progress.json"
    if (Test-Path -LiteralPath $ResultPath) {
        $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
        $Metrics = $Result.simulation.metrics | Select-Object -Last 1
        Write-Host "DONE    $RunLabel"
        Write-Host ("        acc={0:N4}, f1={1:N4}, carbon_g={2:N2}, rounds={3}" -f `
            [double]$Metrics.accuracy, [double]$Metrics.f1, [double]$Metrics.cum_modelled_carbon_g, `
            [int]$Result.simulation.rounds_completed)
    }
    elseif (Test-Path -LiteralPath $ProgressPath) {
        $Progress = Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json
        $Metric = $Progress.latest_metrics
        Write-Host "RUNNING $RunLabel"
        Write-Host ("        round={0}/{1}, acc={2:N4}, loss={3:N4}, carbon_g={4:N2}" -f `
            [int]$Progress.round, [int]$Progress.rounds, [double]$Metric.accuracy, `
            [double]$Metric.loss, [double]$Metric.cum_modelled_carbon_g)
    }
    else {
        Write-Host "PENDING $RunLabel"
    }
}

$LatestLog = Get-ChildItem -LiteralPath $LogDir -Filter "missing_criticalfl_windows_*.log" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestLog) {
    Write-Host ""
    Write-Host "Latest log: $($LatestLog.FullName)"
    if ($Follow) {
        Get-Content -LiteralPath $LatestLog.FullName -Tail 40 -Wait
    }
    else {
        Get-Content -LiteralPath $LatestLog.FullName -Tail 40
    }
}
