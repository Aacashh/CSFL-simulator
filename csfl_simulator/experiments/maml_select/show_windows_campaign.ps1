[CmdletBinding()]
param(
    [switch] $Follow
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\..\.."))
$RunsDir = Join-Path $RepoRoot "runs\maml_select"
$LogDir = Join-Path $RunsDir "logs"
$PidFile = Join-Path $LogDir "campaign_windows.pid"

if (Test-Path -LiteralPath $PidFile) {
    $CampaignProcessId = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($CampaignProcessId -and (Get-Process -Id $CampaignProcessId -ErrorAction SilentlyContinue)) {
        Write-Host "Campaign status: running (PID $CampaignProcessId)"
    }
    else {
        Write-Host "Campaign status: not running (stale PID file)"
    }
}
else {
    Write-Host "Campaign status: not started"
}

$Results = @(Get-ChildItem -LiteralPath $RunsDir -Filter "result.json" -File -Recurse -ErrorAction SilentlyContinue)
Write-Host "Completed result files: $($Results.Count)"

$LatestProgress = Get-ChildItem -LiteralPath $RunsDir -Filter "progress.json" -File -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestProgress) {
    $Progress = Get-Content -LiteralPath $LatestProgress.FullName -Raw | ConvertFrom-Json
    Write-Host "Latest progress: round $($Progress.round)/$($Progress.rounds)"
    Write-Host "  $($LatestProgress.FullName)"
}

$LatestLog = Get-ChildItem -LiteralPath $LogDir -Filter "campaign_windows_*.log" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestLog) {
    Write-Host ""
    Write-Host "Latest log: $($LatestLog.FullName)"
    if ($Follow) {
        Get-Content -LiteralPath $LatestLog.FullName -Tail 30 -Wait
    }
    else {
        Get-Content -LiteralPath $LatestLog.FullName -Tail 30
    }
}
