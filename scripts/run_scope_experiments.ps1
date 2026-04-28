# =============================================================================
# SCOPE-FD Experiment Suite (PowerShell port of run_scope_experiments.sh).
#
# WHY THIS EXISTS (THE WHOLE STORY)
# ---------------------------------
#   Timeline of FD selection attempts in this project:
#     - CALM-FD       -> tied random (0.304 vs 0.307 on the headline config).
#     - PRISM-FD      -> LOST to random (0.271 vs 0.308). Post-mortem: greedy
#                       coverage on static label histograms deterministically
#                       locks onto the same ~15 clients, driving Gini from
#                       random's 0.083 to 0.536 and collapsing data diversity.
#     - SCOPE-FD      -> new design: participation-debt is the PRIMARY ranking
#                       (guarantees Gini -> 0 asymptotically, strictly better
#                       than random's 0.083). Server uncertainty and per-round
#                       diversity layered ON TOP at weights 0.3 and 0.1 so
#                       they can only nudge, never override balance.
#
# WHAT THIS SCRIPT DOES
# ---------------------
#   Validates SCOPE-FD against random across the same configs the earlier
#   v2.1 run covered, plus the MNIST config v2.1 never reached. Existing
#   random curves on disk are re-used; only the SCOPE curves need to be
#   generated for the CIFAR sweeps. Headline and MNIST are run as paired
#   compares (random + SCOPE) for self-contained paper tables.
#
# RANDOM DATA ALREADY ON DISK (from v2.0/v2.1 - not re-run)
#   - exp1_baseline_sota              (alpha=0.5, DL=-20, CIFAR)
#   - exp2_noise_{errfree,dl0,-10,-20,-30}   (5 noise levels, CIFAR)
#   - exp3_alpha_{0_1,0_5,5_0}        (3 alpha levels, CIFAR)
#   - exp4_fl_random                  (FL paradigm, for comm comparison)
#
#   ALL CIFAR random data is already on disk. We never re-run random on CIFAR
#   - scope_headline's random baseline comes from exp2_noise_dl-20 (identical
#   config, deterministic under seed 42).
#
# MISSING RANDOM DATA (this suite backfills - pairs both methods in one run)
#   - MNIST (private) + FMNIST (public): v2.1 killed before reaching its exp5.
#   - FMNIST (private) + MNIST (public): never in any prior suite - a second
#     small-dataset evaluation point for the paper.
#
# =============================================================================
# NEW EXPERIMENTS (EXP 9-12) - ADDED IN RESPONSE TO REVIEWER-STYLE CRITIQUE
# =============================================================================
#
# Why these were added:
#   The original EXP 6 ablation runs at CIFAR, but the paper's Table II and
#   Table III are on FMNIST - so the existing ablation does NOT populate any
#   row in any paper table. The new EXP 9-12 run ablations at the SAME FMNIST
#   configs the paper actually reports, plus multi-seed coverage of the K=1
#   stress test.
#
#   The original ablation set had three variants:
#     - scope_fd            (full: au=0.3, ad=0.1)
#     - scope_fd_no_server  (au=0,   ad=0.1 - debt + diversity)
#     - scope_fd_no_diversity (au=0.3, ad=0  - debt + uncertainty)
#   It was MISSING the most important fourth variant for the paper's narrative:
#     - scope_fd_debt_only  (au=0,   ad=0   - pure round-robin)
#
#   PREREQUISITE: scope_fd_debt_only must be registered as a method in the
#   codebase before EXP 9, 10, and 12 will run.
#
# USAGE
#   powershell -ExecutionPolicy Bypass -File scripts\run_scope_experiments.ps1 -Resume
#   powershell -ExecutionPolicy Bypass -File scripts\run_scope_experiments.ps1 -Exp "9,10,11"
#   powershell -ExecutionPolicy Bypass -File scripts\run_scope_experiments.ps1 -DryRun
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Fast,
    [string]$Exp = "",
    [switch]$Resume,
    [switch]$DryRun,
    [string]$Device = "cuda",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

if ($Cpu) { $Device = "cpu" }
$FastFlag = if ($Fast) { "--fast-mode" } else { "--no-fast-mode" }
$RunOnly  = $Exp

$CIFAR_MODELS = "ResNet18-FD,MobileNetV2-FD,ShuffleNetV2-FD"
$MNIST_MODELS = "FD-CNN1,FD-CNN2,FD-CNN3"

# Build performance flags as an argument list (avoids word-splitting issues).
$PerfFlags = @("--parallel-clients", "2")
if ($Device -eq "cuda") {
    $PerfFlags += @("--use-amp", "--channels-last")
}

$BaseFd = @(
    "--paradigm", "fd",
    "--local-epochs", "2",
    "--public-dataset-size", "2000",
    "--dynamic-steps", "--dynamic-steps-base", "5", "--dynamic-steps-period", "25",
    "--batch-size", "128", "--distillation-batch-size", "500",
    "--distillation-lr", "0.001", "--distillation-epochs", "2", "--temperature", "1.0",
    "--fd-optimizer", "adam",
    "--n-bs-antennas", "64", "--quantization-bits", "8",
    "--eval-every", "10",
    "--profile"
) + $PerfFlags + @("--device", $Device, $FastFlag)

$Rounds = 100

# ---- Method sets ----
$PairSet            = "heuristic.random,fd_native.scope_fd"
$ScopeOnly          = "fd_native.scope_fd"
$ScopeAblation      = "fd_native.scope_fd,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"
$ScopeAblationFull  = "fd_native.scope_fd,fd_native.scope_fd_debt_only,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"
$AblationPair       = "heuristic.random,fd_native.scope_fd,fd_native.scope_fd_debt_only,fd_native.scope_fd_no_server,fd_native.scope_fd_no_diversity"

# Toggle for the optional coefficient sensitivity sweep (EXP 12).
$RunCoefSweep = $false

$script:Total    = 0
$script:Passed   = 0
$script:Failed   = 0
$script:Skipped  = 0
$script:Failures = @()
$script:CurExp   = ""
$script:CurExpNum = 0

function Get-ExpRunCount {
    param([int]$N)
    switch ($N) {
        1  { 1 }   # CIFAR headline
        2  { 5 }   # Noise sweep
        3  { 3 }   # Alpha sweep
        4  { 1 }   # MNIST headline
        5  { 1 }   # FMNIST headline
        6  { 1 }   # SCOPE ablation (CIFAR)
        7  { 7 }   # K-sweep at N=50 on Fashion-MNIST
        8  { 5 }   # FMNIST channel-sweep at N=50, K=5
        9  { 1 }   # FMNIST K=5 ablation
        10 { 1 }   # FMNIST K=1 ablation
        11 { 3 }   # K=1 multi-seed
        12 { 4 }   # OPTIONAL: coefficient sensitivity
        default { 0 }
    }
}

function Write-Log {
    param([string]$Message)
    Write-Host ""
    Write-Host ("========== [{0}] {1} ==========" -f (Get-Date -Format "HH:mm:ss"), $Message)
    Write-Host ""
}

function Should-Run {
    param([int]$N)
    if ([string]::IsNullOrEmpty($RunOnly)) { return $true }
    $arr = $RunOnly.Split(",") | ForEach-Object { $_.Trim() }
    return $arr -contains "$N"
}

function Compute-TotalPlanned {
    $t = 0
    foreach ($e in 1..12) {
        if (Should-Run $e) {
            $t += (Get-ExpRunCount $e)
        }
    }
    return $t
}

function Format-Hms {
    param([int]$S)
    return ("{0}h {1:D2}m {2:D2}s" -f [int]($S / 3600), [int](($S % 3600) / 60), [int]($S % 60))
}

function Get-ProgressPrefix {
    $doneCount = $script:Passed + $script:Failed + $script:Skipped
    $current   = $doneCount + 1
    $pct = 0
    if ($script:TotalPlanned -gt 0) {
        $pct = [int](($current * 100) / $script:TotalPlanned)
    }
    $now = [int][double]::Parse((Get-Date -UFormat %s))
    $elapsed = $now - $script:GlobalStart
    $etaStr = ""
    if ($doneCount -gt 0 -and $script:TotalPlanned -gt $doneCount) {
        $perRun = [int]($elapsed / $doneCount)
        $remaining = ($script:TotalPlanned - $doneCount) * $perRun
        $etaStr = " | ETA~$(Format-Hms $remaining)"
    }
    $expLabel = if ($script:CurExp) { $script:CurExp } else { "?" }
    return ("[{0}/{1} {2}% | {3} | elapsed={4}{5}]" -f `
        $current, $script:TotalPlanned, $pct, $expLabel, (Format-Hms $elapsed), $etaStr)
}

function Test-RunExists {
    param([string]$Name)
    $dirs = Get-ChildItem -Path "artifacts/runs" -Directory -Filter "${Name}_*" -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        $cmpFile = Join-Path $d.FullName "compare_results.json"
        $resFile = Join-Path $d.FullName "results.json"
        if ((Test-Path $cmpFile) -or (Test-Path $resFile)) {
            return $true
        }
    }
    return $false
}

function Invoke-OneRun {
    param(
        [string]$Name,
        [string[]]$Args
    )
    $script:Total++
    $prefix = Get-ProgressPrefix

    if ($Resume -and (Test-RunExists $Name)) {
        Write-Host "  $prefix [SKIP] $Name (completion marker found)"
        $script:Skipped++
        return
    }
    if ($DryRun) {
        $cmdline = "python -m csfl_simulator compare --name $Name " + ($Args -join " ")
        Write-Host "  $prefix [DRY] $cmdline"
        $script:Skipped++
        return
    }

    $t0 = [int][double]::Parse((Get-Date -UFormat %s))
    Write-Host "  $prefix [RUN] $Name"

    $allArgs = @("-m", "csfl_simulator", "compare", "--name", $Name) + $Args
    & python @allArgs
    $exitCode = $LASTEXITCODE
    $dt = [int][double]::Parse((Get-Date -UFormat %s)) - $t0

    if ($exitCode -eq 0) {
        Write-Host ("  {0} [OK]   {1} - {2}" -f $prefix, $Name, (Format-Hms $dt))
        $script:Passed++
    } else {
        Write-Host ("  {0} [FAIL] {1} - {2}" -f $prefix, $Name, (Format-Hms $dt))
        $script:Failed++
        $script:Failures += "  - $Name"
    }
}

$script:GlobalStart   = [int][double]::Parse((Get-Date -UFormat %s))
$script:TotalPlanned  = Compute-TotalPlanned

Write-Log "SCOPE-FD Suite (debt-balanced + server-aware + diversity)"
Write-Host "  Device:         $Device"
Write-Host ("  Mode:           {0}" -f $(if ($FastFlag -eq "--fast-mode") { "FAST (debug)" } else { "FULL" }))
Write-Host "  Resume:         $Resume"
Write-Host "  Dry-run:        $DryRun"
if ($RunOnly) { Write-Host "  Filter:         $RunOnly" }
Write-Host "  Planned:        $($script:TotalPlanned) compare invocations"
Write-Host "  Headline/MNIST: $PairSet"
Write-Host "  CIFAR sweeps:   $ScopeOnly (pair with existing random)"
Write-Host "  Ablation:       $ScopeAblation"
Write-Host "  Rounds:         $Rounds"

# =============================================================================
# EXP 1 - CIFAR headline: SCOPE alone (alpha=0.5, DL=-20 dB).
# =============================================================================
if (Should-Run 1) {
    $script:CurExpNum = 1; $script:CurExp = "S1/12-cifar"
    Write-Log "EXP 1/12: CIFAR headline - SCOPE alone (random from exp2_noise_dl-20)"
    Invoke-OneRun "scope_cifar" (@(
        "--methods", $ScopeOnly
    ) + $BaseFd + @(
        "--dataset", "CIFAR-10", "--public-dataset", "STL-10",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "ResNet18-FD", "--model-heterogeneous", "--model-pool", $CIFAR_MODELS,
        "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 2 - CIFAR noise sweep: SCOPE alone across 5 DL SNR levels.
# =============================================================================
if (Should-Run 2) {
    $script:CurExpNum = 2
    Write-Log "EXP 2/12: Noise sweep - 5 DL SNR levels (SCOPE alone; pair with existing random)"
    $noiseLevels = @(
        @{ Label = "errfree";  Flags = @() },
        @{ Label = "dl0";      Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "0") },
        @{ Label = "dl-10";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-10") },
        @{ Label = "dl-20";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20") },
        @{ Label = "dl-30";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-30") }
    )
    foreach ($lvl in $noiseLevels) {
        $script:CurExp = "S2/12-noise-$($lvl.Label)"
        Write-Log "  Noise sweep level: $($lvl.Label)"
        Invoke-OneRun "scope_noise_$($lvl.Label)" (@(
            "--methods", $ScopeOnly
        ) + $BaseFd + @(
            "--dataset", "CIFAR-10", "--public-dataset", "STL-10",
            "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
            "--model", "ResNet18-FD", "--model-heterogeneous", "--model-pool", $CIFAR_MODELS,
            "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds"
        ) + $lvl.Flags + @(
            "--seed", "42"
        ))
    }
}

# =============================================================================
# EXP 3 - CIFAR alpha sweep: SCOPE alone across 3 non-IID levels.
# =============================================================================
if (Should-Run 3) {
    $script:CurExpNum = 3
    Write-Log "EXP 3/12: Alpha sweep - low / mid / high-IID (SCOPE alone; pair with existing random)"
    foreach ($alpha in @("0.1", "0.5", "5.0")) {
        $alabel = $alpha.Replace(".", "_")
        $script:CurExp = "S3/12-alpha$alabel"
        Write-Log "  alpha = $alpha"
        Invoke-OneRun "scope_alpha_$alabel" (@(
            "--methods", $ScopeOnly
        ) + $BaseFd + @(
            "--dataset", "CIFAR-10", "--public-dataset", "STL-10",
            "--partition", "dirichlet", "--dirichlet-alpha", $alpha,
            "--model", "ResNet18-FD", "--model-heterogeneous", "--model-pool", $CIFAR_MODELS,
            "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds",
            "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
            "--seed", "42"
        ))
    }
}

# =============================================================================
# EXP 4 - MNIST(private) + FMNIST(public) headline: random vs SCOPE.
# =============================================================================
if (Should-Run 4) {
    $script:CurExpNum = 4; $script:CurExp = "S4/12-mnist"
    Write-Log "EXP 4/12: MNIST(private) + FMNIST(public) - random vs SCOPE"
    Invoke-OneRun "scope_mnist" (@(
        "--methods", $PairSet
    ) + $BaseFd + @(
        "--dataset", "MNIST", "--public-dataset", "FMNIST",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
        "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds",
        "--batch-size", "20",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 5 - FMNIST(private) + MNIST(public) headline: random vs SCOPE.
# =============================================================================
if (Should-Run 5) {
    $script:CurExpNum = 5; $script:CurExp = "S5/12-fmnist"
    Write-Log "EXP 5/12: Fashion-MNIST(private) + MNIST(public) - random vs SCOPE"
    Invoke-OneRun "scope_fmnist" (@(
        "--methods", $PairSet
    ) + $BaseFd + @(
        "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
        "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds",
        "--batch-size", "20",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 6 - SCOPE ablation (at CIFAR headline config).
# NOTE: Supplementary, does NOT populate any row in the paper's tables.
# =============================================================================
if (Should-Run 6) {
    $script:CurExpNum = 6; $script:CurExp = "S6/12-ablation-cifar"
    Write-Log "EXP 6/12: SCOPE ablation (CIFAR - supplementary, not in paper tables)"
    Invoke-OneRun "scope_ablation" (@(
        "--methods", $ScopeAblation
    ) + $BaseFd + @(
        "--dataset", "CIFAR-10", "--public-dataset", "STL-10",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "ResNet18-FD", "--model-heterogeneous", "--model-pool", $CIFAR_MODELS,
        "--total-clients", "30", "--clients-per-round", "10", "--rounds", "$Rounds",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 7 - K-sweep at N=50 on Fashion-MNIST (private) + MNIST (public).
# =============================================================================
if (Should-Run 7) {
    $script:CurExpNum = 7
    Write-Log "EXP 7/12: K-sweep - N=50 on Fashion-MNIST (7 K values: 1,5,10,15,25,35,50)"
    foreach ($K in @(1, 5, 10, 15, 25, 35, 50)) {
        $script:CurExp = "S7/12-N50-K$K"
        Write-Log ("  K = {0} ({1}%)" -f $K, ($K * 100 / 50))
        Invoke-OneRun "scope_fmnist_N50_K$K" (@(
            "--methods", $PairSet
        ) + $BaseFd + @(
            "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
            "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
            "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
            "--total-clients", "50", "--clients-per-round", "$K", "--rounds", "$Rounds",
            "--batch-size", "20",
            "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
            "--seed", "42"
        ))
    }
}

# =============================================================================
# EXP 8 - FMNIST channel-noise sweep at N=50, K=5 (5 DL SNR levels).
# =============================================================================
if (Should-Run 8) {
    $script:CurExpNum = 8
    Write-Log "EXP 8/12: FMNIST channel sweep - N=50, K=5, 5 DL SNR levels"
    $noiseLevels = @(
        @{ Label = "errfree";  Flags = @() },
        @{ Label = "dl0";      Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "0") },
        @{ Label = "dl-10";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-10") },
        @{ Label = "dl-20";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20") },
        @{ Label = "dl-30";    Flags = @("--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-30") }
    )
    foreach ($lvl in $noiseLevels) {
        $script:CurExp = "S8/12-fmnist-N50-K5-$($lvl.Label)"
        Write-Log "  FMNIST N=50 K=5 @ $($lvl.Label)"
        Invoke-OneRun "scope_fmnist_N50_K5_noise_$($lvl.Label)" (@(
            "--methods", $PairSet
        ) + $BaseFd + @(
            "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
            "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
            "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
            "--total-clients", "50", "--clients-per-round", "5", "--rounds", "$Rounds",
            "--batch-size", "20"
        ) + $lvl.Flags + @(
            "--seed", "42"
        ))
    }
}

# =============================================================================
# EXP 9 - FMNIST K=5 ABLATION (NEW). Populates the K=5 block of Table IV.
# PREREQUISITE: fd_native.scope_fd_debt_only must be registered.
# =============================================================================
if (Should-Run 9) {
    $script:CurExpNum = 9; $script:CurExp = "S9/12-fmnist-N50-K5-ablation"
    Write-Log "EXP 9/12: FMNIST N=50 K=5 ablation - 4 SCOPE variants + random"
    Invoke-OneRun "scope_fmnist_N50_K5_ablation" (@(
        "--methods", $AblationPair
    ) + $BaseFd + @(
        "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
        "--total-clients", "50", "--clients-per-round", "5", "--rounds", "$Rounds",
        "--batch-size", "20",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 10 - FMNIST K=1 ABLATION (NEW). Populates the K=1 block of Table IV.
# =============================================================================
if (Should-Run 10) {
    $script:CurExpNum = 10; $script:CurExp = "S10/12-fmnist-N50-K1-ablation"
    Write-Log "EXP 10/12: FMNIST N=50 K=1 ablation - 4 SCOPE variants + random"
    Invoke-OneRun "scope_fmnist_N50_K1_ablation" (@(
        "--methods", $AblationPair
    ) + $BaseFd + @(
        "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
        "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
        "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
        "--total-clients", "50", "--clients-per-round", "1", "--rounds", "$Rounds",
        "--batch-size", "20",
        "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
        "--seed", "42"
    ))
}

# =============================================================================
# EXP 11 - K=1 MULTI-SEED (NEW). Adds error bars to the Table II K=1 row.
# =============================================================================
if (Should-Run 11) {
    $script:CurExpNum = 11
    Write-Log "EXP 11/12: FMNIST N=50 K=1 multi-seed - 3 extra seeds (paired random+SCOPE)"
    foreach ($seed in @(7, 123, 2024)) {
        $script:CurExp = "S11/12-fmnist-K1-seed$seed"
        Write-Log "  seed = $seed"
        Invoke-OneRun "scope_fmnist_N50_K1_seed$seed" (@(
            "--methods", $PairSet
        ) + $BaseFd + @(
            "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
            "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
            "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
            "--total-clients", "50", "--clients-per-round", "1", "--rounds", "$Rounds",
            "--batch-size", "20",
            "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20",
            "--seed", "$seed"
        ))
    }
}

# =============================================================================
# EXP 12 - COEFFICIENT SENSITIVITY (NEW, OPTIONAL). Gated by $RunCoefSweep.
# =============================================================================
if ((Should-Run 12) -and $RunCoefSweep) {
    $script:CurExpNum = 12
    Write-Log "EXP 12/12: Coefficient sensitivity - 4 (au, ad) pairs at FMNIST N=50, K=5"
    $coefPairs = @(
        @{ Label = "0_10_0_05"; Flags = @("--scope-au", "0.10", "--scope-ad", "0.05") },
        @{ Label = "0_20_0_10"; Flags = @("--scope-au", "0.20", "--scope-ad", "0.10") },
        @{ Label = "0_30_0_10"; Flags = @("--scope-au", "0.30", "--scope-ad", "0.10") },
        @{ Label = "0_40_0_20"; Flags = @("--scope-au", "0.40", "--scope-ad", "0.20") }
    )
    foreach ($pair in $coefPairs) {
        $script:CurExp = "S12/12-fmnist-K5-coef-$($pair.Label)"
        Write-Log "  (au, ad) labelled $($pair.Label)"
        Invoke-OneRun "scope_fmnist_N50_K5_coef_$($pair.Label)" (@(
            "--methods", $ScopeOnly
        ) + $BaseFd + @(
            "--dataset", "Fashion-MNIST", "--public-dataset", "MNIST",
            "--partition", "dirichlet", "--dirichlet-alpha", "0.5",
            "--model", "FD-CNN1", "--model-heterogeneous", "--model-pool", $MNIST_MODELS,
            "--total-clients", "50", "--clients-per-round", "5", "--rounds", "$Rounds",
            "--batch-size", "20",
            "--channel-noise", "--ul-snr-db", "-8", "--dl-snr-db", "-20"
        ) + $pair.Flags + @(
            "--seed", "42"
        ))
    }
} elseif ((Should-Run 12) -and (-not $RunCoefSweep)) {
    Write-Log "EXP 12/12: Coefficient sensitivity SKIPPED (`$RunCoefSweep = `$false)"
    Write-Host "  To enable: set `$RunCoefSweep = `$true near the top of this script"
    Write-Host "  AND ensure the simulator accepts --scope-au / --scope-ad flags."
}

# =============================================================================
# Summary
# =============================================================================
$globalEnd = [int][double]::Parse((Get-Date -UFormat %s))
$dt = $globalEnd - $script:GlobalStart

Write-Log "SCOPE SUITE COMPLETE"
Write-Host "  Planned:  $($script:TotalPlanned)"
Write-Host "  Total:    $($script:Total)"
Write-Host "  Passed:   $($script:Passed)"
Write-Host "  Failed:   $($script:Failed)"
Write-Host "  Skipped:  $($script:Skipped)"
Write-Host "  Wall:     $(Format-Hms $dt)"

if ($script:Failures.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed runs:"
    $script:Failures | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "  Results in: artifacts/runs/  (prefix: scope_*)"
Write-Host "  Pairing:    random baselines from exp2_*/exp3_* (CIFAR sweeps)."
Write-Host ""
Write-Host "  NEW (EXP 9-12) results to incorporate into the paper:"
Write-Host "    - scope_fmnist_N50_K5_ablation     -> Table IV K=5 block"
Write-Host "    - scope_fmnist_N50_K1_ablation     -> Table IV K=1 block"
Write-Host "    - scope_fmnist_N50_K1_seed{7,123,2024} -> Table II K=1 row error bars"
Write-Host "    - scope_fmnist_N50_K5_coef_*       -> Section IV-E one-sentence claim (if run)"
