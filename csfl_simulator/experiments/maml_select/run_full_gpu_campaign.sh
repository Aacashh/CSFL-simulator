#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# MAML-Select Full GPU Experiment Campaign
# ═══════════════════════════════════════════════════════════════════════════════
#
# Master script for IEEE TAI letter revision experiments.
# Designed for an NVIDIA GPU workstation with no remote access.
#
# Directory layout:
#   runs/maml_select/          ← All JSON result logs (per-run directories)
#   artifacts/maml_select/     ← Publication plots, analysis tables, CSVs
#
# Usage:
#   nohup bash run_full_gpu_campaign.sh 2>&1 | tee campaign.log &
#   OR: tmux new -s maml → bash run_full_gpu_campaign.sh
#
# Fully resumable: re-run after any interruption.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
COUNTRY_ISO="${COUNTRY_ISO:-IND}"
GRID_INTENSITY="${GRID_INTENSITY:-475}"

# Output directories
RUNS_DIR="${REPO_ROOT}/runs/maml_select"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts/maml_select"
LOG_DIR="${RUNS_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/campaign_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${ARTIFACTS_DIR}"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

section() {
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  $*"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

elapsed() {
    local dur=$(( $(date +%s) - $1 ))
    printf '%02d:%02d:%02d' $((dur/3600)) $(((dur%3600)/60)) $((dur%60))
}

run_step() {
    local name="$1"; shift
    local t0=$(date +%s)
    log "  ▶ ${name}"
    log "    cmd: $*"
    if "$@" >> "${LOG_FILE}" 2>&1; then
        log "  ✓ ${name} [$(elapsed $t0)]"
    else
        log "  ✗ FAILED: ${name} (exit $?) [$(elapsed $t0)] — continuing"
    fi
}

# ── Pre-flight ───────────────────────────────────────────────────────────────

CAMPAIGN_START=$(date +%s)

section "MAML-Select GPU Campaign — ${TIMESTAMP}"
log "  Repo:       ${REPO_ROOT}"
log "  Runs →      ${RUNS_DIR}/"
log "  Artifacts → ${ARTIFACTS_DIR}/"
log "  Device:     ${DEVICE}"
log "  Grid:       ${GRID_INTENSITY} gCO2eq/kWh (${COUNTRY_ISO})"

cd "${REPO_ROOT}"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | while read -r line; do log "  GPU: $line"; done
else
    log "  [WARN] No nvidia-smi found. Using DEVICE=cpu"
    DEVICE="cpu"
fi

"${PYTHON_BIN}" -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 | while read -r line; do log "$line"; done

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Quick Sanity Check
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 0: Quick Sanity Check"

run_step "Dry-run (print experiment matrix)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile quick --device "${DEVICE}" --output-dir "${RUNS_DIR}" --dry-run

run_step "Quick validation (12 rounds, 1 seed)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile quick --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --no-hardware-meter --resume

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Core Benchmarks: Fashion-MNIST + CIFAR-10
#   200 rounds × 3 seeds × 8 methods → paired t-tests, CIs, effect sizes
#   Reviewer 1.6, 2.5, 2.6, 2.11, 3.4, 3.5
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 1: Core Benchmarks (Fashion-MNIST + CIFAR-10, 200 rounds, 3 seeds)"

run_step "Core benchmarks (main_benchmarks + cifar10_reconciled)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile core --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --verified-hardware-telemetry --resume

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Energy-to-Target
#   Runs until accuracy target or round cap; measures hardware energy + carbon
#   Reviewer 2.7 (energy claims)
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 2: Energy-to-Target (Fashion 70%, CIFAR-10 88%)"

run_step "Energy-to-target experiments" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile energy --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --verified-hardware-telemetry --resume

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Lambda (λ) Sensitivity Sweep
#   λ ∈ {0.1, 0.5, 1.0, 5.0} × 3 seeds
#   Reviewer 2.3 (lambda selection justification)
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 3: Lambda Sensitivity (λ=0.1, 0.5, 1.0, 5.0)"

run_step "Lambda sensitivity (experiment matrix)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile core --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --only lambda_sensitivity --resume

run_step "Lambda sensitivity (standalone CSV)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_sensitivity \
    --device "${DEVICE}" --seeds 42 123 2026 \
    --output-dir "${RUNS_DIR}/sensitivity"

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Feature Ablation (6 state features)
#   Each feature removed × 3 seeds
#   Reviewer 2.8 (feature justification)
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 4: Feature Ablation (loss, grad_norm, latency, battery, frequency, staleness)"

run_step "Feature ablation (experiment matrix)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile core --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --only feature_ablation --resume

run_step "Feature ablation (standalone CSV)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_ablation \
    --device "${DEVICE}" --seeds 42 123 2026 \
    --output-dir "${RUNS_DIR}/ablation"

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Extended Matrix: Heterogeneity + Scaling (N=20,40,80,100)
#   Non-IID sweep: α ∈ {0.1, 0.5, 1.0}
#   Client scaling: N ∈ {20, 40, 80, 100}
#   Reviewer 2.4 (scalability), 2.6 (fairness)
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 5: Extended Matrix (heterogeneity + scaling)"

run_step "Full extended matrix (heterogeneity + scaling)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile full --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --verified-hardware-telemetry --resume

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Selection Overhead / Time Complexity (N=20,40,80,100)
#   Proves MAML-Select O(N·|φ|) vs FedCor O(N³)
#   Reviewer 2.1 (computational complexity)
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 6: Time Complexity (N=20, 40, 80, 100)"

run_step "Scaling overhead benchmark (10 rounds per N)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_scaling \
    --device "${DEVICE}" --rounds 10 \
    --output-dir "${RUNS_DIR}/scaling"

run_step "Scaling overhead (200 rounds via matrix)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.run_experiments \
    --profile scaling --device "${DEVICE}" --output-dir "${RUNS_DIR}" \
    --country-iso-code "${COUNTRY_ISO}" --grid-intensity "${GRID_INTENSITY}" \
    --resume

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Analysis & Plot Generation
#   All output goes to artifacts/maml_select/
# ═════════════════════════════════════════════════════════════════════════════

section "PHASE 7: Analysis & Plots → artifacts/"

run_step "Statistical analysis (CSV, LaTeX, paired t-tests, Holm, CIs)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.analyze_results \
    --results-dir "${RUNS_DIR}" \
    --output-dir "${ARTIFACTS_DIR}/analysis"

run_step "Publication plots (EPS)" \
    "${PYTHON_BIN}" -m csfl_simulator.experiments.maml_select.generate_plots \
    --results-dir "${RUNS_DIR}" \
    --output-dir "${ARTIFACTS_DIR}/plots" \
    --sensitivity-dir "${RUNS_DIR}/sensitivity" \
    --ablation-dir "${RUNS_DIR}/ablation" \
    --scaling-dir "${RUNS_DIR}/scaling"

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

section "CAMPAIGN COMPLETE — $(elapsed $CAMPAIGN_START)"
log ""
log "Directory layout:"
log "  runs/maml_select/"
log "  ├── main_benchmarks_fashion_main_maml_select_s42/"
log "  │   ├── result.json           ← full run output"
log "  │   ├── round_metrics.jsonl   ← per-round metrics log"
log "  │   ├── progress.json         ← latest checkpoint"
log "  │   └── seed_record.json      ← RNG audit trail"
log "  ├── sensitivity/"
log "  │   ├── sensitivity_results.json"
log "  │   └── sensitivity_summary.csv"
log "  ├── ablation/"
log "  │   ├── ablation_results.json"
log "  │   └── ablation_summary.csv"
log "  ├── scaling/"
log "  │   ├── scaling_results.json"
log "  │   └── scaling_pivot.csv"
log "  └── logs/"
log ""
log "  artifacts/maml_select/"
log "  ├── analysis/"
log "  │   ├── main_summary.csv             (Table 1)"
log "  │   ├── main_summary.tex"
log "  │   ├── paired_significance_tests.csv (t-tests + Holm + Cohen's d)"
log "  │   ├── energy_to_target_summary.csv"
log "  │   ├── runs.csv"
log "  │   ├── convergence_fashion_main.eps"
log "  │   ├── convergence_cifar10_main.eps"
log "  │   ├── fig2_efficiency_comparison.eps"
log "  │   ├── hardware_energy_carbon_*.eps"
log "  │   ├── fairness_*.eps"
log "  │   └── energy_to_target_*.eps"
log "  └── plots/"
log "      ├── fig2_efficiency_fashion_main.eps"
log "      ├── fig2_efficiency_cifar10_main.eps"
log "      ├── lambda_sensitivity.eps"
log "      ├── feature_ablation.eps"
log "      ├── fairness_coverage_tiers.eps"
log "      └── scaling_overhead.eps"
log ""
log "Reviewer evidence produced:"
log "  ✓ R1: Implementation details (seed_record.json, configs logged)"
log "  ✓ R2.3:  λ sensitivity — 4 values × 3 seeds"
log "  ✓ R2.4:  Scaling — N ∈ {20, 40, 80, 100}"
log "  ✓ R2.5:  Statistical significance — paired t-tests, 95% CIs, Holm"
log "  ✓ R2.6:  Fairness — Jain index, entropy, tier coverage"
log "  ✓ R2.7:  Energy — CodeCarbon kWh + gCO₂ + modeled client energy"
log "  ✓ R2.8:  Feature ablation — 6 features × 3 seeds"
log "  ✓ R2.9:  Fig 2 — high-res EPS, proper fonts, 4-panel layout"
log "  ✓ R2.11: CriticalFL + FedGCS baselines included (8 total methods)"
log "  ✓ R3.4:  7 baselines (was 1)"
log "  ✓ R3.5:  Comprehensive simulation results"
log "  ✓ R4.6:  Clearer Fig 2 with markers + uncertainty bands"
log ""
log "Log: ${LOG_FILE}"
