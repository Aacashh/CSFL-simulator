#!/bin/bash
# =============================================================================
# Submit experiments in waves of 4 jobs (HPC queue limit)
#
# Estimated times are based on: 1 method × ResNet18 × CIFAR-10 × 200 rounds ≈ 37 min
#
# Usage:
#   bash scripts/submit_waves.sh 1          # Submit Wave 1 (4 jobs)
#   bash scripts/submit_waves.sh 2          # Submit Wave 2 (4 jobs)
#   bash scripts/submit_waves.sh 3          # Submit Wave 3 (4 jobs)
#   bash scripts/submit_waves.sh all        # Submit all waves with dependencies
#   bash scripts/submit_waves.sh status     # Check all job statuses
#   bash scripts/submit_waves.sh dry        # Print what would be submitted
#
# Wave Plan:
#   Wave 1 — APEX core      (~20-24h each, 4 jobs)
#   Wave 2 — APEX scale + FD light (~12-14h each, 4 jobs)
#   Wave 3 — FD heavy        (~18-24h each, 4 jobs)
#
# Total: 12 jobs across 3 waves ≈ 200 GPU-hours
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/hpc_job.sh"

mkdir -p logs

DRY=false
[[ "${1:-}" == "dry" ]] && DRY=true

submit() {
    local job_name="$1" time="$2" desc="$3"
    shift 3
    # remaining args are passed to hpc_job.sh → run_all_experiments.sh
    local extra_args="$*"

    echo "  ${job_name} (${time}, ${desc})"
    echo "    → ${extra_args}"

    if $DRY; then
        echo "    [DRY] sbatch --job-name=${job_name} --time=${time} ${JOB_SCRIPT} ${extra_args}"
        echo ""
        return
    fi

    local job_id
    job_id=$(sbatch \
        --job-name="${job_name}" \
        --time="${time}" \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --parsable \
        "${JOB_SCRIPT}" ${extra_args})

    echo "    Submitted: Job ${job_id}"
    echo ""
}

case "${1:-help}" in

# =========================================================================
# WAVE 1: APEX core experiments
# Estimated: ~15-24h per job
# =========================================================================
1)
    echo ""
    echo "=== WAVE 1: APEX Core (4 jobs) ==="
    echo ""

    # Job 1: Main benchmark — 8 methods × 3 seeds × 37m = ~15h
    submit "apex-main" "18:00:00" "~15h: Main benchmark (Table I)" \
        --exp apex:1 --resume

    # Job 2: Heterogeneity — (5+6) methods × 3 seeds × 37m = ~20h
    submit "apex-het" "24:00:00" "~20h: Heterogeneity sweep (Table II)" \
        --exp apex:2 --resume

    # Job 3: Ablation + IID + v1v2 — 6×3 + 4×3 + 2×4 = ~23h
    submit "apex-abl-iid" "24:00:00" "~23h: Ablation + IID + v1v2 (Tables IV, VI)" \
        --exp "apex:4,apex:6,apex:7" --resume

    # Job 4: Cross-dataset + LightCNN — (5+5+4)×3 + 5×3 = ~18h
    submit "apex-data" "20:00:00" "~18h: Cross-dataset + LightCNN (Table V)" \
        --exp "apex:5,apex:8" --resume

    echo "  Submit Wave 2 when these complete: bash scripts/submit_waves.sh 2"
    ;;

# =========================================================================
# WAVE 2: APEX scalability (split by seed) + FD light experiments
# Estimated: ~12-14h per job
# =========================================================================
2)
    echo ""
    echo "=== WAVE 2: APEX Scale + FD Light (4 jobs) ==="
    echo ""

    # Job 5: Scalability seed 42 — N={100,200,500} × 4 methods = ~12h
    submit "apex-scale-s42" "14:00:00" "~12h: Scalability all N, seed 42" \
        --exp apex:3 --seed 42 --resume

    # Job 6: Scalability seed 123
    submit "apex-scale-s123" "14:00:00" "~12h: Scalability all N, seed 123" \
        --exp apex:3 --seed 123 --resume

    # Job 7: Scalability seed 456
    submit "apex-scale-s456" "14:00:00" "~12h: Scalability all N, seed 456" \
        --exp apex:3 --seed 456 --resume

    # Job 8: FD light — Exps 1+2+6+7+8+9 = ~18h
    submit "fd-light" "20:00:00" "~18h: FD main + baseline + scale + group + MNIST + comm" \
        --exp "fd:1,fd:2,fd:6,fd:7,fd:8,fd:9" --resume

    echo "  Submit Wave 3 when these complete: bash scripts/submit_waves.sh 3"
    ;;

# =========================================================================
# WAVE 3: FD heavy experiments (sweeps + ablation + multi-seed)
# Estimated: ~15-24h per job
# =========================================================================
3)
    echo ""
    echo "=== WAVE 3: FD Heavy (4 jobs) ==="
    echo ""

    # Job 9: Noise sensitivity sweep — 7 methods × 5 SNR levels = ~15h
    submit "fd-noise" "18:00:00" "~15h: Noise sensitivity sweep (5 SNR levels)" \
        --exp fd:3 --resume

    # Job 10: Alpha sweep — 7 methods × 6 alphas = ~18h
    submit "fd-alpha" "20:00:00" "~18h: Non-IID alpha sweep (6 levels)" \
        --exp fd:4 --resume

    # Job 11: K sweep + antenna sweep = ~15h + 5h = ~20h
    submit "fd-sweeps" "24:00:00" "~20h: K sweep + antenna sweep" \
        --exp "fd:5,fd:10" --resume

    # Job 12: Ablation + multi-seed = ~4h + ~13h = ~17h
    submit "fd-abl-seed" "20:00:00" "~17h: FD ablation + multi-seed significance" \
        --exp "fd:11,fd:12" --resume

    echo "  All waves submitted! Run: bash scripts/submit_waves.sh status"
    ;;

# =========================================================================
# Submit all waves with SLURM dependencies
# =========================================================================
all)
    echo ""
    echo "=== Submitting ALL 3 waves (12 jobs) with dependencies ==="
    echo ""
    echo "--- Wave 1: APEX Core ---"

    W1_1=$(sbatch --job-name=apex-main    --time=18:00:00 --output=logs/apex-main_%j.out    --error=logs/apex-main_%j.err    --parsable "${JOB_SCRIPT}" --exp apex:1 --resume)
    echo "  apex-main:    Job ${W1_1}"
    W1_2=$(sbatch --job-name=apex-het     --time=24:00:00 --output=logs/apex-het_%j.out     --error=logs/apex-het_%j.err     --parsable "${JOB_SCRIPT}" --exp apex:2 --resume)
    echo "  apex-het:     Job ${W1_2}"
    W1_3=$(sbatch --job-name=apex-abl-iid --time=24:00:00 --output=logs/apex-abl-iid_%j.out --error=logs/apex-abl-iid_%j.err --parsable "${JOB_SCRIPT}" --exp "apex:4,apex:6,apex:7" --resume)
    echo "  apex-abl-iid: Job ${W1_3}"
    W1_4=$(sbatch --job-name=apex-data    --time=20:00:00 --output=logs/apex-data_%j.out    --error=logs/apex-data_%j.err    --parsable "${JOB_SCRIPT}" --exp "apex:5,apex:8" --resume)
    echo "  apex-data:    Job ${W1_4}"

    W1_DEP="${W1_1}:${W1_2}:${W1_3}:${W1_4}"

    echo ""
    echo "--- Wave 2: APEX Scale + FD Light (after Wave 1) ---"

    W2_1=$(sbatch --dependency=afterany:${W1_DEP} --job-name=apex-scale-s42  --time=14:00:00 --output=logs/apex-scale-s42_%j.out  --error=logs/apex-scale-s42_%j.err  --parsable "${JOB_SCRIPT}" --exp apex:3 --seed 42 --resume)
    echo "  apex-scale-s42:  Job ${W2_1} (after ${W1_DEP})"
    W2_2=$(sbatch --dependency=afterany:${W1_DEP} --job-name=apex-scale-s123 --time=14:00:00 --output=logs/apex-scale-s123_%j.out --error=logs/apex-scale-s123_%j.err --parsable "${JOB_SCRIPT}" --exp apex:3 --seed 123 --resume)
    echo "  apex-scale-s123: Job ${W2_2}"
    W2_3=$(sbatch --dependency=afterany:${W1_DEP} --job-name=apex-scale-s456 --time=14:00:00 --output=logs/apex-scale-s456_%j.out --error=logs/apex-scale-s456_%j.err --parsable "${JOB_SCRIPT}" --exp apex:3 --seed 456 --resume)
    echo "  apex-scale-s456: Job ${W2_3}"
    W2_4=$(sbatch --dependency=afterany:${W1_DEP} --job-name=fd-light        --time=20:00:00 --output=logs/fd-light_%j.out        --error=logs/fd-light_%j.err        --parsable "${JOB_SCRIPT}" --exp "fd:1,fd:2,fd:6,fd:7,fd:8,fd:9" --resume)
    echo "  fd-light:        Job ${W2_4}"

    W2_DEP="${W2_1}:${W2_2}:${W2_3}:${W2_4}"

    echo ""
    echo "--- Wave 3: FD Heavy (after Wave 2) ---"

    W3_1=$(sbatch --dependency=afterany:${W2_DEP} --job-name=fd-noise    --time=18:00:00 --output=logs/fd-noise_%j.out    --error=logs/fd-noise_%j.err    --parsable "${JOB_SCRIPT}" --exp fd:3 --resume)
    echo "  fd-noise:    Job ${W3_1} (after ${W2_DEP})"
    W3_2=$(sbatch --dependency=afterany:${W2_DEP} --job-name=fd-alpha    --time=20:00:00 --output=logs/fd-alpha_%j.out    --error=logs/fd-alpha_%j.err    --parsable "${JOB_SCRIPT}" --exp fd:4 --resume)
    echo "  fd-alpha:    Job ${W3_2}"
    W3_3=$(sbatch --dependency=afterany:${W2_DEP} --job-name=fd-sweeps   --time=24:00:00 --output=logs/fd-sweeps_%j.out   --error=logs/fd-sweeps_%j.err   --parsable "${JOB_SCRIPT}" --exp "fd:5,fd:10" --resume)
    echo "  fd-sweeps:   Job ${W3_3}"
    W3_4=$(sbatch --dependency=afterany:${W2_DEP} --job-name=fd-abl-seed --time=20:00:00 --output=logs/fd-abl-seed_%j.out --error=logs/fd-abl-seed_%j.err --parsable "${JOB_SCRIPT}" --exp "fd:11,fd:12" --resume)
    echo "  fd-abl-seed: Job ${W3_4}"

    echo ""
    echo "=== All 12 jobs queued ==="
    echo "  Wave 1: immediate"
    echo "  Wave 2: starts after Wave 1 finishes"
    echo "  Wave 3: starts after Wave 2 finishes"
    echo ""
    echo "  Monitor: bash scripts/submit_waves.sh status"
    echo "  Cancel all: scancel ${W1_1} ${W1_2} ${W1_3} ${W1_4} ${W2_1} ${W2_2} ${W2_3} ${W2_4} ${W3_1} ${W3_2} ${W3_3} ${W3_4}"
    ;;

# =========================================================================
# Check status
# =========================================================================
status)
    echo ""
    echo "=== CSFL Job Status ==="
    echo ""
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %.6D %R" --sort=i | \
        grep -E "JOBID|apex|fd|csfl" || echo "  No CSFL jobs in queue"
    echo ""
    echo "=== Recent completed jobs (last 24h) ==="
    sacct -u "$USER" --starttime="$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo 'now-24hours')" \
        -o "JobID%10,JobName%20,State%12,Elapsed%10,MaxRSS%10" \
        --name="apex-main,apex-het,apex-abl-iid,apex-data,apex-scale-s42,apex-scale-s123,apex-scale-s456,fd-light,fd-noise,fd-alpha,fd-sweeps,fd-abl-seed" \
        2>/dev/null || echo "  (sacct not available)"
    echo ""
    echo "=== Completed runs ==="
    python -m csfl_simulator list-runs 2>/dev/null | tail -20 || echo "  (list-runs not available)"
    ;;

# =========================================================================
# Dry run
# =========================================================================
dry)
    echo ""
    echo "=== DRY RUN: All 3 waves ==="
    DRY=true

    echo ""
    echo "--- Wave 1: APEX Core (submit immediately) ---"
    submit "apex-main"    "18:00:00" "~15h: Main benchmark" --exp apex:1
    submit "apex-het"     "24:00:00" "~20h: Heterogeneity"  --exp apex:2
    submit "apex-abl-iid" "24:00:00" "~23h: Ablation+IID+v1v2" --exp "apex:4,apex:6,apex:7"
    submit "apex-data"    "20:00:00" "~18h: Cross-dataset+LightCNN" --exp "apex:5,apex:8"

    echo "--- Wave 2: APEX Scale + FD Light (after Wave 1) ---"
    submit "apex-scale-s42"  "14:00:00" "~12h: Scale seed 42"  --exp apex:3 --seed 42
    submit "apex-scale-s123" "14:00:00" "~12h: Scale seed 123" --exp apex:3 --seed 123
    submit "apex-scale-s456" "14:00:00" "~12h: Scale seed 456" --exp apex:3 --seed 456
    submit "fd-light"        "20:00:00" "~18h: FD light exps"  --exp "fd:1,fd:2,fd:6,fd:7,fd:8,fd:9"

    echo "--- Wave 3: FD Heavy (after Wave 2) ---"
    submit "fd-noise"    "18:00:00" "~15h: Noise sweep"      --exp fd:3
    submit "fd-alpha"    "20:00:00" "~18h: Alpha sweep"      --exp fd:4
    submit "fd-sweeps"   "24:00:00" "~20h: K+antenna sweeps" --exp "fd:5,fd:10"
    submit "fd-abl-seed" "20:00:00" "~17h: Ablation+seeds"   --exp "fd:11,fd:12"
    ;;

# =========================================================================
# Help
# =========================================================================
*)
    echo ""
    echo "Usage: bash scripts/submit_waves.sh {1|2|3|all|status|dry}"
    echo ""
    echo "  1       Submit Wave 1: APEX core (4 jobs, ~18-24h each)"
    echo "  2       Submit Wave 2: APEX scalability + FD light (4 jobs, ~12-20h)"
    echo "  3       Submit Wave 3: FD heavy sweeps (4 jobs, ~15-24h each)"
    echo "  all     Submit all 12 jobs with SLURM dependencies (auto-chained)"
    echo "  status  Check running/completed jobs"
    echo "  dry     Print what would be submitted"
    echo ""
    echo "  Wave Plan:"
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │ Wave 1 (4 jobs, ~24h)                                     │"
    echo "  │   apex-main      APEX 1: main benchmark          ~15h    │"
    echo "  │   apex-het       APEX 2: heterogeneity sweep     ~20h    │"
    echo "  │   apex-abl-iid   APEX 4+6+7: ablation/IID/v1v2  ~23h    │"
    echo "  │   apex-data      APEX 5+8: cross-dataset/LightCNN ~18h  │"
    echo "  ├─────────────────────────────────────────────────────────────┤"
    echo "  │ Wave 2 (4 jobs, ~14h)                after Wave 1         │"
    echo "  │   apex-scale-s42   APEX 3: N=100/200/500 seed 42  ~12h  │"
    echo "  │   apex-scale-s123  APEX 3: N=100/200/500 seed 123 ~12h  │"
    echo "  │   apex-scale-s456  APEX 3: N=100/200/500 seed 456 ~12h  │"
    echo "  │   fd-light         FD 1+2+6+7+8+9: light FD exps  ~18h  │"
    echo "  ├─────────────────────────────────────────────────────────────┤"
    echo "  │ Wave 3 (4 jobs, ~24h)                after Wave 2         │"
    echo "  │   fd-noise       FD 3: noise sensitivity (5 SNR)  ~15h   │"
    echo "  │   fd-alpha       FD 4: alpha sweep (6 levels)     ~18h   │"
    echo "  │   fd-sweeps      FD 5+10: K + antenna sweeps      ~20h   │"
    echo "  │   fd-abl-seed    FD 11+12: ablation + multi-seed  ~17h   │"
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Total: 12 jobs, ~200 GPU-hours, ~3 calendar days (with queuing)"
    echo ""
    ;;
esac
