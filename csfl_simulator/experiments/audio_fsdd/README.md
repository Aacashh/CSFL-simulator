# FSDD Audio Experiment

This folder runs a small auxiliary audio-modality experiment for the MAML-Select
revision using the Free Spoken Digit Dataset (FSDD).

The workflow runs all eight selectors for 100 communication rounds:

- FedAvg
- FedCS
- Oort
- TiFL
- FedCor
- CriticalFL
- FedGCS
- MAML-Select

Default setup:

- Dataset: Free Spoken Digit Dataset, converted to 32x32 log-spectrograms
- Model: `AudioCNN`
- Clients: `N=30`
- Selected clients per round: `K=5`
- Rounds: `100`
- Local epochs: `2`
- Batch size: `16`
- Non-IID split: label-shard, two shards per client
- Seed: `42`

The smaller `N` and `K` are deliberate. FSDD has only 3,000 clips, so using
100 clients with Dirichlet splitting can create empty clients and make the
quick auxiliary experiment fragile.

Run from the repository root:

```bash
bash csfl_simulator/experiments/audio_fsdd/setup_and_run.sh
```

On Windows PowerShell, run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File csfl_simulator\experiments\audio_fsdd\setup_and_run_windows.ps1
```

Or use the CMD wrapper:

```cmd
csfl_simulator\experiments\audio_fsdd\setup_and_run_windows.cmd
```

Useful overrides:

```bash
DEVICE=cuda bash csfl_simulator/experiments/audio_fsdd/setup_and_run.sh
DEVICE=mps bash csfl_simulator/experiments/audio_fsdd/setup_and_run.sh
NO_HARDWARE_METER=0 bash csfl_simulator/experiments/audio_fsdd/setup_and_run.sh
```

Windows PowerShell overrides:

```powershell
$env:DEVICE="cuda"; powershell -NoProfile -ExecutionPolicy Bypass -File csfl_simulator\experiments\audio_fsdd\setup_and_run_windows.ps1
$env:NO_HARDWARE_METER="0"; powershell -NoProfile -ExecutionPolicy Bypass -File csfl_simulator\experiments\audio_fsdd\setup_and_run_windows.ps1
```

Outputs:

- JSON logs: `runs/audio_fsdd/`
- CSV/LaTeX analysis: `artifacts/audio_fsdd/analysis/`
- EPS plots: `artifacts/audio_fsdd/plots/`
