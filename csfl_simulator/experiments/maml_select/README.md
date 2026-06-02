# MAML-Select Resubmission Experiments

This additive suite implements the manuscript's MAML-Select algorithm without
changing legacy simulator files. It prepares the experimental evidence needed
for the IEEE TAI letter revision:

- repeated runs with paired significance tests;
- Fashion-MNIST and CIFAR-10 main benchmarks;
- client-count and non-IID scaling analysis;
- lambda sensitivity and six state-feature ablations;
- participation fairness, tier selection, computation, communication, modeled
  client energy, hardware compute energy, and estimated carbon emissions;
- an integrated CriticalFL cohort-augmentation reproduction;
- a disclosed in-simulator FedGCS-style approximation for controlled loops;
- a separate bootstrap path for the official FedGCS codebase.

## Install

From the repository root:

```bash
pip install -e .
pip install -r csfl_simulator/experiments/maml_select/requirements.txt
python scripts/download_data.py --datasets fashion-mnist cifar10
```

Use a dedicated experiment host with no competing workloads. Keep its GPU power
cap, software environment, and declared grid intensity fixed across methods.
Run `codecarbon detect` before collecting hardware evidence. Add
`--verified-hardware-telemetry` only after confirming that the host exposes real
telemetry. On Apple Silicon, CodeCarbon uses `powermetrics`, which requires
`sudo`; otherwise it falls back to an estimate.

## Reproducibility Defaults

The primary protocol matches the letter: `N=100`, `K=10`, `T=200`, `E=5`,
batch size `32`, Dirichlet `alpha=0.5`, and seeds `42`, `123`, and `2026`.
Client compute rates follow the declared `20% / 50% / 30%` tier split at
`1x / 2x / 4x`. Local models use SGD with learning rate `0.01`. The
`6-64-64-1` MAML-Select policy uses PyTorch's seeded linear-layer
initialization, one inner gradient-descent step at `0.01`, and an Adam outer
optimizer at `0.001`. `T_target` is the mean Tier-2 latency.

MAML-Select uses a disclosed cold-start strategy: one initial coverage pass
over the client pool before policy-only exploitation begins. This is `10`
rounds in the primary `N=100`, `K=10` protocol. Later rounds reserve one cohort
slot for the stalest client. This supplies representative online feedback and
avoids permanently excluding unobserved clients. The latency excess in the
policy target is divided by `T_target`, making the loss-latency trade-off
dimensionless while preserving the soft-deadline interpretation. The inner
support set contains the immediately preceding round's observed client costs,
as defined in the manuscript.

## Run

For a fresh workstation clone, the supported one-command workflow is:

```bash
bash csfl_simulator/experiments/maml_select/setup_and_run.sh
```

It writes JSON logs under `runs/maml_select/`, CSV and LaTeX analysis tables
under `artifacts/maml_select/analysis/`, and EPS-only publication figures under
`artifacts/maml_select/plots/`. Each run may also contain a nested `_scratch/`
directory for the base simulator's audit logs.

Validate the environment first. This prints the matrix and does not train:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile quick --device cuda --dry-run
```

Run the optional local CPU pilot before committing a remote host:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile pilot --device cpu --no-hardware-meter --resume
```

Run the optional local lambda diagnostic:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile pilot_lambda --device cpu --no-hardware-meter --resume
```

Run the manuscript-critical matrix:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile core --device cuda --country-iso-code IND \
  --grid-intensity 475 --verified-hardware-telemetry --resume
```

Run the extended reviewer matrix:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile full --device cuda --country-iso-code IND \
  --grid-intensity 475 --verified-hardware-telemetry --resume
```

On a CPU-only host, start the same full 200-round reviewer matrix as a resumable
background campaign:

```bash
PYTHON_BIN=/path/to/python \
  bash csfl_simulator/experiments/maml_select/start_local_cpu_campaign.sh
bash csfl_simulator/experiments/maml_select/show_local_cpu_campaign.sh
```

The CPU campaign writes an append-only `round_metrics.jsonl` file and a
periodically refreshed `progress.json` checkpoint inside each active run
directory. Completed runs remain resumable through `--resume`.

Measure energy-to-accuracy on a dedicated host:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile energy --device cuda --country-iso-code IND \
  --grid-intensity 475 --verified-hardware-telemetry --resume
```

`--resume` skips completed result files after an interruption. Use
`--only lambda_sensitivity` or another experiment ID to run one block.

## Analyze

```bash
python -m csfl_simulator.experiments.maml_select.analyze_results
```

The analysis command writes CSV and LaTeX tables, standard deviations, paired
tests, paired effect sizes, 95% confidence intervals, and Holm-adjusted
p-values under `artifacts/maml_select/analysis/`. The plot command writes the
larger Figure 2 replacement and reviewer-requested plots as EPS files only:

```bash
python -m csfl_simulator.experiments.maml_select.generate_plots
```

Open
`plot_results.ipynb` in JupyterLab for an interactive summary:

```bash
jupyter lab csfl_simulator/experiments/maml_select/plot_results.ipynb
```

## Interpret Energy Carefully

The suite reports two different quantities:

1. `cum_modelled_energy_wh` attributes client-level energy per FL round using
   declared tier power proxies. It supports mechanistic analysis.
2. `measured_energy_kwh` is collected around `simulator.run()` with
   `codecarbon.OfflineEmissionsTracker`, where host telemetry is available.

Carbon emissions remain estimates derived from electricity use and a declared
grid-intensity value. Check each run's `hardware_energy.status`; do not present
hardware-energy bars unless the tracker reports `measured`.

On macOS, CodeCarbon may fall back to a CPU-TDP estimate when `powermetrics`
telemetry is not configured. Such runs are marked `tracked_unverified`. The
analysis exports them separately as tracked estimates and never labels them as
direct measurements.

The `energy` profile stops each run after reaching the shared dataset accuracy
target, or at the shared round cap when the target is not reached. This allows
energy-to-accuracy comparisons while preserving did-not-reach outcomes.

## Recent Baselines

`research.criticalfl` reproduces CriticalFL's random-base cohort augmentation
mechanism using relative federated-gradient-norm changes and geometric cohort
growth. `research.fedgcs` is a disclosed FedGCS-style approximation for
controlled in-simulator comparisons. It is not the official IJCAI
implementation. Follow `official_fedgcs_protocol.md` before adding a direct
FedGCS row to the paper.

The integrated FedCor implementation is the repository's existing
`FedCor (approx.)` selector. Label it as an approximation in plots and use an
official implementation before making an exact FedCor head-to-head claim.

The practical local overhead sweep uses `N = 20, 40, 80, 100`. It is intended
to show the observed scaling trend without making the campaign prohibitively
long.
