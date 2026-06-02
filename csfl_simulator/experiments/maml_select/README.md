# MAML-Select Resubmission Experiments

This additive suite implements the manuscript's MAML-Select algorithm without
changing legacy simulator files. It prepares the experimental evidence needed
for the IEEE TAI letter revision:

- repeated runs with paired significance tests;
- Fashion-MNIST and CIFAR-10 main benchmarks;
- CIFAR-100 larger-benchmark evaluation;
- client-count and non-IID scaling analysis;
- lambda sensitivity and six state-feature ablations;
- participation fairness, tier selection, computation, communication, modeled
  client energy, hardware compute energy, and estimated carbon emissions;
- an integrated CriticalFL cohort-augmentation reproduction;
- a separate bootstrap path for the official FedGCS codebase.

## Install

From the repository root:

```bash
pip install -e .
pip install -r csfl_simulator/experiments/maml_select/requirements.txt
python scripts/download_data.py --datasets fashion-mnist cifar10 cifar100
```

Use a dedicated experiment host with no competing workloads. Keep its GPU power
cap, software environment, and declared grid intensity fixed across methods.

## Reproducibility Defaults

The primary protocol matches the letter: `N=100`, `K=10`, `T=200`, `E=5`,
batch size `32`, Dirichlet `alpha=0.5`, and seeds `42`, `123`, and `2026`.
Client compute rates follow the declared `20% / 50% / 30%` tier split at
`1x / 2x / 4x`. Local models use SGD with learning rate `0.01`. The
`6-64-64-1` MAML-Select policy uses PyTorch's seeded linear-layer
initialization, one inner gradient-descent step at `0.01`, and an Adam outer
optimizer at `0.001`. `T_target` is the mean Tier-2 latency.

## Run

Validate the environment first. This prints the matrix and does not train:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile quick --device cuda --dry-run
```

Run the manuscript-critical matrix:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile core --device cuda --country-iso-code IND \
  --grid-intensity 475 --resume
```

Run the extended reviewer matrix:

```bash
bash csfl_simulator/experiments/maml_select/run_suite.sh \
  --profile full --device cuda --country-iso-code IND \
  --grid-intensity 475 --resume
```

`--resume` skips completed result files after an interruption. Use
`--only lambda_sensitivity` or another experiment ID to run one block.

## Analyze

```bash
python -m csfl_simulator.experiments.maml_select.analyze_results
```

The command writes CSV and LaTeX tables, standard deviations, paired tests,
paired effect sizes, 95% confidence intervals, Holm-adjusted p-values, and
high-resolution vector EPS figures under `artifacts/maml_select_letter/analysis/`.
PDF and 600-DPI PNG companions are exported for inspection. Open
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
hardware-energy bars when the tracker reports `unavailable`.

## Recent Baselines

`research.criticalfl` reproduces CriticalFL's random-base cohort augmentation
mechanism using relative federated-gradient-norm changes and geometric cohort
growth. FedGCS uses its official external implementation. Follow
`official_fedgcs_protocol.md` before adding a direct FedGCS row to the paper.
