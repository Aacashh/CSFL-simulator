# Official FedGCS Comparison Protocol

FedGCS is not replaced with an in-simulator approximation. Its official IJCAI
2024 repository implements a generative optimization pipeline with its own
environment and should be run directly:

```bash
bash csfl_simulator/experiments/maml_select/prepare_official_fedgcs.sh
cd external/GenerativeFL
conda create -n GenerativeFL python=3.9
conda activate GenerativeFL
pip install -r requirements.txt
bash mnist-lenet5-iid.sh
```

The command above is the official repository's smoke test. For a manuscript
comparison, configure its experiment to use the same dataset, Dirichlet split,
number of clients, cohort size, rounds, local epochs, batch size, model, and
seed as the corresponding MAML-Select scenario. Record any unavoidable
framework-level differences in the paper.

After each comparable official run, add one row to a copy of
`official_baselines_template.csv`. Pass that CSV to the analysis command:

```bash
python -m csfl_simulator.experiments.maml_select.analyze_results \
  --external-csv path/to/official_baselines.csv
```

Do not label the integrated `ml.fedcor` implementation as an exact FedCor
reproduction: it is the repository's existing approximation. Do not label any
local method as FedGCS unless it came from the official repository.

Official repository: <https://github.com/zhiyuan-ning/GenerativeFL>

## Upstream Reproducibility Note

The official repository snapshot cloned on June 2, 2026 was commit
`785773b2a6675048aa3ab2ac7aaf0324b802f368`. Its generator currently exits when
CUDA is unavailable (`autos/train.py`). The published `main.py` also contains
placeholder candidate selectors and a `choose_clients_favor` /
`choose_clients_fovar` naming mismatch. Therefore, the unmodified upstream
snapshot cannot produce a comparable CPU-only FedGCS run on this host.

Do not silently patch these issues and describe the output as an exact official
reproduction. Any compatibility patch must be versioned, disclosed in the
paper, and validated independently before importing a FedGCS row.
