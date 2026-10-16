# OCG-SSM

Online causal-graph-conditioned state-space forecasting with a pure-PyTorch
sequential scan. CPU is supported; GPU runs can be launched from Colab or any
environment with a CUDA-enabled PyTorch installation.

## Install

```bash
python -m pip install -e .
```

## Run experiments

Run a synthetic benchmark from the repository root:

```bash
python -m ocg.run \
  --config configs/synthetic_abrupt.yaml \
  --output results/synthetic_abrupt.json \
  --device cpu
```

Other checked-in synthetic, ablation, and stress configs can be run the same
way:

```bash
python -m ocg.run --config configs/synthetic_gradual.yaml
python -m ocg.run --config configs/ablation.yaml
python -m ocg.run --config configs/stress.yaml
```

The ablation config compares identity, learned, random, and oracle graph modes.
Results include forecast errors, graph scores when synthetic ground truth is
available, online adaptation metrics, efficiency measurements, and regret
against the oracle variant when one is configured. Results are written as JSON
under `results/` by default.

### ETT, Electricity, and Traffic CSVs

Place the numeric time-series CSV at the path in the selected config, or
override it with `--data-path`. For example:

```bash
python -m ocg.run \
  --config configs/ett.yaml \
  --data-path /path/to/ETTh1.csv \
  --output results/ett.json \
  --device cpu
python -m ocg.run \
  --config configs/electricity.yaml \
  --data-path /path/to/electricity.csv \
  --output results/electricity.json
python -m ocg.run \
  --config configs/traffic.yaml \
  --data-path /path/to/traffic.csv \
  --output results/traffic.json
```

CSV files must contain at least one numeric column and enough rows for the
configured chronological train, validation, and test splits. The runner does
not download datasets.

For CUDA, use `--device cuda` with a CUDA-enabled PyTorch build. Hyperparameters
and dataset paths are configured in `configs/*.yaml`.

## Google Colab

Open [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) in Colab. It
clones the repository, installs the project, detects the available accelerator,
runs a synthetic config, and writes the JSON result to Google Drive.

## Tests

```bash
pytest -q
```
