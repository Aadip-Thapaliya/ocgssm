# OCG-SSM

Online causal-graph-conditioned state-space forecasting with a pure-PyTorch
sequential scan. CPU is supported; GPU runs can be launched from Colab or any
environment with a CUDA-enabled PyTorch installation.

## Install

```bash
python -m pip install -e .
```

### This computer's isolated Windows setup

The ready-to-use copy is `C:\Users\aadip\Desktop\OCGSSM-Local`.
It has a private Python 3.12.12 installation in `.python` and an isolated
environment in `.venv`; existing Python installations and global packages
were not changed. Keep both directories with this copy.

Double-click `Run-Synthetic.cmd` to run the synthetic benchmark on CPU and
save `results\synthetic_abrupt_local.json`. Double-click
`Open-Project-Terminal.cmd` for a terminal with this environment activated.
Neither launcher requires changing PowerShell's execution policy.

To run directly from PowerShell:

```powershell
Set-Location 'C:\Users\aadip\Desktop\OCGSSM-Local'
.\.venv\Scripts\python.exe -m ocg.run --config configs\synthetic_abrupt.yaml --output results\synthetic_abrupt_local.json --device cpu
.\.venv\Scripts\python.exe -m pytest
```

All required packages, including the test dependencies, are installed.
`requirements-local-lock.txt` records the installed package versions.
This copy includes a Windows peak-memory compatibility fix: the runner uses
`psutil` on Windows instead of importing the Unix-only `resource` module.

CPU execution was verified on this laptop; it has Intel Iris Xe graphics,
not an NVIDIA CUDA GPU. Real-data configs still require the user's dataset
CSVs; none are downloaded automatically. See `SETUP.txt` for exact paths
and verification results.

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

The configs compare identity, dense, learned, density-matched random, and
oracle graph modes. Identity includes self-lags only; dense and random masks
include self-lags consistently with the generated VAR graph. Results include
forecast errors, timestamp-aligned graph scores when synthetic ground truth
is available, online adaptation metrics, efficiency measurements, and regret
against the oracle variant when one is configured. Results are written as JSON
under `results/` by default.

`configs/research_online.yaml` defines matched online graph controls,
forecasting baselines, and ablations. `configs/synthetic_abrupt.yaml` uses a
cold start; `configs/synthetic_abrupt_warm.yaml` uses only the training split
for warm-up. These protocols must not be mixed in comparisons.

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
