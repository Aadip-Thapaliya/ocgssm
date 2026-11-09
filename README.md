# OCG-SSM: Online Causal-Graph-Conditioned State-Space Forecasting

[![Tests](https://github.com/Aadip-Thapaliya/ocgssm/actions/workflows/tests.yml/badge.svg)](https://github.com/Aadip-Thapaliya/ocgssm/actions/workflows/tests.yml)
[![Citation](https://github.com/Aadip-Thapaliya/ocgssm/actions/workflows/citation.yml/badge.svg)](https://github.com/Aadip-Thapaliya/ocgssm/actions/workflows/citation.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

OCG-SSM is a multivariate time-series forecaster built for a master's thesis.
It learns a sparse, lagged dependency graph between the input variables and
uses that graph to mask the cross-variable inputs of a state-space model
written in pure PyTorch. During online forecasting the graph can be updated,
and a regime-switch detector decides when the graph should adapt to a change
in the data-generating process.

The repository contains the model, the baselines it is compared against,
synthetic data generators with known ground-truth graphs, a leak-safe online
evaluation protocol, the frozen experiment protocols, and the scripts and
summaries that produce every table and figure in the thesis.

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Tests](#tests)
- [Data](#data)
- [Results](#results)
- [Reproducing the thesis](#reproducing-the-thesis)
- [Citation](#citation)
- [License](#license)

## How it works

1. **Graph learning** (`src/ocg/graph.py`, `src/ocg/fast_graph.py`). A lagged
   adjacency over the D input variables is learned with a sparsity penalty. In
   online mode an exponential moving average of the edge logits is the graph
   used by the forecaster.
2. **Graph-conditioned SSM** (`src/ocg/ssm.py`, `src/ocg/model.py`). Each
   variable's state is driven only by the lags of the variables the graph
   connects to it. The sequential scan is plain PyTorch and runs on CPU.
3. **Regime detection** (`src/ocg/graph.py`). A detector tracks a running
   mean and variance of a structural score and flags windows whose z-score
   exceeds a threshold; confirmed detections trigger graph updates.
4. **Evaluation** (`src/ocg/eval.py`, `src/ocg/train.py`, `scripts/`).
   Offline training on a chronological split, prequential online evaluation
   in which every window is forecast before its targets are used, and paired
   circular block-bootstrap confidence intervals for MSE ratios.

The same model is run with five graph modes so that only the graph changes
between comparisons: `identity` (self-lags only), `dense` (all edges),
`random` (density-matched), `oracle` (true generator graph, synthetic data
only) and `learned`. DirectVAR and DLinear (`src/ocg/baselines.py`) are
reported as external baselines.

## Repository layout

```
src/ocg/                  model, graph learning, baselines, training and evaluation
tests/                    pytest suite
configs/                  YAML configs for the general runner (python -m ocg.run)
scripts/                  thesis stage runners and the export script
results/roadmap/          frozen stage protocols (gates.yaml) and stage summaries
results/thesis_export_c2/ final thesis tables (CSV) and figures (PNG and PDF)
results/d20_equal_training/  protocol files used by the D20 tests
notebooks/                Colab notebook for running a config on a GPU
data/                     place the ETT CSVs here (not tracked)
```

## Installation

Python 3.12 is recommended; the pinned requirements and CI use it. The thesis runs used Python 3.12.12 on CPU.

```bash
git clone https://github.com/Aadip-Thapaliya/ocgssm.git
cd ocgssm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` pins the versions used for the thesis. For a CPU-only
PyTorch build on Linux, install torch first with
`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

## Quick start

Run a config with the general runner:

```bash
python -m ocg.run --config configs/synthetic_abrupt.yaml --output results/synthetic_abrupt.json --device cpu
```

The JSON output holds forecast errors per variant, graph-recovery scores
against the true synthetic graph, online adaptation metrics and runtime.
Other configs in `configs/` are run the same way. Real-data configs
(`ett.yaml`, `electricity.yaml`, `traffic.yaml`) take the CSV path from the
config or from `--data-path`.

To use a GPU, pass `--device cuda` with a CUDA build of PyTorch, or open
`notebooks/colab_train.ipynb` in Google Colab.

## Tests

```bash
python -m pytest
```

The suite has 96 tests. One test in `tests/test_roadmap_final.py` is skipped
when `data/ETTh1.csv` is not present. GitHub Actions runs the suite on every
push and pull request.

## Data

The synthetic stages generate sparse structural VAR data from fixed seeds in
code; no download is needed.

The real-data stage (S6) uses ETTh1 and ETTh2 from the ETDataset repository
(https://github.com/zhouhaoyi/ETDataset). Download `ETTh1.csv` and `ETTh2.csv`
from https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ and
place them in `data/`. The CSVs are not redistributed here. ETDataset is
published under CC BY-ND 4.0; check the upstream repository for the current
terms.

## Results

All values are equal-seed mean-MSE ratios of the learned graph against a
comparator at horizon 1, with paired block-bootstrap 95% confidence intervals
(2000 resamples). A ratio below 1 favours the learned graph. Gates were
committed before any held-out data was built. Full per-horizon tables are in
`results/thesis_export_c2/`.

### Stationary synthetic data: supported

| Stage | Data | learned / dense | learned / random | learned / identity |
|---|---|---|---|---|
| S4b | D20, 3 held-out seed pairs | 0.986 [0.984, 0.987] | 0.829 [0.823, 0.836] | 0.809 [0.802, 0.817] |
| S4b-hard | D20, 600 training steps | 0.973 [0.971, 0.975] | 0.853 [0.847, 0.860] | 0.852 [0.845, 0.859] |

![Stationary held-out ratios](results/thesis_export_c2/stationary_s4b_ratios.png)

### Learned graph versus dense after a graph switch: not supported

- S4 (held-out, test slice contains a graph switch): learned / dense
  0.9999 [0.997, 1.002]. The frozen gate failed.
- S5 (online, first 200 windows after a switch): detector-driven learned
  minus dense online MSE is +0.000117 [+0.000026, +0.000211]. Dense adapts
  better.
- S5b (EMA reset with replay updates): learned minus dense online MSE is
  +0.001084 [+0.000846, +0.001334]. The gate failed and was not retried.

![Pooled S5 and S5b post-switch MSE](results/thesis_export_c2/pooled_switch_mse.png)

### Real data: DLinear is stronger

| Case | learned / dense | learned / DLinear |
|---|---|---|
| ETTh1, H24 | 0.973 [0.962, 0.983] | 1.733 |
| ETTh1, H96 | 0.997 [0.990, 1.003] | 1.553 |
| ETTh2, H24 | 0.991 [0.984, 0.997] | 1.153 |
| ETTh2, H96 | incomplete | incomplete |

On ETT the learned graph is at least as good as the dense graph, but DLinear
has lower error in every evaluated case.

### Scope

- Stationary synthetic data: supported.
- Learned graph versus dense after a switch: not supported.
- Real data: DLinear is stronger.
- Online adaptation on real data: not tested.

## Reproducing the thesis

Run all commands from the repository root. The thesis tables and figures in
`results/thesis_export_c2/` are built from the stage summaries in
`results/roadmap/`:

```bash
python -m scripts.export_thesis_assets_c2
```

This rebuilds every CSV and both figures. The CSV and PNG files are
byte-identical to the committed ones; the PDFs differ only in embedded
metadata.

To regenerate the stage summaries themselves, run the stage scripts below.
Each stage reads its frozen protocol or `gates.yaml` under `results/roadmap/`
and refuses to build held-out data unless that file is committed and
unchanged. Runs use `torch.set_num_threads(1)` and are deterministic for a
fixed data and model seed pair.

| Table | Commands |
|---|---|
| `S1.csv` | `python -m scripts.roadmap_s1 --stage s1` |
| `S1b.csv` | `python -m scripts.roadmap_s1 --stage s1b` |
| `S1_diagnostics.csv` | `python -m scripts.roadmap_s1_diagnostics` |
| `S3.csv` | `python -m scripts.roadmap_s3` |
| `S2.csv` | `python -m scripts.roadmap_final --stage s2`, then `--stage s2 --summarize` |
| `S2d.csv` | `python -m scripts.roadmap_final --stage s2d`, then `--stage s2d --summarize` |
| `S4.csv`, `S4-hard.csv` | `python -m scripts.roadmap_final --stage s4 --pairs 0,1,2`, then `--stage s4 --summarize` |
| `S4b.csv` | `python -m scripts.roadmap_s4b --pairs 0,1,2`, then `--summarize` |
| `S4b-hard.csv` | `python -m scripts.roadmap_s4b_hard --pairs 0`, `--pairs 1`, `--pairs 2`, then `--summarize` |
| `S5.csv` | `python -m scripts.roadmap_s5 --phase tuning`, `--phase select`, `--phase heldout --pairs 0,1,2,3,4`, then `--phase heldout --summarize` |
| `S5b.csv` | `python -m scripts.roadmap_s5b --phase tune --pair P --cand C` for P in 0..2 and C in 0..5, `--phase select`, `--phase heldout --pair P --variant all` for P in 0..4, then `--phase summarize` |
| `S6.csv` | `python -m scripts.roadmap_s6 --dataset ETTh1 --horizon 24`, the same with `--horizon 96` and with `--dataset ETTh2`, then `--summarize` |
| `S5_S5b_pooled.csv`, `saturation_diagnostic.csv` | `python -m scripts.roadmap_continuation2` (reads saved outputs only) |
| `S6b.csv` | not run; the table records this |

The full set of stages takes several hours on a laptop CPU.

## Citation

If you use this code, please cite it with the metadata in
[`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on
GitHub.

## License

The code is released under the [MIT License](LICENSE). The ETT data is not
part of this repository and keeps its own licence.
