# OCG-SSM

OCG-SSM is a multivariate time-series forecaster that learns a lagged dependency
graph between variables and uses it to condition a state-space model written in
pure PyTorch. The graph can be updated online, with a regime-switch detector that
reacts to abrupt changes in the data-generating process. The repository contains
the model, synthetic SVAR generators with known graphs, baselines (identity,
dense, random, oracle, DirectVAR, DLinear), a leak-safe online evaluation
protocol and the scripts behind the thesis tables and figures.

## Install

Python 3.10 or newer. The thesis runs used Python 3.12.12 on CPU.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Tests

From the repository root:

```bash
python -m pytest
```

One test in `tests/test_roadmap_final.py` is skipped if `data/ETTh1.csv` is not
present.

## Data

The real-data stage (S6) uses ETTh1 and ETTh2 from the ETDataset repository
(https://github.com/zhouhaoyi/ETDataset). Download `ETTh1.csv` and `ETTh2.csv`
from
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ and place
them in `data/`. The CSVs are not included in this repository. ETDataset is
distributed under the CC BY-ND 4.0 licence; check the upstream repository for
the current terms before redistributing the files.

All other stages use synthetic data generated in code from fixed seeds.

## Reproducing the thesis tables and figures

All commands run from the repository root. The tables and figures in
`results/thesis_export_c2/` are built from the stage summaries in
`results/roadmap/`. Run this command to rebuild every CSV and both figure pairs
(`pooled_switch_mse`, `stationary_s4b_ratios`, PNG and PDF):

```bash
python -m scripts.export_thesis_assets_c2
```

The stage summaries come from the commands below. Each stage reads its frozen
protocol or `gates.yaml` under `results/roadmap/`, and the scripts check that
the file is committed and unchanged before building held-out data. Stages
without a run step only post-process saved outputs.

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
| `S6.csv` | `python -m scripts.roadmap_s6 --dataset ETTh1 --horizon 24`, the same for `--horizon 96` and for `ETTh2`, then `--summarize` |
| `S5_S5b_pooled.csv`, `saturation_diagnostic.csv` | `python -m scripts.roadmap_continuation2` (saved outputs only) |
| `S6b.csv` | not run; the table records this |

Runs use `torch.set_num_threads(1)` and are deterministic for a fixed data and
model seed pair. Seed pairs and gate conditions are listed in the frozen
protocol files.

## Results summary

Ratios are equal-seed mean-MSE ratios (below 1 favours the learned graph), with
paired block-bootstrap 95% intervals. The per-stage values are in
`results/thesis_export_c2/`.

- Stationary synthetic data: supported. On stationary held-out data (S4b,
  S4b-hard) the learned graph beat the dense, random and identity graphs with
  intervals below 1.
- Learned graph versus dense after a graph switch: not supported. The held-out
  S4 gate failed on a test slice that contains a switch. In S5 the
  detector-driven online learned graph did not beat online dense adaptation
  over the first 200 post-switch windows, and the S5b variant failed the same
  comparison.
- Real data (ETTh1, ETTh2): DLinear is stronger than OCG-SSM in every evaluated
  case. The learned graph beat dense on ETTh1 H24 and ETTh2 H24. ETTh1 H96 shows
  no interval-supported gain, and ETTh2 H96 is incomplete.
- Online adaptation on real data: not tested.

## Licence

MIT, see `LICENSE`.
