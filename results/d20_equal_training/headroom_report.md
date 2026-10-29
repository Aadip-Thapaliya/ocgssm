# D20 development headroom gate

## Result

The development-only D20 headroom gate **passed**. On the untouched
development validation segment (raw timestamps 2933-3199; 238 windows),
DirectVAR's mean MSE was 19.4% below identity at H1 and 12.7% below identity
at H2. Paired circular moving-block 95% intervals were below 1.0 at both
horizons. No held-out seed, test target, or held-out forecast result was read.

| Horizon | Identity MSE | DirectVAR MSE | DirectVAR / identity | Paired 95% block-bootstrap CI |
|---:|---:|---:|---:|---:|
| 1 | 0.013027 | 0.010499 | 0.8059 | [0.7892, 0.8244] |
| 2 | 0.013299 | 0.011610 | 0.8730 | [0.8583, 0.8893] |

The predeclared gate required a point ratio <=0.95 and a 95% interval upper
endpoint <1.0 for each horizon. Both conditions passed at H1 and H2. Bootstrap
resampling used 12-window circular blocks and 2,000 resamples.

## Development-only selection and runtime

Forecast-rate screening on training data plus validation timestamps
2400-2665 selected a shared learning rate of **0.001**. Its equal-weight
normalized validation score was 1.00020 versus 1.00614 at 0.0003 and 1.00104
at 0.003. Structural graph-rate screening selected **0.01**: AUROC and
average precision tied at 1.0 with 0.03, so the lower rate won. At the fixed
0.5 threshold, the 0.01 candidate had F1 0.2299 and SHD 67; this cautions
against interpreting perfect ranking metrics as calibrated binary edges.
DirectVAR here is the project's learned linear lag-forecast model trained
with `train_offline`; these values are not from an OLS fit.

The complete development stage (rate selection, graph-rate selection, and
both headroom fits) took **1,142.05 seconds (19m02s)** of measured CPU
runtime. The identity headroom fit took 689.47 seconds and selected epoch 32
of 50; DirectVAR took 15.71 seconds and selected epoch 5. The separate
profile artifact records 24.00 / 14.74 / 0.332 seconds per offline epoch
(identity / learned / DirectVAR) and 0.777 / 0.676 / 0.028 seconds per
stream window.

The machine reported a 14-core / 20-logical-processor Intel i7-13700H,
13.7 GB physical memory, and 3.9 GB free at the time of the capacity check.
CUDA was not used. All 72 project tests pass; the only warning is the
pre-existing PyTorch Transformer nested-tensor warning.

## Scope and next decision

The previous all-variant launch path is disabled. The D20 `runs` directory,
held-out campaign ledger, and held-out summary were not created. The three
reserved seed pairs (37/401, 43/503, 47/607) remain unevaluated. No detector,
real-data, or Colab work was done.

The approved reduced protocol is in `reduced_proposal_v2.yaml`. It retains
D20/K2 and the same chronological split and seed pairs, but limits the
campaign to identity, dense, learned-soft, oracle, and DirectVAR with at most
20 offline epochs and validation early stopping. Its measured-cost estimate
is 2.89 hours of core work sequentially, or approximately **3.18-3.47 hours**
including 10-20% runtime allowance. A separate oracle-40 side-check on
37/401 is included, not part of the five-variant gates. The forecast-rate and
graph-rate values are frozen from development only. The user has now approved
this exact reduced campaign; it must still be frozen and committed locally
before held-out execution.

The estimate assumes a 20-epoch maximum for the three fixed-graph SSM
variants, learned SSM, and DirectVAR, and uses the measured 64-window
streaming rates for the three SSM modes and DirectVAR. The sequential estimate
does not assume parallel speedup. Two seed processes could fit the available
CPU cores, but only 3.9 GB was free and per-process peak memory was not
measured; any concurrency was considered; execution is sequential only as requested.

Raw evidence is in `development/development_result.json`,
`development/headroom_gate.json`, and the two rate-selection JSON files.
