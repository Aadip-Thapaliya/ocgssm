# Synthetic Gate Diagnostics

## Sanity run before Step 4 model changes

Configuration: `configs/sanity.yaml` (seed 17, 4,000 steps, `switch_at = 4,000`,
offline training, 40 epochs, learning rate 0.003). Forecast MSE is over the
configured six-step test forecasts.

| Variant | Overall MSE | Graph SHD | Graph AUROC |
|---|---:|---:|---:|
| identity | 0.019827 | unavailable | unavailable |
| learned | 0.020096 | 15 | 0.500000 |
| random | 0.019453 | unavailable | unavailable |
| oracle | 0.019198 | 0 | 1.000000 |

Oracle was 3.2% lower than identity, below the required 10% margin, so Step 4
diagnosis was required.

## Post-switch direct OLS floor check

Data: the `configs/gate.yaml` synthetic series after `switch_at = 2,800`.
For each direct horizon, the first half of this post-switch segment fits OLS
and the second half evaluates it. Each regression uses the two most recent
observations plus an intercept. The mean baseline fits only the target mean;
per-variable AR fits each variable from its own two lags; full VAR fits all
variables' two lags jointly.

| Horizon | Mean MSE | Per-variable AR MSE | Full VAR MSE |
|---:|---:|---:|---:|
| 1 | 0.020985 | 0.019446 | 0.010457 |
| 2 | 0.020985 | 0.020182 | 0.014366 |
| 3 | 0.020985 | 0.020595 | 0.017847 |
| 4 | 0.020985 | 0.020414 | 0.019541 |
| 5 | 0.020984 | 0.020811 | 0.019971 |
| 6 | 0.020983 | 0.021034 | 0.020748 |
| Mean across horizons | 0.020985 | 0.020414 | 0.017155 |

## Sanity run after Step 4 graph-aware lag readout

The same `configs/sanity.yaml` run, with graph-masked learnable lag input
mixing and a direct graph-masked lag forecast head, produced:

| Variant | Overall MSE | Graph SHD | Graph AUROC |
|---|---:|---:|---:|
| identity | 0.019708 | unavailable | unavailable |
| learned | 0.016494 | 15 | 0.500000 |
| random | 0.016913 | unavailable | unavailable |
| oracle | 0.017144 | 0 | 1.000000 |

Oracle is 13.0% lower than identity, passing the required 10% sanity margin.
The same model and config were used; comparison to the pre-Step-4 run is
identity 0.019827 -> 0.019708 and oracle 0.019198 -> 0.017144.

## Correctness baseline and leak-safe online rerun

The original local copy passed 36 tests before the correctness audit. Its
`configs/synthetic_abrupt.yaml` online loop trained immediately from all six
future horizon labels at every forecast. The baseline output is preserved at
`results/baseline_synthetic_abrupt.json` (seed 17, test MSE 0.143124,
graph AUROC 0.978175); it is explicitly not a fair comparison with the
corrected protocol.

The corrected loop delays each label until its observation time, uses
timestamped pre-update EMA graph snapshots, and labels online evaluation as
cold-start or train-only warm-start. The seed-17 cold-start smoke run is at
`results/correctness_synthetic_abrupt.json`: 202 test windows at input
timestamps 304–505, MSE 2.344530, graph AUROC 0.938492. This is a single
smoke result, not a model-improvement claim. Run the separate
`configs/synthetic_abrupt_warm.yaml` before making any warm-start comparison.

The full corrected suite passes 51 tests. New coverage checks offline
structural VAR/EMA updates, delayed-target leakage, optimizer ownership,
seed/order reproducibility, warm-start labeling, RevIN wiring, timestamped
graph snapshots, stability across gradual coefficient blends, and fail-fast
handling of non-finite forecasts and training state.

## Independent graph-recovery experiment

`scripts/graph_diagnostics.py` evaluates stationary D=5/K=2 followed by
D=20/K=2 with three explicit data/model seed pairs. It fits the structural
learner using training data only, selects its probability threshold and the
sparse-VAR alpha/edge threshold using validation graph F1 only, then reports
test AUROC, average precision (PR-AUC), precision, recall, F1, SHD, loss,
gradient/update magnitudes, and current/EMA edge distributions. Empty,
dense, and density-matched random controls are scored against the same test
truth. The incrementally written `results/graph_diagnostics.json` preserves
individual seed-pair outcomes, source hashes and environment metadata;
per-epoch model, Adam, and shuffle-generator state is in
`results/graph_diagnostics_checkpoints/`.

All six stationary runs completed. Metrics below are mean ± population SD
over the three data/model seed pairs; thresholds were selected by validation
F1, never by test scores.

| D | Estimator/control | F1 | SHD | AUROC | PR-AUC |
|---:|---|---:|---:|---:|---:|
| 5 | Learned structural graph | 0.9740 ± 0.0187 | 0.67 ± 0.47 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 5 | Sparse VAR | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 5 | Empty | 0.0000 ± 0.0000 | 12.33 ± 2.05 | 0.5000 ± 0.0000 | 0.2467 ± 0.0411 |
| 5 | Dense | 0.3940 ± 0.0526 | 37.67 ± 2.05 | 0.5000 ± 0.0000 | 0.2467 ± 0.0411 |
| 5 | Density-matched random | 0.1739 ± 0.0574 | 18.33 ± 0.47 | 0.4719 ± 0.0259 | 0.2411 ± 0.0452 |
| 20 | Learned structural graph | 0.9855 ± 0.0138 | 2.33 ± 2.05 | 0.9999 ± 0.0001 | 0.9992 ± 0.0006 |
| 20 | Sparse VAR | 0.9983 ± 0.0025 | 0.33 ± 0.47 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 20 | Empty | 0.0000 ± 0.0000 | 82.67 ± 9.46 | 0.5000 ± 0.0000 | 0.1033 ± 0.0118 |
| 20 | Dense | 0.1871 ± 0.0193 | 717.33 ± 9.46 | 0.5000 ± 0.0000 | 0.1033 ± 0.0118 |
| 20 | Density-matched random | 0.0877 ± 0.0427 | 148.00 ± 1.41 | 0.4917 ± 0.0203 | 0.1037 ± 0.0156 |

This is strong recovery on the known stationary synthetic generator and
supports a predictive graph signal under that data-generating assumption.
It does not establish causal identification from real observational series.

## Matched model contribution and ablation campaign

`configs/research_online.yaml` compares five graph controls, DLinear,
RevIN-Linear, Transformer-lite, NoGraphSSM, direct VAR, and four ablations
(direct lag head, SSM coupling, online graph updates, detector boost). The
full campaign completed all 14 variants × 202 test windows for three regimes
and three fixed data/model seed pairs (9 runs). Values below are mean ±
sample SD across seed pairs.

| Regime | Identity MSE | Learned MSE | Oracle MSE | DLinear MSE | No-SSM-coupling MSE | Learned graph F1 / SHD |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 0.01404 ± 0.00312 | 35.1531 ± 60.8271 | 0.01410 ± 0.00296 | 0.01347 ± 0.00273 | 0.01382 ± 0.00294 | 0.4527 ± 0.4359 / 8.33 ± 6.66 |
| Abrupt | 0.01377 ± 0.00117 | 35.1049 ± 60.7452 | 0.01384 ± 0.00108 | 0.01322 ± 0.00099 | 0.01351 ± 0.00094 | 0.6934 ± 0.3249 / 4.67 ± 5.69 |
| Gradual | 0.01320 ± 0.00124 | 35.1498 ± 60.8243 | 0.01331 ± 0.00112 | 0.01268 ± 0.00103 | 0.01298 ± 0.00105 | 0.5065 ± 0.1570 / 6.00 ± 4.36 |

The learned-graph means are dominated by finite cold-start spikes in model
seed 303: maximum per-window MSE was 6,967.57; dense reached 130,276.72.
Every per-step value is retained in the raw JSON; there was no clipping or
removal of startup windows. No forecast improvement was supported. The
no-SSM-coupling ablation remained near the other baselines, making graph
coupling a follow-up instability lead, not a confirmed causal mechanism.
Per-horizon means/spreads and pre/post-switch curves are in `summary.json`
and `forecast_curves.png`; the learned pre/post MSE means were 421.27 / 1.8867
for abrupt and 147.49 / 0.1351 for gradual.

The abrupt detector emitted no events in any seed. The gradual detector
emitted two events in one seed (1/3 detection coverage), with 79-step delay;
no false alarms were recorded. The sustained graph-recovery metric requires
10 consecutive SHD observations within 10% of the pre-switch mean. It reports
zero for all abrupt/gradual graph variants because the threshold is already
met at the switch; this is not evidence of forecast recovery. Detector rate,
delay, false alarms, sustained recovery, per-horizon metrics, runtime, and
peak memory are summarized per variant in `summary.json`/`summary.csv`.

The full campaign used 4,766 seconds (79m26s) of measured CPU run time.
`resume_state.json` preserves the run ledger, seeds, environment metadata,
configuration hashes, and source hashes; this folder-backed checkout has no
Git metadata, so no commit revision is available. The exact private package
environment is in `environment.lock.txt`. Each original run JSON has a
deterministic learned-variant replay under `graph_snapshots/results/`; all
nine replays matched the campaign forecast and graph metrics and saved 202
pre-update EMA matrices aligned to timestamps 304–505. Hashes and the replay
command are in `graph_snapshots/index.json`.

Reproduce the full campaign and graph-snapshot replay with:

```powershell
.\.venv\Scripts\python.exe -m scripts.synthetic_evaluation --scenarios stationary abrupt gradual --seed-pairs 11:101 17:202 23:303
.\.venv\Scripts\python.exe -m scripts.replay_graph_snapshots --output-directory results\model_contribution
```

The final test suite passes 51 tests (one pre-existing Transformer warning).
CUDA is unavailable, and the project has no real-data CSVs; no real-data or
GPU experiment was run. The separate stationary structural learner still
recovers the synthetic generator graph well, but that does not establish a
forecasting contribution or causal identification from real observations.

## Warm-start detector follow-up

The earlier abrupt campaign used the 3.5-score threshold and produced no
detections in three abrupt runs. A development-only seed-17 probe was used to
exercise the new detector arming lifecycle. The detector now starts disarmed
after calibration and arms only after a configured streak of baseline inliers;
this prevents a score burst at test entry from immediately consuming the
one-shot alert. The candidate protocol used a fixed threshold floor of 5,
three consecutive outliers to confirm, and five consecutive inliers to arm or
re-arm. The development probe detected the switch at timestamp 322 (two steps
after the true switch at 320), with no pre-switch event. Because this probe
informed the candidate settings, it is not counted as confirmatory evidence.

The settings were then frozen and run on three distinct held-out synthetic
data/model seed pairs. Thresholds above the fixed floor are selected from the
validation maximum robust z-score plus 0.25; test statistics remain frozen.
Coverage means at least one detection at or after the switch; delays are
measured from the switch timestamp. Later alerts after the first post-switch
event are listed as repeated alerts, not counted as additional detections.

| Data/model seed | Validation z max | Threshold | Test events | First delay | Pre-switch false alarms | Sustained recovery |
|---|---:|---:|---|---:|---:|---:|
| 11/101 | 6.3654 | 6.6154 | 327, 357, 433, 487 | 7 | 0 | 57 steps (reference SHD 3) |
| 23/202 | 1.3284 | 5.0000 | 324, 352 | 4 | 0 | 151 steps (reference SHD 3) |
| 29/303 | 1.2287 | 5.0000 | none | unavailable | 0 | unavailable (reference SHD 2) |

Detection coverage is 2/3; the detector is not reliable enough to call fixed.
The first post-switch event is within seven steps in the two detected runs,
but there are three and one repeated post-switch alerts respectively, and one
miss. No pre-switch false alarms occurred in these three held-out runs.
Sustained recovery uses the retrospective evaluation-only reference defined
above; the nonconverged run is null, not zero. This detector-only experiment
does not establish an OCG-SSM forecasting benefit.

Across all 606 held-out learned-variant forecast windows, pooled MSE is mean
0.018619, median 0.017077, and a 10%-per-tail trimmed mean 0.017724; the
maximum remains 0.059657. The trimmed statistic removes only
`floor(0.1 * n)` observations at each tail from its calculation; every raw
window error remains in the source JSON. Mean/median/trimmed and per-horizon
summaries plus log-axis plots are in
`results/warm_start_revision/detector_summary/`. Per-seed raw results and
their exact configs are `detector_data*_model*.json` and
`detector_seed_configs/`.

| Horizon | Mean MSE | Median MSE | 10%-trimmed mean MSE |
|---:|---:|---:|---:|
| 1 | 0.017210 | 0.014760 | 0.015745 |
| 2 | 0.018395 | 0.015198 | 0.016862 |
| 3 | 0.018715 | 0.015971 | 0.017034 |
| 4 | 0.019025 | 0.016007 | 0.017394 |
| 5 | 0.019135 | 0.016253 | 0.017546 |
| 6 | 0.019236 | 0.016513 | 0.017661 |

## Frozen graph-utilization comparison

The fixed warm-start comparison covered scale 1.0 and validation-selected
scale 1.5, each with seven variants and data/model seed pairs 11/101, 23/202,
and 29/303. At scale 1.5, DirectVAR/identity MSE ratios were 0.827
[0.804, 0.852] and 0.893 [0.872, 0.918] at horizons 1 and 2. Oracle was
within 5% of DirectVAR at H1 (ratio 1.026) but not H2 (1.105). Learned was
within 5% of oracle at H2 (0.987), not H1 (1.072); learned was significantly
worse than dense at both horizons by the paired block-bootstrap intervals.
These are synthetic forecasting results, not evidence of causal discovery.

The 50-epoch train-only oracle comparison outperformed matched offline
DirectVAR at both horizons on all three seeds, suggesting that the
architecture can use oracle graph information with a longer offline
optimizer schedule. The validation loss was still improving at the epoch
limit, so this does not prove that online budget alone explains the gap.
Post-switch online learned-graph F1 ranged from 0.322 to 0.616, with SHD
from 5.10 to 9.80, leaving graph-estimation quality as a plausible
contributor alongside graph utilization. No model or hyperparameter changes
were selected from test outcomes.

The frozen protocol, per-seed intervals, full results, branch diagnostics,
and both-scale ratio plot are documented in
`results/graph_utilization/report/graph_utilization_findings.md`;
`graph_utilization_summary.json` and `mse_ratios_to_identity.png` retain the
machine-readable summary and plot.

## D20 equal-training follow-up: runtime profile only

`results/d20_equal_training/protocol.yaml` specifies the proposed equal
50-epoch train-only pretraining and delayed-label test adaptation at D=20,
4,000 steps, plus a hard-thresholded learned-graph ablation. The development
pair (31/301) and held-out pairs (37/401, 43/503, 47/607) are new relative to
the prior graph diagnostics and graph-utilization seed pairs. No validation
headroom comparison or held-out D20 forecast outcome has been run yet.

A one-epoch/64-window CPU profile measured identity CausalSSM at 24.00
seconds per offline epoch and 0.777 seconds per streaming window; learned
CausalSSM at 14.74 seconds and 0.676 seconds; DirectVAR at 0.332 seconds
and 0.028 seconds. Linear extrapolation estimates about 7.54 CPU hours for
the three-pair full campaign, excluding development-only learning-rate
selection and headroom validation. The campaign is paused pending a
compute/scope decision. `results/d20_equal_training/profile_findings.md`
contains the frozen protocol summary, estimation method and limitations;
the measurement is in `profile.json` and the profile chart is
`d20_runtime_profile.png`.

The held-out runs measured 1.060–1.178 seconds per test step (202 windows per
run; approximately 214–238 seconds each, 688 seconds total). Reproduce with:

```powershell
.\.venv\Scripts\python.exe -m ocg.run --config results\warm_start_revision\detector_seed_configs\detector_data11_model101.yaml --output results\warm_start_revision\detector_data11_model101.json
.\.venv\Scripts\python.exe -m ocg.run --config results\warm_start_revision\detector_seed_configs\detector_data23_model202.yaml --output results\warm_start_revision\detector_data23_model202.json
.\.venv\Scripts\python.exe -m ocg.run --config results\warm_start_revision\detector_seed_configs\detector_data29_model303.yaml --output results\warm_start_revision\detector_data29_model303.json
.\.venv\Scripts\python.exe -m scripts.warm_start_report --result data11-model101=results\warm_start_revision\detector_data11_model101.json --result data23-model202=results\warm_start_revision\detector_data23_model202.json --result data29-model303=results\warm_start_revision\detector_data29_model303.json --output-directory results\warm_start_revision\detector_summary
```

Separately, the frozen scale-1.5 seed-17 headroom gate found DirectVAR MSE
0.017080 / 0.018679 versus identity 0.021194 / 0.021274 at horizons 1 / 2.
This is a one-seed check, not a population estimate. The matched seed-17
learned-versus-oracle comparison remains essentially tied overall (0.020238
versus 0.020224), with learned worse at horizon 1 and slightly better at
horizon 2. Robust summaries and log-axis plots for these one-seed comparisons
are in `headroom_summary/` and `learned_oracle_summary/`. Neither result
supports a general forecasting claim. The generator scale is a separate
synthetic-data version: off-diagonal coefficients are
multiplied by 1.5 before stability rescaling, with noise fixed at 0.1 and
companion radius below 0.9. No Colab, real-data, GPU, or larger model campaign
was run.

## D20 development headroom gate and reduced proposal

The development-only D20/K2 run used 4,000 synthetic steps at cross-variable
scale 1.0, with train/validation/test raw intervals 0-2399, 2400-3199, and
3200-3999. Forecast-rate selection chose 0.001 from train-only fitting and
validation indices 2400-2665; structural graph-rate selection chose 0.01 by
validation AUROC (1.0, tied with 0.03; lower rate selected). The complete
rate-selection and headroom stage measured 1,142.05 CPU seconds (19m02s).

On the untouched development validation gate (raw indices 2933-3199,
238 windows), DirectVAR/identity MSE ratios were 0.8059 at H1
(95% paired circular 12-window block interval [0.7892, 0.8244]) and
0.8730 at H2 ([0.8583, 0.8893]). The predeclared requirement was <=0.95
point ratio and interval upper endpoint <1.0 at both horizons; both passed.
This DirectVAR is fitted by the project's gradient-based `train_offline`
optimizer, not OLS. The graph-rate candidate at 0.01 had AUROC/AP 1.0 but
F1 0.2299 and SHD 67 at threshold 0.5, so rank quality should not be mistaken
for threshold calibration.

The former full 50-epoch campaign was stopped before held-out evaluation.
The approved five-variant protocol (identity, dense, learned-soft, oracle,
DirectVAR) is frozen at `results/d20_equal_training/reduced_proposal_v2.yaml`.
It caps each main variant at 20 epochs with validation-MSE patience 5,
strict improvement, and best-checkpoint restoration; any run that reaches
the cap is explicitly marked as not converged. The declared oracle-40
side-check uses data/model seed 37/401, the same patience rule, the same
delayed-label stream, and is excluded from all primary gates.

The reserved held-out pairs are 37/401, 43/503, and 47/607. Required outputs
include per-seed mean/median MSEs, equal-seed mean and median summaries, and
paired 95% circular 12-window block-bootstrap intervals for learned-vs-dense,
learned-vs-oracle, and oracle-vs-DirectVAR at H1/H2. The point gates remain
oracle/DirectVAR <=1.05 and learned/oracle <=1.05; learned must beat dense
with the paired interval upper endpoint below zero. The oracle-40 side-check
does not enter these comparisons.

Maximum-budget extrapolation including the side-check is 3.21 hours core
sequential CPU time, or 3.53-3.85 hours with 10-20% allowance. Execution is
sequential only; the workstation reported 14 physical / 20 logical CPU
processors and 3.9 GB free of 13.7 GB at measurement. No held-out run had
started at this protocol freeze. Do not run the detector, real data, or
Colab; no parallel processes are authorized.

## Roadmap S1 - oracle direct-head masking at H2 (development only)

Hypothesis: the oracle direct readout applied lag-k mask A_k at every horizon, but for K=2 the H2 coefficients are A1@A1+A2 on x[t] and A1@A2 on x[t-1]. Variants (identical 20-epoch cap, patience 5, best-checkpoint restore, also for DirectVAR) on pairs 67/1009, 71/1103, 73/1201; selection on tune 2400-2665; gate on validation 2933-3199 (equal-seed paired 12-window block bootstrap, 2000 resamples). No test or held-out data used.

| Oracle head | H1 / DirectVAR [95% CI] | H2 / DirectVAR [95% CI] | H3 | H6 |
|---|---|---|---:|---:|
| lag (previous default) | 1.0275 [1.0187, 1.0366] | 1.1294 [1.1168, 1.1424] | 1.0585 | 1.0332 |
| reachability (companion support) | 1.0285 [1.0198, 1.0371] | 1.0365 [1.0294, 1.0441] | 1.0402 | 1.0446 |
| iterated (selected) | 1.0288 [1.0198, 1.0380] | 1.0333 [1.0260, 1.0413] | 1.0336 | 1.0295 |

Result: S1 PASS (gate <= 1.05 at H1 and H2). The H2 over-restriction hypothesis is supported on development seeds: fixing horizon masks cut the H2 ratio from 1.129 to 1.033-1.037 without changing H1. The oracle remains ~3% worse than DirectVAR at H1/H2 (CIs exclude 1), and 6/9 oracle runs hit the 20-epoch cap, so convergence is still open (S3). Run time 18.2 min CPU. Artifacts: `results/roadmap/s1/s1/summary.md`, `summary.json`, `s1_ratios_curves.png`.

### S1 diagnostics decision rule (frozen before running, user-directed)
On dev pairs 67/1009, 71/1103, 73/1201 and validation slice 2933-3199 only: report oracle_iterated vs identity/dense (both with the iterated head, S1 training policy), oracle-support OLS (train-only lstsq/gelsd on [x_t, x_(t-1), 1] restricted to true companion reachability support) vs DirectVAR and identity, reachability densities, and oracle_iterated with a 40-epoch cap on 67/1009. "Clearly beats" = paired 95% block-bootstrap CI upper bound of the MSE ratio < 1 at that horizon. Go to S1b if, at H1 or H2, OLS clearly beats DirectVAR while oracle_iterated does not; otherwise go to S3. Iterated remains the accepted S1 selection.

### S1 diagnostics, S1b and S3 results (development only, validation 2933-3199)
- Oracle_iterated/identity: H1 0.820 [0.805, 0.834], H2 0.900 [0.888, 0.911]. Oracle_iterated/dense: H1 0.978 [0.974, 0.983], H2 0.988 [0.985, 0.992].
- Oracle-support OLS/DirectVAR: H1 0.972 [0.966, 0.977], H2 0.986 [0.981, 0.990]; OLS/identity: H1 0.774, H2 0.858. 0/360 regressions were rank-deficient; the maximum condition number was 13.6.
- Reachability density (edges / 400, self-edges counted): H1 0.112 / 0.088 (lag 1 / lag 2), H2 0.288 / 0.183, H3 0.560 / 0.387, H6 0.988 / 0.760.
- Oracle_iterated, 40-epoch cap, on 67/1009: best epoch 16, stopped at 21, cap not reached. Ratios to DirectVAR were unchanged (H1 1.0225, H2 1.0253), so the 20-epoch cap was not the limit.
- The frozen rule triggered S1b at both H1 and H2. S1b with a zero-initialized SSM head did not close the gap: iterated H1 1.0282 [1.0197, 1.0367], H2 1.0327 [1.0256, 1.0403]; reachability H2 1.0368. It passes the =1.05 gate but does not clearly beat DirectVAR. Iterated stays the S1 selection. Untested hypothesis: per-window RevIN in the SSM model, not the SSM residual, costs about 3% against raw-scale linear fits.
- S3 PASS: learned_soft_iterated with a 40-epoch cap and patience 5 hit the cap on 0/3 seeds (best epochs 33, 34, 32). The gain beyond epoch 20 was 0.02-0.03%, under the 0.5% limit. Informative ratios (not S2 gates): learned/oracle H1 1.006 [1.004, 1.008], H2 1.002 [1.000, 1.004]; learned/dense H1 0.984 [0.981, 0.987], H2 0.990 [0.989, 0.992]. Compute: diagnostics 13.2 min, S1b 9.6 min, S3 11.7 min.

### Final mission: S2 and noise-floor qualification
- S2 PASS (see PLAN). SSM variants remain behind linear fits on validation: oracle_iterated/var_direct H1 1.029, H2 1.033 (about 3% above DirectVAR) and, combining with S1 diagnostics (OLS/DirectVAR 0.972 / 0.986), roughly 5-6% above oracle-support OLS at H1 and about 5% at H2. learned_soft/var_direct H1 1.035 [1.027, 1.043].
- Noise-floor claim (user-supplied: "OLS/identity 0.774 is near the noise floor 0.01"): a ratio cannot show proximity to a floor. Checked in absolute units: the series is raw (not standardized) with innovations N(0, 0.1^2), so the Bayes one-step MSE is 0.0100 at H1 only (higher at H2+). Dev gate-slice H1 MSE: var_direct 0.01036, implied OLS ~0.0101 (0.972 x 0.01036), oracle 0.01066, identity 0.01300. So OLS is within about 1% of the theoretical H1 floor and oracle_iterated about 6.6% above it; at H2 the floor exceeds 0.01, so "0.01" applies to H1 only. Finite-slice noise variance is itself sampled (267 windows), so "about 1%" is approximate.
- learned_hard caveat: soft edge probabilities never exceed 0.5 under the default sparsity, so fixed-threshold (0.5) F1 is uninformative for this learner; S5 additionally reports top-|E| F1 (reporting only).
- S2d D40 skip-rule incident (decided before any remaining D40 run, without seeing D40 ratios): the laptop entered Windows Modern Standby twice (about 01:44-13:53 and 13:53-15:53). The D40 learned run for 73/1201 straddled standby and logged 51,133 s wall time, which tripped the 600 s compute guard. All other D40 runs took 68-308 s. The guard is a compute-cost rule, so the trigger is treated as an artifact: `d40_skipped.json` was renamed to `d40_skip_standby_artifact.json`, the guard now applies only to freshly computed runs, and the remaining two D40 runs (oracle and var_direct for 73/1201) are completed. D40 then enters the frozen S4-hard rule unchanged. Training is deterministic and unaffected by standby; only the recorded elapsed seconds for runs spanning standby are inflated. A keep-awake helper (SetThreadExecutionState) now runs during experiments.

## S4 held-out confirmation (gates cbd9040, seeds 53/709, 59/811, 61/913, test 3200-3999) - FAIL -> F2
- Primary (default D20_T2400), all four conditions fail: learned/dense H1 0.9999 [0.9974, 1.0024]; learned/random H1 0.9926 [0.9705, 1.0162]; learned/identity H1 0.9983 [0.9731, 1.0241], H2 1.0005 [0.9835, 1.0183].
- Secondary: learned/oracle H1 1.093, H2 1.054 (both > 1.05). learned/var_direct H1 1.045; learned/dlinear H1 1.042; oracle/dense H1 0.915 [0.906, 0.924].
- S4-hard (D20_T600, secondary only): learned/dense H1 0.9955 [0.9919, 0.9992]; learned/identity H1 0.988 [0.969, 1.007]; learned/oracle H1 1.078.
- Descriptive pre-switch windows only (targets < 3600): learned/dense H1 0.984 [0.982, 0.986], learned/identity 0.813, learned/random 0.825, learned/oracle 1.005, learned/var_direct 1.044. The frozen gate slice spans the 3600 regime switch; after it, every offline-trained model (except oracle, which receives the true post-switch graph) loses its advantage, which removes the pre-switch gains. This is an explanation, not a rescue: the gate is FAIL.
- Decision: per mission, no retuning; F2 recorded ("graph recovery works but no forecast benefit over dense" on the frozen held-out slice). Learned hit epoch cap 40 on 3/3 seeds (default).
- Runtime: about 52 min of compute (17:18-17:51 plus 18:58-19:03; pair 2 was interrupted by a lost shell session and resumed from cached runs). Summarize script fix: pre-switch table renders only H1/H2 (formatting only).

## S5 held-out adaptation (gates df79569; seeds 79/1301, 83/1409, 89/1511, 97/1613, 101/1709; all 5 complete) - FAIL -> F3
- Frozen online config (dev tuning, 8-config grid): EMA 0.95, threshold 3.5, boost 5 (dev post-switch MSE 0.019989).
- Primary metric (first 200 post-switch windows, mean MSE): learned_det_on 0.01728; learned_frozen 0.01894; dense_online 0.01716; identity_online 0.01675; random_online 0.01688; oracle_online 0.01558; learned_det_off 0.01748.
- Gate: det_on - frozen = -0.001661 [-0.001831, -0.001502] (pass); det_on - dense_online = +0.000117 [+0.000026, +0.000211] (FAIL; dense is better). Gate FAIL -> F3 (benefit on stationary data, no adaptation gain over dense online). No retuning.
- Additional: det_on - det_off = -0.000196 [-0.000230, -0.000166] (detector helps); identity_online is the best non-oracle variant post-switch (det_on - identity = +0.000527 [0.000371, 0.000678]). Pre-switch MSE: learned 0.01538-0.01554 versus dense 0.01618 and identity 0.01854.
- Detector (det_on): 5/5 switches detected, 0 false alarms, delay 10, 2, 3, 2, 12 steps. det_off: no detections (by design).
- Graph tracking: pre-switch top-|E| F1 about 1.0; first-200 post-switch top-|E| F1 0.23-0.37 (det_on) versus 0.17-0.31 (det_off); final 0.38-0.65 versus 0.34-0.52; AUROC 0.87-0.89; F1@0.5 is uninformative (probabilities stay below 0.5). The learned graph does not recover the post-switch graph within 200 windows.
- Runtime and budget: S5 wall time about 5.8 h (tuning 23:57-01:44 and about 0.1 h after resume; dev reference 17:17-18:11, 18:57-19:35, 09:34-09:59; held-out 09:59-11:59 on 2-3 slots), against the 3.5 h budget plus a user-approved extension of 2 h (+15 min, unused: last held-out run finished 11:59:18). Summed process time is larger (parallel slots) and is not wall time. The S5 dev reference re-run (`--phase select` reporting on dev with the frozen config) was stopped at the deadline and is incomplete; selection itself (tuning/selection.json) was complete before any held-out access.

## S6 real data (gates 9ef1ded; ETTh1/ETTh2, standardized with train statistics; test 11520-14399) - per-case results
- ETTh1 H24: learned/dense MSE 0.9731 [0.9622, 0.9834] (CI-supported benefit); learned/identity 0.9479 [0.9305, 0.9668]; learned/dlinear 1.733 [1.624, 1.856].
- ETTh1 H96: learned/dense 0.9967 [0.9898, 1.0034] (point target met, no CI benefit); learned/identity 0.9753; learned/dlinear 1.553.
- ETTh2 H24: learned/dense 0.9907 [0.9839, 0.9973] (CI-supported benefit); learned/identity 0.9828 [0.9761, 0.9893]; learned/dlinear 1.153 [1.114, 1.197].
- ETTh2 H96: INCOMPLETE (2 of 3 seeds; S6 stage budget exhausted; not evaluated).
- No iterated-head divergence (all learned validation MSE < persistence), so learned_lag was not needed. All OCG variants hit the 20-epoch cap with best epoch 20 (under-trained under the frozen cap). DLinear is the best model in every case; var_direct is the worst (DirectVAR without per-window normalization is poorly suited to these non-stationary series).
- Learned graphs: mean density 0.43; top-10 edge Jaccard across seeds 0.25-0.43 (ETTh1) and 0.0-0.18 (ETTh2); no ground truth, so graphs are descriptive only.
- Runtime: S6 wall time about 4.8 h (process windows, 1 slot running alongside S5/S2d/S4 jobs, excluding standby) against a 3 h budget - overrun disclosed; ETTh2 H96 was stopped at 12:01 rather than extending silently. Online-adapted real-data evaluation was not run (no time remained).
- Runtime/standby caveat: invalid `elapsed_seconds`/`train s` values from runs that straddled standby (for example ETTh1 H96 random 44,434 s) are not compute times.
- Correction to the S4 entry: S4 wall time was about 62 min (17:18-18:11 and 18:57-19:06), not 52 min.

## Continuation Task 0 (2026-11-03, no new model runs)

Mission record: `docs/MISSION_CONTINUATION.md`. Tag `final-roadmap-v1` = 1c398e2 (pre-continuation; `roadmap-final` unchanged). Report: `results/roadmap/final/continuation/task0/task0.md` (+ `task0.json`).

- **(a) S4 slice design flaw (verified).** Regenerated generator graphs: for all S4 held-out pairs (53/709, 59/811, 61/913) the graph changes exactly once, at t = 3600. The saved S4 outputs have 771 test windows with first target 3224: 371 windows have all targets < 3600, 5 straddle, 395 are fully post-switch. The post-switch graph shares only 6-9% of the pre-switch edges, so ~51% of the S4 test windows evaluate an offline model on a regime it never saw; the learned graph (fit to regime 1) cannot be expected to beat dense there. The earlier D20 reduced campaign (pairs 37/401, 43/503, 47/607) used the same protocol (switch_at 3600, test slice 3200-3999, identical windowing); its graphs also change exactly at 3600 (4-17% shared edges), so it had the switch inside its test slice too. Its saved run files carry no per-window timestamps; the count follows from its frozen protocol and code. S4 remains a frozen FAIL; S4b (Task 1) is a new stationary confirmation, not a replacement.
- **(b) S5 per-window tables.** Stream has 971 windows; first post-switch window 476; only **495** post-switch windows exist, so "200-500" and "0-500" are truncated to [200,495) and [0,495) (not padded). The frozen [0,200) gate numbers reproduce the S5 summary (det_on - dense_online +0.00012 [+0.00002, +0.00021]). Post hoc, descriptive: over [200,495) dense_online is best non-oracle (det_on - dense +0.00089 [+0.00075, +0.00102]); detector on vs off is indistinguishable there (-0.00001 [-0.00003, +0.00001]). Pre-switch [0,471): learned variants beat dense_online (-0.00065 [-0.00075, -0.00054]) and match oracle.
- **(c) ETTh2 H96:** not run in this continuation; original partial outputs (2/3 seeds) and INCOMPLETE row preserved.

## Continuation Task 1: S4b (stationary held-out confirmation; gates 7e941a9)

- Purpose: correct S4's slice-design flaw with the same D20/K2 task but no regime switch (`switch_at = n_steps = 4000`), so the entire 3200-3999 test slice is stationary. Original S4 remains a frozen FAIL; S4b is new confirmation.
- Fresh pairs: 103/1801, 107/1907, 109/2003. Variants: identity, dense, random, learned, oracle, var_direct, DLinear. Same S4 iterated SSM head, train/tune slices, and training policy; var_direct remains the original direct-head baseline and DLinear is unchanged.
- Gate: **PASS (4/4 primary conditions)**. learned/dense H1 0.9857 [0.9841, 0.9874]; learned/random H1 0.8295 [0.8227, 0.8363]; learned/identity H1 0.8093 [0.8021, 0.8166], H2 0.8747 [0.8678, 0.8816]. Secondary learned/oracle point ratios H1/H2 1.0063/1.0039 (both <=1.05).
- Runtime: 20m55s wall (16:54:14-17:15:09), within <=1.5 h. Output: `results/roadmap/final/continuation/s4b/summary.md` and `.json`; per-seed runs under `s4b/default/`.
