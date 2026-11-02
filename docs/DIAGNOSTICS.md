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
