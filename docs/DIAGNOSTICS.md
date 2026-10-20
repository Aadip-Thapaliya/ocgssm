# Synthetic Gate Diagnostics

## Speed

Benchmarked on CPU with `python scripts/bench_speed.py` using `L=24`, batch size
32, and online prequential timing over 160 steps. "Before" is the
`_reference_forward` scan and "After" is the vectorized fast scan path.

| D | Metric | Before (reference) | After (fast) | Speedup |
|---:|---|---:|---:|---:|
| 5 | Forward ms/batch | 4.912 | 2.674 | 1.84x |
| 5 | Backward ms/batch | 20.958 | 12.653 | 1.66x |
| 5 | Online ms/step | 15.236 | 7.956 | 1.92x |
| 20 | Forward ms/batch | 6.481 | 3.776 | 1.72x |
| 20 | Backward ms/batch | 25.291 | 27.394 | 0.92x |
| 20 | Online ms/step | 16.720 | 8.294 | 2.02x |

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
