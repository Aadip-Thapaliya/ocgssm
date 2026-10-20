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

## Quality

### Step a: offline learned-graph warm-start (reduced h=6 ablations)

Using [quality_d5_h6.yaml](/workspaces/ocgssm/configs/quality_d5_h6.yaml) and
[quality_d20_h6.yaml](/workspaces/ocgssm/configs/quality_d20_h6.yaml), seeds
17/18/19, with warm-start epochs toggled from 0 to 5:

| D | Warm-start epochs | Learned validation MSE (mean ± std) | Learned test MSE (mean ± std) | Learned AUROC (mean ± std) |
|---:|---:|---:|---:|---:|
| 5 | 0 | 3.2440 ± 4.5458 | 3.7547 ± 5.2648 | 0.5000 ± 0.0000 |
| 5 | 5 | 0.2594 ± 0.3227 | 0.2544 ± 0.3118 | 0.6194 ± 0.0235 |
| 20 | 0 | 8.60e21 ± 1.22e22 | 8.77e21 ± 1.24e22 | 0.5000 ± 0.0000 |
| 20 | 5 | 6.34e20 ± 8.97e20 | 6.61e20 ± 9.35e20 | 0.5208 ± 0.0398 |

Warm-start updates logits (`warm_start_logit_delta_l1 > 0`) and improves
learned graph AUROC/validation loss, but learned still trails identity on this
reduced setup, so additional stabilization/tuning is still required.

### Step b: validation threshold calibration modes

Learned graph metrics now support `fixed_0.5`, `density_prior`, and `best_f1`
threshold calibration (selected on the validation split). Reduced h=6 runs
with seeds 17/18/19 produced:

| D | Mode | SHD (mean ± std) | F1 (mean ± std) | Threshold (mean ± std) |
|---:|---|---:|---:|---:|
| 5 | fixed_0.5 | 13.67 ± 1.70 | 0.2838 ± 0.1084 | 0.5000 ± 0.0000 |
| 5 | density_prior | 13.67 ± 1.25 | 0.3452 ± 0.0541 | 0.4750 ± 0.0468 |
| 5 | best_f1 | 18.00 ± 7.48 | 0.4459 ± 0.0391 | 0.4267 ± 0.0704 |
| 20 | fixed_0.5 | 152.00 ± 7.07 | 0.1058 ± 0.0377 | 0.5000 ± 0.0000 |
| 20 | density_prior | 144.33 ± 8.26 | 0.0890 ± 0.0226 | 0.5421 ± 0.0112 |
| 20 | best_f1 | 522.67 ± 100.97 | 0.1916 ± 0.0115 | 0.3567 ± 0.0125 |

`density_prior` is the default because it improves (or preserves) SHD without
using test information.

### Step d: stabilized training controls (reduced h=6)

Compared two settings on seeds 17/18/19 with warm-start and density-prior
thresholding:

- Baseline: existing optimizer/training defaults.
- Stabilized: `optimizer=adamw`, `weight_decay=1e-4`, `grad_clip_norm=1.0`,
  `lr_schedule=cosine`, `min_learning_rate=1e-5`,
  `early_stopping_patience=3`, `coupling_init_scale=0.5`.

| D | Setting | Learned val MSE (mean ± std) | Learned test MSE (mean ± std) | Identity test MSE (mean ± std) |
|---:|---|---:|---:|---:|
| 5 | Baseline | 0.2594 ± 0.3227 | 0.2544 ± 0.3118 | 0.0297 ± 0.0102 |
| 5 | Stabilized | 0.0234 ± 0.0023 | 0.0260 ± 0.0039 | 0.0283 ± 0.0029 |
| 20 | Baseline | 6.34e20 ± 8.97e20 | 6.61e20 ± 9.35e20 | 0.0230 ± 0.0017 |
| 20 | Stabilized | 0.0220 ± 0.0016 | 0.0225 ± 0.0021 | 0.0283 ± 0.0029 |

These controls remove the learned-mode divergence seen on reduced D20 runs.

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

## D=20 stability

Setup: [stability_d20.yaml](/workspaces/ocgssm/configs/stability_d20.yaml)
(`sanity_d20.yaml` shrunk to `n_steps=1500`, `switch_at=1500`, 10 epochs,
seed 17, offline, Adam lr 0.003, no clipping) run through
`python scripts/diagnose_stability.py --variants identity oracle`. The
"spectral radius" is the largest |eigenvalue| of the effective one-step state
transition `diag(vec(decay)) + (diag(gate) A_cross) (x) W` evaluated per
window/time step on a 32-window probe batch (max over the probe), and
"hidden max" is max |hidden state| over the same probe.

### Before the fix (`results/diagnose_stability_d20_before.json`)

| Variant | Epoch | Train loss | Spectral radius (max) | Hidden max | Lag-input max row sum | Max grad norm (coupling / gate / input / logA) |
|---|---:|---:|---:|---:|---:|---|
| identity | 0 (init) | - | 0.493 | 0.47 | 0 | - |
| identity | 1 | 0.0405 | 0.530 | 0.34 | 0.054 | 0 / 0 / 4.7e-3 / 5.8e-4 |
| identity | 10 | 0.0236 | 0.624 | 0.26 | 0.356 | 0 / 0 / 1.8e-3 / 2.0e-4 |
| oracle | 0 (init) | - | **2.011** | **9.3e4** | 0 | - |
| oracle | 1 | 3.43e6 | 1.591 | 853 | 0.101 | 6.6e9 / 1.0e9 / 2.0e9 / 2.2e8 |
| oracle | 2 | 4982 | 1.569 | 738 | 0.110 | 8.8e5 / 1.8e5 / 1.8e5 / 4.4e4 |
| oracle | 10 | 2634 | 1.554 | 570 | 0.112 | 5.1e5 / 9.5e4 / 1.0e5 / 2.5e4 |

Finding: the oracle variant is unstable **at initialization, before any
gradient step**. The spectral radius of the one-step transition is 2.0 at
epoch 0 and the hidden state reaches 9.3e4 inside a single 24-step window,
so the first forecast loss is 3.4e6 and the first-epoch gradients are 1e8-1e9.
Training then only slowly shrinks the radius toward ~1.55; it never becomes
contractive. The cause is the un-normalized cross-variable coupling: at D=20,
sparsity 0.1 the aggregate cross-adjacency has row sums up to 8 (mean 4.35;
dense would be 19), `coupling_projection` is initialized with
`||W||_2 ~ 0.97`, and the gate starts at ~0.5, so the coupling term alone has
gain ~4 on top of decay ~0.5. At D=5 (sparsity 0.2) row sums are <= 3 and the
realized radius stayed below one, which is why the same code trained there.
The lag input mixer and lag forecast readout row sums stay below 0.45 for both
variants and are not implicated; `logA`/decay is not implicated either
(identity decay drifts only 0.49 -> 0.62).
