# Continuation 2 Task A: pooled S5 + S5b (saved outputs only)

All comparisons are post hoc and descriptive, including the original frozen first-200 gate displayed here as descriptive pooled analysis. Ten independent held-out seed pairs are pooled: the original five S5 and five fresh S5b pairs. Means give each seed equal weight. Paired circular block bootstrap: 12-window blocks, 2000 resamples; for each bootstrap draw, identical within-seed circular block indices are used across variants. Reported contrasts are learned_det_on minus comparator. All ranges are half-open.

## post_switch_[0,200)_post_hoc_descriptive

Stream window indices [476, 676); 200 windows per seed.

| Variant | Mean MSE [95% CI] | learned_det_on minus variant [95% CI] |
|---|---:|---:|
| learned_det_on | 0.019778 [0.019159, 0.020405] | - |
| dense_online | 0.019282 [0.018724, 0.019874] | +0.000496 [+0.000369, +0.000633] |
| identity_online | 0.019244 [0.018648, 0.019818] | +0.000534 [+0.000426, +0.000653] |
| oracle_online | 0.017565 [0.017045, 0.018100] | +0.002213 [+0.002041, +0.002385] |

## post_switch_[200,495)_post_hoc_descriptive

Stream window indices [676, 971); 295 windows per seed.

| Variant | Mean MSE [95% CI] | learned_det_on minus variant [95% CI] |
|---|---:|---:|
| learned_det_on | 0.018543 [0.018073, 0.019001] | - |
| dense_online | 0.017153 [0.016778, 0.017559] | +0.001391 [+0.001205, +0.001574] |
| identity_online | 0.019146 [0.018664, 0.019645] | -0.000603 [-0.000674, -0.000535] |
| oracle_online | 0.016101 [0.015748, 0.016462] | +0.002442 [+0.002246, +0.002648] |

## pre_switch_[0,471)_post_hoc_descriptive

Stream window indices [0, 471); 471 windows per seed.

| Variant | Mean MSE [95% CI] | learned_det_on minus variant [95% CI] |
|---|---:|---:|
| learned_det_on | 0.015326 [0.015094, 0.015564] | - |
| dense_online | 0.015899 [0.015659, 0.016162] | -0.000573 [-0.000646, -0.000507] |
| identity_online | 0.018057 [0.017723, 0.018409] | -0.002731 [-0.002897, -0.002572] |
| oracle_online | 0.015308 [0.015066, 0.015539] | +0.000019 [-0.000002, +0.000040] |

## Saturation diagnostic availability

UNAVAILABLE: saved records contain no pre-update EMA graph snapshots; only aggregate graph_timeline metrics were retained. No model was rerun and no topology proxy substituted.

Files were checked for saved per-window graph history/snapshot fields. Both the S5 and S5b online JSONs retain forecast errors, detections, and aggregate graph metrics (F1/AUROC/SHD/top-|E| F1), but not the requested pre-update EMA adjacency matrices. Therefore the mean probability on newly true edges vs other entries at windows 0, 50, 100, and 200—and the <0.05 determination—cannot be calculated from saved data. No new model runs were made.
