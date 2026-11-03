# Continuation Task 0 (no new model runs)

## a) Switch location audit

| Source | pair | graph changes at t | test windows | all-pre | straddle | all-post | shared edges |
|---|---|---|---:|---:|---:|---:|---:|
| S4 held-out | [53, 709] | [3600] | 771 (saved 771, first target 3224) | 371 | 5 | 395 | 0.07 |
| S4 held-out | [59, 811] | [3600] | 771 (saved 771, first target 3224) | 371 | 5 | 395 | 0.06 |
| S4 held-out | [61, 913] | [3600] | 771 (saved 771, first target 3224) | 371 | 5 | 395 | 0.09 |
| D20 reduced | [37, 401] | [3600] | 771 | 371 | 5 | 395 | 0.17 |
| D20 reduced | [43, 503] | [3600] | 771 | 371 | 5 | 395 | 0.04 |
| D20 reduced | [47, 607] | [3600] | 771 | 371 | 5 | 395 | 0.10 |

D20 reduced campaign: saved run files contain no per-window timestamps (examples found: 0); its window counts follow from the frozen protocol (results/d20_equal_training/protocol.yaml: test slice [3200, 3999], 771 windows, graph timestamps [3223, 3993], switch_at 3600) and the identical make_windows(series[3200:4000], 24, 6) call (scripts/d20_equal_training.py:796), which the S4 rows verify against saved S4 outputs. Conclusion: both campaigns had the switch inside the test slice; 395/771 windows are fully post-switch under a graph sharing only 4-17% of edges with the training graph.

## b) S5 held-out per-window tables (seeds 79/1301, 83/1409, 89/1511, 97/1613, 101/1709)

Stream windows 971; first post-switch window 476; eligible post-switch windows 495 (fewer than 500: the [200,500) and [0,500) ranges are truncated to the available windows, not padded). Paired circular block bootstrap, 12-window blocks, 2000 resamples, equal-seed means.

### post[0,200) frozen gate - frozen gate window; stream windows [476, 676) = 200 per seed

| Variant | mean MSE [95% CI] | minus dense_online [95% CI] | learned_det_on minus variant [95% CI] |
|---|---|---|---|
| identity_online | 0.01675 [0.01615, 0.01737] | -0.00041 [-0.00063, -0.00019] | +0.00053 [+0.00038, +0.00070] |
| dense_online | 0.01716 [0.01656, 0.01781] | - | +0.00012 [+0.00002, +0.00021] |
| random_online | 0.01688 [0.01628, 0.01746] | -0.00028 [-0.00046, -0.00009] | +0.00040 [+0.00027, +0.00053] |
| learned_frozen | 0.01894 [0.01825, 0.01960] | +0.00178 [+0.00156, +0.00199] | -0.00166 [-0.00183, -0.00148] |
| learned_det_off | 0.01748 [0.01683, 0.01815] | +0.00031 [+0.00023, +0.00040] | -0.00020 [-0.00023, -0.00017] |
| learned_det_on | 0.01728 [0.01662, 0.01795] | +0.00012 [+0.00002, +0.00021] | - |
| oracle_online | 0.01558 [0.01508, 0.01607] | -0.00159 [-0.00178, -0.00140] | +0.00170 [+0.00154, +0.00187] |

### post[200,500) post hoc, descriptive - post hoc, descriptive; stream windows [676, 971) = 295 per seed (TRUNCATED)

| Variant | mean MSE [95% CI] | minus dense_online [95% CI] | learned_det_on minus variant [95% CI] |
|---|---|---|---|
| identity_online | 0.01655 [0.01598, 0.01713] | +0.00115 [+0.00098, +0.00133] | -0.00027 [-0.00033, -0.00020] |
| dense_online | 0.01540 [0.01492, 0.01594] | - | +0.00089 [+0.00075, +0.00102] |
| random_online | 0.01637 [0.01582, 0.01696] | +0.00097 [+0.00082, +0.00112] | -0.00008 [-0.00014, -0.00002] |
| learned_frozen | 0.01882 [0.01816, 0.01946] | +0.00342 [+0.00317, +0.00368] | -0.00253 [-0.00269, -0.00237] |
| learned_det_off | 0.01630 [0.01575, 0.01688] | +0.00090 [+0.00077, +0.00103] | -0.00001 [-0.00003, +0.00001] |
| learned_det_on | 0.01629 [0.01576, 0.01690] | +0.00089 [+0.00075, +0.00102] | - |
| oracle_online | 0.01463 [0.01419, 0.01509] | -0.00077 [-0.00088, -0.00067] | +0.00166 [+0.00149, +0.00183] |

### post[0,500) post hoc, descriptive - post hoc, descriptive; stream windows [476, 971) = 495 per seed (TRUNCATED)

| Variant | mean MSE [95% CI] | minus dense_online [95% CI] | learned_det_on minus variant [95% CI] |
|---|---|---|---|
| identity_online | 0.01663 [0.01620, 0.01709] | +0.00052 [+0.00034, +0.00070] | +0.00005 [-0.00004, +0.00015] |
| dense_online | 0.01611 [0.01571, 0.01653] | - | +0.00058 [+0.00047, +0.00069] |
| random_online | 0.01658 [0.01616, 0.01700] | +0.00046 [+0.00031, +0.00061] | +0.00011 [+0.00004, +0.00019] |
| learned_frozen | 0.01887 [0.01840, 0.01940] | +0.00275 [+0.00254, +0.00297] | -0.00218 [-0.00231, -0.00205] |
| learned_det_off | 0.01677 [0.01635, 0.01722] | +0.00066 [+0.00056, +0.00076] | -0.00008 [-0.00011, -0.00006] |
| learned_det_on | 0.01669 [0.01624, 0.01714] | +0.00058 [+0.00047, +0.00068] | - |
| oracle_online | 0.01501 [0.01466, 0.01538] | -0.00110 [-0.00121, -0.00099] | +0.00168 [+0.00156, +0.00181] |

### pre-switch [0,471) descriptive - descriptive (all targets < 2500); stream windows [0, 471) = 471 per seed

| Variant | mean MSE [95% CI] | minus dense_online [95% CI] | learned_det_on minus variant [95% CI] |
|---|---|---|---|
| identity_online | 0.01854 [0.01802, 0.01906] | +0.00236 [+0.00213, +0.00259] | -0.00301 [-0.00324, -0.00279] |
| dense_online | 0.01618 [0.01577, 0.01658] | - | -0.00065 [-0.00075, -0.00054] |
| random_online | 0.01806 [0.01757, 0.01855] | +0.00188 [+0.00168, +0.00209] | -0.00253 [-0.00273, -0.00233] |
| learned_frozen | 0.01538 [0.01501, 0.01573] | -0.00080 [-0.00094, -0.00067] | +0.00015 [+0.00009, +0.00021] |
| learned_det_off | 0.01554 [0.01514, 0.01591] | -0.00065 [-0.00076, -0.00054] | +0.00000 [+0.00000, +0.00000] |
| learned_det_on | 0.01554 [0.01516, 0.01592] | -0.00065 [-0.00075, -0.00054] | - |
| oracle_online | 0.01554 [0.01514, 0.01591] | -0.00065 [-0.00076, -0.00055] | +0.00000 [-0.00003, +0.00003] |

## c) ETTh2 H96

Not run in this continuation. The original partial ETTh2 H96 outputs (2/3 seeds) and its INCOMPLETE row are preserved unchanged.
