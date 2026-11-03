# S5b held-out (seeds 113/2111, 127/2203, 131/2309, 137/2411, 139/2503)

Gates commit `f334269a4390193e474449173e56ce462bbd21ec`; frozen learned_fast config {'buffer': 32, 'steps': 50, 'reset': True}.

Primary gate (learned_fast - dense_online over first 200 post-switch windows, CI upper < 0): **FAIL** - +0.001084 [+0.000846, +0.001334]

## post[0,200) gate (200 windows/seed)

| Variant | mean MSE | learned_fast minus variant [95% CI] |
|---|---:|---|
| identity_online | 0.02174 | +0.000750 [+0.000570, +0.000953] |
| dense_online | 0.02140 | +0.001084 [+0.000846, +0.001334] |
| learned_frozen | 0.02477 | -0.002283 [-0.002593, -0.001969] |
| learned_det_on | 0.02228 | +0.000209 [+0.000169, +0.000249] |
| learned_fast | 0.02249 | - |
| oracle_online | 0.01955 | +0.002932 [+0.002641, +0.003228] |

## post[200,495) descriptive (295 windows/seed)

| Variant | mean MSE | learned_fast minus variant [95% CI] |
|---|---:|---|
| identity_online | 0.02174 | -0.000950 [-0.001082, -0.000826] |
| dense_online | 0.01890 | +0.001883 [+0.001568, +0.002212] |
| learned_frozen | 0.02507 | -0.004284 [-0.004591, -0.003984] |
| learned_det_on | 0.02080 | -0.000010 [-0.000035, +0.000014] |
| learned_fast | 0.02079 | - |
| oracle_online | 0.01757 | +0.003218 [+0.002869, +0.003564] |

## pre-switch [0,471) descriptive (471 windows/seed)

| Variant | mean MSE | learned_fast minus variant [95% CI] |
|---|---:|---|
| identity_online | 0.01757 | -0.002456 [-0.002675, -0.002249] |
| dense_online | 0.01561 | -0.000500 [-0.000601, -0.000402] |
| learned_frozen | 0.01500 | +0.000117 [+0.000070, +0.000167] |
| learned_det_on | 0.01512 | -0.000003 [-0.000008, +0.000001] |
| learned_fast | 0.01511 | - |
| oracle_online | 0.01508 | +0.000035 [+0.000009, +0.000063] |

## Graph top-|E| F1 and detector

- learned_det_on (113, 2111): F1 pre 1.000, post200 0.269, final 0.486; FA=0 delay=3
- learned_det_on (127, 2203): F1 pre 0.998, post200 0.280, final 0.618; FA=0 delay=6
- learned_det_on (131, 2309): F1 pre 1.000, post200 0.270, final 0.513; FA=1 delay=2
- learned_det_on (137, 2411): F1 pre 1.000, post200 0.334, final 0.763; FA=1 delay=4
- learned_det_on (139, 2503): F1 pre 1.000, post200 0.439, final 0.800; FA=0 delay=7
- learned_fast (113, 2111): F1 pre 1.000, post200 0.197, final 0.343; FA=0 delay=3
- learned_fast (127, 2203): F1 pre 0.998, post200 0.256, final 0.588; FA=0 delay=6
- learned_fast (131, 2309): F1 pre 1.000, post200 0.206, final 0.359; FA=1 delay=63
- learned_fast (137, 2411): F1 pre 1.000, post200 0.289, final 0.553; FA=2 delay=4
- learned_fast (139, 2503): F1 pre 1.000, post200 0.360, final 0.760; FA=0 delay=7
