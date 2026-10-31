# Roadmap s3: learned-graph convergence (development only)

learned_soft_iterated, cap 40, patience 5. Gate: cap hit on <= 1/3 seeds and gain from epochs 21-40 < 0.5% on every seed -> **PASS** (cap hits 0/3).

| Pair | Epochs | Best epoch | Cap hit | Best tune MSE (<=20) | Best (all) | Gain | Min |
|---|---:|---:|---|---:|---:|---:|---:|
| 67/1009 | 38 | 33 | False | 0.012589 | 0.012586 | 0.02% | 3.9 |
| 71/1103 | 39 | 34 | False | 0.011812 | 0.011808 | 0.03% | 4.0 |
| 73/1201 | 37 | 32 | False | 0.012209 | 0.012204 | 0.03% | 3.8 |

Informative validation ratios (learned / reference, paired 95% CI; not S2 gates):

| Reference | H1 | H2 |
|---|---|---|
| oracle_iterated | 1.0059 [1.0036, 1.0083] | 1.0022 [1.0001, 1.0043] |
| dense_iterated | 0.9839 [0.9811, 0.9868] | 0.9904 [0.9885, 0.9924] |
| var_direct | 1.0349 [1.0272, 1.0424] | 1.0356 [1.0275, 1.0427] |
