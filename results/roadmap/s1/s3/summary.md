# Roadmap s3 (development only)

Selected on tune 2400-2665 (mean H1/H2 MSE): **learned_soft_iterated**. Gate (validation 2933-3199): oracle/DirectVAR <= 1.05 at H1 and H2 -> **PASS**. Clearly beats DirectVAR (CI upper < 1): {'H1': False, 'H2': False}. Run time 11.8 min.

| Variant | H1 ratio [95% CI] | H2 ratio [95% CI] | H3 | H6 |
|---|---|---|---:|---:|
| learned_soft_iterated | 1.0349 [1.0269, 1.0428] | 1.0356 [1.0284, 1.0433] | 1.0340 | 1.0298 |

| Variant | Pair | Epochs | Best epoch | Cap hit | Seconds |
|---|---|---:|---:|---|---:|
| learned_soft_iterated | 67/1009 | 38 | 33 | False | 235 |
| learned_soft_iterated | 71/1103 | 39 | 34 | False | 241 |
| learned_soft_iterated | 73/1201 | 37 | 32 | False | 228 |
| var_direct | 67/1009 | 10 | 5 | False | 2 |
| var_direct | 71/1103 | 9 | 4 | False | 1 |
| var_direct | 73/1201 | 12 | 7 | False | 2 |
