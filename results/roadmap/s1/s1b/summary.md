# Roadmap s1b (development only)

Selected on tune 2400-2665 (mean H1/H2 MSE): **oracle_iterated_zero_ssm**. Gate (validation 2933-3199): oracle/DirectVAR <= 1.05 at H1 and H2 -> **PASS**. Clearly beats DirectVAR (CI upper < 1): {'H1': False, 'H2': False}. Run time 9.6 min.

| Variant | H1 ratio [95% CI] | H2 ratio [95% CI] | H3 | H6 |
|---|---|---|---:|---:|
| oracle_iterated_zero_ssm | 1.0282 [1.0197, 1.0367] | 1.0327 [1.0256, 1.0403] | 1.0333 | 1.0287 |
| oracle_reachability_zero_ssm | 1.0307 [1.0229, 1.0385] | 1.0368 [1.0299, 1.0441] | 1.0385 | 1.0422 |

| Variant | Pair | Epochs | Best epoch | Cap hit | Seconds |
|---|---|---:|---:|---|---:|
| oracle_iterated_zero_ssm | 67/1009 | 13 | 8 | False | 76 |
| oracle_iterated_zero_ssm | 71/1103 | 12 | 7 | False | 104 |
| oracle_iterated_zero_ssm | 73/1201 | 17 | 12 | False | 110 |
| oracle_reachability_zero_ssm | 67/1009 | 11 | 6 | False | 88 |
| oracle_reachability_zero_ssm | 71/1103 | 11 | 6 | False | 94 |
| oracle_reachability_zero_ssm | 73/1201 | 12 | 7 | False | 99 |
| var_direct | 67/1009 | 10 | 5 | False | 2 |
| var_direct | 71/1103 | 9 | 4 | False | 2 |
| var_direct | 73/1201 | 12 | 7 | False | 2 |
