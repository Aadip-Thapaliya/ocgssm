# Roadmap s1 (development only)

Selected on tune 2400-2665 (mean H1/H2 MSE): **oracle_iterated**. Gate (validation 2933-3199): oracle/DirectVAR <= 1.05 at H1 and H2 -> **PASS**. Run time 18.2 min.

| Variant | H1 ratio [95% CI] | H2 ratio [95% CI] | H3 | H6 |
|---|---|---|---:|---:|
| oracle_lag | 1.0275 [1.0187, 1.0366] | 1.1294 [1.1168, 1.1424] | 1.0585 | 1.0332 |
| oracle_reachability | 1.0285 [1.0198, 1.0371] | 1.0365 [1.0294, 1.0441] | 1.0402 | 1.0446 |
| oracle_iterated | 1.0288 [1.0198, 1.0380] | 1.0333 [1.0260, 1.0413] | 1.0336 | 1.0295 |

| Variant | Pair | Epochs | Best epoch | Cap hit | Seconds |
|---|---|---:|---:|---|---:|
| oracle_lag | 67/1009 | 16 | 11 | False | 104 |
| oracle_lag | 71/1103 | 20 | 20 | True | 135 |
| oracle_lag | 73/1201 | 20 | 20 | True | 135 |
| oracle_reachability | 67/1009 | 12 | 7 | False | 75 |
| oracle_reachability | 71/1103 | 20 | 20 | True | 163 |
| oracle_reachability | 73/1201 | 13 | 8 | False | 83 |
| oracle_iterated | 67/1009 | 20 | 16 | True | 127 |
| oracle_iterated | 71/1103 | 20 | 20 | True | 134 |
| oracle_iterated | 73/1201 | 20 | 20 | True | 130 |
| var_direct | 67/1009 | 10 | 5 | False | 2 |
| var_direct | 71/1103 | 9 | 4 | False | 2 |
| var_direct | 73/1201 | 12 | 7 | False | 2 |
