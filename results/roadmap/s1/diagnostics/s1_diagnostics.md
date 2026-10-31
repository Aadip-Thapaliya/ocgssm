# S1 diagnostics (development seeds 67/1009, 71/1103, 73/1201; validation only)

Gate slice: raw 2933-3199 (238 windows/seed); identical target timestamps and paired 12-window circular block bootstrap (2000 resamples, equal-seed). Identity/dense use the selected iterated head and the S1 training policy (cap 20, patience 5, tune checkpoint).

| Ratio | H1 [95% CI] | H2 [95% CI] |
|---|---|---|
| oracle_iterated/identity_iterated | 0.8195 [0.8050, 0.8344] | 0.8997 [0.8884, 0.9114] |
| oracle_iterated/dense_iterated | 0.9782 [0.9738, 0.9826] | 0.9883 [0.9850, 0.9916] |
| oracle_iterated/var_direct | 1.0288 [1.0202, 1.0375] | 1.0333 [1.0257, 1.0413] |
| ols/var_direct | 0.9715 [0.9660, 0.9767] | 0.9855 [0.9810, 0.9902] |
| ols/identity_iterated | 0.7738 [0.7599, 0.7888] | 0.8580 [0.8458, 0.8713] |
| dense_iterated/var_direct | 1.0518 [1.0445, 1.0593] | 1.0456 [1.0384, 1.0533] |

Decision rule: S1b if, at a horizon in {H1,H2}, OLS/DirectVAR paired 95% CI upper < 1 and oracle_iterated/DirectVAR CI upper >= 1; else S3.

- H1: OLS clearly beats DirectVAR = True; oracle_iterated clearly beats = False; S1b trigger = True
- H2: OLS clearly beats DirectVAR = True; oracle_iterated clearly beats = False; S1b trigger = True
- **Next branch: S1b**

## Oracle-support OLS rank diagnostics
Per horizon and target variable: lstsq (gelsd, float64) on train windows (raw 0-2399) with features [x_t, x_(t-1), 1] restricted to the true reachability support.

- 67/1009: 0/120 rank-deficient; condition number median 12.4, max 12.9; mean support size 23.9
- 71/1103: 0/120 rank-deficient; condition number median 11.6, max 13.0; mean support size 19.8
- 73/1201: 0/120 rank-deficient; condition number median 12.5, max 13.6; mean support size 23.4

## Reachability mask density (mean over 3 seeds)
Density = edges / D^2 (D=20, 400 entries per lag block). Self-edges (diagonal) count as legitimate lagged self-effects when present, matching the generator and mask conventions.

| H | Lag | Density | Edges | Self-edges |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.112 | 44.7 | 2.3 |
| 1 | 2 | 0.088 | 35.0 | 1.3 |
| 2 | 1 | 0.288 | 115.3 | 5.3 |
| 2 | 2 | 0.183 | 73.3 | 3.7 |
| 3 | 1 | 0.560 | 224.0 | 12.0 |
| 3 | 2 | 0.387 | 154.7 | 8.0 |
| 4 | 1 | 0.807 | 322.7 | 16.7 |
| 4 | 2 | 0.587 | 234.7 | 12.0 |
| 5 | 1 | 0.936 | 374.3 | 19.0 |
| 5 | 2 | 0.714 | 285.7 | 14.3 |
| 6 | 1 | 0.988 | 395.3 | 19.7 |
| 6 | 2 | 0.760 | 304.0 | 15.7 |

## oracle_iterated, 40-epoch cap, 67/1009
Best epoch 16, stopped at epoch 21, cap reached False; 2.6 min.

| Cap | H1 / DirectVAR [95% CI] | H2 / DirectVAR [95% CI] |
|---|---|---|
| 20 | 1.0225 [1.0074, 1.0374] | 1.0253 [1.0103, 1.0400] |
| 40 | 1.0225 [1.0074, 1.0374] | 1.0253 [1.0103, 1.0400] |

Wall time 13.2 min.
