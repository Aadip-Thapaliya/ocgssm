# S5b development-only configuration selection

Candidates were frozen in `tuning_protocol.yaml` (commit 5518b61) before any S5b tuning runs. Selection is the lowest mean learned_fast MSE over the first 200 post-switch stream windows [476,676), averaged equally over development pairs 67/1009, 71/1103, and 73/1201. Results below use dev targets only.

| Candidate | B | N | EMA reset | Dev MSE |
|---:|---:|---:|---|---:|
| 0 | 32 | 50 | on | 0.02016124 |
| 1 | 32 | 100 | off | 0.02018704 |
| 2 | 32 | 200 | on | 0.02017100 |
| 3 | 64 | 50 | off | 0.02021706 |
| 4 | 64 | 100 | on | 0.02026816 |
| 5 | 64 | 200 | off | 0.02026729 |

Selected candidate 0: B=32, N=50, reset on. The frozen S5 learned_det_on configuration 6 reference is 0.01998918 on the same dev metric; learned_fast did not beat that reference during selection. The frozen rule was to select the lowest candidate, not reject the mechanism based on dev performance.

Tuning runtime: 53m37s wall, from 17:23:40 to 18:17:17; 18 runs (6 candidates × 3 dev pairs), no held-out data accessed.
