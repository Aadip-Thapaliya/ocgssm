# Reproducing the final roadmap (S1-S6 and continuation)

Environment: Windows, CPU only (i7-13700H), Python 3.12.12, torch 2.15.0+cpu, numpy 2.5.4. The full package list is in `docs/requirements-freeze.txt`. Every run uses `torch.set_num_threads(1)` and is deterministic for a fixed data/model seed pair.

```powershell
cd C:\Users\aadip\Desktop\OCGSSM-Local
$env:PYTHONPATH="$PWD;$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q
```

## Seeds
- Development: 67/1009, 71/1103, 73/1201. Real data model seeds: 1009, 1103, 1201.
- Burned (never used for gates): 11/101, 17/202, 23/303, 29/303, 31/301, 37/401, 43/503, 47/607.
- S4 held-out: 53/709, 59/811, 61/913. S5 held-out: 79/1301, 83/1409, 89/1511, 97/1613, 101/1709.
- S4b stationary held-out: 103/1801, 107/1907, 109/2003. S5b held-out: 113/2111, 127/2203, 131/2309, 137/2411, 139/2503.
- S5b development-only tuning uses the original dev pairs 67/1009, 71/1103, 73/1201.
- Held-out data are only constructed after the stage `gates.yaml` is committed (`authorize_held_out` asserts the committed hash).
- ETTh2 H96 was not run in this continuation; the original two-seed partial outputs and INCOMPLETE row remain intact. S6b was not run.

## Frozen gate commits
| Stage | gates file | commit |
|---|---|---|
| S1 | results/roadmap/s1/protocol.yaml | 8404801 |
| S3 | results/roadmap/s1/s3 protocol | 882ea09 |
| S2, S2d | results/roadmap/final/{s2,s2d}/gates.yaml | f8273c4 |
| S5 tuning protocol | results/roadmap/final/s5/tuning_protocol.yaml | 234dd56 |
| S6 | results/roadmap/final/s6/gates.yaml | 9ef1ded |
| S4 | results/roadmap/final/s4/gates.yaml | cbd9040 |
| S5 held-out | results/roadmap/final/s5/gates.yaml | df79569 |
| S4b stationary | results/roadmap/final/continuation/s4b/gates.yaml | 7e941a9 |
| S5b tuning candidates | results/roadmap/final/continuation/s5b/tuning_protocol.yaml | 5518b61 |
| S5b held-out | results/roadmap/final/continuation/s5b/gates.yaml | f334269 |

## Commands (cached run JSONs are skipped; delete `runs`/pair folders to recompute)
```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m scripts.roadmap_final --stage s2 ; & $py -m scripts.roadmap_final --stage s2 --summarize
& $py -m scripts.roadmap_final --stage s2d ; & $py -m scripts.roadmap_final --stage s2d --summarize
& $py -m scripts.roadmap_final --stage s4 --pairs 0,1,2 ; & $py -m scripts.roadmap_final --stage s4 --summarize
& $py -m scripts.roadmap_s5 --phase tuning ; & $py -m scripts.roadmap_s5 --phase select
& $py -m scripts.roadmap_s5 --phase heldout --pairs 0,1,2,3,4 ; & $py -m scripts.roadmap_s5 --phase heldout --summarize
& $py -m scripts.roadmap_s6 --dataset ETTh1 --horizon 24   # also --horizon 96, and ETTh2
& $py -m scripts.roadmap_s6 --summarize
```
`scripts.job_queue N <logdir> "module|arg|..." ...` runs commands N at a time with timestamped logs; it was used for all campaigns. ETT CSVs come from https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ (placed in `data/`).

## Continuation analyses
Task 0 reads only saved outputs (no model training):
```powershell
& $py -m scripts.roadmap_task0
```
S4b uses the committed stationary gates and fresh held-out pairs:
```powershell
& $py -m scripts.roadmap_s4b --pairs 0,1,2
& $py -m scripts.roadmap_s4b --summarize
```
S5b's six tuning candidates are frozen in `results/roadmap/final/continuation/s5b/tuning_protocol.yaml`; each is evaluated on all three dev pairs, then `select` freezes the lowest first-200 post-switch dev MSE. The held-out gate (`gates.yaml`) was committed before building any S5b held-out data:
```powershell
foreach ($c in 0..5) { foreach ($p in 0..2) { & $py -m scripts.roadmap_s5b --phase tune --pair $p --cand $c } }
& $py -m scripts.roadmap_s5b --phase select
# Only after the committed S5b gates.yaml:
foreach ($p in 0..4) { & $py -m scripts.roadmap_s5b --phase heldout --pair $p --variant all }
& $py -m scripts.roadmap_s5b --phase summarize
```
S5b selected B=32, N=50, EMA reset on. The frozen primary gate failed; there was no retry. `learned_fast` resets `ema_logits`, the actual representation used by the forecaster, and replays only already-observed input windows for its selected 50 structural updates.

The continuation's S4 audit and pre-switch/post-switch descriptive tables are in `results/roadmap/final/continuation/task0/task0.md`. The S4b and S5b summaries, identity sidecars, and figures are stored alongside their seed-specific result files. Final continuation tag: `final-roadmap-v2`.

## Timing caveat
The laptop entered Windows Modern Standby several times (about 2026-11-02 01:44-15:53 and 2026-11-02 ~19:45 - 2026-11-03 09:33). Runs that straddled standby record inflated `elapsed_seconds` (for example 51,133 s); results are unaffected (deterministic), but those timings are invalid and are excluded or flagged in all runtime reports.
