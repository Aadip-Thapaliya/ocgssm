# Reproducing the final roadmap (S1-S6)

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
- Held-out data are only constructed after the stage `gates.yaml` is committed (`authorize_held_out` asserts the committed hash).

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

## Timing caveat
The laptop entered Windows Modern Standby several times (about 2026-11-02 01:44-15:53 and 2026-11-02 ~19:45 - 2026-11-03 09:33). Runs that straddled standby record inflated `elapsed_seconds` (for example 51,133 s); results are unaffected (deterministic), but those timings are invalid and are excluded or flagged in all runtime reports.
