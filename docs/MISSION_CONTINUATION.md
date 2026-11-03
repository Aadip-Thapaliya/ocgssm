# Mission record: final roadmap continuation (received 2026-11-03 16:43 local)

Verbatim user mission (forwarded by coordinator, assignment call_o3MZS55Rn7kNCL2Cb43nwPZ5):

```text
MISSION: continue the final roadmap in C:\Users\aadip\Desktop\OCGSSM-Local. Same rules as before: select on validation/dev only, freeze and commit gates.yaml before any held-out data is built (authorize_held_out), never edit a gate after seeing results, at most one attempt per branch below, keep all tests passing, commit after each task, no push/PR. Pre-approved; do not ask except if tests fail twice, a leakage assertion fails, or disk/memory runs out. Do NOT delete or overwrite any existing result; add new rows/files and keep the original S4 as reported.

TASK 0 (<=30 min, no new model runs)
a) Verify where the regime switch falls in the S4 held-out data (3600?) and whether the earlier D20 reduced campaign (pairs 37/401, 43/503, 47/607, test slice 3200-3999) had a switch inside its test slice. Document in docs/DIAGNOSTICS.md.
b) From the saved S5 per-window errors, compute post-switch MSE for windows 0-200 (the frozen gate), 200-500, and 0-500 for identity, dense, random, learned_frozen, learned_det_off, learned_det_on, oracle, with paired block-bootstrap CIs. Label 200-500 and 0-500 "post hoc, descriptive". Add the pre-switch table with CIs.
c) ETTh2 H96: record "not run" and do not run it.
d) Tag the current commit "final-roadmap-v1".

TASK 1: S4b stationary held-out confirmation (budget <= 1.5 h)
Same as S4 except: generator with NO switch (switch_at = n_steps), so the whole test slice 3200-3999 is stationary. Fresh seed pairs 103/1801, 107/1907, 109/2003. Variants: identity, dense, random, learned, oracle, var_direct, dlinear; identical training policy to S4 (iterated head, cap 40, patience 5, lr 0.001, tune-slice checkpoint). Primary gate (unchanged from S4): learned/dense CI upper < 1 at H1; learned/random CI upper < 1 at H1; learned/identity CI upper < 1 at H1 and H2. Secondary: learned/oracle <= 1.05 at H1, H2. Freeze gates.yaml, commit, then run. No retry. Report beside the original S4 row, stating clearly that S4b corrects a slice design flaw and that S4 stays a frozen FAIL.

TASK 2: S5b faster post-switch graph tracking (budget <= 3.5 h)
One mechanism change to the learned online variant ("learned_fast"): on detection, (i) reset ema_logits to the current logits, and (ii) for the next N steps compute the structural update on a replay buffer of the most recent B windows instead of one window. Everything else (pretraining cache, boost, detector protocol, layout D=10, switch 2500, stream 2000-2999) unchanged. Tune on the dev-seed stream only, at most 6 configs from B in {32,64}, N in {50,100,200}, reset on/off; pick the lowest dev post-switch MSE (first 200 windows) and freeze. Held-out seeds (fresh): 113/2111, 127/2203, 131/2309, 137/2411, 139/2503. Variants: identity_online, dense_online, learned_frozen, learned_det_on (existing), learned_fast, oracle_online. Primary gate: learned_fast - dense_online CI excludes 0 on the negative side over the first 200 post-switch windows. Secondary: learned_fast - learned_det_on; graph top-|E| F1 timeline; windows 200-500 (descriptive). If the gate fails, F3 stands as final; no further attempts.

TASK 3 (OPTIONAL: only if >= 35% of budget remains after tasks 1-2)
S6b: graph mask on a competitive backbone. Per-channel DLinear plus a cross-channel lag term (last 24 steps of all channels) masked by M. Variants: DLinear (no cross term), +dense, +random (density-matched), +learned mask (structural learner fit on train, soft mask). ETTh1 and ETTh2, horizons 24 and 96, 5 seeds. Select on validation, freeze gates before test. Primary: learned < dense with CI upper < 1 and learned <= DLinear. Report whatever happens.

DOCS (<=45 min, last)
Update docs/THESIS_RESULTS.md: add S4b, S5b (and S6b if run) rows; update the claims table (SUPPORTED / NOT SUPPORTED / INCONCLUSIVE with evidence); add the S4 slice-design explanation and the pre-switch descriptive numbers; keep all original rows. Update docs/REPRODUCE.md with new commands, seeds and gate commits. Tag "final-roadmap-v2".

COMPUTE/CREDITS
Run jobs in the background via scripts.job_queue with logs; poll at most every 10-15 min. Measure peak RSS first; up to 3 parallel single-thread processes only if memory allows. Priority order: Task 0 -> Task 1 -> Task 2 -> docs -> Task 3. Stop experiments at ~8 h of total work and finish the docs. Report one final table: task, gate, pass/fail, key ratios with CIs, runtime.
```

Coordinator constraints (summary): distinct output paths/caches with config+seed identity checks; tag v1 on the pre-task commit (1c398e2) without moving `roadmap-final`; keep the original ETTh2 H96 partial outputs and INCOMPLETE row, adding "not run in this continuation"; Task 0 uses actual saved timestamps, half-open window ranges, discloses <500 eligible windows, never redefines the frozen 200-window gate; S5b candidates frozen before tuning, replay uses only observed windows (delayed labels, pre-update snapshots); EMA reset tested at the actual representation; baseline definitions preserved (DLinear stays DLinear; var_direct vs var_iterated disclosed); S6b disclosed as follow-up evaluation on already-examined test sets; docs reserved before optional S6b; 35% eligibility uses the 8 h budget.

Clock: mission start 16:43 local, 8 h work cap → experiments stop by ~00:00 (docs reserve 45 min).

