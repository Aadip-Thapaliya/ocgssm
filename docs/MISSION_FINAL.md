MISSION
Finish the OCG-SSM roadmap (S2, S4, S5, S6, docs) autonomously in C:\Users\aadip\Desktop\OCGSSM-Local. I pre-approve everything described below, including held-out runs (once each, with frozen gates), real-data runs and the runtime limits stated here. Do not stop to ask except under "STOP ONLY IF". Never fabricate or round up results. A failed gate is a valid result: record it and continue.

WHAT IS ALREADY DECIDED (validation, dev seeds 67/1009, 71/1103, 73/1201)
- S1 PASSED: iterated readout selected. S1b gave no gain (1.028/1.033 vs DirectVAR), 40 epochs gave no change. Record in docs/DIAGNOSTICS.md that oracle_iterated stays ~3% above DirectVAR and ~5% above oracle-support OLS (OLS/identity 0.774 at H1, near the noise floor 0.01). Do not spend more time on this.
- S3 PASSED: learned_soft_iterated, cap 40, patience 5: 0/3 cap hits, gain <0.05%.
- Informative S2 numbers: learned/oracle 1.006 (H1), 1.002 (H2); learned/dense 0.984, 0.990 (CI below 1); learned/DirectVAR 1.035. Reachability masks become near-dense from H3, so H1 and H2 are the primary horizons; report H1..H6 as secondary.
- From now on every SSM variant and DirectVAR uses: iterated head, lr 0.001, batch 32, cap 40, patience 5, tune-slice checkpoint restore.

GLOBAL RULES
1. Select everything on validation or dev seeds. Held-out seeds and test slices are read only after gates.yaml for that stage is committed (git) with its hash recorded. Add or keep assertions that no held-out/test target is read earlier. If this is ever violated, discard those results and rerun with new seeds.
2. Never edit a gate after seeing results. One change at a time. At most 2 attempts per branch.
3. Paired circular block bootstrap, 2000 resamples, equal-seed weighting; 12-window blocks for synthetic, >=96-window blocks for real data.
4. Keep all tests passing; run the full suite at the end of every stage. Commit after every stage. Log decisions in docs/PLAN.md and docs/DIAGNOSTICS.md.
5. Always include var_direct and identity in tables. Report mean, median, per-seed values, best epoch and cap hits. Keep raw per-window errors. No clipping.
6. Seed sets: dev = 67/1009, 71/1103, 73/1201. Burned (never use for gates): 11/101, 17/202, 23/303, 29/303, 31/301, 37/401, 43/503, 47/607. S4 held-out = 53/709, 59/811, 61/913. S5 held-out = 79/1301, 83/1409, 89/1511, 97/1613, 101/1709.

STAGES (run in this order)

S2 formal (dev, validation slices; budget <= 1 h)
Variants: identity, dense, random (density-matched), learned_soft, learned_hard (threshold chosen on tune-slice forecast MSE from {0.3,0.4,0.5,0.6,0.7}, using the learned graph as a fixed mask, then retrain with the same policy), oracle, var_direct.
Gate (gate slice 2933-3199): learned/oracle <= 1.05 at H1 and H2; learned/dense CI upper < 1 at H1 and H2; learned/random CI upper < 1 at H1.
Final learned variant = learned_soft unless learned_hard beats it with CI upper < 1 at both H1 and H2 AND an online update rule for the hard mask (periodic refit with hysteresis) is implemented and tested. Default to soft.
If the gate fails: one attempt of a sparsity-weight sweep (3 values, validation only); if it still fails, record F2 and continue.

S2d scarcity sweep (dev, validation; budget <= 1 h)
Compare dense, learned, oracle at D=20 with training lengths 600, 1200, 2400, and D=10 and D=40 at 2400 (skip D=40 if one run takes >10 min). Report learned/dense and oracle/dense at H1 and H2 with CIs. Choose ONE harder configuration where learned/dense is lowest at H1 with CI upper < 1 and oracle still beats dense; commit it as "S4-hard". If none qualifies, skip S4-hard.

S4 held-out confirmation (budget <= 2 h)
Freeze gates.yaml and commit. Run seeds 53/709, 59/811, 61/913 on the default config (and S4-hard if chosen). Variants: identity, dense, random, learned (final), oracle, var_direct, dlinear.
Primary gate (default config, test slice 3200-3999): learned/dense CI upper < 1 at H1; learned/random CI upper < 1 at H1; learned/identity CI upper < 1 at H1 and H2.
Secondary: learned/oracle <= 1.05 at H1, H2. S4-hard is secondary evidence only.
If the primary gate fails: do NOT retune; write the F2 result (graph recovery works but no forecast benefit over dense) and continue to S5.

S5 adaptation (budget <= 3.5 h)
Setup: D=10, K=2, abrupt switch at 2500; train 0-1599, validation 1600-1999, stream 2000-2999. Pretrain offline on pre-switch train, then run the leak-safe prequential stream (delayed labels, timestamped pre-update graph snapshots). Variants: identity-online, dense-online, random-online, learned-frozen (no online updates), learned-online with detector off, learned-online with detector on, oracle-online (true graph switches at the true time).
Tune once on the dev-seed stream only, at most 8 configurations of EMA decay {0.9,0.95,0.98}, detector threshold {3.5,5}, boost {3,5}; choose lowest dev post-switch MSE; freeze. Reuse the existing frozen-threshold detector protocol (thresholds from validation, consecutive-outlier confirmation).
Primary metric: mean MSE over the first 200 windows after the switch. Gate: learned-online (detector on) < learned-frozen AND < dense-online, paired bootstrap CI excluding 0, over the 5 held-out seeds. Also report: pre-switch MSE, recovery time, post-switch graph F1/AUROC/SHD timeline, detector coverage/delay/false alarms, and detector on vs off.
Freeze gates.yaml and commit before running the held-out seeds. If the gate fails: F3 (benefit on stationary data, no adaptation gain); report honestly and continue.

S6 real data (budget <= 3 h; CPU only; no Colab)
Use ETTh1 and ETTh2 (7 variables, 17,420 hourly rows; standard 12/4/4-month chronological split, standardize with train statistics only). If CSVs are absent, download them from https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv (and ETTh2.csv). Weather/ETTm1 only if a CSV is already present and time remains. If downloads are blocked, report exactly which files to place where, and skip S6.
Input length 96, horizons 24 and 96. To fit CPU time: train on windows with stride 4, batch 64, cap 20 epochs, patience 5 (same for all variants), batched inference for offline evaluation (per-window timing on a 100-window sample only). If the iterated head diverges at horizon 96, report it and add the direct lag head as a named variant; choose the head on validation only.
Variants: identity, dense, random (density-matched to the mean learned adjacency), learned (final), var_direct, dlinear. 3 model seeds. Report standardized MSE/MAE, ratios to identity and to dense, with paired block-bootstrap CIs. Target: learned <= dense. Also report the learned graphs: heatmap, top edges, edge stability across seeds. Run online-adapted evaluation for learned vs dense only if time remains. No benefit claim beyond what the CIs support.

DOCS (budget <= 1 h, do last, do not skip)
Write docs/THESIS_RESULTS.md: one table per stage with CIs, figures (ratio-to-identity plots with per-seed points, scarcity sweep, post-switch MSE curves, graph heatmaps), and a claims table marking each thesis claim SUPPORTED / NOT SUPPORTED / INCONCLUSIVE with the evidence and limits (including: SSM variants ~3% behind DirectVAR on validation; small effect sizes; synthetic-only causal identification; real-data graphs have no ground truth). Write docs/REPRODUCE.md with exact commands, seeds, hashes and environment lock. Tag the final commit.

COMPUTE AND CREDIT MANAGEMENT
- Launch long jobs in the background with logs and the resume ledger; poll at most every 10-15 minutes with minimal output. Do not narrate waits.
- Measure peak memory in the first run of each stage. If peak RSS <= 1 GB and free RAM >= 2 x (n_processes x peak) + 1.5 GB, run up to 3 seed processes in parallel with torch.set_num_threads(1) each; otherwise sequential.
- Order of priority if budget runs short: S2 -> S4 -> S5 -> S6 -> docs. Stop all experiments at about 10.5 h of total work and write the docs with whatever is complete. If S6 is unfinished, finish ETTh1 only.
- Any single job estimated above 2 h: cut seeds or variants (keep identity, dense, learned, oracle, var_direct), declare the cut in gates.yaml, and continue.

STOP ONLY IF
- the test suite fails and two fix attempts do not resolve it; a leakage assertion fails; disk or memory is exhausted; or a required action needs my credentials or a manual step. Otherwise continue.

FINAL MESSAGE
Report one table: stage, gate, pass/fail, key ratios with CIs, runtime, artifact paths; then the claims table and any limits. No push or PR.