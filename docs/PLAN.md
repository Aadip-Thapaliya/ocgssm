# PLAN (Copilot: tick boxes only after acceptance tests pass)
- [ ] M0 skeleton, requirements.txt, pyproject, CI-free pytest config
- [ ] M1 data.py: SVAR generator (abrupt+gradual), real loaders (ETT, Electricity, Traffic via CSV path), windowing, RevIN
- [ ] M2 graph.py: lagged logits, structural score, online update, EMA, regime detector
- [ ] M3 ssm.py + model.py: CausalSSMLayer, stack, head, graph modes
- [ ] M4 baselines.py: DLinear, RevIN-Linear, Transformer-lite, NoGraph-SSM
- [ ] M5 train.py: offline + online prequential loops, logging to results/*.json
- [ ] M6 eval.py: all metrics in SPEC
- [ ] M7 run.py CLI + configs/*.yaml (synthetic_abrupt, synthetic_gradual, ett, electricity, traffic, ablation, stress)
- [ ] M8 notebooks/colab_train.ipynb (clone, install, run config, save results to Drive) + README run commands
- [ ] M9 stretch: graph ensemble + CRPS
## Decisions
(none yet)
## Blockers
(none yet)