# PLAN (Copilot: tick boxes only after acceptance tests pass)
- [x] M0 skeleton, requirements.txt, pyproject, CI-free pytest config
- [x] M1 data.py: SVAR generator (abrupt+gradual), real loaders (ETT, Electricity, Traffic via CSV path), windowing, RevIN
- [x] M2 graph.py: lagged logits, structural score, online update, EMA, regime detector
- [x] M3 ssm.py + model.py: CausalSSMLayer, stack, head, graph modes
- [x] M4 baselines.py: DLinear, RevIN-Linear, Transformer-lite, NoGraph-SSM
- [ ] M5 train.py: offline + online prequential loops, logging to results/*.json
- [ ] M6 eval.py: all metrics in SPEC
- [ ] M7 run.py CLI + configs/*.yaml (synthetic_abrupt, synthetic_gradual, ett, electricity, traffic, ablation, stress)
- [ ] M8 notebooks/colab_train.ipynb (clone, install, run config, save results to Drive) + README run commands
- [ ] M9 stretch: graph ensemble + CRPS
## Decisions
- Synthetic generation returns a dictionary of torch tensors (`series`, `graph`, `coefficients`); during a gradual switch the graph and coefficients interpolate linearly.
- RevIN uses population standard deviation over each window's time dimension.
- Graph sparsity regularization is the mean edge probability, and detector alerts temporarily boost the online graph optimizer's learning rate.
- Identity adjacency contributes node-local embeddings but masks the diagonal from cross-variable coupling; learned forecasts use EMA values with a straight-through gradient to current logits.
## Blockers
(none yet)