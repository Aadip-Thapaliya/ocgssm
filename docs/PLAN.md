# PLAN (Copilot: tick boxes only after acceptance tests pass)
- [x] M0 skeleton, requirements.txt, pyproject, CI-free pytest config
- [x] M1 data.py: SVAR generator (abrupt+gradual), real loaders (ETT, Electricity, Traffic via CSV path), windowing, RevIN
- [x] M2 graph.py: lagged logits, structural score, online update, EMA, regime detector
- [x] M3 ssm.py + model.py: CausalSSMLayer, stack, head, graph modes
- [x] M4 baselines.py: DLinear, RevIN-Linear, Transformer-lite, NoGraph-SSM
- [x] M5 train.py: offline + online prequential loops, logging to results/*.json
- [x] M6 eval.py: all metrics in SPEC
- [x] M7 run.py CLI + configs/*.yaml (synthetic_abrupt, synthetic_gradual, ett, electricity, traffic, ablation, stress)
- [x] M8 notebooks/colab_train.ipynb (clone, install, run config, save results to Drive) + README run commands
- [ ] M9 stretch: graph ensemble + CRPS
## Decisions
- Synthetic generation returns a dictionary of torch tensors (`series`, `graph`, `coefficients`); during a gradual switch the graph and coefficients interpolate linearly.
- RevIN uses population standard deviation over each window's time dimension.
- Graph sparsity regularization is the mean edge probability, and detector alerts temporarily boost the online graph optimizer's learning rate.
- Identity adjacency contributes node-local embeddings but masks the diagonal from cross-variable coupling; learned forecasts use EMA values with a straight-through gradient to current logits.
- Gate work: `tests/test_baselines.py` was already tracked, so the requested Step 0 staging/commit was unnecessary.
- Oracle fix: graph schedules are aligned to each window's last input step (`start + input_length - 1`); the old test schedule was one step late. Offline training/evaluation now groups windows by consecutive graph changes, so no batch crosses a graph regime and each graph is installed for its group. Online prequential forecasts install the matching graph before prediction. This also handles each distinct graph during a gradual blend without changing the SSM forward signature.
- Oracle graph metrics now compare the final evaluation graph with the true graph at the final evaluated window's last input time; a switching synthetic-run regression asserts zero SHD. Forecast-level schedule tests prove forecasts change with the supplied oracle graph.
- Environment finding: `configs/gate.yaml` and `configs/sanity.yaml` were present but untracked before this work. The run-config test now checks the required benchmark configs as a subset, so task-specific configs do not invalidate it.
- Oracle-fix acceptance: `pytest -q` passed all 28 tests (one pre-existing Transformer nested-tensor warning).
## Blockers
(none yet)