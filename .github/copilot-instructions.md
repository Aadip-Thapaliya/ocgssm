# Project rules
- Project: OCG-SSM (online causal-graph-conditioned SSM for non-stationary multivariate forecasting). Read docs/SPEC.md first, docs/PLAN.md second.
- Stack: Python 3.11, PyTorch only. No mamba-ssm, no CUDA kernels. Pure-PyTorch sequential scan.
- Every function: type hints + docstring with tensor shapes.
- Every module ships with a pytest file in tests/. Run pytest after each milestone; fix until green.
- Config via YAML (configs/), no hardcoded hyperparameters. Seeds fixed and logged.
- Never edit docs/SPEC.md. Update docs/PLAN.md checkboxes only after a milestone's acceptance tests pass.
- Do not add dependencies beyond requirements.txt without noting it in PLAN.md.
- CPU smoke runs must finish in under 2 minutes. GPU runs are launched by the user on Colab.
- If a requirement in SPEC.md is ambiguous, pick the simplest option, write the choice in PLAN.md under "Decisions", and continue. Do not stop to ask.