# OCG-SSM spec

Notation: D variables, K lags, N state dim, H horizon, L input window, B batch.

## Data
x_t in R^D. Windows (B, L, D) -> targets (B, H, D). Per-window standardization (RevIN-style, reversible).

## Graph module (graph.py)
- Lagged logits Lg in R^{K x D x D}; A^k = sigmoid(Lg[k] / tau). Entry A^k[i,j] = edge x_j(t-k) -> x_i(t). Lagged => acyclic by construction, no projection.
- Aggregate adjacency: Abar[i,j] = 1 - prod_k (1 - A^k[i,j]).
- Structural score on a window: S(G) = mean_t || x_t - sum_k (A^k * W^k) x_{t-k} ||^2 + lambda * sum |A^k|, with W^k in R^{D x D} learned weights (masked linear VAR). This gives the graph an identifiable signal independent of the SSM.
- Online update: one Adam/SGD step on Lg every `graph_stride` steps using S. EMA of logits (decay beta) used for the forecast. Cost O(K D^2) per step.
- Regime detector: running mean/var of S (EMA). Flag if (S - mean) / std > k. On flag: multiply graph lr by `boost` for `boost_steps` steps, then decay back.

## Causal SSM layer (ssm.py)
Per variable i, state h_t[i] in R^N:
  dt = softplus(Linear([x_t[i], e_i]))          # (B,D,1)
  Abar = exp(dt * -exp(logA))                    # diagonal, (B,D,N)
  coupling[i] = sum_j Abar_adj[i,j] * W h_{t-1}[j]
  h_t[i] = Abar * h_{t-1}[i] + gate * coupling[i] + dt * Bproj([x_t[i], e_i]) * x_t[i]
  y_t[i] = Cproj([x_t[i], e_i]) . h_t[i]
Graph embedding: node embeddings E in R^{D x d}; e = MLP([E_i, sum_j Abar[i,j] E_j]).
Complexity: linear in L, O(D^2 N) per step. Stack n_layers with residual + LayerNorm over the last dim.

## Head
Last-step state (B,D,N) -> Linear -> (B, H, D).

## Training modes
1. offline: standard train/val/test with fixed graph logits trained jointly (used for ablations).
2. online (prequential): stream windows in order; at each step predict, record error, then update model (forecast loss) and graph (structural score). Report running MSE.

## Graph modes (ablation switch)
`identity` (Abar = I), `learned` (online), `oracle` (given true graph, synthetic only), `random`.

## Synthetic data
SVAR(D, K): sparse true graphs G1 then G2, switch at time T (abrupt) or linear blend over `blend` steps (gradual). Stable coefficients (spectral radius < 0.9). Return series + true graph per time step.

## Metrics
Forecast: MSE, MAE. Graph: SHD, F1, precision, recall, AUROC vs true graph. Adaptation: recovery time (steps after switch until SHD back within 10% of pre-switch level), regret vs oracle, detection delay. Efficiency: ms/step, peak memory, params.

## Baselines
DLinear, RevIN+Linear, NoGraph-SSM (identity mode), simple Transformer (PatchTST-lite). Optional if time: S-Mamba-style (bidirectional Mamba block, pure PyTorch).

## Stretch (only after all milestones green)
K Gumbel-sigmoid graph samples, average forecast, CRPS from sample spread.