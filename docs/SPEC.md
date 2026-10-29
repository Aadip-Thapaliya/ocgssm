# OCG-SSM spec

Notation: D variables, K lags, N state dim, H horizon, L input window, B batch.

## Data
x_t in R^D. Windows (B, L, D) -> targets (B, H, D). CausalSSM and RevINLinear normalize each input window reversibly; the structural VAR objective operates on raw observed values. Other forecasting baselines retain their own model-specific input handling.

## Graph module (graph.py)
- Lagged logits Lg in R^{K x D x D}; A^k = sigmoid(Lg[k] / tau). Entry A^k[i,j] = edge x_j(t-k) -> x_i(t). Lagged => acyclic by construction, no projection.
- Aggregate adjacency: Abar[i,j] = 1 - prod_k (1 - A^k[i,j]).
- Structural score on a window: S(G) = mean_t || x_t - sum_k (A^k * W^k) x_{t-k} ||^2 + lambda * sum |A^k|, with W^k in R^{D x D} learned weights (masked linear VAR). This gives the graph an identifiable signal independent of the SSM.
- Online update: one Adam/SGD step on Lg every `graph_stride` steps using S. EMA of logits (decay beta) used for the forecast. Cost O(K D^2) per step.
- Regime detector: robust validation median/MAD baseline for S. During test the baseline is frozen; the detector starts disarmed and arms after `detector_rearm_steps` consecutive inlier scores. It requires `detector_confirm_steps` consecutive outliers to flag, then disarms until another normal streak. On flag: multiply graph lr by `boost` for `boost_steps` steps, then decay back. The threshold is at least the configured floor and validation maximum z-score plus 0.25.

## Causal SSM layer (ssm.py)
Per variable i, state h_t[i] in R^N:
  dt = softplus(Linear([x_t[i], e_i]))          # (B,D,1)
  Abar = exp(dt * -exp(logA))                    # diagonal, (B,D,N)
  coupling[i] = sum_j Abar_adj[i,j] * W h_{t-1}[j]
  h_t[i] = Abar * h_{t-1}[i] + gate * coupling[i] + dt * Bproj([x_t[i], e_i]) * x_t[i]
  y_t[i] = Cproj([x_t[i], e_i]) . h_t[i]
Graph embedding: node embeddings E in R^{D x d}; e = MLP([E_i, sum_j Abar[i,j] E_j]).
Complexity: linear in L, O(D^2 N) per step. Stack n_layers with residual + LayerNorm over the last dim.

Stability: before each scan, the off-diagonal adjacency and state coupling
projection are each divided by `max(1, spectral_norm)`. The coupling gate is
multiplied by `1 - max_i(Abar_i)` per state dimension, so the recurrent
operator norm is bounded by the largest decay plus the remaining coupling
budget. This prevents cross-variable recurrence from exponentially amplifying
hidden states; it does not clip forecasts or training errors.

## Head
Last-step state (B,D,N) -> Linear -> (B, H, D).

## Training modes
1. offline: forecast models train on training windows; learned graphs separately fit their masked VAR structural objective on those windows and refresh EMA logits after each batch.
2. online (prequential): cold-start (default) or train-only warm-start is labeled in the result. Predict before updating, and defer each horizon's label until its observation time. Forecast and graph optimizers have disjoint ownership. The forecast uses the EMA adjacency recorded before each graph update.

## Graph modes (ablation switch)
`identity` (self-lags only), `learned` (structural objective/online updates), `oracle` (given true graph, synthetic only), `random` (density-matched), `dense` (all entries including self-lags).

## Synthetic data
SVAR(D, K): sparse true graphs G1 then G2, switch at time T (abrupt) or linear blend over `blend` steps (gradual). Stable coefficients (spectral radius < 0.9). Return series + true graph per time step. Optional `cross_variable_scale` (default 1.0) increases off-diagonal coefficient magnitudes before companion-radius rescaling; it defines a separate synthetic-data version and leaves noise scale fixed.

## Metrics
Forecast: MSE, MAE. Graph: timestamp-aligned SHD, F1, precision, recall, AUROC vs true graph, including fixed control masks. Online history snapshots are the EMA masks used for each forecast; detector events carry observation timestamps. Adaptation: sustained recovery time is the first 10-step SHD window within 10% (at least one edge) of the median SHD in the final 20 post-switch evaluation steps. Candidate windows must end before this retrospective reference tail; insufficient reference data or no convergence is `null`. This reference is evaluation-only and is never exposed to training or detection. Also report regret vs oracle and detection delay. Efficiency: ms/step, peak memory, params.

For online learned-graph reports, `graph.snapshots` stores the pre-update EMA adjacency used for each forecast, with one lag-indexed `(K, D, D)` matrix per entry in `graph.timestamps`; `graph.snapshot_semantics` identifies this timing. Adaptation reports record `sustained_recovery_time` and `recovery_sustain_steps`. Training stops with a `FloatingPointError` if forecast outputs, losses, gradients, or updated parameters become non-finite.

## Baselines
DLinear, RevIN+Linear, NoGraph-SSM (identity mode), simple Transformer (PatchTST-lite). Optional if time: S-Mamba-style (bidirectional Mamba block, pure PyTorch).

## Stretch (only after all milestones green)
K Gumbel-sigmoid graph samples, average forecast, CRPS from sample spread.