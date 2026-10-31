"""Forecasting model built from graph-conditioned causal SSM layers."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from ocg.data import revin_normalize
from ocg.graph import GraphLearner
from ocg.ssm import CausalSSMLayer


def companion_reachability(adjacency: Tensor, horizon: int) -> Tensor:
    """Return Boolean (H,K,D,D) supports of VAR companion powers.

    Entry [i,j] of lag block k means x_j[t+1-k] -> x_i (generator orientation).
    For horizon h the mask on lag k is the support of block (0,k-1) of C^h with
    C = [[A1, ..., AK], [I, 0, ...], ...]; for K=2, h=2 gives A1@A1+A2 and A1@A2.
    """
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape (K,D,D)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    n_lags, n_variables, _ = adjacency.shape
    size = n_lags * n_variables
    support = (adjacency.detach() > 0).to(torch.float64)
    companion = torch.zeros(size, size, dtype=torch.float64, device=adjacency.device)
    companion[:n_variables] = torch.cat(tuple(support), dim=1)
    if n_lags > 1:
        companion[n_variables:, : size - n_variables] = torch.eye(
            size - n_variables, dtype=torch.float64, device=adjacency.device
        )
    power = torch.eye(size, dtype=torch.float64, device=adjacency.device)
    masks = []
    for _ in range(horizon):
        power = ((power @ companion) > 0).to(torch.float64)
        masks.append(power[:n_variables].reshape(n_variables, n_lags, n_variables).transpose(0, 1))
    return torch.stack(masks).to(dtype=adjacency.dtype)


class CausalSSM(nn.Module):
    """Forecast (B,L,D) windows as (B,H,D), with graph-masked lag input mixing."""

    def __init__(
        self,
        n_variables: int,
        state_dim: int,
        horizon: int,
        n_lags: int,
        n_layers: int = 1,
        embedding_dim: int = 8,
        graph_mode: str = "identity",
        graph_learner: GraphLearner | None = None,
        oracle_graph: Tensor | None = None,
        random_seed: int = 0,
        random_sparsity: float | None = None,
        use_direct_lag_head: bool = True,
        use_ssm_coupling: bool = True,
        graph_mask_threshold: float | None = None,
        direct_head_mode: str = "lag",
        zero_init_ssm_head: bool = False,
    ) -> None:
        """Initialize the SSM stack, graph mode, and per-variable forecast head."""
        super().__init__()
        if direct_head_mode not in {"lag", "reachability", "iterated"}:
            raise ValueError("direct_head_mode must be lag, reachability, or iterated")
        if direct_head_mode == "reachability" and graph_mode == "learned":
            raise ValueError("reachability direct head is not implemented for learned graphs")
        if n_variables < 1 or state_dim < 1 or horizon < 1 or n_lags < 1 or n_layers < 1:
            raise ValueError("model dimensions must be positive")
        if graph_mode not in {"identity", "learned", "oracle", "random", "dense"}:
            raise ValueError(
                "graph_mode must be identity, learned, oracle, random, or dense"
            )
        if graph_mode != "learned" and graph_learner is not None:
            raise ValueError("graph_learner can only be passed in learned mode")
        if graph_mode == "oracle" and oracle_graph is None:
            raise ValueError("oracle_graph is required in oracle mode")
        if graph_mode != "oracle" and oracle_graph is not None:
            raise ValueError("oracle_graph can only be passed in oracle mode")
        if graph_mode == "random" and random_sparsity is None:
            raise ValueError("random_sparsity is required in random mode")
        if random_sparsity is not None and not 0.0 <= random_sparsity <= 1.0:
            raise ValueError("random_sparsity must be in [0,1]")
        if graph_mask_threshold is not None and not 0.0 <= graph_mask_threshold <= 1.0:
            raise ValueError("graph_mask_threshold must be in [0,1]")
        if graph_mask_threshold is not None and graph_mode != "learned":
            raise ValueError("graph_mask_threshold is only valid in learned mode")
        if graph_mode == "learned" and graph_learner is not None:
            if graph_learner.n_variables != n_variables or graph_learner.n_lags != n_lags:
                raise ValueError("graph learner dimensions must match model dimensions")
        if oracle_graph is not None:
            if oracle_graph.shape != (n_lags, n_variables, n_variables):
                raise ValueError("oracle_graph must have shape (K,D,D)")
            if not torch.isfinite(oracle_graph).all() or torch.any((oracle_graph < 0) | (oracle_graph > 1)):
                raise ValueError("oracle_graph entries must be finite and in [0,1]")

        self.n_variables = n_variables
        self.n_lags = n_lags
        self.horizon = horizon
        self.graph_mode = graph_mode
        self.graph_mask_threshold = graph_mask_threshold
        self.use_direct_lag_head = use_direct_lag_head
        self.use_ssm_coupling = use_ssm_coupling
        if graph_mode == "learned":
            self.graph_learner = graph_learner or GraphLearner(n_variables, n_lags)
        else:
            self.graph_learner = None

        if graph_mode == "random":
            generator = torch.Generator().manual_seed(random_seed)
            edges_per_lag = round(float(random_sparsity) * n_variables**2)
            fixed_graph = torch.zeros(n_lags, n_variables, n_variables)
            for lag_index in range(n_lags):
                edge_indices = torch.randperm(n_variables**2, generator=generator)[
                    :edges_per_lag
                ]
                fixed_graph[lag_index].view(-1)[edge_indices] = 1.0
        elif graph_mode == "dense":
            fixed_graph = torch.ones(n_lags, n_variables, n_variables)
        elif graph_mode == "oracle" and oracle_graph is not None:
            fixed_graph = oracle_graph.detach().clone().float()
        else:
            fixed_graph = torch.zeros(n_lags, n_variables, n_variables)
        self.register_buffer("fixed_graph", fixed_graph)

        self.layers = nn.ModuleList(
            CausalSSMLayer(n_variables, state_dim, embedding_dim) for _ in range(n_layers)
        )
        self.lag_input_weights = nn.Parameter(
            torch.zeros(n_lags, n_variables, n_variables)
        )
        self.lag_forecast_weights = nn.Parameter(
            torch.zeros(horizon, n_lags, n_variables, n_variables)
        )
        self.residual_norms = nn.ModuleList(
            nn.LayerNorm(n_variables) for _ in range(n_layers - 1)
        )
        self.head = nn.Linear(state_dim, horizon)
        self.direct_head_mode = direct_head_mode
        if zero_init_ssm_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _lagged_adjacency(self, values: Tensor) -> Tensor:
        """Resolve mode into lag-specific (K,D,D) adjacency probabilities."""
        if self.graph_mode == "identity":
            identity = torch.eye(
                self.n_variables,
                device=values.device,
                dtype=values.dtype,
            )
            return identity.unsqueeze(0).expand(self.n_lags, -1, -1)
        if self.graph_mode == "learned":
            if self.graph_learner is None:
                raise RuntimeError("learned graph mode has no graph learner")
            adjacency = self.graph_learner.adjacency(use_ema=True)
            if self.graph_mask_threshold is not None:
                adjacency = (adjacency >= self.graph_mask_threshold).to(
                    dtype=adjacency.dtype
                )
            return adjacency
        return self.fixed_graph.to(device=values.device, dtype=values.dtype)

    def _mix_lagged_inputs(self, values: Tensor) -> Tensor:
        """Add graph-masked lag projections to values shaped (B,L,D)."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        adjacency = self._lagged_adjacency(values)
        mixed = values
        for lag_index in range(self.n_lags):
            lag = lag_index + 1
            if lag >= values.shape[1]:
                continue
            weights = self.lag_input_weights[lag_index] * adjacency[lag_index]
            lagged_values = torch.einsum(
                "ij,btj->bti",
                weights,
                values[:, :-lag],
            )
            mixed = mixed + functional.pad(lagged_values, (0, 0, lag, 0))
        return mixed

    def _direct_lag_forecast(self, values: Tensor) -> Tensor:
        """Project recent (B,L,D) values directly to graph-masked (B,H,D) forecasts."""
        adjacency = self._lagged_adjacency(values)
        if self.direct_head_mode == "iterated":
            return self._iterated_lag_forecast(values, adjacency)
        if self.direct_head_mode == "reachability":
            horizon_masks = companion_reachability(adjacency, self.horizon)
        else:
            horizon_masks = adjacency.unsqueeze(0).expand(self.horizon, -1, -1, -1)
        forecasts: list[Tensor] = []
        for horizon_index in range(self.horizon):
            forecast = values.new_zeros(values.shape[0], self.n_variables)
            for lag_index in range(self.n_lags):
                lag = lag_index + 1
                if lag > values.shape[1]:
                    continue
                weights = (
                    self.lag_forecast_weights[horizon_index, lag_index]
                    * horizon_masks[horizon_index, lag_index]
                )
                forecast = forecast + torch.einsum(
                    "ij,bj->bi",
                    weights,
                    values[:, -lag, :],
                )
            forecasts.append(forecast)
        return torch.stack(forecasts, dim=1)

    def _iterated_lag_forecast(self, values: Tensor, adjacency: Tensor) -> Tensor:
        """Roll the graph-masked horizon-1 lag readout forward recursively."""
        history = [values[:, -lag, :] for lag in range(self.n_lags, 0, -1) if lag <= values.shape[1]]
        forecasts: list[Tensor] = []
        for _ in range(self.horizon):
            forecast = values.new_zeros(values.shape[0], self.n_variables)
            for lag_index in range(min(self.n_lags, len(history))):
                weights = self.lag_forecast_weights[0, lag_index] * adjacency[lag_index]
                forecast = forecast + torch.einsum(
                    "ij,bj->bi", weights, history[-(lag_index + 1)]
                )
            forecasts.append(forecast)
            history.append(forecast)
        return torch.stack(forecasts, dim=1)

    def _adjacency(self, values: Tensor) -> Tensor:
        """Resolve graph mode into an aggregate adjacency with shape (D,D)."""
        if self.graph_mode == "identity":
            return torch.eye(self.n_variables, device=values.device, dtype=values.dtype)
        if self.graph_mode == "learned":
            if self.graph_learner is None:
                raise RuntimeError("learned graph mode has no graph learner")
            adjacency = self._lagged_adjacency(values)
            return 1.0 - torch.prod(1.0 - adjacency, dim=0)
        return 1.0 - torch.prod(1.0 - self.fixed_graph.to(values), dim=0)

    def forward(self, values: Tensor) -> Tensor:
        """Map an input window (B,L,D) to a forecast (B,H,D)."""
        prediction, _ = self.forward_with_components(values)
        return prediction

    def forward_with_components(
        self, values: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return forecast and raw-unit additive mean, SSM, and direct-lag components."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        normalized, stats = revin_normalize(values)
        adjacency = self._adjacency(normalized)
        layer_input = self._mix_lagged_inputs(normalized)
        final_hidden: Tensor | None = None
        for index, layer in enumerate(self.layers):
            hidden, projected = layer(
                layer_input,
                adjacency,
                coupling_enabled=self.use_ssm_coupling,
            )
            final_hidden = hidden
            if index < len(self.layers) - 1:
                layer_input = self.residual_norms[index](layer_input + projected)
        if final_hidden is None:
            raise RuntimeError("model has no SSM layers")
        last_state = final_hidden[:, -1, :, :]
        ssm_forecast = self.head(last_state).transpose(1, 2)
        if self.use_direct_lag_head:
            direct_forecast = self._direct_lag_forecast(normalized)
        else:
            direct_forecast = torch.zeros_like(ssm_forecast)
        mean = stats["mean"].expand(-1, self.horizon, -1)
        ssm_contribution = ssm_forecast * stats["std"]
        direct_contribution = direct_forecast * stats["std"]
        prediction = mean + ssm_contribution + direct_contribution
        return prediction, {
            "normalization_mean": mean,
            "ssm_contribution": ssm_contribution,
            "direct_lag_contribution": direct_contribution,
        }
