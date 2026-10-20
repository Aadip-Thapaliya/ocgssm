"""Forecasting model built from graph-conditioned causal SSM layers."""

import torch
from torch import Tensor, nn
from torch.nn import functional

from ocg.graph import GraphLearner
from ocg.ssm import CausalSSMLayer


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
        scan_implementation: str = "fast",
    ) -> None:
        """Initialize the SSM stack, graph mode, and per-variable forecast head."""
        super().__init__()
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
        if scan_implementation not in {"fast", "reference"}:
            raise ValueError("scan_implementation must be fast or reference")
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
        self.scan_implementation = scan_implementation
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
            fixed_graph.diagonal(dim1=-2, dim2=-1).zero_()
        elif graph_mode == "oracle" and oracle_graph is not None:
            fixed_graph = oracle_graph.detach().clone().float()
        else:
            fixed_graph = torch.zeros(n_lags, n_variables, n_variables)
        self.register_buffer("fixed_graph", fixed_graph)

        self.layers = nn.ModuleList(
            CausalSSMLayer(n_variables, state_dim, embedding_dim) for _ in range(n_layers)
        )
        for layer in self.layers:
            layer.use_reference_forward = scan_implementation == "reference"
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
            current = self.graph_learner.adjacency(use_ema=False)
            ema = self.graph_learner.adjacency(use_ema=True)
            return ema + current - current.detach()
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
        forecasts: list[Tensor] = []
        for horizon_index in range(self.horizon):
            forecast = values.new_zeros(values.shape[0], self.n_variables)
            for lag_index in range(self.n_lags):
                lag = lag_index + 1
                if lag > values.shape[1]:
                    continue
                weights = (
                    self.lag_forecast_weights[horizon_index, lag_index]
                    * adjacency[lag_index]
                )
                forecast = forecast + torch.einsum(
                    "ij,bj->bi",
                    weights,
                    values[:, -lag, :],
                )
            forecasts.append(forecast)
        return torch.stack(forecasts, dim=1)

    def _adjacency(self, values: Tensor) -> Tensor:
        """Resolve graph mode into an aggregate adjacency with shape (D,D)."""
        if self.graph_mode == "identity":
            return torch.eye(self.n_variables, device=values.device, dtype=values.dtype)
        if self.graph_mode == "learned":
            if self.graph_learner is None:
                raise RuntimeError("learned graph mode has no graph learner")
            current = self.graph_learner.aggregate_adjacency(use_ema=False)
            ema = self.graph_learner.aggregate_adjacency(use_ema=True)
            return ema + current - current.detach()
        return 1.0 - torch.prod(1.0 - self.fixed_graph.to(values), dim=0)

    def forward(self, values: Tensor) -> Tensor:
        """Map an input window (B,L,D) to a forecast (B,H,D)."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        adjacency = self._adjacency(values)
        layer_input = self._mix_lagged_inputs(values)
        if len(self.layers) == 1:
            last_state = self.layers[0].forward_last(layer_input, adjacency)
            return self.head(last_state).transpose(1, 2) + self._direct_lag_forecast(values)
        for index, layer in enumerate(self.layers[:-1]):
            _, projected = layer(layer_input, adjacency)
            layer_input = self.residual_norms[index](layer_input + projected)
        if len(self.layers) < 1:
            raise RuntimeError("model has no SSM layers")
        last_state = self.layers[-1].forward_last(layer_input, adjacency)
        return self.head(last_state).transpose(1, 2) + self._direct_lag_forecast(values)
