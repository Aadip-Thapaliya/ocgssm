"""Forecasting model built from graph-conditioned causal SSM layers."""

import torch
from torch import Tensor, nn

from ocg.graph import GraphLearner
from ocg.ssm import CausalSSMLayer


class CausalSSM(nn.Module):
    """Forecast (B,L,D) input windows as (B,H,D) predictions."""

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
    ) -> None:
        """Initialize the SSM stack, graph mode, and per-variable forecast head."""
        super().__init__()
        if n_variables < 1 or state_dim < 1 or horizon < 1 or n_lags < 1 or n_layers < 1:
            raise ValueError("model dimensions must be positive")
        if graph_mode not in {"identity", "learned", "oracle", "random"}:
            raise ValueError("graph_mode must be identity, learned, oracle, or random")
        if graph_mode != "learned" and graph_learner is not None:
            raise ValueError("graph_learner can only be passed in learned mode")
        if graph_mode == "oracle" and oracle_graph is None:
            raise ValueError("oracle_graph is required in oracle mode")
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
        if graph_mode == "learned":
            self.graph_learner = graph_learner or GraphLearner(n_variables, n_lags)
        else:
            self.graph_learner = None

        if graph_mode == "random":
            generator = torch.Generator().manual_seed(random_seed)
            fixed_graph = torch.rand(n_lags, n_variables, n_variables, generator=generator)
        elif oracle_graph is not None:
            fixed_graph = oracle_graph.detach().clone().float()
        else:
            fixed_graph = torch.zeros(n_lags, n_variables, n_variables)
        self.register_buffer("fixed_graph", fixed_graph)

        self.layers = nn.ModuleList(
            CausalSSMLayer(n_variables, state_dim, embedding_dim) for _ in range(n_layers)
        )
        self.residual_norms = nn.ModuleList(
            nn.LayerNorm(n_variables) for _ in range(n_layers - 1)
        )
        self.head = nn.Linear(state_dim, horizon)

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
        layer_input = values
        final_hidden: Tensor | None = None
        for index, layer in enumerate(self.layers):
            hidden, projected = layer(layer_input, adjacency)
            final_hidden = hidden
            if index < len(self.layers) - 1:
                layer_input = self.residual_norms[index](layer_input + projected)
        if final_hidden is None:
            raise RuntimeError("model has no SSM layers")
        last_state = final_hidden[:, -1, :, :]
        return self.head(last_state).transpose(1, 2)
