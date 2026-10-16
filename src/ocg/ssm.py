"""Pure-PyTorch causal state-space scan."""

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class CausalSSMLayer(nn.Module):
    """Scan values shaped (B,L,D) and produce hidden states shaped (B,L,D,N)."""

    def __init__(self, n_variables: int, state_dim: int, embedding_dim: int) -> None:
        """Create variable embeddings and state-space projections."""
        super().__init__()
        if n_variables < 1 or state_dim < 1 or embedding_dim < 1:
            raise ValueError("n_variables, state_dim, and embedding_dim must be positive")
        self.n_variables = n_variables
        self.state_dim = state_dim
        self.node_embeddings = nn.Parameter(torch.randn(n_variables, embedding_dim) * 0.1)
        self.graph_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Tanh(),
        )
        feature_dim = embedding_dim + 1
        self.dt_projection = nn.Linear(feature_dim, 1)
        self.input_projection = nn.Linear(feature_dim, state_dim)
        self.output_projection = nn.Linear(feature_dim, state_dim)
        self.gate_projection = nn.Linear(feature_dim, 1)
        self.coupling_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.logA = nn.Parameter(torch.zeros(state_dim))

    def forward(self, values: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """Return hidden (B,L,D,N) and projected (B,L,D) sequences."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        if values.shape[1] < 1 or adjacency.shape != (self.n_variables, self.n_variables):
            raise ValueError("sequence must be nonempty and adjacency must have shape (D,D)")
        adjacency = adjacency.to(device=values.device, dtype=values.dtype)
        neighbors = adjacency @ self.node_embeddings
        graph_features = self.graph_encoder(torch.cat((self.node_embeddings, neighbors), dim=-1))
        cross_adjacency = adjacency * (
            1.0 - torch.eye(self.n_variables, device=values.device, dtype=values.dtype)
        )
        state = values.new_zeros(values.shape[0], self.n_variables, self.state_dim)
        hidden_states: list[Tensor] = []
        projected_values: list[Tensor] = []
        decay_rate = torch.exp(self.logA).view(1, 1, self.state_dim)

        for time_index in range(values.shape[1]):
            current = values[:, time_index, :].unsqueeze(-1)
            node_features = graph_features.unsqueeze(0).expand(values.shape[0], -1, -1)
            features = torch.cat((current, node_features), dim=-1)
            dt = functional.softplus(self.dt_projection(features))
            decay = torch.exp(-dt * decay_rate)
            transformed_state = self.coupling_projection(state)
            coupling = torch.einsum("ij,bjn->bin", cross_adjacency, transformed_state)
            gate = torch.sigmoid(self.gate_projection(features))
            input_state = self.input_projection(features) * current
            state = decay * state + gate * coupling + dt * input_state
            output = torch.sum(self.output_projection(features) * state, dim=-1)
            hidden_states.append(state)
            projected_values.append(output)

        return torch.stack(hidden_states, dim=1), torch.stack(projected_values, dim=1)
