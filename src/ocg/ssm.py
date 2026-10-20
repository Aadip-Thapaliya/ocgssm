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
        self.use_reference_forward = False
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

    def _graph_features(self, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """Return graph features (D,E) and cross-variable mask (D,D) from adjacency."""
        neighbors = adjacency @ self.node_embeddings
        graph_features = self.graph_encoder(torch.cat((self.node_embeddings, neighbors), dim=-1))
        cross_adjacency = adjacency * (
            1.0 - torch.eye(self.n_variables, device=adjacency.device, dtype=adjacency.dtype)
        )
        return graph_features, cross_adjacency

    def _fast_scan(
        self,
        values: Tensor,
        cross_adjacency: Tensor,
        decay: Tensor,
        gate: Tensor,
        input_state: Tensor,
        return_hidden_sequence: bool,
    ) -> tuple[Tensor | None, Tensor]:
        """Run recurrence on (B,L,D,*) tensors and return hidden sequence and last state."""
        batch_size, sequence_length, _, _ = input_state.shape
        state = values.new_zeros(batch_size, self.n_variables, self.state_dim)
        hidden = (
            values.new_empty(batch_size, sequence_length, self.n_variables, self.state_dim)
            if return_hidden_sequence
            else None
        )
        for time_index in range(sequence_length):
            transformed_state = self.coupling_projection(state)
            coupling = torch.matmul(cross_adjacency, transformed_state)
            state = (
                decay[:, time_index] * state
                + gate[:, time_index] * coupling
                + input_state[:, time_index]
            )
            if hidden is not None:
                hidden[:, time_index] = state
        return hidden, state

    def _fast_forward(
        self,
        values: Tensor,
        adjacency: Tensor,
        return_hidden_sequence: bool = True,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Run vectorized feature projections and recurrent scan for values (B,L,D)."""
        graph_features, cross_adjacency = self._graph_features(adjacency)
        node_features = graph_features.unsqueeze(0).unsqueeze(0).expand(
            values.shape[0], values.shape[1], -1, -1
        )
        current = values.unsqueeze(-1)
        features = torch.cat((current, node_features), dim=-1)
        dt = functional.softplus(self.dt_projection(features))
        decay_rate = torch.exp(self.logA).view(1, 1, 1, self.state_dim)
        decay = torch.exp(-dt * decay_rate)
        gate = torch.sigmoid(self.gate_projection(features))
        input_state = dt * self.input_projection(features) * current
        hidden, last_state = self._fast_scan(
            values,
            cross_adjacency,
            decay,
            gate,
            input_state,
            return_hidden_sequence=return_hidden_sequence,
        )
        if hidden is None:
            projected = values.new_zeros(values.shape[0], values.shape[1], self.n_variables)
        else:
            output_weights = self.output_projection(features)
            projected = torch.einsum("bldn,bldn->bld", output_weights, hidden)
        return hidden, projected, last_state

    def _reference_forward(self, values: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """Reference scan used for parity tests; returns hidden (B,L,D,N) and projected (B,L,D)."""
        graph_features, cross_adjacency = self._graph_features(adjacency)
        state = values.new_zeros(values.shape[0], self.n_variables, self.state_dim)
        hidden_states: list[Tensor] = []
        projected_values: list[Tensor] = []
        decay_rate = torch.exp(self.logA).view(1, 1, self.state_dim)
        node_features = graph_features.unsqueeze(0).expand(values.shape[0], -1, -1)

        for time_index in range(values.shape[1]):
            current = values[:, time_index, :].unsqueeze(-1)
            features = torch.cat((current, node_features), dim=-1)
            dt = functional.softplus(self.dt_projection(features))
            decay = torch.exp(-dt * decay_rate)
            transformed_state = self.coupling_projection(state)
            coupling = torch.matmul(cross_adjacency, transformed_state)
            gate = torch.sigmoid(self.gate_projection(features))
            input_state = self.input_projection(features) * current
            state = decay * state + gate * coupling + dt * input_state
            output = torch.sum(self.output_projection(features) * state, dim=-1)
            hidden_states.append(state)
            projected_values.append(output)
        return torch.stack(hidden_states, dim=1), torch.stack(projected_values, dim=1)

    def forward(self, values: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """Return hidden (B,L,D,N) and projected (B,L,D) sequences."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        if values.shape[1] < 1 or adjacency.shape != (self.n_variables, self.n_variables):
            raise ValueError("sequence must be nonempty and adjacency must have shape (D,D)")
        adjacency = adjacency.to(device=values.device, dtype=values.dtype)
        if self.use_reference_forward:
            return self._reference_forward(values, adjacency)
        hidden, projected, _ = self._fast_forward(
            values,
            adjacency,
            return_hidden_sequence=True,
        )
        if hidden is None:
            raise RuntimeError("hidden sequence is required for forward")
        return hidden, projected

    def forward_last(self, values: Tensor, adjacency: Tensor) -> Tensor:
        """Return only the final hidden state with shape (B,D,N) from values (B,L,D)."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        if values.shape[1] < 1 or adjacency.shape != (self.n_variables, self.n_variables):
            raise ValueError("sequence must be nonempty and adjacency must have shape (D,D)")
        adjacency = adjacency.to(device=values.device, dtype=values.dtype)
        if self.use_reference_forward:
            hidden, _ = self._reference_forward(values, adjacency)
            return hidden[:, -1]
        _, _, last_state = self._fast_forward(
            values,
            adjacency,
            return_hidden_sequence=False,
        )
        return last_state
