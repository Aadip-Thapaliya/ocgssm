"""Acceptance tests for the causal state-space scan."""

import torch

from ocg.ssm import CausalSSMLayer


def test_layer_returns_hidden_and_projected_sequences() -> None:
    """Layer maps (B,L,D) to hidden (B,L,D,N) and projected (B,L,D)."""
    layer = CausalSSMLayer(n_variables=3, state_dim=4, embedding_dim=5)
    values = torch.randn(2, 12, 3)
    adjacency = torch.eye(3)
    hidden, projected = layer(values, adjacency)
    assert hidden.shape == (2, 12, 3, 4)
    assert projected.shape == (2, 12, 3)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(projected).all()


def test_cross_adjacency_rows_are_normalized_to_at_most_one() -> None:
    """Rows of the cross-variable adjacency are divided by max(1, row sum) with no diagonal."""
    layer = CausalSSMLayer(n_variables=4, state_dim=2, embedding_dim=2)
    adjacency = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _, cross = layer._graph_features(adjacency)
    assert torch.all(torch.diagonal(cross) == 0)
    torch.testing.assert_close(cross[0], torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3]))
    torch.testing.assert_close(cross[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(cross[2], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert torch.all(cross.sum(dim=-1) <= 1.0 + 1e-6)


def test_effective_coupling_weight_row_sums_are_bounded() -> None:
    """The coupling weight used in the scan has max absolute row sum at most the configured bound."""
    layer = CausalSSMLayer(n_variables=3, state_dim=5, embedding_dim=2, coupling_norm_bound=0.7)
    with torch.no_grad():
        layer.coupling_projection.weight.mul_(10.0)
    effective = layer._effective_coupling_weight()
    assert abs(effective.abs().sum(dim=-1).max().item() - 0.7) < 1e-5
    with torch.no_grad():
        layer.coupling_projection.weight.mul_(0.01)
    effective = layer._effective_coupling_weight()
    torch.testing.assert_close(effective, layer.coupling_projection.weight)
    loss = layer._effective_coupling_weight().square().sum()
    loss.backward()
    assert layer.coupling_projection.weight.grad is not None


def test_dense_d20_transition_is_contractive_and_bounded_state() -> None:
    """With a dense D=20 graph the one-step transition has spectral radius <= 1 and bounded states."""
    torch.manual_seed(0)
    n_variables, state_dim = 20, 8
    layer = CausalSSMLayer(n_variables=n_variables, state_dim=state_dim, embedding_dim=4)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(torch.randn_like(parameter) * 2.0)
    adjacency = torch.ones(n_variables, n_variables)
    values = torch.randn(4, 48, n_variables)
    graph_features, cross = layer._graph_features(adjacency)
    features = torch.cat(
        (values.unsqueeze(-1), graph_features.expand(4, 48, -1, -1)), dim=-1
    )
    dt = torch.nn.functional.softplus(layer.dt_projection(features))
    decay = torch.exp(-dt * torch.exp(layer.logA).view(1, 1, 1, -1))
    gate = torch.sigmoid(layer.gate_projection(features))
    weight = layer._effective_coupling_weight()
    transition = torch.diag_embed(decay.reshape(4, 48, -1)) + torch.einsum(
        "blij,mn->blimjn", (1.0 - decay[..., :1]) * gate * cross, weight
    ).reshape(4, 48, n_variables * state_dim, n_variables * state_dim)
    radius = torch.linalg.eigvals(transition.to(torch.float64)).abs().amax(dim=-1)
    assert radius.max().item() <= 1.0 + 1e-6
    hidden, _ = layer(values, adjacency)
    assert torch.isfinite(hidden).all()
    assert hidden.abs().max().item() < 1e3
