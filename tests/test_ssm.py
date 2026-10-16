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
