"""Acceptance tests for the causal state-space scan."""

import torch

from ocg.data import generate_svar, make_windows
from ocg.graph import GraphLearner
from ocg.model import CausalSSM
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


def test_learned_coupling_does_not_amplify_the_seed_303_startup_window() -> None:
    """Regression for the finite recurrent-state explosion in the abrupt campaign."""
    series = generate_svar(
        n_steps=512,
        n_variables=5,
        n_lags=2,
        switch_at=320,
        sparsity=0.2,
        noise_scale=0.1,
        seed=23,
    )["series"]
    inputs, _ = make_windows(series, input_length=24, horizon=6)
    first_test_window = inputs[281:282]

    torch.manual_seed(303)
    learner = GraphLearner(n_variables=5, n_lags=2)
    model = CausalSSM(
        n_variables=5,
        state_dim=8,
        horizon=6,
        n_lags=2,
        n_layers=1,
        embedding_dim=4,
        graph_mode="learned",
        graph_learner=learner,
    ).eval()

    with torch.no_grad():
        forecast = model(first_test_window)

    assert torch.isfinite(forecast).all()
    assert forecast.abs().max().item() < 10.0
