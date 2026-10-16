"""Acceptance tests for graph-conditioned forecasting models."""

import copy
import time

import torch

from ocg.graph import GraphLearner
from ocg.model import CausalSSM


def test_forecast_shape_and_identity_matches_no_cross_coupling() -> None:
    """Identity adjacency has the same result as an oracle identity graph."""
    identity_graph = torch.eye(3).unsqueeze(0)
    identity = CausalSSM(
        n_variables=3,
        state_dim=6,
        horizon=2,
        n_lags=1,
        graph_mode="identity",
        oracle_graph=identity_graph,
    )
    oracle = CausalSSM(
        n_variables=3,
        state_dim=6,
        horizon=2,
        n_lags=1,
        graph_mode="oracle",
        oracle_graph=identity_graph,
    )
    oracle.load_state_dict(copy.deepcopy(identity.state_dict()))
    values = torch.randn(4, 24, 3)
    prediction = identity(values)
    assert prediction.shape == (4, 2, 3)
    torch.testing.assert_close(prediction, oracle(values), rtol=0, atol=0)


def test_tiny_overfit_and_learned_graph_gradient() -> None:
    """Forecast training reduces loss and learned graph logits receive gradients."""
    torch.manual_seed(7)
    inputs = torch.randn(64, 16, 3)
    targets = torch.zeros(64, 2, 3)
    model = CausalSSM(
        n_variables=3,
        state_dim=6,
        horizon=2,
        n_lags=1,
        graph_mode="learned",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final_loss = torch.nn.functional.mse_loss(model(inputs), targets)
    assert final_loss.item() < initial_loss.item()

    learned = CausalSSM(
        n_variables=3,
        state_dim=5,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=GraphLearner(3, 1),
    )
    learned(torch.randn(2, 10, 3)).square().mean().backward()
    assert learned.graph_learner is not None
    gradient = learned.graph_learner.logits.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0


def test_scan_runtime_scales_with_sequence_length() -> None:
    """Four times the sequence length remains within a linear-time tolerance."""
    torch.set_num_threads(1)
    model = CausalSSM(n_variables=3, state_dim=4, horizon=1, n_lags=1).eval()
    short = torch.randn(1, 64, 3)
    long = torch.randn(1, 256, 3)

    with torch.no_grad():
        model(short)
        model(long)
        started = time.perf_counter()
        for _ in range(3):
            model(short)
        short_seconds = (time.perf_counter() - started) / 3
        started = time.perf_counter()
        for _ in range(3):
            model(long)
        long_seconds = (time.perf_counter() - started) / 3

    assert long_seconds < short_seconds * 10
