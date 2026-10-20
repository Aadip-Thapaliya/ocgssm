"""Acceptance tests for graph-conditioned forecasting models."""

import copy
import time

import pytest
import torch
from torch import Tensor

from ocg.data import generate_svar
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
    )
    oracle = CausalSSM(
        n_variables=3,
        state_dim=6,
        horizon=2,
        n_lags=1,
        graph_mode="oracle",
        oracle_graph=identity_graph,
    )
    identity_state = copy.deepcopy(identity.state_dict())
    identity_state.pop("fixed_graph")
    oracle.load_state_dict(identity_state, strict=False)
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


def test_lagged_input_mixing_respects_oracle_edge_and_lag() -> None:
    """Lag mixer applies only graph-permitted source values from the matching lag."""
    graph = torch.zeros(2, 2, 2)
    graph[0, 0, 1] = 1.0
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=2,
        graph_mode="oracle",
        oracle_graph=graph,
    )
    with torch.no_grad():
        model.lag_input_weights.fill_(1.0)
    values = torch.tensor([[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]])

    mixed = model._mix_lagged_inputs(values)

    expected = torch.tensor([[[2.0, 3.0], [8.0, 7.0], [18.0, 13.0]]])
    torch.testing.assert_close(mixed, expected)


def test_lagged_forecast_head_respects_oracle_edge_and_lag() -> None:
    """Direct horizon readout uses the matching graph mask and recent lag values."""
    graph = torch.zeros(2, 2, 2)
    graph[0, 0, 1] = 1.0
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=2,
        n_lags=2,
        graph_mode="oracle",
        oracle_graph=graph,
    )
    with torch.no_grad():
        model.lag_forecast_weights.fill_(1.0)
    values = torch.tensor([[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]])

    direct_forecast = model._direct_lag_forecast(values)

    expected = torch.tensor([[[13.0, 0.0], [13.0, 0.0]]])
    torch.testing.assert_close(direct_forecast, expected)


def test_dense_and_random_graph_masks_match_configured_density() -> None:
    """Dense (K,D,D) and random masks match their respective edge-count rules."""
    n_variables = 5
    n_lags = 2
    sparsity = 0.2
    dense = CausalSSM(
        n_variables=n_variables,
        state_dim=3,
        horizon=1,
        n_lags=n_lags,
        graph_mode="dense",
    )
    random = CausalSSM(
        n_variables=n_variables,
        state_dim=3,
        horizon=1,
        n_lags=n_lags,
        graph_mode="random",
        random_seed=17,
        random_sparsity=sparsity,
    )

    expected_dense = torch.ones(n_lags, n_variables, n_variables)
    expected_dense.diagonal(dim1=-2, dim2=-1).zero_()
    torch.testing.assert_close(dense.fixed_graph, expected_dense)
    assert torch.all((random.fixed_graph == 0) | (random.fixed_graph == 1))
    assert torch.all(random.fixed_graph.sum(dim=(1, 2)) == round(sparsity * n_variables**2))


def test_true_coefficients_never_enter_model_inputs_or_state() -> None:
    """The (T,K,D,D) coefficients never enter (B,L,D) calls or model state."""
    data = generate_svar(
        n_steps=48,
        n_variables=3,
        n_lags=2,
        switch_at=48,
        sparsity=0.3,
        seed=19,
    )
    coefficients = data["coefficients"]
    graph = data["graph"][0]
    values = data["series"][:8].unsqueeze(0)

    for graph_mode in ("identity", "dense", "random", "oracle", "learned"):
        model = CausalSSM(
            n_variables=3,
            state_dim=4,
            horizon=1,
            n_lags=2,
            graph_mode=graph_mode,
            oracle_graph=graph if graph_mode == "oracle" else None,
            random_sparsity=0.3,
        )
        seen_inputs: list[Tensor] = []

        def capture_inputs(module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            """Record direct tensor inputs for the model and all child modules."""
            del module
            seen_inputs.extend(item for item in inputs if isinstance(item, Tensor))

        handles = [
            module.register_forward_pre_hook(capture_inputs)
            for module in model.modules()
        ]
        model(values)
        for handle in handles:
            handle.remove()

        model_tensors = [
            *model.parameters(),
            *model.buffers(),
            *seen_inputs,
        ]
        assert all(tensor is not coefficients for tensor in model_tensors)
        assert all(tensor.data_ptr() != coefficients.data_ptr() for tensor in model_tensors)
        assert all(
            tensor.shape != coefficients[0].shape
            or not torch.equal(tensor, coefficients[0])
            for tensor in model_tensors
        )
        if graph_mode == "oracle":
            torch.testing.assert_close(model.fixed_graph, graph)
        elif graph_mode == "dense":
            assert torch.all((model.fixed_graph == 0) | (model.fixed_graph == 1))
        else:
            assert not torch.equal(model.fixed_graph, coefficients[0])


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


@pytest.mark.parametrize("graph_mode", ["identity", "dense", "random", "oracle", "learned"])
@pytest.mark.parametrize("n_variables", [5, 20])
@pytest.mark.parametrize("batch_size", [1, 32])
def test_fast_scan_matches_reference_forward_and_gradients(
    graph_mode: str,
    n_variables: int,
    batch_size: int,
) -> None:
    """Fast and reference scans match outputs and gradients for all graph modes."""
    torch.manual_seed(23)
    n_lags = 2
    state_dim = 6
    horizon = 3
    sequence_length = 24
    oracle_graph = None
    if graph_mode == "oracle":
        oracle_graph = (torch.rand(n_lags, n_variables, n_variables) > 0.7).float()
    reference = CausalSSM(
        n_variables=n_variables,
        state_dim=state_dim,
        horizon=horizon,
        n_lags=n_lags,
        n_layers=1,
        embedding_dim=4,
        graph_mode=graph_mode,
        graph_learner=GraphLearner(n_variables=n_variables, n_lags=n_lags)
        if graph_mode == "learned"
        else None,
        oracle_graph=oracle_graph,
        random_seed=17,
        random_sparsity=0.2,
        scan_implementation="reference",
    ).double()
    fast = CausalSSM(
        n_variables=n_variables,
        state_dim=state_dim,
        horizon=horizon,
        n_lags=n_lags,
        n_layers=1,
        embedding_dim=4,
        graph_mode=graph_mode,
        graph_learner=GraphLearner(n_variables=n_variables, n_lags=n_lags)
        if graph_mode == "learned"
        else None,
        oracle_graph=oracle_graph,
        random_seed=17,
        random_sparsity=0.2,
        scan_implementation="fast",
    ).double()
    fast.load_state_dict(copy.deepcopy(reference.state_dict()))
    reference.train()
    fast.train()
    input_reference = (
        0.1 * torch.randn(batch_size, sequence_length, n_variables, dtype=torch.float64)
    ).requires_grad_()
    input_fast = input_reference.detach().clone().requires_grad_(True)
    output_reference = reference(input_reference)
    output_fast = fast(input_fast)
    torch.testing.assert_close(output_fast, output_reference, rtol=1e-5, atol=1e-5)
    loss_reference = output_reference.square().mean()
    loss_fast = output_fast.square().mean()
    loss_reference.backward()
    loss_fast.backward()
    torch.testing.assert_close(input_fast.grad, input_reference.grad, rtol=1e-5, atol=1e-5)
    reference_grads = dict(reference.named_parameters())
    fast_grads = dict(fast.named_parameters())
    for name, parameter in reference_grads.items():
        if parameter.grad is None:
            continue
        if fast_grads[name].grad is None:
            raise AssertionError(f"Expected gradient for parameter {name}")
        torch.testing.assert_close(
            fast_grads[name].grad,
            parameter.grad,
            rtol=1e-5,
            atol=1e-5,
        )
