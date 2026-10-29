"""Acceptance tests for graph-conditioned forecasting models."""

import copy
import time

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
    """Forecast fitting works while graph parameters retain structural ownership."""
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
    assert learned.graph_learner.logits.grad is None
    assert learned.graph_learner.weights.grad is None
    structural_values = torch.randn(2, 10, 3)
    learned.graph_learner.structural_loss(structural_values).backward()
    gradient = learned.graph_learner.logits.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0


def test_causal_ssm_uses_reversible_per_window_normalization() -> None:
    """Affine rescaling of a window is inverted after normalized prediction."""
    torch.manual_seed(8)
    model = CausalSSM(
        n_variables=3,
        state_dim=4,
        horizon=2,
        n_lags=1,
        graph_mode="identity",
    ).eval()
    values = torch.randn(2, 12, 3)
    shifted = values * 3.0 + 7.0
    with torch.no_grad():
        original_forecast = model(values)
        shifted_forecast = model(shifted)
    torch.testing.assert_close(
        shifted_forecast, original_forecast * 3.0 + 7.0, rtol=1e-5, atol=1e-5
    )


def test_forecast_components_sum_to_the_model_output() -> None:
    """SSM and direct-lag branches add in raw units with no hidden mixing weight."""
    torch.manual_seed(12)
    model = CausalSSM(
        n_variables=3,
        state_dim=4,
        horizon=2,
        n_lags=2,
        graph_mode="identity",
    )
    values = torch.randn(2, 8, 3)

    prediction, components = model.forward_with_components(values)

    torch.testing.assert_close(
        prediction,
        components["normalization_mean"]
        + components["ssm_contribution"]
        + components["direct_lag_contribution"],
    )
    torch.testing.assert_close(prediction, model(values))
    assert torch.count_nonzero(components["direct_lag_contribution"]) == 0


def test_learned_hard_mask_thresholds_forecast_adjacency_only() -> None:
    """Hard masking binarizes forecast adjacency without changing EMA probabilities."""
    graph = GraphLearner(2, 1)
    probabilities = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]])
    with torch.no_grad():
        graph.ema_logits.copy_(torch.logit(probabilities))
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=graph,
        graph_mask_threshold=0.5,
    )
    soft = graph.adjacency().detach().clone()
    hard = model._lagged_adjacency(torch.zeros(1, 2, 2))
    aggregate = model._adjacency(torch.zeros(1, 2, 2))

    torch.testing.assert_close(graph.adjacency(), soft)
    torch.testing.assert_close(hard, (probabilities >= 0.5).float())
    torch.testing.assert_close(aggregate, hard[0])
    assert torch.all((hard == 0) | (hard == 1))


def test_direct_lag_head_ablation_has_no_direct_head_gradient() -> None:
    """Disabling the direct-lag head removes its forecast contribution."""
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="identity",
        use_direct_lag_head=False,
    )
    model(torch.randn(2, 5, 2)).square().mean().backward()
    assert model.lag_forecast_weights.grad is None


def test_ssm_coupling_ablation_is_wired_into_each_layer() -> None:
    """Disabling SSM coupling prevents cross-variable adjacency from reaching layers."""
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="dense",
        use_ssm_coupling=False,
    )
    layer = model.layers[0]
    with torch.no_grad():
        layer.coupling_projection.weight.fill_(1.0)
    values = torch.randn(2, 5, 2)
    no_coupling = model(values)
    model.use_ssm_coupling = True
    with_coupling = model(values)
    assert not torch.equal(no_coupling, with_coupling)


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
