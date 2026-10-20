"""Tests for the stability diagnostics script."""

from pathlib import Path
from typing import Any

import torch
import yaml

from ocg.model import CausalSSM
from scripts.diagnose_stability import (
    PARAMETER_GROUPS,
    effective_transition_matrices,
    grad_norms_by_group,
    run_diagnostics,
    transition_spectral_radius,
)


def _tiny_config() -> dict[str, Any]:
    """Return a tiny synthetic config with identity and oracle variants."""
    return {
        "seed": 3,
        "data": {
            "type": "synthetic",
            "n_steps": 120,
            "n_variables": 3,
            "n_lags": 2,
            "switch_at": 120,
            "blend_steps": 0,
            "sparsity": 0.4,
            "noise_scale": 0.1,
            "input_length": 6,
            "horizon": 2,
            "train_fraction": 0.5,
            "validation_fraction": 0.2,
            "seed": 3,
        },
        "model": {
            "name": "ocg",
            "state_dim": 4,
            "n_layers": 1,
            "embedding_dim": 2,
            "variants": [
                {"id": "identity", "name": "ocg", "graph_mode": "identity"},
                {"id": "oracle", "name": "ocg", "graph_mode": "oracle"},
            ],
        },
        "training": {"mode": "offline", "epochs": 2, "batch_size": 8, "learning_rate": 0.01},
    }


def _brute_force_transition(model: CausalSSM, values: torch.Tensor, time_index: int) -> torch.Tensor:
    """Apply the layer's linear recurrence to basis states at one time step; returns (D*N,D*N)."""
    layer = model.layers[0]
    adjacency = model._adjacency(values)
    layer_input = model._mix_lagged_inputs(values)
    graph_features, cross_adjacency = layer._graph_features(adjacency)
    features = torch.cat(
        (layer_input[:1, time_index].unsqueeze(-1), graph_features.unsqueeze(0)), dim=-1
    )
    dt = torch.nn.functional.softplus(layer.dt_projection(features))
    decay = torch.exp(-dt * torch.exp(layer.logA).view(1, 1, -1))
    gate = torch.sigmoid(layer.gate_projection(features))
    n_variables, state_dim = layer.n_variables, layer.state_dim
    columns = []
    for basis_index in range(n_variables * state_dim):
        state = torch.zeros(1, n_variables, state_dim)
        state.view(-1)[basis_index] = 1.0
        coupling = cross_adjacency @ layer.coupling_projection(state)
        next_state = decay * state + gate * coupling
        columns.append(next_state.reshape(-1))
    return torch.stack(columns, dim=1)


def test_effective_transition_matches_recurrence_and_identity_radius() -> None:
    """Transition matrices reproduce the recurrence and identity radius equals max decay."""
    torch.manual_seed(5)
    oracle_graph = (torch.rand(2, 4, 4) > 0.5).float()
    model = CausalSSM(
        n_variables=4, state_dim=3, horizon=2, n_lags=2, embedding_dim=2,
        graph_mode="oracle", oracle_graph=oracle_graph,
    )
    with torch.no_grad():
        model.lag_input_weights.normal_()
    values = torch.randn(2, 7, 4)
    with torch.no_grad():
        matrices = effective_transition_matrices(model, values)
    assert matrices.shape == (2, 7, 12, 12)
    expected = _brute_force_transition(model, values, time_index=3)
    torch.testing.assert_close(matrices[0, 3], expected, rtol=1e-5, atol=1e-6)

    identity = CausalSSM(n_variables=4, state_dim=3, horizon=2, n_lags=2, graph_mode="identity")
    with torch.no_grad():
        matrices = effective_transition_matrices(identity, values)
        decay_max = torch.diagonal(matrices, dim1=-2, dim2=-1).max().item()
    radius = transition_spectral_radius(identity, values)
    assert abs(radius["max"] - decay_max) < 1e-6
    assert radius["max"] < 1.0


def test_grad_norm_groups_and_diagnostics_report(tmp_path: Path) -> None:
    """Gradient norms are grouped by parameter and the diagnostics report has per-epoch records."""
    model = CausalSSM(n_variables=3, state_dim=4, horizon=2, n_lags=2, graph_mode="dense")
    model(torch.randn(2, 6, 3)).square().mean().backward()
    norms = grad_norms_by_group(model)
    assert set(norms) == set(PARAMETER_GROUPS)
    assert norms["lag_forecast_weights"] > 0.0
    assert norms["coupling_projection"] > 0.0
    assert norms["graph_learner"] == 0.0

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_config()), encoding="utf-8")
    report = run_diagnostics(config_path, ["oracle"], probe_size=4)
    assert list(report["variants"]) == ["oracle"]
    records = report["variants"]["oracle"]["epochs"]
    assert [record["epoch"] for record in records] == [0, 1, 2]
    assert "train_loss" not in records[0]
    for record in records[1:]:
        assert record["train_loss"] > 0.0
        assert set(record["grad_norm_max"]) == set(PARAMETER_GROUPS)
        assert record["spectral_radius"]["max"] >= record["spectral_radius"]["mean"]
        assert record["hidden_max_abs"] >= 0.0
        assert record["lag_input_max_row_sum"] >= 0.0
        assert record["lag_forecast_max_row_sum"] >= 0.0
