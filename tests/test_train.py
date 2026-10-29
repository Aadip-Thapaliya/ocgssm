"""Tests for offline and prequential forecasting training."""

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ocg.graph import GraphLearner
from ocg.model import CausalSSM
from ocg.train import train_offline, train_online


class _ScalarForecaster(nn.Module):
    """Minimal model that broadcasts one trainable scalar as (B,H,D)."""

    def __init__(self) -> None:
        """Initialize the scalar forecast parameter."""
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        """Broadcast a scalar prediction for inputs shaped (B,L,D)."""
        return self.value.expand(inputs.shape[0], 1, inputs.shape[2])


class _HorizonForecaster(nn.Module):
    """Broadcast one parameter across all configured forecast horizons."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))
        self.horizon = horizon

    def forward(self, inputs: Tensor) -> Tensor:
        return self.value.expand(inputs.shape[0], self.horizon, inputs.shape[2])


class _NonfiniteForecaster(nn.Module):
    """Return a non-finite forecast to verify that online training stops early."""

    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.value.expand(inputs.shape[0], 1, inputs.shape[2]) + float("inf")


class _GraphForecaster(nn.Module):
    """Expose the fixed graph mean as a scalar forecast for schedule tests."""

    def __init__(self) -> None:
        """Initialize a fixed graph buffer and trainable forecast offset."""
        super().__init__()
        self.register_buffer("fixed_graph", torch.zeros(1, 2, 2))
        self.graph_mode = "oracle"
        self.offset = nn.Parameter(torch.zeros(()))
        self.seen_graphs: list[Tensor] = []

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a graph-dependent forecast for inputs shaped (B,L,D)."""
        self.seen_graphs.append(self.fixed_graph.detach().clone())
        value = self.fixed_graph.mean() + self.offset
        return value.expand(inputs.shape[0], 1, inputs.shape[2])


def test_offline_training_logs_validation(tmp_path: Path) -> None:
    """Offline fitting returns losses and writes the same record as JSON."""
    inputs = torch.randn(8, 4, 2)
    targets = torch.ones(8, 1, 2)
    result_path = tmp_path / "nested" / "offline.json"
    result = train_offline(
        _ScalarForecaster(),
        inputs,
        targets,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=17,
        validation_inputs=inputs,
        validation_targets=targets,
        output_path=result_path,
    )

    assert result["mode"] == "offline"
    assert result["seed"] == 17
    assert len(result["train_mse"]) == 2
    assert len(result["validation_mse"]) == 2
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_offline_training_restores_best_validation_checkpoint() -> None:
    """Optional offline checkpoint selection restores the lowest validation loss."""
    model = _ScalarForecaster()
    inputs = torch.zeros(4, 2, 1)

    result = train_offline(
        model,
        inputs,
        torch.ones(4, 1, 1),
        epochs=3,
        batch_size=4,
        learning_rate=0.1,
        validation_inputs=inputs,
        validation_targets=torch.zeros(4, 1, 1),
        restore_best_validation=True,
    )

    assert result["best_validation_epoch"] == 1
    assert result["best_validation_mse"] == result["validation_mse"][0]
    assert result["validation_mse"][-1] > result["best_validation_mse"]
    assert model.value.item() == pytest.approx(0.1)


def test_offline_early_stopping_restores_best_checkpoint_and_logs_epochs() -> None:
    model = _ScalarForecaster()
    inputs = torch.zeros(4, 2, 1)

    result = train_offline(
        model,
        inputs,
        torch.ones(4, 1, 1),
        epochs=20,
        batch_size=4,
        learning_rate=0.1,
        validation_inputs=inputs,
        validation_targets=torch.zeros(4, 1, 1),
        restore_best_validation=True,
        early_stopping_patience=2,
    )

    assert result["epochs_executed"] == 3
    assert result["best_validation_epoch"] == 1
    assert result["stopped_early"] is True
    assert result["reached_epoch_cap"] is False
    assert model.value.item() == pytest.approx(0.1)


def test_offline_early_stopping_marks_epoch_cap_as_not_converged() -> None:
    model = _ScalarForecaster()
    inputs = torch.zeros(4, 2, 1)

    result = train_offline(
        model,
        inputs,
        torch.zeros(4, 1, 1),
        epochs=3,
        batch_size=4,
        learning_rate=0.1,
        validation_inputs=inputs,
        validation_targets=torch.zeros(4, 1, 1),
        restore_best_validation=True,
        early_stopping_patience=5,
    )

    assert result["epochs_executed"] == 3
    assert result["best_validation_epoch"] == 1
    assert result["stopped_early"] is False
    assert result["reached_epoch_cap"] is True


def test_offline_early_stopping_requires_validation_checkpointing() -> None:
    inputs = torch.zeros(4, 2, 1)

    with pytest.raises(ValueError, match="requires best-validation restoration"):
        train_offline(
            _ScalarForecaster(),
            inputs,
            torch.ones(4, 1, 1),
            epochs=3,
            batch_size=4,
            learning_rate=0.1,
            validation_inputs=inputs,
            validation_targets=torch.zeros(4, 1, 1),
            early_stopping_patience=2,
        )


def test_offline_training_updates_structural_graph_and_ema() -> None:
    """Offline fitting trains VAR graph parameters and the evaluated EMA graph."""
    torch.manual_seed(19)
    graph = GraphLearner(n_variables=3, n_lags=1, sparsity_weight=0.0)
    model = CausalSSM(
        n_variables=3,
        state_dim=4,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=graph,
    )
    initial_logits = graph.logits.detach().clone()
    initial_weights = graph.weights.detach().clone()
    initial_adjacency = graph.adjacency().detach().clone()
    inputs = torch.randn(12, 8, 3)
    targets = torch.randn(12, 1, 3)

    train_offline(
        model,
        inputs,
        targets,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
    )

    assert not torch.equal(graph.logits, initial_logits)
    assert not torch.equal(graph.weights, initial_weights)
    assert not torch.equal(graph.adjacency(), initial_adjacency)


def test_offline_best_checkpoint_restores_graph_optimizer_state() -> None:
    """Selected learned-graph weights and their Adam state come from one epoch."""
    graph = GraphLearner(2, 1, sparsity_weight=0.0)
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=graph,
    )
    inputs = torch.randn(4, 5, 2)

    result = train_offline(
        model,
        inputs,
        torch.zeros(4, 1, 2),
        epochs=3,
        batch_size=4,
        learning_rate=0.1,
        validation_inputs=inputs,
        validation_targets=torch.full((4, 1, 2), 100.0),
        restore_best_validation=True,
    )

    assert 1 <= result["best_validation_epoch"] <= 3
    step = graph._optimizer.state[graph.logits]["step"]
    assert int(step.item()) == result["best_validation_epoch"]


def test_online_training_predicts_before_update() -> None:
    """Online forecasts are recorded before each target updates the model."""
    inputs = torch.zeros(3, 2, 1)
    targets = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1)
    result = train_online(_ScalarForecaster(), inputs, targets, learning_rate=0.1)

    predictions = torch.tensor(result["predictions"])
    assert predictions.shape == (3, 1, 1)
    assert predictions[0].item() == 0.0
    assert predictions[2].item() > predictions[0].item()
    assert result["prequential_mse"][0] == 1.0
    assert len(result["running_mse"]) == 3


def test_online_training_delays_labels_until_they_are_observed() -> None:
    """A late horizon label cannot affect forecasts made before its timestamp."""
    inputs = torch.zeros(6, 2, 1)
    targets = torch.zeros(6, 3, 1)
    changed_targets = targets.clone()
    changed_targets[0, 2, 0] = 1.0

    ordinary = train_online(
        _HorizonForecaster(3), inputs, targets, learning_rate=0.1
    )
    perturbed = train_online(
        _HorizonForecaster(3), inputs, changed_targets, learning_rate=0.1
    )

    ordinary_predictions = torch.tensor(ordinary["predictions"])
    perturbed_predictions = torch.tensor(perturbed["predictions"])
    torch.testing.assert_close(ordinary_predictions[:4], perturbed_predictions[:4])
    assert not torch.equal(ordinary_predictions[4], perturbed_predictions[4])


def test_online_training_stops_and_reports_nonfinite_forecasts() -> None:
    """Non-finite model outputs fail immediately instead of yielding bad metrics."""
    with pytest.raises(FloatingPointError, match="online forecast at step 0"):
        train_online(
            _NonfiniteForecaster(),
            torch.zeros(2, 2, 1),
            torch.zeros(2, 1, 1),
            learning_rate=0.1,
        )


def test_online_graph_history_records_forecast_time_adjacency() -> None:
    """Graph history records the pre-update EMA and its aligned timestamps."""
    torch.manual_seed(7)
    graph = GraphLearner(n_variables=2, n_lags=1, sparsity_weight=0.0)
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=graph,
    )
    initial_adjacency = graph.adjacency().detach().clone()
    result = train_online(
        model,
        torch.randn(4, 5, 2),
        torch.randn(4, 1, 2),
        learning_rate=0.001,
        timestamps=[40, 41, 42, 43],
    )

    history = torch.tensor(result["graph_history"])
    assert torch.equal(history[0], initial_adjacency)
    assert result["graph_timestamps"] == [40, 41, 42, 43]
    assert result["detections"] == [False] * 4
    assert result["detection_events"] == []


def test_online_graph_update_ablation_keeps_graph_fixed() -> None:
    """The no-online-graph-updates ablation leaves its initial EMA unchanged."""
    torch.manual_seed(31)
    graph = GraphLearner(n_variables=2, n_lags=1, sparsity_weight=0.0)
    model = CausalSSM(
        n_variables=2,
        state_dim=3,
        horizon=1,
        n_lags=1,
        graph_mode="learned",
        graph_learner=graph,
    )
    initial = graph.adjacency().detach().clone()
    result = train_online(
        model,
        torch.randn(4, 5, 2),
        torch.randn(4, 1, 2),
        learning_rate=0.001,
        update_graph=False,
    )

    assert torch.equal(graph.adjacency(), initial)
    assert all(torch.equal(torch.tensor(snapshot), initial) for snapshot in result["graph_history"])
    assert result["detections"] == [False] * 4


def test_online_oracle_graph_changes_before_each_forecast() -> None:
    """Prequential oracle forecasts use each window's (K,D,D) graph."""
    inputs = torch.zeros(3, 2, 2)
    oracle_graphs = torch.stack(
        [
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, 2),
            torch.full((1, 2, 2), 0.5),
        ]
    )
    targets = oracle_graphs.mean(dim=(1, 2, 3)).reshape(3, 1, 1).expand(-1, 1, 2)

    model = _GraphForecaster()
    result = train_online(
        model,
        inputs,
        targets,
        learning_rate=0.001,
        oracle_graphs=oracle_graphs,
    )

    predictions = torch.tensor(result["predictions"])
    assert predictions[0].mean().item() == 0.0
    assert predictions[1].mean().item() == 1.0
    assert predictions[2].mean().item() == 0.5
    expected_forward_graphs = [
        oracle_graphs[0],
        oracle_graphs[1],
        oracle_graphs[0],
        oracle_graphs[2],
        oracle_graphs[1],
    ]
    assert all(
        torch.equal(seen, expected)
        for seen, expected in zip(
            model.seen_graphs, expected_forward_graphs, strict=True
        )
    )


def test_offline_oracle_batches_are_isolated_by_regime() -> None:
    """Offline oracle batches never mix labels and use the group's graph."""
    inputs = torch.zeros(4, 2, 2)
    targets = torch.zeros(4, 1, 2)
    oracle_graphs = torch.stack(
        [
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, 2),
        ]
    )
    regime_ids = torch.tensor([0, 1, 0, 1])
    model = _GraphForecaster()

    train_offline(
        model,
        inputs,
        targets,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        oracle_graphs=oracle_graphs,
        regime_ids=regime_ids,
    )

    assert len(model.seen_graphs) == 2
    assert torch.equal(model.seen_graphs[0], torch.zeros(1, 2, 2))
    assert torch.equal(model.seen_graphs[1], torch.ones(1, 2, 2))
