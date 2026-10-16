"""Tests for offline and prequential forecasting training."""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

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


def test_online_training_predicts_before_update() -> None:
    """Online forecasts are recorded before each target updates the model."""
    inputs = torch.zeros(3, 2, 1)
    targets = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1)
    result = train_online(_ScalarForecaster(), inputs, targets, learning_rate=0.1)

    predictions = torch.tensor(result["predictions"])
    assert predictions.shape == (3, 1, 1)
    assert predictions[0].item() == 0.0
    assert predictions[1].item() > predictions[0].item()
    assert result["prequential_mse"][0] == 1.0
    assert len(result["running_mse"]) == 3


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
    assert all(
        torch.equal(seen, expected)
        for seen, expected in zip(model.seen_graphs, oracle_graphs, strict=True)
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
