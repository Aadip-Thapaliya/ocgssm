"""Tests for forecasting and graph evaluation metrics."""

import pytest
import torch
from torch import Tensor, nn

from ocg.eval import (
    detection_delay,
    efficiency_metrics,
    forecast_metrics,
    graph_metrics,
    recovery_time,
    regret_vs_oracle,
)


class _SmallModel(nn.Module):
    """One-parameter model used to check parameter counting."""

    def __init__(self) -> None:
        """Initialize the parameter used by the efficiency metric test."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def test_forecast_metrics_and_oracle_regret() -> None:
    """Forecast metrics and regret use mean errors over (N,H,D)."""
    targets = torch.zeros(2, 2, 1)
    predictions = torch.ones_like(targets)
    oracle_predictions = torch.full_like(targets, 0.5)

    assert forecast_metrics(predictions, targets) == {"mse": 1.0, "mae": 1.0}
    assert regret_vs_oracle(predictions, oracle_predictions, targets) == 0.75


def test_graph_metrics_and_single_class_auroc() -> None:
    """Graph scores include SHD, threshold metrics, and optional AUROC."""
    truth = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    probabilities = torch.tensor([[[0.9, 0.8], [0.1, 0.2]]])
    metrics = graph_metrics(probabilities, truth)
    assert metrics["shd"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["auroc"] == 0.75

    single_class = graph_metrics(torch.zeros_like(truth), torch.zeros_like(truth))
    assert single_class["auroc"] is None


def test_recovery_detection_and_efficiency_metrics() -> None:
    """Adaptation timing and efficiency metrics match the supplied sequences."""
    assert recovery_time([2, 2, 5, 4, 2, 1], switch_at=2) == 2
    assert recovery_time([1, 1, 4, 3, 3], switch_at=2) is None
    assert detection_delay([False, False, False, True], switch_at=1) == 2
    assert detection_delay([False, False, False], switch_at=1) is None
    metrics = efficiency_metrics([0.01, 0.03], 2 * 1024 * 1024, _SmallModel())
    assert metrics == {
        "milliseconds_per_step": 20.0,
        "peak_memory_mb": 2.0,
        "parameters": 1,
    }


def test_metrics_reject_invalid_shapes_and_switch_indices() -> None:
    """Metric functions reject incompatible tensors and impossible switch indices."""
    with pytest.raises(ValueError):
        forecast_metrics(torch.zeros(2, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError):
        graph_metrics(torch.zeros(1, 2, 2), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        recovery_time([1.0, 2.0], switch_at=0)
    with pytest.raises(ValueError):
        detection_delay([False], switch_at=1)
