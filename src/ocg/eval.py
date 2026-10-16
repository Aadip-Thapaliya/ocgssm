"""Forecast, graph, adaptation, regret, and efficiency metrics."""

from collections.abc import Sequence

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn


def _validate_forecasts(predictions: Tensor, targets: Tensor) -> None:
    """Validate matching forecast tensors shaped (N,H,D)."""
    if predictions.ndim != 3 or targets.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shape (N,H,D)")
    if predictions.numel() == 0:
        raise ValueError("forecast tensors must not be empty")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("forecasts and targets must contain only finite values")


def forecast_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute MSE and MAE from matching (N,H,D) forecast and target tensors."""
    _validate_forecasts(predictions, targets)
    errors = predictions - targets
    return {
        "mse": float(errors.square().mean().item()),
        "mae": float(errors.abs().mean().item()),
    }


def graph_metrics(
    predicted_graph: Tensor,
    true_graph: Tensor,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compute graph metrics for matching (K,D,D) probabilities and edge labels."""
    if predicted_graph.shape != true_graph.shape or predicted_graph.ndim != 3:
        raise ValueError("predicted_graph and true_graph must match with shape (K,D,D)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if not torch.isfinite(predicted_graph).all() or not torch.isfinite(true_graph).all():
        raise ValueError("graph tensors must contain only finite values")
    if torch.any((predicted_graph < 0) | (predicted_graph > 1)):
        raise ValueError("predicted graph probabilities must be in [0,1]")
    if torch.any((true_graph < 0) | (true_graph > 1)):
        raise ValueError("true graph edge labels must be in [0,1]")

    scores = predicted_graph.detach().cpu().reshape(-1)
    labels = true_graph.detach().cpu().reshape(-1) >= 0.5
    predicted_edges = scores >= threshold
    true_positive = int((predicted_edges & labels).sum().item())
    false_positive = int((predicted_edges & ~labels).sum().item())
    false_negative = int((~predicted_edges & labels).sum().item())
    shd = false_positive + false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auroc = (
        float(roc_auc_score(labels.numpy(), scores.numpy()))
        if labels.any() and (~labels).any()
        else None
    )
    return {
        "shd": shd,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
    }


def recovery_time(shd_by_step: Sequence[float], switch_at: int) -> int | None:
    """Return steps until SHD returns within 10% of pre-switch SHD, or None."""
    if switch_at <= 0 or switch_at >= len(shd_by_step):
        raise ValueError("switch_at must leave nonempty pre- and post-switch periods")
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in shd_by_step):
        raise ValueError("SHD values must be finite and nonnegative")
    baseline = sum(shd_by_step[:switch_at]) / switch_at
    limit = baseline * 1.1
    for step in range(switch_at, len(shd_by_step)):
        if shd_by_step[step] <= limit:
            return step - switch_at
    return None


def detection_delay(detections: Sequence[bool], switch_at: int) -> int | None:
    """Return steps from switch_at to first detection, or None if never detected."""
    if switch_at < 0 or switch_at >= len(detections):
        raise ValueError("switch_at must index the detection sequence")
    for step in range(switch_at, len(detections)):
        if detections[step]:
            return step - switch_at
    return None


def regret_vs_oracle(predictions: Tensor, oracle_predictions: Tensor, targets: Tensor) -> float:
    """Compute mean forecast MSE regret against oracle (N,H,D) predictions."""
    _validate_forecasts(predictions, targets)
    _validate_forecasts(oracle_predictions, targets)
    return float(
        (
            (predictions - targets).square().mean()
            - (oracle_predictions - targets).square().mean()
        ).item()
    )


def efficiency_metrics(
    step_times_seconds: Sequence[float],
    peak_memory_bytes: int,
    model: nn.Module,
) -> dict[str, float | int]:
    """Summarize step latency, peak memory, and parameter count for a model."""
    if not step_times_seconds or any(
        not torch.isfinite(torch.tensor(value)) or value < 0 for value in step_times_seconds
    ):
        raise ValueError("step_times_seconds must contain finite nonnegative values")
    if peak_memory_bytes < 0:
        raise ValueError("peak_memory_bytes must be nonnegative")
    return {
        "milliseconds_per_step": sum(step_times_seconds) / len(step_times_seconds) * 1000.0,
        "peak_memory_mb": peak_memory_bytes / (1024.0 * 1024.0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
