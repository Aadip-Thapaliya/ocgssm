"""Offline and prequential training loops for forecasting models."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ocg.graph import GraphLearner


def _validate_windows(inputs: Tensor, targets: Tensor) -> None:
    """Validate aligned forecasting tensors shaped (N,L,D) and (N,H,D)."""
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must have shapes (N,L,D) and (N,H,D)")
    if inputs.shape[0] < 1 or inputs.shape[2] != targets.shape[2]:
        raise ValueError("inputs and targets need matching nonempty examples and variables")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[1] < 1 or targets.shape[1] < 1:
        raise ValueError("inputs and targets need matching examples and positive window lengths")
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("inputs and targets must contain only finite values")
    if inputs.device != targets.device:
        raise ValueError("inputs and targets must be on the same device")


def _write_results(path: str | Path | None, results: dict[str, Any]) -> None:
    """Write a JSON-compatible result record when a path is provided."""
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2, allow_nan=False)
        result_file.write("\n")


def train_offline(
    model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit on training windows (N,L,D)/(N,H,D) and optionally log validation MSE."""
    _validate_windows(train_inputs, train_targets)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must be provided together")
    if validation_inputs is not None and validation_targets is not None:
        _validate_windows(validation_inputs, validation_targets)
        if validation_inputs.device != train_inputs.device:
            raise ValueError("training and validation tensors must be on the same device")
    torch.manual_seed(seed)
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model must have trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        n_examples = train_inputs.shape[0]
        for start in range(0, n_examples, batch_size):
            inputs = train_inputs[start : start + batch_size]
            targets = train_targets[start : start + batch_size]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * inputs.shape[0]
        train_losses.append(total_loss / n_examples)

        if validation_inputs is not None and validation_targets is not None:
            model.eval()
            with torch.no_grad():
                validation_loss = nn.functional.mse_loss(
                    model(validation_inputs), validation_targets
                )
            validation_losses.append(float(validation_loss.item()))

    results: dict[str, Any] = {
        "mode": "offline",
        "seed": seed,
        "train_mse": train_losses,
    }
    if validation_inputs is not None:
        results["validation_mse"] = validation_losses
    _write_results(output_path, results)
    return results


def train_online(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    learning_rate: float,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run prequential training on ordered windows (N,L,D)/(N,H,D)."""
    _validate_windows(inputs, targets)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    torch.manual_seed(seed)
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model must have trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    graph_learner = getattr(model, "graph_learner", None)
    if graph_learner is not None and not isinstance(graph_learner, GraphLearner):
        raise TypeError("model.graph_learner must be a GraphLearner")

    predictions: list[Tensor] = []
    squared_errors: list[float] = []
    for index in range(inputs.shape[0]):
        model.train()
        window = inputs[index : index + 1]
        target = targets[index : index + 1]
        optimizer.zero_grad()
        prediction = model(window)
        predictions.append(prediction.detach().cpu().clone())
        loss = nn.functional.mse_loss(prediction, target)
        squared_errors.append(float(loss.detach().item()))
        loss.backward()
        optimizer.step()
        if graph_learner is not None:
            graph_learner.online_update(window)

    stacked_predictions = torch.cat(predictions, dim=0)
    results: dict[str, Any] = {
        "mode": "online",
        "seed": seed,
        "prequential_mse": squared_errors,
        "running_mse": [
            sum(squared_errors[: index + 1]) / (index + 1)
            for index in range(len(squared_errors))
        ],
        "predictions": stacked_predictions.tolist(),
    }
    _write_results(output_path, results)
    return results
