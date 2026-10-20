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


def _validate_oracle_schedule(
    model: nn.Module,
    inputs: Tensor,
    oracle_graphs: Tensor | None,
    regime_ids: Tensor | None,
) -> None:
    """Validate oracle graphs (N,K,D,D) and offline regimes (N,)."""
    if (oracle_graphs is None) != (regime_ids is None):
        raise ValueError("oracle_graphs and regime_ids must be provided together")
    if oracle_graphs is None or regime_ids is None:
        return
    fixed_graph = getattr(model, "fixed_graph", None)
    if getattr(model, "graph_mode", None) != "oracle" or not isinstance(fixed_graph, Tensor):
        raise ValueError("oracle graph schedules require an oracle model with fixed_graph")
    if (
        oracle_graphs.ndim != 4
        or oracle_graphs.shape[0] != inputs.shape[0]
        or oracle_graphs.shape[1:] != fixed_graph.shape
    ):
        raise ValueError("oracle_graphs must have shape (N,K,D,D) aligned with inputs")
    if regime_ids.shape != (inputs.shape[0],) or regime_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("regime_ids must be an integer tensor shaped (N,)")
    if oracle_graphs.device != inputs.device or regime_ids.device != inputs.device:
        raise ValueError("oracle schedules must be on the same device as inputs")
    if (
        not torch.isfinite(oracle_graphs).all()
        or torch.any((oracle_graphs < 0) | (oracle_graphs > 1))
    ):
        raise ValueError("oracle graph probabilities must be finite and in [0,1]")


def _set_oracle_graph(model: nn.Module, graph: Tensor) -> None:
    """Set an oracle model's fixed adjacency from one (K,D,D) graph."""
    fixed_graph = getattr(model, "fixed_graph", None)
    if not isinstance(fixed_graph, Tensor) or fixed_graph.shape != graph.shape:
        raise ValueError("oracle model fixed_graph must match graph shape (K,D,D)")
    with torch.no_grad():
        fixed_graph.copy_(graph.to(device=fixed_graph.device, dtype=fixed_graph.dtype))


def _regime_ids_from_graphs(oracle_graphs: Tensor) -> Tensor:
    """Label consecutive constant (K,D,D) graph regimes for (N,K,D,D) windows."""
    if oracle_graphs.ndim != 4 or oracle_graphs.shape[0] < 1:
        raise ValueError("oracle_graphs must have nonempty shape (N,K,D,D)")
    changed = (oracle_graphs[1:] != oracle_graphs[:-1]).reshape(
        oracle_graphs.shape[0] - 1, -1
    ).any(dim=1)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=oracle_graphs.device),
            changed.to(dtype=torch.long).cumsum(dim=0),
        )
    )


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
    oracle_graphs: Tensor | None = None,
    regime_ids: Tensor | None = None,
    validation_oracle_graphs: Tensor | None = None,
    validation_regime_ids: Tensor | None = None,
) -> dict[str, Any]:
    """Fit on (N,L,D)/(N,H,D) windows, isolating oracle batches by regime."""
    _validate_windows(train_inputs, train_targets)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    _validate_oracle_schedule(model, train_inputs, oracle_graphs, regime_ids)
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must be provided together")
    if validation_inputs is not None and validation_targets is not None:
        _validate_windows(validation_inputs, validation_targets)
        if validation_inputs.device != train_inputs.device:
            raise ValueError("training and validation tensors must be on the same device")
        _validate_oracle_schedule(
            model,
            validation_inputs,
            validation_oracle_graphs,
            validation_regime_ids,
        )
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
        if oracle_graphs is not None and regime_ids is not None:
            batch_groups = [
                torch.nonzero(regime_ids == regime, as_tuple=False).flatten()
                for regime in torch.unique(regime_ids, sorted=True)
            ]
        else:
            batch_groups = [torch.arange(n_examples, device=train_inputs.device)]
        for group_indices in batch_groups:
            if oracle_graphs is not None:
                _set_oracle_graph(model, oracle_graphs[group_indices[0]])
            for start in range(0, group_indices.shape[0], batch_size):
                indices = group_indices[start : start + batch_size]
                inputs = train_inputs[indices]
                targets = train_targets[indices]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(inputs), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item() * inputs.shape[0]
        train_losses.append(total_loss / n_examples)

        if validation_inputs is not None and validation_targets is not None:
            model.eval()
            with torch.no_grad():
                if validation_oracle_graphs is not None and validation_regime_ids is not None:
                    validation_total = 0.0
                    for regime in torch.unique(validation_regime_ids, sorted=True):
                        indices = torch.nonzero(
                            validation_regime_ids == regime, as_tuple=False
                        ).flatten()
                        _set_oracle_graph(model, validation_oracle_graphs[indices[0]])
                        loss = nn.functional.mse_loss(
                            model(validation_inputs[indices]),
                            validation_targets[indices],
                        )
                        validation_total += loss.item() * indices.shape[0]
                    validation_loss = torch.tensor(
                        validation_total / validation_inputs.shape[0],
                        device=validation_inputs.device,
                    )
                else:
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
    oracle_graphs: Tensor | None = None,
    return_tensors: bool = False,
) -> dict[str, Any]:
    """Run prequential training on (N,L,D)/(N,H,D) windows and optional oracle (N,K,D,D)."""
    _validate_windows(inputs, targets)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if oracle_graphs is not None:
        _validate_oracle_schedule(
            model,
            inputs,
            oracle_graphs,
            torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device),
        )
    torch.manual_seed(seed)
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model must have trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    graph_learner = getattr(model, "graph_learner", None)
    if graph_learner is not None and not isinstance(graph_learner, GraphLearner):
        raise TypeError("model.graph_learner must be a GraphLearner")

    predictions = torch.empty_like(targets)
    squared_errors = torch.empty(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    graph_history: list[Tensor] = []
    detections: list[bool] = []
    model.train()
    for index in range(inputs.shape[0]):
        window = inputs[index : index + 1]
        target = targets[index : index + 1]
        if oracle_graphs is not None:
            _set_oracle_graph(model, oracle_graphs[index])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(window)
        predictions[index] = prediction.detach()[0]
        loss = nn.functional.mse_loss(prediction, target)
        squared_errors[index] = loss.detach()
        loss.backward()
        optimizer.step()
        if graph_learner is not None:
            graph_learner.online_update(window)
            graph_history.append(graph_learner.adjacency().detach().cpu().clone())
            detections.append(graph_learner.last_detection)

    squared_errors_cpu = squared_errors.detach().cpu()
    running_mse = torch.cumsum(squared_errors_cpu, dim=0) / torch.arange(
        1,
        squared_errors_cpu.shape[0] + 1,
        dtype=squared_errors_cpu.dtype,
    )
    results: dict[str, Any] = {
        "mode": "online",
        "seed": seed,
        "prequential_mse": squared_errors_cpu.tolist(),
        "running_mse": running_mse.tolist(),
        "predictions": predictions.detach().cpu().tolist(),
    }
    if return_tensors:
        results["predictions_tensor"] = predictions.detach()
        results["prequential_mse_tensor"] = squared_errors.detach()
    if graph_learner is not None:
        results["graph_history"] = torch.stack(graph_history).tolist()
        results["detections"] = detections
    _write_results(output_path, results)
    return results
