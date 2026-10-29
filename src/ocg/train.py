"""Offline and prequential training loops for forecasting models."""

import copy
import json
from pathlib import Path
from typing import Any, Sequence

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


def _forecast_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Keep structural VAR parameters under the graph learner's optimizer only."""
    graph_learner = getattr(model, "graph_learner", None)
    graph_parameters = (
        {id(parameter) for parameter in graph_learner.parameters()}
        if isinstance(graph_learner, GraphLearner)
        else set()
    )
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in graph_parameters
    ]


def _require_finite_training_state(
    model: nn.Module,
    phase: str,
    value: Tensor | None = None,
    gradients: bool = False,
) -> None:
    """Stop a training run as soon as its output, gradients, or weights diverge."""
    if value is not None and not torch.isfinite(value).all():
        raise FloatingPointError(f"{phase} produced non-finite values")
    for name, parameter in model.named_parameters():
        checked = parameter.grad if gradients else parameter
        if checked is not None and not torch.isfinite(checked).all():
            description = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"{phase} {description} became non-finite: {name}")


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
    restore_best_validation: bool = False,
    early_stopping_patience: int | None = None,
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
    if restore_best_validation and validation_inputs is None:
        raise ValueError("best-validation checkpointing requires validation data")
    if early_stopping_patience is not None:
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if validation_inputs is None:
            raise ValueError("early stopping requires validation data")
        if not restore_best_validation:
            raise ValueError("early stopping requires best-validation restoration")
    torch.manual_seed(seed)
    parameters = _forecast_parameters(model)
    if not parameters:
        raise ValueError("model must have forecast parameters outside its graph learner")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    graph_learner = getattr(model, "graph_learner", None)
    if graph_learner is not None and not isinstance(graph_learner, GraphLearner):
        raise TypeError("model.graph_learner must be a GraphLearner")
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_validation_epoch: int | None = None
    best_validation_state: dict[str, Tensor] | None = None
    best_graph_optimizer_state: dict[str, Any] | None = None
    epochs_executed = 0
    epochs_without_improvement = 0

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
                prediction = model(inputs)
                loss = nn.functional.mse_loss(prediction, targets)
                _require_finite_training_state(
                    model, "offline forecast", prediction
                )
                _require_finite_training_state(model, "offline forecast loss", loss)
                loss.backward()
                _require_finite_training_state(
                    model, "offline forecast", gradients=True
                )
                optimizer.step()
                _require_finite_training_state(model, "offline forecast")
                total_loss += loss.detach().item() * inputs.shape[0]
                if graph_learner is not None:
                    graph_learner.offline_update(inputs)
        train_losses.append(total_loss / n_examples)
        epochs_executed += 1

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
            if (
                restore_best_validation
                and validation_losses[-1] < best_validation_loss
            ):
                best_validation_loss = validation_losses[-1]
                best_validation_epoch = len(validation_losses)
                epochs_without_improvement = 0
                best_validation_state = copy.deepcopy(model.state_dict())
                if graph_learner is not None:
                    best_graph_optimizer_state = copy.deepcopy(
                        graph_learner._optimizer.state_dict()
                    )
            elif early_stopping_patience is not None:
                epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                break

    if best_validation_state is not None:
        model.load_state_dict(best_validation_state)
        if graph_learner is not None and best_graph_optimizer_state is not None:
            graph_learner._optimizer.load_state_dict(best_graph_optimizer_state)

    results: dict[str, Any] = {
        "mode": "offline",
        "seed": seed,
        "epochs_executed": epochs_executed,
        "train_mse": train_losses,
    }
    if validation_inputs is not None:
        results["validation_mse"] = validation_losses
    if restore_best_validation:
        results["best_validation_epoch"] = best_validation_epoch
        results["best_validation_mse"] = best_validation_loss
    if early_stopping_patience is not None:
        results["early_stopping_patience"] = early_stopping_patience
        results["stopped_early"] = epochs_executed < epochs
        results["reached_epoch_cap"] = epochs_executed == epochs
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
    timestamps: Sequence[int] | None = None,
    update_graph: bool = True,
    freeze_detector: bool = False,
    capture_forecast_components: bool = False,
) -> dict[str, Any]:
    """Run prequential training on (N,L,D)/(N,H,D) windows and optional oracle (N,K,D,D)."""
    _validate_windows(inputs, targets)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if timestamps is not None and (
        len(timestamps) != inputs.shape[0]
        or any(not isinstance(timestamp, int) for timestamp in timestamps)
    ):
        raise ValueError("timestamps must contain one integer per input window")
    if oracle_graphs is not None:
        _validate_oracle_schedule(
            model,
            inputs,
            oracle_graphs,
            torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device),
        )
    torch.manual_seed(seed)
    parameters = _forecast_parameters(model)
    if not parameters:
        raise ValueError("model must have forecast parameters outside its graph learner")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    graph_learner = getattr(model, "graph_learner", None)
    if graph_learner is not None and not isinstance(graph_learner, GraphLearner):
        raise TypeError("model.graph_learner must be a GraphLearner")

    predictions: list[Tensor] = []
    squared_errors: list[float] = []
    graph_history: list[Tensor] = []
    detections: list[bool] = []
    graph_scores: list[float] = []
    forecast_component_rms: list[dict[str, list[float]]] = []
    for index in range(inputs.shape[0]):
        model.train()
        window = inputs[index : index + 1]
        target = targets[index : index + 1]
        if oracle_graphs is not None:
            _set_oracle_graph(model, oracle_graphs[index])
        component_forward = getattr(model, "forward_with_components", None)
        if capture_forecast_components and callable(component_forward):
            prediction, components = component_forward(window)
            forecast_component_rms.append(
                {
                    name: component.square()
                    .mean(dim=(0, 2))
                    .sqrt()
                    .detach()
                    .cpu()
                    .tolist()
                    for name, component in components.items()
                }
            )
        else:
            prediction = model(window)
        if prediction.shape != target.shape:
            raise ValueError("model predictions must match target shape (N,H,D)")
        _require_finite_training_state(model, f"online forecast at step {index}", prediction)
        predictions.append(prediction.detach().cpu().clone())
        squared_errors.append(
            float(nn.functional.mse_loss(prediction.detach(), target).item())
        )
        if graph_learner is not None:
            graph_history.append(
                graph_learner.adjacency().detach().cpu().clone()
            )
            if update_graph:
                graph_score = graph_learner.online_update(
                    window, update_detector=not freeze_detector
                )
                detections.append(graph_learner.last_detection)
            else:
                graph_score = float(
                    graph_learner.structural_loss(window).detach().item()
                )
                detections.append(False)
            graph_scores.append(graph_score)
        observed_horizons = min(index, targets.shape[1])
        for horizon_index in range(observed_horizons):
            target_index = index - horizon_index - 1
            if oracle_graphs is not None:
                _set_oracle_graph(model, oracle_graphs[target_index])
            optimizer.zero_grad()
            delayed_prediction = model(
                inputs[target_index : target_index + 1]
            )[:, horizon_index : horizon_index + 1]
            delayed_target = targets[
                target_index : target_index + 1,
                horizon_index : horizon_index + 1,
            ]
            delayed_loss = nn.functional.mse_loss(
                delayed_prediction, delayed_target
            )
            _require_finite_training_state(
                model,
                f"delayed forecast at step {index}",
                delayed_prediction,
            )
            _require_finite_training_state(
                model, f"delayed forecast loss at step {index}", delayed_loss
            )
            delayed_loss.backward()
            _require_finite_training_state(
                model, f"delayed forecast at step {index}", gradients=True
            )
            optimizer.step()
            _require_finite_training_state(model, f"delayed forecast at step {index}")

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
    if graph_learner is not None:
        results["graph_history"] = torch.stack(graph_history).tolist()
        results["graph_scores"] = graph_scores
        results["detections"] = detections
        results["graph_timestamps"] = list(
            timestamps if timestamps is not None else range(inputs.shape[0])
        )
        results["detection_events"] = [
            {"step": index, "timestamp": results["graph_timestamps"][index]}
            for index, detected in enumerate(detections)
            if detected
        ]
    if forecast_component_rms:
        results["forecast_component_rms"] = forecast_component_rms
    _write_results(output_path, results)
    return results
