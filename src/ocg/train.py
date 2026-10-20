"""Offline and prequential training loops for forecasting models."""

import json
from copy import deepcopy
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


def warm_start_graph(
    model: nn.Module,
    inputs: Tensor,
    epochs: int,
    batch_size: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Warm-start learned graph logits on (N,L,D) windows via structural loss only."""
    _validate_windows(inputs, inputs[:, :1, :])
    if epochs < 0 or batch_size < 1:
        raise ValueError("epochs must be nonnegative and batch_size positive")
    graph_learner = getattr(model, "graph_learner", None)
    if not isinstance(graph_learner, GraphLearner):
        raise ValueError("warm_start_graph requires a model with GraphLearner")
    torch.manual_seed(seed)
    initial_logits = graph_learner.logits.detach().clone()
    n_examples = inputs.shape[0]
    losses: list[float] = []
    if epochs == 0:
        return {
            "warm_start_structural_loss": losses,
            "warm_start_logit_delta_l1": 0.0,
        }
    for _ in range(epochs):
        permutation = torch.randperm(n_examples, device=inputs.device)
        total_loss = 0.0
        for start in range(0, n_examples, batch_size):
            indices = permutation[start : start + batch_size]
            batch = inputs[indices]
            graph_learner._optimizer.zero_grad(set_to_none=True)
            loss = graph_learner.structural_loss(batch)
            loss.backward()
            graph_learner._optimizer.step()
            with torch.no_grad():
                graph_learner.ema_logits.mul_(graph_learner.ema_decay).add_(
                    graph_learner.logits, alpha=1.0 - graph_learner.ema_decay
                )
            total_loss += loss.detach().item() * batch.shape[0]
        losses.append(total_loss / n_examples)
    with torch.no_grad():
        logit_delta_l1 = torch.mean(
            torch.abs(graph_learner.logits - initial_logits)
        ).item()
    return {
        "warm_start_structural_loss": losses,
        "warm_start_logit_delta_l1": float(logit_delta_l1),
    }


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
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    lr_schedule: str = "none",
    min_learning_rate: float = 0.0,
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    """Fit on (N,L,D)/(N,H,D) windows, isolating oracle batches by regime."""
    _validate_windows(train_inputs, train_targets)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be adam or adamw")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided")
    if lr_schedule not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be none or cosine")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be nonnegative")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when provided")
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
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_learning_rate,
        )
        if lr_schedule == "cosine"
        else None
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_state: dict[str, Tensor] | None = None
    best_validation = float("inf")
    patience_remaining = early_stopping_patience

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
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
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
            if validation_losses[-1] < best_validation:
                best_validation = validation_losses[-1]
                best_state = deepcopy(model.state_dict())
                patience_remaining = early_stopping_patience
            elif patience_remaining is not None:
                patience_remaining -= 1
                if patience_remaining <= 0:
                    break
        if scheduler is not None:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)

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
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
) -> dict[str, Any]:
    """Run prequential training on (N,L,D)/(N,H,D) windows and optional oracle (N,K,D,D)."""
    _validate_windows(inputs, targets)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be adam or adamw")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when provided")
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
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
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
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
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
