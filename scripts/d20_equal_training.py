"""Select D20 settings on development data and run the headroom gate."""

import json
import hashlib
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.data import generate_svar, make_windows
from ocg.eval import graph_metrics
from ocg.graph import GraphLearner
from ocg.run import _build_model, _select_graph_threshold
from ocg.train import _regime_ids_from_graphs, train_offline, train_online


ROOT = Path("results/d20_equal_training")
PROTOCOL = ROOT / "protocol.yaml"
REDUCED_PROTOCOL = ROOT / "reduced_proposal_v2.yaml"
FREEZE_MANIFEST = ROOT / "freeze_manifest_v4.json"
CAMPAIGN_LEDGER = ROOT / "campaign_state_v4.json"
DATA_LR_CANDIDATES = (0.0003, 0.001, 0.003)
GRAPH_LR_CANDIDATES = (0.001, 0.01, 0.03)
SELECTION_EPOCHS = 10
FINAL_EPOCHS = 50
REDUCED_EPOCHS = 20
ORACLE_SIDE_CHECK_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32
BOOTSTRAP_BLOCK_LENGTH = 12
BOOTSTRAP_RESAMPLES = 2000


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _configuration() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def _development_data() -> dict[str, Any]:
    protocol = _configuration()
    data_config = protocol["data"]
    seed = int(protocol["seeds"]["development_pair"]["data"])
    data = generate_svar(
        n_steps=int(data_config["steps"]),
        n_variables=int(data_config["variables"]),
        n_lags=int(data_config["lags"]),
        switch_at=int(data_config["switch_at"]),
        blend_steps=int(data_config["blend_steps"]),
        sparsity=float(data_config["sparsity"]),
        noise_scale=float(data_config["noise_scale"]),
        seed=seed,
        cross_variable_scale=float(data_config["cross_variable_scale"]),
    )
    series = data["series"]
    graph = data["graph"]
    train_x, train_y = make_windows(series[0:2400], 24, 6)
    tune_x, tune_y = make_windows(series[2400:2666], 24, 6)
    threshold_x, threshold_y = make_windows(series[2666:2933], 24, 6)
    gate_x, gate_y = make_windows(series[2933:3200], 24, 6)
    train_graphs = graph[23 : 23 + len(train_x)]
    tune_graphs = graph[2423 : 2423 + len(tune_x)]
    threshold_graphs = graph[2689 : 2689 + len(threshold_x)]
    gate_graphs = graph[2956 : 2956 + len(gate_x)]
    return {
        "data": data,
        "train": (train_x, train_y),
        "tune": (tune_x, tune_y),
        "threshold": (threshold_x, threshold_y),
        "gate": (gate_x, gate_y),
        "train_graphs": train_graphs,
        "tune_graphs": tune_graphs,
        "threshold_graphs": threshold_graphs,
        "gate_graphs": gate_graphs,
    }


def _graph_config(learning_rate: float) -> dict[str, Any]:
    return {
        "learning_rate": learning_rate,
        "sparsity_weight": 0.001,
        "temperature": 1.0,
        "ema_decay": 0.95,
        "graph_stride": 1,
        "boost": 1.0,
        "boost_steps": 20,
        "detector_decay": 0.95,
        "detector_threshold": 1e9,
        "detector_warmup": 30,
        "detector_min_std": 1e-6,
    }


def _build(
    variant: str,
    forecast_lr: float,
    graph_lr: float,
    model_seed: int,
    true_graph: torch.Tensor | None = None,
    graph_mask_threshold: float | None = None,
) -> torch.nn.Module:
    model_config: dict[str, Any]
    if variant in {
        "identity",
        "dense",
        "random",
        "learned",
        "learned_hard",
        "oracle",
    }:
        mode = "learned" if variant in {"learned", "learned_hard"} else variant
        model_config = {
            "id": variant,
            "name": "ocg",
            "input_length": 24,
            "state_dim": 8,
            "n_layers": 1,
            "embedding_dim": 4,
            "graph_mode": mode,
            "graph": _graph_config(graph_lr),
            "graph_mask_threshold": graph_mask_threshold,
        }
    elif variant == "var_direct":
        model_config = {"id": variant, "name": "var_direct", "input_length": 24}
    elif variant == "dlinear":
        model_config = {
            "id": variant,
            "name": "dlinear",
            "input_length": 24,
            "moving_average": 3,
        }
    else:
        raise ValueError(f"unknown D20 variant: {variant}")
    torch.manual_seed(model_seed)
    return _build_model(
        model_config,
        n_variables=20,
        horizon=6,
        n_lags=2,
        true_graph=true_graph,
        seed=model_seed,
        random_sparsity=0.1,
    )


def _forecast_lr_selection(data: dict[str, Any], model_seed: int) -> dict[str, Any]:
    path = ROOT / "development" / "forecast_lr_selection.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "selected_learning_rate" in cached:
        return cached
    train_x, train_y = data["train"]
    tune_x, tune_y = data["tune"]
    rows = cached.get("candidates", [])
    for learning_rate in DATA_LR_CANDIDATES:
        for variant in ("identity", "var_direct"):
            if any(
                row["learning_rate"] == learning_rate and row["variant"] == variant
                for row in rows
            ):
                continue
            model = _build(variant, learning_rate, 0.01, model_seed)
            started = time.perf_counter()
            result = train_offline(
                model,
                train_x,
                train_y,
                epochs=SELECTION_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=learning_rate,
                seed=model_seed,
                validation_inputs=tune_x,
                validation_targets=tune_y,
                restore_best_validation=True,
            )
            row = {
                "learning_rate": learning_rate,
                "variant": variant,
                "elapsed_seconds": time.perf_counter() - started,
                "best_validation_epoch": result["best_validation_epoch"],
                "best_validation_mse": result["best_validation_mse"],
                "validation_mse": result["validation_mse"],
            }
            rows.append(row)
            print(json.dumps({"stage": "forecast_lr", **row}), flush=True)
            _write_json(path, {"candidates": rows})
    best_by_variant = {
        variant: min(
            row["best_validation_mse"]
            for row in rows
            if row["variant"] == variant
        )
        for variant in ("identity", "var_direct")
    }
    scores = {}
    for learning_rate in DATA_LR_CANDIDATES:
        values = [
            next(
                row["best_validation_mse"]
                for row in rows
                if row["variant"] == variant
                and row["learning_rate"] == learning_rate
            )
            / best_by_variant[variant]
            for variant in ("identity", "var_direct")
        ]
        scores[str(learning_rate)] = statistics.fmean(values)
    selected = min(DATA_LR_CANDIDATES, key=lambda lr: (scores[str(lr)], lr))
    result = {
        "candidates": rows,
        "normalized_equal_variant_validation_scores": scores,
        "selected_learning_rate": selected,
        "selection_source": "training windows and development validation raw indices 2400-2665 only",
    }
    _write_json(path, result)
    return result


def _graph_lr_selection(data: dict[str, Any], model_seed: int) -> dict[str, Any]:
    path = ROOT / "development" / "graph_lr_selection.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "selected_learning_rate" in cached:
        return cached
    train_x, _ = data["train"]
    tune_graphs = data["tune_graphs"]
    rows = cached.get("candidates", [])
    for learning_rate in GRAPH_LR_CANDIDATES:
        if any(row["learning_rate"] == learning_rate for row in rows):
            continue
        torch.manual_seed(model_seed)
        learner = GraphLearner(
            n_variables=20,
            n_lags=2,
            learning_rate=learning_rate,
            sparsity_weight=0.001,
            temperature=1.0,
            ema_decay=0.95,
        )
        started = time.perf_counter()
        for _ in range(SELECTION_EPOCHS):
            for start in range(0, len(train_x), BATCH_SIZE):
                learner.offline_update(train_x[start : start + BATCH_SIZE])
        elapsed = time.perf_counter() - started
        probabilities = learner.adjacency().detach()
        metrics = graph_metrics(
            probabilities,
            tune_graphs[0],
            threshold=0.5,
        )
        row = {
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "validation_graph_auroc": metrics["auroc"],
            "validation_graph_average_precision": metrics["average_precision"],
            "validation_graph_f1_at_0_5": metrics["f1"],
            "validation_graph_shd_at_0_5": metrics["shd"],
            "structural_loss": float(learner.structural_loss(train_x[:32]).detach()),
        }
        rows.append(row)
        print(json.dumps({"stage": "graph_lr", **row}), flush=True)
        _write_json(path, {"candidates": rows})
    selected_row = min(
        rows,
        key=lambda row: (
            -float(row["validation_graph_auroc"]),
            -float(row["validation_graph_average_precision"]),
            float(row["learning_rate"]),
        ),
    )
    result = {
        "candidates": rows,
        "selected_learning_rate": selected_row["learning_rate"],
        "selection_source": "training windows and development validation graph at raw indices 2400-2665 only",
        "criterion": "maximum AUROC, then average precision, then lowest learning rate",
    }
    _write_json(path, result)
    return result


def _select_hard_threshold(
    model: torch.nn.Module,
    validation_graphs: torch.Tensor,
) -> tuple[float, float]:
    learner = getattr(model, "graph_learner", None)
    if not isinstance(learner, GraphLearner):
        raise TypeError("threshold calibration requires a learned graph model")
    probabilities = learner.adjacency().detach().unsqueeze(0).expand(
        len(validation_graphs), -1, -1, -1
    )
    return _select_graph_threshold(probabilities, validation_graphs)


def _predict_errors(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[float], list[list[float]]]:
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
    errors = (predictions - targets).square()
    return (
        errors.mean(dim=(1, 2)).detach().cpu().tolist(),
        errors.mean(dim=2).detach().cpu().tolist(),
    )


def _moving_block_ratio_interval(
    numerator: list[float],
    denominator: list[float],
    seed: int,
) -> list[float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap inputs must be nonempty and aligned")
    generator = np.random.default_rng(seed)
    n = len(numerator)
    n_blocks = math.ceil(n / BOOTSTRAP_BLOCK_LENGTH)
    draws = np.empty(BOOTSTRAP_RESAMPLES)
    for draw in range(BOOTSTRAP_RESAMPLES):
        starts = generator.integers(0, n, size=n_blocks)
        indices = [
            (int(start) + offset) % n
            for start in starts
            for offset in range(BOOTSTRAP_BLOCK_LENGTH)
        ][:n]
        draws[draw] = (
            statistics.fmean(numerator[index] for index in indices)
            / statistics.fmean(denominator[index] for index in indices)
        )
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _headroom_gate(data: dict[str, Any], rates: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "development" / "headroom_gate.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    protocol = _configuration()
    model_seed = int(protocol["seeds"]["development_pair"]["model"])
    forecast_lr = float(rates["forecast"]["selected_learning_rate"])
    graph_lr = float(rates["graph"]["selected_learning_rate"])
    train_x, train_y = data["train"]
    tune_x, tune_y = data["tune"]
    gate_x, gate_y = data["gate"]
    gate_errors: dict[str, list[list[float]]] = {}
    training = {}
    for variant in ("identity", "var_direct"):
        model = _build(variant, forecast_lr, graph_lr, model_seed)
        started = time.perf_counter()
        result = train_offline(
            model,
            train_x,
            train_y,
            epochs=FINAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=forecast_lr,
            seed=model_seed,
            validation_inputs=tune_x,
            validation_targets=tune_y,
            restore_best_validation=True,
        )
        elapsed = time.perf_counter() - started
        _, per_horizon = _predict_errors(model, gate_x, gate_y)
        gate_errors[variant] = per_horizon
        training[variant] = {
            "elapsed_seconds": elapsed,
            "best_validation_epoch": result["best_validation_epoch"],
            "best_validation_mse": result["best_validation_mse"],
            "training_mse": result["train_mse"],
            "validation_mse": result["validation_mse"],
        }
        print(
            json.dumps(
                {
                    "stage": "headroom_fit",
                    "variant": variant,
                    "elapsed_seconds": elapsed,
                    "best_validation_epoch": result["best_validation_epoch"],
                    "best_validation_mse": result["best_validation_mse"],
                }
            ),
            flush=True,
        )
    horizon_results = []
    for horizon in (0, 1):
        identity = [row[horizon] for row in gate_errors["identity"]]
        direct = [row[horizon] for row in gate_errors["var_direct"]]
        ratio = statistics.fmean(direct) / statistics.fmean(identity)
        interval = _moving_block_ratio_interval(
            direct,
            identity,
            seed=500000 + horizon,
        )
        horizon_results.append(
            {
                "horizon": horizon + 1,
                "identity_mse": statistics.fmean(identity),
                "var_direct_mse": statistics.fmean(direct),
                "var_direct_identity_ratio": ratio,
                "paired_circular_block_bootstrap_95pct_ci": interval,
                "point_ratio_le_0_95": ratio <= 0.95,
                "ci_upper_below_1": interval[1] < 1.0,
                "passed": ratio <= 0.95 and interval[1] < 1.0,
            }
        )
    result = {
        "data_model_seed": [
            int(protocol["seeds"]["development_pair"]["data"]),
            model_seed,
        ],
        "validation_gate_raw_indices": [2933, 3199],
        "validation_gate_windows": len(gate_x),
        "bootstrap": {
            "method": "paired circular moving-block percentile",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        },
        "training": training,
        "per_horizon": horizon_results,
        "passed": all(row["passed"] for row in horizon_results),
        "errors": {
            variant: [
                [float(value) for value in row]
                for row in rows
            ]
            for variant, rows in gate_errors.items()
        },
    }
    _write_json(path, result)
    return result


def _save_pretrain_checkpoint(
    model: torch.nn.Module,
    path: Path,
    training: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    learner = getattr(model, "graph_learner", None)
    checkpoint = {
        "model_state": model.state_dict(),
        "training": training,
        "elapsed_seconds": elapsed_seconds,
        "graph_optimizer_state": (
            learner._optimizer.state_dict()
            if isinstance(learner, GraphLearner)
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _load_pretrain_checkpoint(
    variant: str,
    path: Path,
    forecast_lr: float,
    graph_lr: float,
    model_seed: int,
    true_graph: torch.Tensor,
    hard_threshold: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = _build(
        variant,
        forecast_lr,
        graph_lr,
        model_seed,
        true_graph=true_graph,
        graph_mask_threshold=hard_threshold if variant == "learned_hard" else None,
    )
    model.load_state_dict(checkpoint["model_state"])
    learner = getattr(model, "graph_learner", None)
    if (
        isinstance(learner, GraphLearner)
        and checkpoint["graph_optimizer_state"] is not None
    ):
        learner._optimizer.load_state_dict(checkpoint["graph_optimizer_state"])
    return model, checkpoint


def _mean_graph_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("auroc", "average_precision", "precision", "recall", "f1", "shd"):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = statistics.fmean(values) if values else None
    return result


def _graph_quality(
    snapshots: torch.Tensor,
    truth: torch.Tensor,
    timestamps: list[int],
    threshold: float,
) -> dict[str, Any]:
    if snapshots.shape != truth.shape or snapshots.shape[0] != len(timestamps):
        raise ValueError("graph snapshots, truth, and timestamps must align")
    metrics = [
        graph_metrics(snapshots[index], truth[index], threshold)
        for index in range(len(timestamps))
    ]
    periods = {}
    for label, selection in (
        ("pre_switch", [index for index, value in enumerate(timestamps) if value < 3600]),
        ("post_switch", [index for index, value in enumerate(timestamps) if value >= 3600]),
    ):
        selected = [metrics[index] for index in selection]
        periods[label] = {
            "snapshots": len(selection),
            **_mean_graph_metrics(selected),
        }
    return {
        "threshold": threshold,
        "timestamp_alignment": "pre-update EMA graph at last input observation",
        "periods": periods,
    }


def _run_variant(
    variant: str,
    data: dict[str, Any],
    data_seed: int,
    model_seed: int,
    rates: dict[str, float],
    hard_threshold: float,
    checkpoint_path: Path,
    graphs_path: Path,
    epoch_cap: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    train_x, train_y = data["train"]
    validation_x, validation_y = data["validation"]
    test_x, test_y = data["test"]
    train_graphs = data["train_graphs"]
    validation_graphs = data["validation_graphs"]
    test_graphs = data["test_graphs"]
    forecast_lr = rates["forecast"]
    graph_lr = rates["graph"]
    if variant == "learned_hard":
        learned_path = checkpoint_path.with_name("learned.pt")
        soft, soft_checkpoint = _load_pretrain_checkpoint(
            "learned",
            learned_path,
            forecast_lr,
            graph_lr,
            model_seed,
            data["all_graphs"],
            hard_threshold,
        )
        learner = getattr(soft, "graph_learner", None)
        if not isinstance(learner, GraphLearner):
            raise TypeError("learned pretraining checkpoint has no graph learner")
        _save_pretrain_checkpoint(
            soft,
            checkpoint_path,
            {
                "mode": "copied learned-soft pretraining; no second fit",
                "selected_validation_epoch": soft_checkpoint["training"][
                    "best_validation_epoch"
                ],
            },
            0.0,
        )
        model, pretrain = _load_pretrain_checkpoint(
            "learned_hard",
            checkpoint_path,
            forecast_lr,
            graph_lr,
            model_seed,
            data["all_graphs"],
            hard_threshold,
        )
    elif checkpoint_path.exists():
        model, pretrain = _load_pretrain_checkpoint(
            variant,
            checkpoint_path,
            forecast_lr,
            graph_lr,
            model_seed,
            data["all_graphs"],
            hard_threshold,
        )
    else:
        model = _build(
            variant,
            forecast_lr,
            graph_lr,
            model_seed,
            true_graph=data["all_graphs"],
        )
        oracle = variant == "oracle"
        started = time.perf_counter()
        train_result = train_offline(
            model,
            train_x,
            train_y,
            epochs=epoch_cap,
            batch_size=BATCH_SIZE,
            learning_rate=forecast_lr,
            seed=model_seed,
            validation_inputs=validation_x,
            validation_targets=validation_y,
            oracle_graphs=train_graphs if oracle else None,
            regime_ids=(
                _regime_ids_from_graphs(train_graphs) if oracle else None
            ),
            validation_oracle_graphs=validation_graphs if oracle else None,
            validation_regime_ids=(
                _regime_ids_from_graphs(validation_graphs) if oracle else None
            ),
            restore_best_validation=True,
            early_stopping_patience=early_stopping_patience,
        )
        pretrain = {
            "epochs_executed": train_result["epochs_executed"],
            "epoch_cap": epoch_cap,
            "early_stopping_patience": early_stopping_patience,
            "stopped_early": train_result["stopped_early"],
            "reached_epoch_cap": train_result["reached_epoch_cap"],
            "convergence_status": (
                "not_converged_cap_reached"
                if train_result["reached_epoch_cap"]
                else "stopped_by_validation_patience"
            ),
            "best_validation_epoch": train_result["best_validation_epoch"],
            "best_validation_mse": train_result["best_validation_mse"],
            "training_mse": train_result["train_mse"],
            "validation_mse": train_result["validation_mse"],
        }
        _save_pretrain_checkpoint(
            model,
            checkpoint_path,
            pretrain,
            time.perf_counter() - started,
        )
        model, checkpoint = _load_pretrain_checkpoint(
            variant,
            checkpoint_path,
            forecast_lr,
            graph_lr,
            model_seed,
            data["all_graphs"],
            hard_threshold,
        )
        pretrain = checkpoint["training"]

    learner = getattr(model, "graph_learner", None)
    if isinstance(learner, GraphLearner):
        learner.detector.threshold = float("inf")
        learner.boost = 1.0
    threshold = 0.5
    if isinstance(model, torch.nn.Module) and getattr(model, "graph_mode", None):
        if model.graph_mode == "oracle":
            validation_snapshots = validation_graphs
        elif isinstance(learner, GraphLearner):
            validation_snapshots = learner.adjacency().detach().unsqueeze(0).expand(
                len(validation_graphs), -1, -1, -1
            )
        else:
            fixed = model._lagged_adjacency(validation_x).detach()
            validation_snapshots = fixed.unsqueeze(0).expand(
                len(validation_graphs), -1, -1, -1
            )
        threshold, validation_f1 = _select_graph_threshold(
            validation_snapshots,
            validation_graphs,
        )
        validation_timestamps = list(range(2423, 2423 + len(validation_graphs)))
        validation_quality = _graph_quality(
            validation_snapshots,
            validation_graphs,
            validation_timestamps,
            threshold,
        )
        validation_quality["threshold_selection_f1"] = validation_f1
    else:
        validation_quality = None

    test_timestamps = list(range(3223, 3223 + len(test_x)))
    started = time.perf_counter()
    stream_result = train_online(
        model,
        test_x,
        test_y,
        learning_rate=forecast_lr,
        seed=model_seed,
        oracle_graphs=test_graphs if variant == "oracle" else None,
        timestamps=test_timestamps,
        update_graph=True,
        freeze_detector=True,
    )
    elapsed = time.perf_counter() - started
    predictions = torch.tensor(stream_result["predictions"], dtype=test_y.dtype)
    per_horizon = (
        (predictions - test_y).square().mean(dim=2).detach().cpu().tolist()
    )
    per_window = (
        (predictions - test_y).square().mean(dim=(1, 2)).detach().cpu().tolist()
    )

    graph_quality = None
    if getattr(model, "graph_mode", None):
        if isinstance(learner, GraphLearner):
            graph_snapshots = torch.tensor(stream_result["graph_history"])
        elif variant == "oracle":
            graph_snapshots = test_graphs.detach().cpu()
        else:
            fixed = model._lagged_adjacency(test_x).detach().cpu()
            graph_snapshots = fixed.unsqueeze(0).expand(len(test_x), -1, -1, -1)
        graph_quality = _graph_quality(
            graph_snapshots,
            test_graphs.detach().cpu(),
            test_timestamps,
            threshold,
        )
        graphs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "timestamps": test_timestamps,
                "pre_update_probabilities": graph_snapshots,
                "truth": test_graphs.detach().cpu(),
            },
            graphs_path,
        )
    return {
        "variant": variant,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "pretraining": pretrain,
        "pretraining_elapsed_seconds": (
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
                "elapsed_seconds"
            ]
        ),
        "validation_graph_quality": validation_quality,
        "online_adaptation": {
            "schedule": "delayed-label prequential; forecast Adam reset at stream boundary",
            "elapsed_seconds": elapsed,
            "test_windows": len(test_x),
        },
        "forecast": {
            "per_window_mse": per_window,
            "per_window_per_horizon_mse": per_horizon,
            "per_horizon_mse": [
                statistics.fmean(row[h] for row in per_horizon)
                for h in range(test_y.shape[1])
            ],
        },
        "graph_quality": graph_quality,
        "graph_snapshots_file": str(graphs_path) if graph_quality is not None else None,
    }


def _heldout_data(data_seed: int) -> dict[str, Any]:
    protocol = _configuration()
    data_config = protocol["data"]
    generated = generate_svar(
        n_steps=int(data_config["steps"]),
        n_variables=int(data_config["variables"]),
        n_lags=int(data_config["lags"]),
        switch_at=int(data_config["switch_at"]),
        blend_steps=int(data_config["blend_steps"]),
        sparsity=float(data_config["sparsity"]),
        noise_scale=float(data_config["noise_scale"]),
        seed=data_seed,
        cross_variable_scale=float(data_config["cross_variable_scale"]),
    )
    series = generated["series"]
    graph = generated["graph"]
    train_x, train_y = make_windows(series[:2400], 24, 6)
    validation_x, validation_y = make_windows(series[2400:3200], 24, 6)
    test_x, test_y = make_windows(series[3200:4000], 24, 6)
    train_graphs = graph[23 : 23 + len(train_x)]
    validation_graphs = graph[2423 : 2423 + len(validation_x)]
    test_graphs = graph[3223 : 3223 + len(test_x)]
    return {
        "train": (train_x, train_y),
        "validation": (validation_x, validation_y),
        "test": (test_x, test_y),
        "all_graphs": graph,
        "train_graphs": train_graphs,
        "validation_graphs": validation_graphs,
        "test_graphs": test_graphs,
    }


def _run_heldout_pair(
    data_seed: int,
    model_seed: int,
    rates: dict[str, float],
    hard_threshold: float,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    directory = ROOT / "runs" / f"data{data_seed}_model{model_seed}"
    output_path = directory / "result.json"
    checkpoint_dir = directory / "checkpoints"
    graph_dir = directory / "graphs"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("data_seed") != data_seed or existing.get("model_seed") != model_seed:
            raise ValueError(f"refusing to reuse mismatched run output {output_path}")
        if existing.get("protocol_sha256") != hashlib.sha256(REDUCED_PROTOCOL.read_bytes()).hexdigest():
            raise ValueError(f"protocol hash changed; refusing to resume {output_path}")
        if set(existing.get("variants", {})) - {
            "identity",
            "dense",
            "learned",
            "oracle",
            "var_direct",
        }:
            raise ValueError(f"unexpected variant found in reduced run output {output_path}")
    else:
        existing = {
            "data_seed": data_seed,
            "model_seed": model_seed,
            "protocol": str(REDUCED_PROTOCOL),
            "protocol_sha256": hashlib.sha256(REDUCED_PROTOCOL.read_bytes()).hexdigest(),
            "rates": rates,
            "hard_mask_threshold": hard_threshold,
            "variants": {},
        }
    data = _heldout_data(data_seed)
    variants = ("identity", "dense", "learned", "oracle", "var_direct")
    key = f"data{data_seed}_model{model_seed}"
    ledger["runs"].setdefault(
        key,
        {
            "status": "in_progress",
            "output": str(output_path),
            "completed_variants": list(existing["variants"]),
        },
    )
    _write_json(CAMPAIGN_LEDGER, ledger)
    for variant in variants:
        if variant in existing["variants"]:
            continue
        checkpoint_path = checkpoint_dir / f"{variant}.pt"
        graphs_path = graph_dir / f"{variant}.pt"
        result = _run_variant(
            variant,
            data,
            data_seed,
            model_seed,
            rates,
            0.5,
            checkpoint_path,
            graphs_path,
            REDUCED_EPOCHS,
            EARLY_STOPPING_PATIENCE,
        )
        existing["variants"][variant] = result
        existing["completed_variants"] = list(existing["variants"])
        _write_json(output_path, existing)
        ledger["runs"][key]["completed_variants"] = list(existing["variants"])
        ledger["runs"][key]["status"] = (
            "complete"
            if len(existing["variants"]) == len(variants)
            else "in_progress"
        )
        _write_json(CAMPAIGN_LEDGER, ledger)
        print(
            json.dumps(
                {
                    "stage": "heldout_variant_complete",
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "variant": variant,
                    "elapsed_seconds": result["online_adaptation"]["elapsed_seconds"],
                    "output": str(output_path),
                }
            ),
            flush=True,
        )
    return existing


def _paired_summary(
    runs: list[dict[str, Any]],
    variant: str,
    reference: str,
    horizon_index: int,
    ratio: bool,
) -> dict[str, Any]:
    numerators = []
    denominators = []
    for run in runs:
        candidate = [
            row[horizon_index]
            for row in run["variants"][variant]["forecast"][
                "per_window_per_horizon_mse"
            ]
        ]
        base = [
            row[horizon_index]
            for row in run["variants"][reference]["forecast"][
                "per_window_per_horizon_mse"
            ]
        ]
        numerators.append(candidate)
        denominators.append(base)
    seed_statistics = []
    for candidate, base in zip(numerators, denominators, strict=True):
        candidate_mean = statistics.fmean(candidate)
        base_mean = statistics.fmean(base)
        seed_statistics.append(
            candidate_mean / base_mean
            if ratio
            else candidate_mean - base_mean
        )
    generator = np.random.default_rng(
        700000 + 1000 * horizon_index + sum(ord(c) for c in variant + reference)
    )
    seed_draws = []
    for candidate, base in zip(numerators, denominators, strict=True):
        n = len(candidate)
        starts = generator.integers(
            0,
            n,
            size=(BOOTSTRAP_RESAMPLES, math.ceil(n / BOOTSTRAP_BLOCK_LENGTH)),
        )
        offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH)
        indices = (
            (starts[:, :, None] + offsets[None, None, :]) % n
        ).reshape(BOOTSTRAP_RESAMPLES, -1)[:, :n]
        candidate_means = np.asarray(candidate)[indices].mean(axis=1)
        base_means = np.asarray(base)[indices].mean(axis=1)
        seed_draws.append(
            candidate_means / base_means
            if ratio
            else candidate_means - base_means
        )
    draws = np.mean(np.stack(seed_draws, axis=0), axis=0)
    return {
        "seed_statistics": seed_statistics,
        "median_seed_statistic": statistics.median(seed_statistics),
        "seed_paired_circular_block_bootstrap_95pct_ci": [
            [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for values in seed_draws
        ],
        "equal_seed_mean": statistics.fmean(seed_statistics),
        "paired_circular_block_bootstrap_95pct_ci": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
    }


def _build_heldout_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    variants = (
        "identity",
        "dlinear",
        "var_direct",
        "dense",
        "random",
        "learned",
        "oracle",
        "learned_hard",
    )
    horizon_results = {}
    for horizon in range(6):
        horizon_results[str(horizon + 1)] = {
            variant: _paired_summary(runs, variant, "identity", horizon, ratio=True)
            for variant in variants
        }
        horizon_results[str(horizon + 1)]["oracle_to_var_direct"] = _paired_summary(
            runs, "oracle", "var_direct", horizon, ratio=True
        )
        horizon_results[str(horizon + 1)]["learned_to_oracle"] = _paired_summary(
            runs, "learned", "oracle", horizon, ratio=True
        )
        horizon_results[str(horizon + 1)]["learned_hard_to_oracle"] = _paired_summary(
            runs, "learned_hard", "oracle", horizon, ratio=True
        )
        for control in ("dense", "random"):
            horizon_results[str(horizon + 1)][f"learned_minus_{control}"] = _paired_summary(
                runs, "learned", control, horizon, ratio=False
            )
            horizon_results[str(horizon + 1)][
                f"learned_hard_minus_{control}"
            ] = _paired_summary(runs, "learned_hard", control, horizon, ratio=False)
    return {
        "protocol": str(PROTOCOL),
        "bootstrap": {
            "method": "paired circular moving-block percentile",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "seed_aggregation": "mean seed statistic, each seed equally weighted",
        },
        "seed_pairs": [
            [run["data_seed"], run["model_seed"]]
            for run in runs
        ],
        "per_seed_horizon_mse": {
            f"{run['data_seed']}_{run['model_seed']}": {
                variant: run["variants"][variant]["forecast"]["per_horizon_mse"]
                for variant in variants
            }
            for run in runs
        },
        "per_horizon": horizon_results,
        "primary_gate_horizons": [1, 2],
        "point_gates": {
            "oracle_within_5pct_of_var_direct": {
                horizon: row["oracle_to_var_direct"]["equal_seed_mean"] <= 1.05
                for horizon, row in horizon_results.items()
                if horizon in {"1", "2"}
            },
            "learned_within_5pct_of_oracle": {
                horizon: row["learned_to_oracle"]["equal_seed_mean"] <= 1.05
                for horizon, row in horizon_results.items()
                if horizon in {"1", "2"}
            },
            "learned_hard_within_5pct_of_oracle": {
                horizon: row["learned_hard_to_oracle"]["equal_seed_mean"] <= 1.05
                for horizon, row in horizon_results.items()
                if horizon in {"1", "2"}
            },
            "learned_beats_dense_by_paired_ci": {
                horizon: row["learned_minus_dense"][
                    "paired_circular_block_bootstrap_95pct_ci"
                ][1]
                < 0
                for horizon, row in horizon_results.items()
                if horizon in {"1", "2"}
            },
            "learned_beats_random_by_paired_ci": {
                horizon: row["learned_minus_random"][
                    "paired_circular_block_bootstrap_95pct_ci"
                ][1]
                < 0
                for horizon, row in horizon_results.items()
                if horizon in {"1", "2"}
            },
        },
    }


def _validate_frozen_commit() -> tuple[str, str, dict[str, str]]:
    manifest_path = FREEZE_MANIFEST
    if not manifest_path.exists():
        raise RuntimeError("frozen protocol manifest is missing; refusing held-out run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_protocol_hash = hashlib.sha256(REDUCED_PROTOCOL.read_bytes()).hexdigest()
    if manifest.get("protocol_sha256") != expected_protocol_hash:
        raise RuntimeError("reduced protocol differs from its frozen manifest")
    source_hashes = manifest.get("source_sha256", {})
    if not source_hashes:
        raise RuntimeError("frozen source hashes are missing")
    for relative_path, expected_hash in source_hashes.items():
        path = Path(relative_path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"frozen source hash mismatch: {relative_path}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path.cwd(),
        text=True,
    )
    if status.strip():
        raise RuntimeError("working tree is not clean after protocol freeze commit")
    return commit, expected_protocol_hash, source_hashes


def _run_oracle40_side_check(
    rates: dict[str, float],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    data_seed, model_seed = 37, 401
    directory = ROOT / "runs" / f"data{data_seed}_model{model_seed}"
    output_path = directory / "oracle40_side_check.json"
    protocol_hash = hashlib.sha256(REDUCED_PROTOCOL.read_bytes()).hexdigest()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("data_seed") != data_seed
            or existing.get("model_seed") != model_seed
            or existing.get("protocol_sha256") != protocol_hash
        ):
            raise ValueError(f"refusing to resume mismatched side check {output_path}")
        ledger["side_check"] = {
            "status": "complete",
            "output": str(output_path),
            "variant": "oracle",
            "epoch_cap": ORACLE_SIDE_CHECK_EPOCHS,
            "data_seed": data_seed,
            "model_seed": model_seed,
        }
        _write_json(CAMPAIGN_LEDGER, ledger)
        return existing
    data = _heldout_data(data_seed)
    checkpoint_path = directory / "checkpoints" / "oracle40_side_check.pt"
    graphs_path = directory / "graphs" / "oracle40_side_check.pt"
    result = _run_variant(
        "oracle",
        data,
        data_seed,
        model_seed,
        rates,
        0.5,
        checkpoint_path,
        graphs_path,
        ORACLE_SIDE_CHECK_EPOCHS,
        EARLY_STOPPING_PATIENCE,
    )
    result["side_check"] = "oracle_40_epoch_cap; excluded from primary variant and gates"
    payload = {
        "protocol": str(REDUCED_PROTOCOL),
        "protocol_sha256": protocol_hash,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "epoch_cap": ORACLE_SIDE_CHECK_EPOCHS,
        "result": result,
    }
    _write_json(output_path, payload)
    ledger["side_check"] = {
        "status": "complete",
        "output": str(output_path),
        "variant": "oracle",
        "epoch_cap": ORACLE_SIDE_CHECK_EPOCHS,
        "data_seed": data_seed,
        "model_seed": model_seed,
    }
    _write_json(CAMPAIGN_LEDGER, ledger)
    print(
        json.dumps(
            {
                "stage": "oracle40_side_check_complete",
                "data_seed": data_seed,
                "model_seed": model_seed,
                "pretraining": result["pretraining"],
                "elapsed_seconds": (
                    result["pretraining_elapsed_seconds"]
                    + result["online_adaptation"]["elapsed_seconds"]
                ),
                "output": str(output_path),
            }
        ),
        flush=True,
    )
    return payload


def _build_reduced_report(runs: list[dict[str, Any]], side_check: dict[str, Any]) -> dict[str, Any]:
    variants = ("identity", "dense", "learned", "oracle", "var_direct")
    per_seed: dict[str, Any] = {}
    variant_summary: dict[str, Any] = {}
    for run in runs:
        key = f"{run['data_seed']}_{run['model_seed']}"
        per_seed[key] = {}
        for variant in variants:
            result = run["variants"][variant]
            forecast = result["forecast"]
            per_seed[key][variant] = {
                "best_validation_epoch": result["pretraining"]["best_validation_epoch"],
                "epochs_executed": result["pretraining"]["epochs_executed"],
                "epoch_cap": result["pretraining"]["epoch_cap"],
                "convergence_status": result["pretraining"]["convergence_status"],
                "per_horizon_mean_mse": forecast["per_horizon_mse"][:2],
                "per_horizon_median_mse": [
                    statistics.median(row[horizon] for row in forecast["per_window_per_horizon_mse"])
                    for horizon in (0, 1)
                ],
            }
    for variant in variants:
        variant_summary[variant] = {}
        for horizon in (0, 1):
            seed_means = [
                run["variants"][variant]["forecast"]["per_horizon_mse"][horizon]
                for run in runs
            ]
            seed_medians = [
                statistics.median(
                    row[horizon]
                    for row in run["variants"][variant]["forecast"]["per_window_per_horizon_mse"]
                )
                for run in runs
            ]
            variant_summary[variant][f"H{horizon + 1}"] = {
                "per_seed_mean_mse": seed_means,
                "mean_of_seed_means": statistics.fmean(seed_means),
                "median_of_seed_means": statistics.median(seed_means),
                "per_seed_median_mse": seed_medians,
                "mean_of_seed_medians": statistics.fmean(seed_medians),
                "median_of_seed_medians": statistics.median(seed_medians),
            }

    contrasts: dict[str, Any] = {}
    for candidate, reference, label in (
        ("learned", "dense", "learned_vs_dense"),
        ("learned", "oracle", "learned_vs_oracle"),
        ("oracle", "var_direct", "oracle_vs_var_direct"),
    ):
        contrasts[label] = {}
        for horizon in (0, 1):
            contrasts[label][f"H{horizon + 1}"] = {
                "paired_mean_mse_difference": _paired_summary(
                    runs, candidate, reference, horizon, ratio=False
                ),
                "paired_mse_ratio": _paired_summary(
                    runs, candidate, reference, horizon, ratio=True
                ),
            }
    return {
        "protocol": str(REDUCED_PROTOCOL),
        "protocol_sha256": hashlib.sha256(REDUCED_PROTOCOL.read_bytes()).hexdigest(),
        "bootstrap": {
            "method": "paired circular moving-block percentile",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "seed_aggregation": "mean seed statistic, each seed equally weighted",
        },
        "seed_pairs": [[run["data_seed"], run["model_seed"]] for run in runs],
        "per_seed": per_seed,
        "variant_summary": variant_summary,
        "contrasts": contrasts,
        "primary_gates": {
            "oracle_over_var_direct_ratio_le_1_05_at_H1_H2": {
                f"H{horizon + 1}": contrasts["oracle_vs_var_direct"][
                    f"H{horizon + 1}"
                ]["paired_mse_ratio"]["equal_seed_mean"]
                <= 1.05
                for horizon in (0, 1)
            },
            "learned_over_oracle_ratio_le_1_05_at_H1_H2": {
                f"H{horizon + 1}": contrasts["learned_vs_oracle"][
                    f"H{horizon + 1}"
                ]["paired_mse_ratio"]["equal_seed_mean"]
                <= 1.05
                for horizon in (0, 1)
            },
            "learned_beats_dense_by_95pct_paired_interval_at_H1_H2": {
                f"H{horizon + 1}": contrasts["learned_vs_dense"][
                    f"H{horizon + 1}"
                ]["paired_mean_mse_difference"][
                    "paired_circular_block_bootstrap_95pct_ci"
                ][1]
                < 0
                for horizon in (0, 1)
            },
        },
        "oracle_40_epoch_side_check": side_check,
        "side_check_excluded_from_primary_gates": True,
    }


def _write_reduced_plot(report: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ("identity", "dense", "learned", "oracle", "var_direct")
    colors = ("#1f77b4", "#ff7f0e")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for variant_index, variant in enumerate(variants):
        for horizon_index in (0, 1):
            summary = report["variant_summary"][variant][f"H{horizon_index + 1}"]
            values = summary["per_seed_mean_mse"]
            identity_values = [
                run["variants"]["identity"]["forecast"]["per_horizon_mse"][horizon_index]
                for run in report["_runs"]
            ]
            ratios = [value / base for value, base in zip(values, identity_values, strict=True)]
            offset = (horizon_index - 0.5) * 0.15
            jitter = np.linspace(-0.035, 0.035, len(ratios))
            axis.scatter(
                [variant_index + offset + jitter[index] for index in range(len(ratios))],
                ratios,
                color=colors[horizon_index],
                label=f"H{horizon_index + 1} per-seed ratios" if variant_index == 0 else None,
                s=30,
            )
            mean = summary["mean_of_seed_means"] / statistics.fmean(identity_values)
            axis.scatter(variant_index + offset, mean, marker="D", color=colors[horizon_index], s=40)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    axis.set_ylabel("Per-horizon mean MSE ratio to identity")
    axis.set_title("D20 reduced campaign: three held-out seed pairs")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output = ROOT / "report" / "reduced_mse_ratios_to_identity.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def run_reduced_campaign() -> dict[str, Any]:
    commit, protocol_hash, source_hashes = _validate_frozen_commit()
    torch.set_num_threads(1)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    ledger_path = CAMPAIGN_LEDGER
    manifest_hash = hashlib.sha256(
        FREEZE_MANIFEST.read_bytes()
    ).hexdigest()
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (
            ledger.get("freeze_commit") != commit
            or ledger.get("freeze_manifest_sha256") != manifest_hash
        ):
            raise RuntimeError("campaign ledger belongs to a different frozen commit")
        previous_elapsed = float(ledger.get("elapsed_seconds", 0.0))
    else:
        previous_elapsed = 0.0
        ledger = {
            "freeze_commit": commit,
            "freeze_manifest_sha256": manifest_hash,
            "protocol_sha256": protocol_hash,
            "source_sha256": source_hashes,
            "started_at": started_at,
            "elapsed_seconds": 0.0,
            "execution": "sequential; torch single-thread",
            "runs": {},
        }
        _write_json(ledger_path, ledger)
    protocol = yaml.safe_load(REDUCED_PROTOCOL.read_text(encoding="utf-8"))
    rates = {
        "forecast": float(protocol["training"]["forecast_learning_rate"]),
        "graph": float(protocol["training"]["graph_learning_rate"]),
    }
    runs = []
    for pair in protocol["seeds"]["held_out_pairs"]:
        data_seed = int(pair["data"])
        model_seed = int(pair["model"])
        run = _run_heldout_pair(data_seed, model_seed, rates, 0.5, ledger)
        runs.append(run)
        ledger["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
        _write_json(ledger_path, ledger)
    side_check = _run_oracle40_side_check(rates, ledger)
    ledger["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
    ledger["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(ledger_path, ledger)
    report = _build_reduced_report(runs, side_check)
    report["campaign"] = {
        "freeze_commit": commit,
        "freeze_manifest_sha256": manifest_hash,
        "started_at": ledger["started_at"],
        "completed_at": ledger["completed_at"],
        "elapsed_seconds": ledger["elapsed_seconds"],
        "execution": "sequential; torch single-thread",
    }
    report["_runs"] = runs
    plot_path = _write_reduced_plot(report)
    report.pop("_runs")
    report["plot"] = str(plot_path)
    _write_json(ROOT / "report" / "reduced_summary_v2.json", report)
    return report


def _write_ratio_plot(report: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = (
        "identity",
        "dlinear",
        "var_direct",
        "dense",
        "random",
        "learned",
        "oracle",
        "learned_hard",
    )
    fig, axis = plt.subplots(figsize=(11, 5))
    colors = ("#1f77b4", "#ff7f0e")
    for variant_index, variant in enumerate(variants):
        for horizon_index, horizon in enumerate(("1", "2")):
            row = report["per_horizon"][horizon][variant]
            offset = (horizon_index - 0.5) * 0.16
            seed_ratios = row["seed_statistics"]
            jitter = np.linspace(-0.035, 0.035, len(seed_ratios))
            axis.scatter(
                [
                    variant_index + offset + jitter[index]
                    for index in range(len(seed_ratios))
                ],
                seed_ratios,
                color=colors[horizon_index],
                alpha=0.8,
                s=30,
                label=f"H{horizon} seeds" if variant_index == 0 else None,
            )
            mean = row["equal_seed_mean"]
            lower, upper = row["paired_circular_block_bootstrap_95pct_ci"]
            axis.errorbar(
                variant_index + offset,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt="D",
                color=colors[horizon_index],
                capsize=3,
                markersize=4,
            )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(variants)), variants, rotation=25, ha="right")
    axis.set_ylabel("Per-horizon MSE ratio to identity (lower is better)")
    axis.set_title("D20 equal-pretraining forecast comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output = ROOT / "report" / "d20_mse_ratios_to_identity.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def run_all() -> dict[str, Any]:
    raise RuntimeError(
        "Held-out execution is disabled pending approval of a reduced protocol."
    )


def run_development() -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    protocol = _configuration()
    model_seed = int(protocol["seeds"]["development_pair"]["model"])
    data = _development_data()
    lr_selection = _forecast_lr_selection(data, model_seed)
    graph_selection = _graph_lr_selection(data, model_seed)
    rates = {"forecast": lr_selection, "graph": graph_selection}

    threshold_result = {
        "status": "deferred",
        "reason": "hard-mask ablation is excluded from the reduced proposal",
    }

    gate = _headroom_gate(data, rates)
    result = {
        "protocol": str(PROTOCOL),
        "started_at": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "seed_pair": [31, model_seed],
        "forecast_learning_rate_selection": lr_selection,
        "graph_learning_rate_selection": graph_selection,
        "hard_mask_threshold": threshold_result,
        "headroom_gate": gate,
        "held_out_campaign_authorized_by_gate": bool(gate["passed"]),
    }
    _write_json(ROOT / "development" / "development_result.json", result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "reduced"), required=True)
    args = parser.parse_args()
    if args.phase == "reduced":
        result = run_reduced_campaign()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(ROOT / "report" / "reduced_summary_v2.json"),
                    "elapsed_seconds": result["campaign"]["elapsed_seconds"],
                    "freeze_commit": result["campaign"]["freeze_commit"],
                }
            ),
            flush=True,
        )
        return
    result = run_development()
    print(
        json.dumps(
            {
                "development_gate_passed": result["headroom_gate"]["passed"],
                "held_out_campaign_authorized_by_gate": False,
                "result": str(ROOT / "development" / "development_result.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
