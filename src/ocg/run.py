"""Command-line experiment runner for configured forecasting benchmarks."""

import argparse
import copy
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from ocg.baselines import DirectVAR, DLinear, NoGraphSSM, RevINLinear, TransformerLite
from ocg.data import generate_svar, load_csv, make_windows
from ocg.eval import (
    RECOVERY_REFERENCE_STEPS,
    SUSTAINED_RECOVERY_STEPS,
    detection_delay,
    efficiency_metrics,
    forecast_metrics,
    graph_metrics,
    regret_vs_oracle,
    recovery_time,
)
from ocg.graph import GraphLearner, RegimeDetector
from ocg.model import CausalSSM
from ocg.train import _regime_ids_from_graphs, _set_oracle_graph, train_offline, train_online


def _peak_cpu_memory_bytes() -> int:
    """Return the process peak resident memory in bytes on the current platform."""
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Require a YAML mapping at a named configuration location."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment config into a mapping."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return _mapping(config, "configuration")


def _load_series(
    data_config: dict[str, Any],
    data_path: str | Path | None,
) -> tuple[Tensor, Tensor | None, int | None]:
    """Load a (T,D) series, optional (T,K,D,D) truth, and switch index."""
    data_type = data_config.get("type")
    if data_type == "synthetic":
        n_steps = int(data_config["n_steps"])
        switch_at = int(data_config["switch_at"])
        result = generate_svar(
            n_steps=n_steps,
            n_variables=int(data_config["n_variables"]),
            n_lags=int(data_config["n_lags"]),
            switch_at=switch_at,
            blend_steps=int(data_config.get("blend_steps", 0)),
            sparsity=float(data_config["sparsity"]),
            noise_scale=float(data_config["noise_scale"]),
            seed=int(data_config["seed"]),
            cross_variable_scale=float(data_config.get("cross_variable_scale", 1.0)),
        )
        return result["series"], result["graph"], switch_at
    if data_type == "csv":
        source = data_path if data_path is not None else data_config.get("csv_path")
        if source is None:
            raise ValueError("CSV data requires data.csv_path or --data-path")
        columns = data_config.get("columns")
        if columns is not None and (
            not isinstance(columns, list) or not all(isinstance(item, str) for item in columns)
        ):
            raise ValueError("data.columns must be a list of column names")
        return load_csv(source, columns), None, None
    raise ValueError("data.type must be synthetic or csv")


def _split_windows(
    series: Tensor,
    input_length: int,
    horizon: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor], int]:
    """Split series chronologically, returning windows and the test start index."""
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("train_fraction must be in (0,1) and validation_fraction in [0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than one")
    train_end = int(series.shape[0] * train_fraction)
    test_start = int(series.shape[0] * (train_fraction + validation_fraction))
    minimum = input_length + horizon
    if train_end < minimum or test_start - train_end < minimum or series.shape[0] - test_start < minimum:
        raise ValueError("each chronological split must fit input_length + horizon values")
    train = make_windows(series[:train_end], input_length, horizon)
    validation = make_windows(series[train_end:test_start], input_length, horizon)
    test = make_windows(series[test_start:], input_length, horizon)
    return train, validation, test, test_start


def _build_model(
    config: dict[str, Any],
    n_variables: int,
    horizon: int,
    n_lags: int,
    true_graph: Tensor | None,
    seed: int,
    random_sparsity: float | None = None,
) -> nn.Module:
    """Construct a baseline or OCG model for (B,L,D), using random density p."""
    name = config.get("name")
    if name == "dlinear":
        return DLinear(
            int(config["input_length"]),
            horizon,
            n_variables,
            int(config["moving_average"]),
        )
    if name == "revin_linear":
        return RevINLinear(int(config["input_length"]), horizon, n_variables)
    if name == "var_direct":
        return DirectVAR(n_lags, horizon, n_variables)
    if name == "var_iterated":
        return DirectVAR(n_lags, horizon, n_variables, iterated=True)
    if name == "transformer_lite":
        return TransformerLite(
            int(config["input_length"]),
            horizon,
            n_variables,
            int(config["model_dim"]),
            int(config["n_heads"]),
        )
    if name == "no_graph_ssm":
        return NoGraphSSM(
            n_variables,
            int(config["state_dim"]),
            horizon,
            n_lags,
            int(config["n_layers"]),
            int(config["embedding_dim"]),
        )
    if name == "ocg":
        graph_mode = str(config["graph_mode"])
        oracle_graph = true_graph[0] if graph_mode == "oracle" and true_graph is not None else None
        graph_config = _mapping(config.get("graph", {}), "model.graph")
        graph_learner = None
        if graph_mode == "learned":
            detector = RegimeDetector(
                decay=float(graph_config["detector_decay"]),
                threshold=float(graph_config["detector_threshold"]),
                warmup=int(graph_config["detector_warmup"]),
                min_std=float(graph_config["detector_min_std"]),
            )
            graph_learner = GraphLearner(
                n_variables=n_variables,
                n_lags=n_lags,
                learning_rate=float(graph_config["learning_rate"]),
                sparsity_weight=float(graph_config["sparsity_weight"]),
                temperature=float(graph_config["temperature"]),
                ema_decay=float(graph_config["ema_decay"]),
                graph_stride=int(graph_config["graph_stride"]),
                detector=detector,
                detector_confirm_steps=int(
                    graph_config.get("detector_confirm_steps", 1)
                ),
                detector_rearm_steps=int(
                    graph_config.get("detector_rearm_steps", 10)
                ),
                boost=float(graph_config["boost"]),
                boost_steps=int(graph_config["boost_steps"]),
            )
        return CausalSSM(
            n_variables=n_variables,
            state_dim=int(config["state_dim"]),
            horizon=horizon,
            n_lags=n_lags,
            n_layers=int(config["n_layers"]),
            embedding_dim=int(config["embedding_dim"]),
            graph_mode=graph_mode,
            graph_learner=graph_learner,
            oracle_graph=oracle_graph,
            random_seed=seed,
            random_sparsity=random_sparsity,
            use_direct_lag_head=bool(config.get("use_direct_lag_head", True)),
            use_ssm_coupling=bool(config.get("use_ssm_coupling", True)),
            direct_head_mode=str(config.get("direct_head_mode", "lag")),
            zero_init_ssm_head=bool(config.get("zero_init_ssm_head", False)),
            graph_mask_threshold=(
                float(config["graph_mask_threshold"])
                if config.get("graph_mask_threshold") is not None
                else None
            ),
        )
    raise ValueError(f"unsupported model name: {name!r}")


def _predict_timed(
    model: nn.Module,
    inputs: Tensor,
    oracle_graphs: Tensor | None = None,
    capture_forecast_components: bool = False,
) -> tuple[Tensor, list[float], list[dict[str, list[float]]]]:
    """Predict each window individually and time each (1,L,D) inference step."""
    if oracle_graphs is not None and oracle_graphs.shape[0] != inputs.shape[0]:
        raise ValueError("oracle_graphs must align with the number of input windows")
    model.eval()
    predictions: list[Tensor] = []
    step_times: list[float] = []
    component_rms: list[dict[str, list[float]]] = []
    with torch.no_grad():
        for index in range(inputs.shape[0]):
            if oracle_graphs is not None:
                _set_oracle_graph(model, oracle_graphs[index])
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            started = time.perf_counter()
            window = inputs[index : index + 1]
            component_forward = getattr(model, "forward_with_components", None)
            if capture_forecast_components and callable(component_forward):
                prediction, components = component_forward(window)
                component_rms.append(
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
            predictions.append(prediction.detach())
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            step_times.append(time.perf_counter() - started)
    return torch.cat(predictions, dim=0), step_times, component_rms


def _model_variants(model_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a single model config or explicit named ablation variants."""
    variants = model_config.get("variants")
    if variants is None:
        return [{"id": str(model_config.get("name", "model")), **model_config}]
    if not isinstance(variants, list) or not variants:
        raise ValueError("model.variants must be a nonempty list")
    shared = {key: value for key, value in model_config.items() if key != "variants"}
    output: list[dict[str, Any]] = []
    ids: set[str] = set()
    for variant in variants:
        item = _mapping(variant, "model variant")
        variant_id = item.get("id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in ids:
            raise ValueError("each model variant needs a unique nonempty string id")
        ids.add(variant_id)
        merged = {**shared, **item}
        if isinstance(shared.get("graph"), dict) and isinstance(item.get("graph"), dict):
            merged["graph"] = {**shared["graph"], **item["graph"]}
        output.append(merged)
    return output


def _select_graph_threshold(
    predicted_graphs: Tensor,
    true_graphs: Tensor,
) -> tuple[float, float]:
    """Select an edge threshold maximizing pooled validation F1; ties prefer higher thresholds."""
    if (
        predicted_graphs.shape != true_graphs.shape
        or predicted_graphs.ndim != 4
        or predicted_graphs.shape[0] < 1
    ):
        raise ValueError("graph batches must match with shape (N,K,D,D)")
    scores = predicted_graphs.detach().cpu().reshape(-1)
    labels = true_graphs.detach().cpu().reshape(-1) >= 0.5
    if not torch.isfinite(scores).all() or not torch.isfinite(true_graphs).all():
        raise ValueError("validation graph batches must contain finite values")
    if torch.any((scores < 0) | (scores > 1)):
        raise ValueError("graph probabilities must be in [0,1]")
    if not labels.any():
        raise ValueError("validation graphs must contain at least one positive edge")
    candidates = torch.unique(torch.cat((scores, torch.tensor([0.0, 1.0])))).tolist()
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int((predicted & labels).sum().item())
        false_positive = int((predicted & ~labels).sum().item())
        false_negative = int((~predicted & labels).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold, best_f1


def run_experiment(
    config_path: str | Path,
    output_path: str | Path | None = None,
    data_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run configured variants on a device and return their evaluation metrics."""
    config = _load_config(config_path)
    if "cpu_threads" in config:
        cpu_threads = int(config["cpu_threads"])
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    seed = int(config["seed"])
    data_config = _mapping(config["data"], "data")
    model_config = _mapping(config["model"], "model")
    training_config = _mapping(config["training"], "training")
    series, graph, switch_at = _load_series(data_config, data_path)
    selected_device = torch.device(device or config.get("device", "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    input_length = int(data_config["input_length"])
    horizon = int(data_config["horizon"])
    n_lags = int(data_config.get("n_lags", 1))
    train, validation, test, test_start = _split_windows(
        series,
        input_length,
        horizon,
        float(data_config["train_fraction"]),
        float(data_config["validation_fraction"]),
    )
    train_inputs, train_targets = train
    validation_inputs, validation_targets = validation
    test_inputs, test_targets = test
    train_inputs = train_inputs.to(selected_device)
    train_targets = train_targets.to(selected_device)
    validation_inputs = validation_inputs.to(selected_device)
    validation_targets = validation_targets.to(selected_device)
    test_inputs = test_inputs.to(selected_device)
    test_targets = test_targets.to(selected_device)
    if graph is not None:
        graph = graph.to(selected_device)
    train_end = int(series.shape[0] * float(data_config["train_fraction"]))
    test_graph = (
        graph[
            test_start + input_length - 1 :
            test_start + input_length - 1 + test_inputs.shape[0]
        ]
        if graph is not None
        else None
    )
    train_graphs = (
        graph[input_length - 1 : input_length - 1 + train_inputs.shape[0]]
        if graph is not None
        else None
    )
    validation_graphs = (
        graph[
            train_end + input_length - 1 :
            train_end + input_length - 1 + validation_inputs.shape[0]
        ]
        if graph is not None
        else None
    )
    mode = str(training_config["mode"])
    if mode not in {"offline", "online"}:
        raise ValueError("training.mode must be offline or online")
    online_start = str(training_config.get("online_start", "cold_start"))
    if mode == "online" and online_start not in {"cold_start", "warm_start"}:
        raise ValueError("training.online_start must be cold_start or warm_start")
    evaluation_timestamps = list(
        range(test_start + input_length - 1, test_start + input_length - 1 + len(test_inputs))
    )

    reports: dict[str, Any] = {}
    predictions_by_variant: dict[str, Tensor] = {}
    for variant in _model_variants(model_config):
        variant_id = str(variant["id"])
        variant_seed = seed
        variant.setdefault("input_length", input_length)
        torch.manual_seed(variant_seed)
        model = _build_model(
            variant,
            int(series.shape[1]),
            horizon,
            n_lags,
            (
                graph
                if variant.get("graph_mode") == "oracle"
                else None
            ),
            variant_seed,
            (
                float(data_config["sparsity"])
                if "sparsity" in data_config
                else None
            ),
        ).to(selected_device)
        selected_graph_threshold = None
        validation_graph_f1 = None
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        if mode == "offline":
            is_oracle = variant.get("graph_mode") == "oracle"
            training_result = train_offline(
                model,
                train_inputs,
                train_targets,
                epochs=int(training_config["epochs"]),
                batch_size=int(training_config["batch_size"]),
                learning_rate=float(training_config["learning_rate"]),
                seed=variant_seed,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                oracle_graphs=train_graphs if is_oracle else None,
                regime_ids=(
                    _regime_ids_from_graphs(train_graphs)
                    if is_oracle and train_graphs is not None
                    else None
                ),
                validation_oracle_graphs=validation_graphs if is_oracle else None,
                validation_regime_ids=(
                    _regime_ids_from_graphs(validation_graphs)
                    if is_oracle and validation_graphs is not None
                    else None
                ),
                restore_best_validation=bool(
                    training_config.get("restore_best_validation", False)
                ),
            )
            predictions, step_times, component_rms = _predict_timed(
                model,
                test_inputs,
                test_graph if is_oracle else None,
                capture_forecast_components=bool(
                    training_config.get("capture_forecast_components", False)
                ),
            )
            if component_rms:
                training_result["forecast_component_rms"] = component_rms
        else:
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            started = time.perf_counter()
            warm_start_result = None
            detector_calibration = None
            if online_start == "warm_start":
                warm_start_result = train_online(
                    model,
                    train_inputs,
                    train_targets,
                    learning_rate=float(training_config["learning_rate"]),
                    seed=variant_seed,
                    oracle_graphs=(
                        train_graphs if variant.get("graph_mode") == "oracle" else None
                    ),
                    timestamps=list(
                        range(input_length - 1, input_length - 1 + len(train_inputs))
                    ),
                    update_graph=bool(variant.get("online_graph_updates", True)),
                )
                if (
                    isinstance(model, CausalSSM)
                    and model.graph_learner is not None
                    and len(validation_inputs) >= 2
                ):
                    calibration_learner = copy.deepcopy(model.graph_learner)
                    calibration_learner.detector.threshold = float("inf")
                    calibration_scores = []
                    validation_graph_snapshots = []
                    for validation_window in validation_inputs:
                        validation_graph_snapshots.append(
                            calibration_learner.adjacency().detach().cpu().clone()
                        )
                        calibration_scores.append(
                            float(
                                calibration_learner.online_update(
                                    validation_window.unsqueeze(0),
                                    update_detector=False,
                                )
                            )
                        )
                    if validation_graphs is not None:
                        selected_graph_threshold, validation_graph_f1 = (
                            _select_graph_threshold(
                                torch.stack(validation_graph_snapshots),
                                validation_graphs,
                            )
                        )
                    detector_calibration = model.graph_learner.calibrate_detector(
                        calibration_scores
                    )
                    validation_z_scores = [
                        (score - detector_calibration["median"])
                        / detector_calibration["robust_std"]
                        for score in calibration_scores
                    ]
                    validation_max_z = max(validation_z_scores)
                    validation_selected_threshold = max(
                        model.graph_learner.detector.threshold,
                        validation_max_z + 0.25,
                    )
                    model.graph_learner.detector.threshold = (
                        validation_selected_threshold
                    )
                    detector_calibration.update(
                        {
                            "threshold": validation_selected_threshold,
                            "validation_max_zscore": validation_max_z,
                            "confirm_steps": model.graph_learner.detector_confirm_steps,
                            "rearm_steps": model.graph_learner.detector_rearm_steps,
                            "validation_false_alarms": 0,
                        }
                    )
            if (
                selected_graph_threshold is None
                and isinstance(model, CausalSSM)
                and model.graph_learner is not None
                and validation_graphs is not None
            ):
                threshold_learner = copy.deepcopy(model.graph_learner)
                threshold_snapshots = []
                for validation_window in validation_inputs:
                    threshold_snapshots.append(
                    threshold_learner.adjacency().detach().cpu().clone()
                    )
                    threshold_learner.online_update(
                    validation_window.unsqueeze(0),
                    update_detector=False,
                    )
                selected_graph_threshold, validation_graph_f1 = (
                    _select_graph_threshold(
                    torch.stack(threshold_snapshots),
                    validation_graphs,
                    )
                )
            training_result = train_online(
                model,
                test_inputs,
                test_targets,
                learning_rate=float(training_config["learning_rate"]),
                seed=variant_seed,
                oracle_graphs=(
                    test_graph if variant.get("graph_mode") == "oracle" else None
                ),
                timestamps=evaluation_timestamps,
                update_graph=bool(variant.get("online_graph_updates", True)),
                freeze_detector=detector_calibration is not None,
                capture_forecast_components=bool(
                    training_config.get("capture_forecast_components", False)
                ),
            )
            if warm_start_result is not None:
                training_result["warm_start"] = {
                    "mode": "train_only",
                    "windows": len(train_inputs),
                    "final_running_mse": warm_start_result["running_mse"][-1],
                }
                if detector_calibration is not None:
                    training_result["warm_start"]["detector_calibration"] = {
                        **detector_calibration,
                        "source": "validation structural scores; no validation updates",
                        "statistics_frozen_during_test": True,
                    }
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            elapsed = time.perf_counter() - started
            predictions = torch.tensor(
                training_result["predictions"],
                dtype=test_targets.dtype,
                device=selected_device,
            )
            step_times = [elapsed / test_inputs.shape[0]] * test_inputs.shape[0]
        predictions_by_variant[variant_id] = predictions

        report: dict[str, Any] = {
            "mode": mode,
            "protocol": {
                "start": online_start if mode == "online" else "offline_train",
                "warm_start_windows": len(train_inputs)
                if mode == "online" and online_start == "warm_start"
                else 0,
                "online_graph_updates": bool(
                    variant.get("online_graph_updates", True)
                )
                if mode == "online"
                else None,
            },
            "forecast": forecast_metrics(predictions, test_targets),
            "training": {
                key: value
                for key, value in training_result.items()
                if key not in {
                    "predictions",
                    "graph_history",
                    "graph_timestamps",
                    "graph_scores",
                    "detections",
                    "detection_events",
                }
            },
            "efficiency": efficiency_metrics(
                step_times,
                (
                    torch.cuda.max_memory_allocated(selected_device)
                    if selected_device.type == "cuda"
                    else _peak_cpu_memory_bytes()
                ),
                model,
            ),
        }
        report["forecast"]["per_window_mse"] = (
            (predictions - test_targets).square().mean(dim=(1, 2)).detach().cpu().tolist()
        )
        report["forecast"]["per_window_per_horizon_mse"] = (
            (predictions - test_targets)
            .square()
            .mean(dim=2)
            .detach()
            .cpu()
            .tolist()
        )
        report["forecast"]["per_horizon_mse"] = (
            (predictions - test_targets)
            .square()
            .mean(dim=(0, 2))
            .detach()
            .cpu()
            .tolist()
        )
        report["forecast"]["per_horizon_mae"] = (
            (predictions - test_targets)
            .abs()
            .mean(dim=(0, 2))
            .detach()
            .cpu()
            .tolist()
        )
        report["forecast"]["input_timestamps"] = evaluation_timestamps

        learner = getattr(model, "graph_learner", None)
        if graph is not None and isinstance(model, CausalSSM):
            if model.graph_mode == "oracle":
                history = test_graph
            elif learner is not None and mode == "online":
                history = torch.tensor(training_result["graph_history"])
            elif learner is not None:
                history = learner.adjacency().detach().unsqueeze(0).expand(
                    len(test_graph), -1, -1, -1
                )
            else:
                fixed = model._lagged_adjacency(test_inputs).detach()
                history = fixed.unsqueeze(0).expand(len(test_graph), -1, -1, -1)
            graph_timestamps = (
                training_result["graph_timestamps"]
                if learner is not None and mode == "online"
                else evaluation_timestamps
            )
            metric_threshold = (
                selected_graph_threshold
                if selected_graph_threshold is not None
                else 0.5
            )
            per_step_graph = [
                graph_metrics(
                    history[index],
                    test_graph[index],
                    threshold=metric_threshold,
                )
                for index in range(len(test_graph))
            ]
            report["graph"] = {
                **per_step_graph[-1],
                "reference_timestamp": graph_timestamps[-1],
                "timestamps": graph_timestamps,
                "per_step": per_step_graph,
                "metric_threshold": metric_threshold,
                "metric_threshold_source": (
                    "validation-only pooled F1; tie-breaking selects highest threshold"
                    if selected_graph_threshold is not None
                    else "fixed default 0.5"
                ),
            }
            if selected_graph_threshold is not None:
                report["graph"]["validation_selected_f1"] = validation_graph_f1
            if learner is not None and mode == "online":
                report["graph"]["structural_scores"] = training_result[
                    "graph_scores"
                ]
                report["graph"]["snapshots"] = history.detach().cpu().tolist()
                report["graph"]["snapshot_semantics"] = (
                    "EMA adjacency used for each forecast before that step's graph update"
                )
        if (
            graph is not None
            and mode == "online"
            and learner is not None
            and test_graph is not None
        ):
            detection_value = training_result.get("detections")
            if detection_value is not None:
                history = torch.tensor(training_result["graph_history"])
                history_cpu = history.detach().cpu()
                test_graph_cpu = test_graph.detach().cpu()
                step_shd = [
                    float(
                        (
                            (history_cpu[index] >= 0.5)
                            != (test_graph_cpu[index] >= 0.5)
                        )
                        .sum()
                        .item()
                    )
                    for index in range(history.shape[0])
                ]
                if switch_at is not None:
                    switch_index = switch_at - evaluation_timestamps[0]
                    if 0 <= switch_index < len(step_shd):
                        report["adaptation"] = {
                            "detection_delay": detection_delay(
                                detection_value, switch_index
                            )
                        }
                        if switch_index > 0:
                            reference_start = len(step_shd) - RECOVERY_REFERENCE_STEPS
                            reference_shd = (
                                float(
                                    torch.tensor(
                                        step_shd[reference_start:],
                                        dtype=torch.float64,
                                    )
                                    .median()
                                    .item()
                                )
                                if reference_start >= switch_index
                                else None
                            )
                            report["adaptation"]["sustained_recovery_time"] = (
                                recovery_time(
                                    step_shd,
                                    switch_index,
                                    sustain_steps=SUSTAINED_RECOVERY_STEPS,
                                    reference_steps=RECOVERY_REFERENCE_STEPS,
                                )
                            )
                            report["adaptation"]["recovery_sustain_steps"] = (
                                SUSTAINED_RECOVERY_STEPS
                            )
                            report["adaptation"]["recovery_reference_steps"] = (
                                RECOVERY_REFERENCE_STEPS
                            )
                            report["adaptation"]["recovery_reference_shd_median"] = (
                                reference_shd
                            )
                            report["adaptation"]["recovery_reference_scope"] = (
                                "retrospective evaluation-only post-switch tail; "
                                "not used for training or detection"
                            )
                            report["adaptation"]["recovery_time"] = (
                                report["adaptation"]["sustained_recovery_time"]
                            )
                report["graph"]["online_final_shd"] = int(report["graph"]["shd"])
                report["graph"]["detection_events"] = training_result[
                    "detection_events"
                ]
        reports[variant_id] = report

    oracle_variant = next(
        (
            variant
            for variant in _model_variants(model_config)
            if variant.get("name") == "ocg" and variant.get("graph_mode") == "oracle"
        ),
        None,
    )
    if oracle_variant is not None:
        oracle_predictions = predictions_by_variant[str(oracle_variant["id"])]
        for variant_id, predictions in predictions_by_variant.items():
            reports[variant_id]["regret_vs_oracle"] = regret_vs_oracle(
                predictions, oracle_predictions, test_targets
            )

    result: dict[str, Any] = {
        "config": str(config_path),
        "seed": seed,
        "device": str(selected_device),
        "cpu_threads": torch.get_num_threads(),
        "evaluation": {
            "switch_at": switch_at,
            "test_start": test_start,
            "input_length": input_length,
            "test_window_count": int(test_inputs.shape[0]),
            "online_start": online_start if mode == "online" else None,
            "graph_timestamp_alignment": "last input observation",
            "forecast_target_timestamp_ranges": [
                [timestamp + 1, timestamp + horizon]
                for timestamp in evaluation_timestamps
            ],
        },
        "variants": reports,
    }
    destination = Path(output_path) if output_path is not None else Path("results") / f"{Path(config_path).stem}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, allow_nan=False)
        result_file.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, execute an experiment, and print its result path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config")
    parser.add_argument("--output", help="JSON output path (default: results/<config>.json)")
    parser.add_argument("--data-path", help="Override the CSV path configured for real datasets")
    parser.add_argument("--device", help="Torch device, such as cpu or cuda")
    arguments = parser.parse_args(argv)
    result = run_experiment(
        arguments.config, arguments.output, arguments.data_path, arguments.device
    )
    print(json.dumps({"seed": result["seed"], "variants": list(result["variants"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
