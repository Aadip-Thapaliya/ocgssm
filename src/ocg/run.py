"""Command-line experiment runner for configured forecasting benchmarks."""

import argparse
import json
import math
import resource
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from ocg.baselines import DLinear, NoGraphSSM, RevINLinear, TransformerLite
from ocg.data import generate_svar, load_csv, make_windows
from ocg.eval import (
    detection_delay,
    efficiency_metrics,
    forecast_metrics,
    graph_metrics,
    regret_vs_oracle,
    recovery_time,
)
from ocg.graph import GraphLearner, RegimeDetector
from ocg.model import CausalSSM
from ocg.train import (
    _regime_ids_from_graphs,
    _set_oracle_graph,
    train_offline,
    train_online,
    warm_start_graph,
)


def _calibrated_graph_threshold(
    predicted_graph: Tensor,
    validation_true_graph: Tensor,
) -> float:
    """Select threshold in [0,1] maximizing validation F1 for (K,D,D) graphs."""
    candidate_thresholds = torch.linspace(
        0.05,
        0.95,
        steps=91,
        device=predicted_graph.device,
        dtype=predicted_graph.dtype,
    )
    best_threshold = 0.5
    best_f1 = float("-inf")
    best_shd = float("inf")
    for threshold_tensor in candidate_thresholds:
        threshold = float(threshold_tensor.item())
        metrics = graph_metrics(predicted_graph, validation_true_graph, threshold=threshold)
        f1 = float(metrics["f1"])
        shd = float(metrics["shd"])
        if f1 > best_f1 or (f1 == best_f1 and shd < best_shd):
            best_f1 = f1
            best_shd = shd
            best_threshold = threshold
    return best_threshold


def _density_prior_threshold(predicted_graph: Tensor, density: float) -> float:
    """Select threshold so predicted (K,D,D) edge count matches a target density."""
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0,1]")
    scores = predicted_graph.reshape(-1)
    edge_count = int(round(density * scores.numel()))
    if edge_count <= 0:
        return 1.0
    if edge_count >= scores.numel():
        return 0.0
    kth_value = torch.topk(scores, k=edge_count, largest=True).values[-1]
    return float(kth_value.item())


DEFAULT_GRAD_CLIP_NORM = 1.0


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Require a YAML mapping at a named configuration location."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _sanitize_non_finite(value: Any) -> tuple[Any, bool]:
    """Recursively replace non-finite floats with None and report if any were found."""
    if isinstance(value, float) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None, True
        return value, False
    if isinstance(value, dict):
        sanitized_dict: dict[str, Any] = {}
        found_non_finite = False
        for key, item in value.items():
            sanitized_item, item_non_finite = _sanitize_non_finite(item)
            sanitized_dict[key] = sanitized_item
            found_non_finite = found_non_finite or item_non_finite
        return sanitized_dict, found_non_finite
    if isinstance(value, list):
        sanitized_list: list[Any] = []
        found_non_finite = False
        for item in value:
            sanitized_item, item_non_finite = _sanitize_non_finite(item)
            sanitized_list.append(sanitized_item)
            found_non_finite = found_non_finite or item_non_finite
        return sanitized_list, found_non_finite
    return value, False


def _grad_clip_norm(training_config: dict[str, Any]) -> float | None:
    """Return the gradient-norm clip (default 1.0); `grad_clip_norm: null` disables it."""
    value = training_config.get("grad_clip_norm", DEFAULT_GRAD_CLIP_NORM)
    return None if value is None else float(value)


DIVERGENCE_RELATIVE_FACTOR = 10.0
DIVERGENCE_EPOCH_FACTOR = 100.0


def _finite_float(value: Any) -> float | None:
    """Return value as a finite float, or None if missing or non-finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _divergence_reasons(report: dict[str, Any], identity_report: dict[str, Any] | None) -> list[str]:
    """List finite-blow-up reasons for one variant report, relative to the identity report.

    Flags test or final validation MSE above 10x identity's, and any epoch
    (or online running) loss above 100x the first one. Non-finite values are
    handled separately by the caller.
    """
    reasons: list[str] = []
    training = report.get("training", {})
    if identity_report is not None and identity_report is not report:
        identity_training = identity_report.get("training", {})
        test_mse = _finite_float(report.get("forecast", {}).get("mse"))
        identity_test = _finite_float(identity_report.get("forecast", {}).get("mse"))
        if test_mse is not None and identity_test is not None and identity_test > 0:
            if test_mse > DIVERGENCE_RELATIVE_FACTOR * identity_test:
                reasons.append("test_mse_exceeds_10x_identity")
        validation = training.get("validation_mse") or []
        identity_validation = identity_training.get("validation_mse") or []
        if validation and identity_validation:
            validation_mse = _finite_float(validation[-1])
            identity_val = _finite_float(identity_validation[-1])
            if validation_mse is not None and identity_val is not None and identity_val > 0:
                if validation_mse > DIVERGENCE_RELATIVE_FACTOR * identity_val:
                    reasons.append("validation_mse_exceeds_10x_identity")
    losses = training.get("train_mse") or training.get("running_mse") or []
    first = _finite_float(losses[0]) if losses else None
    if first is not None and first > 0:
        for loss in losses[1:]:
            number = _finite_float(loss)
            if number is not None and number > DIVERGENCE_EPOCH_FACTOR * first:
                reasons.append("epoch_loss_exceeds_100x_first_epoch")
                break
    return reasons


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
            scan_implementation=str(config.get("scan_implementation", "fast")),
            coupling_init_scale=float(config.get("coupling_init_scale", 1.0)),
            coupling_norm_bound=float(config.get("coupling_norm_bound", 1.0)),
        )
    raise ValueError(f"unsupported model name: {name!r}")


def _predict_timed(
    model: nn.Module,
    inputs: Tensor,
    oracle_graphs: Tensor | None = None,
) -> tuple[Tensor, list[float]]:
    """Predict each window individually and time each (1,L,D) inference step."""
    if oracle_graphs is not None and oracle_graphs.shape[0] != inputs.shape[0]:
        raise ValueError("oracle_graphs must align with the number of input windows")
    model.eval()
    predictions: list[Tensor] = []
    step_times: list[float] = []
    with torch.no_grad():
        for index in range(inputs.shape[0]):
            if oracle_graphs is not None:
                _set_oracle_graph(model, oracle_graphs[index])
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            started = time.perf_counter()
            predictions.append(model(inputs[index : index + 1]).detach())
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            step_times.append(time.perf_counter() - started)
    return torch.cat(predictions, dim=0), step_times


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
        output.append({**shared, **item})
    return output


def run_experiment(
    config_path: str | Path,
    output_path: str | Path | None = None,
    data_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run configured variants on a device and return their evaluation metrics."""
    config = _load_config(config_path)
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

    reports: dict[str, Any] = {}
    predictions_by_variant: dict[str, Tensor] = {}
    for variant in _model_variants(model_config):
        variant_id = str(variant["id"])
        variant_seed = seed
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
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        if mode == "offline":
            is_oracle = variant.get("graph_mode") == "oracle"
            is_learned = variant.get("graph_mode") == "learned"
            warm_start_result: dict[str, Any] | None = None
            if is_learned:
                graph_config = _mapping(variant.get("graph", {}), "model.graph")
                warm_start_epochs = int(graph_config.get("warm_start_epochs", 5))
                warm_start_batch_size = int(
                    graph_config.get("warm_start_batch_size", training_config["batch_size"])
                )
                warm_start_result = warm_start_graph(
                    model,
                    train_inputs,
                    epochs=warm_start_epochs,
                    batch_size=warm_start_batch_size,
                    seed=variant_seed,
                )
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
                optimizer_name=str(training_config.get("optimizer", "adam")),
                weight_decay=float(training_config.get("weight_decay", 0.0)),
                grad_clip_norm=_grad_clip_norm(training_config),
                lr_schedule=str(training_config.get("lr_schedule", "none")),
                min_learning_rate=float(training_config.get("min_learning_rate", 0.0)),
                early_stopping_patience=(
                    int(training_config["early_stopping_patience"])
                    if "early_stopping_patience" in training_config
                    else None
                ),
            )
            if warm_start_result is not None:
                training_result.update(warm_start_result)
            predictions, step_times = _predict_timed(
                model,
                test_inputs,
                test_graph if is_oracle else None,
            )
        else:
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            started = time.perf_counter()
            training_result = train_online(
                model,
                test_inputs,
                test_targets,
                learning_rate=float(training_config["learning_rate"]),
                seed=variant_seed,
                oracle_graphs=(
                    test_graph if variant.get("graph_mode") == "oracle" else None
                ),
                return_tensors=True,
                optimizer_name=str(training_config.get("optimizer", "adam")),
                weight_decay=float(training_config.get("weight_decay", 0.0)),
                grad_clip_norm=_grad_clip_norm(training_config),
            )
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            elapsed = time.perf_counter() - started
            prediction_tensor = training_result.get("predictions_tensor")
            if not isinstance(prediction_tensor, Tensor):
                raise TypeError("online training did not return predictions_tensor")
            predictions = prediction_tensor.to(
                device=selected_device, dtype=test_targets.dtype
            )
            step_times = [elapsed / test_inputs.shape[0]] * test_inputs.shape[0]
        predictions_by_variant[variant_id] = predictions

        report: dict[str, Any] = {
            "mode": mode,
            "forecast": forecast_metrics(predictions, test_targets),
            "training": {
                key: value
                for key, value in training_result.items()
                if key
                not in {
                    "predictions",
                    "graph_history",
                    "detections",
                    "predictions_tensor",
                    "prequential_mse_tensor",
                }
            },
            "efficiency": efficiency_metrics(
                step_times,
                (
                    torch.cuda.max_memory_allocated(selected_device)
                    if selected_device.type == "cuda"
                    else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
                model,
            ),
        }
        report["forecast"]["per_window_mse"] = (
            (predictions - test_targets).square().mean(dim=(1, 2)).detach().cpu().tolist()
        )

        learner = getattr(model, "graph_learner", None)
        if graph is not None:
            truth_graph = test_graph.mean(dim=0)
            graph_threshold = 0.5
            graph_options = _mapping(variant.get("graph", {}), "model.graph")
            calibration_mode = str(graph_options.get("threshold_calibration", "density_prior"))
            if learner is not None and validation_graphs is not None:
                if calibration_mode == "best_f1":
                    graph_threshold = _calibrated_graph_threshold(
                        learner.adjacency().detach(),
                        validation_graphs.mean(dim=0),
                    )
                elif calibration_mode == "density_prior":
                    graph_threshold = _density_prior_threshold(
                        learner.adjacency().detach(),
                        float(data_config.get("sparsity", 0.5)),
                    )
                elif calibration_mode != "fixed_0.5":
                    raise ValueError(
                        "model.graph.threshold_calibration must be best_f1, density_prior, or fixed_0.5"
                    )
            if learner is not None:
                report["graph"] = graph_metrics(
                    learner.adjacency().detach(),
                    truth_graph,
                    threshold=graph_threshold,
                )
                report["graph"]["threshold"] = graph_threshold
            elif isinstance(model, CausalSSM):
                if model.graph_mode == "identity":
                    report["graph"] = {
                        "available": False,
                        "reason": "identity mode has no trained graph logits",
                    }
                elif model.graph_mode == "random":
                    report["graph"] = {
                        "available": False,
                        "reason": "random mode has no trained graph logits",
                    }
                elif model.graph_mode == "oracle" and test_graph is not None:
                    report["graph"] = graph_metrics(
                        model.fixed_graph.detach(),
                        test_graph[-1],
                    )
            if mode == "online" and learner is not None and test_graph is not None:
                history_value = training_result.get("graph_history")
                detection_value = training_result.get("detections")
                if history_value is not None and detection_value is not None:
                    history = torch.tensor(history_value)
                    step_shd = [
                        float(
                            ((history[index] >= 0.5) != (test_graph[index] >= 0.5))
                            .sum()
                            .item()
                        )
                        for index in range(history.shape[0])
                    ]
                    if switch_at is not None:
                        switch_index = switch_at - (test_start + input_length - 1)
                        if 0 <= switch_index < len(step_shd):
                            report["adaptation"] = {
                                "detection_delay": detection_delay(
                                    detection_value, switch_index
                                )
                            }
                            if switch_index > 0:
                                report["adaptation"]["recovery_time"] = recovery_time(
                                    step_shd, switch_index
                                )
                    report["graph"]["online_final_shd"] = int(
                        graph_metrics(
                            history[-1],
                            test_graph[-1],
                            threshold=graph_threshold,
                        )["shd"]
                    )
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

    identity_report = next(
        (
            reports[str(variant["id"])]
            for variant in _model_variants(model_config)
            if variant.get("name") == "ocg" and variant.get("graph_mode") == "identity"
        ),
        None,
    )
    for variant_id, report in list(reports.items()):
        reasons = _divergence_reasons(report, identity_report)
        sanitized_report, non_finite = _sanitize_non_finite(report)
        if non_finite:
            reasons.insert(0, "non_finite_values")
        if reasons:
            sanitized_report["diverged"] = True
            sanitized_report["divergence_reasons"] = reasons
            print(f"WARNING: variant {variant_id} diverged: {', '.join(reasons)}")
        reports[variant_id] = sanitized_report

    result: dict[str, Any] = {
        "config": str(config_path),
        "seed": seed,
        "device": str(selected_device),
        "evaluation": {
            "switch_at": switch_at,
            "test_start": test_start,
            "input_length": input_length,
            "test_window_count": int(test_inputs.shape[0]),
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
