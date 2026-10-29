"""Tune synthetic cross-variable signal using only the chronological validation split."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ocg.data import generate_svar
from ocg.run import _build_model, _split_windows
from ocg.train import train_online


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _block_bootstrap_interval(
    differences: torch.Tensor,
    block_length: int = 6,
    resamples: int = 2000,
    seed: int = 0,
) -> list[float]:
    """Estimate a circular moving-block 95% interval for paired window errors."""
    if differences.ndim != 1 or differences.numel() < 1:
        raise ValueError("differences must be a nonempty vector")
    block_length = min(block_length, differences.numel())
    blocks_per_sample = (differences.numel() + block_length - 1) // block_length
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.arange(block_length)
    starts = torch.randint(
        differences.numel(),
        (resamples, blocks_per_sample),
        generator=generator,
    )
    indices = (starts.unsqueeze(-1) + offsets).remainder(differences.numel())
    samples = differences[indices.flatten(1)[:, : differences.numel()]]
    means = samples.mean(dim=1)
    return [
        float(torch.quantile(means, 0.025).item()),
        float(torch.quantile(means, 0.975).item()),
    ]


def run_sweep(
    config_path: Path,
    output_path: Path,
    scales: list[float],
) -> dict[str, Any]:
    """Run matched warm-start identity/DirectVAR validation checks for each scale."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    results: dict[str, Any] = {
        "protocol": {
            "purpose": "development/validation-only synthetic cross-variable signal selection",
            "training_split": "chronological training prefix only",
            "evaluation_split": "chronological validation segment only",
            "test_split_used": False,
            "online_start": "warm_start",
            "cross_variable_scale_candidates": scales,
            "noise_scale_fixed": float(data_config["noise_scale"]),
            "data_seed": int(data_config["seed"]),
            "model_seed": int(config["seed"]),
            "source_sha256": {
                name: _sha256(Path(name))
                for name in (
                    "src/ocg/data.py",
                    "src/ocg/run.py",
                    "src/ocg/ssm.py",
                    "src/ocg/train.py",
                    "src/ocg/baselines.py",
                    str(config_path),
                )
            },
        },
        "candidates": [],
    }

    for scale in scales:
        started = time.perf_counter()
        generated = generate_svar(
            n_steps=int(data_config["n_steps"]),
            n_variables=int(data_config["n_variables"]),
            n_lags=int(data_config["n_lags"]),
            switch_at=int(data_config["switch_at"]),
            blend_steps=int(data_config.get("blend_steps", 0)),
            sparsity=float(data_config["sparsity"]),
            noise_scale=float(data_config["noise_scale"]),
            seed=int(data_config["seed"]),
            cross_variable_scale=scale,
        )
        train, validation, _, _ = _split_windows(
            generated["series"],
            int(data_config["input_length"]),
            int(data_config["horizon"]),
            float(data_config["train_fraction"]),
            float(data_config["validation_fraction"]),
        )
        train_inputs, train_targets = train
        validation_inputs, validation_targets = validation
        variants = (
            {
                "id": "identity",
                "name": "ocg",
                "graph_mode": "identity",
                **{
                    key: value
                    for key, value in model_config.items()
                    if key not in {"variants", "name", "graph_mode"}
                },
            },
            {"id": "var_direct", "name": "var_direct"},
        )
        variant_results: dict[str, Any] = {}
        for variant in variants:
            model_seed = int(config["seed"])
            torch.manual_seed(model_seed)
            model = _build_model(
                variant,
                int(data_config["n_variables"]),
                int(data_config["horizon"]),
                int(data_config["n_lags"]),
                None,
                model_seed,
                float(data_config["sparsity"]),
            )
            warm = train_online(
                model,
                train_inputs,
                train_targets,
                learning_rate=float(training_config["learning_rate"]),
                seed=model_seed,
            )
            evaluation = train_online(
                model,
                validation_inputs,
                validation_targets,
                learning_rate=float(training_config["learning_rate"]),
                seed=model_seed,
            )
            predictions = torch.tensor(evaluation["predictions"])
            squared_errors = (predictions - validation_targets).square()
            variant_results[variant["id"]] = {
                "mse": float(squared_errors.mean().item()),
                "per_horizon_mse": squared_errors.mean(dim=(0, 2)).tolist(),
                "per_window_per_horizon_mse": squared_errors.mean(dim=2).tolist(),
                "per_window_mse": squared_errors.mean(dim=(1, 2)).tolist(),
                "warm_start_windows": len(train_inputs),
                "warm_start_final_running_mse": warm["running_mse"][-1],
            }

        identity_errors = torch.tensor(
            variant_results["identity"]["per_window_per_horizon_mse"]
        )
        direct_errors = torch.tensor(
            variant_results["var_direct"]["per_window_per_horizon_mse"]
        )
        paired_differences = direct_errors - identity_errors
        candidate = {
            "cross_variable_scale": scale,
            "maximum_companion_radius": max(
                _companion_radius(frame)
                for frame in generated["coefficients"]
            ),
            "training_windows": len(train_inputs),
            "validation_windows": len(validation_inputs),
            "variants": variant_results,
            "var_direct_minus_identity_by_horizon": [
                {
                    "mean_difference": float(paired_differences[:, horizon].mean().item()),
                    "circular_block_bootstrap_95pct_ci": _block_bootstrap_interval(
                        paired_differences[:, horizon],
                        seed=int(config["seed"]) + horizon,
                    ),
                    "block_length_windows": min(6, len(validation_inputs)),
                }
                for horizon in range(int(data_config["horizon"]))
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        results["candidates"].append(candidate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        identity = variant_results["identity"]["per_horizon_mse"]
        direct = variant_results["var_direct"]["per_horizon_mse"]
        print(
            json.dumps(
                {
                    "cross_variable_scale": scale,
                    "identity_h1_h2": identity[:2],
                    "var_direct_h1_h2": direct[:2],
                    "candidate_seconds": candidate["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    return results


def _companion_radius(coefficients: torch.Tensor) -> float:
    """Compute one companion spectral radius without importing private helpers."""
    n_lags, n_variables, _ = coefficients.shape
    companion = torch.zeros(
        n_variables * n_lags, n_variables * n_lags, dtype=torch.float64
    )
    companion[:n_variables] = torch.cat(tuple(coefficients), dim=1).to(torch.float64)
    if n_lags > 1:
        companion[n_variables:, :-n_variables] = torch.eye(
            n_variables * (n_lags - 1), dtype=torch.float64
        )
    return float(torch.linalg.eigvals(companion).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_abrupt_warm.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/warm_start_revision/validation_sweep.json"),
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0, 3.0],
    )
    arguments = parser.parse_args()
    run_sweep(arguments.config, arguments.output, arguments.scales)


if __name__ == "__main__":
    main()
