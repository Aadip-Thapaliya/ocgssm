"""Summarize warm-start JSON outputs with robust errors and readable log plots."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


TRIM_FRACTION_EACH_TAIL = 0.1


def _block_bootstrap_interval(
    differences: list[float],
    seed: int,
    block_length: int = 6,
    resamples: int = 2000,
) -> list[float]:
    """Estimate a circular moving-block 95% interval for paired window errors."""
    import torch

    values = torch.tensor(differences, dtype=torch.float64)
    if values.numel() < 1:
        raise ValueError("paired differences must be nonempty")
    block_length = min(block_length, values.numel())
    blocks_per_sample = math.ceil(values.numel() / block_length)
    offsets = torch.arange(block_length)
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(values.numel(), (resamples, blocks_per_sample), generator=generator)
    indices = (starts.unsqueeze(-1) + offsets).remainder(values.numel())
    samples = values[indices.flatten(1)[:, : values.numel()]]
    means = samples.mean(dim=1)
    return [
        float(torch.quantile(means, 0.025).item()),
        float(torch.quantile(means, 0.975).item()),
    ]


def _trimmed_mean(values: list[float]) -> float:
    """Mean after removing the lowest and highest 10%, retaining raw inputs."""
    if not values:
        raise ValueError("trimmed mean requires at least one observation")
    ordered = sorted(values)
    trim = min(
        math.floor(len(ordered) * TRIM_FRACTION_EACH_TAIL),
        (len(ordered) - 1) // 2,
    )
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(kept)


def _robust_metrics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("robust forecast metrics require at least one observation")
    return {
        "n": len(values),
        "mean_mse": statistics.fmean(values),
        "median_mse": statistics.median(values),
        "trimmed_mean_mse": _trimmed_mean(values),
        "maximum_mse": max(values),
        "trim_fraction_each_tail": TRIM_FRACTION_EACH_TAIL,
    }


def build_report(result_specs: list[tuple[str, Path]]) -> dict[str, Any]:
    """Read raw result files and compute robust per-window and per-horizon metrics."""
    source_files: dict[str, str] = {}
    runs_by_variant: dict[str, list[dict[str, Any]]] = {}
    values_by_variant: dict[str, list[list[float]]] = {}
    horizons_by_variant: dict[str, list[list[list[float]]]] = {}
    per_window_by_source: dict[str, dict[str, list[list[float]]]] = {}
    seed_by_source: dict[str, int] = {}
    if len({label for label, _ in result_specs}) != len(result_specs):
        raise ValueError("each result file must have a unique source label")
    for label, path in result_specs:
        result = json.loads(path.read_text(encoding="utf-8"))
        source_files[label] = str(path)
        seed_by_source[label] = int(result["seed"])
        per_window_by_source[label] = {}
        for variant_id, report in result["variants"].items():
            forecast = report["forecast"]
            values = [float(value) for value in forecast["per_window_mse"]]
            values_by_variant.setdefault(variant_id, []).append(values)
            row = {
                "source": label,
                "protocol": result["evaluation"].get("online_start"),
                "seed": result["seed"],
                "overall": _robust_metrics(values),
                "raw_values_retained_in": str(path),
            }
            by_horizon = forecast.get("per_window_per_horizon_mse")
            if by_horizon is not None:
                horizons = list(zip(*by_horizon, strict=True))
                row["by_horizon"] = [
                    _robust_metrics([float(value) for value in horizon])
                    for horizon in horizons
                ]
                horizon_values = [
                    [float(value) for value in window]
                    for window in by_horizon
                ]
                per_window_by_source[label][variant_id] = horizon_values
                horizons_by_variant.setdefault(variant_id, []).append(
                    horizon_values
                )
            runs_by_variant.setdefault(variant_id, []).append(row)

    variants: dict[str, dict[str, Any]] = {}
    for variant_id, runs in runs_by_variant.items():
        all_values = [
            value for run_values in values_by_variant[variant_id] for value in run_values
        ]
        horizon_runs = horizons_by_variant.get(variant_id, [])
        by_horizon = None
        if horizon_runs:
            horizon_count = min(len(run[0]) for run in horizon_runs if run)
            by_horizon = [
                _robust_metrics(
                    [
                        window[horizon_index]
                        for run in horizon_runs
                        for window in run
                    ]
                )
                for horizon_index in range(horizon_count)
            ]
        paths = [run["raw_values_retained_in"] for run in runs]
        variants[variant_id] = {
            "n_runs": len(runs),
            "runs": runs,
            "overall": _robust_metrics([float(value) for value in all_values]),
            "raw_values_retained_in": paths[0] if len(paths) == 1 else paths,
        }
        if by_horizon is not None:
            variants[variant_id]["by_horizon"] = by_horizon

    paired_comparisons = {}
    for reference, candidate in (
        ("identity", "var_direct"),
        ("identity", "oracle"),
        ("oracle", "learned"),
    ):
        by_source = {}
        for source, source_variants in per_window_by_source.items():
            if reference not in source_variants or candidate not in source_variants:
                continue
            reference_values = source_variants[reference]
            candidate_values = source_variants[candidate]
            if (
                len(reference_values) != len(candidate_values)
                or not reference_values
                or len(reference_values[0]) != len(candidate_values[0])
            ):
                continue
            by_source[source] = [
                {
                    "mean_difference": statistics.fmean(
                        candidate_values[index][horizon]
                        - reference_values[index][horizon]
                        for index in range(len(reference_values))
                    ),
                    "circular_block_bootstrap_95pct_ci": _block_bootstrap_interval(
                        [
                            candidate_values[index][horizon]
                            - reference_values[index][horizon]
                            for index in range(len(reference_values))
                        ],
                        seed=seed_by_source[source] + horizon,
                    ),
                    "block_length_windows": min(6, len(reference_values)),
                }
                for horizon in range(len(reference_values[0]))
            ]
        if by_source:
            paired_comparisons[f"{candidate}_minus_{reference}"] = (
                next(iter(by_source.values()))
                if len(by_source) == 1
                else {"by_source": by_source}
            )
    return {
        "protocol": {
            "trimmed_mean": (
                "sort per-window MSE values and remove floor(10% of n) from "
                "each tail; if this would remove every observation, trim "
                "at most floor((n-1)/2) per tail"
            ),
            "raw_spikes_retained": True,
            "log_plot_zero_handling": (
                "zero-valued metrics are placed at the smallest positive plotted "
                "value divided by ten (or 1e-12 if every value is zero) and labeled 0"
            ),
        },
        "source_files": source_files,
        "variants": variants,
        "paired_horizon_comparisons": paired_comparisons,
    }


def write_plots(report: dict[str, Any], output_directory: Path) -> None:
    """Write robust-metric and per-horizon plots without modifying source results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    available = report["variants"]
    variant_ids = [
        variant_id
        for variant_id in ("identity", "var_direct", "learned", "oracle", "ocg")
        if variant_id in available
    ]
    metric_names = ("mean_mse", "median_mse", "trimmed_mean_mse")
    values = [
        available[variant_id]["overall"][metric]
        for variant_id in variant_ids
        for metric in metric_names
    ]
    positive = [value for value in values if value > 0]
    zero_floor = min(positive) / 10.0 if positive else 1e-12

    figure, axis = plt.subplots(figsize=(max(8, len(variant_ids) * 2.2), 5))
    width = 0.24
    x_positions = list(range(len(variant_ids)))
    for metric_index, metric in enumerate(metric_names):
        centers = [
            available[variant_id]["overall"][metric]
            for variant_id in variant_ids
        ]
        plotted = [value if value > 0 else zero_floor for value in centers]
        offsets = [
            index + (metric_index - 1) * width for index in x_positions
        ]
        bars = axis.bar(offsets, plotted, width=width, label=metric.removesuffix("_mse"))
        for bar, value in zip(bars, centers, strict=True):
            if value == 0:
                axis.annotate(
                    "0",
                    (bar.get_x() + bar.get_width() / 2, zero_floor),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    axis.set_yscale("log")
    axis.set_xticks(x_positions, variant_ids)
    axis.set_ylabel("Per-window MSE (log scale)")
    axis.set_title("Warm-start forecast error: mean, median, trimmed mean")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "forecast_mse_log.png", dpi=160)
    plt.close(figure)

    horizon_variants = [
        variant_id
        for variant_id in variant_ids
        if "by_horizon" in available[variant_id]
    ]
    if horizon_variants:
        horizon_count = max(
            len(available[variant_id]["by_horizon"])
            for variant_id in horizon_variants
        )
        figure, axis = plt.subplots(figsize=(8, 5))
        for variant_id in horizon_variants:
            means = [
                available[variant_id]["by_horizon"][index]["mean_mse"]
                for index in range(horizon_count)
            ]
            axis.plot(range(1, horizon_count + 1), means, marker="o", label=variant_id)
        axis.set_yscale("log")
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel("Per-horizon MSE (mean, log scale)")
        axis.set_title("Warm-start forecast error by horizon")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_directory / "horizon_mse_log.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Warm-start result JSON; repeat to combine distinct variant runs.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/warm_start_revision"),
    )
    arguments = parser.parse_args()
    specs = []
    for item in arguments.result:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            parser.error("--result must use LABEL=PATH")
        specs.append((label, Path(path)))
    report = build_report(specs)
    output_path = arguments.output_directory / "robust_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_plots(report, arguments.output_directory)
    print(json.dumps({"summary": str(output_path), "variants": list(report["variants"])}))


if __name__ == "__main__":
    main()
