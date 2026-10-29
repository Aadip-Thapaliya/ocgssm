"""Summarize matched graph-utilization runs without pooling seeds as IID data."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


VARIANTS = ("identity", "dlinear", "var_direct", "dense", "random", "learned", "oracle")
SEED_PAIRS = ((11, 101), (23, 202), (29, 303))
HORIZONS = (0, 1)
BLOCK_LENGTH = 6
BOOTSTRAP_RESAMPLES = 2000


def _read_runs(specs: list[tuple[float, int, int, Path]]) -> dict[float, dict[tuple[int, int], dict[str, Any]]]:
    runs: dict[float, dict[tuple[int, int], dict[str, Any]]] = {}
    for scale, data_seed, model_seed, path in specs:
        key = (data_seed, model_seed)
        if key in runs.setdefault(scale, {}):
            raise ValueError(f"duplicate run for scale {scale} and seed pair {key}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result["seed"]) != model_seed:
            raise ValueError(f"{path} has model seed {result['seed']}, expected {model_seed}")
        if set(result["variants"]) != set(VARIANTS):
            raise ValueError(f"{path} does not contain exactly the seven frozen variants")
        for variant in VARIANTS:
            errors = result["variants"][variant]["forecast"]["per_window_per_horizon_mse"]
            if not errors or any(len(row) < 2 for row in errors):
                raise ValueError(f"{path} is missing horizon 1/2 window errors for {variant}")
        runs[scale][key] = result
    expected = set(SEED_PAIRS)
    for scale, scale_runs in runs.items():
        if set(scale_runs) != expected:
            raise ValueError(
                f"scale {scale} needs fixed seed pairs {sorted(expected)}; "
                f"found {sorted(scale_runs)}"
            )
        reference_lengths = {
            len(scale_runs[pair]["variants"]["identity"]["forecast"]["per_window_mse"])
            for pair in SEED_PAIRS
        }
        if reference_lengths != {202}:
            raise ValueError(f"scale {scale} must contain 202 test windows per seed")
    return runs


def _sample_indices(n: int, generator: Any) -> list[int]:
    blocks = math.ceil(n / BLOCK_LENGTH)
    starts = generator.integers(0, n, size=blocks)
    return [
        int((start + offset) % n)
        for start in starts
        for offset in range(BLOCK_LENGTH)
    ][:n]


def _bootstrap_ratio_ci(
    numerator: list[list[float]],
    denominator: list[list[float]],
    bootstrap_seed: int,
) -> list[float]:
    import numpy as np

    generator = np.random.default_rng(bootstrap_seed)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        seed_ratios = []
        for n_values, d_values in zip(numerator, denominator, strict=True):
            indices = _sample_indices(len(n_values), generator)
            n_mean = statistics.fmean(n_values[index] for index in indices)
            d_mean = statistics.fmean(d_values[index] for index in indices)
            seed_ratios.append(n_mean / d_mean)
        draws.append(statistics.fmean(seed_ratios))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _bootstrap_difference_ci(
    differences: list[list[float]],
    bootstrap_seed: int,
) -> list[float]:
    import numpy as np

    generator = np.random.default_rng(bootstrap_seed)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        seed_means = []
        for values in differences:
            indices = _sample_indices(len(values), generator)
            seed_means.append(statistics.fmean(values[index] for index in indices))
        draws.append(statistics.fmean(seed_means))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _per_seed_ratio_cis(
    numerator: list[list[float]],
    denominator: list[list[float]],
    bootstrap_seed: int,
) -> list[list[float]]:
    return [
        _bootstrap_ratio_ci([n_values], [d_values], bootstrap_seed + index)
        for index, (n_values, d_values) in enumerate(
            zip(numerator, denominator, strict=True)
        )
    ]


def _per_seed_difference_cis(
    differences: list[list[float]],
    bootstrap_seed: int,
) -> list[list[float]]:
    return [
        _bootstrap_difference_ci([values], bootstrap_seed + index)
        for index, values in enumerate(differences)
    ]


def _per_seed_errors(
    runs: dict[tuple[int, int], dict[str, Any]],
    variant: str,
    horizon: int,
) -> list[list[float]]:
    return [
        [
            float(window[horizon])
            for window in runs[pair]["variants"][variant]["forecast"][
                "per_window_per_horizon_mse"
            ]
        ]
        for pair in SEED_PAIRS
    ]


def _ratio_table(
    runs: dict[tuple[int, int], dict[str, Any]],
    reference: str = "identity",
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for variant in VARIANTS:
        horizon_rows = []
        for horizon in HORIZONS:
            numerator = _per_seed_errors(runs, variant, horizon)
            denominator = _per_seed_errors(runs, reference, horizon)
            seed_ratios = [
                statistics.fmean(n) / statistics.fmean(d)
                for n, d in zip(numerator, denominator, strict=True)
            ]
            horizon_rows.append(
                {
                    "horizon": horizon + 1,
                    "seed_ratios": seed_ratios,
                    "seed_paired_block_bootstrap_95pct_ci": _per_seed_ratio_cis(
                        numerator,
                        denominator,
                        300000 + 10000 * VARIANTS.index(variant) + 100 * horizon,
                    ),
                    "equal_seed_mean_ratio": statistics.fmean(seed_ratios),
                    "paired_block_bootstrap_95pct_ci": _bootstrap_ratio_ci(
                        numerator,
                        denominator,
                        100000 + 1000 * (VARIANTS.index(variant) + 1) + horizon,
                    ),
                }
            )
        table[variant] = horizon_rows
    return table


def _paired_comparison(
    runs: dict[tuple[int, int], dict[str, Any]],
    candidate: str,
    reference: str,
) -> list[dict[str, Any]]:
    output = []
    for horizon in HORIZONS:
        differences = []
        for pair in SEED_PAIRS:
            candidate_errors = [
                float(row[horizon])
                for row in runs[pair]["variants"][candidate]["forecast"][
                    "per_window_per_horizon_mse"
                ]
            ]
            reference_errors = [
                float(row[horizon])
                for row in runs[pair]["variants"][reference]["forecast"][
                    "per_window_per_horizon_mse"
                ]
            ]
            if len(candidate_errors) != len(reference_errors):
                raise ValueError("paired variants must have identical test windows")
            differences.append(
                [c - r for c, r in zip(candidate_errors, reference_errors, strict=True)]
            )
        seed_means = [statistics.fmean(values) for values in differences]
        output.append(
            {
                "horizon": horizon + 1,
                "seed_mean_differences": seed_means,
                "seed_paired_block_bootstrap_95pct_ci": _per_seed_difference_cis(
                    differences,
                    400000 + 10000 * VARIANTS.index(candidate) + 100 * horizon,
                ),
                "equal_seed_mean_difference": statistics.fmean(seed_means),
                "paired_block_bootstrap_95pct_ci": _bootstrap_difference_ci(
                    differences,
                    200000 + 1000 * VARIANTS.index(candidate) + horizon,
                ),
            }
        )
    return output


def _graph_quality(
    runs: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for data_seed, model_seed in SEED_PAIRS:
        result = runs[(data_seed, model_seed)]
        graph = result["variants"]["learned"]["graph"]
        timestamps = graph["timestamps"]
        values = graph["per_step"]
        for period, indices in (
            ("pre_switch", [i for i, t in enumerate(timestamps) if t < 320]),
            ("post_switch", [i for i, t in enumerate(timestamps) if t >= 320]),
        ):
            if not indices:
                raise ValueError(f"no learned graph snapshots for {period}")
            metrics = {}
            for metric in ("auroc", "average_precision", "precision", "recall", "f1", "shd"):
                present = [values[index][metric] for index in indices if values[index][metric] is not None]
                metrics[metric] = statistics.fmean(present) if present else None
            rows.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "period": period,
                    "snapshots": len(indices),
                    "threshold": graph["metric_threshold"],
                    **metrics,
                }
            )
    return {
        "timing": "EMA snapshots before the graph update, aligned to last input timestamp",
        "threshold_source": "validation-only maximum pooled F1",
        "seed_rows": rows,
    }


def build_report(specs: list[tuple[float, int, int, Path]]) -> dict[str, Any]:
    runs_by_scale = _read_runs(specs)
    result: dict[str, Any] = {
        "protocol": {
            "path": "results/graph_utilization/protocol.yaml",
            "seed_pairs": [list(pair) for pair in SEED_PAIRS],
            "primary_gate": "Scale 1.5, horizons 1 and 2, ratios are equal-weight seed means.",
            "block_bootstrap": {
                "kind": "paired circular moving-block percentile",
                "block_length": BLOCK_LENGTH,
                "resamples": BOOTSTRAP_RESAMPLES,
                "confidence": 0.95,
                "aggregation": "resample windows within each seed, then average the three seed statistics equally",
            },
            "directvar_is_ols": False,
        },
        "scales": {},
    }
    for scale in sorted(runs_by_scale):
        runs = runs_by_scale[scale]
        ratios = _ratio_table(runs)
        result["scales"][str(scale)] = {
            "ratios_to_identity": ratios,
            "oracle_to_var_direct": _ratio_table(runs, reference="var_direct")[
                "oracle"
            ],
            "learned_to_oracle": _ratio_table(runs, reference="oracle")[
                "learned"
            ],
            "learned_minus_dense": _paired_comparison(runs, "learned", "dense"),
            "learned_minus_random": _paired_comparison(runs, "learned", "random"),
            "learned_graph_quality": _graph_quality(runs),
            "per_seed_horizon_mse": {
                f"{data_seed}_{model_seed}": {
                    variant: [
                        float(value)
                        for value in runs[(data_seed, model_seed)]["variants"][
                            variant
                        ]["forecast"]["per_horizon_mse"][:2]
                    ]
                    for variant in VARIANTS
                }
                for data_seed, model_seed in SEED_PAIRS
            },
        }
    primary = result["scales"].get("1.5")
    if primary is not None:
        result["primary_pass_criteria"] = {
            "oracle_within_5pct_of_directvar": {
                str(row["horizon"]): row["equal_seed_mean_ratio"] <= 1.05
                for row in primary["oracle_to_var_direct"]
            },
            "learned_within_5pct_of_oracle": {
                str(row["horizon"]): row["equal_seed_mean_ratio"] <= 1.05
                for row in primary["learned_to_oracle"]
            },
            "learned_beats_dense_by_paired_ci": {
                str(row["horizon"]): row["paired_block_bootstrap_95pct_ci"][1] < 0
                for row in primary["learned_minus_dense"]
            },
            "learned_beats_random_by_paired_ci": {
                str(row["horizon"]): row["paired_block_bootstrap_95pct_ci"][1] < 0
                for row in primary["learned_minus_random"]
            },
            "note": "The ratio booleans are point-estimate gates; ratio confidence intervals are reported separately and are not substituted for those gates.",
        }
    return result


def write_ratio_plot(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scales = [scale for scale in ("1.0", "1.5") if scale in report["scales"]]
    fig, axes = plt.subplots(1, len(scales), figsize=(max(8, 8 * len(scales)), 5), squeeze=False)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for axis, scale in zip(axes[0], scales, strict=True):
        table = report["scales"][scale]["ratios_to_identity"]
        for x, variant in enumerate(VARIANTS):
            for horizon_index, horizon in enumerate(HORIZONS):
                row = table[variant][horizon_index]
                offset = (horizon_index - 0.5) * 0.16
                seed_ratios = row["seed_ratios"]
                jitter = np.linspace(-0.035, 0.035, len(seed_ratios))
                axis.scatter(
                    [x + offset + jitter[index] for index in range(len(seed_ratios))],
                    seed_ratios,
                    color=colors[horizon_index],
                    alpha=0.8,
                    s=28,
                    label=f"H{horizon + 1} seeds" if x == 0 else None,
                )
                mean = row["equal_seed_mean_ratio"]
                lower, upper = row["paired_block_bootstrap_95pct_ci"]
                axis.errorbar(
                    x + offset,
                    mean,
                    yerr=[[mean - lower], [upper - mean]],
                    fmt="D",
                    color=colors[horizon_index],
                    capsize=3,
                    markersize=4,
                )
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=25, ha="right")
        axis.set_ylabel("Per-horizon MSE ratio to identity (lower is better)")
        axis.set_title(f"Cross-variable scale {scale}")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SCALE,DATA_SEED,MODEL_SEED,PATH",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    specs = []
    for item in args.run:
        fields = item.split(",", 3)
        if len(fields) != 4:
            parser.error("--run must be SCALE,DATA_SEED,MODEL_SEED,PATH")
        specs.append((float(fields[0]), int(fields[1]), int(fields[2]), Path(fields[3])))
    report = build_report(specs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output = args.output_directory / "graph_utilization_summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_ratio_plot(report, args.output_directory / "mse_ratios_to_identity.png")
    print(json.dumps({"summary": str(output), "scales": list(report["scales"])}))


if __name__ == "__main__":
    main()
