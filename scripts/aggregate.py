"""Aggregate multi-seed OCG result JSON files into summary tables."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_results(paths: list[Path]) -> list[dict[str, Any]]:
    """Load result JSON files produced by ocg.run."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return population mean/std for a nonempty float list."""
    if not values:
        raise ValueError("values must be nonempty")
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-variant test MSE and oracle paired differences across seeds."""
    per_variant: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[int, dict[str, float]] = {}
    for result in results:
        seed = int(result["seed"])
        seed_variants: dict[str, float] = {}
        for variant, report in result["variants"].items():
            mse = float(report["forecast"]["mse"])
            per_variant[variant].append(mse)
            seed_variants[variant] = mse
        per_seed[seed] = seed_variants

    summary = {
        variant: {"mse_mean": _mean_std(values)[0], "mse_std": _mean_std(values)[1]}
        for variant, values in sorted(per_variant.items())
    }

    oracle = "oracle"
    paired: dict[str, dict[str, Any]] = {}
    if oracle in per_variant:
        for variant in sorted(per_variant):
            if variant == oracle:
                continue
            diffs: list[float] = []
            wins = 0
            for seed, seed_metrics in per_seed.items():
                if oracle not in seed_metrics or variant not in seed_metrics:
                    continue
                diff = seed_metrics[variant] - seed_metrics[oracle]
                diffs.append(diff)
                if seed_metrics[oracle] < seed_metrics[variant]:
                    wins += 1
            paired[variant] = {
                "oracle_minus_variant_mse_mean": _mean_std(diffs)[0] if diffs else None,
                "oracle_minus_variant_mse_std": _mean_std(diffs)[1] if diffs else None,
                "oracle_win_count": wins,
                "seed_count": len(diffs),
            }
    return {"variants": summary, "paired_vs_oracle": paired}


def _markdown_table(aggregate_result: dict[str, Any]) -> str:
    """Format aggregate metrics as markdown tables."""
    lines = [
        "| Variant | Test MSE mean | Test MSE std |",
        "|---|---:|---:|",
    ]
    for variant, metrics in aggregate_result["variants"].items():
        lines.append(
            f"| {variant} | {metrics['mse_mean']:.6f} | {metrics['mse_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Variant | (Variant - Oracle) MSE mean | (Variant - Oracle) MSE std | Oracle wins | Seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, metrics in aggregate_result["paired_vs_oracle"].items():
        if metrics["oracle_minus_variant_mse_mean"] is None:
            continue
        lines.append(
            f"| {variant} | {metrics['oracle_minus_variant_mse_mean']:.6f} | "
            f"{metrics['oracle_minus_variant_mse_std']:.6f} | "
            f"{metrics['oracle_win_count']} | {metrics['seed_count']} |"
        )
    return "\n".join(lines)


def main() -> int:
    """Aggregate result JSON files and print markdown summary tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="Result JSON paths or glob patterns")
    parser.add_argument("--json-out", help="Optional output JSON path for aggregate stats")
    arguments = parser.parse_args()

    paths: list[Path] = []
    for pattern in arguments.results:
        expanded = sorted(Path().glob(pattern))
        if expanded:
            paths.extend(path for path in expanded if path.is_file())
        else:
            path = Path(pattern)
            if path.is_file():
                paths.append(path)
    if not paths:
        raise ValueError("no result files matched")

    aggregate_result = aggregate(_load_results(paths))
    print(_markdown_table(aggregate_result))
    if arguments.json_out is not None:
        destination = Path(arguments.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(aggregate_result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
