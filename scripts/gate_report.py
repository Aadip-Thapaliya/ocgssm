"""Summarize per-window synthetic gate results from a runner JSON file."""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean of a nonempty sequence of per-window MSE values."""
    return sum(values) / len(values) if values else None


def summarize_results(
    results: dict[str, Any],
) -> dict[str, dict[str, float | int | None]]:
    """Summarize overall and switch-relative MSE plus graph metrics per variant."""
    evaluation = results.get("evaluation")
    variants = results.get("variants")
    if not isinstance(evaluation, dict) or not isinstance(variants, dict):
        raise ValueError("results must contain evaluation metadata and variants")
    try:
        switch_at = int(evaluation["switch_at"])
        test_start = int(evaluation["test_start"])
        input_length = int(evaluation["input_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "evaluation must provide integer switch_at, test_start, and input_length"
        ) from error
    if test_start < 0 or input_length < 1:
        raise ValueError("test_start must be nonnegative and input_length must be positive")

    first_last_input = test_start + input_length - 1
    post_start = max(0, switch_at - first_last_input)
    summaries: dict[str, dict[str, float | int | None]] = {}
    for variant_id, variant in variants.items():
        if not isinstance(variant, dict):
            raise ValueError(f"variant {variant_id!r} must be an object")
        forecast = variant.get("forecast")
        if not isinstance(forecast, dict):
            raise ValueError(f"variant {variant_id!r} must include forecast metrics")
        errors = forecast.get("per_window_mse")
        if (
            not isinstance(errors, list)
            or not errors
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in errors)
        ):
            raise ValueError(f"variant {variant_id!r} needs finite per_window_mse values")

        pre_switch = errors[:post_start]
        post_switch = errors[post_start:]
        early = post_switch[:200]
        late = post_switch[-400:]
        graph = variant.get("graph")
        if not isinstance(graph, dict) or graph.get("available") is False:
            shd = None
            auroc = None
        else:
            shd = graph.get("shd")
            auroc = graph.get("auroc")
        adaptation = variant.get("adaptation")
        delay = adaptation.get("detection_delay") if isinstance(adaptation, dict) else None
        summaries[str(variant_id)] = {
            "overall_mse": _mean(errors),
            "pre_switch_mse": _mean(pre_switch),
            "post_switch_early_mse": _mean(early),
            "post_switch_late_mse": _mean(late),
            "shd": shd,
            "auroc": auroc,
            "detection_delay": delay,
        }
    return summaries


def format_report(summaries: dict[str, dict[str, float | int | None]]) -> str:
    """Format gate summaries as a Markdown table."""
    headers = (
        "Variant",
        "Overall MSE",
        "Pre-switch",
        "Post early (200)",
        "Post late (400)",
        "SHD",
        "AUROC",
        "Detection delay",
    )
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for variant_id, metrics in summaries.items():
        values = [
            variant_id,
            _format_value(metrics["overall_mse"]),
            _format_value(metrics["pre_switch_mse"]),
            _format_value(metrics["post_switch_early_mse"]),
            _format_value(metrics["post_switch_late_mse"]),
            _format_value(metrics["shd"]),
            _format_value(metrics["auroc"]),
            _format_value(metrics["detection_delay"]),
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_value(value: float | int | None) -> str:
    """Format numeric metrics consistently, showing unavailable values as dashes."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Read a results JSON file and print its switch-relative gate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args(argv)
    with args.results_json.open(encoding="utf-8") as results_file:
        results = json.load(results_file)
    if not isinstance(results, dict):
        raise ValueError("results JSON root must be an object")
    print(format_report(summarize_results(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
