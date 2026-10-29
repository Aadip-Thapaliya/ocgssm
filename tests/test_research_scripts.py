"""Tests for persisted research summaries and reproduction drivers."""

import json
from pathlib import Path
from typing import Any

from scripts.synthetic_evaluation import SCENARIOS, _write_summary
from scripts.warm_start_report import build_report, write_plots


def test_synthetic_summary_persists_seed_rows_tables_and_plots(
    tmp_path: Path,
) -> None:
    """Completed run results produce raw seed rows, mean/spread tables, and plots."""
    completed_runs: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        result_path = tmp_path / f"{scenario}.json"
        switch_at = SCENARIOS[scenario]["switch_at"]
        result_path.write_text(
            json.dumps(
                {
                    "evaluation": {"switch_at": switch_at},
                    "variants": {
                        "learned": {
                            "forecast": {
                                "mse": 0.25,
                                "mae": 0.5,
                                "per_window_mse": [0.2, 0.3],
                                "input_timestamps": [1, 2],
                                "per_horizon_mse": [0.25],
                            },
                            "graph": {
                                "shd": 1,
                                "auroc": 0.9,
                                "precision_recall_auc": 0.8,
                                "f1": 0.7,
                                "detection_events": [],
                                "timestamps": list(range(switch_at - 1, switch_at + 11)),
                                "per_step": [
                                    {"shd": value}
                                    for value in [2, 4, *([1] * 10)]
                                ],
                            },
                            "adaptation": {
                                "detection_delay": 2,
                            },
                            "efficiency": {
                                "milliseconds_per_step": 2.0,
                                "peak_memory_mb": 10.0,
                            },
                            "regret_vs_oracle": 0.1,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        completed_runs.append(
            {
                "scenario": scenario,
                "data_seed": 11,
                "model_seed": 101,
                "result_path": str(result_path),
            }
        )

    _write_summary(tmp_path, {"completed_runs": completed_runs})

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["seed_rows"]) == 3
    assert len(summary["means_and_spreads"]) == 3
    assert summary["means_and_spreads"][0]["mse_mean"] == 0.25
    assert "retrospective evaluation-only reference" in summary["protocol"]["recovery_metric"]
    assert summary["seed_rows"][0]["detection_event_count"] == 0
    assert summary["seed_rows"][0]["sustained_recovery_time"] is None
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "forecast_mse.png").exists()
    assert (tmp_path / "forecast_curves.png").exists()


def test_warm_start_report_keeps_zeros_and_writes_log_plots(tmp_path: Path) -> None:
    """Robust summaries preserve zeros/raw values and render log-scale reports."""
    result_path = tmp_path / "run.json"
    mse_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
    result_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "evaluation": {"online_start": "warm_start"},
                "variants": {
                    "identity": {
                        "forecast": {
                            "per_window_mse": mse_values,
                            "per_window_per_horizon_mse": [
                                [value, value + 1.0] for value in mse_values
                            ],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_report([("test", result_path)])
    identity = report["variants"]["identity"]
    assert identity["overall"]["n"] == 10
    assert identity["overall"]["median_mse"] == 4.5
    assert identity["overall"]["trimmed_mean_mse"] == 4.5
    assert identity["overall"]["maximum_mse"] == 100.0
    assert identity["raw_values_retained_in"] == str(result_path)

    write_plots(report, tmp_path)
    assert (tmp_path / "forecast_mse_log.png").exists()
    assert (tmp_path / "horizon_mse_log.png").exists()


def test_warm_start_report_aggregates_repeated_variant_runs(tmp_path: Path) -> None:
    """Multiple seed runs contribute to pooled robust metrics without losing rows."""
    paths = []
    for seed, values in ((11, [1.0, 2.0]), (23, [3.0, 4.0])):
        path = tmp_path / f"run-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "evaluation": {"online_start": "warm_start"},
                    "variants": {
                        "ocg": {
                            "forecast": {
                                "per_window_mse": values,
                                "per_window_per_horizon_mse": [
                                    [value, value + 1.0] for value in values
                                ],
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    report = build_report(
        [(f"seed-{seed}", path) for seed, path in zip((11, 23), paths, strict=True)]
    )
    ocg = report["variants"]["ocg"]
    assert ocg["n_runs"] == 2
    assert ocg["overall"]["n"] == 4
    assert ocg["overall"]["mean_mse"] == 2.5
    assert ocg["overall"]["median_mse"] == 2.5
    assert len(ocg["runs"]) == 2
    assert len(ocg["raw_values_retained_in"]) == 2
    assert ocg["by_horizon"][0]["mean_mse"] == 2.5
