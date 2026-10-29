"""Tests for the resumable graph-utilization experiment runner."""

import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_graph_utilization import _run_one
from scripts.graph_utilization_report import (
    _per_seed_difference_cis,
    _per_seed_ratio_cis,
)


def test_run_one_resumes_existing_result_with_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    output_path = tmp_path / "run.json"
    config_path.write_text(yaml.safe_dump({"seed": 101}), encoding="utf-8")
    output_path.write_text(json.dumps({"seed": 101}), encoding="utf-8")
    ledger: dict[str, Any] = {
        "runs": {
            "run": {
                "status": "complete",
                "output": str(output_path),
                "elapsed_seconds": 12.5,
                "test_windows": 202,
            }
        }
    }

    _run_one(config_path, output_path, ledger)

    assert ledger["runs"]["run"]["status"] == "complete"
    assert ledger["runs"]["run"]["output"] == str(output_path)
    assert ledger["runs"]["run"]["elapsed_seconds"] == 12.5
    assert ledger["runs"]["run"]["test_windows"] == 202


def test_report_bootstraps_each_seed_separately() -> None:
    ratios = _per_seed_ratio_cis(
        [[2.0] * 8, [3.0] * 8],
        [[1.0] * 8, [1.0] * 8],
        bootstrap_seed=17,
    )
    differences = _per_seed_difference_cis(
        [[1.0] * 8, [2.0] * 8],
        bootstrap_seed=23,
    )

    assert ratios == [[2.0, 2.0], [3.0, 3.0]]
    assert differences == [[1.0, 1.0], [2.0, 2.0]]
