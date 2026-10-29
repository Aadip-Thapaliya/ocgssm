"""Replay learned variants and persist pre-update EMA graph snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from ocg.run import run_experiment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_equal(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError(f"{label} keys changed during snapshot replay")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise RuntimeError(f"{label} length changed during snapshot replay")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equal(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(
                f"{label} changed during snapshot replay: {expected} != {actual}"
            )
        return
    if expected != actual:
        raise RuntimeError(
            f"{label} changed during snapshot replay: {expected!r} != {actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/model_contribution"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory
    progress = json.loads(
        (output_directory / "resume_state.json").read_text(encoding="utf-8")
    )
    replay_directory = output_directory / "graph_snapshots"
    config_directory = replay_directory / "configs"
    result_directory = replay_directory / "results"
    config_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    index_path = replay_directory / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"completed_replays": []}
    )
    replay_sources = {
        "src/ocg/run.py": _sha256(Path("src/ocg/run.py")),
        "scripts/replay_graph_snapshots.py": _sha256(
            Path("scripts/replay_graph_snapshots.py")
        ),
    }

    for completed in progress["completed_runs"]:
        stem = (
            f"{completed['scenario']}_data{completed['data_seed']}"
            f"_model{completed['model_seed']}"
        )
        reference_path = Path(completed["result_path"])
        replay_path = result_directory / f"{stem}.json"
        previous = next(
            (
                entry
                for entry in index["completed_replays"]
                if entry["stem"] == stem
            ),
            None,
        )
        if (
            previous is not None
            and replay_path.exists()
            and previous["reference_sha256"] == _sha256(reference_path)
            and previous["replay_sha256"] == _sha256(replay_path)
            and previous["replay_sources"] == replay_sources
        ):
            continue

        config = yaml.safe_load(
            Path(completed["config_path"]).read_text(encoding="utf-8")
        )
        variants = config["model"].get("variants", [])
        learned = [variant for variant in variants if variant.get("id") == "learned"]
        if len(learned) != 1:
            raise ValueError(
                f"{completed['config_path']} must define one learned variant"
            )
        config["model"]["variants"] = learned
        config_path = config_directory / f"{stem}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        replay_result = run_experiment(config_path, replay_path, device="cpu")
        reference_result = json.loads(reference_path.read_text(encoding="utf-8"))
        expected = reference_result["variants"]["learned"]
        actual = replay_result["variants"]["learned"]

        for field in (
            "mse",
            "mae",
            "per_window_mse",
            "per_horizon_mse",
            "per_horizon_mae",
            "input_timestamps",
        ):
            _assert_equal(
                expected["forecast"][field],
                actual["forecast"][field],
                f"{stem}.forecast.{field}",
            )
        for field in (
            "shd",
            "f1",
            "precision",
            "recall",
            "auroc",
            "reference_timestamp",
            "timestamps",
            "per_step",
            "online_final_shd",
            "detection_events",
        ):
            if field in expected.get("graph", {}):
                _assert_equal(
                    expected["graph"][field],
                    actual["graph"][field],
                    f"{stem}.graph.{field}",
                )
        snapshots = actual["graph"].get("snapshots")
        if snapshots is None or len(snapshots) != len(actual["graph"]["timestamps"]):
            raise RuntimeError(f"{stem} did not save timestamp-aligned graph snapshots")
        n_lags = config["data"]["n_lags"]
        n_variables = config["data"]["n_variables"]
        for timestamp, snapshot in zip(
            actual["graph"]["timestamps"], snapshots, strict=True
        ):
            if len(snapshot) != n_lags or any(
                len(lag) != n_variables
                or any(len(row) != n_variables for row in lag)
                for lag in snapshot
            ):
                raise RuntimeError(
                    f"{stem} has an invalid graph snapshot at timestamp {timestamp}"
                )
            if not all(
                math.isfinite(value)
                for lag in snapshot
                for row in lag
                for value in row
            ):
                raise FloatingPointError(
                    f"{stem} has non-finite graph probabilities at timestamp {timestamp}"
                )

        index["completed_replays"] = [
            entry for entry in index["completed_replays"] if entry["stem"] != stem
        ]
        index["completed_replays"].append(
            {
                "stem": stem,
                "reference_result": str(reference_path),
                "reference_sha256": _sha256(reference_path),
                "replay_config": str(config_path),
                "config_sha256": _sha256(config_path),
                "replay_result": str(replay_path),
                "replay_sha256": _sha256(replay_path),
                "replay_sources": replay_sources,
                "snapshot_count": len(snapshots),
                "first_timestamp": actual["graph"]["timestamps"][0],
                "last_timestamp": actual["graph"]["timestamps"][-1],
            }
        )
        index["campaign_source_sha256"] = progress["protocol"]["source_sha256"]
        index["command"] = (
            "python -m scripts.replay_graph_snapshots "
            "--output-directory results/model_contribution"
        )
        index["completed_replays"].sort(key=lambda entry: entry["stem"])
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "stem": stem,
                    "snapshot_count": len(snapshots),
                    "first_timestamp": actual["graph"]["timestamps"][0],
                    "last_timestamp": actual["graph"]["timestamps"][-1],
                    "replay_result": str(replay_path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
