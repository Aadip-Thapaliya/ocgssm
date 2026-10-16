"""Tests for YAML-driven experiment loading and CLI execution."""

import json
from pathlib import Path
from typing import Any

import yaml

from ocg.run import main, run_experiment


def _synthetic_config(mode: str = "offline") -> dict[str, Any]:
    """Return a compact experiment config suitable for CPU smoke tests."""
    return {
        "seed": 11,
        "data": {
            "type": "synthetic",
            "n_steps": 96,
            "n_variables": 2,
            "n_lags": 1,
            "switch_at": 56,
            "blend_steps": 0,
            "sparsity": 0.5,
            "noise_scale": 0.1,
            "input_length": 4,
            "horizon": 1,
            "train_fraction": 0.4,
            "validation_fraction": 0.1,
            "seed": 11,
        },
        "model": {
            "name": "ocg",
            "state_dim": 4,
            "n_layers": 1,
            "embedding_dim": 2,
            "variants": [
                {"id": "identity", "name": "ocg", "graph_mode": "identity"},
                {"id": "oracle", "name": "ocg", "graph_mode": "oracle"},
            ],
        },
        "training": {
            "mode": mode,
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
        },
    }


def test_run_experiment_variants_and_oracle_regret(tmp_path: Path) -> None:
    """Configured variants produce saved forecasts, graph metrics, and oracle regret."""
    config_path = tmp_path / "ablation.yaml"
    output_path = tmp_path / "results" / "run.json"
    config_path.write_text(yaml.safe_dump(_synthetic_config()), encoding="utf-8")

    result = run_experiment(config_path, output_path)
    assert set(result["variants"]) == {"identity", "oracle"}
    assert result["variants"]["identity"]["forecast"]["mse"] >= 0
    assert "graph" in result["variants"]["oracle"]
    assert isinstance(result["variants"]["identity"]["regret_vs_oracle"], float)
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_online_run_reports_graph_adaptation(tmp_path: Path) -> None:
    """Learned-graph prequential runs include adaptation summaries."""
    config = _synthetic_config("online")
    config["model"] = {
        "name": "ocg",
        "graph_mode": "learned",
        "state_dim": 4,
        "n_layers": 1,
        "embedding_dim": 2,
    }
    config_path = tmp_path / "online.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "online.json")
    report = result["variants"]["ocg"]
    assert len(report["training"]["prequential_mse"]) == 44
    assert report["graph"]["online_final_shd"] >= 0
    assert "recovery_time" in report["adaptation"]
    assert "detection_delay" in report["adaptation"]


def test_cli_and_all_benchmark_configs(tmp_path: Path, capsys: Any) -> None:
    """CLI runs a config and all seven benchmark files parse as YAML."""
    repository = Path(__file__).resolve().parents[1]
    config_paths = sorted((repository / "configs").glob("*.yaml"))
    assert {path.stem for path in config_paths} == {
        "synthetic_abrupt",
        "synthetic_gradual",
        "ett",
        "electricity",
        "traffic",
        "ablation",
        "stress",
    }
    for path in config_paths:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

    config_path = tmp_path / "cli.yaml"
    config = _synthetic_config()
    config["model"] = {
        "name": "revin_linear",
        "input_length": 4,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "cli.json"
    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0
    assert '"variants": ["revin_linear"]' in capsys.readouterr().out
    assert output_path.exists()
