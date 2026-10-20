"""Tests for YAML-driven experiment loading and CLI execution."""

import json
from pathlib import Path
from typing import Any

import yaml

from ocg.run import main, run_experiment


def _graph_config() -> dict[str, int | float]:
    """Return explicit graph-learning and regime-detector hyperparameters."""
    return {
        "learning_rate": 0.01,
        "sparsity_weight": 0.001,
        "temperature": 1.0,
        "ema_decay": 0.95,
        "graph_stride": 1,
        "boost": 3.0,
        "boost_steps": 20,
        "detector_decay": 0.95,
        "detector_threshold": 3.5,
        "detector_warmup": 30,
        "detector_min_std": 0.000001,
    }


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
            "graph": _graph_config(),
            "variants": [
                {"id": "identity", "name": "ocg", "graph_mode": "identity"},
                {"id": "random", "name": "ocg", "graph_mode": "random"},
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
    assert set(result["variants"]) == {"identity", "random", "oracle"}
    assert result["evaluation"] == {
        "switch_at": 56,
        "test_start": 48,
        "input_length": 4,
        "test_window_count": 44,
    }
    assert result["variants"]["identity"]["forecast"]["mse"] >= 0
    assert len(result["variants"]["identity"]["forecast"]["per_window_mse"]) == 44
    assert result["variants"]["identity"]["graph"]["available"] is False
    assert result["variants"]["random"]["graph"]["available"] is False
    assert "graph" in result["variants"]["oracle"]
    assert result["variants"]["oracle"]["graph"]["shd"] == 0
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
        "graph": _graph_config(),
    }
    config_path = tmp_path / "online.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "online.json")
    report = result["variants"]["ocg"]
    assert len(report["training"]["prequential_mse"]) == 44
    assert report["graph"]["online_final_shd"] >= 0
    assert "recovery_time" in report["adaptation"]
    assert "detection_delay" in report["adaptation"]


def test_offline_learned_run_reports_graph_warm_start(tmp_path: Path) -> None:
    """Offline learned runs report structural warm-start loss and logit drift."""
    config = _synthetic_config("offline")
    config["model"] = {
        "name": "ocg",
        "graph_mode": "learned",
        "state_dim": 4,
        "n_layers": 1,
        "embedding_dim": 2,
        "graph": _graph_config(),
    }
    config_path = tmp_path / "offline_learned.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "offline_learned.json")
    training = result["variants"]["ocg"]["training"]
    assert len(training["warm_start_structural_loss"]) == 5
    assert training["warm_start_logit_delta_l1"] > 0


def test_cli_and_all_benchmark_configs(tmp_path: Path, capsys: Any) -> None:
    """CLI runs a config and all seven benchmark files parse as YAML."""
    repository = Path(__file__).resolve().parents[1]
    config_paths = sorted((repository / "configs").glob("*.yaml"))
    assert {
        "synthetic_abrupt",
        "synthetic_gradual",
        "ett",
        "electricity",
        "traffic",
        "ablation",
        "stress",
    }.issubset({path.stem for path in config_paths})
    for path in config_paths:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
    d20_config = yaml.safe_load(
        (repository / "configs" / "sanity_d20.yaml").read_text(encoding="utf-8")
    )
    assert d20_config["data"]["n_variables"] == 20
    assert d20_config["data"]["sparsity"] == 0.1
    assert d20_config["data"]["n_steps"] == d20_config["data"]["switch_at"] == 6000
    assert d20_config["training"]["mode"] == "offline"
    assert d20_config["training"]["epochs"] == 40
    assert {
        variant["graph_mode"] for variant in d20_config["model"]["variants"]
    } == {"identity", "dense", "random", "oracle", "learned"}
    ablation_config = yaml.safe_load(
        (repository / "configs" / "ablation.yaml").read_text(encoding="utf-8")
    )
    assert {
        variant["graph_mode"] for variant in ablation_config["model"]["variants"]
    } == {"identity", "dense", "random", "oracle", "learned"}

    config_path = tmp_path / "cli.yaml"
    config = _synthetic_config()
    config["model"] = {
        "name": "revin_linear",
        "input_length": 4,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "cli.json"
    assert main(
        ["--config", str(config_path), "--output", str(output_path), "--device", "cpu"]
    ) == 0
    assert '"variants": ["revin_linear"]' in capsys.readouterr().out
    assert output_path.exists()


def test_colab_notebook_is_valid_and_runs_to_drive() -> None:
    """The Colab notebook includes cloning, installation, execution, and Drive output."""
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads((repository / "notebooks" / "colab_train.ipynb").read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell["source"]
    )
    assert "git clone" in source
    assert "pip install -e ." in source
    assert "configs/synthetic_abrupt.yaml" in source
    assert "/content/drive/MyDrive/ocgssm/results/" in source


def test_real_csv_path_override(tmp_path: Path) -> None:
    """The runner loads a user-provided CSV in place of the configured path."""
    csv_path = tmp_path / "series.csv"
    csv_path.write_text(
        "first,second\n" + "".join(f"{index},{index % 7}\n" for index in range(100)),
        encoding="utf-8",
    )
    config = {
        "seed": 5,
        "data": {
            "type": "csv",
            "csv_path": "not-present.csv",
            "input_length": 4,
            "horizon": 1,
            "train_fraction": 0.5,
            "validation_fraction": 0.2,
            "n_lags": 1,
        },
        "model": {"name": "revin_linear", "input_length": 4},
        "training": {
            "mode": "offline",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
    }
    config_path = tmp_path / "csv.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "csv.json", data_path=csv_path)
    assert result["variants"]["revin_linear"]["forecast"]["mse"] >= 0


def test_run_experiment_sanitizes_non_finite_variant_values(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """A variant report with inf produces valid JSON with diverged=True while others stay unchanged."""
    import ocg.run as run_module

    original_train_offline = run_module.train_offline
    call_count = 0

    def patched_train_offline(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        result = original_train_offline(*args, **kwargs)
        if call_count == 2:
            result["train_mse"] = [float("inf")]
        return result

    monkeypatch.setattr(run_module, "train_offline", patched_train_offline)

    config_path = tmp_path / "diverged.yaml"
    output_path = tmp_path / "diverged.json"
    config_path.write_text(yaml.safe_dump(_synthetic_config()), encoding="utf-8")

    result = run_experiment(config_path, output_path)
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "random" in captured
    assert result["variants"]["random"]["diverged"] is True
    assert result["variants"]["random"]["training"]["train_mse"] == [None]
    assert "diverged" not in result["variants"]["identity"]
    assert "diverged" not in result["variants"]["oracle"]
    assert all(
        value is not None
        for value in result["variants"]["identity"]["training"]["train_mse"]
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == result



def test_grad_clip_defaults_on_and_can_be_disabled(tmp_path: Path, monkeypatch: Any) -> None:
    """Offline and online runs clip gradients at 1.0 unless the config sets grad_clip_norm null."""
    import ocg.run as run_module

    seen: list[float | None] = []

    def capture(original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs.get("grad_clip_norm"))
            return original(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(run_module, "train_offline", capture(run_module.train_offline))
    monkeypatch.setattr(run_module, "train_online", capture(run_module.train_online))

    for mode in ("offline", "online"):
        config = _synthetic_config(mode)
        config["model"]["variants"] = config["model"]["variants"][:1]
        config_path = tmp_path / f"default_{mode}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run_experiment(config_path, tmp_path / f"default_{mode}.json")
        config["training"]["grad_clip_norm"] = None
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run_experiment(config_path, tmp_path / f"disabled_{mode}.json")
    assert seen == [1.0, None, 1.0, None]
