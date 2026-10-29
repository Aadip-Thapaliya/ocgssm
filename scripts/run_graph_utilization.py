"""Run/resume the frozen graph-utilization comparisons and matched offline trial."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from ocg.run import run_experiment


ROOT = Path("results/graph_utilization")
PROFILE = ROOT / "profile_scale15_data11_model101.yaml"
SEED_PAIRS = ((11, 101), (23, 202), (29, 303))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_for(scale: float, data_seed: int, model_seed: int) -> dict[str, Any]:
    config = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    config["seed"] = model_seed
    config["data"]["seed"] = data_seed
    config["data"]["cross_variable_scale"] = scale
    return config


def _offline_config(data_seed: int, model_seed: int) -> dict[str, Any]:
    config = _config_for(1.5, data_seed, model_seed)
    config["model"]["variants"] = [
        {"id": "var_direct", "name": "var_direct"},
        {"id": "oracle", "name": "ocg", "graph_mode": "oracle"},
    ]
    config["training"] = {
        "mode": "offline",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "restore_best_validation": True,
        "capture_forecast_components": True,
    }
    return config


def _run_one(config_path: Path, output_path: Path, ledger: dict[str, Any]) -> None:
    key = output_path.stem
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing.get("seed") != config["seed"]:
            raise ValueError(f"refusing to reuse mismatched existing output {output_path}")
        prior = ledger["runs"].get(key, {})
        ledger["runs"][key] = {
            **prior,
            "status": prior.get("status", "already_present"),
            "output": str(output_path),
        }
        return
    started = time.perf_counter()
    result = run_experiment(config_path, output_path)
    elapsed = time.perf_counter() - started
    ledger["runs"][key] = {
        "status": "complete",
        "output": str(output_path),
        "elapsed_seconds": elapsed,
        "test_windows": result["evaluation"]["test_window_count"],
    }
    ledger["updated_at_epoch_seconds"] = time.time()
    (ROOT / "run_state.json").write_text(
        json.dumps(ledger, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run": key, "elapsed_seconds": elapsed}), flush=True)


def run(profile_only: bool, offline_oracle: bool) -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT / "configs"
    output_dir = ROOT / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    online_specs = [
        (1.5, *SEED_PAIRS[0])
    ] if profile_only else [
        (scale, data_seed, model_seed)
        for scale in (1.0, 1.5)
        for data_seed, model_seed in SEED_PAIRS
    ]
    offline_specs = list(SEED_PAIRS) if offline_oracle else []
    sources = [
        Path("src/ocg/run.py"),
        Path("src/ocg/train.py"),
        Path("src/ocg/model.py"),
        Path("src/ocg/eval.py"),
        Path("src/ocg/baselines.py"),
        Path("src/ocg/ssm.py"),
        Path("src/ocg/data.py"),
        Path("scripts/graph_utilization_report.py"),
        Path("results/graph_utilization/protocol.yaml"),
        PROFILE,
    ]
    ledger_path = ROOT / "run_state.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        current_hashes = {str(path): _sha256(path) for path in sources}
        if ledger.get("source_sha256") != current_hashes:
            if ledger.get("runs"):
                raise ValueError("protocol/source hashes changed; refusing to resume this ledger")
            ledger["source_sha256"] = current_hashes
    else:
        ledger = {
            "protocol": str(ROOT / "protocol.yaml"),
            "source_sha256": {str(path): _sha256(path) for path in sources},
            "runs": {},
        }
        ledger_path.write_text(
            json.dumps(ledger, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    for scale, data_seed, model_seed in online_specs:
        label = f"online_scale{str(scale).replace('.', 'p')}_data{data_seed}_model{model_seed}"
        config_path = config_dir / f"{label}.yaml"
        output_path = output_dir / f"{label}.json"
        config_path.write_text(
            yaml.safe_dump(
                _config_for(scale, data_seed, model_seed),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _run_one(config_path, output_path, ledger)

    for data_seed, model_seed in offline_specs:
        label = f"offline_scale1p5_data{data_seed}_model{model_seed}"
        config_path = config_dir / f"{label}.yaml"
        output_path = output_dir / f"{label}.json"
        config_path.write_text(
            yaml.safe_dump(
                _offline_config(data_seed, model_seed),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _run_one(config_path, output_path, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile-only", action="store_true")
    group.add_argument("--offline-oracle", action="store_true")
    args = parser.parse_args()
    ledger = run(args.profile_only, args.offline_oracle)
    print(json.dumps({"run_count": len(ledger["runs"]), "ledger": str(ROOT / "run_state.json")}))


if __name__ == "__main__":
    main()
