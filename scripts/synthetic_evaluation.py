"""Run resumable matched online synthetic comparisons across regimes and seeds."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
import sklearn

from ocg.eval import RECOVERY_REFERENCE_STEPS, SUSTAINED_RECOVERY_STEPS, recovery_time
from ocg.run import run_experiment


SEED_PAIRS = [(11, 101), (17, 202), (23, 303)]
SCENARIOS = {
    "stationary": {"switch_at": 512, "blend_steps": 0},
    "abrupt": {"switch_at": 320, "blend_steps": 0},
    "gradual": {"switch_at": 352, "blend_steps": 64},
}
SOURCE_FILES = [
    "src/ocg/data.py",
    "src/ocg/graph.py",
    "src/ocg/model.py",
    "src/ocg/train.py",
    "src/ocg/run.py",
    "src/ocg/baselines.py",
    "src/ocg/ssm.py",
    "configs/research_online.yaml",
    "scripts/synthetic_evaluation.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_summary(output_directory: Path, progress: dict[str, Any]) -> None:
    """Write per-seed rows, mean/spread summaries, and an overview plot."""
    rows = []
    curves: dict[tuple[str, str], list[tuple[list[int], list[float]]]] = {}
    for run in progress["completed_runs"]:
        result = json.loads(Path(run["result_path"]).read_text(encoding="utf-8"))
        scenario = run["scenario"]
        switch_at = result["evaluation"]["switch_at"]
        for variant_id, report in result["variants"].items():
            forecast = report["forecast"]
            errors = forecast["per_window_mse"]
            timestamps = forecast["input_timestamps"]
            pre = [
                error
                for error, timestamp in zip(errors, timestamps, strict=True)
                if switch_at is not None and timestamp < switch_at
            ]
            post = [
                error
                for error, timestamp in zip(errors, timestamps, strict=True)
                if switch_at is not None and timestamp >= switch_at
            ]
            graph = report.get("graph", {})
            adaptation = report.get("adaptation", {})
            events = graph.get("detection_events", [])
            graph_timestamps = graph.get("timestamps", [])
            graph_steps = graph.get("per_step", [])
            switch_index = (
                switch_at - graph_timestamps[0]
                if switch_at is not None and graph_timestamps
                else None
            )
            sustained_recovery = (
                recovery_time(
                    [step["shd"] for step in graph_steps],
                    switch_index,
                    sustain_steps=SUSTAINED_RECOVERY_STEPS,
                    reference_steps=RECOVERY_REFERENCE_STEPS,
                )
                if switch_index is not None
                and 0 < switch_index < len(graph_steps)
                and len(graph_steps) == len(graph_timestamps)
                else None
            )
            reference_start = len(graph_steps) - RECOVERY_REFERENCE_STEPS
            recovery_reference = (
                statistics.median(
                    step["shd"] for step in graph_steps[reference_start:]
                )
                if switch_index is not None
                and switch_index > 0
                and reference_start >= switch_index
                and len(graph_steps) == len(graph_timestamps)
                else None
            )
            detector_available = "detection_events" in graph
            curves.setdefault((scenario, variant_id), []).append(
                (timestamps, errors)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant_id,
                    "data_seed": run["data_seed"],
                    "model_seed": run["model_seed"],
                    "mse": forecast["mse"],
                    "mae": forecast["mae"],
                    "pre_switch_mse": statistics.fmean(pre) if pre else None,
                    "post_switch_mse": statistics.fmean(post) if post else None,
                    "per_horizon_mse": forecast["per_horizon_mse"],
                    "graph_shd": graph.get("shd"),
                    "graph_auroc": graph.get("auroc"),
                    "graph_pr_auc": graph.get("precision_recall_auc"),
                    "graph_f1": graph.get("f1"),
                    "detection_event_count": (
                        len(events) if detector_available else None
                    ),
                    "detection_rate": (
                        float(bool(events)) if detector_available else None
                    ),
                    "detection_false_alarms": (
                        sum(
                            event["timestamp"] < switch_at
                            for event in events
                            if switch_at is not None
                        )
                        if detector_available
                        else None
                    ),
                    "detection_delay": adaptation.get("detection_delay"),
                    "sustained_recovery_time": sustained_recovery,
                    "sustained_recovery_reference_shd": recovery_reference,
                    "regret_vs_oracle": report.get("regret_vs_oracle"),
                    "milliseconds_per_step": report["efficiency"][
                        "milliseconds_per_step"
                    ],
                    "peak_memory_mb": report["efficiency"]["peak_memory_mb"],
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["variant"]), []).append(row)
    summaries = []
    for (scenario, variant), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "scenario": scenario,
            "variant": variant,
            "n": len(group),
        }
        for field in (
            "mse",
            "mae",
            "pre_switch_mse",
            "post_switch_mse",
            "graph_shd",
            "graph_auroc",
            "graph_pr_auc",
            "graph_f1",
            "detection_event_count",
            "detection_rate",
            "detection_false_alarms",
            "detection_delay",
            "sustained_recovery_time",
            "sustained_recovery_reference_shd",
            "regret_vs_oracle",
            "milliseconds_per_step",
            "peak_memory_mb",
        ):
            values = [row[field] for row in group if row[field] is not None]
            summary[f"{field}_mean"] = statistics.fmean(values) if values else None
            summary[f"{field}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary["per_horizon_mse_mean"] = [
            statistics.fmean(row["per_horizon_mse"][index] for row in group)
            for index in range(len(group[0]["per_horizon_mse"]))
        ]
        summary["per_horizon_mse_sd"] = [
            statistics.stdev(row["per_horizon_mse"][index] for row in group)
            if len(group) > 1
            else 0.0
            for index in range(len(group[0]["per_horizon_mse"]))
        ]
        summaries.append(summary)
    (output_directory / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "recovery_metric": (
                        "first sustained window within 10% (minimum tolerance one edge) "
                        f"of median SHD in the final {RECOVERY_REFERENCE_STEPS} "
                        "post-switch evaluation steps; candidate windows end before "
                        f"that reference window and require {SUSTAINED_RECOVERY_STEPS} "
                        "consecutive steps; retrospective evaluation-only reference"
                    ),
                    "campaign_source_sha256": progress.get("protocol", {}).get(
                        "source_sha256", {}
                    ),
                },
                "seed_rows": rows,
                "means_and_spreads": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_directory / "summary.csv"
    csv_fields = [
        key
        for key in summaries[0]
        if key not in {"per_horizon_mse_mean", "per_horizon_mse_sd"}
    ] if summaries else ["scenario", "variant", "n"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {key: summary.get(key) for key in csv_fields}
            )
    if summaries:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        variants = [
            "identity",
            "learned",
            "oracle",
            "dense",
            "random",
            "dlinear",
            "revin_linear",
            "var_direct",
        ]
        figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5), sharey=True)
        for axis, scenario in zip(axes, SCENARIOS, strict=True):
            for variant in variants:
                match = next(
                    (
                        item
                        for item in summaries
                        if item["scenario"] == scenario
                        and item["variant"] == variant
                    ),
                    None,
                )
                if match is not None:
                    axis.bar(
                        variant,
                        match["mse_mean"],
                        yerr=match["mse_sd"],
                        capsize=3,
                    )
            axis.set_title(scenario)
            axis.tick_params(axis="x", labelrotation=60)
            axis.set_ylabel("Forecast MSE (mean ± SD)")
        figure.tight_layout()
        figure.savefig(output_directory / "forecast_mse.png", dpi=160)
        plt.close(figure)
        curve_figure, curve_axes = plt.subplots(
            1, len(SCENARIOS), figsize=(18, 5), sharey=True
        )
        curve_variants = (
            "identity",
            "learned",
            "oracle",
            "dense",
            "random",
            "dlinear",
            "revin_linear",
        )
        for axis, scenario in zip(curve_axes, SCENARIOS, strict=True):
            for variant in curve_variants:
                traces = curves.get((scenario, variant), [])
                if not traces:
                    continue
                timestamps = traces[0][0]
                values = [trace for _, trace in traces]
                mean = [
                    statistics.fmean(trace[index] for trace in values)
                    for index in range(len(timestamps))
                ]
                spread = [
                    statistics.stdev(trace[index] for trace in values)
                    if len(values) > 1
                    else 0.0
                    for index in range(len(timestamps))
                ]
                axis.plot(timestamps, mean, label=variant)
                axis.fill_between(
                    timestamps,
                    [max(0.0, value - sd) for value, sd in zip(mean, spread, strict=True)],
                    [value + sd for value, sd in zip(mean, spread, strict=True)],
                    alpha=0.12,
                )
            switch = SCENARIOS[scenario]["switch_at"]
            if switch < 512:
                axis.axvline(switch, color="black", linestyle="--", linewidth=1)
            if scenario == "gradual":
                axis.axvspan(switch, switch + SCENARIOS[scenario]["blend_steps"], alpha=0.1)
            axis.set_title(scenario)
            axis.set_xlabel("Last observed input timestamp")
            axis.set_ylabel("Per-window MSE (mean ± SD)")
        curve_axes[-1].legend(fontsize="small")
        curve_figure.tight_layout()
        curve_figure.savefig(output_directory / "forecast_curves.png", dpi=160)
        plt.close(curve_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--seed-pairs",
        nargs="+",
        choices=("11:101", "17:202", "23:303"),
        default=["11:101", "17:202", "23:303"],
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/model_contribution"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory
    config_directory = output_directory / "configs"
    output_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    progress_path = output_directory / "resume_state.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"completed_runs": []}
    )
    progress["protocol"] = {
        "online_start": "cold_start",
        "training_scope": "test stream only after its observed timestamps; no future horizon labels",
        "source_sha256": {
            filename: _sha256(Path(filename)) for filename in SOURCE_FILES
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "command": (
            "python -m scripts.synthetic_evaluation --scenarios stationary abrupt gradual "
            "--seed-pairs 11:101 17:202 23:303"
        ),
        "resume_unit": "completed scenario/data-seed/model-seed JSON run",
    }
    pairs = [tuple(map(int, pair.split(":"))) for pair in arguments.seed_pairs]
    for scenario in arguments.scenarios:
        for data_seed, model_seed in pairs:
            stem = f"{scenario}_data{data_seed}_model{model_seed}"
            result_path = output_directory / f"{stem}.json"
            if any(
                run["scenario"] == scenario
                and run["data_seed"] == data_seed
                and run["model_seed"] == model_seed
                for run in progress["completed_runs"]
            ) and result_path.exists():
                continue
            config = yaml.safe_load(
                Path("configs/research_online.yaml").read_text(encoding="utf-8")
            )
            config["seed"] = model_seed
            config["data"]["seed"] = data_seed
            config["data"]["switch_at"] = SCENARIOS[scenario]["switch_at"]
            config["data"]["blend_steps"] = SCENARIOS[scenario]["blend_steps"]
            config_path = config_directory / f"{stem}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            started = time.perf_counter()
            result = run_experiment(config_path, result_path, device="cpu")
            elapsed = time.perf_counter() - started
            metadata = {
                "scenario": scenario,
                "data_seed": data_seed,
                "model_seed": model_seed,
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "elapsed_seconds": elapsed,
            }
            progress["completed_runs"] = [
                run
                for run in progress["completed_runs"]
                if not (
                    run["scenario"] == scenario
                    and run["data_seed"] == data_seed
                    and run["model_seed"] == model_seed
                )
            ]
            progress["completed_runs"].append(metadata)
            progress_path.write_text(
                json.dumps(progress, indent=2) + "\n", encoding="utf-8"
            )
            _write_summary(output_directory, progress)
            print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
