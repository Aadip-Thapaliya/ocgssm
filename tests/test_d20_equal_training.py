"""Tests for the gated D20 equal-training evaluation harness."""

import json

import pytest
import torch

from ocg.train import train_online
from scripts import d20_equal_training
from scripts.d20_equal_training import (
    _build,
    _build_heldout_report,
    _build_reduced_report,
    _development_data,
    _paired_summary,
    _write_json,
    run_all,
)


def test_reduced_runner_binds_online_trainer() -> None:
    assert d20_equal_training.train_online is train_online


def test_campaign_json_writer_serializes_tensor_values(tmp_path) -> None:
    output = tmp_path / "campaign.json"

    _write_json(
        output,
        {"scalar": torch.tensor(0.5), "series": torch.tensor([1.0, 2.0])},
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "scalar": 0.5,
        "series": [1.0, 2.0],
    }


def test_d20_development_subsplits_have_frozen_window_counts() -> None:
    data = _development_data()

    assert len(data["train"][0]) == 2371
    assert len(data["tune"][0]) == 237
    assert len(data["threshold"][0]) == 238
    assert len(data["gate"][0]) == 238


def test_d20_learned_hard_variant_builds_with_fixed_threshold() -> None:
    model = _build(
        "learned_hard",
        forecast_lr=0.001,
        graph_lr=0.01,
        model_seed=401,
        graph_mask_threshold=0.4,
    )

    assert model.graph_mode == "learned"
    assert model.graph_mask_threshold == 0.4


def test_d20_full_heldout_runner_requires_new_approval() -> None:
    with pytest.raises(RuntimeError, match="pending approval"):
        run_all()


def test_d20_paired_summary_keeps_seed_level_statistics() -> None:
    runs = []
    for data_seed, candidate_mse in ((37, 0.9), (43, 0.8), (47, 0.7)):
        runs.append(
            {
                "data_seed": data_seed,
                "variants": {
                    "candidate": {
                        "forecast": {
                            "per_window_per_horizon_mse": [
                                [candidate_mse, candidate_mse] for _ in range(24)
                            ]
                        }
                    },
                    "reference": {
                        "forecast": {
                            "per_window_per_horizon_mse": [
                                [1.0, 1.0] for _ in range(24)
                            ]
                        }
                    },
                },
            }
        )

    result = _paired_summary(runs, "candidate", "reference", 0, ratio=True)

    assert result["seed_statistics"] == pytest.approx([0.9, 0.8, 0.7])
    assert result["equal_seed_mean"] == pytest.approx(0.8)
    assert result["paired_circular_block_bootstrap_95pct_ci"] == pytest.approx(
        [0.8, 0.8]
    )
    assert len(result["seed_paired_circular_block_bootstrap_95pct_ci"]) == 3


def test_d20_heldout_report_retains_all_horizons_and_gates_only_h1_h2() -> None:
    variants = (
        "identity",
        "dlinear",
        "var_direct",
        "dense",
        "random",
        "learned",
        "oracle",
        "learned_hard",
    )
    runs = []
    for data_seed, model_seed in ((37, 401), (43, 503), (47, 607)):
        run_variants = {}
        for variant_index, variant in enumerate(variants):
            run_variants[variant] = {
                "forecast": {
                    "per_horizon_mse": [
                        1.0 + variant_index * 0.1 + horizon * 0.01
                        for horizon in range(6)
                    ],
                    "per_window_per_horizon_mse": [
                        [
                            1.0 + variant_index * 0.1 + horizon * 0.01
                            for horizon in range(6)
                        ]
                        for _ in range(20)
                    ],
                }
            }
        runs.append(
            {
                "data_seed": data_seed,
                "model_seed": model_seed,
                "variants": run_variants,
            }
        )

    report = _build_heldout_report(runs)

    assert list(report["per_horizon"]) == ["1", "2", "3", "4", "5", "6"]
    assert set(report["per_seed_horizon_mse"]["37_401"]) == set(variants)
    assert all(
        len(values) == 6
        for values in report["per_seed_horizon_mse"]["37_401"].values()
    )
    assert report["point_gates"]["oracle_within_5pct_of_var_direct"].keys() == {
        "1",
        "2",
    }


def test_d20_reduced_report_keeps_seed_mean_median_and_side_check_separate() -> None:
    variants = ("identity", "dense", "learned", "oracle", "var_direct")
    runs = []
    for data_seed, model_seed, scale in ((37, 401, 1.0), (43, 503, 0.9), (47, 607, 1.1)):
        run_variants = {}
        for variant_index, variant in enumerate(variants):
            per_horizon = [
                [scale * (1.0 + variant_index * 0.1 + horizon * 0.01)] * 12
                for horizon in range(6)
            ]
            run_variants[variant] = {
                "pretraining": {
                    "best_validation_epoch": 3,
                    "epochs_executed": 8,
                    "epoch_cap": 20,
                    "convergence_status": "stopped_by_validation_patience",
                },
                "forecast": {
                    "per_horizon_mse": [
                        float(sum(row[h] for row in per_horizon) / len(per_horizon))
                        for h in range(6)
                    ],
                    "per_window_per_horizon_mse": per_horizon,
                },
            }
        runs.append(
            {
                "data_seed": data_seed,
                "model_seed": model_seed,
                "variants": run_variants,
            }
        )
    side_check = {"side_check": "oracle_40_epoch_cap", "data_seed": 37}

    report = _build_reduced_report(runs, side_check)

    assert report["variant_summary"]["learned"]["H1"]["mean_of_seed_means"] > 0
    assert report["variant_summary"]["learned"]["H1"]["median_of_seed_medians"] > 0
    assert report["per_seed"]["37_401"]["learned"]["best_validation_epoch"] == 3
    assert report["oracle_40_epoch_side_check"] == side_check
    assert report["side_check_excluded_from_primary_gates"] is True
    for contrast in ("learned_vs_dense", "learned_vs_oracle", "oracle_vs_var_direct"):
        assert len(
            report["contrasts"][contrast]["H1"]["paired_mean_mse_difference"][
                "paired_circular_block_bootstrap_95pct_ci"
            ]
        ) == 2
        assert len(
            report["contrasts"][contrast]["H2"]["paired_mse_ratio"][
                "paired_circular_block_bootstrap_95pct_ci"
            ]
        ) == 2
