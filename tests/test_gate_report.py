"""Tests for synthetic gate result reporting."""

from typing import Any

from scripts.gate_report import summarize_results


def _results() -> dict[str, Any]:
    """Build an ordered result fixture with enough windows for the early/late periods."""
    errors = [float(index) for index in range(704)]
    return {
        "evaluation": {
            "switch_at": 108,
            "test_start": 100,
            "input_length": 5,
        },
        "variants": {
            "learned": {
                "forecast": {"per_window_mse": errors},
                "graph": {"shd": 2, "auroc": 0.75},
                "adaptation": {"detection_delay": 3},
            },
            "identity": {
                "forecast": {"per_window_mse": errors},
                "graph": {
                    "available": False,
                    "reason": "identity mode has no trained graph logits",
                },
            },
        },
    }


def test_gate_summary_uses_switch_and_test_window_alignment() -> None:
    """Split windows at switch_at relative to the first test window's last input."""
    report = summarize_results(_results())

    assert report["learned"]["overall_mse"] == 351.5
    assert report["learned"]["pre_switch_mse"] == 1.5
    assert report["learned"]["post_switch_early_mse"] == 103.5
    assert report["learned"]["post_switch_late_mse"] == 503.5
    assert report["learned"]["shd"] == 2
    assert report["learned"]["auroc"] == 0.75
    assert report["learned"]["detection_delay"] == 3


def test_gate_summary_marks_unavailable_graph_metrics() -> None:
    """Identity graph scores remain empty when no graph logits were trained."""
    report = summarize_results(_results())

    assert report["identity"]["shd"] is None
    assert report["identity"]["auroc"] is None
    assert report["identity"]["detection_delay"] is None


def test_gate_summary_handles_switch_before_test_start() -> None:
    """A test split entirely after the switch has no pre-switch window MSE."""
    results = _results()
    results["evaluation"]["switch_at"] = 90

    report = summarize_results(results)

    assert report["learned"]["pre_switch_mse"] is None
    assert report["learned"]["post_switch_early_mse"] == 99.5
    assert report["learned"]["post_switch_late_mse"] == 503.5
