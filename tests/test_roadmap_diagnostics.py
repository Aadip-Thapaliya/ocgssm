"""Oracle-support OLS reference used in roadmap S1 diagnostics."""

from scripts.roadmap_s1_diagnostics import oracle_support_ols
from scripts.roadmap_s1 import _seed_data


def test_oracle_support_ols_is_full_rank_and_near_noise_floor() -> None:
    config = {
        "steps": 4000, "variables": 5, "lags": 2, "switch_at": 3600, "blend_steps": 0,
        "sparsity": 0.2, "noise_scale": 0.1, "cross_variable_scale": 1.0,
    }
    data = _seed_data(5, config)
    result = oracle_support_ols(data, horizon=2)
    assert result["rank_deficient_regressions"] == 0
    assert result["regressions"] == 10
    h1 = sum(row[0] for row in result["gate_errors"]) / len(result["gate_errors"])
    assert 0.007 < h1 < 0.013  # one-step noise variance is 0.01


def test_s3_convergence_gain() -> None:
    from scripts.roadmap_s3 import convergence

    result = convergence([1.0] * 19 + [0.9] + [0.8] * 5)
    assert result["best_first_20"] == 0.9 and result["best_all"] == 0.8
    assert abs(result["relative_gain"] - 1 / 9) < 1e-12
