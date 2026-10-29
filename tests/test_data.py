"""Acceptance tests for synthetic data, windowing, and RevIN."""

from pathlib import Path

import numpy as np
import torch

from ocg.data import (
    _companion_spectral_radius,
    generate_svar,
    load_csv,
    make_windows,
    revin_denormalize,
    revin_normalize,
)


def test_svar_shapes_and_stationary_graph_recovery() -> None:
    """Generated series and graph support OLS edge recovery."""
    result = generate_svar(
        n_steps=2400,
        n_variables=6,
        n_lags=2,
        switch_at=1600,
        sparsity=0.2,
        seed=11,
    )
    series = result["series"]
    graph = result["graph"]
    coefficients = result["coefficients"]

    assert series.shape == (2400, 6)
    assert graph.shape == (2400, 2, 6, 6)
    assert coefficients.shape == (2400, 2, 6, 6)

    start, stop = 250, 1500
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for time_index in range(start + 2, stop):
        rows.append(torch.cat([series[time_index - lag] for lag in (1, 2)]))
        targets.append(series[time_index])
    design = torch.stack(rows)
    target = torch.stack(targets)
    estimate = torch.linalg.lstsq(design, target).solution.T.reshape(6, 2, 6).permute(1, 0, 2)
    recovered = estimate.abs() > 0.12
    truth = graph[start:stop].mean(dim=0) > 0.5
    true_positive = (recovered & truth).sum().item()
    f1 = 2 * true_positive / (recovered.sum().item() + truth.sum().item())
    assert f1 > 0.8
    assert not torch.equal(coefficients[100], coefficients[-100])


def test_gradual_switch_and_window_shapes() -> None:
    """Gradual data switches coefficients smoothly and windows match requested shapes."""
    data = generate_svar(
        n_steps=300,
        n_variables=3,
        n_lags=1,
        switch_at=150,
        blend_steps=30,
        seed=3,
    )
    assert data["series"].shape == (300, 3)
    assert data["graph"].shape == (300, 1, 3, 3)
    assert not torch.equal(data["coefficients"][145], data["coefficients"][165])

    inputs, targets = make_windows(data["series"], input_length=32, horizon=5)
    assert inputs.shape == (264, 32, 3)
    assert targets.shape == (264, 5, 3)
    torch.testing.assert_close(inputs[0, -1], data["series"][31])
    torch.testing.assert_close(targets[0, 0], data["series"][32])


def test_svar_coefficients_remain_stable_through_blends() -> None:
    """Every generated VAR regime and gradual blend stays below radius 0.9."""
    for dimensions, seed in ((5, 3), (20, 0)):
        data = generate_svar(
            n_steps=160,
            n_variables=dimensions,
            n_lags=2,
            switch_at=40,
            blend_steps=80,
            sparsity=0.2,
            seed=seed,
        )
        maximum_radius = max(
            _companion_spectral_radius(coefficients)
            for coefficients in data["coefficients"]
        )
        assert maximum_radius < 0.9


def test_cross_variable_strengthening_is_opt_in_and_stable() -> None:
    """The stronger synthetic version preserves defaults and checks every regime."""
    arguments = {
        "n_steps": 512,
        "n_variables": 5,
        "n_lags": 2,
        "switch_at": 320,
        "blend_steps": 32,
        "sparsity": 0.2,
        "noise_scale": 0.1,
        "seed": 17,
    }
    original = generate_svar(**arguments)
    default_strength = generate_svar(**arguments, cross_variable_scale=1.0)
    strengthened = generate_svar(**arguments, cross_variable_scale=1.5)
    for key in ("series", "graph", "coefficients"):
        torch.testing.assert_close(original[key], default_strength[key], rtol=0, atol=0)

    cross_variable_mask = 1.0 - torch.eye(5)
    original_cross_magnitude = (
        original["coefficients"][0].abs() * cross_variable_mask
    ).sum()
    strengthened_cross_magnitude = (
        strengthened["coefficients"][0].abs() * cross_variable_mask
    ).sum()
    assert strengthened_cross_magnitude > original_cross_magnitude
    maximum_radius = max(
        _companion_spectral_radius(coefficients)
        for coefficients in strengthened["coefficients"]
    )
    assert maximum_radius < 0.89


def test_revin_round_trip() -> None:
    """RevIN normalization and inversion preserve (B,L,D) tensors."""
    values = torch.randn(12, 48, 5)
    normalized, stats = revin_normalize(values)
    restored = revin_denormalize(normalized, stats)
    assert normalized.shape == values.shape
    assert restored.shape == values.shape
    assert torch.max(torch.abs(restored - values)).item() < 1e-5


def test_load_csv(tmp_path: Path) -> None:
    """CSV loader returns selected numeric columns as (T,D) tensors."""
    path = tmp_path / "series.csv"
    np.savetxt(path, np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]]), delimiter=",", header="a,b,c", comments="")
    series = load_csv(path, columns=["a", "c"])
    assert series.shape == (2, 2)
    torch.testing.assert_close(series, torch.tensor([[1.0, 7.0], [2.0, 8.0]]))
