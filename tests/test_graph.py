"""Acceptance tests for online graph learning and regime detection."""

import torch

from ocg.data import generate_svar
from ocg.graph import GraphLearner, RegimeDetector


def _shd(prediction: torch.Tensor, truth: torch.Tensor) -> int:
    """Compute edge edit distance between (K,D,D) boolean tensors."""
    return int(torch.logical_xor(prediction, truth).sum().item())


def test_adjacency_shape_range_and_logit_gradients() -> None:
    """Lagged adjacency has expected dimensions and differentiable logits."""
    learner = GraphLearner(n_variables=5, n_lags=2)
    adjacency = learner.adjacency()
    assert adjacency.shape == (2, 5, 5)
    assert torch.all((adjacency >= 0) & (adjacency <= 1))

    values = torch.randn(4, 20, 5)
    loss = learner.structural_loss(values)
    loss.backward()
    assert learner.logits.grad is not None
    assert torch.isfinite(learner.logits.grad).all()
    assert learner.logits.grad.abs().sum().item() > 0


def test_online_updates_improve_stationary_svar_shd() -> None:
    """Three hundred graph updates improve SHD over initialization."""
    torch.manual_seed(0)
    data = generate_svar(
        n_steps=700,
        n_variables=10,
        n_lags=2,
        switch_at=700,
        sparsity=0.12,
        noise_scale=0.08,
        seed=19,
    )
    truth = data["graph"][0].bool()
    learner = GraphLearner(n_variables=10, n_lags=2)
    initial_shd = _shd(learner.adjacency() > 0.5, truth)

    for time_index in range(64, 364):
        learner.online_update(data["series"][time_index - 64 : time_index].unsqueeze(0))

    learned_shd = _shd(learner.adjacency() > 0.5, truth)
    assert learned_shd < initial_shd


def test_detector_switch_delay_and_stationary_false_alarms() -> None:
    """Detector catches a synthetic coefficient switch with under-five-percent alarms."""
    data = generate_svar(
        n_steps=900,
        n_variables=8,
        n_lags=1,
        switch_at=500,
        sparsity=0.3,
        noise_scale=0.1,
        seed=29,
    )
    fixed_coefficients = data["coefficients"][100]
    scores: list[float] = []
    for time_index in range(1, 900):
        prediction = fixed_coefficients[0] @ data["series"][time_index - 1]
        scores.append(torch.mean((data["series"][time_index] - prediction) ** 2).item())

    detector = RegimeDetector(decay=0.95, threshold=3.5, warmup=30)
    alarms: list[int] = []
    for index, score in enumerate(scores, start=1):
        if detector.update(score):
            alarms.append(index)

    stationary_alarms = sum(index < 500 for index in alarms)
    assert stationary_alarms / (500 - 30) < 0.05
    post_switch_alarms = [index for index in alarms if 500 <= index <= 550]
    assert post_switch_alarms
