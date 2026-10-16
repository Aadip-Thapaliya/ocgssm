"""Acceptance tests for forecasting baselines."""

import pytest
import torch

from ocg.baselines import (
    DLinear,
    NoGraphSSM,
    RevINLinear,
    TransformerLite,
    predict_last_value,
    train_one_epoch,
)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: DLinear(input_length=8, horizon=2, n_variables=3),
        lambda: RevINLinear(input_length=8, horizon=2, n_variables=3),
        lambda: TransformerLite(input_length=8, horizon=2, n_variables=3, model_dim=8, n_heads=2),
        lambda: NoGraphSSM(n_variables=3, state_dim=4, horizon=2, n_lags=1),
    ],
)
def test_baseline_trains_one_epoch_and_beats_last_value(model_factory: object) -> None:
    """One synthetic training epoch beats the held-out predict-last baseline."""
    torch.manual_seed(13)
    train_inputs = torch.randn(96, 8, 3)
    train_targets = train_inputs.mean(dim=1, keepdim=True).expand(-1, 2, -1)
    test_inputs = torch.randn(32, 8, 3)
    test_targets = test_inputs.mean(dim=1, keepdim=True).expand(-1, 2, -1)
    model = model_factory()

    train_one_epoch(model, train_inputs, train_targets, batch_size=96, learning_rate=0.001)
    prediction = model(test_inputs)
    baseline = predict_last_value(test_inputs, horizon=2)
    model_error = torch.nn.functional.mse_loss(prediction, test_targets)
    baseline_error = torch.nn.functional.mse_loss(baseline, test_targets)

    assert prediction.shape == (32, 2, 3)
    assert model_error.item() < baseline_error.item()
