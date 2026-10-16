"""Forecasting baselines used to compare against the graph-conditioned SSM."""

import torch
from torch import Tensor, nn

from ocg.data import revin_denormalize, revin_normalize
from ocg.model import CausalSSM


def predict_last_value(values: Tensor, horizon: int) -> Tensor:
    """Repeat the final value in each (B,L,D) input for an (B,H,D) forecast."""
    if values.ndim != 3 or horizon < 1 or values.shape[1] < 1:
        raise ValueError("values must be (B,L,D) and horizon must be positive")
    return values[:, -1:, :].expand(-1, horizon, -1)


def _mean_forecast(values: Tensor, horizon: int) -> Tensor:
    """Repeat each variable's window mean from (B,L,D) as (B,H,D)."""
    return values.mean(dim=1, keepdim=True).expand(-1, horizon, -1)


class DLinear(nn.Module):
    """Decomposition-linear forecaster mapping (B,L,D) inputs to (B,H,D)."""

    def __init__(
        self,
        input_length: int,
        horizon: int,
        n_variables: int,
        moving_average: int = 3,
    ) -> None:
        """Initialize per-variable trend and seasonal linear projections."""
        super().__init__()
        if input_length < 1 or horizon < 1 or n_variables < 1:
            raise ValueError("model dimensions must be positive")
        if moving_average < 1 or moving_average % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        self.input_length = input_length
        self.horizon = horizon
        self.n_variables = n_variables
        self.moving_average = moving_average
        self.trend_projection = nn.Linear(input_length, horizon)
        self.seasonal_projection = nn.Linear(input_length, horizon)
        with torch.no_grad():
            self.trend_projection.weight.fill_(1.0 / input_length)
            self.trend_projection.bias.zero_()
            self.seasonal_projection.weight.zero_()
            self.seasonal_projection.bias.zero_()

    def forward(self, values: Tensor) -> Tensor:
        """Return a decomposition-linear forecast for (B,L,D) values."""
        if values.ndim != 3 or values.shape[1:] != (self.input_length, self.n_variables):
            raise ValueError("values must have shape (B,input_length,n_variables)")
        radius = self.moving_average // 2
        channels_first = values.transpose(1, 2)
        padded = nn.functional.pad(channels_first, (radius, radius), mode="replicate")
        trend = nn.functional.avg_pool1d(padded, kernel_size=self.moving_average, stride=1)
        seasonal = channels_first - trend
        forecast = self.trend_projection(trend).transpose(1, 2)
        forecast = forecast + self.seasonal_projection(seasonal).transpose(1, 2)
        return forecast


class RevINLinear(nn.Module):
    """Per-window RevIN followed by a shared linear time projection."""

    def __init__(self, input_length: int, horizon: int, n_variables: int) -> None:
        """Initialize a normalized linear forecaster for (B,L,D) windows."""
        super().__init__()
        if input_length < 1 or horizon < 1 or n_variables < 1:
            raise ValueError("model dimensions must be positive")
        self.input_length = input_length
        self.horizon = horizon
        self.n_variables = n_variables
        self.projection = nn.Linear(input_length, horizon)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: Tensor) -> Tensor:
        """Predict (B,H,D) from normalized (B,L,D) windows and invert RevIN."""
        if values.ndim != 3 or values.shape[1:] != (self.input_length, self.n_variables):
            raise ValueError("values must have shape (B,input_length,n_variables)")
        normalized, stats = revin_normalize(values)
        prediction = self.projection(normalized.transpose(1, 2)).transpose(1, 2)
        return revin_denormalize(prediction, {"mean": stats["mean"], "std": stats["std"]})


class TransformerLite(nn.Module):
    """Small temporal Transformer residual forecaster with a mean-value skip."""

    def __init__(
        self,
        input_length: int,
        horizon: int,
        n_variables: int,
        model_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        """Initialize a lightweight encoder and (B,H,D) forecast head."""
        super().__init__()
        if input_length < 1 or horizon < 1 or n_variables < 1 or model_dim < 1:
            raise ValueError("model dimensions must be positive")
        if n_heads < 1 or model_dim % n_heads:
            raise ValueError("model_dim must be divisible by positive n_heads")
        self.input_length = input_length
        self.horizon = horizon
        self.n_variables = n_variables
        self.input_projection = nn.Linear(n_variables, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=2 * model_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.forecast_head = nn.Linear(model_dim, horizon * n_variables)
        nn.init.zeros_(self.forecast_head.weight)
        nn.init.zeros_(self.forecast_head.bias)

    def forward(self, values: Tensor) -> Tensor:
        """Return a Transformer residual around the mean forecast (B,H,D)."""
        if values.ndim != 3 or values.shape[1:] != (self.input_length, self.n_variables):
            raise ValueError("values must have shape (B,input_length,n_variables)")
        encoded = self.encoder(self.input_projection(values))
        residual = self.forecast_head(encoded.mean(dim=1))
        residual = residual.reshape(-1, self.horizon, self.n_variables)
        return _mean_forecast(values, self.horizon) + residual


class NoGraphSSM(nn.Module):
    """CausalSSM forecasting baseline with graph interactions disabled."""

    def __init__(
        self,
        n_variables: int,
        state_dim: int,
        horizon: int,
        n_lags: int,
        n_layers: int = 1,
        embedding_dim: int = 8,
    ) -> None:
        """Initialize an identity-graph SSM with a mean forecast skip."""
        super().__init__()
        self.horizon = horizon
        self.ssm = CausalSSM(
            n_variables=n_variables,
            state_dim=state_dim,
            horizon=horizon,
            n_lags=n_lags,
            n_layers=n_layers,
            embedding_dim=embedding_dim,
            graph_mode="identity",
        )
        nn.init.zeros_(self.ssm.head.weight)
        nn.init.zeros_(self.ssm.head.bias)

    def forward(self, values: Tensor) -> Tensor:
        """Forecast (B,H,D) using identity-graph SSM residuals and a mean skip."""
        return _mean_forecast(values, self.horizon) + self.ssm(values)


def train_one_epoch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    learning_rate: float,
) -> float:
    """Train on (N,L,D) inputs and (N,H,D) targets; return mean epoch MSE."""
    if inputs.ndim != 3 or targets.ndim != 3 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have shapes (N,L,D) and (N,H,D)")
    if inputs.shape[0] < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("training data, batch_size, and learning_rate must be positive")
    if inputs.shape[2] != targets.shape[2]:
        raise ValueError("inputs and targets must have the same variable dimension")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_sum = 0.0
    n_examples = inputs.shape[0]
    for start in range(0, n_examples, batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(batch_inputs), batch_targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item() * batch_inputs.shape[0]
    return loss_sum / n_examples
