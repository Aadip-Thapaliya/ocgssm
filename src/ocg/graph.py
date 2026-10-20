"""Differentiable lagged graph estimation and online regime detection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GraphLearner(nn.Module):
    """Learn lagged adjacency and masked VAR weights from (B,L,D) windows."""

    def __init__(
        self,
        n_variables: int,
        n_lags: int,
        learning_rate: float = 0.01,
        sparsity_weight: float = 0.001,
        temperature: float = 1.0,
        ema_decay: float = 0.95,
        graph_stride: int = 1,
        detector: RegimeDetector | None = None,
        boost: float = 3.0,
        boost_steps: int = 20,
    ) -> None:
        """Initialize graph logits and structural VAR weights."""
        super().__init__()
        if n_variables < 1 or n_lags < 1:
            raise ValueError("n_variables and n_lags must be positive")
        if learning_rate <= 0 or sparsity_weight < 0 or temperature <= 0:
            raise ValueError("learning rate and temperature must be positive; sparsity weight nonnegative")
        if not 0.0 <= ema_decay < 1.0 or graph_stride < 1:
            raise ValueError("ema_decay must be in [0,1) and graph_stride positive")
        if boost < 1.0 or boost_steps < 1:
            raise ValueError("boost must be at least one and boost_steps positive")
        self.n_variables = n_variables
        self.n_lags = n_lags
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.graph_stride = graph_stride
        self.detector = detector if detector is not None else RegimeDetector()
        self.boost = boost
        self.boost_steps = boost_steps
        self._base_learning_rate = learning_rate
        self._boost_remaining = 0
        self.logits = nn.Parameter(torch.full((n_lags, n_variables, n_variables), -0.3))
        self.weights = nn.Parameter(torch.empty(n_lags, n_variables, n_variables))
        nn.init.normal_(self.weights, mean=0.0, std=0.2)
        self.register_buffer("ema_logits", self.logits.detach().clone())
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self._steps = 0
        self.last_detection = False

    def adjacency(self, use_ema: bool = True) -> Tensor:
        """Return lagged edge probabilities (K,D,D) in the closed interval [0,1]."""
        logits = self.ema_logits if use_ema else self.logits
        return torch.sigmoid(logits / self.temperature)

    def aggregate_adjacency(self, use_ema: bool = True) -> Tensor:
        """Combine lagged probabilities into an aggregate adjacency (D,D)."""
        adjacency = self.adjacency(use_ema)
        return 1.0 - torch.prod(1.0 - adjacency, dim=0)

    def structural_loss(self, values: Tensor) -> Tensor:
        """Compute masked VAR score for input windows with shape (B,L,D)."""
        if values.ndim != 3 or values.shape[-1] != self.n_variables:
            raise ValueError("values must have shape (B,L,D) with configured D")
        if values.shape[1] <= self.n_lags:
            raise ValueError("window length must exceed n_lags")
        target = values[:, self.n_lags :, :]
        adjacency = torch.sigmoid(self.logits / self.temperature)
        prediction = torch.zeros_like(target)
        for lag_index in range(self.n_lags):
            lag = lag_index + 1
            previous = values[:, self.n_lags - lag : values.shape[1] - lag, :]
            effective_weights = adjacency[lag_index] * self.weights[lag_index]
            prediction = prediction + torch.einsum("bti,oi->bto", previous, effective_weights)
        residual_loss = torch.mean((target - prediction) ** 2)
        sparsity_loss = self.sparsity_weight * adjacency.mean()
        return residual_loss + sparsity_loss

    def online_update(self, values: Tensor) -> float:
        """Apply one online optimizer step to a batch of (B,L,D) windows."""
        self._steps += 1
        should_step = self._steps % self.graph_stride == 0
        if should_step:
            loss = self.structural_loss(values)
        else:
            with torch.no_grad():
                loss = self.structural_loss(values)
        self.last_detection = self.detector.update(loss.detach())
        if self.last_detection:
            self._boost_remaining = self.boost_steps
        if should_step:
            learning_rate = self._base_learning_rate
            if self._boost_remaining > 0:
                learning_rate *= self.boost
            for parameter_group in self._optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
            with torch.no_grad():
                self.ema_logits.mul_(self.ema_decay).add_(self.logits, alpha=1.0 - self.ema_decay)
            self._boost_remaining = max(0, self._boost_remaining - 1)
        return float(loss.detach().item())


class RegimeDetector:
    """Detect unusually high structural scores using running moments."""

    def __init__(
        self,
        decay: float = 0.95,
        threshold: float = 3.5,
        warmup: int = 30,
        min_std: float = 1e-6,
    ) -> None:
        """Configure the score EMA, z-score threshold, and warmup period."""
        if not 0.0 <= decay < 1.0 or threshold <= 0 or warmup < 2 or min_std <= 0:
            raise ValueError("invalid regime detector configuration")
        self.decay = decay
        self.threshold = threshold
        self.warmup = warmup
        self.min_std = min_std
        self.count = 0
        self.mean = 0.0
        self.variance = 0.0

    def update(self, score: float | Tensor) -> bool:
        """Consume a scalar score and return whether it exceeds the running threshold."""
        value = float(score.detach().item()) if isinstance(score, Tensor) else float(score)
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        self.count += 1
        if self.count == 1:
            self.mean = value
            return False
        if self.count <= self.warmup:
            delta = value - self.mean
            self.mean += delta / self.count
            self.variance = (
                (self.count - 2) * self.variance + delta * (value - self.mean)
            ) / (self.count - 1)
            return False

        std = max(math.sqrt(max(self.variance, 0.0)), self.min_std)
        detected = value > self.mean + self.threshold * std
        delta = value - self.mean
        self.mean = self.decay * self.mean + (1.0 - self.decay) * value
        self.variance = self.decay * self.variance + (1.0 - self.decay) * delta * (value - self.mean)
        return detected
