"""Synthetic and real-world time-series data utilities."""

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor


def _stable_coefficients(
    graph: Tensor,
    generator: torch.Generator,
    max_radius: float = 0.8,
) -> Tensor:
    """Build stable lag matrices for an adjacency tensor (K,D,D)."""
    coefficients = graph * torch.where(
        torch.rand(graph.shape, generator=generator) < 0.5,
        -1.0,
        1.0,
    ) * (0.35 + 0.2 * torch.rand(graph.shape, generator=generator))
    n_lags, n_variables, _ = graph.shape
    companion = torch.zeros(n_variables * n_lags, n_variables * n_lags)
    companion[:n_variables] = torch.cat(tuple(coefficients), dim=1)
    if n_lags > 1:
        companion[n_variables:, :-n_variables] = torch.eye(n_variables * (n_lags - 1))
    radius = torch.linalg.eigvals(companion.to(torch.float64)).abs().max().item()
    if radius >= max_radius:
        coefficients = coefficients * (max_radius / radius)
    return coefficients


def generate_svar(
    n_steps: int,
    n_variables: int,
    n_lags: int,
    switch_at: int,
    blend_steps: int = 0,
    sparsity: float = 0.2,
    noise_scale: float = 0.1,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Generate a stable switching SVAR.

    Returns `series` with shape (T,D), `graph` with shape (T,K,D,D), and
    `coefficients` with shape (T,K,D,D).
    """
    if n_steps <= n_lags or n_variables < 1 or n_lags < 1:
        raise ValueError("n_steps must exceed positive n_lags and n_variables")
    if not 0 <= switch_at <= n_steps or blend_steps < 0 or switch_at + blend_steps > n_steps:
        raise ValueError("switch and blend must fit within n_steps")
    if not 0.0 <= sparsity <= 1.0 or noise_scale <= 0:
        raise ValueError("sparsity must be in [0,1] and noise_scale must be positive")

    generator = torch.Generator().manual_seed(seed)
    graph_one = (torch.rand(n_lags, n_variables, n_variables, generator=generator) < sparsity).float()
    graph_two = (torch.rand(n_lags, n_variables, n_variables, generator=generator) < sparsity).float()
    if torch.equal(graph_one, graph_two):
        graph_two[0, 0, 0] = 1.0 - graph_two[0, 0, 0]
    coefficients_one = _stable_coefficients(graph_one, generator)
    coefficients_two = _stable_coefficients(graph_two, generator)

    coefficients = torch.empty(n_steps, n_lags, n_variables, n_variables)
    graphs = torch.empty_like(coefficients)
    for time_index in range(n_steps):
        if blend_steps > 0 and switch_at <= time_index < switch_at + blend_steps:
            weight = (time_index - switch_at + 1) / blend_steps
        else:
            weight = float(time_index >= switch_at + blend_steps)
        coefficients[time_index] = torch.lerp(coefficients_one, coefficients_two, weight)
        graphs[time_index] = torch.lerp(graph_one, graph_two, weight)

    series = torch.zeros(n_steps, n_variables)
    noise = torch.randn(n_steps, n_variables, generator=generator) * noise_scale
    series[:n_lags] = noise[:n_lags]
    for time_index in range(n_lags, n_steps):
        prediction = torch.zeros(n_variables)
        for lag in range(n_lags):
            prediction += coefficients[time_index, lag] @ series[time_index - lag - 1]
        series[time_index] = prediction + noise[time_index]
    return {"series": series, "graph": graphs, "coefficients": coefficients}


def load_csv(path: str | Path, columns: list[str] | None = None) -> Tensor:
    """Load numeric series from a CSV path into a (T,D) float tensor."""
    frame = pd.read_csv(path)
    if columns is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing requested columns: {missing}")
        frame = frame.loc[:, columns]
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError("CSV contains no numeric time-series columns")
    if not torch.isfinite(torch.as_tensor(numeric.to_numpy())).all():
        raise ValueError("CSV series contains non-finite values")
    return torch.as_tensor(numeric.to_numpy(), dtype=torch.float32)


def make_windows(series: Tensor, input_length: int, horizon: int) -> tuple[Tensor, Tensor]:
    """Create aligned input (N,L,D) and target (N,H,D) windows from (T,D)."""
    if series.ndim != 2:
        raise ValueError("series must have shape (T,D)")
    if input_length < 1 or horizon < 1 or series.shape[0] < input_length + horizon:
        raise ValueError("input_length and horizon must be positive and fit in series")
    windows = series.unfold(0, input_length + horizon, 1).permute(0, 2, 1)
    return windows[:, :input_length], windows[:, input_length:]


def revin_normalize(
    values: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Normalize (B,L,D) per window and return reversible mean/std statistics."""
    if values.ndim != 3:
        raise ValueError("values must have shape (B,L,D)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (values - mean) / std, {"mean": mean, "std": std}


def revin_denormalize(values: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Invert RevIN for values and statistics with shape (B,L,D)/(B,1,D)."""
    if values.ndim != 3 or set(stats) != {"mean", "std"}:
        raise ValueError("values must be (B,L,D) and stats must contain mean and std")
    return values * stats["std"] + stats["mean"]
