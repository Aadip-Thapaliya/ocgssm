"""Benchmark scan-related forward, backward, and online CPU timings."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from ocg.model import CausalSSM
from ocg.train import train_online


@dataclass(frozen=True)
class SpeedRow:
    """Benchmark row for one variable count and implementation."""

    n_variables: int
    implementation: str
    forward_ms: float
    backward_ms: float
    online_ms_per_step: float


def _make_model(n_variables: int, implementation: str) -> CausalSSM:
    """Build a one-layer identity-graph model for windows shaped (B,L,D)."""
    return CausalSSM(
        n_variables=n_variables,
        state_dim=16,
        horizon=6,
        n_lags=2,
        n_layers=1,
        embedding_dim=8,
        graph_mode="identity",
        scan_implementation=implementation,
    )


def _time_forward(model: CausalSSM, inputs: torch.Tensor, repeats: int) -> float:
    """Return average forward latency in milliseconds for inputs (B,L,D)."""
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(inputs)
        started = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        return (time.perf_counter() - started) * 1000.0 / repeats


def _time_backward(model: CausalSSM, inputs: torch.Tensor, repeats: int) -> float:
    """Return average backward latency in milliseconds for inputs (B,L,D)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
    started = time.perf_counter()
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
    return (time.perf_counter() - started) * 1000.0 / repeats


def _time_online(model: CausalSSM, inputs: torch.Tensor, repeats: int) -> float:
    """Return prequential milliseconds per step for inputs (N,L,D)."""
    targets = torch.zeros(inputs.shape[0], 6, inputs.shape[2], dtype=inputs.dtype)
    started = time.perf_counter()
    train_online(
        model,
        inputs,
        targets,
        learning_rate=0.001,
        seed=17,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms / repeats


def benchmark(n_variables: int, repeats: int, online_steps: int, seed: int) -> list[SpeedRow]:
    """Benchmark reference and fast implementations for one variable dimension D."""
    torch.manual_seed(seed + n_variables)
    batch_inputs = torch.randn(32, 24, n_variables)
    online_inputs = torch.randn(online_steps, 24, n_variables)
    rows: list[SpeedRow] = []
    for implementation in ("reference", "fast"):
        model = _make_model(n_variables, implementation)
        rows.append(
            SpeedRow(
                n_variables=n_variables,
                implementation=implementation,
                forward_ms=_time_forward(model, batch_inputs, repeats),
                backward_ms=_time_backward(model, batch_inputs, repeats),
                online_ms_per_step=_time_online(model, online_inputs, online_steps),
            )
        )
    return rows


def _format_table(rows: list[SpeedRow]) -> str:
    """Format speed rows as a markdown table with before/after speedups."""
    by_dimension = {
        n_variables: {row.implementation: row for row in rows if row.n_variables == n_variables}
        for n_variables in sorted({row.n_variables for row in rows})
    }
    lines = [
        "| D | Metric | Before (reference) | After (fast) | Speedup |",
        "|---:|---|---:|---:|---:|",
    ]
    for n_variables in sorted(by_dimension):
        reference = by_dimension[n_variables]["reference"]
        fast = by_dimension[n_variables]["fast"]
        for metric, before, after in (
            ("Forward ms/batch", reference.forward_ms, fast.forward_ms),
            ("Backward ms/batch", reference.backward_ms, fast.backward_ms),
            ("Online ms/step", reference.online_ms_per_step, fast.online_ms_per_step),
        ):
            lines.append(
                f"| {n_variables} | {metric} | {before:.3f} | {after:.3f} | {before / after:.2f}x |"
            )
    return "\n".join(lines)


def main() -> int:
    """Run CPU benchmarks and print a markdown speed table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--online-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()

    torch.set_num_threads(1)
    rows = benchmark(5, arguments.repeats, arguments.online_steps, arguments.seed)
    rows.extend(benchmark(20, arguments.repeats, arguments.online_steps, arguments.seed))
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
