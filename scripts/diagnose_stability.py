"""Per-epoch stability diagnostics for OCG graph variants on a config.

Logs train loss, per-group gradient norms, hidden-state magnitude, the
spectral radius of the effective one-step state transition, and the largest
masked lag-mixer row sum, so the first quantity to blow up can be identified.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ocg.model import CausalSSM
from ocg.run import (
    _build_model,
    _load_config,
    _load_series,
    _mapping,
    _model_variants,
    _split_windows,
)
from ocg.train import _regime_ids_from_graphs, _set_oracle_graph

PARAMETER_GROUPS: tuple[str, ...] = (
    "lag_input_weights",
    "lag_forecast_weights",
    "coupling_projection",
    "gate_projection",
    "dt_projection",
    "input_projection",
    "output_projection",
    "logA",
    "node_embeddings",
    "graph_encoder",
    "head",
    "graph_learner",
    "other",
)


def parameter_group(name: str) -> str:
    """Map a parameter name to one of PARAMETER_GROUPS."""
    for group in PARAMETER_GROUPS[:-1]:
        if group in name:
            return group
    return "other"


def grad_norms_by_group(model: nn.Module) -> dict[str, float]:
    """Return the L2 gradient norm per parameter group (0.0 for groups without grads)."""
    squared: dict[str, float] = {group: 0.0 for group in PARAMETER_GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared[parameter_group(name)] += float(parameter.grad.detach().square().sum().item())
    return {group: math.sqrt(value) for group, value in squared.items()}


def effective_transition_matrices(
    model: CausalSSM,
    values: Tensor,
    layer_index: int = 0,
) -> Tensor:
    """Build one-step transitions (B,L,D*N,D*N) for state vec(S) with S shaped (D,N).

    The recurrence is ``S' = decay * S + (1 - decay) * gate * (A_cross @ S @ W^T) + input``,
    so the linear part in flattened ``(d,n)`` order is
    ``diag(vec(decay)) + diag(vec((1 - decay) * gate)) (A_cross ⊗ W)``.
    """
    layer = model.layers[layer_index]
    adjacency = model._adjacency(values).to(dtype=values.dtype)
    layer_input = model._mix_lagged_inputs(values)
    graph_features, cross_adjacency = layer._graph_features(adjacency)
    node_features = graph_features.unsqueeze(0).unsqueeze(0).expand(
        layer_input.shape[0], layer_input.shape[1], -1, -1
    )
    features = torch.cat((layer_input.unsqueeze(-1), node_features), dim=-1)
    dt = nn.functional.softplus(layer.dt_projection(features))
    decay = torch.exp(-dt * torch.exp(layer.logA).view(1, 1, 1, -1))
    gate = torch.sigmoid(layer.gate_projection(features))
    batch_size, sequence_length, n_variables, state_dim = decay.shape
    weight = layer._effective_coupling_weight()
    coupling_gate = (1.0 - decay) * gate
    coupling_part = torch.einsum(
        "blij,blim,mn->blimjn",
        cross_adjacency.expand(batch_size, sequence_length, -1, -1),
        coupling_gate,
        weight,
    ).reshape(
        batch_size, sequence_length, n_variables * state_dim, n_variables * state_dim
    )
    return torch.diag_embed(decay.reshape(batch_size, sequence_length, -1)) + coupling_part


def transition_spectral_radius(model: CausalSSM, values: Tensor) -> dict[str, float]:
    """Return max/mean spectral radius over (B,L) one-step transitions for values (B,L,D)."""
    with torch.no_grad():
        matrices = effective_transition_matrices(model, values).to(torch.float64)
        radii = torch.linalg.eigvals(matrices).abs().amax(dim=-1)
    return {"max": float(radii.max().item()), "mean": float(radii.mean().item())}


def hidden_state_max_abs(model: CausalSSM, values: Tensor) -> float:
    """Return max |hidden state| over a window batch (B,L,D) for the first layer."""
    with torch.no_grad():
        adjacency = model._adjacency(values)
        layer_input = model._mix_lagged_inputs(values)
        hidden, _ = model.layers[0](layer_input, adjacency)
    return float(hidden.abs().max().item())


def masked_lag_row_sums(model: CausalSSM, values: Tensor) -> dict[str, float]:
    """Return the largest row sum of |weights * adjacency| for the lag mixer and lag readout."""
    with torch.no_grad():
        adjacency = model._lagged_adjacency(values)
        input_rows = (model.lag_input_weights * adjacency).abs().sum(dim=-1)
        forecast_rows = (model.lag_forecast_weights * adjacency.unsqueeze(0)).abs().sum(dim=-1)
    return {
        "lag_input_max_row_sum": float(input_rows.max().item()),
        "lag_forecast_max_row_sum": float(forecast_rows.max().item()),
    }


def _group_batches(
    n_examples: int,
    regime_ids: Tensor | None,
    device: torch.device,
) -> list[Tensor]:
    """Return index groups so no batch spans two oracle graph regimes."""
    if regime_ids is None:
        return [torch.arange(n_examples, device=device)]
    return [
        torch.nonzero(regime_ids == regime, as_tuple=False).flatten()
        for regime in torch.unique(regime_ids, sorted=True)
    ]


def snapshot(model: CausalSSM, probe: Tensor) -> dict[str, Any]:
    """Collect state-independent diagnostics on a probe batch (B,L,D)."""
    model.eval()
    record: dict[str, Any] = {
        "spectral_radius": transition_spectral_radius(model, probe),
        "hidden_max_abs": hidden_state_max_abs(model, probe),
        "coupling_weight_max_row_sum": float(
            model.layers[0]._effective_coupling_weight().detach().abs().sum(dim=-1).max().item()
        ),
        "decay_min": float(
            torch.exp(-torch.exp(model.layers[0].logA.detach())).min().item()
        ),
    }
    record.update(masked_lag_row_sums(model, probe))
    return record


def run_variant_diagnostics(
    config: dict[str, Any],
    variant: dict[str, Any],
    series: Tensor,
    graph: Tensor | None,
    probe_size: int = 32,
) -> dict[str, Any]:
    """Train one variant with the config's optimizer settings and log per-epoch diagnostics."""
    seed = int(config["seed"])
    data_config = _mapping(config["data"], "data")
    training_config = _mapping(config["training"], "training")
    input_length = int(data_config["input_length"])
    horizon = int(data_config["horizon"])
    n_lags = int(data_config.get("n_lags", 1))
    (train_inputs, train_targets), _, _, _ = _split_windows(
        series,
        input_length,
        horizon,
        float(data_config["train_fraction"]),
        float(data_config["validation_fraction"]),
    )
    is_oracle = variant.get("graph_mode") == "oracle"
    model = _build_model(
        variant,
        int(series.shape[1]),
        horizon,
        n_lags,
        graph if is_oracle else None,
        seed,
        float(data_config["sparsity"]) if "sparsity" in data_config else None,
    )
    if not isinstance(model, CausalSSM):
        raise TypeError("stability diagnostics require a CausalSSM model")
    train_graphs = (
        graph[input_length - 1 : input_length - 1 + train_inputs.shape[0]]
        if is_oracle and graph is not None
        else None
    )
    regime_ids = _regime_ids_from_graphs(train_graphs) if train_graphs is not None else None
    if train_graphs is not None:
        _set_oracle_graph(model, train_graphs[0])

    torch.manual_seed(seed)
    parameters = list(model.parameters())
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if str(training_config.get("optimizer", "adam")) == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    grad_clip_norm = (
        float(training_config["grad_clip_norm"]) if "grad_clip_norm" in training_config else None
    )
    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    probe = train_inputs[:probe_size]
    epoch_records: list[dict[str, Any]] = [{"epoch": 0, **snapshot(model, probe)}]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        grad_mean: dict[str, float] = {group: 0.0 for group in PARAMETER_GROUPS}
        grad_max: dict[str, float] = {group: 0.0 for group in PARAMETER_GROUPS}
        for group_indices in _group_batches(
            train_inputs.shape[0], regime_ids, train_inputs.device
        ):
            if train_graphs is not None:
                _set_oracle_graph(model, train_graphs[group_indices[0]])
            for start in range(0, group_indices.shape[0], batch_size):
                indices = group_indices[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(train_inputs[indices]), train_targets[indices])
                loss.backward()
                norms = grad_norms_by_group(model)
                for group, value in norms.items():
                    grad_mean[group] += value
                    grad_max[group] = max(grad_max[group], value)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
                optimizer.step()
                total_loss += float(loss.detach().item()) * indices.shape[0]
                batch_count += 1
        record = {
            "epoch": epoch,
            "train_loss": total_loss / train_inputs.shape[0],
            "grad_norm_mean": {
                group: value / max(batch_count, 1) for group, value in grad_mean.items()
            },
            "grad_norm_max": grad_max,
        }
        record.update(snapshot(model, probe))
        epoch_records.append(record)
        if not math.isfinite(record["train_loss"]):
            break
    return {"variant": str(variant["id"]), "epochs": epoch_records}


def run_diagnostics(
    config_path: str | Path,
    variant_ids: list[str] | None = None,
    probe_size: int = 32,
) -> dict[str, Any]:
    """Run per-epoch stability diagnostics for selected variants of a config."""
    config = _load_config(config_path)
    data_config = _mapping(config["data"], "data")
    series, graph, _ = _load_series(data_config, None)
    reports: dict[str, Any] = {}
    for variant in _model_variants(_mapping(config["model"], "model")):
        variant_id = str(variant["id"])
        if variant_ids is not None and variant_id not in variant_ids:
            continue
        reports[variant_id] = run_variant_diagnostics(
            config, variant, series, graph, probe_size=probe_size
        )
    return {"config": str(config_path), "variants": reports}


def format_report(report: dict[str, Any]) -> str:
    """Render a compact per-epoch table for each variant."""
    lines: list[str] = []
    for variant_id, variant_report in report["variants"].items():
        lines.append(f"== {variant_id} ==")
        lines.append(
            "epoch | loss | rho_max | rho_mean | hidden_max | lag_in_row | lag_fc_row | "
            "||W||_inf | g:lag_in | g:lag_fc | g:coup | g:gate | g:dt | g:in | g:out | g:logA"
        )
        for record in variant_report["epochs"]:
            grads = record.get("grad_norm_max", {})
            lines.append(
                " | ".join(
                    [
                        f"{record['epoch']:>3d}",
                        f"{record.get('train_loss', float('nan')):.4g}",
                        f"{record['spectral_radius']['max']:.4g}",
                        f"{record['spectral_radius']['mean']:.4g}",
                        f"{record['hidden_max_abs']:.4g}",
                        f"{record['lag_input_max_row_sum']:.4g}",
                        f"{record['lag_forecast_max_row_sum']:.4g}",
                        f"{record['coupling_weight_max_row_sum']:.4g}",
                    ]
                    + [
                        f"{grads.get(group, float('nan')):.3g}"
                        for group in (
                            "lag_input_weights",
                            "lag_forecast_weights",
                            "coupling_projection",
                            "gate_projection",
                            "dt_projection",
                            "input_projection",
                            "output_projection",
                            "logA",
                        )
                    ]
                )
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point writing a JSON report and printing a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stability_d20.yaml")
    parser.add_argument("--variants", nargs="*", default=["identity", "oracle"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--probe-size", type=int, default=32)
    arguments = parser.parse_args(argv)
    report = run_diagnostics(arguments.config, arguments.variants, arguments.probe_size)
    print(format_report(report))
    output = (
        Path(arguments.output)
        if arguments.output is not None
        else Path("results") / f"diagnose_{Path(arguments.config).stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
