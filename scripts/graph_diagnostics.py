"""Reproducible stationary graph-recovery diagnostics for learned and control graphs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, roc_auc_score

from ocg.data import generate_svar
from ocg.graph import GraphLearner


THRESHOLDS = [round(value / 100, 2) for value in range(10, 96, 5)]
SEED_PAIRS = [(11, 101), (17, 202), (23, 303)]


def _graph_metrics(
    scores: torch.Tensor,
    truth: torch.Tensor,
    threshold: float,
) -> dict[str, float | int | None]:
    """Score (K,D,D) edge values against known validation/test graph truth."""
    truth_edges = truth.detach().cpu().reshape(-1).numpy() >= 0.5
    edge_scores = scores.detach().cpu().reshape(-1).numpy()
    predicted = edge_scores >= threshold
    true_positive = int((predicted & truth_edges).sum())
    false_positive = int((predicted & ~truth_edges).sum())
    false_negative = int((~predicted & truth_edges).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    has_both_classes = truth_edges.any() and (~truth_edges).any()
    return {
        "threshold": threshold,
        "shd": false_positive + false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": (
            float(roc_auc_score(truth_edges, edge_scores))
            if has_both_classes
            else None
        ),
        "precision_recall_auc": (
            float(average_precision_score(truth_edges, edge_scores))
            if truth_edges.any()
            else None
        ),
    }


def _choose_threshold(scores: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    """Select the edge threshold by validation F1 only."""
    candidates = [
        _graph_metrics(scores, truth, threshold) for threshold in THRESHOLDS
    ]
    return max(candidates, key=lambda item: (item["f1"], item["threshold"]))


def _edge_distribution(values: torch.Tensor) -> dict[str, float]:
    """Summarize probabilities without using the test graph."""
    flattened = values.detach().cpu().flatten()
    quantiles = torch.quantile(flattened, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
    return {
        "minimum": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "fraction_at_least_half": float((flattened >= 0.5).float().mean()),
    }


def _design(series: torch.Tensor, n_lags: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a lag-major linear design from one time segment."""
    features = torch.stack(
        [
            torch.cat(
                [series[index - lag - 1] for lag in range(n_lags)], dim=0
            )
            for index in range(n_lags, len(series))
        ]
    )
    return features, series[n_lags:]


def _fit_sparse_var(
    train: torch.Tensor,
    validation: torch.Tensor,
    n_lags: int,
    truth: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit target-wise Lasso VAR on train and tune sparsity/threshold on validation."""
    train_features, train_targets = _design(train, n_lags)
    validation_features, _ = _design(validation, n_lags)
    _, validation_targets = _design(validation, n_lags)
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025]
    best: tuple[tuple[float, float, float], torch.Tensor, dict[str, Any]] | None = None
    for alpha in alphas:
        coefficients = torch.zeros(
            truth.shape, dtype=torch.float32
        )
        for target_index in range(truth.shape[-1]):
            estimator = Lasso(alpha=alpha, max_iter=5000)
            estimator.fit(
                train_features.numpy(),
                train_targets[:, target_index].numpy(),
            )
            coefficients[:, target_index] = torch.from_numpy(
                estimator.coef_.reshape(n_lags, truth.shape[-1])
            )
        scores = coefficients.abs()
        threshold_result = _choose_threshold(scores, truth)
        key = (
            threshold_result["f1"],
            threshold_result["recall"],
            -alpha,
        )
        if best is None or key > best[0]:
            best = (
                key,
                scores,
                {
                    "alpha": alpha,
                    "threshold": threshold_result["threshold"],
                    "validation_f1": threshold_result["f1"],
                },
            )
    if best is None:
        raise RuntimeError("sparse VAR validation did not produce a candidate")
    del validation_features, validation_targets
    return best[1], best[2]


def _source_hashes() -> dict[str, str]:
    """Hash implementation and protocol sources for local reproducibility."""
    paths = [
        Path("src/ocg/data.py"),
        Path("src/ocg/graph.py"),
        Path("src/ocg/model.py"),
        Path("src/ocg/train.py"),
        Path("src/ocg/run.py"),
        Path("scripts/graph_diagnostics.py"),
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _run_pair(
    n_variables: int,
    data_seed: int,
    model_seed: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train and evaluate graph estimators for one stationary synthetic seed pair."""
    n_lags = 2
    n_steps = 1200 if n_variables == 5 else 1600
    train_end = int(n_steps * 0.6)
    validation_end = int(n_steps * 0.8)
    input_length = 24
    data = generate_svar(
        n_steps=n_steps,
        n_variables=n_variables,
        n_lags=n_lags,
        switch_at=n_steps,
        sparsity=0.2 if n_variables == 5 else 0.1,
        seed=data_seed,
    )
    series = data["series"]
    truth = data["graph"][0]
    train_inputs = torch.stack(
        [series[index - input_length : index] for index in range(input_length, train_end)]
    )
    validation_series = series[train_end:validation_end]
    test_series = series[validation_end:]
    graph = GraphLearner(
        n_variables=n_variables,
        n_lags=n_lags,
        learning_rate=0.01,
        sparsity_weight=0.001,
        ema_decay=0.95,
    )
    torch.manual_seed(model_seed)
    torch.nn.init.normal_(graph.weights, mean=0.0, std=0.2)
    initial_parameters = torch.cat(
        [graph.logits.detach().flatten(), graph.weights.detach().flatten()]
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    batch_size = 32
    generator = torch.Generator().manual_seed(model_seed)
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        graph.load_state_dict(checkpoint["graph_state"])
        graph._optimizer.load_state_dict(checkpoint["optimizer_state"])
        generator.set_state(checkpoint["shuffle_generator_state"])
        start_epoch = int(checkpoint["next_epoch"])
        losses = checkpoint["losses"]
        gradient_norms = checkpoint["gradient_norms"]
        update_norms = checkpoint["update_norms"]
    for epoch in range(start_epoch, 8):
        order = torch.randperm(len(train_inputs), generator=generator)
        epoch_losses: list[float] = []
        for batch_indices in order.split(batch_size):
            batch = train_inputs[batch_indices]
            graph._optimizer.zero_grad()
            loss = graph.structural_loss(batch)
            loss.backward()
            gradient_norms.append(
                float(
                    torch.sqrt(
                        sum(
                            parameter.grad.square().sum()
                            for parameter in graph.parameters()
                            if parameter.grad is not None
                        )
                    )
                )
            )
            before = torch.cat(
                [graph.logits.detach().flatten(), graph.weights.detach().flatten()]
            )
            graph._optimizer.step()
            graph._update_ema()
            after = torch.cat(
                [graph.logits.detach().flatten(), graph.weights.detach().flatten()]
            )
            update_norms.append(float(torch.linalg.vector_norm(after - before)))
            epoch_losses.append(float(loss.detach()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graph_state": graph.state_dict(),
                "optimizer_state": graph._optimizer.state_dict(),
                "shuffle_generator_state": generator.get_state(),
                "next_epoch": epoch + 1,
                "losses": losses,
                "gradient_norms": gradient_norms,
                "update_norms": update_norms,
            },
            checkpoint_path,
        )

    validation_graph = data["graph"][train_end + n_lags : validation_end].mean(dim=0)
    test_graph = data["graph"][validation_end + n_lags :].mean(dim=0)
    current = graph.adjacency(use_ema=False).detach()
    ema = graph.adjacency(use_ema=True).detach()
    validation_selection = _choose_threshold(ema, validation_graph)
    learned_metrics = _graph_metrics(
        ema, test_graph, validation_selection["threshold"]
    )
    generator = torch.Generator().manual_seed(model_seed + 1)
    random_graph = torch.zeros_like(truth)
    edges_per_lag = round((0.2 if n_variables == 5 else 0.1) * n_variables**2)
    for lag in range(n_lags):
        chosen = torch.randperm(n_variables**2, generator=generator)[:edges_per_lag]
        random_graph[lag].view(-1)[chosen] = 1.0

    sparse_var_scores, sparse_var_selection = _fit_sparse_var(
        series[:train_end], validation_series, n_lags, truth
    )
    sparse_var_metrics = _graph_metrics(
        sparse_var_scores,
        test_graph,
        sparse_var_selection["threshold"],
    )
    final_parameters = torch.cat(
        [graph.logits.detach().flatten(), graph.weights.detach().flatten()]
    )
    return {
        "dimensions": {"variables": n_variables, "lags": n_lags},
        "seed_pair": {"data": data_seed, "model": model_seed},
        "samples": {
            "total": n_steps,
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
        },
        "source_sha256": _source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "git_revision": None,
            "revision_note": "Folder-backed local copy; no Git metadata is present.",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "completed_epochs": 8,
            "resume_state_includes": [
                "model state",
                "Adam optimizer state",
                "epoch and shuffled-batch generator state",
                "loss, gradient, and parameter-update histories",
            ],
        },
        "protocol": {
            "epochs": 8,
            "batch_size": batch_size,
            "thresholds": THRESHOLDS,
            "threshold_selection": "validation F1, ties prefer higher threshold",
            "structural_sparsity_weight": graph.sparsity_weight,
            "graph_lr": graph._base_learning_rate,
            "ema_decay": graph.ema_decay,
            "structural_objective": "masked VAR MSE plus mean edge probability penalty",
        },
        "training": {
            "epoch_structural_loss": losses,
            "mean_gradient_l2": sum(gradient_norms) / len(gradient_norms),
            "mean_update_l2": sum(update_norms) / len(update_norms),
            "total_parameter_update_l2": float(
                torch.linalg.vector_norm(final_parameters - initial_parameters)
            ),
        },
        "edge_distributions": {
            "current_logits_probability": _edge_distribution(current),
            "ema_probability": _edge_distribution(ema),
        },
        "truth_edge_count": int((truth >= 0.5).sum()),
        "validation": {"learned_threshold": validation_selection},
        "test": {
            "learned": learned_metrics,
            "empty_control": _graph_metrics(torch.zeros_like(truth), test_graph, 0.5),
            "dense_control": _graph_metrics(torch.ones_like(truth), test_graph, 0.5),
            "random_control": _graph_metrics(random_graph, test_graph, 0.5),
            "sparse_var_selection": sparse_var_selection,
            "sparse_var": sparse_var_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        choices=("5", "20", "all"),
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/graph_diagnostics.json"),
    )
    arguments = parser.parse_args()
    selected_dimensions = (
        [5, 20]
        if arguments.dimensions == "all"
        else [int(arguments.dimensions)]
    )
    results: list[dict[str, Any]] = (
        json.loads(arguments.output.read_text(encoding="utf-8"))
        if arguments.output.exists()
        else []
    )
    existing = {
        (
            item["dimensions"]["variables"],
            item["seed_pair"]["data"],
        ): index
        for index, item in enumerate(results)
    }
    checkpoint_directory = arguments.output.with_name(
        f"{arguments.output.stem}_checkpoints"
    )
    for n_variables in selected_dimensions:
        for data_seed, model_seed in SEED_PAIRS:
            checkpoint_path = checkpoint_directory / (
                f"d{n_variables}_data{data_seed}_model{model_seed}.pt"
            )
            result_index = existing.get((n_variables, data_seed))
            if result_index is not None and checkpoint_path.exists():
                continue
            result = _run_pair(
                n_variables, data_seed, model_seed, checkpoint_path
            )
            if result_index is None:
                results.append(result)
                existing[(n_variables, data_seed)] = len(results) - 1
            else:
                results[result_index] = result
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "dimensions": result["dimensions"],
                        "seed_pair": result["seed_pair"],
                        "learned": result["test"]["learned"],
                        "sparse_var": result["test"]["sparse_var"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
