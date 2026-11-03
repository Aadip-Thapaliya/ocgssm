"""Roadmap S6: ETTh1/ETTh2 real-data evaluation (CPU, standardized, chronological split)."""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.baselines import predict_last_value
from ocg.data import load_csv, make_windows
from ocg.run import _build_model
from ocg.train import train_offline
from scripts import roadmap_final as rf
from scripts.d20_equal_training import _graph_config, _write_json

ROOT = rf.ROOT / "s6"
GATES = ROOT / "gates.yaml"
COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
TRAIN_END, VAL_END, TEST_END = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
L, K, BLOCK = 96, 2, 96
SEEDS = [1009, 1103, 1201]


def ensure_csv(name: str) -> Path:
    path = Path("data") / f"{name}.csv"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path)
    return path


def ett_data(name: str, horizon: int, train_stride: int = 4) -> dict[str, Any]:
    """Chronological 12/4/4-month split; standardize with train-row statistics only."""
    series = load_csv(ensure_csv(name), COLUMNS)[:TEST_END].float()
    if series.shape != (TEST_END, 7) or not torch.isfinite(series).all():
        raise RuntimeError(f"unexpected {name} shape {tuple(series.shape)}")
    mean = series[:TRAIN_END].mean(dim=0)
    std = series[:TRAIN_END].std(dim=0)
    z = (series - mean) / std
    out: dict[str, Any] = {"mean": mean.tolist(), "std": std.tolist(), "variables": 7}
    for split, (start, stop) in {"train": (0, TRAIN_END), "val": (TRAIN_END - L, VAL_END),
                                 "test": (VAL_END - L, TEST_END)}.items():
        x, y = make_windows(z[start:stop], L, horizon)
        if split == "train":
            x, y = x[::train_stride], y[::train_stride]
        out[split] = (x, y)
        out[f"{split}_target_range"] = [start + L, stop - 1]
    return out


def build(spec: dict[str, Any], horizon: int, seed: int) -> torch.nn.Module:
    kind = spec.get("model", "ocg")
    if kind == "var_direct":
        config: dict[str, Any] = {"name": "var_direct", "input_length": L}
    elif kind == "dlinear":
        config = {"name": "dlinear", "input_length": L, "moving_average": 25}
    else:
        graph = _graph_config(0.01)
        graph["sparsity_weight"] = float(spec.get("sparsity_weight", graph["sparsity_weight"]))
        config = {"name": "ocg", "input_length": L, "state_dim": 8, "n_layers": 1,
                  "embedding_dim": 4, "graph": graph, "graph_mode": spec["graph_mode"],
                  "direct_head_mode": spec.get("direct_head_mode", "iterated")}
    torch.manual_seed(seed)
    return _build_model(config, n_variables=7, horizon=horizon, n_lags=K, true_graph=None,
                        seed=seed, random_sparsity=float(spec.get("random_density", 0.1)))


def _predict(model: torch.nn.Module, x: torch.Tensor, batch: int = 512) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def train_eval(path: Path, spec: dict[str, Any], data: dict[str, Any], horizon: int, seed: int,
               training: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    model = build(spec, horizon, seed)
    tx, ty = data["train"]
    vx, vy = data["val"]
    started = time.perf_counter()
    result = train_offline(model, tx, ty, epochs=int(training["epoch_cap"]),
                           batch_size=int(training["batch_size"]),
                           learning_rate=float(training["learning_rate"]), seed=seed,
                           validation_inputs=vx, validation_targets=vy,
                           restore_best_validation=True,
                           early_stopping_patience=int(training["patience"]))
    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {"spec": spec, "seed": seed, "horizon": horizon,
                              "elapsed_seconds": elapsed,
                              "epochs_executed": result["epochs_executed"],
                              "reached_epoch_cap": result["reached_epoch_cap"],
                              "best_validation_epoch": result["best_validation_epoch"],
                              "best_validation_mse": result["best_validation_mse"],
                              "validation_mse_curve": result["validation_mse"],
                              "peak_rss_mb": rf.peak_rss_mb()}
    if all(np.isfinite(result["validation_mse"])) and data.get("_test_authorized"):
        ex, ey = data["test"]
        pred = _predict(model, ex)
        record["finite_test"] = bool(torch.isfinite(pred).all())
        diff = pred - ey
        record["window_mse"] = diff.square().mean(dim=(1, 2)).tolist()
        record["window_mae"] = diff.abs().mean(dim=(1, 2)).tolist()
        record["mse_by_horizon_step"] = diff.square().mean(dim=(0, 2)).tolist()
        sample = ex[:100]
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(len(sample)):
                model(sample[i:i + 1])
        record["seconds_per_window"] = (time.perf_counter() - t0) / len(sample)
    learner = getattr(model, "graph_learner", None)
    if learner is not None:
        record["adjacency"] = learner.adjacency().detach().tolist()
    _write_json(path, record)
    print(json.dumps({"run": str(path), "s": round(elapsed, 1), "ep": result["epochs_executed"],
                      "best": result["best_validation_epoch"],
                      "val": round(result["best_validation_mse"], 4)}), flush=True)
    return record


def persistence_val_mse(data: dict[str, Any], horizon: int) -> float:
    vx, vy = data["val"]
    return float((predict_last_value(vx, horizon) - vy).square().mean())


def diverged(record: dict[str, Any], persistence: float) -> bool:
    curve = record["validation_mse_curve"]
    return (not all(np.isfinite(curve))) or record["best_validation_mse"] > persistence


def run_job(dataset: str, horizon: int, seeds: list[int], gates: dict[str, Any],
            force_lag: bool = False) -> None:
    data = ett_data(dataset, horizon, int(gates["training"]["train_stride"]))
    data["_test_authorized"] = True
    training = gates["training"]
    base = ROOT / dataset / f"H{horizon}"
    persistence = persistence_val_mse(data, horizon)
    for seed in seeds:
        d = base / f"seed{seed}"
        learned = train_eval(d / "learned.json", {"graph_mode": "learned"}, data, horizon, seed,
                             training)
        density = float(np.mean(learned["adjacency"]))
        variants = [("identity", {"graph_mode": "identity"}), ("dense", {"graph_mode": "dense"}),
                    ("random", {"graph_mode": "random", "random_density": density}),
                    ("var_direct", {"model": "var_direct"}), ("dlinear", {"model": "dlinear"})]
        if force_lag or diverged(learned, persistence):
            variants.append(("learned_lag", {"graph_mode": "learned", "direct_head_mode": "lag"}))
        for name, spec in variants:
            train_eval(d / f"{name}.json", spec, data, horizon, seed, training)
    _write_json(base / "persistence.json", {"validation_mse": persistence})


def _load(base: Path, name: str) -> list[dict[str, Any]]:
    out = []
    for seed in SEEDS:
        p = base / f"seed{seed}" / f"{name}.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def summarize(datasets: list[str], horizons: list[int], commit: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary: dict[str, Any] = {"gates_commit": commit, "results": {}}
    lines = ["# S6 real data (ETTh1/ETTh2)", "", f"Gates commit `{commit}`. Standardized units "
             "(train-split statistics). Ratios are equal-seed mean MSE ratios with paired circular "
             f"block bootstrap CIs (block {BLOCK} windows, {rf.RESAMPLES} resamples).", ""]
    for ds in datasets:
        for h in horizons:
            base = ROOT / ds / f"H{h}"
            if not base.exists():
                continue
            names = ["identity", "dense", "random", "learned", "var_direct", "dlinear", "learned_lag"]
            recs = {n: _load(base, n) for n in names}
            done = {n: len(r) for n, r in recs.items() if r}
            recs = {n: r for n, r in recs.items() if len(r) == len(SEEDS)}
            if "learned" not in recs:
                summary["results"][f"{ds}_H{h}"] = {"status": "INCOMPLETE", "seeds_done": done}
                lines += [f"## {ds}_H{h}: INCOMPLETE (stage budget exhausted; seeds done per "
                          f"variant {done}); not evaluated", ""]
                continue
            persistence = json.loads((base / "persistence.json").read_text())["validation_mse"]
            head = "learned"
            divergent = any(diverged(r, persistence) for r in recs["learned"])
            if divergent and "learned_lag" in recs:
                lv = np.mean([r["best_validation_mse"] for r in recs["learned_lag"]])
                iv = np.mean([r["best_validation_mse"] for r in recs["learned"]])
                head = "learned_lag" if lv < iv else "learned"
            key = f"{ds}_H{h}"
            res: dict[str, Any] = {"persistence_val_mse": persistence, "divergent_iterated": divergent,
                                   "final_learned": head, "variants": {}, "ratios": {}}
            for n, rs in recs.items():
                res["variants"][n] = {
                    "test_mse_mean": float(np.mean([np.mean(r["window_mse"]) for r in rs])),
                    "test_mae_mean": float(np.mean([np.mean(r["window_mae"]) for r in rs])),
                    "test_mse_per_seed": [float(np.mean(r["window_mse"])) for r in rs],
                    "median_window_mse": float(np.median(np.concatenate([r["window_mse"] for r in rs]))),
                    "best_val_mse": [r["best_validation_mse"] for r in rs],
                    "best_epochs": [r["best_validation_epoch"] for r in rs],
                    "cap_hits": sum(bool(r["reached_epoch_cap"]) for r in rs),
                    "train_seconds": sum(r["elapsed_seconds"] for r in rs),
                    "seconds_per_window": float(np.mean([r["seconds_per_window"] for r in rs]))}
            for i, (num, den) in enumerate([(head, "dense"), (head, "identity"), (head, "random"),
                                            (head, "var_direct"), (head, "dlinear"),
                                            ("dense", "identity"), ("var_direct", "identity"),
                                            ("dlinear", "identity")]):
                for metric in ("window_mse", "window_mae"):
                    res["ratios"][f"{num}/{den} {metric[7:]}"] = rf.paired_ratio(
                        [np.asarray(r[metric]) for r in recs[num]],
                        [np.asarray(r[metric]) for r in recs[den]], 930000 + 10 * i, block=BLOCK)
            ld = res["ratios"][f"{head}/dense mse"]
            res["target_learned_le_dense"] = ld["ratio"] <= 1.0
            res["benefit_vs_dense_ci"] = ld["ci95"][1] < 1.0
            adj = np.asarray([r["adjacency"] for r in recs[head]])  # (S,K,D,D) [target, source]
            res["graph"] = _graph_report(adj)
            summary["results"][key] = res
            _heatmap(plt, adj, base / "learned_graph.png", key)
            lines += [f"## {key} (final learned head: {head}; iterated divergent: {divergent}; "
                      f"persistence val MSE {persistence:.4f})", "",
                      "| Variant | test MSE | test MAE | per-seed MSE | median window MSE | best val | best epochs | cap hits | train s | ms/window |",
                      "|---|---:|---:|---|---:|---|---|---:|---:|---:|"]
            for n, v in res["variants"].items():
                lines.append(f"| {n} | {v['test_mse_mean']:.4f} | {v['test_mae_mean']:.4f} | " +
                             ", ".join(f"{x:.4f}" for x in v["test_mse_per_seed"]) +
                             f" | {v['median_window_mse']:.4f} | " +
                             ", ".join(f"{x:.4f}" for x in v["best_val_mse"]) +
                             f" | {v['best_epochs']} | {v['cap_hits']} | {v['train_seconds']:.0f} | "
                             f"{1000 * v['seconds_per_window']:.2f} |")
            lines += ["", "| Ratio | value | 95% CI | per seed |", "|---|---:|---|---|"]
            for k2, r in res["ratios"].items():
                lines.append(f"| {k2} | {r['ratio']:.4f} | [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | " +
                             ", ".join(f"{x:.4f}" for x in r["per_seed"]) + " |")
            g = res["graph"]
            lines += ["", f"Target learned <= dense (point): **{'met' if res['target_learned_le_dense'] else 'NOT met'}**; "
                      f"CI-supported benefit: **{'yes' if res['benefit_vs_dense_ci'] else 'no'}**.",
                      f"Learned graph: mean density {g['mean_density']:.3f}; top-10 edge Jaccard across seeds "
                      f"{g['top10_pairwise_jaccard']}; mean across-seed std of edge probabilities "
                      f"{g['mean_edge_std']:.3f}. Top edges (lag, source->target, mean prob): " +
                      "; ".join(f"L{e['lag']} {e['source']}->{e['target']} {e['mean']:.2f}"
                                for e in g["top_edges"]), "",
                      f"![{key} learned graph]({ds}/H{h}/learned_graph.png)", ""]
    _write_json(ROOT / "summary.json", summary)
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _graph_report(adj: np.ndarray) -> dict[str, Any]:
    mean = adj.mean(axis=0)
    flat = [(float(mean[k, i, j]), k, i, j) for k in range(adj.shape[1])
            for i in range(7) for j in range(7) if i != j]
    flat.sort(reverse=True)
    tops = []
    for s in range(adj.shape[0]):
        e = sorted(((adj[s, k, i, j], k, i, j) for k in range(adj.shape[1])
                    for i in range(7) for j in range(7) if i != j), reverse=True)[:10]
        tops.append({(k, i, j) for _, k, i, j in e})
    jac = [round(len(a & b) / len(a | b), 3) for x, a in enumerate(tops) for b in tops[x + 1:]]
    return {"mean_density": float(mean.mean()), "mean_edge_std": float(adj.std(axis=0).mean()),
            "top10_pairwise_jaccard": jac,
            "top_edges": [{"lag": k + 1, "source": COLUMNS[j], "target": COLUMNS[i], "mean": p,
                           "std": float(adj[:, k, i, j].std())} for p, k, i, j in flat[:10]]}


def _heatmap(plt: Any, adj: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, adj.shape[1], figsize=(5 * adj.shape[1], 4.3))
    for k, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(adj[:, k].mean(axis=0), vmin=0, vmax=1, cmap="viridis")
        ax.set(xticks=range(7), yticks=range(7), xlabel="source", ylabel="target",
               title=f"{title} lag {k + 1} (mean of {adj.shape[0]} seeds)")
        ax.set_xticklabels(COLUMNS, rotation=45, fontsize=7)
        ax.set_yticklabels(COLUMNS, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force-lag", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    commit = rf.committed_hash(GATES)
    if args.summarize:
        summarize(gates["datasets"], gates["horizons"], commit)
        return
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(gates["seeds"])
    run_job(args.dataset, args.horizon, seeds, gates, args.force_lag)


if __name__ == "__main__":
    sys.exit(main())
