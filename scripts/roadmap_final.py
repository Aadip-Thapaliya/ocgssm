"""Final roadmap stages S2, S2d and S4 on the D-variable K=2 SVAR benchmark.

Development runs read only targets with index <= 3199. Held-out (S4) data is
built only after the stage gates file is committed; every loader asserts it.
"""

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from ocg.data import generate_svar, make_windows
from ocg.run import _build_model
from ocg.train import _regime_ids_from_graphs, _set_oracle_graph, train_offline
from scripts.d20_equal_training import _graph_config, _write_json

ROOT = Path("results/roadmap/final")
DEV_PAIRS = [(67, 1009), (71, 1103), (73, 1201)]
BURNED_PAIRS = {(11, 101), (17, 202), (23, 303), (29, 303), (31, 301), (37, 401),
                (43, 503), (47, 607)}
S4_PAIRS = [(53, 709), (59, 811), (61, 913)]
S5_PAIRS = [(79, 1301), (83, 1409), (89, 1511), (97, 1613), (101, 1709)]
HELD_OUT = set(S4_PAIRS) | set(S5_PAIRS)
TRAINING = {"learning_rate": 0.001, "batch_size": 32, "epoch_cap": 40, "patience": 5}
DATA = {"steps": 4000, "lags": 2, "sparsity": 0.1, "noise_scale": 0.1,
        "cross_variable_scale": 1.0, "switch_at": 3600, "blend_steps": 0,
        "input_length": 24, "horizon": 6}
BLOCK, RESAMPLES = 12, 2000
_AUTHORIZED_HELD_OUT: set[tuple[int, int]] = set()


def committed_hash(path: Path) -> str:
    """Return the last commit hash of ``path``; refuse if it has local changes."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(path)],
                           capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)],
                            capture_output=True, text=True, check=True).stdout.strip()
    if dirty or not commit:
        raise RuntimeError(f"{path} must be committed and unchanged before running")
    return commit


def authorize_held_out(gates: Path, pairs: list[tuple[int, int]]) -> str:
    """Unlock held-out data for ``pairs`` only once ``gates`` is committed."""
    commit = committed_hash(gates)
    _AUTHORIZED_HELD_OUT.update(pairs)
    return commit


def synthetic_data(pair: tuple[int, int], variables: int = 20, train_length: int = 2400,
                   split: str = "dev") -> dict[str, Any]:
    """Build train/tune and either the dev gate slice (<=3199) or the test slice."""
    if pair in BURNED_PAIRS:
        raise RuntimeError(f"burned pair {pair} may not be used")
    if split == "dev" and pair not in DEV_PAIRS:
        raise RuntimeError(f"{pair} is not a development pair")
    if split == "test" and pair not in _AUTHORIZED_HELD_OUT:
        raise RuntimeError(f"LEAKAGE GUARD: held-out pair {pair} used before gates commit")
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if not 0 < train_length <= 2400:
        raise ValueError("train_length must be in (0, 2400]")
    data = generate_svar(
        n_steps=DATA["steps"], n_variables=variables, n_lags=DATA["lags"],
        switch_at=DATA["switch_at"], blend_steps=DATA["blend_steps"],
        sparsity=DATA["sparsity"], noise_scale=DATA["noise_scale"], seed=pair[0],
        cross_variable_scale=DATA["cross_variable_scale"],
    )
    series, graph = data["series"], data["graph"]
    length, horizon = DATA["input_length"], DATA["horizon"]
    segments = {"train": (2400 - train_length, 2400), "tune": (2400, 2666)}
    segments["gate" if split == "dev" else "test"] = (2933, 3200) if split == "dev" else (3200, 4000)
    out: dict[str, Any] = {"graph0": graph[0].clone(), "variables": variables,
                           "segments": segments}
    for name, (start, stop) in segments.items():
        x, y = make_windows(series[start:stop], length, horizon)
        last_target = start + length + len(x) - 1 + horizon - 1
        if split == "dev" and last_target > 3199:
            raise RuntimeError("LEAKAGE GUARD: development target beyond 3199")
        out[name] = (x, y, graph[start + length - 1: start + length - 1 + len(x)])
        out[f"{name}_target_start"] = start + length
    if not torch.equal(out["train"][2][0], out["tune"][2][-1]):
        raise RuntimeError("graph changed between train and tune")
    return out


def build(spec: dict[str, Any], variables: int, model_seed: int, graph0: torch.Tensor,
          mask: torch.Tensor | None = None) -> torch.nn.Module:
    kind = spec.get("model", "ocg")
    if kind in {"var_direct", "var_iterated"}:
        config: dict[str, Any] = {"name": kind, "input_length": 24}
    elif kind == "dlinear":
        config = {"name": "dlinear", "input_length": 24,
                  "moving_average": int(spec.get("moving_average", 3))}
    else:
        mode = spec["graph_mode"]
        graph = _graph_config(0.01)
        graph["sparsity_weight"] = float(spec.get("sparsity_weight", graph["sparsity_weight"]))
        config = {"name": "ocg", "input_length": 24, "state_dim": 8, "n_layers": 1,
                  "embedding_dim": 4, "graph": graph,
                  "graph_mode": "oracle" if mode in {"oracle", "fixed_mask"} else mode,
                  "direct_head_mode": spec.get("direct_head_mode", "iterated")}
    fixed = None
    if config.get("graph_mode") == "oracle":
        fixed = (mask if spec.get("graph_mode") == "fixed_mask" else graph0).unsqueeze(0)
    torch.manual_seed(model_seed)
    return _build_model(config, n_variables=variables, horizon=6, n_lags=2, true_graph=fixed,
                        seed=model_seed,
                        random_sparsity=float(spec.get("random_density", 0.1)))


def predict_errors(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                   graphs: torch.Tensor | None = None) -> list[list[float]]:
    """Per-window per-horizon MSE (N,H); oracle graphs are set per constant segment."""
    model.eval()
    with torch.no_grad():
        if graphs is None:
            pred = model(x)
        else:
            ids = _regime_ids_from_graphs(graphs)
            pred = torch.empty_like(y)
            for regime in torch.unique(ids):
                idx = torch.nonzero(ids == regime).flatten()
                _set_oracle_graph(model, graphs[idx[0]])
                pred[idx] = model(x[idx])
    return (pred - y).square().mean(dim=2).tolist()


def peak_rss_mb() -> float:
    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 2**20


def train_and_eval(path: Path, spec: dict[str, Any], pair: tuple[int, int], data: dict[str, Any],
                   eval_slice: str, mask: torch.Tensor | None = None,
                   training: dict[str, Any] | None = None, save_state: bool = False,
                   ) -> dict[str, Any]:
    """Train with the shared policy (tune checkpoint) and score ``eval_slice`` windows."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    training = training or TRAINING
    model = build(spec, data["variables"], pair[1], data["graph0"], mask)
    oracle = spec.get("graph_mode") == "oracle"
    tx, ty, tg = data["train"]
    vx, vy, vg = data["tune"]
    ex, ey, eg = data[eval_slice]
    started = time.perf_counter()
    result = train_offline(
        model, tx, ty, epochs=int(training["epoch_cap"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]), seed=pair[1],
        validation_inputs=vx, validation_targets=vy,
        oracle_graphs=tg if oracle else None,
        regime_ids=_regime_ids_from_graphs(tg) if oracle else None,
        validation_oracle_graphs=vg if oracle else None,
        validation_regime_ids=_regime_ids_from_graphs(vg) if oracle else None,
        restore_best_validation=True, early_stopping_patience=int(training["patience"]),
    )
    elapsed = time.perf_counter() - started
    tune_rows = predict_errors(model, vx, vy, vg if oracle else None)
    rows = predict_errors(model, ex, ey, eg if oracle else None)
    record: dict[str, Any] = {
        "spec": {k: v for k, v in spec.items()}, "pair": list(pair),
        "eval_slice": eval_slice, "eval_target_start": data[f"{eval_slice}_target_start"],
        "elapsed_seconds": elapsed, "epochs_executed": result["epochs_executed"],
        "reached_epoch_cap": result["reached_epoch_cap"],
        "best_validation_epoch": result["best_validation_epoch"],
        "best_validation_mse": result["best_validation_mse"],
        "validation_mse_curve": result["validation_mse"],
        "tune_mse": statistics.fmean(statistics.fmean(r) for r in tune_rows),
        "tune_mse_by_horizon": [statistics.fmean(r[h] for r in tune_rows) for h in range(6)],
        "errors": rows, "peak_rss_mb": peak_rss_mb(),
    }
    learner = getattr(model, "graph_learner", None)
    if learner is not None:
        record["adjacency"] = learner.adjacency().detach().tolist()
    if mask is not None:
        record["mask_density"] = float(mask.mean())
    _write_json(path, record)
    if save_state:
        torch.save(model.state_dict(), path.with_suffix(".pt"))
    print(json.dumps({"run": str(path), "s": round(elapsed, 1),
                      "epochs": result["epochs_executed"],
                      "best": result["best_validation_epoch"],
                      "rss_mb": round(record["peak_rss_mb"])}), flush=True)
    return record


# ---------------------------------------------------------------- statistics

def _block_index(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]


def paired_ratio(num: list[np.ndarray], den: list[np.ndarray], seed: int,
                 block: int = BLOCK, resamples: int = RESAMPLES) -> dict[str, Any]:
    """Equal-seed ratio of mean errors with a paired circular block bootstrap CI."""
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for d in range(resamples):
        ns, ds = [], []
        for a, b in zip(num, den):
            idx = _block_index(rng, len(a), block)
            ns.append(a[idx].mean())
            ds.append(b[idx].mean())
        draws[d] = np.mean(ns) / np.mean(ds)
    ratio = float(np.mean([a.mean() for a in num]) / np.mean([b.mean() for b in den]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"ratio": ratio, "ci95": [float(lo), float(hi)],
            "per_seed": [float(a.mean() / b.mean()) for a, b in zip(num, den)]}


def paired_difference(a: list[np.ndarray], b: list[np.ndarray], seed: int,
                      block: int = BLOCK, resamples: int = RESAMPLES) -> dict[str, Any]:
    """Equal-seed mean(a) - mean(b) with a paired circular block bootstrap CI."""
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for d in range(resamples):
        diffs = []
        for x, y in zip(a, b):
            idx = _block_index(rng, len(x), block)
            diffs.append(x[idx].mean() - y[idx].mean())
        draws[d] = np.mean(diffs)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"difference": float(np.mean([x.mean() - y.mean() for x, y in zip(a, b)])),
            "ci95": [float(lo), float(hi)],
            "per_seed": [float(x.mean() - y.mean()) for x, y in zip(a, b)]}


def horizon_errors(records: list[dict[str, Any]], h: int) -> list[np.ndarray]:
    return [np.asarray([row[h] for row in r["errors"]]) for r in records]


def ratio_table(records: dict[str, list[dict[str, Any]]], pairs: list[tuple[str, str]],
                horizons: int = 6) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for num, den in pairs:
        out[f"{num}/{den}"] = [
            {"horizon": h + 1, **paired_ratio(horizon_errors(records[num], h),
                                              horizon_errors(records[den], h), 700000 + h)}
            for h in range(horizons)
        ]
    return out


def variant_stats(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for v, rs in records.items():
        out[v] = {
            "mean_mse_by_horizon": [float(np.mean([np.mean(e) for e in horizon_errors(rs, h)]))
                                    for h in range(6)],
            "median_window_mse_H1": float(np.median(np.concatenate(horizon_errors(rs, 0)))),
            "per_seed_mse_H1": [float(e.mean()) for e in horizon_errors(rs, 0)],
            "best_epochs": [r["best_validation_epoch"] for r in rs],
            "epochs": [r["epochs_executed"] for r in rs],
            "cap_hits": sum(bool(r["reached_epoch_cap"]) for r in rs),
            "seconds": sum(r["elapsed_seconds"] for r in rs),
            "peak_rss_mb": max(r.get("peak_rss_mb", 0.0) for r in rs),
        }
    return out


def fmt(entry: dict[str, Any]) -> str:
    return f"{entry['ratio']:.4f} [{entry['ci95'][0]:.4f}, {entry['ci95'][1]:.4f}]"


def markdown_ratios(table: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["| Ratio | H1 [95% CI] | H2 [95% CI] | H3 | H4 | H5 | H6 | per-seed H1 |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for name, rows in table.items():
        lines.append(f"| {name} | {fmt(rows[0])} | {fmt(rows[1])} | " + " | ".join(
            (f"{rows[h]['ratio']:.4f}" if h < len(rows) else "-") for h in range(2, 6)) + " | " +
            ", ".join(f"{v:.3f}" for v in rows[0]["per_seed"]) + " |")
    return lines


def markdown_stats(stats: dict[str, Any]) -> list[str]:
    lines = ["| Variant | mean MSE H1 | mean MSE H2 | median window MSE H1 | per-seed MSE H1 | "
             "best epochs | cap hits | seconds |", "|---|---:|---:|---:|---|---|---:|---:|"]
    for v, s in stats.items():
        lines.append(f"| {v} | {s['mean_mse_by_horizon'][0]:.5f} | {s['mean_mse_by_horizon'][1]:.5f} | "
                     f"{s['median_window_mse_H1']:.5f} | "
                     + ", ".join(f"{x:.5f}" for x in s["per_seed_mse_H1"])
                     + f" | {s['best_epochs']} | {s['cap_hits']} | {s['seconds']:.0f} |")
    return lines


# ---------------------------------------------------------------- S2

S2_VARIANTS: dict[str, dict[str, Any]] = {
    "identity": {"graph_mode": "identity"},
    "dense": {"graph_mode": "dense"},
    "random": {"graph_mode": "random"},
    "learned_soft": {"graph_mode": "learned"},
    "oracle": {"graph_mode": "oracle"},
    "var_direct": {"model": "var_direct"},
    "var_iterated": {"model": "var_iterated"},
}
HARD_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _tune_mse(model: torch.nn.Module, data: dict[str, Any]) -> float:
    vx, vy, _ = data["tune"]
    return statistics.fmean(statistics.fmean(r) for r in predict_errors(model, vx, vy))


def select_hard_mask(state_path: Path, data: dict[str, Any], pair: tuple[int, int],
                     spec: dict[str, Any]) -> dict[str, Any]:
    """Pick the threshold minimizing tune MSE of the trained soft model with a hard mask."""
    model = build(spec, data["variables"], pair[1], data["graph0"])
    model.load_state_dict(torch.load(state_path))
    scores = {}
    for threshold in HARD_THRESHOLDS:
        model.graph_mask_threshold = threshold
        scores[threshold] = _tune_mse(model, data)
    best = min(HARD_THRESHOLDS, key=lambda t: (scores[t], -t))
    model.graph_mask_threshold = None
    mask = (model.graph_learner.adjacency().detach() >= best).float()
    return {"threshold": best, "tune_mse_by_threshold": scores, "mask": mask,
            "density": float(mask.mean())}


def run_pair_s2(stage_dir: Path, pair: tuple[int, int], variants: dict[str, dict[str, Any]],
                data: dict[str, Any], eval_slice: str, with_hard: bool = True,
                sparsity_weight: float | None = None) -> dict[str, dict[str, Any]]:
    runs = stage_dir / "runs" / f"data{pair[0]}_model{pair[1]}"
    density = float(data["graph0"].mean())
    out = {}
    for name, spec in variants.items():
        spec = dict(spec)
        if spec.get("graph_mode") == "random":
            spec["random_density"] = density
        if sparsity_weight is not None and spec.get("graph_mode") == "learned":
            spec["sparsity_weight"] = sparsity_weight
        out[name] = train_and_eval(runs / f"{name}.json", spec, pair, data, eval_slice,
                                   save_state=spec.get("graph_mode") == "learned")
    if with_hard:
        soft_spec = dict(variants["learned_soft"])
        if sparsity_weight is not None:
            soft_spec["sparsity_weight"] = sparsity_weight
        selection_path = runs / "learned_hard_selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            mask = torch.tensor(selection["mask"])
        else:
            selection = select_hard_mask(runs / "learned_soft.pt", data, pair, soft_spec)
            mask = selection["mask"]
            _write_json(selection_path, {**selection, "mask": mask.tolist()})
        out["learned_hard"] = train_and_eval(
            runs / "learned_hard.json",
            {"graph_mode": "fixed_mask", "threshold": selection["threshold"]},
            pair, data, eval_slice, mask=mask)
    return out


def s2_gate(table: dict[str, list[dict[str, Any]]], learned: str) -> dict[str, bool]:
    g = {
        "learned/oracle<=1.05 H1": table[f"{learned}/oracle"][0]["ratio"] <= 1.05,
        "learned/oracle<=1.05 H2": table[f"{learned}/oracle"][1]["ratio"] <= 1.05,
        "learned/dense CI<1 H1": table[f"{learned}/dense"][0]["ci95"][1] < 1,
        "learned/dense CI<1 H2": table[f"{learned}/dense"][1]["ci95"][1] < 1,
        "learned/random CI<1 H1": table[f"{learned}/random"][0]["ci95"][1] < 1,
    }
    return g


def run_s2(pairs_subset: list[int] | None, summarize: bool, attempt: str = "s2") -> None:
    gates = ROOT / "s2" / "gates.yaml"
    commit = committed_hash(gates)
    torch.set_num_threads(1)
    cfg = yaml.safe_load(gates.read_text(encoding="utf-8"))
    stage_dir = ROOT / attempt
    sweep = cfg.get("fallback_sparsity_weights") if attempt != "s2" else None
    weights = sweep if sweep else [None]
    indices = pairs_subset if pairs_subset is not None else range(len(DEV_PAIRS))
    if not summarize:
        for i in indices:
            pair = DEV_PAIRS[i]
            data = synthetic_data(pair)
            for w in weights:
                sub = stage_dir if w is None else stage_dir / f"sw{w}"
                run_pair_s2(sub, pair, S2_VARIANTS, data, "gate", sparsity_weight=w)
        return
    summaries = {}
    for w in weights:
        sub = stage_dir if w is None else stage_dir / f"sw{w}"
        records: dict[str, list[dict[str, Any]]] = {}
        selections = []
        for pair in DEV_PAIRS:
            runs = sub / "runs" / f"data{pair[0]}_model{pair[1]}"
            for name in list(S2_VARIANTS) + ["learned_hard"]:
                records.setdefault(name, []).append(
                    json.loads((runs / f"{name}.json").read_text(encoding="utf-8")))
            sel = json.loads((runs / "learned_hard_selection.json").read_text(encoding="utf-8"))
            selections.append({"pair": list(pair), "threshold": sel["threshold"],
                               "density": sel["density"],
                               "tune_mse_by_threshold": sel["tune_mse_by_threshold"]})
        comparisons = [(a, b) for a in ("learned_soft", "learned_hard", "oracle")
                       for b in ("identity", "dense", "random", "oracle", "var_direct",
                                 "var_iterated") if a != b]
        comparisons += [("learned_hard", "learned_soft"), ("dense", "identity"),
                        ("var_iterated", "var_direct")]
        table = ratio_table(records, comparisons)
        hard_beats = all(table["learned_hard/learned_soft"][h]["ci95"][1] < 1 for h in (0, 1))
        final = "learned_soft"  # hard requires an implemented online hysteresis rule
        gate = s2_gate(table, final)
        summaries[str(w)] = {
            "attempt": attempt, "sparsity_weight": w, "final_learned": final,
            "learned_hard_beats_soft_ci": hard_beats, "gate": gate,
            "passed": all(gate.values()), "hard_mask_selection": selections,
            "ratios": table, "stats": variant_stats(records),
            "true_graph_density": [float(r["spec"].get("random_density", float("nan")))
                                   for r in records["random"]],
            "learned_soft_mean_adjacency": [float(np.mean(r["adjacency"]))
                                            for r in records["learned_soft"]],
        }
    summary = {"gates_commit": commit, "attempts": summaries}
    _write_json(stage_dir / "summary.json", summary)
    lines = [f"# {attempt.upper()} (dev seeds, validation gate slice 2933-3199)", "",
             f"Gates commit `{commit}`.", ""]
    for key, s in summaries.items():
        lines += [f"## sparsity weight {s['sparsity_weight'] or 'default 0.001'}: "
                  f"{'PASS' if s['passed'] else 'FAIL'} (final learned = {s['final_learned']})", "",
                  "Gate: " + ", ".join(f"{k}: {'pass' if v else 'FAIL'}"
                                       for k, v in s["gate"].items()), "",
                  f"learned_hard beats soft with CI<1 at H1 and H2: {s['learned_hard_beats_soft_ci']}. "
                  "Hard thresholds/densities: " + "; ".join(
                      f"{x['pair'][0]}/{x['pair'][1]}: t={x['threshold']} d={x['density']:.3f}"
                      for x in s["hard_mask_selection"]), "",
                  *markdown_ratios(s["ratios"]), "", *markdown_stats(s["stats"]), ""]
    (stage_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {"passed": s["passed"], "gate": s["gate"]} for k, s in summaries.items()}))


# ---------------------------------------------------------------- S2d

S2D_VARIANTS = {"identity": {"graph_mode": "identity"}, "dense": {"graph_mode": "dense"},
                "learned": {"graph_mode": "learned"}, "oracle": {"graph_mode": "oracle"},
                "var_direct": {"model": "var_direct"}}


def s2d_configs() -> list[tuple[str, int, int]]:
    return [("D20_T600", 20, 600), ("D20_T1200", 20, 1200), ("D20_T2400", 20, 2400),
            ("D10_T2400", 10, 2400), ("D40_T2400", 40, 2400)]


def run_s2d(config_names: list[str] | None, summarize: bool) -> None:
    gates = ROOT / "s2d" / "gates.yaml"
    commit = committed_hash(gates)
    torch.set_num_threads(1)
    stage_dir = ROOT / "s2d"
    skip_path = stage_dir / "d40_skipped.json"
    if not summarize:
        for name, d, t in s2d_configs():
            if config_names and name not in config_names:
                continue
            for pair in DEV_PAIRS:
                data = synthetic_data(pair, variables=d, train_length=t)
                runs = stage_dir / name / f"data{pair[0]}_model{pair[1]}"
                for v, spec in S2D_VARIANTS.items():
                    if name == "D20_T2400":
                        source = ROOT / "s2" / "runs" / f"data{pair[0]}_model{pair[1]}" / (
                            "learned_soft.json" if v == "learned" else f"{v}.json")
                        if source.exists():
                            continue
                    if d == 40 and skip_path.exists():
                        return
                    fresh = not (runs / f"{v}.json").exists()
                    rec = train_and_eval(runs / f"{v}.json", spec, pair, data, "gate")
                    if d == 40 and fresh and rec["elapsed_seconds"] > 600:
                        _write_json(skip_path, {"reason": "one D40 run took > 10 min",
                                                "run": f"{pair}/{v}",
                                                "seconds": rec["elapsed_seconds"]})
                        return
        return
    results = {}
    for name, d, t in s2d_configs():
        if d == 40 and skip_path.exists():
            results[name] = {"skipped": json.loads(skip_path.read_text(encoding="utf-8"))}
            continue
        records: dict[str, list[dict[str, Any]]] = {}
        for pair in DEV_PAIRS:
            for v in S2D_VARIANTS:
                path = stage_dir / name / f"data{pair[0]}_model{pair[1]}" / f"{v}.json"
                if name == "D20_T2400":
                    path = ROOT / "s2" / "runs" / f"data{pair[0]}_model{pair[1]}" / (
                        "learned_soft.json" if v == "learned" else f"{v}.json")
                records.setdefault(v, []).append(json.loads(path.read_text(encoding="utf-8")))
        table = ratio_table(records, [("learned", "dense"), ("oracle", "dense"),
                                      ("learned", "oracle"), ("learned", "identity"),
                                      ("learned", "var_direct")])
        qualifies = (table["learned/dense"][0]["ci95"][1] < 1
                     and table["oracle/dense"][0]["ci95"][1] < 1)
        results[name] = {"variables": d, "train_length": t, "ratios": table,
                         "stats": variant_stats(records), "qualifies": qualifies}
    candidates = [n for n, r in results.items() if r.get("qualifies") and n != "D20_T2400"]
    chosen = min(candidates, key=lambda n: results[n]["ratios"]["learned/dense"][0]["ratio"],
                 default=None)
    summary = {"gates_commit": commit, "configs": results, "s4_hard": chosen}
    _write_json(stage_dir / "summary.json", summary)
    lines = ["# S2d scarcity sweep (dev seeds, validation gate slice)", "",
             f"Gates commit `{commit}`. S4-hard: **{chosen or 'none qualifies (skipped)'}**.", "",
             "| Config | learned/dense H1 | learned/dense H2 | oracle/dense H1 | oracle/dense H2 | qualifies |",
             "|---|---|---|---|---|---|"]
    for n, r in results.items():
        if "skipped" in r:
            lines.append(f"| {n} | skipped ({r['skipped']['reason']}) | | | | |")
            continue
        t = r["ratios"]
        lines.append(f"| {n} | {fmt(t['learned/dense'][0])} | {fmt(t['learned/dense'][1])} | "
                     f"{fmt(t['oracle/dense'][0])} | {fmt(t['oracle/dense'][1])} | {r['qualifies']} |")
    for n, r in results.items():
        if "skipped" not in r:
            lines += ["", f"## {n}", "", *markdown_ratios(r["ratios"]), "",
                      *markdown_stats(r["stats"])]
    (stage_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_s2d(results, stage_dir / "s2d_scarcity.png")
    print(json.dumps({"s4_hard": chosen}))


def _plot_s2d(results: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, r in results.items() if "skipped" not in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, (0, 1)):
        for k, key in enumerate(("learned/dense", "oracle/dense")):
            vals = [results[n]["ratios"][key][h] for n in names]
            x = np.arange(len(names)) + (k - 0.5) * 0.3
            ax.errorbar(x, [v["ratio"] for v in vals],
                        yerr=[[v["ratio"] - v["ci95"][0] for v in vals],
                              [v["ci95"][1] - v["ratio"] for v in vals]],
                        fmt="o", capsize=3, label=key)
            for xi, v in zip(x, vals):
                ax.scatter([xi] * len(v["per_seed"]), v["per_seed"], s=8, color="gray")
        ax.axhline(1, color="black", lw=0.6)
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_title(f"H{h + 1}: ratio to dense (validation)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- S4

S4_VARIANTS = {"identity": {"graph_mode": "identity"}, "dense": {"graph_mode": "dense"},
               "random": {"graph_mode": "random"}, "learned": {"graph_mode": "learned"},
               "oracle": {"graph_mode": "oracle"}, "var_direct": {"model": "var_direct"},
               "var_iterated": {"model": "var_iterated"},
               "dlinear": {"model": "dlinear", "moving_average": 3}}


def run_s4(pairs_subset: list[int] | None, summarize: bool) -> None:
    gates_path = ROOT / "s4" / "gates.yaml"
    cfg = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    commit = authorize_held_out(gates_path, S4_PAIRS)
    torch.set_num_threads(1)
    configs = [("default", 20, 2400)]
    if cfg.get("s4_hard"):
        configs.append(("hard", int(cfg["s4_hard"]["variables"]),
                        int(cfg["s4_hard"]["train_length"])))
    learned_spec = {"graph_mode": "learned", **cfg.get("learned_spec_overrides", {})}
    variants = dict(S4_VARIANTS, learned=learned_spec)
    hard_variants = {k: variants[k] for k in cfg.get("s4_hard_variants", list(variants))}
    indices = pairs_subset if pairs_subset is not None else range(len(S4_PAIRS))
    if not summarize:
        for name, d, t in configs:
            for i in indices:
                pair = S4_PAIRS[i]
                data = synthetic_data(pair, variables=d, train_length=t, split="test")
                runs = ROOT / "s4" / name / f"data{pair[0]}_model{pair[1]}"
                for v, spec in (variants if name == "default" else hard_variants).items():
                    spec = dict(spec)
                    if spec.get("graph_mode") == "random":
                        spec["random_density"] = float(data["graph0"].mean())
                    train_and_eval(runs / f"{v}.json", spec, pair, data, "test")
        return
    out = {"gates_commit": commit, "configs": {}}
    for name, d, t in configs:
        records: dict[str, list[dict[str, Any]]] = {}
        names = list(variants if name == "default" else hard_variants)
        for pair in S4_PAIRS:
            for v in names:
                path = ROOT / "s4" / name / f"data{pair[0]}_model{pair[1]}" / f"{v}.json"
                records.setdefault(v, []).append(json.loads(path.read_text(encoding="utf-8")))
        comps = [("learned", b) for b in names if b != "learned"]
        comps += [("oracle", "dense"), ("dense", "identity"), ("oracle", "var_direct")]
        comps = [c for c in comps if c[0] in names and c[1] in names]
        table = ratio_table(records, comps)
        pre = {}
        for v, rs in records.items():  # windows whose targets all precede the 3600 switch
            n_pre = 3600 - 6 - 3224 + 1
            pre[v] = [{"errors": r["errors"][:n_pre]} for r in rs]
        pre_table = ratio_table(pre, [c for c in comps if c[0] == "learned"], horizons=2)
        entry: dict[str, Any] = {"ratios": table, "pre_switch_ratios": pre_table,
                                 "stats": variant_stats(records)}
        if name == "default":
            primary = {
                "learned/dense CI<1 H1": table["learned/dense"][0]["ci95"][1] < 1,
                "learned/random CI<1 H1": table["learned/random"][0]["ci95"][1] < 1,
                "learned/identity CI<1 H1": table["learned/identity"][0]["ci95"][1] < 1,
                "learned/identity CI<1 H2": table["learned/identity"][1]["ci95"][1] < 1,
            }
            entry["primary_gate"] = primary
            entry["primary_passed"] = all(primary.values())
            entry["secondary"] = {f"learned/oracle<=1.05 H{h + 1}":
                                  table["learned/oracle"][h]["ratio"] <= 1.05 for h in (0, 1)}
        out["configs"][name] = entry
    _write_json(ROOT / "s4" / "summary.json", out)
    lines = ["# S4 held-out confirmation (test slice 3200-3999, seeds 53/709, 59/811, 61/913)",
             "", f"Gates commit `{commit}`.", ""]
    for name, e in out["configs"].items():
        lines += [f"## {name}", ""]
        if "primary_gate" in e:
            lines += [f"Primary gate: **{'PASS' if e['primary_passed'] else 'FAIL'}** - " +
                      ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in e["primary_gate"].items()),
                      "", "Secondary: " + ", ".join(f"{k}: {v}" for k, v in e["secondary"].items()), ""]
        lines += [*markdown_ratios(e["ratios"]), "", "Pre-switch windows only (targets < 3600):", "",
                  *markdown_ratios(e["pre_switch_ratios"]).__iter__(), "", *markdown_stats(e["stats"]), ""]
    (ROOT / "s4" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({n: e.get("primary_passed") for n, e in out["configs"].items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["s2", "s2_sweep", "s2d", "s4"])
    parser.add_argument("--pairs", default=None, help="comma-separated pair indices")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    pairs = [int(i) for i in args.pairs.split(",")] if args.pairs else None
    if args.stage == "s2":
        run_s2(pairs, args.summarize)
    elif args.stage == "s2_sweep":
        run_s2(pairs, args.summarize, attempt="s2_sweep")
    elif args.stage == "s2d":
        run_s2d(args.configs.split(",") if args.configs else None, args.summarize)
    else:
        run_s4(pairs, args.summarize)


if __name__ == "__main__":
    sys.exit(main())
