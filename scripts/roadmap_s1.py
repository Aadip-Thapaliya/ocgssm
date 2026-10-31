"""Roadmap S1: oracle direct-head masking at H2 versus DirectVAR (development only)."""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.data import generate_svar, make_windows
from ocg.run import _build_model
from ocg.train import _regime_ids_from_graphs, train_offline
from scripts.d20_equal_training import _graph_config, _predict_errors, _write_json

ROOT = Path("results/roadmap/s1")
PROTOCOL = ROOT / "protocol.yaml"
EXCLUDED_PAIRS = {(11, 101), (17, 202), (23, 303), (31, 301), (37, 401), (43, 503), (47, 607)}
HELD_OUT_PAIRS = {(53, 709), (59, 811), (61, 913)}


def _protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def _check_frozen() -> str:
    """Require the protocol to be committed and unchanged; return commit hash."""
    diff = subprocess.run(
        ["git", "status", "--porcelain", "--", str(PROTOCOL)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if diff:
        raise RuntimeError("S1 protocol must be committed before running")
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(PROTOCOL)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _seed_data(data_seed: int, data_config: dict[str, Any]) -> dict[str, Any]:
    data = generate_svar(
        n_steps=int(data_config["steps"]),
        n_variables=int(data_config["variables"]),
        n_lags=int(data_config["lags"]),
        switch_at=int(data_config["switch_at"]),
        blend_steps=int(data_config["blend_steps"]),
        sparsity=float(data_config["sparsity"]),
        noise_scale=float(data_config["noise_scale"]),
        seed=data_seed,
        cross_variable_scale=float(data_config["cross_variable_scale"]),
    )
    series, graph = data["series"], data["graph"]
    if int(data_config["switch_at"]) <= 3199:
        raise RuntimeError("S1 assumes one regime across train and validation")
    del data
    out: dict[str, Any] = {"graph0": graph[0].clone()}
    for name, (start, stop) in {
        "train": (0, 2400), "tune": (2400, 2666), "gate": (2933, 3200),
    }.items():
        x, y = make_windows(series[start:stop], 24, 6)
        graphs = graph[start + 23 : start + 23 + len(x)]
        out[name] = (x, y, graphs)
    if not torch.equal(out["train"][2][0], out["gate"][2][-1]):
        raise RuntimeError("graph changed inside development span")
    return out


def _build(variant: str, spec: dict[str, Any], model_seed: int, graph0: torch.Tensor):
    if variant == "var_direct":
        config: dict[str, Any] = {"id": variant, "name": "var_direct", "input_length": 24}
    else:
        config = {
            "id": variant, "name": "ocg", "input_length": 24, "state_dim": 8,
            "n_layers": 1, "embedding_dim": 4, "graph_mode": spec.get("graph_mode", "oracle"),
            "graph": _graph_config(0.01),
            "direct_head_mode": spec["direct_head_mode"],
            "zero_init_ssm_head": bool(spec.get("zero_init_ssm_head", False)),
        }
    torch.manual_seed(model_seed)
    return _build_model(
        config, n_variables=20, horizon=6, n_lags=2,
        true_graph=graph0.unsqueeze(0) if config.get("graph_mode", "oracle") == "oracle" else None,
        seed=model_seed, random_sparsity=0.1,
    )


def _run_one(stage: str, variant: str, spec: dict[str, Any], pair: tuple[int, int],
             data: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / stage / "runs" / f"data{pair[0]}_model{pair[1]}" / f"{variant}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    model = _build(variant, spec, pair[1], data["graph0"])
    train_x, train_y, train_g = data["train"]
    tune_x, tune_y, tune_g = data["tune"]
    gate_x, gate_y, _ = data["gate"]
    oracle = variant != "var_direct" and spec.get("graph_mode", "oracle") == "oracle"
    started = time.perf_counter()
    result = train_offline(
        model, train_x, train_y,
        epochs=int(training["epoch_cap"]), batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]), seed=pair[1],
        validation_inputs=tune_x, validation_targets=tune_y,
        oracle_graphs=train_g if oracle else None,
        regime_ids=_regime_ids_from_graphs(train_g) if oracle else None,
        validation_oracle_graphs=tune_g if oracle else None,
        validation_regime_ids=_regime_ids_from_graphs(tune_g) if oracle else None,
        restore_best_validation=True,
        early_stopping_patience=int(training["patience"]),
    )
    elapsed = time.perf_counter() - started
    _, tune_rows = _predict_errors(model, tune_x, tune_y)
    _, gate_rows = _predict_errors(model, gate_x, gate_y)
    record = {
        "stage": stage, "variant": variant, "pair": list(pair), "spec": spec,
        "elapsed_seconds": elapsed,
        "epochs_executed": result["epochs_executed"],
        "reached_epoch_cap": result["reached_epoch_cap"],
        "best_validation_epoch": result["best_validation_epoch"],
        "best_validation_mse": result["best_validation_mse"],
        "train_mse_curve": result["train_mse"],
        "validation_mse_curve": result["validation_mse"],
        "tune_mse_by_horizon": [statistics.fmean(r[h] for r in tune_rows) for h in range(6)],
        "gate_errors": gate_rows,
    }
    _write_json(path, record)
    print(json.dumps({k: record[k] for k in ("variant", "pair", "elapsed_seconds",
                      "epochs_executed", "best_validation_epoch")}), flush=True)
    return record


def _paired_ratio_ci(num: list[list[float]], den: list[list[float]], seed: int,
                     block: int, resamples: int) -> list[float]:
    """Equal-seed paired circular moving-block CI for mean(num)/mean(den)."""
    rng = np.random.default_rng(seed)
    num_a = [np.asarray(v) for v in num]
    den_a = [np.asarray(v) for v in den]
    draws = np.empty(resamples)
    for draw in range(resamples):
        ns, ds = [], []
        for a, b in zip(num_a, den_a):
            n = len(a)
            starts = rng.integers(0, n, size=math.ceil(n / block))
            idx = ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]
            ns.append(a[idx].mean())
            ds.append(b[idx].mean())
        draws[draw] = np.mean(ns) / np.mean(ds)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def _summarize(stage: str, records: dict[str, list[dict[str, Any]]],
               protocol: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    stats = protocol["statistics"]
    gate_cfg = protocol["gate"]
    selection = {
        v: statistics.fmean(
            statistics.fmean(r["tune_mse_by_horizon"][h] for r in records[v])
            for h in (0, 1)
        )
        for v in candidates
    }
    selected = min(candidates, key=lambda v: selection[v])
    table: dict[str, Any] = {}
    for v in candidates:
        rows = []
        for h in range(6):
            num = [[w[h] for w in r["gate_errors"]] for r in records[v]]
            den = [[w[h] for w in r["gate_errors"]] for r in records["var_direct"]]
            ratio = statistics.fmean(map(statistics.fmean, num)) / statistics.fmean(
                map(statistics.fmean, den))
            ci = _paired_ratio_ci(num, den, 700000 + h, int(stats["block_length"]),
                                  int(stats["resamples"]))
            per_seed = [statistics.fmean(a) / statistics.fmean(b) for a, b in zip(num, den)]
            rows.append({"horizon": h + 1, "ratio": ratio, "ci95": ci, "per_seed": per_seed})
        table[v] = rows
    limit = float(gate_cfg["max_ratio"])
    gate_rows = [table[selected][h] for h in (0, 1)]
    summary = {
        "stage": stage,
        "selection_by_tune_mean_H1_H2_mse": selection,
        "selected_variant": selected,
        "ratios_vs_var_direct": table,
        "gate": {f"H{r['horizon']}": r["ratio"] <= limit for r in gate_rows},
        "clearly_beats_var_direct": {f"H{r['horizon']}": r["ci95"][1] < 1.0 for r in gate_rows},
        "passed": all(r["ratio"] <= limit for r in gate_rows),
        "training": {
            v: [{k: r[k] for k in ("pair", "elapsed_seconds", "epochs_executed",
                                   "reached_epoch_cap", "best_validation_epoch",
                                   "best_validation_mse")} for r in records[v]]
            for v in records
        },
        "run_seconds": sum(r["elapsed_seconds"] for rs in records.values() for r in rs),
    }
    _write_json(ROOT / stage / "summary.json", summary)
    _write_report(stage, summary, records, candidates)
    return summary


def _write_report(stage: str, summary: dict[str, Any], records: dict[str, list[dict[str, Any]]],
                  candidates: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    horizons = np.arange(1, 7)
    width = 0.8 / len(candidates)
    for i, v in enumerate(candidates):
        rows = summary["ratios_vs_var_direct"][v]
        ratios = np.array([r["ratio"] for r in rows])
        lo = ratios - np.array([r["ci95"][0] for r in rows])
        hi = np.array([r["ci95"][1] for r in rows]) - ratios
        ax1.bar(horizons + (i - (len(candidates) - 1) / 2) * width, ratios, width,
                yerr=[lo, hi], capsize=2, label=v)
    ax1.axhline(1.05, color="red", linestyle="--", linewidth=1, label="gate 1.05")
    ax1.axhline(1.0, color="black", linewidth=0.6)
    ax1.set(xlabel="horizon", ylabel="MSE / DirectVAR (validation 2933-3199)",
            title=f"{stage}: ratio with paired 95% block CI")
    ax1.set_ylim(0.8, max(1.2, ax1.get_ylim()[1]))
    ax1.legend(fontsize=7)
    for v in candidates + ["var_direct"]:
        for j, r in enumerate(records[v]):
            ax2.plot(range(1, len(r["validation_mse_curve"]) + 1), r["validation_mse_curve"],
                     color=f"C{(candidates + ['var_direct']).index(v)}", alpha=0.7,
                     label=v if j == 0 else None)
    ax2.set(xlabel="epoch", ylabel="validation MSE (2400-2665)", title="validation curves",
            yscale="log")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / stage / "s1_ratios_curves.png", dpi=120)
    plt.close(fig)

    lines = [f"# Roadmap {stage} (development only)", "",
             f"Selected on tune 2400-2665 (mean H1/H2 MSE): **{summary['selected_variant']}**. "
             f"Gate (validation 2933-3199): oracle/DirectVAR <= 1.05 at H1 and H2 -> "
             f"**{'PASS' if summary['passed'] else 'FAIL'}**. Clearly beats DirectVAR (CI upper < 1): "
             f"{summary['clearly_beats_var_direct']}. Run time "
             f"{summary['run_seconds'] / 60:.1f} min.", "",
             "| Variant | H1 ratio [95% CI] | H2 ratio [95% CI] | H3 | H6 |", "|---|---|---|---:|---:|"]
    for v in candidates:
        r = summary["ratios_vs_var_direct"][v]
        lines.append(
            f"| {v} | {r[0]['ratio']:.4f} [{r[0]['ci95'][0]:.4f}, {r[0]['ci95'][1]:.4f}] | "
            f"{r[1]['ratio']:.4f} [{r[1]['ci95'][0]:.4f}, {r[1]['ci95'][1]:.4f}] | "
            f"{r[2]['ratio']:.4f} | {r[5]['ratio']:.4f} |")
    lines += ["", "| Variant | Pair | Epochs | Best epoch | Cap hit | Seconds |", "|---|---|---:|---:|---|---:|"]
    for v, rows in summary["training"].items():
        for t in rows:
            lines.append(f"| {v} | {t['pair'][0]}/{t['pair'][1]} | {t['epochs_executed']} | "
                         f"{t['best_validation_epoch']} | {t['reached_epoch_cap']} | "
                         f"{t['elapsed_seconds']:.0f} |")
    (ROOT / stage / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(stage: str) -> dict[str, Any]:
    protocol = _protocol()
    commit = _check_frozen()
    torch.set_num_threads(1)
    cfg = protocol["stages"][stage]
    pairs = [tuple(p) for p in protocol["seeds"]["development_pairs"]]
    if set(pairs) & (EXCLUDED_PAIRS | HELD_OUT_PAIRS):
        raise RuntimeError("development pairs overlap seen or held-out pairs")
    training = dict(protocol["training"], **cfg.get("training_overrides", {}))
    variants: dict[str, dict[str, Any]] = dict(cfg["variants"])
    variants["var_direct"] = {}
    budget = float(protocol["budget"]["max_run_seconds"])
    records: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    started = time.perf_counter()
    for index, pair in enumerate(pairs):
        data = _seed_data(pair[0], protocol["data"])
        for variant, spec in variants.items():
            records[variant].append(_run_one(stage, variant, spec, pair, data, training))
        spent = sum(r["elapsed_seconds"] for rs in records.values() for r in rs)
        projected = spent / (index + 1) * len(pairs)
        print(json.dumps({"after_pair": list(pair), "spent_s": spent,
                          "projected_s": projected}), flush=True)
        if projected > budget:
            raise SystemExit(f"projected run time {projected:.0f}s exceeds budget {budget:.0f}s")
    summary = _summarize(stage, records, protocol, list(cfg["variants"]))
    summary["protocol_commit"] = commit
    summary["wall_seconds_this_invocation"] = time.perf_counter() - started
    _write_json(ROOT / stage / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="s1")
    args = parser.parse_args()
    summary = run_stage(args.stage)
    print(json.dumps({k: summary[k] for k in ("selected_variant", "gate", "passed",
                                               "run_seconds")}))


if __name__ == "__main__":
    sys.exit(main())
