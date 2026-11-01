"""Roadmap S5: leak-safe prequential adaptation after an abrupt graph switch (D10, K2)."""

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.data import generate_svar, make_windows
from ocg.eval import graph_metrics, recovery_time
from ocg.train import train_online
from scripts import roadmap_final as rf
from scripts.d20_equal_training import _write_json

ROOT = rf.ROOT / "s5"
GATES = ROOT / "gates.yaml"
SWITCH, STREAM_START, L, H = 2500, 2000, 24, 6
FIRST_POST = SWITCH - STREAM_START - L          # first window whose first target >= switch
DETECT_INDEX = SWITCH - STREAM_START - L + 1    # first window whose inputs include the switch
LAST_PRE = SWITCH - STREAM_START - L - H        # windows [0, LAST_PRE] have all targets < switch
POST_WINDOWS = 200
DATA = {"steps": 3000, "variables": 10, "lags": 2, "sparsity": 0.2, "noise_scale": 0.1,
        "cross_variable_scale": 1.5}
PRETRAIN = {"identity": {"graph_mode": "identity"}, "dense": {"graph_mode": "dense"},
            "random": {"graph_mode": "random"}, "learned": {"graph_mode": "learned"},
            "oracle": {"graph_mode": "oracle"}}
TUNING_GRID = ([{"ema_decay": e, "threshold": t, "boost": 3.0}
                for e in (0.9, 0.95, 0.98) for t in (3.5, 5.0)]
               + [{"ema_decay": 0.95, "threshold": t, "boost": 5.0} for t in (3.5, 5.0)])


def s5_data(pair: tuple[int, int]) -> dict[str, Any]:
    if pair in rf.BURNED_PAIRS:
        raise RuntimeError(f"burned pair {pair} may not be used")
    if pair not in rf.DEV_PAIRS and pair not in rf._AUTHORIZED_HELD_OUT:
        raise RuntimeError(f"LEAKAGE GUARD: held-out pair {pair} used before gates commit")
    data = generate_svar(
        n_steps=DATA["steps"], n_variables=DATA["variables"], n_lags=DATA["lags"],
        switch_at=SWITCH, sparsity=DATA["sparsity"], noise_scale=DATA["noise_scale"],
        seed=pair[0], cross_variable_scale=DATA["cross_variable_scale"])
    series, graph = data["series"], data["graph"]
    out: dict[str, Any] = {"graph0": graph[0].clone(), "graph1": graph[-1].clone(),
                           "variables": DATA["variables"]}
    for name, (start, stop) in {"train": (0, 1600), "tune": (1600, 2000),
                                "stream": (STREAM_START, 3000)}.items():
        x, y = make_windows(series[start:stop], L, H)
        out[name] = (x, y, graph[start + L - 1: start + L - 1 + len(x)])
        out[f"{name}_target_start"] = start + L
    if not torch.equal(out["train"][2][0], out["tune"][2][-1]):
        raise RuntimeError("graph changed before the stream")
    return out


def _model(spec: dict[str, Any], pair: tuple[int, int], data: dict[str, Any], state: Path):
    model = rf.build(spec, data["variables"], pair[1], data["graph0"])
    model.load_state_dict(torch.load(state))
    return model


def _calibrate(model: torch.nn.Module, data: dict[str, Any], config: dict[str, Any]) -> dict:
    """Frozen-threshold protocol: robust stats of validation structural scores, no updates."""
    learner = model.graph_learner
    probe = copy.deepcopy(learner)
    probe.detector.threshold = float("inf")
    scores = [float(probe.online_update(w.unsqueeze(0), update_detector=False))
              for w in data["tune"][0]]
    cal = learner.calibrate_detector(scores)
    max_z = max((s - cal["median"]) / cal["robust_std"] for s in scores)
    threshold = max(float(config["threshold"]), max_z + 0.25)
    learner.detector.threshold = threshold
    return {**cal, "threshold": threshold, "validation_max_zscore": max_z}


def run_online(variant: str, pair: tuple[int, int], data: dict[str, Any], state_dir: Path,
               config: dict[str, Any], path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    base = variant.replace("_online", "").replace("_det_on", "").replace("_det_off", "")
    spec = dict(PRETRAIN[base])
    if base == "random":
        spec["random_density"] = float(data["graph0"].mean())
    model = _model(spec, pair, data, state_dir / f"{base}.pt")
    sx, sy, sg = data["stream"]
    calibration = None
    if base == "learned":
        learner = model.graph_learner
        learner.ema_decay = float(config["ema_decay"])
        learner.boost = float(config["boost"])
        learner.boost_steps = 20
        learner.detector_confirm_steps = 2
        learner.detector_rearm_steps = 10
        if variant.endswith("det_on"):
            calibration = _calibrate(model, data, config)
        else:
            learner.detector.threshold = float("inf")
    started = time.perf_counter()
    torch.manual_seed(pair[1])
    result = train_online(
        model, sx, sy, learning_rate=0.001, seed=pair[1],
        oracle_graphs=sg if base == "oracle" else None,
        timestamps=list(range(STREAM_START + L - 1, STREAM_START + L - 1 + len(sx))),
        update_graph=True, freeze_detector=calibration is not None)
    elapsed = time.perf_counter() - started
    pred = torch.tensor(result["predictions"])
    errors = (pred - sy).square().mean(dim=2).tolist()
    record: dict[str, Any] = {"variant": variant, "pair": list(pair), "config": config,
                              "elapsed_seconds": elapsed, "errors": errors,
                              "calibration": calibration}
    if base == "learned":
        record["detections"] = result["detections"]
        history = torch.tensor(result["graph_history"])
        record["graph_timeline"] = _graph_timeline(history, sg)
    _write_json(path, record)
    print(json.dumps({"run": str(path), "s": round(elapsed, 1)}), flush=True)
    return record


def _graph_timeline(history: torch.Tensor, truth: torch.Tensor) -> dict[str, list]:
    rows = [graph_metrics(history[i], truth[i]) for i in range(len(history))]
    out = {k: [r[k] for r in rows] for k in ("f1", "auroc", "shd")}
    # Reporting-only: F1 of the top-|E| edges (|E| = true edge count), since soft
    # adjacency probabilities can stay below the fixed 0.5 threshold.
    topk = []
    for i in range(len(history)):
        n_true = int(truth[i].sum())
        pred = torch.zeros_like(truth[i]).reshape(-1)
        pred[torch.topk(history[i].reshape(-1), n_true).indices] = 1.0
        tp = float((pred * truth[i].reshape(-1)).sum())
        topk.append(tp / n_true if n_true else 0.0)
    out["f1_topk"] = topk
    return out


def pretrain(pair: tuple[int, int], data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    out = {}
    for name, spec in PRETRAIN.items():
        spec = dict(spec)
        if name == "random":
            spec["random_density"] = float(data["graph0"].mean())
        out[name] = rf.train_and_eval(base_dir / f"{name}.json", spec, pair, data, "stream",
                                      save_state=True)
    return out


def window_mse(record: dict[str, Any]) -> np.ndarray:
    return np.asarray([statistics.fmean(r) for r in record["errors"]])


def post(record: dict[str, Any]) -> np.ndarray:
    return window_mse(record)[FIRST_POST: FIRST_POST + POST_WINDOWS]


def forecast_recovery(errors: np.ndarray) -> int | None:
    """Windows after the switch until a 25-window rolling MSE returns within 10% of pre-switch."""
    reference = errors[: LAST_PRE + 1].mean()
    for i in range(FIRST_POST, len(errors) - 25):
        if errors[i: i + 25].mean() <= 1.1 * reference:
            return i - FIRST_POST
    return None


def detector_summary(record: dict[str, Any]) -> dict[str, Any]:
    det = record["detections"]
    events = [i for i, d in enumerate(det) if d]
    after = [i for i in events if DETECT_INDEX <= i < DETECT_INDEX + POST_WINDOWS]
    return {"false_alarms_pre_switch": sum(i < DETECT_INDEX for i in events),
            "detected_within_200": bool(after),
            "delay_windows": (after[0] - DETECT_INDEX) if after else None,
            "events": events}


def run_pair(pair: tuple[int, int], phase: str, config: dict[str, Any] | None) -> None:
    data = s5_data(pair)
    base = ROOT / phase / f"data{pair[0]}_model{pair[1]}"
    pretrain(pair, data, base / "pretrain")
    if phase == "tuning":
        for k, cfg in enumerate(TUNING_GRID):
            run_online("learned_det_on", pair, data, base / "pretrain", cfg,
                       base / f"config{k}" / "learned_det_on.json")
        return
    for v in ("identity_online", "dense_online", "random_online", "oracle_online",
              "learned_det_off", "learned_det_on"):
        run_online(v, pair, data, base / "pretrain", config, base / f"{v}.json")


def select_config() -> dict[str, Any]:
    scores = []
    for k, cfg in enumerate(TUNING_GRID):
        vals = [post(json.loads((ROOT / "tuning" / f"data{p[0]}_model{p[1]}" / f"config{k}" /
                                 "learned_det_on.json").read_text(encoding="utf-8"))).mean()
                for p in rf.DEV_PAIRS]
        scores.append({"config_index": k, **cfg, "dev_post_switch_mse": float(np.mean(vals))})
    best = min(scores, key=lambda s: s["dev_post_switch_mse"])
    return {"grid": scores, "selected": best}


def summarize(phase: str, pairs: list[tuple[int, int]], commit: str) -> dict[str, Any]:
    names = ["identity_online", "dense_online", "random_online", "oracle_online",
             "learned_frozen", "learned_det_off", "learned_det_on"]
    records: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for p in pairs:
        base = ROOT / phase / f"data{p[0]}_model{p[1]}"
        for n in names:
            path = base / "pretrain" / "learned.json" if n == "learned_frozen" else base / f"{n}.json"
            records[n].append(json.loads(path.read_text(encoding="utf-8")))
    target = "learned_det_on"
    diffs = {f"{target}-{b}": rf.paired_difference([post(r) for r in records[target]],
                                                   [post(r) for r in records[b]], 910000 + i)
             for i, b in enumerate(n for n in names if n != target)}
    gate = {f"{target} < learned_frozen (CI excl 0)": diffs[f"{target}-learned_frozen"]["ci95"][1] < 0,
            f"{target} < dense_online (CI excl 0)": diffs[f"{target}-dense_online"]["ci95"][1] < 0}
    per_variant = {}
    for n, rs in records.items():
        mses = [window_mse(r) for r in rs]
        per_variant[n] = {
            "post_switch_mse_mean": float(np.mean([m[FIRST_POST:FIRST_POST + POST_WINDOWS].mean() for m in mses])),
            "post_switch_mse_per_seed": [float(m[FIRST_POST:FIRST_POST + POST_WINDOWS].mean()) for m in mses],
            "post_switch_window_median": float(np.median(np.concatenate(
                [m[FIRST_POST:FIRST_POST + POST_WINDOWS] for m in mses]))),
            "pre_switch_mse_mean": float(np.mean([m[: LAST_PRE + 1].mean() for m in mses])),
            "forecast_recovery_windows": [forecast_recovery(m) for m in mses],
        }
        if "detections" in rs[0]:
            per_variant[n]["detector"] = [detector_summary(r) for r in rs]
            per_variant[n]["graph"] = []
            for r in rs:
                tl = r["graph_timeline"]
                post_slice = slice(DETECT_INDEX, DETECT_INDEX + POST_WINDOWS)
                per_variant[n]["graph"].append({
                    "pre_switch_f1": float(np.mean(tl["f1"][:DETECT_INDEX])),
                    "post200_f1_mean": float(np.mean(tl["f1"][post_slice])),
                    "final_f1": tl["f1"][-1], "final_auroc": tl["auroc"][-1],
                    "final_shd": tl["shd"][-1],
                    "pre_switch_f1_topk": (float(np.mean(tl["f1_topk"][:DETECT_INDEX]))
                                           if "f1_topk" in tl else None),
                    "post200_f1_topk": (float(np.mean(tl["f1_topk"][post_slice]))
                                        if "f1_topk" in tl else None),
                    "final_f1_topk": tl["f1_topk"][-1] if "f1_topk" in tl else None,
                    "shd_recovery_windows": recovery_time(tl["shd"], DETECT_INDEX)})
        if n == "learned_frozen":
            per_variant[n]["best_epochs"] = [r["best_validation_epoch"] for r in rs]
            per_variant[n]["cap_hits"] = sum(bool(r["reached_epoch_cap"]) for r in rs)
    summary = {"phase": phase, "gates_commit": commit, "pairs": [list(p) for p in pairs],
               "gate": gate, "passed": all(gate.values()), "differences": diffs,
               "variants": per_variant}
    _write_json(ROOT / phase / "summary.json", summary)
    _report(phase, summary, records)
    return summary


def _report(phase: str, s: dict[str, Any], records: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, rs in records.items():
        curve = np.mean([np.convolve(window_mse(r), np.ones(15) / 15, mode="same") for r in rs], axis=0)
        a1.plot(np.arange(len(curve)) - FIRST_POST, curve, label=n)
    a1.axvline(0, color="black", lw=0.8)
    a1.axvspan(0, POST_WINDOWS, color="gray", alpha=0.1)
    a1.set(yscale="log", xlabel="windows since first post-switch target",
           ylabel="prequential MSE (15-window smooth, seed mean)", title=f"S5 {phase}")
    a1.legend(fontsize=7)
    for n in ("learned_det_off", "learned_det_on"):
        key = "f1_topk" if "f1_topk" in records[n][0]["graph_timeline"] else "auroc"
        f1 = np.mean([[np.nan if v is None else v for v in r["graph_timeline"][key]]
                      for r in records[n]], axis=0)
        a2.plot(np.arange(len(f1)) - DETECT_INDEX, f1, label=f"{n} ({key})")
    a2.axvline(0, color="black", lw=0.8)
    a2.set(xlabel="windows since switch enters input", ylabel="graph recovery vs current true graph",
           title="pre-update graph snapshot (top-|E| F1 or AUROC)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / phase / "s5_post_switch.png", dpi=120)
    plt.close(fig)
    v = s["variants"]
    lines = [f"# S5 adaptation ({phase})", "", f"Gates commit `{s['gates_commit']}`. "
             f"Pairs: {s['pairs']}. Gate: **{'PASS' if s['passed'] else 'FAIL'}** - " +
             ", ".join(f"{k}: {'pass' if x else 'FAIL'}" for k, x in s["gate"].items()), "",
             "Primary metric: mean MSE over the first 200 windows whose first target is post-switch.", "",
             "| Variant | post-switch MSE | per seed | median window | pre-switch MSE | forecast recovery (windows) |",
             "|---|---:|---|---:|---:|---|"]
    for n, x in v.items():
        lines.append(f"| {n} | {x['post_switch_mse_mean']:.5f} | " +
                     ", ".join(f"{y:.5f}" for y in x["post_switch_mse_per_seed"]) +
                     f" | {x['post_switch_window_median']:.5f} | {x['pre_switch_mse_mean']:.5f} | "
                     f"{x['forecast_recovery_windows']} |")
    lines += ["", "| Difference (post-switch MSE) | mean | 95% CI | per seed |", "|---|---:|---|---|"]
    for k, d in s["differences"].items():
        lines.append(f"| {k} | {d['difference']:.6f} | [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}] | " +
                     ", ".join(f"{y:.6f}" for y in d["per_seed"]) + " |")
    for n in ("learned_det_off", "learned_det_on"):
        lines += ["", f"{n}: detector " + "; ".join(
            f"FA={d['false_alarms_pre_switch']} hit={d['detected_within_200']} delay={d['delay_windows']}"
            for d in v[n]["detector"]), "", f"{n}: graph " + "; ".join(
            f"preF1={g['pre_switch_f1']:.3f} post200F1={g['post200_f1_mean']:.3f} finalF1={g['final_f1']:.3f} "
            f"topkF1 pre/post200/final={g['pre_switch_f1_topk']}/{g['post200_f1_topk']}/{g['final_f1_topk']} "
            f"AUROC={g['final_auroc']:.3f} SHD={g['final_shd']} rec={g['shd_recovery_windows']}"
            for g in v[n]["graph"])]
    (ROOT / phase / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["tuning", "select", "heldout"])
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "tuning":
        commit = rf.committed_hash(ROOT / "tuning_protocol.yaml")
        idx = [int(i) for i in args.pairs.split(",")] if args.pairs else range(3)
        for i in idx:
            run_pair(rf.DEV_PAIRS[i], "tuning", None)
        return
    if args.phase == "select":
        rf.committed_hash(ROOT / "tuning_protocol.yaml")
        sel = select_config()
        _write_json(ROOT / "tuning" / "selection.json", sel)
        for p in rf.DEV_PAIRS:  # dev-stream reference run of all variants with the chosen config
            run_pair(p, "tuning", None)
            base = ROOT / "tuning" / f"data{p[0]}_model{p[1]}"
            data = s5_data(p)
            for v in ("identity_online", "dense_online", "random_online", "oracle_online",
                      "learned_det_off", "learned_det_on"):
                run_online(v, p, data, base / "pretrain", sel["selected"], base / f"{v}.json")
        summarize("tuning", rf.DEV_PAIRS, "dev")
        print(json.dumps(sel["selected"]))
        return
    cfg = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    commit = rf.authorize_held_out(GATES, rf.S5_PAIRS)
    config = cfg["frozen_online_config"]
    if args.summarize:
        s = summarize("heldout", rf.S5_PAIRS, commit)
        print(json.dumps({"passed": s["passed"], "gate": s["gate"]}))
        return
    idx = [int(i) for i in args.pairs.split(",")] if args.pairs else range(5)
    for i in idx:
        run_pair(rf.S5_PAIRS[i], "heldout", config)


if __name__ == "__main__":
    sys.exit(main())
