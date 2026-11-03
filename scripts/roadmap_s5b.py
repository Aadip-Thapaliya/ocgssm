"""Continuation Task 2 (S5b): learned_fast = EMA reset + replay-buffer structural updates.

Everything except the learned_fast mechanism is the frozen S5 protocol (scripts/roadmap_s5.py,
gates results/roadmap/final/s5/gates.yaml). Dev tuning reuses the S5 dev pretraining cache.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.fast_graph import enable_fast_tracking
from ocg.train import train_online
from scripts import roadmap_final as rf
from scripts import roadmap_s5 as s5
from scripts.d20_equal_training import _write_json

ROOT = rf.ROOT / "continuation" / "s5b"
TUNING_PROTOCOL = ROOT / "tuning_protocol.yaml"
GATES = ROOT / "gates.yaml"
HELD_OUT = [(113, 2111), (127, 2203), (131, 2309), (137, 2411), (139, 2503)]
S5_CONFIG = {"ema_decay": 0.95, "threshold": 3.5, "boost": 5.0}  # frozen S5 config 6
BASES = {"identity_online": "identity", "dense_online": "dense", "oracle_online": "oracle",
         "learned_det_on": "learned", "learned_fast": "learned", "learned_both": "learned"}


def _protocol() -> dict[str, Any]:
    cfg = yaml.safe_load(TUNING_PROTOCOL.read_text(encoding="utf-8"))
    rf.committed_hash(TUNING_PROTOCOL)
    return cfg


def _record_identity(path: Path, identity: dict[str, Any]) -> None:
    """Write an immutable sidecar identity, refusing to replace conflicting metadata."""
    sidecar = path.with_suffix(".identity.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"cached result identity mismatch: {path}")
        return
    _write_json(sidecar, identity)


def run_fast(pair: tuple[int, int], data: dict[str, Any], state_dir: Path,
             fast: dict[str, Any], path: Path) -> dict[str, Any]:
    protocol_commit = rf.committed_hash(TUNING_PROTOCOL)
    phase = "tuning" if path.is_relative_to(ROOT / "tuning") else "heldout"
    identity = {"stage": "S5b", "phase": phase, "pair": list(pair), "fast": fast,
                "online_config": S5_CONFIG, "tuning_protocol_commit": protocol_commit}
    if phase == "tuning":
        identity["candidate"] = int(path.stem.removeprefix("cand"))
    else:
        identity.update({"variant": "learned_fast", "s5b_gates_commit": rf.committed_hash(GATES)})
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["pair"] != list(pair) or rec["fast"] != fast:
            raise RuntimeError(f"identity mismatch in cached {path}")
        sidecar = path.with_suffix(".identity.json")
        if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"identity sidecar mismatch in {sidecar}")
        if path.is_relative_to(ROOT / "heldout") and not sidecar.exists():
            raise RuntimeError(f"held-out result lacks identity sidecar: {path}")
        return rec
    model = s5._model(dict(s5.PRETRAIN["learned"]), pair, data, state_dir / "learned.pt")
    learner = model.graph_learner
    learner.ema_decay = float(S5_CONFIG["ema_decay"])
    learner.boost = float(S5_CONFIG["boost"])
    learner.boost_steps = 20
    learner.detector_confirm_steps = 2
    learner.detector_rearm_steps = 10
    calibration = s5._calibrate(model, data, S5_CONFIG)
    enable_fast_tracking(learner, int(fast["buffer"]), int(fast["steps"]), bool(fast["reset"]))
    sx, sy, sg = data["stream"]
    started = time.perf_counter()
    torch.manual_seed(pair[1])
    result = train_online(
        model, sx, sy, learning_rate=0.001, seed=pair[1],
        timestamps=list(range(s5.STREAM_START + s5.L - 1, s5.STREAM_START + s5.L - 1 + len(sx))),
        update_graph=True, freeze_detector=True)
    elapsed = time.perf_counter() - started
    errors = (torch.tensor(result["predictions"]) - sy).square().mean(dim=2).tolist()
    record = {"variant": "learned_fast", "pair": list(pair), "config": S5_CONFIG, "fast": fast,
              "elapsed_seconds": elapsed, "errors": errors, "calibration": calibration,
              "detections": result["detections"], "fast_events": learner.fast_events,
              "graph_timeline": s5._graph_timeline(torch.tensor(result["graph_history"]), sg)}
    _write_json(path, record)
    _record_identity(path, identity)
    print(json.dumps({"run": str(path), "s": round(elapsed, 1)}), flush=True)
    return record


def tune(pair_index: int, config_index: int) -> None:
    cfg = _protocol()
    pair = rf.DEV_PAIRS[pair_index]
    data = s5.s5_data(pair)
    state = s5.ROOT / "tuning" / f"data{pair[0]}_model{pair[1]}" / "pretrain"
    meta = json.loads((state / "learned.json").read_text(encoding="utf-8"))
    if meta["pair"] != list(pair) or meta["spec"] != s5.PRETRAIN["learned"]:
        raise RuntimeError("S5 dev pretraining cache identity mismatch")
    fast = cfg["candidates"][config_index]
    run_fast(pair, data, state, fast,
             ROOT / "tuning" / f"data{pair[0]}_model{pair[1]}" / f"cand{config_index}.json")


def select() -> dict[str, Any]:
    cfg = _protocol()
    rows = []
    for k, fast in enumerate(cfg["candidates"]):
        vals = []
        for p in rf.DEV_PAIRS:
            path = ROOT / "tuning" / f"data{p[0]}_model{p[1]}" / f"cand{k}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["pair"] != list(p) or rec["fast"] != fast or rec["variant"] != "learned_fast":
                raise RuntimeError(f"dev run identity mismatch: {path}")
            identity = {"stage": "S5b", "phase": "tuning", "pair": list(p), "candidate": k,
                        "fast": fast, "tuning_protocol_commit": rf.committed_hash(TUNING_PROTOCOL)}
            sidecar = path.with_suffix(".identity.json")
            if sidecar.exists():
                saved = json.loads(sidecar.read_text(encoding="utf-8"))
                generic = {"stage": "S5b", "pair": list(p), "fast": fast,
                           "online_config": S5_CONFIG,
                           "tuning_protocol_commit": rf.committed_hash(TUNING_PROTOCOL)}
                rich = {**identity, "online_config": S5_CONFIG}
                if saved not in (identity, generic, rich):
                    raise RuntimeError(f"dev sidecar identity mismatch: {sidecar}")
            else:
                _record_identity(path, identity)
            vals.append(s5.post(rec).mean())
        rows.append({"candidate": k, **fast, "dev_post_switch_mse": float(np.mean(vals))})
    ref = [s5.post(json.loads((s5.ROOT / "tuning" / f"data{p[0]}_model{p[1]}" / "config6" /
                               "learned_det_on.json").read_text(encoding="utf-8"))).mean()
           for p in rf.DEV_PAIRS]
    out = {"grid": rows, "selected": min(rows, key=lambda r: r["dev_post_switch_mse"]),
           "reference_learned_det_on_config6_dev_post_switch_mse": float(np.mean(ref))}
    _write_json(ROOT / "tuning" / "selection.json", out)
    print(json.dumps(out["selected"]))
    return out


def _held_out_data(pair: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    rf.committed_hash(GATES)
    if gates["heldout_pairs"] != [list(p) for p in HELD_OUT]:
        raise RuntimeError("gates pairs do not match runner")
    if set(HELD_OUT) & (rf.HELD_OUT | rf.BURNED_PAIRS | set(rf.DEV_PAIRS)):
        raise RuntimeError("S5b held-out pairs must be fresh")
    rf._AUTHORIZED_HELD_OUT.update(HELD_OUT)
    return s5.s5_data(pair), gates


def heldout_job(pair_index: int, variant: str) -> None:
    torch.set_num_threads(1)
    pair = HELD_OUT[pair_index]
    data, gates = _held_out_data(pair)
    base_dir = ROOT / "heldout" / f"data{pair[0]}_model{pair[1]}"
    variants = [v for v in BASES if v != "learned_both"] if variant == "all" else (
        ["learned_det_on", "learned_fast"] if variant == "learned_both" else [variant])
    for base in dict.fromkeys(BASES[v] for v in variants):
        pretrain_path = base_dir / "pretrain" / f"{base}.json"
        rec = rf.train_and_eval(pretrain_path,
                                dict(s5.PRETRAIN[base]), pair, data, "stream", save_state=True)
        if rec["pair"] != list(pair):
            raise RuntimeError("pretrain identity mismatch")
        _record_identity(pretrain_path, {"stage": "S5b", "phase": "heldout-pretrain",
                                         "pair": list(pair), "base": base,
                                         "spec": dict(s5.PRETRAIN[base]),
                                         "s5_gates_commit": rf.committed_hash(s5.GATES),
                                         "s5b_gates_commit": rf.committed_hash(GATES)})
    for v in variants:
        if v == "learned_fast":
            run_fast(pair, data, base_dir / "pretrain", gates["frozen_fast_config"],
                     base_dir / "learned_fast.json")
        else:
            output_path = base_dir / f"{v}.json"
            s5.run_online(v, pair, data, base_dir / "pretrain", S5_CONFIG, output_path)
            _record_identity(output_path, {"stage": "S5b", "phase": "heldout-online",
                                           "pair": list(pair), "variant": v,
                                           "online_config": S5_CONFIG,
                                           "s5_gates_commit": rf.committed_hash(s5.GATES),
                                           "s5b_gates_commit": rf.committed_hash(GATES)})


def summarize() -> None:
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    commit = rf.committed_hash(GATES)
    names = ["identity_online", "dense_online", "learned_frozen", "learned_det_on",
             "learned_fast", "oracle_online"]
    records: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for p in HELD_OUT:
        base = ROOT / "heldout" / f"data{p[0]}_model{p[1]}"
        for n in names:
            path = base / "pretrain" / "learned.json" if n == "learned_frozen" else base / f"{n}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["pair"] != list(p):
                raise RuntimeError(f"pair mismatch {path}")
            records[n].append(rec)
    mses = {n: [s5.window_mse(r) for r in rs] for n, rs in records.items()}
    n_stream = len(mses["dense_online"][0])
    a, b = s5.FIRST_POST, s5.FIRST_POST + s5.POST_WINDOWS
    gate_diff = rf.paired_difference([m[a:b] for m in mses["learned_fast"]],
                                     [m[a:b] for m in mses["dense_online"]], 980000)
    windows = {"post[0,200) gate": (a, b), "post[200,495) descriptive": (b, n_stream),
               "pre-switch [0,471) descriptive": (0, s5.LAST_PRE + 1)}
    tables: dict[str, Any] = {}
    for label, (lo, hi) in windows.items():
        tables[label] = {
            "n_windows": hi - lo,
            "mean": {n: float(np.mean([m[lo:hi].mean() for m in ms])) for n, ms in mses.items()},
            "learned_fast_minus": {n: rf.paired_difference(
                [m[lo:hi] for m in mses["learned_fast"]], [m[lo:hi] for m in ms], 981000 + i)
                for i, (n, ms) in enumerate(mses.items()) if n != "learned_fast"}}
    tables["post[0,200) gate"]["learned_fast_minus"]["dense_online"] = gate_diff
    graph = {}
    for n in ("learned_det_on", "learned_fast"):
        rows = []
        for r in records[n]:
            tl = r["graph_timeline"]["f1_topk"]
            rows.append({"pre": float(np.mean(tl[:s5.DETECT_INDEX])),
                         "post200": float(np.mean(tl[s5.DETECT_INDEX:s5.DETECT_INDEX + 200])),
                         "final": tl[-1], "detector": s5.detector_summary(r)})
        graph[n] = rows
    passed = gate_diff["ci95"][1] < 0
    out = {"gates_commit": commit, "frozen_fast_config": gates["frozen_fast_config"],
           "gate": {"learned_fast - dense_online CI upper < 0 (post[0,200))": passed},
           "passed": passed, "gate_difference": gate_diff, "tables": tables, "graph_topk_f1": graph}
    _write_json(ROOT / "heldout" / "summary.json", out)
    _plot(records)
    lines = [f"# S5b held-out (seeds {', '.join(f'{p}/{q}' for p, q in HELD_OUT)})", "",
             f"Gates commit `{commit}`; frozen learned_fast config {gates['frozen_fast_config']}.", "",
             f"Primary gate (learned_fast - dense_online over first 200 post-switch windows, CI upper < 0): "
             f"**{'PASS' if passed else 'FAIL'}** - {gate_diff['difference']:+.6f} "
             f"[{gate_diff['ci95'][0]:+.6f}, {gate_diff['ci95'][1]:+.6f}]", ""]
    for label, t in tables.items():
        lines += [f"## {label} ({t['n_windows']} windows/seed)", "",
                  "| Variant | mean MSE | learned_fast minus variant [95% CI] |", "|---|---:|---|"]
        for n, m in t["mean"].items():
            d = t["learned_fast_minus"].get(n)
            lines.append(f"| {n} | {m:.5f} | " + (f"{d['difference']:+.6f} [{d['ci95'][0]:+.6f}, "
                                                  f"{d['ci95'][1]:+.6f}]" if d else "-") + " |")
        lines.append("")
    lines += ["## Graph top-|E| F1 and detector", ""]
    for n, rows in graph.items():
        for p, r in zip(HELD_OUT, rows):
            d = r["detector"]
            lines.append(f"- {n} {p}: F1 pre {r['pre']:.3f}, post200 {r['post200']:.3f}, final "
                         f"{r['final']:.3f}; FA={d['false_alarms_pre_switch']} delay={d['delay_windows']}")
    (ROOT / "heldout" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["gate"]), json.dumps(gate_diff))


def _plot(records: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for n in ("dense_online", "identity_online", "learned_det_on", "learned_fast", "oracle_online"):
        curve = np.mean([s5.window_mse(r) for r in records[n]], axis=0)
        smooth = np.convolve(curve, np.ones(15) / 15, mode="same")
        a1.plot(np.arange(len(curve)) - s5.FIRST_POST, smooth, label=n)
    for n in ("learned_det_on", "learned_fast"):
        f1 = np.mean([r["graph_timeline"]["f1_topk"] for r in records[n]], axis=0)
        a2.plot(np.arange(len(f1)) - s5.FIRST_POST, f1, label=n)
    for ax in (a1, a2):
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(0, 200, color="gray", alpha=0.1)
        ax.legend(fontsize=8)
    a1.set_ylabel("window MSE (15-window mean)")
    a2.set_ylabel("top-|E| F1")
    a2.set_xlabel("window relative to first post-switch window")
    fig.tight_layout()
    fig.savefig(ROOT / "heldout" / "s5b_post_switch.png", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["tune", "select", "heldout", "summarize"])
    parser.add_argument("--pair", type=int)
    parser.add_argument("--cand", type=int)
    parser.add_argument("--variant", choices=[*BASES, "all"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "tune":
        tune(args.pair, args.cand)
    elif args.phase == "select":
        select()
    elif args.phase == "heldout":
        heldout_job(args.pair, args.variant)
    else:
        summarize()


if __name__ == "__main__":
    sys.exit(main())
