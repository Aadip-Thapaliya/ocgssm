"""Continuation 2 Task B: stationary held-out D20_T600 confirmation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from ocg.data import generate_svar, make_windows
from scripts import roadmap_final as rf
from scripts.d20_equal_training import _write_json

ROOT = rf.ROOT / "continuation2" / "s4b_hard"
GATES = ROOT / "gates.yaml"
PAIRS = [(149, 2609), (151, 2713), (157, 2819)]
VARIANTS = {k: rf.S4_VARIANTS[k] for k in
            ("identity", "dense", "random", "learned", "oracle", "var_direct")}


def _data(pair: tuple[int, int]) -> dict[str, Any]:
    data = generate_svar(n_steps=4000, n_variables=20, n_lags=2, switch_at=4000,
                         blend_steps=0, sparsity=0.1, noise_scale=0.1,
                         seed=pair[0], cross_variable_scale=1.0)
    series, graph = data["series"], data["graph"]
    if any(not torch.equal(graph[t], graph[0]) for t in range(4000)):
        raise RuntimeError("stationarity assertion failed")
    segments = {"train": (1800, 2400), "tune": (2400, 2666), "test": (3200, 4000)}
    out: dict[str, Any] = {"graph0": graph[0].clone(), "variables": 20, "segments": segments}
    for name, (start, stop) in segments.items():
        x, y = make_windows(series[start:stop], 24, 6)
        out[name] = (x, y, graph[start + 23: start + 23 + len(x)])
        out[f"{name}_target_start"] = start + 24
    if len(out["test"][0]) != 771 or out["test_target_start"] != 3224:
        raise RuntimeError("test window geometry mismatch")
    return out


def run(indices: list[int] | None = None, summarize: bool = False) -> None:
    cfg = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    if cfg["pairs"] != [list(p) for p in PAIRS] or cfg["variants"] != list(VARIANTS):
        raise RuntimeError("frozen gate configuration mismatch")
    if set(PAIRS) & (rf.BURNED_PAIRS | set(rf.DEV_PAIRS) | rf.HELD_OUT):
        raise RuntimeError("S4b-hard pairs are not fresh")
    gates_commit = rf.authorize_held_out(GATES, PAIRS)
    torch.set_num_threads(1)
    if not summarize:
        for i in indices if indices is not None else range(len(PAIRS)):
            pair = PAIRS[i]
            data = _data(pair)
            runs = ROOT / "default" / f"data{pair[0]}_model{pair[1]}"
            for name, spec in VARIANTS.items():
                spec = dict(spec)
                if spec.get("graph_mode") == "random":
                    spec["random_density"] = float(data["graph0"].mean())
                path = runs / f"{name}.json"
                if path.exists():
                    rec = json.loads(path.read_text(encoding="utf-8"))
                    if rec.get("pair") != list(pair) or rec.get("spec") != spec:
                        raise RuntimeError(f"saved-result identity mismatch: {path}")
                    continue
                rec = rf.train_and_eval(path, spec, pair, data, "test",
                                        training={**rf.TRAINING, "epoch_cap": 40,
                                                  "patience": 5})
                if rec["pair"] != list(pair) or rec["eval_target_start"] != 3224:
                    raise RuntimeError(f"run identity mismatch: {path}")
        return
    records: dict[str, list[dict[str, Any]]] = {}
    for pair in PAIRS:
        for name in VARIANTS:
            path = ROOT / "default" / f"data{pair[0]}_model{pair[1]}" / f"{name}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["pair"] != list(pair) or rec["eval_target_start"] != 3224:
                raise RuntimeError(f"saved result identity mismatch: {path}")
            records.setdefault(name, []).append(rec)
    table = rf.ratio_table(records, [("learned", n) for n in VARIANTS if n != "learned"])
    primary = {
        "learned/dense H1 CI upper < 1": table["learned/dense"][0]["ci95"][1] < 1,
        "learned/dense H1 point ratio < 0.9857": table["learned/dense"][0]["ratio"] < 0.9857,
        "learned/identity H1 CI upper < 1": table["learned/identity"][0]["ci95"][1] < 1,
        "learned/random H1 CI upper < 1": table["learned/random"][0]["ci95"][1] < 1,
    }
    out = {"gates_commit": gates_commit, "ratios": table, "primary_gate": primary,
           "primary_passed": all(primary.values()), "stats": rf.variant_stats(records)}
    _write_json(ROOT / "summary.json", out)
    lines = ["# S4b-hard stationary D20_T600 (seeds 149/2609, 151/2713, 157/2819)", "",
             f"Gates commit `{gates_commit}`. This is a new stationary confirmation; original S4-hard "
             "mixed-regime evidence is not replaced.", "",
             f"Primary gate: **{'PASS' if out['primary_passed'] else 'FAIL'}** - " +
             ", ".join(f"{key}: {'pass' if val else 'FAIL'}" for key, val in primary.items()), "",
             *rf.markdown_ratios(table), "", *rf.markdown_stats(out["stats"]), ""]
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"primary_passed": out["primary_passed"], **primary}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    run([int(i) for i in args.pairs.split(",")] if args.pairs else None, args.summarize)


if __name__ == "__main__":
    sys.exit(main())
