"""Continuation Task 1 (S4b): stationary held-out confirmation, same policy as S4.

Identical to S4 except the generator has no switch (switch_at = n_steps = 4000), fresh seed
pairs, and the variant list given in the frozen gates. Held-out data is built only after
``s4b/gates.yaml`` is committed.
"""

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

ROOT = rf.ROOT / "continuation" / "s4b"
GATES = ROOT / "gates.yaml"
PAIRS = [(103, 1801), (107, 1907), (109, 2003)]
VARIANTS = {k: rf.S4_VARIANTS[k] for k in
            ("identity", "dense", "random", "learned", "oracle", "var_direct", "dlinear")}
_AUTHORIZED: set[tuple[int, int]] = set()


def stationary_data(pair: tuple[int, int]) -> dict[str, Any]:
    if pair not in _AUTHORIZED:
        raise RuntimeError(f"LEAKAGE GUARD: S4b pair {pair} used before gates commit")
    if pair in rf.BURNED_PAIRS or pair in rf.HELD_OUT or pair in rf.DEV_PAIRS:
        raise RuntimeError(f"pair {pair} is not fresh")
    steps = rf.DATA["steps"]
    data = generate_svar(n_steps=steps, n_variables=20, n_lags=rf.DATA["lags"], switch_at=steps,
                         blend_steps=0, sparsity=rf.DATA["sparsity"],
                         noise_scale=rf.DATA["noise_scale"], seed=pair[0],
                         cross_variable_scale=rf.DATA["cross_variable_scale"])
    series, graph = data["series"], data["graph"]
    if any(not torch.equal(graph[t], graph[0]) for t in range(steps)):
        raise RuntimeError("S4b generator must be stationary")
    length, horizon = rf.DATA["input_length"], rf.DATA["horizon"]
    segments = {"train": (0, 2400), "tune": (2400, 2666), "test": (3200, 4000)}
    out: dict[str, Any] = {"graph0": graph[0].clone(), "variables": 20, "segments": segments}
    for name, (start, stop) in segments.items():
        x, y = make_windows(series[start:stop], length, horizon)
        out[name] = (x, y, graph[start + length - 1: start + length - 1 + len(x)])
        out[f"{name}_target_start"] = start + length
    return out


def run(indices: list[int] | None, summarize: bool) -> None:
    cfg = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    if cfg["pairs"] != [list(p) for p in PAIRS] or cfg["variants"] != list(VARIANTS):
        raise RuntimeError("gates.yaml does not match the runner configuration")
    commit = rf.committed_hash(GATES)
    _AUTHORIZED.update(PAIRS)
    torch.set_num_threads(1)
    if not summarize:
        for i in (indices if indices is not None else range(len(PAIRS))):
            pair = PAIRS[i]
            data = stationary_data(pair)
            runs = ROOT / "default" / f"data{pair[0]}_model{pair[1]}"
            for v, spec in VARIANTS.items():
                spec = dict(spec)
                if spec.get("graph_mode") == "random":
                    spec["random_density"] = float(data["graph0"].mean())
                rec = rf.train_and_eval(runs / f"{v}.json", spec, pair, data, "test")
                if rec["pair"] != list(pair) or rec["eval_target_start"] != 3224:
                    raise RuntimeError(f"identity check failed for {runs / v}")
        return
    records: dict[str, list[dict[str, Any]]] = {}
    for pair in PAIRS:
        for v in VARIANTS:
            path = ROOT / "default" / f"data{pair[0]}_model{pair[1]}" / f"{v}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["pair"] != list(pair):
                raise RuntimeError(f"pair mismatch in {path}")
            records.setdefault(v, []).append(rec)
    comps = [("learned", b) for b in VARIANTS if b != "learned"]
    comps += [("oracle", "dense"), ("dense", "identity"), ("oracle", "var_direct")]
    table = rf.ratio_table(records, comps)
    primary = {
        "learned/dense CI<1 H1": table["learned/dense"][0]["ci95"][1] < 1,
        "learned/random CI<1 H1": table["learned/random"][0]["ci95"][1] < 1,
        "learned/identity CI<1 H1": table["learned/identity"][0]["ci95"][1] < 1,
        "learned/identity CI<1 H2": table["learned/identity"][1]["ci95"][1] < 1,
    }
    secondary = {f"learned/oracle<=1.05 H{h + 1}": table["learned/oracle"][h]["ratio"] <= 1.05
                 for h in (0, 1)}
    out = {"gates_commit": commit, "ratios": table, "primary_gate": primary,
           "primary_passed": all(primary.values()), "secondary": secondary,
           "stats": rf.variant_stats(records)}
    _write_json(ROOT / "summary.json", out)
    lines = ["# S4b stationary held-out confirmation (no switch; test slice 3200-3999; seeds "
             + ", ".join(f"{a}/{b}" for a, b in PAIRS) + ")", "",
             f"Gates commit `{commit}`. S4b corrects the S4 slice design flaw (switch at 3600 inside "
             "the S4 test slice); the original S4 remains a frozen FAIL.", "",
             f"Primary gate: **{'PASS' if out['primary_passed'] else 'FAIL'}** - " +
             ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in primary.items()), "",
             "Secondary: " + ", ".join(f"{k}: {v}" for k, v in secondary.items()), "",
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
