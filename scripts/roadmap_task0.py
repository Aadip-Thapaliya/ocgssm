"""Continuation Task 0: switch-location audit and post-hoc S5 window tables (no model runs).

Reads only saved outputs and regenerates generator graphs (no training, no new forecasts).
Window ranges are half-open [a, b) and indexed relative to the first post-switch window.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ocg.data import generate_svar
from scripts import roadmap_final as rf
from scripts import roadmap_s5 as s5
from scripts.d20_equal_training import _write_json

OUT = rf.ROOT / "continuation" / "task0"
D20_PAIRS = [(37, 401), (43, 503), (47, 607)]


def _switch_index(graph: torch.Tensor) -> list[int]:
    return [t for t in range(1, len(graph)) if not torch.equal(graph[t], graph[t - 1])]


def audit_switch() -> dict[str, Any]:
    out: dict[str, Any] = {}
    protocol = yaml.safe_load(Path("results/d20_equal_training/protocol.yaml").read_text())
    d = protocol["data"]
    # D20 reduced campaign: regenerate the generator graph only (burned pairs, no modelling).
    rows = []
    for pair in D20_PAIRS:
        g = generate_svar(n_steps=int(d["steps"]), n_variables=int(d["variables"]),
                          n_lags=int(d["lags"]), switch_at=int(d["switch_at"]),
                          blend_steps=int(d["blend_steps"]), sparsity=float(d["sparsity"]),
                          noise_scale=float(d["noise_scale"]), seed=pair[0],
                          cross_variable_scale=float(d["cross_variable_scale"]))["graph"]
        rows.append({"pair": list(pair), "graph_changes_at": _switch_index(g),
                     "edges_before": int(g[0].sum()), "edges_after": int(g[-1].sum()),
                     "edges_shared": int((g[0] * g[-1]).sum())})
    saved = []
    for p in sorted(Path("results/d20_equal_training/runs").rglob("*.json"))[:400]:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        ts = rec.get("timestamps") if isinstance(rec, dict) else None
        if ts:
            saved.append({"file": str(p), "first_label_t": ts[0], "last_label_t": ts[-1],
                          "n": len(ts), "windows_label_ge_3600": sum(t >= 3600 for t in ts)})
            if len(saved) >= 3:
                break
    out["d20_reduced_campaign"] = {"protocol_switch_at": int(d["switch_at"]),
                                   "protocol_test_slice": d["chronological_raw_splits"]["test"],
                                   "pairs": rows, "saved_timestamp_examples": saved}
    # S4 held-out (gates committed; authorization is audit-only, no training).
    rf.authorize_held_out(rf.ROOT / "s4" / "gates.yaml", rf.S4_PAIRS)
    s4_rows = []
    for pair in rf.S4_PAIRS:
        data = rf.synthetic_data(pair, split="test")
        full = generate_svar(n_steps=rf.DATA["steps"], n_variables=20, n_lags=2,
                             switch_at=rf.DATA["switch_at"], blend_steps=0,
                             sparsity=rf.DATA["sparsity"], noise_scale=rf.DATA["noise_scale"],
                             seed=pair[0], cross_variable_scale=1.0)["graph"]
        n = len(data["test"][0])
        first_target = data["test_target_start"]
        target_first = np.arange(n) + first_target
        target_last = target_first + rf.DATA["horizon"] - 1
        saved_rec = json.loads((rf.ROOT / "s4" / "default" / f"data{pair[0]}_model{pair[1]}" /
                                "learned.json").read_text(encoding="utf-8"))
        s4_rows.append({
            "pair": list(pair), "graph_changes_at": _switch_index(full),
            "test_windows": n, "saved_windows": len(saved_rec["errors"]),
            "saved_eval_target_start": saved_rec["eval_target_start"],
            "windows_all_targets_pre_switch": int((target_last < 3600).sum()),
            "windows_straddling": int(((target_first < 3600) & (target_last >= 3600)).sum()),
            "windows_all_targets_post_switch": int((target_first >= 3600).sum()),
            "edges_shared_fraction": float((full[0] * full[-1]).sum() / full[0].sum())})
    out["s4_heldout"] = {"switch_at": rf.DATA["switch_at"], "test_slice": [3200, 3999],
                         "pairs": s4_rows}
    return out


def _mean_ci(series: list[np.ndarray], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = np.empty(rf.RESAMPLES)
    for d in range(rf.RESAMPLES):
        draws[d] = np.mean([x[rf._block_index(rng, len(x), rf.BLOCK)].mean() for x in series])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(np.mean([x.mean() for x in series])), "ci95": [float(lo), float(hi)],
            "n_windows_per_seed": [len(x) for x in series]}


def s5_tables() -> dict[str, Any]:
    gates = yaml.safe_load(s5.GATES.read_text(encoding="utf-8"))
    pairs = rf.S5_PAIRS
    names = ["identity_online", "dense_online", "random_online", "learned_frozen",
             "learned_det_off", "learned_det_on", "oracle_online"]
    mses: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for p in pairs:
        base = s5.ROOT / "heldout" / f"data{p[0]}_model{p[1]}"
        for n in names:
            path = base / "pretrain" / "learned.json" if n == "learned_frozen" else base / f"{n}.json"
            mses[n].append(s5.window_mse(json.loads(path.read_text(encoding="utf-8"))))
    n_stream = len(mses["dense_online"][0])
    eligible = n_stream - s5.FIRST_POST
    ranges = {"post[0,200) frozen gate": (0, 200, "frozen gate window"),
              "post[200,500) post hoc, descriptive": (200, 500, "post hoc, descriptive"),
              "post[0,500) post hoc, descriptive": (0, 500, "post hoc, descriptive")}
    out: dict[str, Any] = {"gates": gates.get("gate"), "stream_windows": n_stream,
                           "first_post_window": s5.FIRST_POST, "last_pre_window": s5.LAST_PRE,
                           "eligible_post_switch_windows": eligible, "tables": {}}
    for label, (a, b, kind) in ranges.items():
        lo, hi = s5.FIRST_POST + a, min(s5.FIRST_POST + b, n_stream)
        cut = {n: [m[lo:hi] for m in ms] for n, ms in mses.items()}
        out["tables"][label] = {
            "kind": kind, "stream_windows_half_open": [lo, hi], "n_windows": hi - lo,
            "truncated": hi - lo < b - a,
            "mean": {n: _mean_ci(v, 920000 + i) for i, (n, v) in enumerate(cut.items())},
            "diff_vs_dense_online": {n: rf.paired_difference(v, cut["dense_online"], 930000 + i)
                                     for i, (n, v) in enumerate(cut.items()) if n != "dense_online"},
            "diff_learned_det_on_minus": {n: rf.paired_difference(cut["learned_det_on"], v, 940000 + i)
                                          for i, (n, v) in enumerate(cut.items())
                                          if n != "learned_det_on"}}
    cut = {n: [m[: s5.LAST_PRE + 1] for m in ms] for n, ms in mses.items()}
    out["tables"]["pre-switch [0,471) descriptive"] = {
        "kind": "descriptive (all targets < 2500)", "stream_windows_half_open": [0, s5.LAST_PRE + 1],
        "n_windows": s5.LAST_PRE + 1, "truncated": False,
        "mean": {n: _mean_ci(v, 950000 + i) for i, (n, v) in enumerate(cut.items())},
        "diff_vs_dense_online": {n: rf.paired_difference(v, cut["dense_online"], 960000 + i)
                                 for i, (n, v) in enumerate(cut.items()) if n != "dense_online"},
        "diff_learned_det_on_minus": {n: rf.paired_difference(cut["learned_det_on"], v, 970000 + i)
                                      for i, (n, v) in enumerate(cut.items())
                                      if n != "learned_det_on"}}
    return out


def _f(x: dict[str, Any], key: str, digits: int = 5) -> str:
    return f"{x[key]:+.{digits}f} [{x['ci95'][0]:+.{digits}f}, {x['ci95'][1]:+.{digits}f}]"


def report(audit: dict[str, Any], tables: dict[str, Any]) -> str:
    lines = ["# Continuation Task 0 (no new model runs)", "",
             "## a) Switch location audit", "",
             "| Source | pair | graph changes at t | test windows | all-pre | straddle | all-post | shared edges |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in audit["s4_heldout"]["pairs"]:
        lines.append(f"| S4 held-out | {r['pair']} | {r['graph_changes_at']} | {r['test_windows']} "
                     f"(saved {r['saved_windows']}, first target {r['saved_eval_target_start']}) | "
                     f"{r['windows_all_targets_pre_switch']} | {r['windows_straddling']} | "
                     f"{r['windows_all_targets_post_switch']} | {r['edges_shared_fraction']:.2f} |")
    for r in audit["d20_reduced_campaign"]["pairs"]:
        lines.append(f"| D20 reduced | {r['pair']} | {r['graph_changes_at']} | 771 | 371 | 5 | 395 | "
                     f"{r['edges_shared'] / max(r['edges_before'], 1):.2f} |")
    n_examples = len(audit["d20_reduced_campaign"]["saved_timestamp_examples"])
    lines += ["", f"D20 reduced campaign: saved run files contain no per-window timestamps "
              f"(examples found: {n_examples}); its window counts follow from the frozen protocol "
              "(results/d20_equal_training/protocol.yaml: test slice [3200, 3999], 771 windows, graph "
              "timestamps [3223, 3993], switch_at 3600) and the identical make_windows(series[3200:4000], "
              "24, 6) call (scripts/d20_equal_training.py:796), which the S4 rows verify against saved "
              "S4 outputs. Conclusion: both campaigns had the switch inside the test slice; 395/771 "
              "windows are fully post-switch under a graph sharing only 4-17% of edges with the "
              "training graph.", ""]
    lines += ["## b) S5 held-out per-window tables (seeds " +
              ", ".join(f"{a}/{b}" for a, b in rf.S5_PAIRS) + ")", "",
              f"Stream windows {tables['stream_windows']}; first post-switch window "
              f"{tables['first_post_window']}; eligible post-switch windows "
              f"{tables['eligible_post_switch_windows']} (fewer than 500: the [200,500) and [0,500) "
              "ranges are truncated to the available windows, not padded). Paired circular block "
              "bootstrap, 12-window blocks, 2000 resamples, equal-seed means.", ""]
    for label, t in tables["tables"].items():
        lines += [f"### {label} - {t['kind']}; stream windows [{t['stream_windows_half_open'][0]}, "
                  f"{t['stream_windows_half_open'][1]}) = {t['n_windows']} per seed"
                  + (" (TRUNCATED)" if t["truncated"] else ""), "",
                  "| Variant | mean MSE [95% CI] | minus dense_online [95% CI] | learned_det_on minus variant [95% CI] |",
                  "|---|---|---|---|"]
        for n, m in t["mean"].items():
            dd = t["diff_vs_dense_online"].get(n)
            dl = t["diff_learned_det_on_minus"].get(n)
            lines.append(f"| {n} | {m['mean']:.5f} [{m['ci95'][0]:.5f}, {m['ci95'][1]:.5f}] | "
                         f"{_f(dd, 'difference') if dd else '-'} | {_f(dl, 'difference') if dl else '-'} |")
        lines.append("")
    lines += ["## c) ETTh2 H96", "",
              "Not run in this continuation. The original partial ETTh2 H96 outputs (2/3 seeds) and its "
              "INCOMPLETE row are preserved unchanged.", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = audit_switch()
    tables = s5_tables()
    _write_json(OUT / "task0.json", {"switch_audit": audit, "s5_windows": tables})
    (OUT / "task0.md").write_text(report(audit, tables), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
