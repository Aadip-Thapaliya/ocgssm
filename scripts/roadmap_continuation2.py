"""Saved-output-only analyses for roadmap continuation 2; never trains models."""

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts import roadmap_final as rf
from scripts import roadmap_s5 as s5
from scripts.d20_equal_training import _write_json

OUT = rf.ROOT / "continuation2" / "task_a"
S5_PAIRS = [(79, 1301), (83, 1409), (89, 1511), (97, 1613), (101, 1709)]
S5B_PAIRS = [(113, 2111), (127, 2203), (131, 2309), (137, 2411), (139, 2503)]
VARIANTS = ["learned_det_on", "dense_online", "identity_online", "oracle_online"]


def _read(root: Path, pairs: list[tuple[int, int]], variant: str) -> list[np.ndarray]:
    result = []
    for pair in pairs:
        path = root / f"data{pair[0]}_model{pair[1]}" / f"{variant}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["pair"] != list(pair) or len(rec.get("errors", [])) != 971:
            raise RuntimeError(f"saved-result identity/length mismatch: {path}")
        result.append(s5.window_mse(rec))
    return result


def _mean_ci(series: list[np.ndarray], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = np.empty(rf.RESAMPLES)
    for d in range(rf.RESAMPLES):
        draws[d] = np.mean([x[rf._block_index(rng, len(x), rf.BLOCK)].mean() for x in series])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(np.mean([x.mean() for x in series])),
            "ci95": [float(lo), float(hi)]}


def _bootstrap_table() -> dict[str, Any]:
    roots_pairs = [(s5.ROOT / "heldout", S5_PAIRS), (rf.ROOT / "continuation" / "s5b" /
                                                       "heldout", S5B_PAIRS)]
    data: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    seed_labels = []
    for root, pairs in roots_pairs:
        for pair in pairs:
            seed_labels.append(f"{pair[0]}/{pair[1]}")
        for variant in VARIANTS:
            data[variant].extend(_read(root, pairs, variant))
    if len(seed_labels) != 10:
        raise RuntimeError("expected exactly 10 held-out seeds")
    # Relative to the first wholly post-switch window; all ranges are half-open.
    ranges = {
        "post_switch_[0,200)_post_hoc_descriptive": (s5.FIRST_POST, s5.FIRST_POST + 200),
        "post_switch_[200,495)_post_hoc_descriptive": (s5.FIRST_POST + 200,
                                                       s5.FIRST_POST + 495),
        "pre_switch_[0,471)_post_hoc_descriptive": (0, s5.LAST_PRE + 1),
    }
    out: dict[str, Any] = {"seed_pairs": seed_labels, "n_seeds": len(seed_labels),
                           "bootstrap": {"method": "paired circular block bootstrap",
                                         "block_windows": rf.BLOCK,
                                         "resamples": rf.RESAMPLES,
                                         "aggregation": "equal-seed mean",
                                         "pairing": "same seed and time-block indices across variants"},
                           "ranges": {}}
    for j, (label, (lo, hi)) in enumerate(ranges.items()):
        cut = {v: [x[lo:hi] for x in arrs] for v, arrs in data.items()}
        out["ranges"][label] = {
            "stream_window_range_half_open": [lo, hi],
            "relative_post_switch_range": ([0, 200] if "0,200" in label else
                                           [200, 495] if "200,495" in label else None),
            "n_windows_per_seed": hi - lo,
            "means": {v: _mean_ci(a, 500100 + j * 10 + k)
                      for k, (v, a) in enumerate(cut.items())},
            "learned_det_on_minus": {
                v: rf.paired_difference(cut["learned_det_on"], vals, 501100 + j * 10 + k)
                for k, (v, vals) in enumerate(cut.items()) if v != "learned_det_on"}}
    return out


def saturation_availability() -> dict[str, Any]:
    checks = []
    requested = [("S5", s5.ROOT / "heldout", S5_PAIRS, ["learned_det_on"]),
                 ("S5b", rf.ROOT / "continuation" / "s5b" / "heldout",
                  S5B_PAIRS, ["learned_det_on", "learned_fast"])]
    for stage, root, pairs, variants in requested:
        for pair in pairs:
            for variant in variants:
                path = root / f"data{pair[0]}_model{pair[1]}" / f"{variant}.json"
                rec = json.loads(path.read_text(encoding="utf-8"))
                keys = sorted(rec.keys())
                timeline_keys = sorted(rec.get("graph_timeline", {}).keys())
                has_snapshots = any(k in rec for k in ("graph_history", "ema_graph_snapshots",
                                                       "pre_update_graphs", "adjacency_history"))
                checks.append({"stage": stage, "pair": list(pair), "variant": variant,
                               "path": str(path), "top_level_keys": keys,
                               "graph_timeline_keys": timeline_keys,
                               "saved_pre_update_ema_snapshots": has_snapshots})
    return {"status": "UNAVAILABLE: saved records contain no pre-update EMA graph snapshots; "
                      "only aggregate graph_timeline metrics were retained. No model was rerun and "
                      "no topology proxy substituted.",
            "checks": checks}


def write_report(result: dict[str, Any], saturation: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "pooled.json", result)
    _write_json(OUT / "saturation_availability.json", saturation)
    lines = ["# Continuation 2 Task A: pooled S5 + S5b (saved outputs only)", "",
             "All comparisons are post hoc and descriptive, including the original frozen first-200 "
             "gate displayed here as descriptive pooled analysis. Ten independent held-out seed pairs "
             "are pooled: the original five S5 and five fresh S5b pairs. Means give each seed equal "
             "weight. Paired circular block bootstrap: 12-window blocks, 2000 resamples; for each "
             "bootstrap draw, identical within-seed circular block indices are used across variants. "
             "Reported contrasts are learned_det_on minus comparator. All ranges are half-open.", ""]
    for label, item in result["ranges"].items():
        lines += [f"## {label}", "",
                  f"Stream window indices [{item['stream_window_range_half_open'][0]}, "
                  f"{item['stream_window_range_half_open'][1]}); {item['n_windows_per_seed']} "
                  "windows per seed.", "",
                  "| Variant | Mean MSE [95% CI] | learned_det_on minus variant [95% CI] |",
                  "|---|---:|---:|"]
        for variant, stats in item["means"].items():
            diff = item["learned_det_on_minus"].get(variant)
            dtext = (f"{diff['difference']:+.6f} [{diff['ci95'][0]:+.6f}, "
                     f"{diff['ci95'][1]:+.6f}]" if diff else "-")
            lines.append(f"| {variant} | {stats['mean']:.6f} [{stats['ci95'][0]:.6f}, "
                         f"{stats['ci95'][1]:.6f}] | {dtext} |")
        lines.append("")
    lines += ["## Saturation diagnostic availability", "", saturation["status"], "",
              "Files were checked for saved per-window graph history/snapshot fields. Both the S5 "
              "and S5b online JSONs retain forecast errors, detections, and aggregate graph metrics "
              "(F1/AUROC/SHD/top-|E| F1), but not the requested pre-update EMA adjacency matrices. "
              "Therefore the mean probability on newly true edges vs other entries at windows 0, "
              "50, 100, and 200—and the <0.05 determination—cannot be calculated from saved data. "
              "No new model runs were made.", ""]
    (OUT / "pooled.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_report(_bootstrap_table(), saturation_availability())
    print(OUT / "pooled.md")


if __name__ == "__main__":
    sys.exit(main())
