"""Roadmap S3 convergence check for the learned graph (development seeds only)."""

import json
import statistics
import sys
from typing import Any

import torch

from scripts.d20_equal_training import _write_json
from scripts.roadmap_s1 import ROOT, _paired_ratio_ci, _protocol


def convergence(curve: list[float], early_cap: int = 20) -> dict[str, Any]:
    """Relative gain of best tune MSE over the full curve versus the first early_cap epochs."""
    best_early = min(curve[:early_cap])
    best_all = min(curve)
    return {"best_first_20": best_early, "best_all": best_all,
            "relative_gain": (best_early - best_all) / best_early}


def main(stage: str = "s3") -> dict[str, Any]:
    protocol = _protocol()
    cfg = protocol["stages"][stage]
    variant = next(iter(cfg["variants"]))
    cap = int(cfg["training_overrides"]["epoch_cap"])
    pairs = [tuple(p) for p in protocol["seeds"]["development_pairs"]]
    runs, refs = [], {"oracle_iterated": [], "dense_iterated": [], "var_direct": []}
    for pair in pairs:
        folder = f"data{pair[0]}_model{pair[1]}"
        runs.append(json.loads((ROOT / stage / "runs" / folder / f"{variant}.json").read_text()))
        for ref in refs:
            refs[ref].append(json.loads((ROOT / "s1" / "runs" / folder / f"{ref}.json").read_text()))
    seeds = []
    for pair, run in zip(pairs, runs):
        seeds.append({"pair": list(pair), "epochs_executed": run["epochs_executed"],
                      "best_epoch": run["best_validation_epoch"],
                      "cap_reached": run["reached_epoch_cap"],
                      "minutes": run["elapsed_seconds"] / 60,
                      **convergence(run["validation_mse_curve"])})
    cap_hits = sum(s["cap_reached"] for s in seeds)
    gains_ok = all(s["relative_gain"] < 0.005 for s in seeds)
    ratios = {}
    for ref, rows in refs.items():
        ratios[ref] = {}
        for h in (0, 1):
            a = [[w[h] for w in r["gate_errors"]] for r in runs]
            b = [[w[h] for w in r["gate_errors"]] for r in rows]
            ratio = statistics.fmean(map(statistics.fmean, a)) / statistics.fmean(map(statistics.fmean, b))
            ratios[ref][f"H{h + 1}"] = {"ratio": ratio, "ci95": _paired_ratio_ci(a, b, 900000 + h, 12, 2000)}
    report = {"stage": stage, "variant": variant, "epoch_cap": cap, "seeds": seeds,
              "cap_hits": cap_hits, "all_gains_below_0_5pct": gains_ok,
              "passed": cap_hits <= 1 and gains_ok, "informative_ratios": ratios}
    _write_json(ROOT / stage / "convergence.json", report)
    _plot(report, runs)
    lines = [f"# Roadmap {stage}: learned-graph convergence (development only)", "",
             f"{variant}, cap {cap}, patience 5. Gate: cap hit on <= 1/3 seeds and gain from epochs 21-{cap} "
             f"< 0.5% on every seed -> **{'PASS' if report['passed'] else 'FAIL'}** (cap hits {cap_hits}/3).", "",
             "| Pair | Epochs | Best epoch | Cap hit | Best tune MSE (<=20) | Best (all) | Gain | Min |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for s in seeds:
        lines.append(f"| {s['pair'][0]}/{s['pair'][1]} | {s['epochs_executed']} | {s['best_epoch']} | "
                     f"{s['cap_reached']} | {s['best_first_20']:.6f} | {s['best_all']:.6f} | "
                     f"{100 * s['relative_gain']:.2f}% | {s['minutes']:.1f} |")
    lines += ["", "Informative validation ratios (learned / reference, paired 95% CI; not S2 gates):", "",
              "| Reference | H1 | H2 |", "|---|---|---|"]
    for ref, row in ratios.items():
        lines.append(f"| {ref} | {row['H1']['ratio']:.4f} [{row['H1']['ci95'][0]:.4f}, {row['H1']['ci95'][1]:.4f}] | "
                     f"{row['H2']['ratio']:.4f} [{row['H2']['ci95'][0]:.4f}, {row['H2']['ci95'][1]:.4f}] |")
    (ROOT / stage / "convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cap_hits": cap_hits}))
    return report


def _plot(report: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, run in zip(report["seeds"], runs):
        curve = run["validation_mse_curve"]
        ax.plot(range(1, len(curve) + 1), curve, label=f"{seed['pair'][0]}/{seed['pair'][1]}")
        ax.scatter([seed["best_epoch"]], [min(curve)], marker="o")
    ax.axvline(20, color="grey", linestyle="--", linewidth=0.8, label="old cap 20")
    ax.set(xlabel="epoch", ylabel="tune MSE (2400-2665)", yscale="log",
           title=f"{report['variant']} validation curves (cap {report['epoch_cap']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / report["stage"] / "s3_validation_curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main(sys.argv[1] if len(sys.argv) > 1 else "s3")
