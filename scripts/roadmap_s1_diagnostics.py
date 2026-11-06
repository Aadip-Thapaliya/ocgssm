"""S1 diagnostics (development seeds, validation only)."""

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocg.model import companion_reachability
from scripts.d20_equal_training import _write_json
from scripts.roadmap_s1 import ROOT, _check_frozen, _paired_ratio_ci, _protocol, _run_one, _seed_data

OUT = ROOT / "diagnostics"
REFERENCES = {
    "identity_iterated": {"graph_mode": "identity", "direct_head_mode": "iterated"},
    "dense_iterated": {"graph_mode": "dense", "direct_head_mode": "iterated"},
}


def _load(stage: str, variant: str, pair: tuple[int, int]) -> dict[str, Any]:
    path = ROOT / stage / "runs" / f"data{pair[0]}_model{pair[1]}" / f"{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_support_ols(data: dict[str, Any], horizon: int = 6) -> dict[str, Any]:
    """Per-horizon least squares on [x_t, x_{t-1}, 1], restricted to true reachability support."""
    graph0 = data["graph0"]
    masks = companion_reachability(graph0, horizon)  # (H,K,D,D)
    n_lags, d = graph0.shape[0], graph0.shape[1]

    def features(x: torch.Tensor) -> torch.Tensor:
        lags = [x[:, -(k + 1), :] for k in range(n_lags)]
        return torch.cat(lags + [torch.ones(x.shape[0], 1)], dim=1).double()

    train_x, train_y, _ = data["train"]
    gate_x, gate_y, _ = data["gate"]
    f_train, f_gate = features(train_x), features(gate_x)
    predictions = torch.zeros(gate_y.shape, dtype=torch.float64)
    ranks, deficits, conds = [], 0, []
    for h in range(horizon):
        for i in range(d):
            support = torch.cat([masks[h, k, i] for k in range(n_lags)] + [torch.ones(1)]).bool()
            design = f_train[:, support]
            solution = torch.linalg.lstsq(design, train_y[:, h, i].double(), driver="gelsd")
            rank = int(solution.rank)
            singular = torch.linalg.svdvals(design)
            ranks.append(rank)
            deficits += int(rank < int(support.sum()))
            conds.append(float(singular[0] / singular[-1]))
            predictions[:, h, i] = f_gate[:, support] @ solution.solution
    errors = (predictions - gate_y.double()).square().mean(dim=2)
    return {
        "gate_errors": errors.tolist(),
        "rank_deficient_regressions": deficits,
        "regressions": len(ranks),
        "max_condition_number": max(conds),
        "median_condition_number": statistics.median(conds),
        "mean_support_size_incl_intercept": statistics.fmean(ranks),
    }


def _densities(graph0: torch.Tensor) -> list[dict[str, Any]]:
    masks = companion_reachability(graph0, 6)
    d = graph0.shape[1]
    rows = []
    for h in range(6):
        for k in range(graph0.shape[0]):
            block = masks[h, k]
            rows.append({
                "horizon": h + 1, "lag": k + 1,
                "density": float(block.sum() / d**2),
                "edges": int(block.sum()),
                "self_edges": int(torch.diagonal(block).sum()),
            })
    return rows


def _compare(num: list[list[list[float]]], den: list[list[list[float]]], horizons=(0, 1)) -> dict[str, Any]:
    out = {}
    for h in horizons:
        a = [[w[h] for w in rows] for rows in num]
        b = [[w[h] for w in rows] for rows in den]
        ratio = statistics.fmean(map(statistics.fmean, a)) / statistics.fmean(map(statistics.fmean, b))
        ci = _paired_ratio_ci(a, b, 800000 + h, 12, 2000)
        out[f"H{h + 1}"] = {
            "ratio": ratio, "ci95": ci,
            "per_seed": [statistics.fmean(x) / statistics.fmean(y) for x, y in zip(a, b)],
        }
    return out


def main() -> None:
    protocol = _protocol()
    commit = _check_frozen()
    torch.set_num_threads(1)
    pairs = [tuple(p) for p in protocol["seeds"]["development_pairs"]]
    training = dict(protocol["training"])
    started = time.perf_counter()
    errors: dict[str, list] = {k: [] for k in ("oracle_iterated", "var_direct", "ols", *REFERENCES)}
    ols_meta, density = [], {}
    for pair in pairs:
        data = _seed_data(pair[0], protocol["data"])
        for variant, spec in REFERENCES.items():
            errors[variant].append(_run_one("s1", variant, spec, pair, data, training)["gate_errors"])
        for variant in ("oracle_iterated", "var_direct"):
            errors[variant].append(_load("s1", variant, pair)["gate_errors"])
        ols = oracle_support_ols(data)
        errors["ols"].append(ols.pop("gate_errors"))
        ols_meta.append({"pair": list(pair), **ols})
        density[f"{pair[0]}/{pair[1]}"] = _densities(data["graph0"])
        if pair == (67, 1009):
            cap40 = _run_one("s1_cap40", "oracle_iterated", {"direct_head_mode": "iterated"},
                             pair, data, dict(training, epoch_cap=40))
            cap40_ratio = _compare([cap40["gate_errors"]], [errors["var_direct"][-1]])
            cap20_ratio = _compare([errors["oracle_iterated"][-1]], [errors["var_direct"][-1]])
    comparisons = {
        "oracle_iterated/identity_iterated": _compare(errors["oracle_iterated"], errors["identity_iterated"]),
        "oracle_iterated/dense_iterated": _compare(errors["oracle_iterated"], errors["dense_iterated"]),
        "oracle_iterated/var_direct": _compare(errors["oracle_iterated"], errors["var_direct"]),
        "ols/var_direct": _compare(errors["ols"], errors["var_direct"]),
        "ols/identity_iterated": _compare(errors["ols"], errors["identity_iterated"]),
        "dense_iterated/var_direct": _compare(errors["dense_iterated"], errors["var_direct"]),
    }
    decision = {}
    for h in ("H1", "H2"):
        ols_clear = comparisons["ols/var_direct"][h]["ci95"][1] < 1.0
        oracle_clear = comparisons["oracle_iterated/var_direct"][h]["ci95"][1] < 1.0
        decision[h] = {"ols_clearly_beats_var_direct": ols_clear,
                       "oracle_iterated_clearly_beats_var_direct": oracle_clear,
                       "s1b_trigger": ols_clear and not oracle_clear}
    next_branch = "S1b" if all(decision[h]["s1b_trigger"] for h in ("H1", "H2")) else (
        "S1b" if any(decision[h]["s1b_trigger"] for h in ("H1", "H2")) else "S3")
    mean_density = {}
    for row_list in density.values():
        for row in row_list:
            mean_density.setdefault((row["horizon"], row["lag"]), []).append(row)
    density_table = [
        {"horizon": h, "lag": k,
         "mean_density": statistics.fmean(r["density"] for r in rows),
         "mean_edges": statistics.fmean(r["edges"] for r in rows),
         "mean_self_edges": statistics.fmean(r["self_edges"] for r in rows)}
        for (h, k), rows in sorted(mean_density.items())
    ]
    cap40_record = {k: cap40[k] for k in ("epochs_executed", "best_validation_epoch",
                                          "reached_epoch_cap", "best_validation_mse",
                                          "validation_mse_curve", "elapsed_seconds")}
    report = {
        "protocol_commit": commit,
        "comparisons": comparisons,
        "decision_rule": "S1b if, at a horizon in {H1,H2}, OLS/DirectVAR paired 95% CI upper < 1 and oracle_iterated/DirectVAR CI upper >= 1; else S3",
        "decision": decision, "next_branch": next_branch,
        "ols_rank_diagnostics": ols_meta,
        "reachability_density": density_table, "reachability_density_per_seed": density,
        "cap40_67_1009": {**cap40_record, "ratio_vs_var_direct": cap40_ratio,
                          "cap20_ratio_vs_var_direct": cap20_ratio},
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(OUT / "s1_diagnostics.json", report)
    _write_outputs(report)
    print(json.dumps({"next_branch": next_branch, "decision": decision}))


def _write_outputs(report: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = report["comparisons"]
    names = list(comps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, h in zip(axes, ("H1", "H2")):
        ratios = np.array([comps[n][h]["ratio"] for n in names])
        lo = ratios - np.array([comps[n][h]["ci95"][0] for n in names])
        hi = np.array([comps[n][h]["ci95"][1] for n in names]) - ratios
        ax.barh(range(len(names)), ratios, xerr=[lo, hi], capsize=3, color="C0")
        ax.axvline(1.0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlim(min(0.7, ratios.min() - 0.05), max(1.2, ratios.max() + 0.05))
        ax.set_title(f"{h}: MSE ratio, paired 95% block CI (validation 2933-3199)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "s1_diagnostics_ratios.png", dpi=120)
    plt.close(fig)

    lines = ["# S1 diagnostics (development seeds 67/1009, 71/1103, 73/1201; validation only)", "",
             "Gate slice: raw 2933-3199 (238 windows/seed); identical target timestamps and paired "
             "12-window circular block bootstrap (2000 resamples, equal-seed). Identity/dense use the "
             "selected iterated head and the S1 training policy (cap 20, patience 5, tune checkpoint).", "",
             "| Ratio | H1 [95% CI] | H2 [95% CI] |", "|---|---|---|"]
    for n, c in comps.items():
        lines.append(f"| {n} | {c['H1']['ratio']:.4f} [{c['H1']['ci95'][0]:.4f}, {c['H1']['ci95'][1]:.4f}] | "
                     f"{c['H2']['ratio']:.4f} [{c['H2']['ci95'][0]:.4f}, {c['H2']['ci95'][1]:.4f}] |")
    lines += ["", f"Decision rule: {report['decision_rule']}.", ""]
    for h, d in report["decision"].items():
        lines.append(f"- {h}: OLS clearly beats DirectVAR = {d['ols_clearly_beats_var_direct']}; "
                     f"oracle_iterated clearly beats = {d['oracle_iterated_clearly_beats_var_direct']}; "
                     f"S1b trigger = {d['s1b_trigger']}")
    lines += [f"- **Next branch: {report['next_branch']}**", "",
              "## Oracle-support OLS rank diagnostics",
              "Per horizon and target variable: lstsq (gelsd, float64) on train windows (raw 0-2399) "
              "with features [x_t, x_(t-1), 1] restricted to the true reachability support.", ""]
    for m in report["ols_rank_diagnostics"]:
        lines.append(f"- {m['pair'][0]}/{m['pair'][1]}: {m['rank_deficient_regressions']}/{m['regressions']} "
                     f"rank-deficient; condition number median {m['median_condition_number']:.1f}, "
                     f"max {m['max_condition_number']:.1f}; mean support size {m['mean_support_size_incl_intercept']:.1f}")
    lines += ["", "## Reachability mask density (mean over 3 seeds)",
              "Density = edges / D^2 (D=20, 400 entries per lag block). Self-edges (diagonal) count as "
              "legitimate lagged self-effects when present, matching the generator and mask conventions.", "",
              "| H | Lag | Density | Edges | Self-edges |", "|---:|---:|---:|---:|---:|"]
    for r in report["reachability_density"]:
        lines.append(f"| {r['horizon']} | {r['lag']} | {r['mean_density']:.3f} | {r['mean_edges']:.1f} | "
                     f"{r['mean_self_edges']:.1f} |")
    c40 = report["cap40_67_1009"]
    lines += ["", "## oracle_iterated, 40-epoch cap, 67/1009",
              f"Best epoch {c40['best_validation_epoch']}, stopped at epoch {c40['epochs_executed']}, "
              f"cap reached {c40['reached_epoch_cap']}; {c40['elapsed_seconds'] / 60:.1f} min.", "",
              "| Cap | H1 / DirectVAR [95% CI] | H2 / DirectVAR [95% CI] |", "|---|---|---|"]
    for label, key in (("20", "cap20_ratio_vs_var_direct"), ("40", "ratio_vs_var_direct")):
        c = c40[key]
        lines.append(f"| {label} | {c['H1']['ratio']:.4f} [{c['H1']['ci95'][0]:.4f}, {c['H1']['ci95'][1]:.4f}] | "
                     f"{c['H2']['ratio']:.4f} [{c['H2']['ci95'][0]:.4f}, {c['H2']['ci95'][1]:.4f}] |")
    lines.append(f"\nWall time {report['wall_seconds'] / 60:.1f} min.")
    (OUT / "s1_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
