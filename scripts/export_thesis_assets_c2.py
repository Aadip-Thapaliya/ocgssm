"""Export continuation-2 thesis tables and consistent-style figures."""

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import roadmap_final as rf

ROOT = Path("results/thesis_export_c2")
BASE = Path("results/roadmap")
S5 = BASE / "final/s5/heldout/summary.json"
S5B = BASE / "final/continuation/s5b/heldout/summary.json"
TASKA = BASE / "final/continuation2/task_a/pooled.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = ROOT / f"{name}.csv"
    columns = ["stage", "comparison", "horizon", "estimate", "ci95_low", "ci95_high",
               "status", "note"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio_rows(stage: str, ratios: dict[str, Any], status: str, note: str) -> list[dict[str, Any]]:
    rows = []
    for comparison, entries in ratios.items():
        if isinstance(entries, dict):
            entries = [entries]
        for i, entry in enumerate(entries):
            h = entry.get("horizon", i + 1)
            ci = entry.get("ci95", [None, None])
            estimate = entry.get("ratio", entry.get("difference"))
            rows.append({"stage": stage, "comparison": comparison, "horizon": h,
                         "estimate": estimate, "ci95_low": ci[0], "ci95_high": ci[1],
                         "status": status, "note": note})
    return rows


def stage_csvs() -> None:
    s1 = load(BASE / "s1/s1/summary.json")
    write_csv("S1", ratio_rows("S1", s1["ratios_vs_var_direct"], "PASS",
                               "Iterated head selected on validation."))
    s1b = load(BASE / "s1/s1b/summary.json")
    write_csv("S1b", ratio_rows("S1b", s1b["ratios_vs_var_direct"], "NO GAIN",
                                "Zero-initialized SSM residual on masked VAR backbone."))
    s3 = load(BASE / "s1/s3/summary.json")
    write_csv("S3", ratio_rows("S3", s3["ratios_vs_var_direct"], "PASS",
                               "Convergence/cap diagnostic; validation only."))
    diag = load(BASE / "s1/diagnostics/s1_diagnostics.json")
    rows = []
    for comparison, horizons in diag["comparisons"].items():
        for horizon, entry in horizons.items():
            rows.append({"stage": "S1-diagnostics", "comparison": comparison,
                         "horizon": horizon, "estimate": entry["ratio"],
                         "ci95_low": entry["ci95"][0], "ci95_high": entry["ci95"][1],
                         "status": diag["next_branch"], "note": diag["decision_rule"]})
    write_csv("S1_diagnostics", rows)

    s2 = load(BASE / "final/s2/summary.json")["attempts"]["None"]
    write_csv("S2", ratio_rows("S2", s2["ratios"], "PASS", "Learned soft mask; dev validation gate."))
    s2d = load(BASE / "final/s2d/summary.json")
    rows = []
    for config, item in s2d["configs"].items():
        rows += ratio_rows("S2d", item["ratios"], "SCARCITY SWEEP",
                           f"{config}; chosen S4-hard setting is {s2d['s4_hard']}.")
    write_csv("S2d", rows)

    s4 = load(BASE / "final/s4/summary.json")
    write_csv("S4", ratio_rows("S4", s4["configs"]["default"]["ratios"], "FAIL",
                               "Mixed-regime test slice; switch at t=3600 is inside test."))
    write_csv("S4-hard", ratio_rows("S4-hard", s4["configs"]["hard"]["ratios"],
                                    "COMPROMISED, DESCRIPTIVE ONLY",
                                    "The frozen mixed-regime test slice includes t=3600; do not treat as stationary evidence."))
    s4b = load(BASE / "final/continuation/s4b/summary.json")
    rows = ratio_rows("S4b", s4b["ratios"], "PASS",
                      "Stationary held-out confirmation; original S4 unchanged.")
    for variant, stats in s4b["stats"].items():
        rows.append({"stage": "S4b", "comparison": variant, "horizon": "cap_hits",
                     "estimate": stats["cap_hits"], "status": "TRAINING DIAGNOSTIC",
                     "note": f"best epochs={stats['best_epochs']}; epochs={stats['epochs']}"})
    write_csv("S4b", rows)
    s4bh = load(BASE / "final/continuation2/s4b_hard/summary.json")
    rows = ratio_rows("S4b-hard", s4bh["ratios"], "PASS",
                      "Stationary D20_T600; gate includes learned/dense point <0.9857.")
    for variant, stats in s4bh["stats"].items():
        rows.append({"stage": "S4b-hard", "comparison": variant, "horizon": "cap_hits",
                     "estimate": stats["cap_hits"], "status": "TRAINING DIAGNOSTIC",
                     "note": f"best epochs={stats['best_epochs']}; epochs={stats['epochs']}"})
    write_csv("S4b-hard", rows)

    s5 = load(S5)
    rows = []
    for variant, stats in s5["variants"].items():
        for metric, value in (("post_switch_first200_mean", stats["post_switch_mse_mean"]),
                              ("pre_switch_mean", stats["pre_switch_mse_mean"])):
            rows.append({"stage": "S5", "comparison": variant, "horizon": metric,
                         "estimate": value, "status": "GATE FAIL" if not s5["passed"] else "PASS",
                         "note": "Original S5 held-out summary; paired comparison CIs are in difference rows."})
    rows += ratio_rows("S5", s5["differences"], "GATE FAIL",
                       "Paired learned_det_on minus comparator, frozen first-200 gate.")
    write_csv("S5", rows)

    s5b = load(S5B)
    rows = []
    for window, table in s5b["tables"].items():
        for variant, mean in table["mean"].items():
            rows.append({"stage": "S5b", "comparison": variant, "horizon": window,
                         "estimate": mean, "status": "GATE FAIL" if not s5b["passed"] else "PASS",
                         "note": "learned_fast contrast CIs recorded below."})
        rows += ratio_rows("S5b", table["learned_fast_minus"], "GATE FAIL" if not s5b["passed"] else "PASS",
                           f"learned_fast minus comparator; {window}.")
    rows += ratio_rows("S5b", {"learned_fast-dense_online": s5b["gate_difference"]},
                       "GATE FAIL", "Frozen first-200 post-switch primary gate.")
    write_csv("S5b", rows)

    s6 = load(BASE / "final/s6/summary.json")
    rows = []
    for case, item in s6["results"].items():
        if "ratios" not in item:
            rows.append({"stage": "S6", "comparison": case, "horizon": "all",
                         "status": "INCOMPLETE", "note": "Original 2/3-seed case; not run in continuation 2."})
            continue
        rows += ratio_rows(f"S6 {case}", item["ratios"], "REAL-DATA CASE", case)
    write_csv("S6", rows)
    write_csv("S6b", [{"stage": "S6b", "comparison": "all cases", "horizon": "H24,H96",
                       "status": "NOT RUN", "note": "Follow-up on examined test sets; budget did not permit."}])

    pooled = load(TASKA)
    rows = []
    for label, item in pooled["ranges"].items():
        for variant, stats in item["means"].items():
            rows.append({"stage": "S5+S5b pooled", "comparison": variant, "horizon": label,
                         "estimate": stats["mean"], "ci95_low": stats["ci95"][0],
                         "ci95_high": stats["ci95"][1], "status": "POST HOC, DESCRIPTIVE",
                         "note": f"10 held-out seeds; n={item['n_windows_per_seed']} windows/seed."})
        rows += ratio_rows("S5+S5b pooled", item["learned_det_on_minus"],
                           "POST HOC, DESCRIPTIVE", f"learned_det_on minus comparator; {label}.")
    write_csv("S5_S5b_pooled", rows)
    sat = load(BASE / "final/continuation2/task_a/saturation_availability.json")
    write_csv("saturation_diagnostic", [{"stage": "Saturation", "comparison": "pre-update EMA snapshots",
                                        "horizon": "windows 0,50,100,200",
                                        "status": "UNAVAILABLE",
                                        "note": sat["status"]}])


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 160,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pooled_figure() -> None:
    pooled = load(TASKA)
    labels = ["First 200\nafter switch", "Later [200,495)", "Pre-switch"]
    keys = list(pooled["ranges"])
    variants = ["learned_det_on", "dense_online", "identity_online", "oracle_online"]
    colors = {"learned_det_on": "#0072B2", "dense_online": "#D55E00",
              "identity_online": "#009E73", "oracle_online": "#666666"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, key, title in zip(axes, keys, labels):
        table = pooled["ranges"][key]["means"]
        for variant in variants:
            m = table[variant]["mean"]
            low, high = table[variant]["ci95"]
            ax.errorbar(variant.replace("_online", "").replace("_", "\n"),
                        m, yerr=[[m - low], [high - m]], fmt="o", capsize=3,
                        color=colors[variant], markersize=6)
        ax.set_title(title)
        ax.set_ylabel("Per-window MSE (mean, 95% CI)")
        ax.tick_params(axis="x", labelrotation=0)
    fig.suptitle("Pooled S5 + S5b held-out forecasts (10 seeds; descriptive)", y=1.03)
    fig.tight_layout()
    save_both(fig, "pooled_switch_mse")


def stationary_figure() -> None:
    s4b = load(BASE / "final/continuation/s4b/summary.json")
    s4bh = load(BASE / "final/continuation2/s4b_hard/summary.json")
    comparisons = ["learned/dense", "learned/identity", "learned/random"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    markers = {"S4b": ("#0072B2", "o"), "S4b-hard": ("#D55E00", "s")}
    for y, (label, summary) in enumerate((("S4b", s4b), ("S4b-hard", s4bh))):
        color, marker = markers[label]
        for i, comparison in enumerate(comparisons):
            entry = summary["ratios"][comparison][0]
            x = entry["ratio"]
            low, high = entry["ci95"]
            ax.errorbar(x, y + (i - 1) * 0.16,
                        xerr=[[x - low], [high - x]], fmt=marker, capsize=3,
                        color=color, label=label if i == 0 else None)
            ax.text(x + 0.004, y + (i - 1) * 0.16, comparison.split("/")[1],
                    va="center", fontsize=8)
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--", label="no ratio benefit")
    ax.axvline(0.9857, color="#777777", linewidth=1, linestyle=":", label="S4b-hard point gate")
    ax.set_yticks([0, 1], ["S4b", "S4b-hard"])
    ax.set_xlabel("Learned / comparator H1 MSE ratio (95% paired bootstrap CI)")
    ax.set_title("Stationary held-out synthetic confirmations")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_both(fig, "stationary_s4b_ratios")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    stage_csvs()
    _style()
    pooled_figure()
    stationary_figure()
    (ROOT / "README.md").write_text(
        "# Thesis export, continuation 2\n\n"
        "CSV files provide one table per roadmap stage plus pooled S5/S5b and saturation availability. "
        "Figures use a shared matplotlib style and are exported as PNG and PDF. "
        "Saturation values are unavailable because per-window pre-update EMA adjacency snapshots "
        "were not persisted; no model rerun was used.\n", encoding="utf-8")
    print(f"Exported {len(list(ROOT.iterdir()))} assets to {ROOT}")


if __name__ == "__main__":
    main()
