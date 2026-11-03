"""Final cross-stage figure: learned/dense and learned/identity H1 ratios with 95% CIs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/roadmap/final")


def main() -> None:
    rows = []
    s2 = json.loads((ROOT / "s2" / "summary.json").read_text(encoding="utf-8"))
    att = s2["attempts"][0] if isinstance(s2["attempts"], list) else next(iter(s2["attempts"].values()))
    for key in ("learned/dense", "learned/identity"):
        rows.append(("S2 dev (val)", key, att["ratios"][key.replace("learned", "learned_soft")][0]))
    s4 = json.loads((ROOT / "s4" / "summary.json").read_text(encoding="utf-8"))
    for cfg, label in (("default", "S4 held-out"), ("hard", "S4-hard held-out")):
        for key in ("learned/dense", "learned/identity"):
            rows.append((label, key, s4["configs"][cfg]["ratios"][key][0]))
    s6 = json.loads((ROOT / "s6" / "summary.json").read_text(encoding="utf-8"))
    for case, res in s6["results"].items():
        if "ratios" not in res:
            continue
        for key in ("learned/dense", "learned/identity"):
            rows.append((f"S6 {case}", key, res["ratios"][f"{key} mse"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = sorted({r[0] for r in rows}, key=[r[0] for r in rows].index)
    for k, key in enumerate(("learned/dense", "learned/identity")):
        sel = [r for r in rows if r[1] == key]
        x = [labels.index(r[0]) + (k - 0.5) * 0.25 for r in sel]
        y = [r[2]["ratio"] for r in sel]
        err = [[v - r[2]["ci95"][0] for v, r in zip(y, sel)], [r[2]["ci95"][1] - v for v, r in zip(y, sel)]]
        ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, label=key)
    ax.axhline(1, color="black", lw=0.6)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("MSE ratio (H1 / H24 / H96), 95% CI")
    ax.set_title("Learned graph vs dense and identity across stages")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "final_ratios.png", dpi=120)
    print(json.dumps([(r[0], r[1], round(r[2]["ratio"], 4)) for r in rows]))


if __name__ == "__main__":
    main()
