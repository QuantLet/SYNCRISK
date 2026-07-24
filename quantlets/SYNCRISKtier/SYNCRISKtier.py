#!/usr/bin/env python3
"""SYNCRISKtier - exploratory tier extension (Appendix G), B-stabilized. Frontier scale
deepens synchronization (frontier-3 0.55 > efficiency 0.50; frontier-2 0.61); a single
reasoning pair (GPT-Gemini frontier) is the low outlier. Stress reload now in the B regime."""
import json, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
T = json.load(open(PUB / "phase3x9_Bstab_results.json")); TA = T["arms"]["P0"]["C3x"]; fp = T["frontier_pairwise"]["P0"]; sr = T["stress_r2_by_tier"]
sets = [("within-efficiency", "efficiency-3", "within_efficiency"), ("within-frontier", "frontier-3", "within_frontier"), ("within-open", "open-3", "within_open"),
        ("frontier-2 (claude+gpt)", "frontier-2", "frontier2"), ("lineage (same-provider cross-tier)", "lineage", "lineage_same_provider_cross_tier"), ("all-nine", "all-9", "all9")]
with open(H / "SYNCRISKtier.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["block", "model_set", "excess_afsi"])
    for b, ms, k in sets: w.writerow([b, ms, TA[k]["normal"]])
    for p in fp: w.writerow([f"pair {p}", "frontier pair", fp[p]])
with open(H / "SYNCRISKtier_stress_r2.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["tier", "model_set", "r2_normal", "r2_stress"])
    for t in sr: w.writerow([t, "P3", sr[t]["normal"], sr[t]["stress"]])
fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(range(len(sets)), [TA[k]["normal"] for _, _, k in sets], color="#a8743a")
ax.set_xticks(range(len(sets))); ax.set_xticklabels([b for b, _, _ in sets], rotation=25, ha="right", fontsize=7); ax.set_ylabel("Excess-AFSI (C3x, P0, normal)"); ax.grid(alpha=.3, axis="y"); fig.savefig(H / "SYNCRISKtier.png", bbox_inches="tight")
print("tier written; eff reload", sr["efficiency"]["stress"])
