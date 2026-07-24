#!/usr/bin/env python3
"""SYNCRISKfamily - family-block decomposition, B-stabilized. Blocks statistically equal
(contrasts cover zero); cross-family of the same order as within-family."""
import json, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
A = json.load(open(PUB / "upg_Bstab_results.json"))["arms"]; T = json.load(open(PUB / "phase3x9_Bstab_results.json"))["arms"]
rows = [("all-6", "all"), ("within-commercial", "within_comm"), ("within-open", "within_open"), ("cross-family", "cross_family")]
with open(H / "SYNCRISKfamily.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["block", "P0_normal", "P0_stress", "P3_normal", "P3_stress"])
    for b, g in rows: w.writerow([b, A["P0"]["C3x"][g]["normal"], A["P0"]["C3x"][g]["stress"], A["P3"]["C3x"][g]["normal"], A["P3"]["C3x"][g]["stress"]])
PS = [("all-9", "all9"), ("within-efficiency", "within_efficiency"), ("within-frontier", "within_frontier"), ("within-open", "within_open"), ("frontier-2 (claude+gpt)", "frontier2"), ("lineage (same-provider cross-tier)", "lineage_same_provider_cross_tier")]
with open(H / "SYNCRISKfamily_tier.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["pair_set", "P0_normal", "P0_stress", "P3_normal", "P3_stress"])
    for lab, k in PS: w.writerow([lab, T["P0"]["C3x"][k]["normal"], T["P0"]["C3x"][k]["stress"], T["P3"]["C3x"][k]["normal"], T["P3"]["C3x"][k]["stress"]])
fig, ax = plt.subplots(figsize=(6, 4)); xs = range(len(rows))
ax.bar([x - 0.2 for x in xs], [A["P3"]["C3x"][g]["normal"] for _, g in rows], 0.4, label="P3 normal")
ax.bar([x + 0.2 for x in xs], [A["P0"]["C3x"][g]["normal"] for _, g in rows], 0.4, label="P0 normal")
ax.set_xticks(list(xs)); ax.set_xticklabels([b for b, _ in rows], rotation=20, ha="right", fontsize=7); ax.set_ylabel("Excess-AFSI (C3x, normal)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); ax.grid(alpha=.3, axis="y"); fig.savefig(H / "SYNCRISKfamily.png", bbox_inches="tight")
print("family written")
