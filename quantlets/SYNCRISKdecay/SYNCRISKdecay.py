#!/usr/bin/env python3
"""SYNCRISKdecay - decay chain (Figure 1), B-stabilized (Eq.1 standardization, burn-in 36).
All-6 Raw -> C2x -> C3x, arms P0/P3 (0.69 -> 0.36 chain), and the 9-model per-tier chains.
Reproduces from the published B results."""
import json, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
B = json.load(open(PUB / "upg_Bstab_results.json")); T = json.load(open(PUB / "phase3x9_Bstab_results.json"))
A, R = B["arms"], B["raw"]
with open(H / "SYNCRISKdecay.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["arm", "raw", "C2x", "C3x"])
    for a in ["P0", "P3"]: w.writerow([a, R[a]["all"]["normal"], A[a]["C2x"]["all"]["normal"], A[a]["C3x"]["all"]["normal"]])
TI = {"all9": "all-9", "within_efficiency": "efficiency", "within_frontier": "frontier", "within_open": "open"}
MS = {"P0": "P0=9 models", "P3": "P3=8 models (frontier=claude+gpt)"}
with open(H / "SYNCRISKdecay_tier.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["arm", "model_set", "tier", "raw", "C2x", "C3x"])
    for a in ["P0", "P3"]:
        for t in TI: w.writerow([a, MS[a], TI[t], T["arms"][a]["RAW"][t]["normal"], T["arms"][a]["C2x"][t]["normal"], T["arms"][a]["C3x"][t]["normal"]])
fig, ax = plt.subplots(figsize=(6, 4))
for a, c in [("P0", "#3b5b92"), ("P3", "#a8743a")]:
    ax.plot(["Raw", "C2x", "C3x"], [R[a]["all"]["normal"], A[a]["C2x"]["all"]["normal"], A[a]["C3x"]["all"]["normal"]], marker="o", color=c, label=a)
ax.set_ylabel("Excess-AFSI (all-6, normal)"); ax.set_xlabel("cleaning rung"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); ax.grid(alpha=.3); fig.savefig(H / "SYNCRISKdecay.png", bbox_inches="tight")
print("decay: P3 chain", R["P3"]["all"]["normal"], "->", A["P3"]["C3x"]["all"]["normal"])
