#!/usr/bin/env python3
"""SYNCRISKladder - cleaning-ladder levels and orthogonality (P0, all-6, normal), B-stabilized.
The linear C2x rung leaves per-week heteroskedasticity (max R2 ~0.08); the boosted C3x rung
clears the 0.05 ceiling (max 0.036), which is why C3x is the headline rung."""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
B = json.load(open(PUB / "upg_Bstab_results.json")); A, R, om = B["arms"], B["raw"], B["ortho"]
out = [["Raw", R["P0"]["all"]["normal"], ""], ["C1x", A["P0"]["C1x"]["all"]["normal"], ""],
       ["C2x", A["P0"]["C2x"]["all"]["normal"], round(max(om["C2x"]["normal"].values()), 4)],
       ["C3x", A["P0"]["C3x"]["all"]["normal"], round(max(om["C3x"]["normal"].values()), 4)]]
with open(H / "SYNCRISKladder.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["spec", "excess_normal_all6", "max_resid_R2_normal"]); [w.writerow(r) for r in out]
print("ladder written")
