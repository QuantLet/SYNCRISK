#!/usr/bin/env python3
"""SYNCRISKstress - stress premium + reload, B-stabilized. No positive premium survives
(point negative on 10 stress weeks, not interpreted directionally); milder OOS reload
(C3x 0.07-0.18) keeps stress reported only as an upper bound."""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
B = json.load(open(PUB / "upg_Bstab_results.json")); pc = B["premium_ci"]; oc = B["ortho"]["C3x"]
with open(H / "SYNCRISKstress.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["arm", "cleaning", "premium", "ci_lo", "ci_hi"])
    for k in ["P0_C2x", "P0_C3x", "P3_C2x", "P3_C3x"]: w.writerow([k.split("_")[0], k.split("_")[1]] + pc[k])
with open(H / "SYNCRISKstress_reload.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["model", "C3x_R2_normal", "C3x_R2_stress"])
    for m in oc["normal"]: w.writerow([m, round(oc["normal"][m], 4), round(oc["stress"][m], 4)])
print("stress premium P3/C3x", pc["P3_C3x"])
