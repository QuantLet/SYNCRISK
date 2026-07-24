#!/usr/bin/env python3
"""SYNCRISKcore - 13-asset high-coverage core robustness (Table F.17), B-stabilized."""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
B = json.load(open(PUB / "upg_Bstab_results.json")); A, co = B["arms"], B["core"]
with open(H / "SYNCRISKcore.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["cleaning", "arm", "full_normal", "full_stress", "core_normal", "core_stress"])
    for cl in ["C2x", "C3x"]:
        for ar in ["P0", "P3"]: w.writerow([cl, ar, A[ar][cl]["all"]["normal"], A[ar][cl]["all"]["stress"], co[f"{ar}_{cl}"]["normal"], co[f"{ar}_{cl}"]["stress"]])
print("core written")
