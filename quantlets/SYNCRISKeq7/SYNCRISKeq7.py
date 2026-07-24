#!/usr/bin/env python3
"""SYNCRISKeq7 - descriptive co-movement projection (Table 13), B-stabilized. All cells
insignificant (smallest p 0.26); no sign reversal."""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent; PUB = H  # input data bundled alongside this Quantlet
rows = json.load(open(PUB / "upg_Bstab_eq7.json"))
with open(H / "SYNCRISKeq7.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["arm", "cleaning", "horizon_w", "beta_excess", "nw_se", "t", "p"])
    for r in rows: w.writerow(r)
print("eq7 min p", min(r[6] for r in rows))
