#!/usr/bin/env python3
"""SYNCRISKoverlap - the headline as a curve, not a point (paper Figure, Section 5.7).

The matched-overlap no-model-channel benchmark measures how much residual agreement
shared input alone produces at each designed news overlap J (Table 12; both anchors
pinned: estimator floor 0.083 at J=0, same-channel pole 0.561 at J=1). This QuantLet
reads that published sweep, maps it to an effective diversity
K_eff(J) = 6 / [1 + 5 * rho_null(J)], and marks the empirical LLM panel
(J=0.57, rho=0.36, K_eff=2.1) on the curve: it sits just above the pure-shared-input
null at the same overlap (K_eff ~ 1.9), so at the realized overlap the
effective-diversity deficit is essentially what input overlap alone produces. Open
markers are analytic extrapolations to 50%/90% subsamples via J=f/(2-f) and the
induced news correlation 2J/(1+J)=f -- NOT empirical measurements.

The matched-overlap sweep is generated from the primary run by
revision/p1_repin_bothends.py (SEED=0, expanding-window C3x, two-way bootstrap); its
published values ship here as data/published/matched_overlap_sweep.csv. Deterministic;
no licensed data.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.transparent": True, "savefig.bbox": "tight",
                     "figure.facecolor": "none", "axes.facecolor": "none",
                     "legend.frameon": False})

MAIN_BLUE = "#003DA5"
IDA_RED = "#C8102E"
K = 6
keff = lambda rho: K / (1.0 + (K - 1) * rho)

H = Path(__file__).resolve().parent
PUB = H

# --- read the published matched-overlap sweep rho_null(J) ---
J, rho = [], []
for r in csv.DictReader(open(PUB / "matched_overlap_sweep.csv")):
    J.append(float(r["jaccard"])); rho.append(float(r["rho_null_c3x"]))
J = np.array(J); rho = np.array(rho)

RHO_EMP = 0.36          # empirical P3/C3x headline at the realized overlap
J_EMP = 0.57            # realized news Jaccard (P3, 70% subsample)

# --- write the K_eff(J) curve + empirical/extrapolated points ---
with open(H / "SYNCRISKoverlap.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["kind", "J", "rho", "K_eff"])
    for j, rn in zip(J, rho):
        w.writerow(["null_sweep", round(float(j), 2), rn, round(keff(rn), 3)])
    w.writerow(["empirical", J_EMP, RHO_EMP, round(keff(RHO_EMP), 3)])
    for fsub in [0.50, 0.90]:                       # analytic subsample extrapolation
        jf = fsub / (2 - fsub); rf = float(np.interp(jf, J, rho))
        w.writerow([f"analytic_{int(fsub*100)}pct", round(jf, 3), round(rf, 3), round(keff(rf), 2)])

# --- figure ---
Jg = np.linspace(0, 1, 200); rho_g = np.interp(Jg, J, rho)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(Jg, keff(rho_g), color=MAIN_BLUE, lw=2.3, zorder=3,
        label=r"shared-input null $K_{\mathrm{eff}}(J)=6/[1+5\,\rho_{\mathrm{null}}(J)]$")
ax.plot(J, keff(rho), "o", color=MAIN_BLUE, ms=4, zorder=4)
ax.axhline(K, ls="--", color="#888888", lw=1.2, label="nominal headcount (6)")
ax.axvline(J_EMP, ls=":", color="#999999", lw=1.0, zorder=1)
ax.plot([J_EMP], [keff(RHO_EMP)], "s", color=IDA_RED, ms=9, zorder=5,
        label=r"empirical LLM panel ($J{=}0.57$, $\rho{=}0.36$)")
ax.annotate("empirical panel:\n" + r"$K_{\mathrm{eff}}=2.1$ at $J=0.57$",
            xy=(J_EMP, keff(RHO_EMP)), xytext=(0.24, 3.55), fontsize=8.5, color=IDA_RED,
            ha="left", arrowprops=dict(arrowstyle="-", color=IDA_RED, lw=0.8))
ax.annotate(r"pure shared input at $J=0.57$: $K_{\mathrm{eff}}\approx1.9$",
            xy=(J_EMP, keff(0.433)), xytext=(0.05, 1.45), fontsize=8.0, color=MAIN_BLUE,
            arrowprops=dict(arrowstyle="-", color=MAIN_BLUE, lw=0.7))
for fsub in [0.50, 0.90]:
    jf = fsub / (2 - fsub); rf = float(np.interp(jf, J, rho))
    ax.plot([jf], [keff(rf)], "o", mfc="white", mec="#555555", ms=7, zorder=4)
    ax.annotate(f"{int(fsub*100)}%", xy=(jf, keff(rf)), xytext=(jf - 0.02, keff(rf) + 0.28),
                fontsize=7.5, color="#555555")
ax.plot([], [], "o", mfc="white", mec="#555555", ms=7, label="analytic 50%/90% subsample (not empirical)")
ax.set_xlabel(r"designed news-input overlap (Jaccard $J$)")
ax.set_ylabel(r"effective independent validators $K_{\mathrm{eff}}$")
ax.set_xlim(0, 1); ax.set_ylim(0, 6.4); ax.set_yticks(range(0, 7)); ax.grid(alpha=0.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.6, frameon=False)
fig.tight_layout()
fig.savefig(H / "SYNCRISKoverlap.png", dpi=140, bbox_inches="tight")
print("K_eff(rho=0.36)=%.3f ; null K_eff@J=0.57=%.3f ; wrote SYNCRISKoverlap.{csv,png}"
      % (keff(RHO_EMP), keff(0.433)))
