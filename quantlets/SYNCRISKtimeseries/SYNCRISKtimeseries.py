#!/usr/bin/env python3
"""SYNCRISKtimeseries - per-week Excess-AFSI (C3x, P3, all-6), B-stabilized (OOS from mid-2022).
Recomputed from the published B residuals; stress weeks at the 90th-pct VIX rule."""
import sys, csv
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})
H = Path(__file__).resolve().parent; PUB = H  # input data + syncrisk lib bundled alongside this Quantlet
sys.path.insert(0, str(H))
from syncrisk.families import afsi_series
r = pd.read_parquet(PUB / "residuals_Bstab_P3_C3x.parquet"); r["date"] = pd.to_datetime(r["date"])
ts = afsi_series(r[["date", "asset_ticker", "model", "info_set", "u"]], "u")[["date", "all"]]
ts.to_csv(H / "SYNCRISKtimeseries.csv", index=False)
vix = pd.read_csv(PUB / "main_vix.csv"); vix["date"] = pd.to_datetime(vix["date"]); vix["stress"] = vix["VIX"] >= vix["VIX"].quantile(0.90)
fig, ax = plt.subplots(figsize=(10, 4)); ts["date"] = pd.to_datetime(ts["date"]); ax.plot(ts["date"], ts["all"], lw=1, color="navy", label="Excess-AFSI (C3x, P3, all six)")
for _, x in vix[vix["stress"]].iterrows():
    if ts["date"].min() <= x["date"] <= ts["date"].max(): ax.axvspan(x["date"] - pd.Timedelta(days=3), x["date"] + pd.Timedelta(days=3), color="red", alpha=.12)
ax.set_ylabel("Excess-AFSI (C3x, P3)"); ax.set_xlabel("week"); ax.legend(handles=[ax.get_lines()[0], Patch(facecolor="red", alpha=0.12, label="stress weeks")], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); ax.grid(alpha=.3); fig.savefig(H / "SYNCRISKtimeseries.png", bbox_inches="tight")
print("timeseries OOS", ts["date"].min().date(), "->", ts["date"].max().date())
