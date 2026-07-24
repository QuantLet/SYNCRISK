#!/usr/bin/env python3
"""
SYNCRISKpanel — realized-panel construction and coverage report (publishable data).

Summarizes the realized SYNCRISK panel (30 US equities x 244 weeks, 2021-02 -> 2026-06)
and its coverage:
  - weekly usable cross-section N over time (with the N>=30 inclusion rule),
  - per-asset training/overall coverage and the Amendment-4 keep flags,
  - the realized-panel metadata (stress weeks, train window, year distribution).

Uses ONLY publishable derived data (no raw EODHD prices/news). Self-contained: reads
../../data/published with relative paths; writes SYNCRISKpanel.csv/.png alongside.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.transparent": True, "savefig.bbox": "tight", "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})  # transparent + legend-below sweep
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUB = HERE


def main():
    info = json.load(open(PUB / "realized_panel.json"))
    panel = pd.read_parquet(PUB / "panel_features.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    cov = pd.read_csv(PUB / "training_coverage_per_asset_v2.csv")

    # weekly cross-section size from the (publishable) feature panel
    wn = panel.groupby("date")["asset_ticker"].nunique().rename("N_assets").reset_index()
    wn.to_csv(HERE / "SYNCRISKpanel.csv", index=False)

    print(f"Realized panel: {info['n_assets']} assets x {info['n_weeks']} weeks "
          f"({info['week_first']} -> {info['week_last']})")
    print(f"Train window {info['train_window']}, train weeks {info['n_train_weeks']}, "
          f"OOS weeks {info['n_oos_weeks']}, stress weeks {info['n_stress_in_panel']}")
    print(f"Year distribution: {info['year_distribution']}")
    print(f"Kept assets (Amendment-4 rule): {int(cov['keep'].sum())} / {len(cov)}")
    print(f"Weekly N: min={wn['N_assets'].min()}, median={int(wn['N_assets'].median())}, "
          f"max={wn['N_assets'].max()}")

    # Top: weekly cross-section over time. Bottom: per-asset coverage for all the
    # candidate names, split into two readable columns (~half the assets each).
    kept = cov.sort_values("pct_overall").reset_index(drop=True)
    n = len(kept); half = (n + 1) // 2
    xmax = max(100.0, float(kept["pct_overall"].max()))
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.6], hspace=0.30, wspace=0.55)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(wn["date"], wn["N_assets"], lw=1.2, color="navy")
    ax_ts.axhline(30, color="red", ls="--", lw=.8, label="N>=30 inclusion rule")
    ax_ts.set_title("Weekly usable cross-section"); ax_ts.set_ylabel("N assets")
    ax_ts.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False); ax_ts.grid(alpha=.3)
    for col, (lo, hi) in enumerate([(0, half), (half, n)]):
        ax = fig.add_subplot(gs[1, col]); sub = kept.iloc[lo:hi]
        ax.barh(sub["asset"], sub["pct_overall"],
                color=["seagreen" if k else "lightgray" for k in sub["keep"]])
        ax.set_xlim(0, xmax); ax.margins(y=0.01)
        ax.tick_params(labelsize=8); ax.set_xlabel("% usable weeks overall")
        ax.grid(alpha=.3, axis="x")
        if col == 0:
            ax.set_title("Per-asset overall coverage (green = kept under the Amendment-4 rule)",
                         loc="left", fontsize=11)
    fig.savefig(HERE / "SYNCRISKpanel.png", dpi=150, bbox_inches="tight")
    print("wrote SYNCRISKpanel.csv / .png")


if __name__ == "__main__":
    main()