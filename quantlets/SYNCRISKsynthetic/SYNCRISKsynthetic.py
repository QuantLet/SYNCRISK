#!/usr/bin/env python3
"""
SYNCRISKsynthetic — full end-to-end pipeline on the SYNTHETIC panel (no licensed data).

Runs the complete SYNCRISK chain on the synthetic generator so the methodology can be
exercised with zero EODHD/FRED inputs:
  Eq.1 standardization -> Eq.2 Raw-AFSI -> Eq.3/4 cleaning + Excess-AFSI (frozen linear
  AND expanding-window C2x/C3x) -> Eq.5 Tail-AFSI -> permutation test ->
  Eq.6 SCM -> Eq.7 descriptive projection.

Confirms the qualitative signatures hold by construction: Excess-AFSI < Raw-AFSI
(cleaning removes shared drivers) and the permutation test rejects independence.

Self-contained: regenerates the synthetic panel via ../../data/synthetic with relative
paths (no published data needed); writes SYNCRISKsynthetic.csv alongside this script.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
from make_synthetic_panel import make_synthetic_panel
from syncrisk.indices import (standardize_expanding_per_asset, raw_afsi,
                              fit_cleaning_betas, apply_cleaning, excess_afsi,
                              fit_tail_thresholds, tail_afsi, permutation_test_panel)
from syncrisk.cleaning import expanding_residuals, C2_FEATURES
from syncrisk.outcomes import scm_panel
from syncrisk.families import afsi_series
from syncrisk.regression import newey_west


def main():
    d = make_synthetic_panel(n_assets=30, n_weeks=200, n_models=6,
                             correlation_strength=0.6, seed=42)
    panel, scores, rets, weeks = d["panel"], d["scores_long"], d["returns_daily"], d["weekly_dates"]
    z = standardize_expanding_per_asset(scores, min_history=36)

    raw = raw_afsi(z, info_set="prices_only")
    raw_mean = float(raw["raw_afsi"].mean())

    fits = fit_cleaning_betas(z, panel, cleaning_vars=["ret_5d", "rv_20d", "dd_60d", "tone_7d"],
                              train_start="2018-01-01", train_end="2021-12-31")
    ex_frozen = float(excess_afsi(apply_cleaning(z, panel, fits))["excess_afsi"].mean())

    resid_c2x = expanding_residuals(z.dropna(subset=["Z"]), panel, C2_FEATURES, "lin")
    ex_c2x = float(afsi_series(resid_c2x[["date", "asset_ticker", "model", "info_set", "u"]], "u")["all"].mean())
    resid_c3x = expanding_residuals(z.dropna(subset=["Z"]), panel, C2_FEATURES, "gbm")
    ex_c3x = float(afsi_series(resid_c3x[["date", "asset_ticker", "model", "info_set", "u"]], "u")["all"].mean())

    th = fit_tail_thresholds(z, train_start="2018-01-01", train_end="2021-12-31", quantile=0.90)
    tafsi = tail_afsi(z, th, q_values=(0.5, 0.67))
    perm = permutation_test_panel(z, value_col="Z", n_perms=200)
    reject = float((perm["p_value"] < 0.05).mean())

    scm = scm_panel(rets, weeks[:-5], h_days=20, direction="forward")
    exser = afsi_series(resid_c2x[["date", "asset_ticker", "model", "info_set", "u"]], "u")[["date", "all"]]
    j = exser.merge(scm[["date", "scm"]], on="date").dropna()
    m = newey_west(j["scm"].to_numpy(), j[["all"]].to_numpy(), maxlags=4)

    summary = {
        "raw_afsi_mean": round(raw_mean, 4),
        "excess_frozen_C2": round(ex_frozen, 4),
        "excess_C2x": round(ex_c2x, 4),
        "excess_C3x": round(ex_c3x, 4),
        "tail_afsi_q50_mean": round(float(tafsi["tail_afsi_q50"].mean()), 4),
        "tail_afsi_q67_mean": round(float(tafsi["tail_afsi_q67"].mean()), 4),
        "perm_reject_rate": round(reject, 3),
        "scm_mean": round(float(scm["scm"].dropna().mean()), 4),
        "eq7_beta_excess_h4": round(float(m.params[1]), 4),
        "eq7_n": int(m.nobs),
    }
    pd.DataFrame([summary]).to_csv(HERE / "SYNCRISKsynthetic.csv", index=False)
    for k, v in summary.items():
        print(f"  {k:24s} {v}")

    assert raw_mean > ex_c2x > 0, "expected Raw-AFSI > Excess-AFSI(C2x) > 0"
    assert reject > 0.5, "expected the permutation test to reject independence often"
    print("\nSanity checks PASS: Raw-AFSI > Excess-AFSI(C2x) > 0; permutation rejects.")
    print("wrote SYNCRISKsynthetic.csv")


if __name__ == "__main__":
    main()
