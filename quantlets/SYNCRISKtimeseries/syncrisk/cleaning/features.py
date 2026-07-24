"""Cleaning-regression feature sets (Eq. 3 observables).

These are the asset-level risk drivers regressed out of the standardized scores
to form residuals u (Eq. 3). Market-wide W_t variables are excluded from cleaning
(they are scalar in t and absorbed by the per-date intercept alpha_{k,t}).

Specs:
  C1   : minimal linear cleaning (4 drivers)
  C2   : full linear cleaning (11 drivers)
  C3   : full non-linear cleaning (LightGBM on the C2 drivers)
  C2x  : C2 drivers, expanding-window OOS fit (MAIN spec, Amendment 6)
  C3x  : C2 drivers, expanding-window OOS LightGBM fit (MAIN spec, Amendment 6)
"""
from __future__ import annotations

C1_FEATURES = ["ret_5d", "rv_20d", "dd_60d", "tone_7d"]

C2_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "rv_20d", "rv_60d", "dd_60d",
    "min_ret_20d", "beta_60d", "rv_zscore", "tone_7d", "tone_delta",
]
