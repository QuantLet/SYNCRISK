"""
Cleaning regression (Eq. 3) and Excess-AFSI (Eq. 4) for SYNCRISK.

Eq. (3):
    Z_{i,t}^{(k)} = alpha_{k,t} + beta_k' * X_tilde_{i,t} + u_{i,t}^{(k)}

where X_tilde_{i,t} = X_{i,t} - mean_i(X_{i,t}) is the cross-sectionally
demeaned vector of asset-level risk drivers (lagged return, 20-day RV,
drawdown, asset-level news tone). Market-wide variables W_t are EXCLUDED from
cleaning (Note 1 in the paper: they are scalar in t and would be absorbed by
alpha_{k,t}).

OOS protocol (from concept note v0.7):
- beta_k estimated ONCE on the training window (2018-2021) per model.
- alpha_{k,t} recomputed cross-sectionally for each (k, t) in the OOS window.
- Holding beta_k fixed OOS prevents leakage of post-2022 dependencies into the
  cleaning step.

Eq. (4):
    ExcessAFSI_t = (2 / (K*(K-1))) * sum_{k<l} corr_i(u_{i,t}^{(k)}, u_{i,t}^{(l)})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .standardization import _pairwise_mean_corr

log = logging.getLogger("syncrisk.indices.excess_afsi")


DEFAULT_CLEANING_VARS = [
    "ret_5d",         # lagged 5-day return
    "rv_20d",         # 20-day realized volatility
    "dd_60d",         # 60-day drawdown
    "tone_7d",        # asset-level news tone over past 7 days
]


# ---------------------------------------------------------------------------
# Cross-sectional demeaning
# ---------------------------------------------------------------------------

def cross_sectional_demean(
    df: pd.DataFrame,
    cleaning_vars: list[str],
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute X_tilde_{i,t} = X_{i,t} - mean_i(X_{i,t}) for each cleaning var.

    Adds columns named "<var>_tilde" to df. Missing values within a date pass
    through; the date-mean is computed over non-NaN observations.
    """
    df = df.copy()
    for v in cleaning_vars:
        if v not in df.columns:
            log.warning("cleaning var '%s' not in dataframe; skipping", v)
            continue
        mean_by_date = df.groupby(date_col)[v].transform("mean")
        df[f"{v}_tilde"] = df[v] - mean_by_date
    return df


# ---------------------------------------------------------------------------
# Beta fit (training window)
# ---------------------------------------------------------------------------

@dataclass
class ModelCleaningFit:
    """Fitted cleaning regression for one model.

    beta is the vector of slope coefficients for the cleaning variables.
    Note: no global intercept is fit, because alpha_{k,t} (the date FE) is
    absorbed by within-date demeaning of both Z and X_tilde at evaluation time.
    """
    model_name: str
    cleaning_vars: list[str]
    beta: np.ndarray              # shape (p,)
    n_train_obs: int
    train_date_range: tuple


def _within_date_demean(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Subtract per-date mean. Equivalent to absorbing a date fixed effect."""
    return values - values.groupby(dates).transform("mean")


def fit_cleaning_betas(
    scores_z: pd.DataFrame,
    panel_features: pd.DataFrame,
    *,
    cleaning_vars: list[str] = DEFAULT_CLEANING_VARS,
    train_start: str,
    train_end: str,
    info_set: str = "prices_only",
    z_col: str = "Z",
) -> dict[str, ModelCleaningFit]:
    """Fit beta_k for each model on the training window.

    Method: within-date demean Z and each X_tilde (to absorb alpha_{k,t}),
    then OLS on the demeaned panel within model k.

    Returns dict {model_name: ModelCleaningFit}.
    """
    sub = scores_z[scores_z["info_set"] == info_set].copy()
    sub["date"] = pd.to_datetime(sub["date"])

    panel = panel_features.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    merge_cols = ["date", "asset_ticker"]
    merged = sub.merge(panel[merge_cols + cleaning_vars], on=merge_cols, how="left")

    # Cross-sectional demean X
    merged = cross_sectional_demean(merged, cleaning_vars)
    tilde_cols = [f"{v}_tilde" for v in cleaning_vars if f"{v}_tilde" in merged.columns]

    # Restrict to training window
    mask = (merged["date"] >= train_start) & (merged["date"] <= train_end)
    train = merged.loc[mask].dropna(subset=[z_col] + tilde_cols).copy()
    log.info("cleaning fit: training obs = %d (%s → %s)",
             len(train), train_start, train_end)

    fits: dict[str, ModelCleaningFit] = {}
    for model_name, g in train.groupby("model"):
        if len(g) < 100:
            log.warning("model %s: only %d training obs; skipping cleaning fit",
                        model_name, len(g))
            continue

        # Absorb date FE by within-date demeaning of both LHS and RHS
        z_demean = _within_date_demean(g[z_col], g["date"]).to_numpy()
        X_demean = np.column_stack([
            _within_date_demean(g[c], g["date"]).to_numpy() for c in tilde_cols
        ])

        # OLS (no intercept, both sides demeaned)
        try:
            beta, *_ = np.linalg.lstsq(X_demean, z_demean, rcond=None)
        except np.linalg.LinAlgError as e:
            log.error("model %s: lstsq failed: %s", model_name, e)
            continue

        fits[model_name] = ModelCleaningFit(
            model_name=model_name,
            cleaning_vars=cleaning_vars,
            beta=beta,
            n_train_obs=len(g),
            train_date_range=(str(g["date"].min().date()), str(g["date"].max().date())),
        )
        log.info("model %s: beta = %s (n_train=%d)",
                 model_name,
                 dict(zip(cleaning_vars, np.round(beta, 4).tolist())),
                 len(g))

    return fits


# ---------------------------------------------------------------------------
# OOS residualization
# ---------------------------------------------------------------------------

def apply_cleaning(
    scores_z: pd.DataFrame,
    panel_features: pd.DataFrame,
    fits: dict[str, ModelCleaningFit],
    *,
    info_set: str = "prices_only",
    z_col: str = "Z",
) -> pd.DataFrame:
    """Compute residuals u_{i,t}^{(k)} for the full sample using fitted beta_k.

    alpha_{k,t} is re-estimated cross-sectionally per (k, t) (this is just the
    within-(k, t) mean of Z - X_tilde @ beta).
    """
    sub = scores_z[scores_z["info_set"] == info_set].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    panel = panel_features.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    cleaning_vars = next(iter(fits.values())).cleaning_vars
    merge_cols = ["date", "asset_ticker"]
    merged = sub.merge(panel[merge_cols + cleaning_vars], on=merge_cols, how="left")
    merged = cross_sectional_demean(merged, cleaning_vars)
    tilde_cols = [f"{v}_tilde" for v in cleaning_vars]

    merged["u"] = np.nan
    for model_name, fit in fits.items():
        m = merged["model"] == model_name
        sub_m = merged[m]
        # Predicted: X_tilde @ beta
        valid = sub_m[tilde_cols].notna().all(axis=1) & sub_m[z_col].notna()
        X = sub_m.loc[valid, tilde_cols].to_numpy()
        pred = X @ fit.beta
        z_vals = sub_m.loc[valid, z_col].to_numpy()
        raw_resid = z_vals - pred
        # Absorb alpha_{k,t}: within (k, t) demean
        tmp = sub_m.loc[valid, ["date"]].copy()
        tmp["raw_resid"] = raw_resid
        date_means = tmp.groupby("date")["raw_resid"].transform("mean")
        u = raw_resid - date_means.to_numpy()
        merged.loc[sub_m.index[valid], "u"] = u

    return merged


# ---------------------------------------------------------------------------
# Excess-AFSI (Eq. 4)
# ---------------------------------------------------------------------------

def excess_afsi(
    residuals: pd.DataFrame,
    *,
    u_col: str = "u",
) -> pd.DataFrame:
    """Compute Eq. (4): Excess-AFSI per date.

    `residuals` must contain columns date, asset_ticker, model, u, info_set.
    """
    df = residuals.dropna(subset=[u_col]).copy()
    rows = []
    for (date, iset), g in df.groupby(["date", "info_set"]):
        wide = g.pivot_table(index="asset_ticker", columns="model", values=u_col)
        afsi_val = _pairwise_mean_corr(wide.to_numpy())
        rows.append({
            "date": date,
            "info_set": iset,
            "excess_afsi": afsi_val,
            "n_assets": int(wide.shape[0]),
            "n_models": int(wide.shape[1]),
        })
    return pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)
