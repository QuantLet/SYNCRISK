"""
Systemic Co-Movement (SCM) outcomes (Eq. 6) for SYNCRISK.

Eq. (6): Primary outcome — dominant eigenvalue of the realized correlation
matrix over forward horizon (t, t+h], normalized by N:

    SCM_{t,t+h} = lambda_1(Sigma_{t,t+h}) / N

Sigma is the realized correlation matrix of asset returns. We use h = 20
trading days as the main horizon (so weekly forward window ~4 weeks).

Also computed:
- Absorption ratio (Kritzman, Li, Page, Rigobon, 2011): share of total
  variance captured by the top K=3 PCs.
- Average pairwise correlation: simple alternative for transparency.

Inputs
------
returns_daily.parquet (from build_panel.py): long-format daily log returns
panel_weekly.parquet:                       Friday-anchored panel dates

For each panel date t, we form the forward correlation matrix of asset
returns over (t, t+h] business days using the same trading calendar as the
returns file.

Notes
-----
- Returns are forward-looking outcomes. This module computes them from the
  RAW returns_daily file, not from any model output.
- For SCM_{t-h,t} (the lagged outcome used as control in Eq. 7), we provide
  realized_correlation_backward() that uses (t-h, t] instead.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("syncrisk.outcomes.scm")


def _realized_correlation_matrix(
    returns_daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    min_overlap: int = 10,
    return_col: str = "log_return",
) -> tuple[np.ndarray, list[str]]:
    """Build the asset × asset correlation matrix over (start, end].

    Returns (corr_matrix, asset_list). corr_matrix may have NaNs if some
    asset pairs have fewer than min_overlap overlapping observations.
    """
    mask = (returns_daily["date"] > start) & (returns_daily["date"] <= end)
    window = returns_daily.loc[mask]
    if window.empty:
        return (np.array([]), [])

    wide = window.pivot_table(index="date", columns="asset_ticker", values=return_col)
    # Drop assets with too few obs
    counts = wide.notna().sum()
    keep = counts[counts >= min_overlap].index.tolist()
    if len(keep) < 5:
        return (np.array([]), keep)

    wide = wide[keep]
    # Pearson correlation, pairwise complete obs
    corr = wide.corr(method="pearson", min_periods=min_overlap).to_numpy()
    return (corr, keep)


def _safe_eigh(corr_matrix: np.ndarray) -> np.ndarray | None:
    """Symmetric eigvals of corr_matrix. Replace NaNs with 0 for stability."""
    if corr_matrix.size == 0:
        return None
    if np.isnan(corr_matrix).any():
        corr_matrix = np.where(np.isnan(corr_matrix), 0.0, corr_matrix)
    # Symmetrize against numerical noise
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    try:
        eigvals = np.linalg.eigvalsh(corr_matrix)
    except np.linalg.LinAlgError as e:
        log.warning("eigvalsh failed: %s", e)
        return None
    return np.sort(eigvals)[::-1]   # descending


def scm_one_window(
    returns_daily: pd.DataFrame,
    t: pd.Timestamp, h_days: int,
    *,
    direction: str = "forward",
    n_pcs_for_absorption: int = 3,
    min_overlap: int = 10,
) -> dict:
    """Compute SCM and related statistics for one window anchored at date t.

    direction = 'forward':  (t, t + h_days business days]
    direction = 'backward': (t - h_days business days, t]

    Returns dict with keys:
        scm (lambda_1 / N), absorption (top-K share), avg_pairwise_corr,
        n_assets, eigenvalues (numpy array, descending), window_start, window_end.
    """
    bday = pd.tseries.offsets.BDay()
    if direction == "forward":
        start = t
        end = t + bday * h_days
    elif direction == "backward":
        start = t - bday * h_days
        end = t
    else:
        raise ValueError(direction)

    corr, assets = _realized_correlation_matrix(
        returns_daily, start, end, min_overlap=min_overlap
    )
    if corr.size == 0:
        return {"scm": np.nan, "absorption": np.nan,
                "avg_pairwise_corr": np.nan, "n_assets": len(assets),
                "eigenvalues": np.array([]),
                "window_start": start, "window_end": end}

    N = corr.shape[0]
    eigs = _safe_eigh(corr)
    if eigs is None:
        return {"scm": np.nan, "absorption": np.nan,
                "avg_pairwise_corr": np.nan, "n_assets": N,
                "eigenvalues": np.array([]),
                "window_start": start, "window_end": end}

    scm = float(eigs[0] / N)
    K = min(n_pcs_for_absorption, len(eigs))
    absorption = float(eigs[:K].sum() / eigs.sum()) if eigs.sum() > 0 else np.nan

    # Average pairwise corr (exclude diagonal)
    upper = corr[np.triu_indices_from(corr, k=1)]
    avg_corr = float(np.nanmean(upper))

    return {
        "scm": scm,
        "absorption": absorption,
        "avg_pairwise_corr": avg_corr,
        "n_assets": N,
        "eigenvalues": eigs,
        "window_start": start,
        "window_end": end,
    }


def scm_panel(
    returns_daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    h_days: int = 20,
    *,
    direction: str = "forward",
    min_overlap: int = 10,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Compute SCM for each date in `dates`, with forward or backward window.

    `dates` is typically the Friday-anchored weekly panel dates.
    """
    rd = returns_daily.copy()
    rd["date"] = pd.to_datetime(rd["date"])
    rows = []
    for t in pd.to_datetime(dates):
        out = scm_one_window(rd, t, h_days, direction=direction,
                             min_overlap=min_overlap)
        # Also compute cross-sectional return dispersion at time t (for Eq. 7 control)
        if direction == "forward":
            disp_window = rd[(rd["date"] > t - pd.tseries.offsets.BDay() * 5) &
                              (rd["date"] <= t)]
        else:
            disp_window = rd[(rd["date"] > t - pd.tseries.offsets.BDay() * h_days) &
                              (rd["date"] <= t)]
        if not disp_window.empty:
            per_asset_std = (disp_window.groupby("asset_ticker")[return_col]
                             .std(ddof=0).dropna())
            disp = float(per_asset_std.std(ddof=0)) if len(per_asset_std) > 1 else np.nan
        else:
            disp = np.nan
        rows.append({
            "date": t,
            "h_days": h_days,
            "direction": direction,
            "scm": out["scm"],
            "absorption_ratio": out["absorption"],
            "avg_pairwise_corr": out["avg_pairwise_corr"],
            "cross_sectional_dispersion": disp,
            "n_assets_in_window": out["n_assets"],
            "window_start": out["window_start"],
            "window_end": out["window_end"],
        })
    return pd.DataFrame(rows)
