"""
Standardization (Eq. 1) and Raw-AFSI (Eq. 2) for SYNCRISK.

Eq. (1), production: per-(model, asset) EXPANDING-window standardization, using only
weeks strictly prior to t (no contemporaneous information), with a burn-in of at least
``min_history`` weeks. This is the defensible, strictly out-of-sample estimator behind
the B-stabilized results in the paper -> ``standardize_expanding_per_asset``.
    Z_{i,t}^{(k)} = (R_{i,t}^{(k)} - mu_hat_{k,i,<t}) / sigma_hat_{k,i,<t}

Superseded scheme: contemporaneous within-(model, date) cross-sectional standardization
(``standardize_within_model_date``). It uses the same-week cross-section and is therefore
in-sample; it is retained only for the frozen-window / pilot reproductions and is NOT the
production estimator.
    Z_{i,t}^{(k)} = (R_{i,t}^{(k)} - mu_t^{(k)}) / sigma_t^{(k)}

Eq. (2): Raw-AFSI as average pairwise cross-sectional correlation across model
pairs, computed per date.
    RawAFSI_t = (2 / (K*(K-1))) * sum_{k<l} corr_i(Z_{i,t}^{(k)}, Z_{i,t}^{(l)})

Input
-----
Long-format scores DataFrame with columns:
    date            datetime
    asset_ticker    str
    model           str   (e.g. "claude", "gpt", "gemini")
    info_set        str   ("prices_only" or "prices_plus_news")
    raw_score       float (e.g. relative_risk_score in 0-100)

Output
------
- standardized scores: same shape + 'Z' column
- raw_afsi: one row per (date, info_set) with raw_afsi value and metadata
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("syncrisk.indices.standardization")


def standardize_within_model_date(
    scores: pd.DataFrame,
    *,
    raw_score_col: str = "raw_score",
    group_cols: tuple = ("date", "model", "info_set"),
    min_assets: int = 5,
) -> pd.DataFrame:
    """Apply Eq. (1): z-score within each (date, model, info_set) cross-section.

    If the within-group std is zero or fewer than min_assets observations are
    present, Z is set to NaN for that group (degenerate cross-section).
    """
    df = scores.copy()
    df["Z"] = np.nan

    grp = df.groupby(list(group_cols), dropna=False)
    for key, g in grp:
        x = g[raw_score_col].astype(float)
        n_valid = x.notna().sum()
        if n_valid < min_assets:
            continue
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - x.mean()) / sd
        df.loc[g.index, "Z"] = z

    n_groups_total = df.groupby(list(group_cols), dropna=False).ngroups
    n_groups_with_z = df.dropna(subset=["Z"]).groupby(list(group_cols), dropna=False).ngroups
    log.info("standardization: %d / %d groups have valid Z (%.1f%%)",
             n_groups_with_z, n_groups_total,
             100 * n_groups_with_z / max(1, n_groups_total))
    return df


def standardize_expanding_per_asset(
    scores: pd.DataFrame,
    *,
    raw_score_col: str = "raw_score",
    group_cols: tuple = ("model", "asset_ticker"),
    min_history: int = 36,
) -> pd.DataFrame:
    """Apply Eq. (1), production form: per-(model, asset) expanding-window z-score.

    For each (model, asset) series sorted by date, the mean and standard deviation are
    estimated on the weeks STRICTLY PRIOR to each observation (an expanding window), so
    the standardization carries no contemporaneous information. Z is defined only once at
    least ``min_history`` prior weeks are available (the burn-in); earlier weeks get NaN.
    This is the strictly out-of-sample estimator behind the B-stabilized paper results.
    """
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([*group_cols, "date"]).reset_index(drop=True)
    g = df.groupby(list(group_cols))[raw_score_col]
    x = df[raw_score_col].astype(float)
    cs = g.cumsum() - x                                   # sum over strictly-prior weeks
    cnt = g.cumcount()                                    # count of strictly-prior weeks
    cs2 = g.transform(lambda s: (s.astype(float) ** 2).cumsum()) - x ** 2
    mu = cs / cnt.replace(0, np.nan)
    sd = np.sqrt((cs2 / cnt.replace(0, np.nan) - mu ** 2).clip(lower=0))
    df["Z"] = np.where(cnt >= min_history, (x - mu) / sd.replace(0, np.nan), np.nan)
    n_valid = int(df["Z"].notna().sum())
    log.info("expanding standardization (burn-in %d): %d / %d rows have valid Z (%.1f%%)",
             min_history, n_valid, len(df), 100 * n_valid / max(1, len(df)))
    return df


def _pairwise_mean_corr(z_matrix: np.ndarray) -> float:
    """Average pairwise Pearson correlation across columns of z_matrix.

    z_matrix has shape (N_assets, K_models). NaN-safe via pairwise complete obs.
    Returns NaN if there are fewer than 2 columns with any data or fewer than 5
    overlapping observations.
    """
    N, K = z_matrix.shape
    if K < 2 or N < 5:
        return np.nan
    corrs = []
    for k in range(K):
        for l in range(k + 1, K):
            x = z_matrix[:, k]
            y = z_matrix[:, l]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            c = np.corrcoef(x[mask], y[mask])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def raw_afsi(
    scores_z: pd.DataFrame,
    *,
    info_set: str | None = None,
    z_col: str = "Z",
) -> pd.DataFrame:
    """Compute Eq. (2): Raw-AFSI per date.

    If info_set is provided, filters to that information set; otherwise computes
    one Raw-AFSI per (date, info_set).
    """
    df = scores_z.dropna(subset=[z_col]).copy()
    if info_set is not None:
        df = df[df["info_set"] == info_set]

    group_cols = ["date"] if info_set is not None else ["date", "info_set"]
    rows = []
    for key, g in df.groupby(group_cols):
        date = key if info_set is not None else key[0]
        iset = info_set if info_set is not None else key[1]
        wide = g.pivot_table(index="asset_ticker", columns="model", values=z_col)
        z = wide.to_numpy()
        afsi_val = _pairwise_mean_corr(z)
        rows.append({
            "date": date,
            "info_set": iset,
            "raw_afsi": afsi_val,
            "n_assets": int(wide.shape[0]),
            "n_models": int(wide.shape[1]),
        })
    return pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)
