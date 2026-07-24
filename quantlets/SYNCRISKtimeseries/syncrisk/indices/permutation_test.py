"""
Permutation test for SYNCRISK (concept note section 3, "Permutation test").

Null hypothesis: model risk rankings are independent given common information.
Under this null, the cross-sectional asset rankings of different models are
exchangeable across asset labels within each (k, t) cell.

Procedure (per date t):
    1. Form the N × K matrix of standardized scores Z (or residuals u).
    2. Independently permute the asset row labels within each model column.
    3. Recompute the index (Raw-AFSI or Excess-AFSI) on the shuffled matrix.
    4. Repeat n_perms times to form a null distribution.
    5. p-value = share of null draws >= observed value (one-sided right tail).

We provide:
- permutation_pvalue_one_date():  test for a single date
- permutation_test_panel():       per-date p-values across the panel

Used for criterion (iii) in the pilot: rejection rate higher in stress than
in normal periods.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .standardization import _pairwise_mean_corr

log = logging.getLogger("syncrisk.indices.permutation")


def permutation_pvalue_one_date(
    z_matrix: np.ndarray,
    *,
    n_perms: int = 500,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """One-sided permutation p-value for AFSI on a single date.

    Parameters
    ----------
    z_matrix : (N_assets, K_models) numpy array
    n_perms  : number of permutations
    seed     : RNG seed for reproducibility

    Returns
    -------
    (observed_afsi, p_value, null_draws)
    """
    rng = np.random.default_rng(seed)
    z_obs = z_matrix.copy()
    # Drop assets with any NaN across models (need exchangeable rows)
    mask = ~np.isnan(z_obs).any(axis=1)
    z_obs = z_obs[mask]
    if z_obs.shape[0] < 5 or z_obs.shape[1] < 2:
        return (np.nan, np.nan, np.array([]))

    observed = _pairwise_mean_corr(z_obs)
    if not np.isfinite(observed):
        return (np.nan, np.nan, np.array([]))

    null_draws = np.empty(n_perms)
    N, K = z_obs.shape
    for i in range(n_perms):
        z_perm = z_obs.copy()
        for k in range(K):
            rng.shuffle(z_perm[:, k])
        null_draws[i] = _pairwise_mean_corr(z_perm)

    p_value = float(np.mean(null_draws >= observed))
    return (float(observed), p_value, null_draws)


def permutation_test_panel(
    scores_or_residuals: pd.DataFrame,
    *,
    value_col: str = "Z",
    info_set: str = "prices_only",
    n_perms: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-date permutation p-values across the entire panel.

    Returns a DataFrame with: date, observed_afsi, p_value, n_assets, n_models.
    """
    df = scores_or_residuals[scores_or_residuals["info_set"] == info_set].dropna(
        subset=[value_col]
    )
    rows = []
    for date, g in df.groupby("date"):
        wide = g.pivot_table(index="asset_ticker", columns="model", values=value_col)
        z = wide.to_numpy()
        obs, p, _ = permutation_pvalue_one_date(
            z, n_perms=n_perms, seed=seed + int(pd.Timestamp(date).value % 1_000_000)
        )
        rows.append({
            "date": date,
            "info_set": info_set,
            "observed_afsi": obs,
            "p_value": p,
            "n_assets": int(wide.shape[0]),
            "n_models": int(wide.shape[1]),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
