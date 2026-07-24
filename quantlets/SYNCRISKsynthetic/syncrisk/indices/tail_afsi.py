"""
Tail-AFSI (Eq. 5) for SYNCRISK.

Eq. (5): Share of assets that a qualified majority of models flag as high-risk:

    H_{i,t}^{(k)} = 1{Z_{i,t}^{(k)} > c_k}
    TailAFSI_t(q) = (1/N) * sum_i 1{(1/K) * sum_k H_{i,t}^{(k)} >= q}

with c_k the 90th percentile of model k's standardized scores in the
training sample (held fixed OOS to avoid look-ahead).

q ∈ {0.5, 0.67}: half-majority and two-thirds majority thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("syncrisk.indices.tail_afsi")


@dataclass
class TailThresholds:
    """Per-model 90th-percentile thresholds, frozen on training data."""
    thresholds: dict[str, float]
    quantile: float
    train_date_range: tuple


def fit_tail_thresholds(
    scores_z: pd.DataFrame,
    *,
    info_set: str = "prices_only",
    z_col: str = "Z",
    train_start: str,
    train_end: str,
    quantile: float = 0.90,
) -> TailThresholds:
    """Compute c_k on the training window for each model."""
    sub = scores_z[scores_z["info_set"] == info_set].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    mask = (sub["date"] >= train_start) & (sub["date"] <= train_end)
    train = sub.loc[mask].dropna(subset=[z_col])

    thresholds: dict[str, float] = {}
    for model_name, g in train.groupby("model"):
        c = float(g[z_col].quantile(quantile))
        thresholds[model_name] = c
        log.info("tail threshold c_%s @ q=%.2f = %.3f (n_train=%d)",
                 model_name, quantile, c, len(g))

    return TailThresholds(
        thresholds=thresholds,
        quantile=quantile,
        train_date_range=(train_start, train_end),
    )


def tail_afsi(
    scores_z: pd.DataFrame,
    thresholds: TailThresholds,
    *,
    info_set: str = "prices_only",
    z_col: str = "Z",
    q_values: tuple = (0.5, 0.67),
) -> pd.DataFrame:
    """Compute TailAFSI_t(q) for each date and each q in q_values."""
    sub = scores_z[scores_z["info_set"] == info_set].copy().dropna(subset=[z_col])

    # H_{i,t}^{(k)} = indicator
    sub["c_k"] = sub["model"].map(thresholds.thresholds)
    valid = sub.dropna(subset=["c_k"])
    valid["H"] = (valid[z_col] > valid["c_k"]).astype(int)

    rows = []
    for date, g in valid.groupby("date"):
        wide = g.pivot_table(index="asset_ticker", columns="model", values="H",
                             aggfunc="max")
        K = wide.shape[1]
        if K < 2:
            continue
        agreement_share = wide.mean(axis=1)  # average over models per asset
        row = {"date": date, "info_set": info_set, "n_assets": int(wide.shape[0]),
               "n_models": K}
        for q in q_values:
            row[f"tail_afsi_q{int(q*100)}"] = float((agreement_share >= q).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
