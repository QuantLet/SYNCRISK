"""
Shared helpers for the 6-model SYNCRISK evaluation with family decomposition.

Family groupings of model PAIRS:
  - all          : every unordered model pair
  - within_comm  : both models commercial (claude/gpt/gemini)
  - within_open  : both models open-weight (qwen/mistral/gemma)
  - cross_family : one commercial, one open

Provides per-date Raw-AFSI for any grouping and a permutation test that reshuffles
each model's asset ordering independently and recomputes the grouping AFSI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMMERCIAL = ["claude", "gpt", "gemini"]
OPEN = ["qwen3-32b", "mistral-24b", "gemma-3-27b"]
FAMILY = {**{m: "commercial" for m in COMMERCIAL}, **{m: "open" for m in OPEN}}

# Groupings used across the analysis (order matters for table layout).
GROUPINGS = ["all", "within_comm", "within_open", "cross_family"]


def _z(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def wide_z_for_date(panel: pd.DataFrame, score_col="score_relative_risk_score"):
    """Return (models list, Z matrix N_assets x K) for one date; assets with any
    NaN across models dropped. Z is within-(model) standardized across assets."""
    wide = panel.pivot_table(index="asset_ticker", columns="model", values=score_col)
    wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")
    models = list(wide.columns)
    if len(models) < 2 or wide.shape[0] < 5:
        return models, None
    Z = np.column_stack([_z(wide[m].to_numpy()) for m in models])
    return models, Z


def _pair_indices(models, grouping):
    pairs = []
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            fa, fb = FAMILY.get(models[a]), FAMILY.get(models[b])
            if grouping == "all":
                ok = True
            elif grouping == "within_comm":
                ok = fa == "commercial" and fb == "commercial"
            elif grouping == "within_open":
                ok = fa == "open" and fb == "open"
            elif grouping == "cross_family":
                ok = fa != fb
            else:
                ok = False
            if ok:
                pairs.append((a, b))
    return pairs


def _afsi_from_pairs(Z, pairs):
    cs = []
    for (a, b) in pairs:
        x, y = Z[:, a], Z[:, b]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 5:
            c = np.corrcoef(x[m], y[m])[0, 1]
            if np.isfinite(c):
                cs.append(c)
    return float(np.mean(cs)) if cs else np.nan


def raw_afsi_grouping(panel, grouping, score_col="score_relative_risk_score"):
    models, Z = wide_z_for_date(panel, score_col)
    if Z is None:
        return np.nan
    pairs = _pair_indices(models, grouping)
    if not pairs:
        return np.nan
    return _afsi_from_pairs(Z, pairs)


def afsi_series(resid_or_z, value_col, groupings=GROUPINGS):
    """Per-date AFSI for each grouping from a long frame with `value_col`."""
    recs = []
    for date, g in resid_or_z.dropna(subset=[value_col]).groupby("date"):
        wide = g.pivot_table(index="asset_ticker", columns="model", values=value_col).dropna(axis=0, how="any")
        models = list(wide.columns)
        Z = wide.to_numpy()
        row = {"date": date}
        if len(models) >= 2 and wide.shape[0] >= 5:
            for grp in groupings:
                pairs = _pair_indices(models, grp)
                row[grp] = _afsi_from_pairs(Z, pairs) if pairs else np.nan
        recs.append(row)
    return pd.DataFrame(recs).sort_values("date")


def permutation_pvalue_grouping(panel, grouping, n_perms=500, seed=0,
                                score_col="score_relative_risk_score"):
    models, Z = wide_z_for_date(panel, score_col)
    if Z is None:
        return (np.nan, np.nan)
    pairs = _pair_indices(models, grouping)
    if not pairs:
        return (np.nan, np.nan)
    obs = _afsi_from_pairs(Z, pairs)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for i in range(n_perms):
        Zp = Z.copy()
        for k in range(Zp.shape[1]):
            rng.shuffle(Zp[:, k])
        null[i] = _afsi_from_pairs(Zp, pairs)
    p = float(np.mean(null >= obs))
    return (float(obs), p)


def diag_signal_vs_score(scores: pd.DataFrame, info_set: str = "prices_plus_news") -> dict:
    """Risk score should be related to but not collinear with de_risking_signal.

    Separates SYNCRISK from machine-forecast-disagreement work on expected returns
    (where the signal *is* the score). Returns per-model corr (target high but <0.95).
    """
    sub = scores[scores["info_set"] == info_set].copy()
    sub["signal_numeric"] = sub["score_de_risking_signal"].map(
        {"reduce": 1.0, "hold": 0.0, "increase": -1.0})
    out = {}
    for model, m_sub in sub.groupby("model"):
        if m_sub["signal_numeric"].nunique() < 2:
            out[model] = None
            continue
        corr = float(m_sub["score_relative_risk_score"].corr(m_sub["signal_numeric"]))
        out[model] = round(corr, 3)
    return {"diagnostic": "score vs signal correlation (should be high but < 0.95)",
            "by_model": out}
