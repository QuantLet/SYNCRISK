#!/usr/bin/env python3
"""
Synthetic SYNCRISK panel generator (no licensed data).

Produces a SYNCRISK-shaped panel so the FULL pipeline (Eq.1-7, including the
expanding-window C2x/C3x cleaning and family decomposition) can run end to end
without any EODHD/FRED inputs:

  - returns_daily : daily log returns with a common market factor + idiosyncratic noise
  - panel         : weekly cross-sectional features (the full C2 driver set)
  - scores_long   : 6-model LLM-style relative-risk scores with a shared component
                    (the "excess synchronization" the pipeline is meant to recover)

The six model names match the real family split:
  commercial = claude, gpt, gemini ; open-weight = qwen3-32b, mistral-24b, gemma-3-27b.

Run as a script to write synthetic_{returns,panel,scores}.parquet next to this file
(seed fixed at 42). Imported by tests/test_pipeline.py and the SYNCRISKsynthetic Quantlet.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_ALL = ["claude", "gpt", "gemini", "qwen3-32b", "mistral-24b", "gemma-3-27b"]
MODEL_BIAS = {"claude": 5, "gpt": -3, "gemini": 0,
              "qwen3-32b": 2, "mistral-24b": -2, "gemma-3-27b": 1}


def make_synthetic_panel(n_assets: int = 30, n_weeks: int = 200, n_models: int = 3,
                         correlation_strength: float = 0.6, seed: int = 42):
    """Generate a synthetic SYNCRISK panel.

    Returns dict: panel, scores_long, returns_daily, weekly_dates.
    Scores carry info_set='prices_only' (the test uses the frozen linear cleaning).
    """
    rng = np.random.default_rng(seed)
    weekly_dates = pd.bdate_range("2018-01-05", periods=n_weeks, freq="W-FRI")
    daily_dates = pd.bdate_range(weekly_dates[0] - pd.Timedelta(days=200),
                                 weekly_dates[-1] + pd.Timedelta(days=60))
    assets = [f"A{i:02d}" for i in range(n_assets)]

    market = rng.normal(0, 0.01, len(daily_dates))
    beta_per_asset = rng.uniform(0.5, 1.5, n_assets)
    idio_vol = rng.uniform(0.005, 0.02, n_assets)

    rec = []
    for i, asset in enumerate(assets):
        idio = rng.normal(0, idio_vol[i], len(daily_dates))
        returns = beta_per_asset[i] * market + idio
        for d, r in zip(daily_dates, returns):
            rec.append({"date": d, "asset_ticker": asset, "log_return": r})
    returns_daily = pd.DataFrame(rec)
    mkt = returns_daily.groupby("date")["log_return"].mean().rename("mkt")

    panel_rec = []
    for asset in assets:
        a = returns_daily[returns_daily["asset_ticker"] == asset].set_index("date")
        ar = a["log_return"]
        for wd in weekly_dates:
            w1 = ar[(ar.index > wd - pd.Timedelta(days=3)) & (ar.index <= wd)]
            w5 = ar[(ar.index > wd - pd.Timedelta(days=10)) & (ar.index <= wd)]
            w20 = ar[(ar.index > wd - pd.Timedelta(days=35)) & (ar.index <= wd)]
            w60 = ar[(ar.index > wd - pd.Timedelta(days=100)) & (ar.index <= wd)]
            w90 = ar[(ar.index > pd.Timestamp(wd) - pd.Timedelta(days=140)) & (ar.index <= wd)]
            w250 = ar[(ar.index > wd - pd.Timedelta(days=400)) & (ar.index <= wd)]
            if len(w20) < 15 or len(w60) < 30:
                continue
            rv20 = w20.std(ddof=0) * np.sqrt(252)
            rv60 = w60.std(ddof=0) * np.sqrt(252)
            rv_hist = w250.rolling(20).std().dropna() * np.sqrt(252)
            rv_z = ((rv20 - rv_hist.mean()) / rv_hist.std()) if len(rv_hist) > 5 and rv_hist.std() > 0 else 0.0
            # beta to market over last 60 obs
            mw = mkt.reindex(w60.index)
            beta = float(np.cov(w60.values, mw.values)[0, 1] / np.var(mw.values)) if np.var(mw.values) > 0 else 1.0
            panel_rec.append({
                "date": wd, "asset_ticker": asset,
                "ret_1d": float(w1.sum()) if len(w1) else 0.0,
                "ret_5d": float(w5.sum()) if len(w5) else 0.0,
                "ret_20d": float(w20.sum()),
                "rv_20d": float(rv20), "rv_60d": float(rv60),
                "rv_90d": float(w90.std(ddof=0) * np.sqrt(252)) if len(w90) else float(rv60),
                "dd_60d": float(w60.cumsum().min()),
                "dd_90d": float(w90.cumsum().min()) if len(w90) else float(w60.cumsum().min()),
                "min_ret_20d": float(w20.min()),
                "beta_60d": beta,
                "rv_zscore": float(rv_z),
                "tone_7d": float(rng.normal(0, 0.2)),
                "tone_delta": float(rng.normal(0, 0.2)),
            })
    panel = pd.DataFrame(panel_rec)

    # shared component = cross-sectional rank of rv_20d (the common risk driver)
    shared = {}
    for wd in weekly_dates:
        sub = panel.loc[panel["date"] == wd]
        rv_rank = sub["rv_20d"].rank(pct=True) * 100
        for a, v in zip(sub["asset_ticker"], rv_rank):
            shared[(wd, a)] = v

    models = MODELS_ALL[:n_models]
    srec = []
    for model in models:
        for wd in weekly_dates:
            sub = panel[panel["date"] == wd]
            for _, row in sub.iterrows():
                sh = shared.get((wd, row["asset_ticker"]), 50.0)
                raw = (correlation_strength * sh
                       + (1 - correlation_strength) * rng.uniform(0, 100)
                       + rng.normal(0, 10) + MODEL_BIAS[model])
                srec.append({"date": wd, "asset_ticker": row["asset_ticker"],
                             "model": model, "info_set": "prices_only",
                             "raw_score": float(np.clip(raw, 0, 100))})
    scores_long = pd.DataFrame(srec)
    return {"panel": panel, "scores_long": scores_long,
            "returns_daily": returns_daily, "weekly_dates": weekly_dates}


def main():
    here = Path(__file__).resolve().parent
    data = make_synthetic_panel(n_assets=30, n_weeks=200, n_models=6,
                                correlation_strength=0.6, seed=42)
    data["panel"].to_parquet(here / "synthetic_panel.parquet", index=False)
    data["scores_long"].to_parquet(here / "synthetic_scores.parquet", index=False)
    data["returns_daily"].to_parquet(here / "synthetic_returns.parquet", index=False)
    print(f"panel   {data['panel'].shape}")
    print(f"scores  {data['scores_long'].shape} ({data['scores_long']['model'].nunique()} models)")
    print(f"returns {data['returns_daily'].shape}")
    print("wrote synthetic_{panel,scores,returns}.parquet to", here)


if __name__ == "__main__":
    main()
