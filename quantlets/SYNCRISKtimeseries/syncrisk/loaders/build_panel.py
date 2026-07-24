#!/usr/bin/env python3
"""
Build the SYNCRISK weekly panel from EODHD All-In-One.

Inputs
------
- config/tickers.yaml: the 80-asset panel with EODHD symbol mappings
- environment: EODHD_API_TOKEN
- date range: 2018-01-01 to today

Outputs (under --out directory)
-------
- prices_daily.parquet      one row per (asset, date), OHLCV + adjusted close
- returns_daily.parquet     log returns + rolling features (RV, drawdown, etc.)
- panel_weekly.parquet      Friday-close weekly panel with all features needed
                            for LLM prompts (the input to run_pilot.py)
- news_articles.parquet     raw articles, one row per (article, query_ticker)
- sentiment_weekly.parquet  weekly aggregated sentiment from /sentiments endpoint
- data_quality.csv          coverage report

Usage
-----
    export EODHD_API_TOKEN=...
    python build_panel.py \
        --tickers config/tickers.yaml \
        --from 2018-01-01 \
        --to   2026-05-23 \
        --out  data/panel_v1/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .eodhd_client import EODHDClient

log = logging.getLogger("syncrisk.panel")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_returns_and_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset rolling features from daily prices.

    Assumes df_prices has columns: date, asset_ticker, adjusted_close (or close).
    """
    out = []
    for asset, g in df_prices.sort_values(["asset_ticker", "date"]).groupby("asset_ticker"):
        g = g.copy().reset_index(drop=True)
        # Use adjusted_close when available, else close
        price_col = "adjusted_close" if "adjusted_close" in g.columns and g["adjusted_close"].notna().any() else "close"
        p = g[price_col].astype(float)
        g["log_return"] = np.log(p / p.shift(1))

        # Rolling volatility (annualized)
        g["rv_20d"] = g["log_return"].rolling(20).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
        g["rv_60d"] = g["log_return"].rolling(60).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
        g["rv_250d"] = g["log_return"].rolling(250).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Cumulative returns
        g["ret_5d"] = g["log_return"].rolling(5).sum()
        g["ret_20d"] = g["log_return"].rolling(20).sum()

        # Drawdown over 60d window
        roll_max = p.rolling(60, min_periods=1).max()
        g["dd_60d"] = p / roll_max - 1.0

        # Min single-day return over window
        g["min_ret_20d"] = g["log_return"].rolling(20).min()

        # Skew/kurt
        g["skew_250d"] = g["log_return"].rolling(250).skew()
        g["kurt_250d"] = g["log_return"].rolling(250).kurt()

        # RV z-score vs trailing 1y distribution
        rv = g["rv_20d"]
        rv_mean = rv.rolling(250).mean()
        rv_std = rv.rolling(250).std(ddof=0)
        g["rv_zscore"] = (rv - rv_mean) / rv_std.replace(0, np.nan)

        out.append(g)
    return pd.concat(out, ignore_index=True)


def compute_beta_to_market(
    df: pd.DataFrame, market_ticker: str,
    window: int = 60, return_col: str = "log_return",
) -> pd.DataFrame:
    """Add beta_60d column: rolling beta of each asset to a benchmark ticker."""
    market = df[df["asset_ticker"] == market_ticker][["date", return_col]].copy()
    market = market.rename(columns={return_col: "market_return"})
    merged = df.merge(market, on="date", how="left")

    out = []
    for asset, g in merged.sort_values(["asset_ticker", "date"]).groupby("asset_ticker"):
        g = g.copy().reset_index(drop=True)
        cov = g[return_col].rolling(window).cov(g["market_return"])
        var = g["market_return"].rolling(window).var(ddof=0)
        g["beta_60d"] = cov / var.replace(0, np.nan)
        out.append(g.drop(columns=["market_return"]))
    return pd.concat(out, ignore_index=True)


def to_weekly_friday_panel(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Take the Friday observation for each asset-week.

    Uses last available trading day in each Friday-anchored week. If no trade
    on Friday, falls back to last available day that week.
    """
    df = df_daily.sort_values(["asset_ticker", "date"]).copy()
    df["week_anchor"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d") + pd.Timedelta(days=4)
    # week_anchor is the Friday of each ISO week
    weekly = df.groupby(["asset_ticker", "week_anchor"]).tail(1)
    weekly = weekly.rename(columns={"week_anchor": "week_date"})
    weekly = weekly.drop(columns=["date"], errors="ignore")
    weekly = weekly.rename(columns={"week_date": "date"})
    return weekly.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Headlines per (asset, week)
# ---------------------------------------------------------------------------

def aggregate_weekly_headlines(
    df_articles: pd.DataFrame, top_n: int = 10,
) -> pd.DataFrame:
    """Collapse article-level news into weekly top-N headlines per asset.

    df_articles must have: query_ticker, date (timestamp), title, sent_polarity.
    Returns one row per (asset_ticker, week_date) with headlines (list[str]),
    tone_7d (mean polarity over the week), and tone_delta vs previous week.
    """
    if df_articles.empty:
        return pd.DataFrame(columns=["asset_ticker", "date", "headlines",
                                     "tone_7d", "tone_delta"])

    df = df_articles.dropna(subset=["title", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["week_date"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d")
                       + pd.Timedelta(days=4))
    df["week_date"] = df["week_date"].dt.normalize()

    rows = []
    for (ticker, week), g in df.groupby(["query_ticker", "week_date"]):
        g_sorted = g.sort_values("date", ascending=False)
        headlines = g_sorted["title"].dropna().astype(str).head(top_n).tolist()
        tone = g_sorted["sent_polarity"].dropna().mean() if "sent_polarity" in g.columns else None
        rows.append({
            "asset_ticker": ticker,
            "date": week,
            "headlines": headlines,
            "tone_7d": float(tone) if tone is not None and not pd.isna(tone) else None,
            "n_articles_week": int(len(g)),
        })
    out = pd.DataFrame(rows).sort_values(["asset_ticker", "date"]).reset_index(drop=True)
    # tone_delta = tone_7d - tone_7d.shift(1) per asset
    out["tone_delta"] = out.groupby("asset_ticker")["tone_7d"].diff()
    return out


# ---------------------------------------------------------------------------
# Main async orchestrator
# ---------------------------------------------------------------------------

async def fetch_one_asset(
    client: EODHDClient, eodhd_symbol: str, from_date: str, to_date: str,
) -> dict:
    """Fetch all data needed for one asset. Returns dict of DataFrames."""
    log.info("[%s] fetching EOD + news + sentiment", eodhd_symbol)
    prices_task = client.eod_history(eodhd_symbol, from_date, to_date)
    news_task = client.news_all_pages(eodhd_symbol, from_date, to_date)
    sent_task = client.sentiments_history(eodhd_symbol, from_date, to_date)
    prices, news, sent = await asyncio.gather(
        prices_task, news_task, sent_task, return_exceptions=True
    )
    out = {"ticker": eodhd_symbol}
    for key, val in [("prices", prices), ("news", news), ("sentiment", sent)]:
        if isinstance(val, Exception):
            log.error("[%s] %s failed: %s", eodhd_symbol, key, val)
            out[key] = pd.DataFrame()
        else:
            out[key] = val
    log.info("[%s] prices=%d news=%d sent=%d",
             eodhd_symbol, len(out["prices"]), len(out["news"]), len(out["sentiment"]))
    return out


async def build_panel(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.tickers) as f:
        config = yaml.safe_load(f)
    tickers = config["assets"]  # list of dicts: {ticker, name, eodhd_symbol, category, ...}
    log.info("loaded %d tickers from %s", len(tickers), args.tickers)

    market_proxy = config.get("market_benchmark", "SPY.US")

    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise SystemExit("EODHD_API_TOKEN environment variable not set")

    async with EODHDClient(token, concurrency=args.concurrency) as client:
        # Concurrent fetch for all tickers
        results = await asyncio.gather(*[
            fetch_one_asset(client, t["eodhd_symbol"], args.from_date, args.to_date)
            for t in tickers
        ])

    # Assemble prices
    prices_frames = [r["prices"] for r in results if not r["prices"].empty]
    if not prices_frames:
        raise SystemExit("no price data downloaded; check tickers/api token")
    prices = pd.concat(prices_frames, ignore_index=True)

    # Ticker → asset_ticker (use EODHD symbol as the canonical ID throughout)
    prices.to_parquet(out_dir / "prices_daily.parquet", index=False)
    log.info("wrote prices_daily.parquet (%d rows)", len(prices))

    # Features
    feat = compute_returns_and_features(prices)
    feat = compute_beta_to_market(feat, market_ticker=market_proxy)
    # ret_1d is just log_return today
    feat["ret_1d"] = feat["log_return"]
    feat.to_parquet(out_dir / "returns_daily.parquet", index=False)
    log.info("wrote returns_daily.parquet (%d rows)", len(feat))

    # Weekly snapshot (Friday close)
    weekly_features = to_weekly_friday_panel(feat)

    # News aggregation per (asset, week)
    news_frames = [r["news"] for r in results if not r["news"].empty]
    if news_frames:
        news_all = pd.concat(news_frames, ignore_index=True)
        news_all.to_parquet(out_dir / "news_articles.parquet", index=False)
        log.info("wrote news_articles.parquet (%d articles)", len(news_all))
        weekly_news = aggregate_weekly_headlines(news_all, top_n=args.top_n_headlines)
    else:
        log.warning("no news data downloaded")
        weekly_news = pd.DataFrame(columns=["asset_ticker", "date", "headlines",
                                            "tone_7d", "tone_delta"])

    # Daily sentiment → weekly (mean over the week)
    sent_frames = [r["sentiment"] for r in results if not r["sentiment"].empty]
    if sent_frames:
        sent_all = pd.concat(sent_frames, ignore_index=True)
        sent_all["week_date"] = (sent_all["date"]
                                 - pd.to_timedelta(sent_all["date"].dt.dayofweek, unit="d")
                                 + pd.Timedelta(days=4)).dt.normalize()
        sent_weekly = (sent_all.groupby(["asset_ticker", "week_date"])
                       .agg(sent_normalized=("sent_normalized", "mean"),
                            n_articles_sentiment=("n_articles", "sum"))
                       .reset_index()
                       .rename(columns={"week_date": "date"}))
        sent_weekly.to_parquet(out_dir / "sentiment_weekly.parquet", index=False)
        log.info("wrote sentiment_weekly.parquet (%d rows)", len(sent_weekly))
    else:
        sent_weekly = pd.DataFrame(columns=["asset_ticker", "date",
                                            "sent_normalized", "n_articles_sentiment"])

    # Join everything → panel_weekly
    panel = weekly_features.merge(weekly_news, on=["asset_ticker", "date"], how="left")
    panel = panel.merge(sent_weekly, on=["asset_ticker", "date"], how="left")

    # Add asset_name column from tickers config
    name_map = {t["eodhd_symbol"]: t.get("name", t["eodhd_symbol"]) for t in tickers}
    panel["asset_name"] = panel["asset_ticker"].map(name_map)
    category_map = {t["eodhd_symbol"]: t.get("category", "unknown") for t in tickers}
    panel["category"] = panel["asset_ticker"].map(category_map)

    # Drop rows with missing essentials
    essential_cols = ["ret_1d", "ret_5d", "ret_20d", "rv_20d", "rv_60d",
                      "dd_60d", "min_ret_20d", "beta_60d", "rv_zscore"]
    before = len(panel)
    panel = panel.dropna(subset=essential_cols)
    log.info("panel: %d → %d rows after dropping NaN essentials", before, len(panel))

    panel.to_parquet(out_dir / "panel_weekly.parquet", index=False)
    log.info("wrote panel_weekly.parquet (%d rows)", len(panel))

    # Data quality
    qa = (panel.groupby("asset_ticker")
          .agg(n_weeks=("date", "size"),
               first_date=("date", "min"),
               last_date=("date", "max"),
               pct_with_news=("headlines", lambda s: s.apply(
                   lambda h: bool(h) if isinstance(h, list) else False).mean()),
               mean_articles_per_week=("n_articles_week", "mean"),
               mean_sent_normalized=("sent_normalized", "mean"))
          .round(3))
    qa.to_csv(out_dir / "data_quality.csv")
    log.info("wrote data_quality.csv")

    print(f"\n✓ Panel built: {len(panel)} weekly rows × {panel['asset_ticker'].nunique()} assets")
    print(f"✓ Output: {out_dir.resolve()}")
    print(f"\nNext step:")
    print(f"  python run_pilot.py --panel {out_dir}/panel_weekly.parquet --out runs/pilot_01")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", required=True, type=Path,
                   help="YAML config with assets list")
    p.add_argument("--from", dest="from_date", required=True,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", required=True,
                   help="End date YYYY-MM-DD")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--top-n-headlines", type=int, default=10)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(build_panel(args))


if __name__ == "__main__":
    main()
