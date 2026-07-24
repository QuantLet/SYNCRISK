"""
Macro stress data loader for SYNCRISK.

Downloads the time-varying market-wide controls W_t = (VIX, OFR FSI,
R^market, HY spread, average news tone) at daily frequency, then resamples to
weekly (Friday close) to match the SYNCRISK panel.

Sources:
- FRED (Federal Reserve Economic Data) via direct CSV API. No credential is
  needed for the CSV endpoint used here.
- OFR FSI via direct CSV from the Office of Financial Research.

Series fetched
--------------
- VIXCLS:       CBOE VIX (daily close)
- BAMLH0A0HYM2: ICE BofA US High Yield Option-Adjusted Spread
- DGS10:        10-Year Treasury Constant Maturity Rate
- DGS2:         2-Year Treasury Constant Maturity Rate
- DCOILWTICO:   WTI crude oil (auxiliary control)
- NFCI:         Chicago Fed National Financial Conditions Index (weekly, ffilled)
- OFR FSI:      Office of Financial Research Financial Stress Index

Output
------
- macro_daily.parquet:   date, vix, hy_spread, dgs10, dgs2, term_slope, nfci,
                         ofr_fsi, wti
- macro_weekly.parquet:  same columns, Friday-anchored, last value of each week
- W_t panel:             ready to merge with panel_weekly.parquet on 'date'
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger("syncrisk.macro")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OFR_FSI_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

FRED_SERIES = {
    "VIXCLS":       "vix",
    "BAMLH0A0HYM2": "hy_spread",
    "DGS10":        "dgs10",
    "DGS2":         "dgs2",
    "DCOILWTICO":   "wti",
    "NFCI":         "nfci",
}


def fetch_fred_csv(series_id: str, from_date: str, to_date: str,
                   timeout: float = 30.0) -> pd.DataFrame:
    """Fetch one FRED series via CSV. Returns DataFrame with [date, value]."""
    params = {"id": series_id, "cosd": from_date, "coed": to_date}
    log.info("[FRED] %s [%s → %s]", series_id, from_date, to_date)
    r = httpx.get(FRED_CSV_URL, params=params, timeout=timeout,
                  follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if df.empty or df.shape[1] < 2:
        log.warning("[FRED] %s: empty response", series_id)
        return pd.DataFrame(columns=["date", "value"])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_ofr_fsi(timeout: float = 30.0) -> pd.DataFrame:
    """Fetch the OFR Financial Stress Index. Returns DataFrame [date, ofr_fsi].

    The OFR FSI CSV layout has been stable: 'Date' column plus 'OFR FSI'.
    If the schema changes, this function will warn and return empty.
    """
    log.info("[OFR] fetching FSI")
    try:
        r = httpx.get(OFR_FSI_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except (httpx.HTTPError, httpx.NetworkError) as e:
        log.error("[OFR] fetch failed: %s — returning empty", e)
        return pd.DataFrame(columns=["date", "ofr_fsi"])

    try:
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        log.error("[OFR] CSV parse failed: %s", e)
        return pd.DataFrame(columns=["date", "ofr_fsi"])

    # Identify date and FSI columns regardless of slight name changes
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    fsi_col = next((c for c in df.columns if "fsi" in c.lower() or "ofr" in c.lower()),
                    None)
    if not date_col:
        date_col = df.columns[0]
    if not fsi_col:
        fsi_col = df.columns[1] if df.shape[1] > 1 else None

    if fsi_col is None:
        log.error("[OFR] could not identify FSI column; columns: %s", list(df.columns))
        return pd.DataFrame(columns=["date", "ofr_fsi"])

    out = df[[date_col, fsi_col]].copy()
    out.columns = ["date", "ofr_fsi"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ofr_fsi"] = pd.to_numeric(out["ofr_fsi"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def build_macro_daily(from_date: str, to_date: str) -> pd.DataFrame:
    """Assemble the daily macro stress panel."""
    panels = []
    for series_id, col_name in FRED_SERIES.items():
        df = fetch_fred_csv(series_id, from_date, to_date)
        df = df.rename(columns={"value": col_name})
        panels.append(df)

    # Outer-join on date
    merged = panels[0]
    for p in panels[1:]:
        merged = merged.merge(p, on="date", how="outer")

    # OFR FSI
    ofr = fetch_ofr_fsi()
    ofr = ofr[(ofr["date"] >= from_date) & (ofr["date"] <= to_date)]
    merged = merged.merge(ofr, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Derived: term slope = DGS10 - DGS2
    if "dgs10" in merged.columns and "dgs2" in merged.columns:
        merged["term_slope"] = merged["dgs10"] - merged["dgs2"]

    # Forward-fill NFCI (which is weekly) and OFR FSI gaps that are <= 5 days
    if "nfci" in merged.columns:
        merged["nfci"] = merged["nfci"].ffill(limit=7)
    if "ofr_fsi" in merged.columns:
        merged["ofr_fsi"] = merged["ofr_fsi"].ffill(limit=5)

    return merged


def add_market_return(macro_daily: pd.DataFrame,
                      prices_daily_path: Path,
                      market_ticker: str = "SPY.US") -> pd.DataFrame:
    """Compute daily market log return from the panel's market benchmark.

    Merges into macro_daily so W_t has R^market available.
    """
    if not prices_daily_path.exists():
        log.warning("prices file %s missing; market return left null", prices_daily_path)
        macro_daily["market_return"] = pd.NA
        return macro_daily

    prices = pd.read_parquet(prices_daily_path)
    mkt = prices[prices["asset_ticker"] == market_ticker].copy()
    if mkt.empty:
        log.warning("market ticker %s not in prices; market return left null", market_ticker)
        macro_daily["market_return"] = pd.NA
        return macro_daily

    price_col = "adjusted_close" if "adjusted_close" in mkt.columns else "close"
    mkt = mkt.sort_values("date").reset_index(drop=True)
    import numpy as np
    mkt["market_return"] = np.log(mkt[price_col] / mkt[price_col].shift(1))
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
    macro_daily["date"] = pd.to_datetime(macro_daily["date"]).dt.normalize()
    out = macro_daily.merge(mkt[["date", "market_return"]], on="date", how="left")
    return out


def to_weekly_friday(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample to weekly Friday-anchored, last value of each week."""
    df = macro_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_date"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d")
                       + pd.Timedelta(days=4)).dt.normalize()

    # For levels: last observation of the week
    levels = ["vix", "hy_spread", "dgs10", "dgs2", "term_slope", "nfci",
              "ofr_fsi", "wti"]
    agg = {c: "last" for c in levels if c in df.columns}
    # For market_return: sum the daily log returns over the week
    if "market_return" in df.columns:
        agg["market_return"] = "sum"

    weekly = df.groupby("week_date").agg(agg).reset_index()
    weekly = weekly.rename(columns={"week_date": "date"})
    return weekly


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--prices", type=Path, default=None,
                   help="Path to prices_daily.parquet for market return")
    p.add_argument("--market-ticker", default="SPY.US")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    daily = build_macro_daily(args.from_date, args.to_date)
    if args.prices and args.prices.exists():
        daily = add_market_return(daily, args.prices, args.market_ticker)
    daily.to_parquet(args.out / "macro_daily.parquet", index=False)
    log.info("wrote macro_daily.parquet (%d rows)", len(daily))

    weekly = to_weekly_friday(daily)
    weekly.to_parquet(args.out / "macro_weekly.parquet", index=False)
    log.info("wrote macro_weekly.parquet (%d rows)", len(weekly))

    coverage = weekly.notna().sum()
    print("\n=== Macro weekly coverage ===")
    print(coverage.to_string())


if __name__ == "__main__":
    sys.exit(main())
