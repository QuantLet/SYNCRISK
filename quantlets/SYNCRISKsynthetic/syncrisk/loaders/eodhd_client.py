"""
EODHD API client for SYNCRISK — All-In-One plan endpoints.

Endpoints used:
  - /api/eod/{ticker}              EOD OHLCV historical
  - /api/news                       Financial news per ticker, with sentiment per article
  - /api/sentiments                 Daily aggregated sentiment per ticker
  - /api/fundamentals/{ticker}      Company fundamentals (for market cap stratification)
  - /api/exchange-symbol-list/{exch} Ticker universe (for sanity checks)

Plan accounting (All-In-One, May 2026):
  - 100,000 API calls/day, 1,000 calls/minute soft limit
  - EOD: 1 call per ticker per request
  - News: 5 calls per ticker per request
  - Sentiment: 5 calls per ticker per request
  - Fundamentals: 10 calls per request

For SYNCRISK panel (80 assets × 8 years):
  - EOD download: 80 calls — trivial
  - News: 80 × ~30 batches (1000 articles each) × 5 = ~12,000 calls
  - Sentiment: 80 × ~30 windows × 5 = ~12,000 calls
  - Total well within daily budget.

Usage:
  client = EODHDClient(api_token=os.environ["EODHD_API_TOKEN"])
  df_prices = await client.eod_history("AAPL.US", "2018-01-01", "2026-05-01")
  df_news   = await client.news_all_pages("AAPL.US", "2018-01-01", "2026-05-01")
  df_sent   = await client.sentiments_history("AAPL.US", "2018-01-01", "2026-05-01")
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import httpx
import pandas as pd

log = logging.getLogger("syncrisk.eodhd")

BASE_URL = "https://eodhd.com/api"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
MAX_RETRIES = 5
BASE_BACKOFF = 1.5


class EODHDError(Exception):
    pass


class EODHDRateLimitError(EODHDError):
    pass


@dataclass
class APIUsage:
    """Track approximate API call consumption against the daily budget."""
    calls_today: int = 0
    daily_budget: int = 100_000

    def add(self, cost: int) -> None:
        self.calls_today += cost
        if self.calls_today > self.daily_budget * 0.95:
            log.warning("EODHD daily call budget at %d / %d (95%% threshold)",
                        self.calls_today, self.daily_budget)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EODHDClient:
    """Async EODHD client. One instance per process. Thread-unsafe."""

    def __init__(
        self,
        api_token: str,
        *,
        concurrency: int = 10,
        usage: Optional[APIUsage] = None,
    ):
        if not api_token:
            raise ValueError("EODHD_API_TOKEN missing")
        self.api_token = api_token
        self.usage = usage or APIUsage()
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "syncrisk-eodhd/0.1"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        await self._client.aclose()

    # ---------- Core request with retries ----------
    async def _get(self, path: str, params: dict, cost: int = 1) -> dict | list:
        """Generic GET with exponential backoff and rate-limit handling."""
        params = {**params, "api_token": self.api_token, "fmt": "json"}
        url = f"{BASE_URL}{path}"

        async with self._sem:
            last_exc: Exception | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await self._client.get(url, params=params)
                    if resp.status_code == 429:
                        wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                        log.warning("[EODHD] 429 on %s, sleeping %.1fs", path, wait)
                        await asyncio.sleep(wait)
                        last_exc = EODHDRateLimitError(resp.text[:200])
                        continue
                    if resp.status_code >= 500:
                        wait = BASE_BACKOFF * (1.5 ** attempt)
                        log.warning("[EODHD] %d on %s, sleeping %.1fs",
                                    resp.status_code, path, wait)
                        await asyncio.sleep(wait)
                        last_exc = EODHDError(f"HTTP {resp.status_code}")
                        continue
                    resp.raise_for_status()
                    self.usage.add(cost)
                    return resp.json()
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    wait = BASE_BACKOFF * (1.5 ** attempt)
                    log.warning("[EODHD] network error on %s (attempt %d): %s",
                                path, attempt + 1, e)
                    await asyncio.sleep(wait)
                    last_exc = e
            raise EODHDError(f"exhausted retries on {path}: {last_exc}")

    # ---------- EOD prices ----------
    async def eod_history(
        self, ticker: str, from_date: str, to_date: str,
        *, period: str = "d",
    ) -> pd.DataFrame:
        """End-of-day OHLCV for one ticker.

        Returns DataFrame with columns: date, open, high, low, close, adjusted_close, volume.
        Empty DataFrame on no data.
        """
        data = await self._get(
            f"/eod/{ticker}",
            params={"from": from_date, "to": to_date, "period": period},
            cost=1,
        )
        if not data:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                          "adjusted_close", "volume"])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["asset_ticker"] = ticker
        return df

    # ---------- News ----------
    async def news_page(
        self, ticker: str, from_date: str, to_date: str,
        *, offset: int = 0, limit: int = 1000,
    ) -> list[dict]:
        """One page of news for a ticker. Max 1000 articles per page."""
        data = await self._get(
            "/news",
            params={
                "s": ticker, "from": from_date, "to": to_date,
                "offset": offset, "limit": limit,
            },
            cost=5,
        )
        return data if isinstance(data, list) else []

    async def news_all_pages(
        self, ticker: str, from_date: str, to_date: str,
        *, page_size: int = 1000, max_articles: int = 50_000,
    ) -> pd.DataFrame:
        """Paginate through all news for a ticker over a date range.

        Returns DataFrame with one row per article. The 'symbols' column lists all
        tickers tagged (sometimes the article is multi-ticker). The 'sentiment'
        column is the per-article sentiment dict from EODHD.

        Fetches PER CALENDAR YEAR (each year as its own from/to window). EODHD
        returns news newest-first, so a single whole-range request capped at
        max_articles truncates the EARLY years for high-volume names; year-chunking
        recovers full history (per-year counts are well under the cap). The cap is
        applied per year as a backstop.
        """
        all_articles: list[dict] = []
        start_year = int(str(from_date)[:4])
        end_year = int(str(to_date)[:4])
        for yr in range(start_year, end_year + 1):
            y_from = max(from_date, f"{yr}-01-01")
            y_to = min(to_date, f"{yr}-12-31")
            offset = 0
            year_count = 0
            while True:
                page = await self.news_page(ticker, y_from, y_to,
                                            offset=offset, limit=page_size)
                if not page:
                    break
                all_articles.extend(page)
                year_count += len(page)
                if len(page) < page_size:
                    break
                offset += page_size
                if year_count >= max_articles:
                    log.warning("[EODHD] news_all_pages: capped at %d for %s in %d",
                                max_articles, ticker, yr)
                    break

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        # Normalize fields
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["query_ticker"] = ticker
        # Extract sentiment polarity if present
        if "sentiment" in df.columns:
            df["sent_polarity"] = df["sentiment"].apply(
                lambda s: float(s.get("polarity")) if isinstance(s, dict) and s.get("polarity") is not None else None
            )
            df["sent_neg"] = df["sentiment"].apply(
                lambda s: float(s.get("neg")) if isinstance(s, dict) else None
            )
            df["sent_pos"] = df["sentiment"].apply(
                lambda s: float(s.get("pos")) if isinstance(s, dict) else None
            )
            df["sent_neu"] = df["sentiment"].apply(
                lambda s: float(s.get("neu")) if isinstance(s, dict) else None
            )
        return df

    # ---------- Daily aggregated sentiment ----------
    async def sentiments_history(
        self, ticker: str, from_date: str, to_date: str,
    ) -> pd.DataFrame:
        """Daily aggregated sentiment for one ticker.

        Returns DataFrame with one row per date: date, normalized polarity in [-1, 1],
        count of articles. This is the cleaner per-day signal vs aggregating
        news_all_pages manually.
        """
        data = await self._get(
            "/sentiments",
            params={"s": ticker, "from": from_date, "to": to_date},
            cost=5,
        )
        # Response is {ticker: [{date, count, normalized}, ...]}
        if not isinstance(data, dict):
            return pd.DataFrame()
        rows = []
        for tk, days in data.items():
            for d in days or []:
                rows.append({
                    "asset_ticker": tk,
                    "date": pd.to_datetime(d.get("date")),
                    "sent_normalized": d.get("normalized"),
                    "n_articles": d.get("count"),
                })
        return pd.DataFrame(rows)

    # ---------- Fundamentals (for market cap / sector stratification) ----------
    async def fundamentals(self, ticker: str) -> dict:
        return await self._get(f"/fundamentals/{ticker}", params={}, cost=10)
