"""Indian market helpers: IST clock, NSE session times, symbols and free data.

Symbols use the ``EXCHANGE:TRADINGSYMBOL`` convention, e.g. ``NSE:INFY`` or
``BSE:500325``. A bare symbol defaults to NSE.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

#: India has no daylight saving, so a fixed offset is exact (and avoids
#: needing the tz database, which Windows and some Android setups lack).
IST = dt.timezone(dt.timedelta(hours=5, minutes=30), name="IST")

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def parse_symbol(symbol: str) -> tuple[str, str]:
    """``"NSE:INFY"`` -> ``("NSE", "INFY")``; ``"INFY"`` -> ``("NSE", "INFY")``."""
    if ":" in symbol:
        exchange, tradingsymbol = symbol.split(":", 1)
        return exchange.strip().upper(), tradingsymbol.strip().upper()
    return "NSE", symbol.strip().upper()


def normalize_symbol(symbol: str) -> str:
    ex, ts = parse_symbol(symbol)
    return f"{ex}:{ts}"


@dataclass
class MarketCalendar:
    """Weekday NSE sessions minus a user-maintained holiday list."""

    holidays: frozenset[dt.date] = frozenset()
    open_time: dt.time = MARKET_OPEN
    close_time: dt.time = MARKET_CLOSE

    @classmethod
    def from_strings(cls, dates: list[str] | None) -> "MarketCalendar":
        parsed = set()
        for d in dates or []:
            try:
                parsed.add(dt.date.fromisoformat(str(d).strip()))
            except ValueError:
                log.warning("ignoring bad holiday date %r", d)
        return cls(frozenset(parsed))

    def is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def is_open(self, when: dt.datetime) -> bool:
        when = when.astimezone(IST)
        return (self.is_trading_day(when.date())
                and self.open_time <= when.time() < self.close_time)


def round_to_tick(price: float, tick: float = 0.05, side: float = 0.0) -> float:
    """Round to the exchange tick; up for buys (side > 0), down for sells."""
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if side > 0 else math.floor(steps + 1e-9) if side < 0 \
        else round(steps)
    return round(steps * tick, 2)


# ---------------------------------------------------------------- free data
YAHOO_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
YAHOO_INTERVALS = {"day": "1d", "60minute": "60m", "30minute": "30m", "15minute": "15m",
                   "5minute": "5m", "minute": "1m"}
#: Yahoo only serves limited history for intraday intervals.
YAHOO_RANGE = {"day": "5y", "60minute": "730d", "30minute": "60d", "15minute": "60d",
               "5minute": "60d", "minute": "7d"}


def yahoo_ticker(symbol: str) -> str:
    ex, ts = parse_symbol(symbol)
    if ex not in YAHOO_SUFFIX:
        raise ValueError(f"Yahoo data only covers NSE/BSE, not {ex}")
    return ts + YAHOO_SUFFIX[ex]


def fetch_yahoo(symbol: str, interval: str = "day", session=None) -> pd.DataFrame:
    """Free OHLCV from Yahoo's public chart endpoint (no yfinance needed).

    Daily bars are adjusted for splits/dividends. Data can be delayed; good
    for daily strategies, less so for fast intraday ones.
    """
    import requests

    sess = session or requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker(symbol)}"
    params = {"interval": YAHOO_INTERVALS[interval], "range": YAHOO_RANGE[interval],
              "includeAdjustedClose": "true"}
    resp = sess.get(url, params=params, timeout=20,
                    headers={"User-Agent": "Mozilla/5.0 (algo-agent)"})
    resp.raise_for_status()
    return parse_yahoo_chart(resp.json())


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise ValueError(f"Yahoo returned no data: {err}")
    r = result[0]
    q = r["indicators"]["quote"][0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(IST)
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                      index=idx, dtype=float)
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        factor = pd.Series(adj, index=idx, dtype=float) / df["close"]
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    df = df.dropna(subset=["open", "high", "low", "close"])
    if r.get("meta", {}).get("dataGranularity") == "1d":
        # Daily bars: index by calendar date (09:15 IST session start).
        df.index = pd.DatetimeIndex(df.index.date).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
    df["volume"] = df["volume"].fillna(0.0)
    return df[~df.index.duplicated(keep="last")]


def completed_bars(df: pd.DataFrame, interval: str, now: dt.datetime | None = None) -> pd.DataFrame:
    """Drop the still-forming bar."""
    if df.empty:
        return df
    now = now or now_ist()
    if interval == "day":
        today = now.astimezone(IST).date()
        after_close = now.astimezone(IST).time() >= dt.time(15, 35)
        keep = [(d.date() < today) or (d.date() == today and after_close) for d in df.index]
        return df[keep]
    minutes = {"60minute": 60, "30minute": 30, "15minute": 15, "5minute": 5, "minute": 1}[interval]
    ends = df.index + pd.Timedelta(minutes=minutes)
    return df[ends <= pd.Timestamp(now)]
