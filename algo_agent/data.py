"""Market data: the Bar type and historical data sources.

All sources return a DataFrame indexed by a tz-naive or tz-aware
DatetimeIndex with lower-case columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_row(cls, timestamp, row) -> "Bar":
        return cls(
            timestamp=pd.Timestamp(timestamp),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, require OHLC, sort by time, drop bad rows."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("adj close", "adj_close") for c in df.columns]
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with a date/time column (first column if none is named so)."""
    raw = pd.read_csv(path)
    lower = {c.lower(): c for c in raw.columns}
    time_col = next(
        (lower[c] for c in ("timestamp", "datetime", "date", "time") if c in lower),
        raw.columns[0],
    )
    raw[time_col] = pd.to_datetime(raw[time_col])
    return normalize_ohlcv(raw.set_index(time_col))


def synthetic_ohlcv(
    n_bars: int = 1500,
    start_price: float = 100.0,
    seed: int | None = 7,
    start: str = "2018-01-01",
    freq: str = "B",
    annual_vol: float = 0.25,
    regime_length: int = 120,
) -> pd.DataFrame:
    """Generate a realistic-looking daily OHLCV series.

    Prices follow geometric Brownian motion whose drift switches between bull,
    bear and sideways regimes, which gives trend and mean-reversion strategies
    something to work with. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    drifts = np.array([0.35, -0.30, 0.0])
    n_regimes = int(np.ceil(n_bars / regime_length))
    regime = np.repeat(rng.integers(0, len(drifts), n_regimes), regime_length)[:n_bars]
    vol = annual_vol * np.exp(rng.normal(0, 0.15, n_bars))
    log_ret = (drifts[regime] - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n_bars)
    close = start_price * np.exp(np.cumsum(log_ret))
    prev_close = np.concatenate([[start_price], close[:-1]])
    gap = rng.normal(0, 0.25, n_bars) * vol * np.sqrt(dt)
    open_ = prev_close * np.exp(gap)
    intrabar = np.abs(rng.normal(0, 0.6, (2, n_bars))) * vol * np.sqrt(dt)
    high = np.maximum(open_, close) * np.exp(intrabar[0])
    low = np.minimum(open_, close) * np.exp(-intrabar[1])
    volume = rng.lognormal(mean=13, sigma=0.4, size=n_bars).round()
    index = pd.date_range(start=start, periods=n_bars, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def load_yahoo(symbol: str, start: str | None = None, end: str | None = None,
               interval: str = "1d") -> pd.DataFrame:
    """Download bars via yfinance (optional dependency: ``pip install yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("yfinance is not installed; run `pip install yfinance`") from exc
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return normalize_ohlcv(df)


def load_data(source: str, symbols: list[str], *, start: str | None = None,
              end: str | None = None, data_dir: str | None = None,
              n_bars: int = 1500, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Load one DataFrame per symbol from ``synthetic``, ``csv`` or ``yahoo``."""
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if source == "synthetic":
            df = synthetic_ohlcv(n_bars=n_bars, seed=seed + i, start=start or "2018-01-01")
        elif source == "csv":
            base = Path(data_dir or "data")
            path = base / f"{sym}.csv"
            if not path.exists():
                raise FileNotFoundError(f"expected CSV for {sym} at {path}")
            df = load_csv(path)
        elif source == "yahoo":
            df = load_yahoo(sym, start=start, end=end)
        else:
            raise ValueError(f"unknown data source: {source!r}")
        if start and source != "synthetic":
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        out[sym] = df
    return out
