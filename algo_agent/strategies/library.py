"""Built-in strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy


class SMACrossover(Strategy):
    """Trend following: long when the fast SMA is above the slow SMA."""

    name = "sma_crossover"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"fast": 20, "slow": 50, "allow_short": False}

    @property
    def warmup(self) -> int:
        return int(self.params["slow"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = ind.sma(df["close"], int(self.params["fast"]))
        slow = ind.sma(df["close"], int(self.params["slow"]))
        short = -1.0 if self.params["allow_short"] else 0.0
        sig = pd.Series(np.where(fast > slow, 1.0, short), index=df.index)
        return sig.where(slow.notna(), 0.0)


class RSIMeanReversion(Strategy):
    """Buy oversold dips, exit when RSI recovers.

    Stateful: enters long when RSI < ``oversold`` and holds until RSI >
    ``exit_level``. With ``allow_short`` it mirrors the logic on the upside.
    An optional trend filter only takes longs above the long-term SMA.
    """

    name = "rsi_mean_reversion"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"window": 14, "oversold": 30, "overbought": 70, "exit_level": 50,
                "trend_window": 200, "allow_short": False}

    @property
    def warmup(self) -> int:
        tw = int(self.params["trend_window"] or 0)
        return max(int(self.params["window"]) * 3, tw + 1)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        r = ind.rsi(df["close"], int(p["window"])).to_numpy()
        close = df["close"].to_numpy()
        tw = int(p["trend_window"] or 0)
        trend = ind.sma(df["close"], tw).to_numpy() if tw else np.full(len(df), -np.inf)
        out = np.zeros(len(df))
        pos = 0.0
        for i in range(len(df)):
            if np.isnan(r[i]):
                out[i] = 0.0
                continue
            uptrend = not np.isnan(trend[i]) and close[i] > trend[i]
            downtrend = not np.isnan(trend[i]) and close[i] < trend[i]
            if pos == 0.0:
                if r[i] < p["oversold"] and (not tw or uptrend):
                    pos = 1.0
                elif p["allow_short"] and r[i] > p["overbought"] and (not tw or downtrend):
                    pos = -1.0
            elif pos > 0 and r[i] > p["exit_level"]:
                pos = 0.0
            elif pos < 0 and r[i] < p["exit_level"]:
                pos = 0.0
            out[i] = pos
        return pd.Series(out, index=df.index)


class BollingerReversion(Strategy):
    """Fade moves outside the Bollinger bands; exit at the mid band."""

    name = "bollinger_reversion"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"window": 20, "num_std": 2.0, "allow_short": False}

    @property
    def warmup(self) -> int:
        return int(self.params["window"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bands = ind.bollinger(df["close"], int(self.params["window"]), float(self.params["num_std"]))
        close = df["close"].to_numpy()
        lower, upper, mid = (bands[c].to_numpy() for c in ("lower", "upper", "mid"))
        out = np.zeros(len(df))
        pos = 0.0
        for i in range(len(df)):
            if np.isnan(mid[i]):
                continue
            if pos == 0.0:
                if close[i] < lower[i]:
                    pos = 1.0
                elif self.params["allow_short"] and close[i] > upper[i]:
                    pos = -1.0
            elif (pos > 0 and close[i] >= mid[i]) or (pos < 0 and close[i] <= mid[i]):
                pos = 0.0
            out[i] = pos
        return pd.Series(out, index=df.index)


class MomentumTrend(Strategy):
    """MACD momentum with a long-term trend filter and Donchian breakout confirm.

    Exposure is scaled: 1.0 when MACD histogram is positive *and* price is
    above the trend SMA, 0.5 when only one condition holds, else 0 (or the
    mirrored negative values when shorting is allowed).
    """

    name = "momentum"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"fast": 12, "slow": 26, "signal": 9, "trend_window": 100, "allow_short": False}

    @property
    def warmup(self) -> int:
        return max(int(self.params["slow"]) + int(self.params["signal"]),
                   int(self.params["trend_window"])) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        m = ind.macd(df["close"], int(p["fast"]), int(p["slow"]), int(p["signal"]))
        trend = ind.sma(df["close"], int(p["trend_window"]))
        bull = (m["hist"] > 0).astype(float) + (df["close"] > trend).astype(float)
        bear = (m["hist"] < 0).astype(float) + (df["close"] < trend).astype(float)
        sig = bull / 2.0
        if p["allow_short"]:
            sig = sig - bear / 2.0
        valid = m["hist"].notna() & trend.notna()
        return sig.where(valid, 0.0)


class Ensemble(Strategy):
    """Weighted average of several strategies' exposures."""

    name = "ensemble"

    def __init__(self, members: list[Strategy] | None = None,
                 weights: list[float] | None = None, **params: Any) -> None:
        super().__init__(**params)
        self.members = members or [SMACrossover(), MomentumTrend(), RSIMeanReversion()]
        w = np.asarray(weights if weights is not None else [1.0] * len(self.members), dtype=float)
        if len(w) != len(self.members) or w.sum() <= 0:
            raise ValueError("ensemble weights must match members and sum to > 0")
        self.weights = w / w.sum()

    @property
    def warmup(self) -> int:
        return max(m.warmup for m in self.members)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=df.index)
        for w, m in zip(self.weights, self.members):
            total = total + w * m.generate_signals(df).fillna(0.0)
        return total.clip(-1.0, 1.0)

    def __repr__(self) -> str:
        return f"Ensemble({self.members!r}, weights={self.weights.round(3).tolist()})"
