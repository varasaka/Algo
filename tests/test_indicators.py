import numpy as np
import pandas as pd

from algo_agent import indicators as ind


def test_sma_matches_manual():
    s = pd.Series(np.arange(10, dtype=float))
    out = ind.sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == 8.0


def test_rsi_bounds_and_extremes(prices):
    r = ind.rsi(prices["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.arange(50, 1, -1, dtype=float))
    assert ind.rsi(down).iloc[-1] < 1.0


def test_atr_positive(prices):
    a = ind.atr(prices).dropna()
    assert (a > 0).all()


def test_indicators_are_causal(prices):
    """Value at t must not change when future rows are appended."""
    cut = 250
    for fn in (lambda d: ind.atr(d), lambda d: ind.rsi(d["close"]),
               lambda d: ind.macd(d["close"])["hist"], lambda d: ind.realized_vol(d["close"])):
        full = fn(prices)
        part = fn(prices.iloc[:cut])
        assert np.isclose(full.iloc[cut - 1], part.iloc[-1])
