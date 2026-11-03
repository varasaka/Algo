import pandas as pd
import pytest

from algo_agent.data import synthetic_ohlcv


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic_ohlcv(n_bars=400, seed=3)


def flat_bars(n: int = 100, price: float = 100.0, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
                         "volume": 1_000.0}, index=idx)
