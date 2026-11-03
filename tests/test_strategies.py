import numpy as np
import pandas as pd
import pytest

from algo_agent.strategies import REGISTRY, create_strategy


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_signals_bounded_and_aligned(name, prices):
    s = create_strategy(name)
    sig = s.generate_signals(prices)
    assert sig.index.equals(prices.index)
    assert sig.fillna(0).between(-1, 1).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_lookahead(name, prices):
    """Signal at bar t is identical whether or not later bars exist."""
    s = create_strategy(name)
    full = s.generate_signals(prices).fillna(0)
    for cut in (260, 300, 399):
        assert full.iloc[cut - 1] == pytest.approx(s.signal(prices.iloc[:cut]), abs=1e-9)


def test_sma_crossover_direction():
    up = pd.DataFrame({"close": np.linspace(10, 20, 100)})
    up["open"] = up["high"] = up["low"] = up["close"]
    s = create_strategy("sma", fast=5, slow=20)
    assert s.generate_signals(up).iloc[-1] == 1.0
    down = up[::-1].reset_index(drop=True)
    assert s.generate_signals(down).iloc[-1] == 0.0
    s_short = create_strategy("sma", fast=5, slow=20, allow_short=True)
    assert s_short.generate_signals(down).iloc[-1] == -1.0


def test_unknown_names_and_params():
    with pytest.raises(ValueError):
        create_strategy("nope")
    with pytest.raises(ValueError):
        create_strategy("sma", fastt=3)


def test_ensemble_from_config():
    e = create_strategy("ensemble", members=[{"name": "sma", "params": {"fast": 5, "slow": 10}},
                                             {"name": "momentum"}], weights=[3, 1])
    assert e.weights.tolist() == [0.75, 0.25]
