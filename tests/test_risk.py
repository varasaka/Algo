import pytest

from algo_agent.risk import RiskConfig, RiskManager


def test_long_only_clamps_short_signals():
    rm = RiskManager(RiskConfig(allow_short=False))
    assert rm.target_quantity(-1.0, 100, 100_000, 2.0, 0.2, 1) == 0


def test_vol_target_and_position_cap():
    rm = RiskManager(RiskConfig(target_vol=0.1, risk_per_trade=None, max_position_pct=1.0))
    # vol 0.2 vs target 0.1 -> half the budget
    assert rm.target_quantity(1.0, 100, 100_000, None, 0.2, 1) == pytest.approx(500)
    # low vol would lever up, but the position cap stops it
    assert rm.target_quantity(1.0, 100, 100_000, None, 0.01, 1) == pytest.approx(1000)
    # budget split across symbols
    assert rm.target_quantity(1.0, 100, 100_000, None, 0.1, 4) == pytest.approx(250)


def test_risk_per_trade_caps_size():
    rm = RiskManager(RiskConfig(target_vol=None, risk_per_trade=0.01, stop_atr_mult=2.0))
    # lose at most 1% of 100k = 1000 if a 2*ATR=10 stop is hit -> 100 shares
    assert rm.target_quantity(1.0, 100, 100_000, 5.0, None, 1) == pytest.approx(100)


def test_trailing_stop_only_ratchets_favourably():
    rm = RiskManager(RiskConfig(stop_atr_mult=2.0))
    s = rm.initial_stop(1, 100, 1.0)
    assert s == 98
    assert rm.trail_stop(1, s, 105, 1.0) == 103
    assert rm.trail_stop(1, 103, 101, 1.0) == 103  # never loosens
    assert rm.trail_stop(-1, 102, 95, 1.0) == 97
    assert rm.stop_hit(1, 98, low=97.5, high=101)
    assert not rm.stop_hit(-1, 97, low=90, high=96)


def test_unknown_risk_key_rejected():
    with pytest.raises(ValueError):
        RiskConfig.from_dict({"max_drawdwn": 0.1})
