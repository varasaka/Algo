import pandas as pd
import pytest

from algo_agent.agent import TradingAgent
from algo_agent.backtest import run_backtest
from algo_agent.broker import PaperBroker
from algo_agent.data import Bar, synthetic_ohlcv
from algo_agent.live import LiveRunner, ReplayFeed
from algo_agent.risk import RiskConfig, RiskManager
from algo_agent.strategies import Strategy, create_strategy

from .conftest import flat_bars


class AlwaysLong(Strategy):
    name = "always_long"

    @property
    def warmup(self):
        return 1

    def generate_signals(self, df):
        return pd.Series(1.0, index=df.index)


def test_flat_market_loses_only_costs():
    data = {"X": flat_bars(200)}
    res = run_backtest(data, AlwaysLong(), RiskConfig(target_vol=None, risk_per_trade=None),
                       commission_bps=0, slippage_bps=0)
    assert res.equity.iloc[-1] == pytest.approx(100_000)
    assert res.positions["X"].iloc[5] > 0


@pytest.mark.parametrize("name", ["sma_crossover", "rsi_mean_reversion", "momentum", "ensemble"])
def test_precomputed_matches_bar_by_bar(name):
    """Fast backtest path == the exact path live trading uses."""
    data = {"A": synthetic_ohlcv(400, seed=1), "B": synthetic_ohlcv(400, seed=2)}
    kw = dict(risk=RiskConfig(), commission_bps=1, slippage_bps=2)
    fast = run_backtest(data, create_strategy(name), precompute=True, **kw)
    # bar-by-bar recomputes on a bounded lookback window; EMA-based values differ only
    # negligibly from the full-history ones, so equity must agree to tight tolerance
    slow = run_backtest(data, create_strategy(name), precompute=False, **kw)
    pd.testing.assert_series_equal(fast.equity, slow.equity, rtol=1e-6)


def test_replay_runner_matches_backtest():
    data = {"A": synthetic_ohlcv(400, seed=5)}
    strat = create_strategy("sma", fast=10, slow=30)
    warm = 100
    bt = run_backtest(data, strat, RiskConfig(), trade_start=data["A"].index[warm],
                      liquidate_at_end=False, precompute=False)
    broker = PaperBroker()
    agent = TradingAgent(create_strategy("sma", fast=10, slow=30), broker, ["A"],
                         RiskManager(RiskConfig()))
    LiveRunner(agent, ReplayFeed(data, warmup_bars=warm), poll_seconds=0).run()
    assert broker.equity() == pytest.approx(bt.equity.iloc[-1])


def test_stop_loss_exits_and_cools_down():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    close = [100.0] * 30 + [80.0] * 10   # crash on bar 30
    df = pd.DataFrame({"open": close, "high": [c + 1 for c in close],
                       "low": [c - 1 for c in close], "close": close, "volume": 1.0}, index=idx)
    cfg = RiskConfig(target_vol=None, risk_per_trade=None, stop_atr_mult=3, cooldown_bars=3,
                     max_drawdown=None, daily_loss_limit=None)
    res = run_backtest({"X": df}, AlwaysLong(), cfg, commission_bps=0, slippage_bps=0,
                       liquidate_at_end=False)
    reasons = [e["reason"] for e in res.journal if e["action"] == "exit"]
    assert any(r.startswith("stop_hit") for r in reasons)
    # flat during cooldown, then re-enters
    assert res.positions["X"].iloc[32] == 0
    assert res.positions["X"].iloc[-1] > 0


def test_max_drawdown_halts_agent():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    close = [100.0 - i for i in range(60)]
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1.0}, index=idx)
    cfg = RiskConfig(target_vol=None, risk_per_trade=None, stop_atr_mult=0, max_drawdown=0.1,
                     daily_loss_limit=None)
    broker = PaperBroker(commission_bps=0, slippage_bps=0)
    agent = TradingAgent(AlwaysLong(), broker, ["X"], RiskManager(cfg))
    for ts, row in df.iterrows():
        broker.process_bars({"X": Bar.from_row(ts, row)})
        agent.on_bar(ts, {"X": Bar.from_row(ts, row)})
    assert agent.halted and "max_drawdown" in agent.halt_reason
    assert broker.positions["X"].qty == 0
    assert 1 - broker.equity() / 100_000 < 0.2  # loss contained near the 10% limit


def test_gross_exposure_cap():
    data = {s: flat_bars(50) for s in "ABCD"}
    cfg = RiskConfig(target_vol=None, risk_per_trade=None, max_position_pct=1.0,
                     max_gross_exposure=0.5)
    res = run_backtest(data, AlwaysLong(), cfg, commission_bps=0, slippage_bps=0,
                       liquidate_at_end=False)
    gross = (res.positions.iloc[-1] * 100).sum() / res.equity.iloc[-1]
    assert gross == pytest.approx(0.5, abs=0.01)
