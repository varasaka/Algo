import pandas as pd
import pytest

from algo_agent.broker import PaperBroker
from algo_agent.data import Bar

T = pd.Timestamp


def bar(day, o, h=None, l=None, c=None):
    c = o if c is None else c
    return Bar(T(f"2024-01-{day:02d}"), o, h or max(o, c), l or min(o, c), c)


def test_order_fills_at_next_open_with_costs():
    b = PaperBroker(cash=10_000, commission_bps=10, slippage_bps=50)
    b.submit_order("X", 10)
    assert b.pending and b.cash == 10_000  # nothing happens until the next bar
    fills = b.process_bars({"X": bar(2, 100, c=110)})
    assert fills[0].price == pytest.approx(100.5)        # +50bp slippage
    assert fills[0].commission == pytest.approx(1.005)   # 10bp of 1005
    assert b.cash == pytest.approx(10_000 - 1005 - 1.005)
    assert b.equity() == pytest.approx(b.cash + 10 * 110)


def test_round_trip_records_trade_and_flip():
    b = PaperBroker(cash=10_000, commission_bps=0, slippage_bps=0)
    b.submit_order("X", 10)
    b.process_bars({"X": bar(2, 100)})
    b.submit_order("X", -15)  # close 10 long, open 5 short
    b.process_bars({"X": bar(3, 110)})
    assert len(b.trades) == 1
    t = b.trades[0]
    assert (t.side, t.qty, t.pnl) == ("long", 10, pytest.approx(100.0))
    assert b.positions["X"].qty == -5 and b.positions["X"].avg_price == 110
    b.submit_order("X", 5)
    b.process_bars({"X": bar(4, 100)})
    assert b.trades[-1].side == "short" and b.trades[-1].pnl == pytest.approx(50.0)
    assert b.equity() == pytest.approx(10_150)


def test_fractional_truncation_and_missing_bar():
    b = PaperBroker(cash=1_000)
    assert b.submit_order("X", 0.7).qty == 0 and not b.pending
    b.submit_order("X", 2.9)
    b.process_bars({"Y": bar(2, 10)})
    assert b.pending  # no X bar yet, order waits
    b.process_bars({"X": bar(3, 10)})
    assert b.positions["X"].qty == 2
