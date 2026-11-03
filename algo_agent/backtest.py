"""Event-driven backtester.

Replays bars through the real ``TradingAgent`` and a ``PaperBroker``, so the
code path is identical to paper/live trading. Per time step:

    broker fills orders from the previous step at this bar's open
    -> broker marks positions to this bar's close
    -> agent sees the closed bar and submits orders for the next step
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .agent import TradingAgent, precompute_features
from .broker import Fill, PaperBroker, Trade
from .data import Bar
from .metrics import compute_metrics, infer_periods_per_year
from .risk import RiskConfig, RiskManager
from .strategies import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.DataFrame
    trades: list[Trade]
    fills: list[Fill]
    journal: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def journal_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.journal)


def run_backtest(data: dict[str, pd.DataFrame], strategy: Strategy,
                 risk: RiskConfig | None = None, initial_cash: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 2.0,
                 liquidate_at_end: bool = True,
                 trade_start: pd.Timestamp | str | None = None,
                 precompute: bool = True) -> BacktestResult:
    """Backtest ``strategy`` on ``data``.

    Bars before ``trade_start`` are only *observed* (they warm up the agent's
    indicators without trading); results cover ``trade_start`` onwards.
    """
    if not data:
        raise ValueError("no data to backtest")
    symbols = list(data)
    full_index = pd.DatetimeIndex(sorted(set(
        itertools.chain.from_iterable(df.index for df in data.values()))))
    ppy = infer_periods_per_year(full_index)
    if trade_start is not None:
        trade_start = pd.Timestamp(trade_start)
        index = full_index[full_index >= trade_start]
        warmup_index = full_index[full_index < trade_start]
    else:
        index, warmup_index = full_index, full_index[:0]
    if len(index) == 0:
        raise ValueError("no bars on or after trade_start")

    broker = PaperBroker(cash=initial_cash, commission_bps=commission_bps,
                         slippage_bps=slippage_bps)
    rm = RiskManager(risk)
    features = ({s: precompute_features(strategy, df, rm, ppy) for s, df in data.items()}
                if precompute else None)
    agent = TradingAgent(strategy, broker, symbols, rm, periods_per_year=ppy, features=features)

    # Pre-convert rows to Bars once: much faster than DataFrame lookups per step.
    bars_by_time: dict[pd.Timestamp, dict[str, Bar]] = {}
    for sym, df in data.items():
        for ts, o, h, l, c, v in zip(df.index, df["open"], df["high"], df["low"],
                                     df["close"], df["volume"]):
            bars_by_time.setdefault(ts, {})[sym] = Bar(ts, o, h, l, c, v)

    for ts in warmup_index:
        agent.observe(bars_by_time.get(ts, {}))

    equity, pos_rows = [], []
    for ts in index:
        bars = bars_by_time.get(ts, {})
        broker.process_bars(bars)
        agent.on_bar(ts, bars)
        equity.append(broker.equity())
        pos_rows.append({s: broker.positions.get(s).qty if s in broker.positions else 0.0
                         for s in symbols})

    if liquidate_at_end:
        last_ts = index[-1]
        for sym, pos in list(broker.positions.items()):
            if pos.qty:
                price = broker.prices[sym]
                broker._fill(broker.submit_order(sym, -pos.qty, "end_of_backtest", last_ts),
                             price, last_ts)
        broker.pending.clear()
        equity[-1] = broker.equity()

    eq = pd.Series(equity, index=index, name="equity")
    positions = pd.DataFrame(pos_rows, index=index)
    exposure = (positions.abs().sum(axis=1) > 0).astype(float)
    metrics = compute_metrics(eq, broker.trades, ppy, exposure)
    metrics["benchmark_return"] = buy_and_hold_return(
        {s: df[df.index >= index[0]] for s, df in data.items()})
    return BacktestResult(eq, positions, broker.trades, broker.fills, agent.journal, metrics)


def buy_and_hold_return(data: dict[str, pd.DataFrame]) -> float:
    """Equal-weight buy-and-hold return over each symbol's full history."""
    rets = [df["close"].iloc[-1] / df["open"].iloc[0] - 1 for df in data.values() if len(df)]
    return float(sum(rets) / len(rets)) if rets else 0.0


def split_point(data: dict[str, pd.DataFrame], train_frac: float) -> pd.Timestamp:
    """Timestamp separating the first ``train_frac`` of bars from the rest."""
    index = pd.DatetimeIndex(sorted(set(
        itertools.chain.from_iterable(df.index for df in data.values()))))
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    return index[int(len(index) * train_frac)]


def slice_until(data: dict[str, pd.DataFrame], end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {s: df[df.index < end] for s, df in data.items()}
