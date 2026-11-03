"""The trading agent: observe bars -> decide target positions -> act via a broker.

The same ``TradingAgent`` drives backtests, paper trading and live trading;
only the broker and the source of bars change. On each bar the agent:

1. appends the bar to its rolling history,
2. reads the account and checks circuit breakers (max drawdown, daily loss),
3. for each symbol: checks the trailing stop, asks the strategy for a target
   exposure, and asks the risk manager to turn that into a position size,
4. scales everything down if gross exposure exceeds the cap,
5. submits the difference between target and current positions as orders,
6. writes every decision, with its reason, to a journal.

Orders are filled by the broker later (next bar's open in simulation), so
decisions only ever use data available at the bar's close.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ind
from .broker import Broker
from .data import OHLCV, Bar
from .risk import RiskManager
from .strategies import Strategy

log = logging.getLogger(__name__)


@dataclass
class SymbolState:
    history: deque = field(default_factory=deque)
    stop: float | None = None
    side: float = 0.0           # side of the position the stop protects
    cooldown: int = 0
    last_signal: float = 0.0


class TradingAgent:
    def __init__(self, strategy: Strategy, broker: Broker, symbols: list[str],
                 risk: RiskManager | None = None, lookback: int | None = None,
                 periods_per_year: int = 252,
                 features: dict[str, pd.DataFrame] | None = None) -> None:
        if not symbols:
            raise ValueError("agent needs at least one symbol")
        self.strategy = strategy
        self.broker = broker
        self.symbols = list(symbols)
        self.risk = risk or RiskManager()
        cfg = self.risk.config
        # Enough history for the strategy and the risk indicators, with slack
        # so EMA-style indicators converge; bounded so each bar costs O(1).
        need = max(strategy.warmup, cfg.atr_window, cfg.vol_window) + 1
        self.lookback = lookback or max(3 * need, 250)
        self.periods_per_year = periods_per_year
        self.state = {s: SymbolState(history=deque(maxlen=self.lookback)) for s in self.symbols}
        #: Optional precomputed per-symbol columns ``signal, atr, vol`` (backtests).
        #: Values must only depend on data up to each row's timestamp.
        self.features = features or {}
        self.journal: list[dict[str, Any]] = []
        self.peak_equity: float | None = None
        self.halted = False
        self.halt_reason = ""
        self._day: Any = None
        self._day_start_equity: float | None = None
        self._day_blocked = False

    # ------------------------------------------------------------------ API
    def warm_start(self, data: dict[str, pd.DataFrame]) -> None:
        """Preload history without trading (used when going live)."""
        for sym, df in data.items():
            if sym in self.state:
                for ts, row in df.tail(self.lookback).iterrows():
                    self.state[sym].history.append(Bar.from_row(ts, row))

    def observe(self, bars: dict[str, Bar]) -> None:
        """Record bars without making decisions (warm-up)."""
        for sym, bar in bars.items():
            if sym in self.state:
                self.state[sym].history.append(bar)

    def on_bar(self, timestamp: pd.Timestamp, bars: dict[str, Bar]) -> list[dict[str, Any]]:
        """Process one time step. Returns the journal entries it produced."""
        self.observe(bars)
        self._consume_fills()
        start = len(self.journal)

        account = self.broker.account()
        equity = account.equity
        # Size with the allocated capital if it is smaller than the account.
        sizing_equity = min(equity, account.capital) if account.capital else equity
        if self._check_circuit_breakers(timestamp, equity):
            self._flatten(timestamp, account.positions, self.halt_reason or "daily_loss_limit")
            return self.journal[start:]

        targets: dict[str, float] = {}
        reasons: dict[str, str] = {}
        prices: dict[str, float] = {}
        atrs: dict[str, float | None] = {}
        n_active = sum(1 for s in self.symbols if self.state[s].history)

        for sym in self.symbols:
            st = self.state[sym]
            if sym not in bars or not st.history:
                continue  # no new information: leave the position alone
            bar = bars[sym]
            signal, atr, vol = self._features(sym, bar.timestamp)
            prices[sym], atrs[sym] = bar.close, atr
            held = account.positions.get(sym, 0.0)

            if st.cooldown > 0:
                st.cooldown -= 1

            if held and self.risk.stop_hit(math.copysign(1, held), st.stop, bar.low, bar.high):
                targets[sym], reasons[sym] = 0.0, f"stop_hit@{st.stop:.4f}"
                st.cooldown = self.risk.config.cooldown_bars
                st.stop, st.side = None, 0.0
                continue

            st.last_signal = signal
            if self._day_blocked and abs(signal) > 0 and not held:
                targets[sym], reasons[sym] = 0.0, "blocked:daily_loss"
                continue
            if st.cooldown > 0 and not held:
                targets[sym], reasons[sym] = 0.0, f"cooldown({st.cooldown})"
                continue
            qty = self.risk.target_quantity(signal, bar.close, sizing_equity, atr, vol, n_active)
            targets[sym], reasons[sym] = qty, f"signal={signal:+.2f}"

        targets = self._apply_gross_cap(targets, prices, sizing_equity)
        self._execute(timestamp, targets, reasons, prices, atrs, account.positions, sizing_equity)
        return self.journal[start:]

    # ------------------------------------------------------------ internals
    def _features(self, sym: str, ts: pd.Timestamp) -> tuple[float, float | None, float | None]:
        """(signal, atr, realised vol) as of the close of bar ``ts``."""
        pre = self.features.get(sym)
        if pre is not None and ts in pre.index:
            row = pre.loc[ts]
            return (float(row["signal"]), _clean(row["atr"]), _clean(row["vol"]))
        df = self._frame(sym)
        cfg = self.risk.config
        atr = _last(ind.atr(df, cfg.atr_window))
        vol = _last(ind.realized_vol(df["close"], cfg.vol_window, self.periods_per_year))
        return self.strategy.signal(df), atr, vol

    def _frame(self, sym: str) -> pd.DataFrame:
        hist = self.state[sym].history
        idx = pd.DatetimeIndex([b.timestamp for b in hist])
        values = np.array([[b.open, b.high, b.low, b.close, b.volume] for b in hist], dtype=float)
        return pd.DataFrame(values, index=idx, columns=OHLCV)

    def _consume_fills(self) -> None:
        for fill in self.broker.drain_fills():
            self.journal.append({"time": fill.timestamp, "symbol": fill.symbol, "action": "fill",
                                 "qty": fill.qty, "price": round(fill.price, 4),
                                 "reason": fill.reason})

    def _check_circuit_breakers(self, ts: pd.Timestamp, equity: float) -> bool:
        cfg = self.risk.config
        if self.halted:
            return True
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)
        day = pd.Timestamp(ts).date()
        if day != self._day:
            self._day, self._day_start_equity, self._day_blocked = day, equity, False
        if cfg.max_drawdown and self.peak_equity > 0:
            dd = 1 - equity / self.peak_equity
            if dd >= cfg.max_drawdown:
                self.halted = True
                self.halt_reason = f"max_drawdown({dd:.1%})"
                log.warning("circuit breaker: %s - flattening and halting", self.halt_reason)
                return True
        if (cfg.daily_loss_limit and self._day_start_equity and not self._day_blocked
                and equity <= self._day_start_equity * (1 - cfg.daily_loss_limit)):
            self._day_blocked = True
            log.warning("daily loss limit hit - flattening until next session")
            return True
        return False

    def flatten(self, ts: pd.Timestamp, reason: str = "manual_flatten") -> None:
        """Close every position in the agent's symbols."""
        positions = self.broker.account().positions
        self._flatten(ts, {s: q for s, q in positions.items() if s in self.state}, reason)

    def _flatten(self, ts: pd.Timestamp, positions: dict[str, float], reason: str) -> None:
        self.broker.cancel_all()
        for sym, qty in positions.items():
            if sym not in self.state:
                continue  # never touch positions the agent doesn't manage
            if qty:
                last = self.state[sym].history[-1].close if self.state[sym].history else None
                self.broker.submit_order(sym, -qty, reason=reason, timestamp=ts, ref_price=last)
                self.journal.append({"time": ts, "symbol": sym, "action": "order", "qty": -qty,
                                     "reason": reason})
            if sym in self.state:
                self.state[sym].stop, self.state[sym].side = None, 0.0

    def _apply_gross_cap(self, targets: dict[str, float], prices: dict[str, float],
                         equity: float) -> dict[str, float]:
        cap = self.risk.config.max_gross_exposure
        gross = sum(abs(q) * prices[s] for s, q in targets.items()) / equity if equity > 0 else 0
        if gross > cap > 0:
            scale = cap / gross
            return {s: q * scale for s, q in targets.items()}
        return targets

    def _execute(self, ts, targets, reasons, prices, atrs, positions, equity) -> None:
        cfg = self.risk.config
        self.broker.cancel_all()
        for sym, target in targets.items():
            st = self.state[sym]
            held = positions.get(sym, 0.0)
            price = prices[sym]
            target = float(int(target))  # whole units
            delta = target - held
            flipping = held and target and (held > 0) != (target > 0)
            closing = held and not target
            opening = target and not held
            small = abs(delta) * price < cfg.rebalance_threshold * equity
            if delta and (closing or opening or flipping or not small):
                self.broker.submit_order(sym, delta, reason=reasons[sym], timestamp=ts,
                                         ref_price=price)
                action = "exit" if closing else "enter" if opening or flipping else "rebalance"
                self.journal.append({"time": ts, "symbol": sym, "action": action, "qty": delta,
                                     "target": target, "price": round(price, 4),
                                     "reason": reasons[sym]})
            else:
                target = held  # keep current position
            # maintain the trailing stop for whatever we'll be holding next bar
            if target:
                side = 1.0 if target > 0 else -1.0
                if side != st.side:
                    st.stop, st.side = self.risk.initial_stop(side, price, atrs[sym]), side
                else:
                    st.stop = self.risk.trail_stop(side, st.stop, price, atrs[sym])
            else:
                st.stop, st.side = None, 0.0


def precompute_features(strategy: Strategy, df: pd.DataFrame, risk: RiskManager,
                        periods_per_year: int = 252) -> pd.DataFrame:
    """Vectorised equivalent of what the agent computes bar by bar.

    Valid because every indicator is causal: the value at row *t* only uses
    rows <= *t*. ``tests/test_agent.py`` checks this equivalence.
    """
    cfg = risk.config
    sig = strategy.generate_signals(df).fillna(0.0).clip(-1.0, 1.0)
    sig.iloc[: max(strategy.warmup - 1, 0)] = 0.0
    return pd.DataFrame({
        "signal": sig,
        "atr": ind.atr(df, cfg.atr_window),
        "vol": ind.realized_vol(df["close"], cfg.vol_window, periods_per_year),
    }, index=df.index)


def _clean(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    v = series.iloc[-1]
    return None if pd.isna(v) else float(v)
