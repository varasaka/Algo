"""Brokers: where orders go.

``PaperBroker`` is a deterministic simulator used for backtests and paper
trading. ``AlpacaBroker`` routes orders to Alpaca's REST API (paper or live
account). Both expose the same interface so the agent never knows which one
it is talking to.

Quantities are signed: positive = buy / long, negative = sell / short.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data import Bar

log = logging.getLogger(__name__)


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    reason: str = ""
    created: pd.Timestamp | None = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    qty: float
    price: float
    commission: float
    timestamp: pd.Timestamp
    reason: str = ""


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0
    opened: pd.Timestamp | None = None


@dataclass
class Trade:
    """A completed round trip (entry to flat, or entry to flip)."""

    symbol: str
    side: str
    qty: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        sign = 1.0 if self.side == "long" else -1.0
        return sign * (self.exit_price / self.entry_price - 1.0)


@dataclass
class Account:
    cash: float
    equity: float
    positions: dict[str, float] = field(default_factory=dict)


class Broker(ABC):
    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None) -> Order: ...

    @abstractmethod
    def cancel_all(self) -> None: ...

    def position(self, symbol: str) -> float:
        return self.account().positions.get(symbol, 0.0)

    def process_bars(self, bars: dict[str, Bar]) -> list[Fill]:
        """Advance simulated time (no-op for real brokers)."""
        return []

    def mark(self, prices: dict[str, float]) -> None:
        """Update mark-to-market prices (no-op for real brokers)."""

    def drain_fills(self) -> list[Fill]:
        """Fills since the last call (real brokers may return [])."""
        return []


class PaperBroker(Broker):
    """Simulated broker.

    Market orders submitted while processing bar *t* fill at the *open* of
    bar *t+1*, adjusted by ``slippage_bps`` against the trader. Commission is
    ``commission_bps`` of notional plus ``commission_per_order``.
    """

    def __init__(self, cash: float = 100_000.0, commission_bps: float = 1.0,
                 commission_per_order: float = 0.0, slippage_bps: float = 2.0,
                 allow_fractional: bool = False) -> None:
        self.cash = float(cash)
        self.commission_bps = commission_bps
        self.commission_per_order = commission_per_order
        self.slippage_bps = slippage_bps
        self.allow_fractional = allow_fractional
        self.positions: dict[str, Position] = {}
        self.pending: list[Order] = []
        self.prices: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self._unread_fills: list[Fill] = []
        self._ids = itertools.count(1)

    # -- Broker interface -------------------------------------------------
    def account(self) -> Account:
        return Account(
            cash=self.cash,
            equity=self.equity(),
            positions={s: p.qty for s, p in self.positions.items() if p.qty != 0},
        )

    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None) -> Order:
        if not self.allow_fractional:
            qty = float(int(qty))  # truncate toward zero
        order = Order(id=str(next(self._ids)), symbol=symbol, qty=qty, reason=reason,
                      created=timestamp)
        if qty != 0:
            self.pending.append(order)
        return order

    def cancel_all(self) -> None:
        self.pending.clear()

    def drain_fills(self) -> list[Fill]:
        out, self._unread_fills = self._unread_fills, []
        return out

    def mark(self, prices: dict[str, float]) -> None:
        self.prices.update(prices)

    def process_bars(self, bars: dict[str, Bar]) -> list[Fill]:
        """Fill pending orders at this bar's open, then mark to the close."""
        new_fills: list[Fill] = []
        still_pending: list[Order] = []
        for order in self.pending:
            bar = bars.get(order.symbol)
            if bar is None:  # no data for this symbol yet: keep waiting
                still_pending.append(order)
                continue
            new_fills.append(self._fill(order, bar.open, bar.timestamp))
        self.pending = still_pending
        self.mark({s: b.close for s, b in bars.items()})
        self._unread_fills.extend(new_fills)
        return new_fills

    # -- internals ----------------------------------------------------------
    def equity(self) -> float:
        value = sum(p.qty * self.prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + value

    def _fill(self, order: Order, ref_price: float, ts: pd.Timestamp) -> Fill:
        side = 1.0 if order.qty > 0 else -1.0
        price = ref_price * (1.0 + side * self.slippage_bps / 10_000)
        notional = abs(order.qty) * price
        commission = notional * self.commission_bps / 10_000 + self.commission_per_order
        self.cash -= order.qty * price + commission
        self._update_position(order.symbol, order.qty, price, ts, order.reason)
        fill = Fill(order.id, order.symbol, order.qty, price, commission, ts, order.reason)
        self.fills.append(fill)
        return fill

    def _update_position(self, symbol: str, qty: float, price: float, ts: pd.Timestamp,
                         reason: str) -> None:
        pos = self.positions.setdefault(symbol, Position())
        if pos.qty == 0 or (pos.qty > 0) == (qty > 0):
            # opening or adding: weighted average entry
            new_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / new_qty
            if pos.qty == 0:
                pos.opened = ts
            pos.qty = new_qty
            return
        # reducing, closing or flipping
        closing = min(abs(qty), abs(pos.qty))
        sign = 1.0 if pos.qty > 0 else -1.0
        pnl = sign * closing * (price - pos.avg_price)  # gross; commissions hit cash
        self.trades.append(Trade(
            symbol=symbol, side="long" if sign > 0 else "short", qty=closing,
            entry_time=pos.opened, exit_time=ts, entry_price=pos.avg_price,
            exit_price=price, pnl=pnl, exit_reason=reason,
        ))
        remaining = pos.qty + qty
        if abs(remaining) < 1e-12:
            self.positions[symbol] = Position()
        elif (remaining > 0) == (pos.qty > 0):
            pos.qty = remaining  # partial reduce: avg price unchanged
        else:
            self.positions[symbol] = Position(qty=remaining, avg_price=price, opened=ts)


class AlpacaBroker(Broker):
    """Alpaca Trading API v2 adapter (https://alpaca.markets).

    Reads ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` from the environment.
    Uses the paper endpoint unless ``paper=False`` *and* the environment
    variable ``ALGO_AGENT_ALLOW_LIVE=1`` is set — a deliberate two-key
    safety switch against accidentally trading real money.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(self, paper: bool = True, api_key: str | None = None,
                 secret_key: str | None = None, timeout: float = 10.0) -> None:
        import requests  # optional dependency

        self._requests = requests
        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY to use AlpacaBroker")
        if not paper and os.environ.get("ALGO_AGENT_ALLOW_LIVE") != "1":
            raise RuntimeError("live trading requires ALGO_AGENT_ALLOW_LIVE=1")
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _req(self, method: str, url: str, **kwargs: Any) -> Any:
        resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else None

    def account(self) -> Account:
        acct = self._req("GET", f"{self.base_url}/v2/account")
        positions = self._req("GET", f"{self.base_url}/v2/positions")
        return Account(
            cash=float(acct["cash"]),
            equity=float(acct["equity"]),
            positions={p["symbol"]: float(p["qty"]) for p in positions},
        )

    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None) -> Order:
        qty = float(int(qty))
        if qty == 0:
            return Order(id="noop", symbol=symbol, qty=0.0, reason=reason, created=timestamp)
        body = {"symbol": symbol, "qty": str(abs(int(qty))), "side": "buy" if qty > 0 else "sell",
                "type": "market", "time_in_force": "day"}
        log.info("alpaca order %s", json.dumps(body))
        resp = self._req("POST", f"{self.base_url}/v2/orders", json=body)
        return Order(id=resp["id"], symbol=symbol, qty=qty, reason=reason, created=timestamp)

    def cancel_all(self) -> None:
        self._req("DELETE", f"{self.base_url}/v2/orders")

    def latest_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 300) -> pd.DataFrame:
        """Recent historical bars from Alpaca's market data API."""
        data = self._req("GET", f"{self.DATA_URL}/v2/stocks/{symbol}/bars",
                         params={"timeframe": timeframe, "limit": limit, "adjustment": "all", "feed": "iex",
                                 "start": (pd.Timestamp.utcnow() - pd.Timedelta(days=limit * 2)).isoformat()})
        rows = data.get("bars") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                                "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
