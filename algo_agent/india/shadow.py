"""Shadow (dry-run) broker: real account data, orders recorded but never sent."""

from __future__ import annotations

import itertools
import logging

import pandas as pd

from ..broker import Account, Broker, Order

log = logging.getLogger(__name__)


class ShadowBroker(Broker):
    """Wraps a real broker. Reads go to the real account; ``submit_order`` only
    records what *would* have been sent. Use this to verify the agent against
    your real holdings before switching to live."""

    def __init__(self, real: Broker) -> None:
        self.real = real
        self.would_send: list[Order] = []
        self._ids = itertools.count(1)

    def account(self) -> Account:
        return self.real.account()

    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None,
                     ref_price: float | None = None) -> Order:
        order = Order(id=f"shadow-{next(self._ids)}", symbol=symbol, qty=float(int(qty)),
                      reason=reason, created=timestamp)
        if order.qty:
            log.info("SHADOW (not sent): %s %+d %s @~%s", reason, order.qty, symbol, ref_price)
            self.would_send.append(order)
        return order

    def cancel_all(self) -> None:
        pass  # nothing was sent, so nothing to cancel
