"""Risk management: position sizing, stops and portfolio-level circuit breakers."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class RiskConfig:
    #: Largest |position value| per symbol as a fraction of equity.
    max_position_pct: float = 1.0
    #: Cap on sum of |position values| / equity across all symbols.
    max_gross_exposure: float = 1.0
    #: Annualised volatility target per symbol budget (None/0 disables scaling).
    target_vol: float | None = 0.20
    #: Equity fraction risked per position if the stop is hit (None/0 disables).
    risk_per_trade: float | None = 0.02
    #: Trailing stop distance in ATRs (0 disables stops).
    stop_atr_mult: float = 3.0
    #: ATR lookback used for stops and risk sizing.
    atr_window: int = 14
    #: Realised-volatility lookback for vol targeting.
    vol_window: int = 20
    #: Halt trading and flatten if equity falls this far from its peak.
    max_drawdown: float | None = 0.25
    #: Flatten and block new entries for the rest of the day after this loss.
    daily_loss_limit: float | None = 0.05
    #: Ignore rebalances smaller than this fraction of equity (reduces churn).
    rebalance_threshold: float = 0.05
    #: Bars to wait before re-entering a symbol after a stop-out.
    cooldown_bars: int = 5
    #: Permit short positions (strategy signals < 0).
    allow_short: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RiskConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown risk settings: {sorted(unknown)}")
        return cls(**d)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def target_weight(self, exposure: float, realized_vol: float | None, n_symbols: int) -> float:
        """Signed fraction of equity to hold, before the stop-based cap."""
        c = self.config
        if not c.allow_short:
            exposure = max(exposure, 0.0)
        if exposure == 0.0:
            return 0.0
        budget = c.max_gross_exposure / max(n_symbols, 1)
        if c.target_vol and realized_vol and realized_vol > 0 and math.isfinite(realized_vol):
            budget *= c.target_vol / realized_vol
        budget = min(budget, c.max_position_pct)
        return exposure * budget

    def target_quantity(self, exposure: float, price: float, equity: float,
                        atr: float | None, realized_vol: float | None,
                        n_symbols: int) -> float:
        """Signed number of units to hold."""
        if price <= 0 or equity <= 0:
            return 0.0
        weight = self.target_weight(exposure, realized_vol, n_symbols)
        qty = weight * equity / price
        c = self.config
        if qty and c.risk_per_trade and c.stop_atr_mult and atr and atr > 0:
            max_qty = c.risk_per_trade * equity / (c.stop_atr_mult * atr)
            qty = math.copysign(min(abs(qty), max_qty), qty)
        return qty

    def initial_stop(self, side: float, price: float, atr: float | None) -> float | None:
        if not self.config.stop_atr_mult or not atr or not math.isfinite(atr):
            return None
        return price - side * self.config.stop_atr_mult * atr

    def trail_stop(self, side: float, current: float | None, price: float,
                   atr: float | None) -> float | None:
        """Ratchet a stop in the trade's favour only."""
        fresh = self.initial_stop(side, price, atr)
        if current is None:
            return fresh
        if fresh is None:
            return current
        return max(current, fresh) if side > 0 else min(current, fresh)

    @staticmethod
    def stop_hit(side: float, stop: float | None, low: float, high: float) -> bool:
        if stop is None:
            return False
        return low <= stop if side > 0 else high >= stop
