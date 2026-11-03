"""Strategy interface.

A strategy turns price history into a *target exposure* per bar, a float in
[-1, 1]: +1 = fully long, -1 = fully short, 0 = flat. It says nothing about
position size in shares; turning exposure into orders is the risk manager's
job. That keeps strategies small, testable and easy to combine.

Signals must only use information up to and including each bar's close. The
agent acts on the signal at the *next* bar, so there is no look-ahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    #: Registry name, set by subclasses.
    name: str = "base"

    def __init__(self, **params: Any) -> None:
        defaults = self.default_params()
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"{type(self).__name__} got unknown params: {sorted(unknown)}")
        self.params: dict[str, Any] = {**defaults, **params}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @property
    def warmup(self) -> int:
        """Bars of history needed before the signal is meaningful."""
        return 50

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target exposure in [-1, 1] for every bar of ``df``."""

    def signal(self, df: pd.DataFrame) -> float:
        """Target exposure for the latest bar (0.0 during warm-up)."""
        if len(df) < self.warmup:
            return 0.0
        value = self.generate_signals(df).iloc[-1]
        if pd.isna(value):
            return 0.0
        return float(max(-1.0, min(1.0, value)))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
