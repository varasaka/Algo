from __future__ import annotations

from typing import Any

from .base import Strategy
from .library import BollingerReversion, Ensemble, MomentumTrend, RSIMeanReversion, SMACrossover

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (SMACrossover, RSIMeanReversion, BollingerReversion, MomentumTrend, Ensemble)
}
ALIASES = {"sma": "sma_crossover", "rsi": "rsi_mean_reversion", "bollinger": "bollinger_reversion",
           "macd": "momentum"}


def create_strategy(name: str, **params: Any) -> Strategy:
    """Build a strategy by registry name (or alias) with optional params.

    For ``ensemble``, pass ``members=[{"name": ..., "params": {...}}, ...]`` and
    optional ``weights``.
    """
    key = ALIASES.get(name, name)
    if key not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(REGISTRY)}")
    if key == "ensemble":
        members_cfg = params.pop("members", None)
        weights = params.pop("weights", None)
        members = None
        if members_cfg:
            members = [create_strategy(m["name"], **m.get("params", {})) for m in members_cfg]
        return Ensemble(members=members, weights=weights, **params)
    return REGISTRY[key](**params)


__all__ = ["Strategy", "SMACrossover", "RSIMeanReversion", "BollingerReversion",
           "MomentumTrend", "Ensemble", "REGISTRY", "create_strategy"]
