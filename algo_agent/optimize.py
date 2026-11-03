"""Parameter search with an out-of-sample check.

Fits parameters on the first part of the data and reports how the winners
do on the unseen remainder. A big gap between in-sample and out-of-sample
results is the classic sign of overfitting.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from .backtest import run_backtest, slice_until, split_point
from .risk import RiskConfig
from .strategies import create_strategy


def parse_grid(spec: str) -> dict[str, list[Any]]:
    """Parse ``"fast=10,20 slow=50,100"`` into ``{"fast": [10, 20], "slow": [50, 100]}``."""
    grid: dict[str, list[Any]] = {}
    for part in spec.split():
        key, _, values = part.partition("=")
        if not values:
            raise ValueError(f"bad grid entry {part!r}; expected key=v1,v2")
        grid[key] = [_coerce(v) for v in values.split(",")]
    return grid


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def grid_search(data: dict[str, pd.DataFrame], strategy_name: str, grid: dict[str, list[Any]],
                risk: RiskConfig | None = None, train_frac: float = 0.7,
                metric: str = "sharpe", top: int = 5, base_params: dict[str, Any] | None = None,
                **bt_kwargs: Any) -> pd.DataFrame:
    cut = split_point(data, train_frac)
    train = slice_until(data, cut)
    rows = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**(base_params or {}), **dict(zip(keys, combo))}
        try:
            strat = create_strategy(strategy_name, **params)
        except ValueError:
            continue
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue  # meaningless crossover
        res = run_backtest(train, strat, risk, **bt_kwargs)
        rows.append({**dict(zip(keys, combo)), **{f"is_{k}": v for k, v in res.metrics.items()}})
    if not rows:
        raise ValueError("grid produced no valid parameter combinations")
    table = pd.DataFrame(rows).sort_values(f"is_{metric}", ascending=False).head(top)

    oos = []
    for _, row in table.iterrows():
        params = {**(base_params or {}), **{k: _native(row[k]) for k in keys}}
        res = run_backtest(data, create_strategy(strategy_name, **params), risk,
                           trade_start=cut, **bt_kwargs)
        oos.append({f"oos_{k}": res.metrics.get(k) for k in
                    ("total_return", "sharpe", "max_drawdown", "trades")})
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(oos)], axis=1)


def _native(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v
