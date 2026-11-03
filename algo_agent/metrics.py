"""Performance statistics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .broker import Trade


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Guess bars/year from median bar spacing (daily -> 252, hourly -> ~1764...)."""
    if len(index) < 3:
        return 252
    step = pd.Series(index).diff().median()
    if step >= pd.Timedelta(days=28):
        return 12
    if step >= pd.Timedelta(days=5):
        return 52
    if step >= pd.Timedelta(hours=20):
        return 252
    # intraday: assume 6.5 trading hours a day
    per_day = pd.Timedelta(hours=6.5) / step
    return int(round(252 * max(per_day, 1)))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min()) if len(equity) else 0.0


def compute_metrics(equity: pd.Series, trades: Iterable[Trade] = (),
                    periods_per_year: int | None = None,
                    exposure: pd.Series | None = None) -> dict[str, float]:
    equity = equity.dropna()
    trades = list(trades)
    ppy = periods_per_year or infer_periods_per_year(equity.index)
    out: dict[str, float] = {}
    if len(equity) < 2:
        return {"total_return": 0.0}
    rets = equity.pct_change().dropna()
    years = len(rets) / ppy
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    out["total_return"] = float(total)
    out["cagr"] = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0
    vol = rets.std(ddof=1)
    out["volatility"] = float(vol * math.sqrt(ppy))
    out["sharpe"] = float(rets.mean() / vol * math.sqrt(ppy)) if vol > 0 else 0.0
    downside = rets[rets < 0]
    dvol = math.sqrt((downside**2).sum() / len(rets)) if len(rets) else 0.0
    out["sortino"] = float(rets.mean() / dvol * math.sqrt(ppy)) if dvol > 0 else 0.0
    mdd = max_drawdown(equity)
    out["max_drawdown"] = mdd
    out["calmar"] = float(out["cagr"] / abs(mdd)) if mdd < 0 else 0.0
    if exposure is not None and len(exposure):
        out["exposure"] = float((exposure.abs() > 1e-9).mean())
    out["trades"] = float(len(trades))
    if trades:
        pnls = np.array([t.pnl for t in trades])
        wins, losses = pnls[pnls > 0], pnls[pnls < 0]
        out["win_rate"] = float(len(wins) / len(pnls))
        out["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) else float("inf")
        out["avg_trade_return"] = float(np.mean([t.return_pct for t in trades]))
        holds = [(t.exit_time - t.entry_time) for t in trades if t.entry_time is not None]
        if holds:
            out["avg_hold_days"] = float(np.mean([h / pd.Timedelta(days=1) for h in holds]))
    return out


PERCENT_KEYS = {"total_return", "cagr", "volatility", "max_drawdown", "exposure", "win_rate",
                "avg_trade_return", "benchmark_return"}


def format_metrics(metrics: dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        if k in PERCENT_KEYS:
            s = f"{v:9.2%}"
        elif k == "trades":
            s = f"{int(v):9d}"
        else:
            s = f"{v:9.2f}"
        lines.append(f"  {k:<18}{s}")
    return "\n".join(lines)
