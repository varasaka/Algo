"""Run the agent on a schedule against a live (or replayed) bar feed."""

from __future__ import annotations

import json
import logging
import signal
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .agent import TradingAgent
from .data import Bar

log = logging.getLogger(__name__)


class BarFeed(ABC):
    """Yields newly *completed* bars per symbol."""

    @abstractmethod
    def history(self, symbol: str) -> pd.DataFrame:
        """Recent completed bars, used to warm the agent up."""

    @abstractmethod
    def poll(self) -> dict[str, Bar]:
        """Bars completed since the last poll (at most one per symbol)."""

    def exhausted(self) -> bool:
        return False


class ReplayFeed(BarFeed):
    """Replays historical data one bar per poll - a safe way to dry-run the loop."""

    def __init__(self, data: dict[str, pd.DataFrame], warmup_bars: int = 250) -> None:
        self.data = data
        self.warmup = warmup_bars
        self.index = pd.DatetimeIndex(sorted(set().union(*(df.index for df in data.values()))))
        self.pos = min(warmup_bars, len(self.index))

    def history(self, symbol: str) -> pd.DataFrame:
        df = self.data[symbol]
        return df[df.index < self.index[self.pos]] if self.pos < len(self.index) else df

    def poll(self) -> dict[str, Bar]:
        if self.exhausted():
            return {}
        ts = self.index[self.pos]
        self.pos += 1
        return {s: Bar.from_row(ts, df.loc[ts]) for s, df in self.data.items() if ts in df.index}

    def exhausted(self) -> bool:
        return self.pos >= len(self.index)


class PollingFeed(BarFeed):
    """Polls a ``fetch(symbol) -> DataFrame`` function for recent bars.

    A bar counts as complete once ``timestamp + bar_duration <= now``, so a
    still-forming bar is never traded on.
    """

    def __init__(self, fetch: Callable[[str], pd.DataFrame], symbols: list[str],
                 bar_duration: pd.Timedelta = pd.Timedelta(days=1)) -> None:
        self.fetch = fetch
        self.symbols = symbols
        self.bar_duration = bar_duration
        self.last_seen: dict[str, pd.Timestamp] = {}

    def _completed(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
        return df[df.index + self.bar_duration <= now]

    def history(self, symbol: str) -> pd.DataFrame:
        df = self._completed(self.fetch(symbol))
        if len(df):
            self.last_seen[symbol] = df.index[-1]
        return df

    def poll(self) -> dict[str, Bar]:
        out = {}
        for sym in self.symbols:
            try:
                df = self._completed(self.fetch(sym))
            except Exception as exc:  # network hiccups shouldn't kill the loop
                log.error("fetch failed for %s: %s", sym, exc)
                continue
            if df.empty:
                continue
            ts = df.index[-1]
            if sym not in self.last_seen or ts > self.last_seen[sym]:
                self.last_seen[sym] = ts
                out[sym] = Bar.from_row(ts, df.iloc[-1])
        return out


class LiveRunner:
    def __init__(self, agent: TradingAgent, feed: BarFeed, poll_seconds: float = 60.0,
                 journal_path: str | Path | None = None) -> None:
        self.agent = agent
        self.feed = feed
        self.poll_seconds = poll_seconds
        self.journal_path = Path(journal_path) if journal_path else None
        self._stop = False

    def stop(self, *_: Any) -> None:
        log.info("stop requested")
        self._stop = True

    def run(self, max_steps: int | None = None) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.agent.warm_start({s: self.feed.history(s) for s in self.agent.symbols})
        log.info("agent warmed up on %d symbols; polling every %ss",
                 len(self.agent.symbols), self.poll_seconds)
        steps = 0
        while not self._stop and not self.feed.exhausted():
            bars = self.feed.poll()
            if bars:
                ts = max(b.timestamp for b in bars.values())
                self.agent.broker.process_bars(bars)  # no-op for real brokers
                entries = self.agent.on_bar(ts, bars)
                acct = self.agent.broker.account()
                log.info("%s equity=%.2f positions=%s", ts, acct.equity, acct.positions)
                for e in entries:
                    log.info("  %s", e)
                self._write(entries)
                steps += 1
                if self.agent.halted:
                    log.warning("agent halted: %s", self.agent.halt_reason)
                    break
                if max_steps and steps >= max_steps:
                    break
            if self.poll_seconds and not self.feed.exhausted():
                time.sleep(self.poll_seconds)

    def _write(self, entries: list[dict[str, Any]]) -> None:
        if not self.journal_path or not entries:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a") as fh:
            for e in entries:
                fh.write(json.dumps(e, default=str) + "\n")
