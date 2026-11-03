"""The automation engine behind the app.

It owns the broker connection, builds a ``TradingAgent`` from the settings,
and runs it on an IST schedule in a background thread:

* ``day`` timeframe: once per trading day at ``run_at`` (default 09:20), using
  completed daily candles, so orders go in near the open, as in the backtest.
* intraday timeframes: after each candle closes, during market hours, with
  an automatic square-off at ``square_off_at`` for MIS positions.

Three modes:

* ``paper``: simulated money, real market data. Nothing touches a broker.
* ``shadow``: reads your real broker account, but *records* orders instead
  of sending them. Use this to verify behaviour against your real holdings.
* ``live``: sends real orders. Needs an explicit confirmation in the app.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import zlib
import threading
import traceback
from typing import Any, Callable

import pandas as pd

from ..agent import TradingAgent
from ..broker import Broker, PaperBroker
from ..data import Bar
from ..india.kite import KiteBroker
from ..india.market import IST, MarketCalendar, completed_bars, fetch_yahoo, now_ist
from ..india.paytm import PaytmMoneyBroker
from ..india.shadow import ShadowBroker
from ..risk import RiskConfig, RiskManager
from ..strategies import create_strategy
from .settings import Settings, Store

log = logging.getLogger(__name__)

INTERVAL_MINUTES = {"60minute": 60, "30minute": 30, "15minute": 15, "5minute": 5}
DataFetcher = Callable[[str, str], pd.DataFrame]


class Engine:
    def __init__(self, store: Store, fetcher: DataFetcher | None = None,
                 clock: Callable[[], dt.datetime] = now_ist,
                 broker_factory: Callable[[Settings, str | None], Broker] | None = None) -> None:
        self.store = store
        self.settings = store.load_settings()
        self.clock = clock
        self._fetcher_override = fetcher
        self._broker_factory = broker_factory
        self.lock = threading.RLock()
        self.running = False
        self.agent: TradingAgent | None = None
        self.exec_broker: Broker | None = None
        self.paper: PaperBroker | None = None
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.last_cycle: dt.datetime | None = None
        self.last_cycle_day: dt.date | None = None
        self.squared_off_day: dt.date | None = None
        self.last_error: str = ""
        self.equity_history: list[dict[str, Any]] = store.read_json("equity.json", []) or []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load_state()

    # ----------------------------------------------------------- brokers
    def real_broker(self, token: str | None = None) -> Broker:
        s = self.settings
        token = token if token is not None else self.store.load_token(s.broker)
        if self._broker_factory:
            return self._broker_factory(s, token)
        common = dict(managed_symbols=s.symbols, capital=s.capital,
                      order_style=s.order_style, limit_buffer_pct=s.limit_buffer_pct)
        if s.broker == "kite":
            return KiteBroker(s.kite_api_key, s.kite_api_secret, token, product=s.product, **common)
        if s.broker == "paytm":
            product = s.paytm_product if s.product == "CNC" else "I"
            return PaytmMoneyBroker(s.paytm_api_key, s.paytm_api_secret, token,
                                    product=product, source=s.paytm_source, **common)
        raise ValueError("paper broker has no real account")

    def broker_connected(self) -> bool:
        if self.settings.broker == "paper":
            return True
        return self.store.load_token(self.settings.broker) is not None

    def login_url(self) -> str:
        return self.real_broker(token="").login_url()

    def complete_login(self, request_token: str) -> dict[str, Any]:
        broker = self.real_broker(token="")
        data = broker.generate_session(request_token.strip())
        self.store.save_token(self.settings.broker, broker.access_token)
        self.event("login", f"connected to {self.settings.broker}")
        return {"ok": True}

    def logout(self) -> None:
        self.store.clear_token(self.settings.broker)
        self.event("logout", f"disconnected {self.settings.broker}")

    def test_connection(self) -> dict[str, Any]:
        broker = self.real_broker()
        acct = broker.account()
        return {"cash": acct.cash, "equity_managed": acct.equity, "positions": acct.positions,
                "profile": _safe(lambda: _profile_name(broker.profile()))}

    def _build_exec_broker(self) -> Broker:
        s = self.settings
        if s.mode == "paper":
            if self.paper is None:
                self.paper = PaperBroker(cash=s.capital, commission_bps=3.0, slippage_bps=5.0)
            return self.paper
        if not self.broker_connected():
            raise RuntimeError(f"log in to {s.broker} first (tokens expire daily)")
        real = self.real_broker()
        if s.mode == "shadow":
            return ShadowBroker(real)
        if not s.live_confirmed:
            raise RuntimeError("live mode has not been confirmed in the app")
        return real

    # -------------------------------------------------------------- data
    def fetch(self, symbol: str, interval: str) -> pd.DataFrame:
        if self._fetcher_override:
            return self._fetcher_override(symbol, interval)
        if os.environ.get("ALGO_AGENT_DEMO_DATA") == "1":
            return demo_bars(symbol, interval)
        if self.settings.data_source == "broker" and self.settings.broker == "kite":
            return self.real_broker().historical(symbol, interval)
        return fetch_yahoo(symbol, interval)

    # ----------------------------------------------------------- control
    def update_settings(self, changes: dict[str, Any]) -> Settings:
        with self.lock:
            if self.running:
                raise RuntimeError("stop the agent before changing settings")
            new = self.settings.update(changes)
            if new.capital != self.settings.capital or new.mode != self.settings.mode:
                self.paper = None
            self.settings = new
            self.store.save_settings(new)
            self.agent = None  # rebuilt with the new settings on next start
            self.last_bar = {}
            self._save_state()
            return new

    def confirm_live(self, phrase: str) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("stop the agent first")
            s = self.settings
            if s.broker == "paper":
                raise ValueError("choose Zerodha or Paytm Money first")
            if phrase.strip().upper() != "I UNDERSTAND THE RISK":
                raise ValueError('type exactly "I UNDERSTAND THE RISK" to enable live trading')
            s.mode, s.live_confirmed = "live", True
            self.store.save_settings(s)
            self.agent = None
            self.event("live_enabled", "live trading confirmed by user")

    def start(self) -> None:
        with self.lock:
            self.settings.validate()
            self.exec_broker = self._build_exec_broker()
            if self.agent is None or self.agent.broker is not self.exec_broker:
                self.agent = self._build_agent(self.exec_broker)
            self.running = True
            self.last_error = ""
            self.event("start", f"agent started in {self.settings.mode} mode")
        self._ensure_thread()

    def stop(self) -> None:
        with self.lock:
            if self.running:
                self.event("stop", "agent stopped")
            self.running = False

    def kill_switch(self) -> dict[str, Any]:
        """Stop the agent and close every position it manages."""
        with self.lock:
            self.running = False
            broker = self.exec_broker or self._build_exec_broker()
            agent = self.agent or self._build_agent(broker)
            agent.flatten(pd.Timestamp(self.clock()), reason="kill_switch")
            if isinstance(broker, PaperBroker):
                broker.fill_pending(broker.prices, pd.Timestamp(self.clock()))
            self._record(agent)
            self.event("kill_switch", "all managed positions closed and agent stopped")
            return {"ok": True}

    def shutdown(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------- schedule
    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="algo-engine", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self.tick()
            except Exception as exc:  # keep the loop alive no matter what
                log.exception("engine tick failed")
                self.last_error = f"{type(exc).__name__}: {exc}"

    def due(self, now: dt.datetime) -> bool:
        s = self.settings
        cal = MarketCalendar.from_strings(s.holidays)
        if not cal.is_trading_day(now.date()):
            return False
        if s.timeframe == "day":
            run_at = dt.time.fromisoformat(s.run_at)
            return now.time() >= run_at and now.time() < cal.close_time \
                and self.last_cycle_day != now.date()
        if not cal.is_open(now):
            return False
        minutes = INTERVAL_MINUTES[s.timeframe]
        open_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        elapsed = (now - open_dt).total_seconds() / 60
        boundary = open_dt + dt.timedelta(minutes=minutes * int(elapsed // minutes))
        ready = now >= boundary + dt.timedelta(seconds=20)  # let the candle settle
        return ready and (self.last_cycle is None or self.last_cycle < boundary)

    def tick(self) -> None:
        if not self.running:
            return
        now = self.clock()
        s = self.settings
        if s.product == "MIS" and self._square_off_due(now):
            self.squared_off_day = now.date()
            self.event("square_off", "intraday square-off")
            self.kill_switch()
            self.running = True  # resume tomorrow
            return
        if self.due(now):
            self.run_cycle(now)

    def _square_off_due(self, now: dt.datetime) -> bool:
        cal = MarketCalendar.from_strings(self.settings.holidays)
        sq = dt.time.fromisoformat(self.settings.square_off_at)
        return (cal.is_trading_day(now.date()) and sq <= now.time() < cal.close_time
                and self.squared_off_day != now.date())

    # -------------------------------------------------------------- core
    def _build_agent(self, broker: Broker) -> TradingAgent:
        s = self.settings
        strategy = create_strategy(s.strategy, **dict(s.strategy_params))
        risk = RiskConfig.from_dict(s.risk)
        ppy = 252 if s.timeframe == "day" else int(252 * 375 / INTERVAL_MINUTES[s.timeframe])
        self.last_bar = {}
        return TradingAgent(strategy, broker, s.symbols, RiskManager(risk), periods_per_year=ppy)

    def run_cycle(self, now: dt.datetime | None = None) -> list[dict[str, Any]]:
        """Fetch data, let the agent decide, place orders. Safe to call manually."""
        with self.lock:
            now = now or self.clock()
            s = self.settings
            if self.exec_broker is None or self.agent is None:
                self.exec_broker = self._build_exec_broker()
                self.agent = self._build_agent(self.exec_broker)
            agent = self.agent
            new_bars: dict[str, Bar] = {}
            warm: dict[str, pd.DataFrame] = {}
            for sym in s.symbols:
                try:
                    df = completed_bars(self.fetch(sym, s.timeframe), s.timeframe, now)
                except Exception as exc:
                    self.event("data_error", f"{sym}: {exc}", level="error")
                    continue
                if df.empty:
                    continue
                if sym not in self.last_bar:
                    # First sight of this symbol: warm up on everything but the last bar.
                    warm[sym] = df.iloc[:-1]
                    self.last_bar[sym] = df.index[-2] if len(df) > 1 else df.index[0] - pd.Timedelta(seconds=1)
                last_ts = df.index[-1]
                if last_ts > self.last_bar[sym]:
                    new_bars[sym] = Bar.from_row(last_ts, df.iloc[-1])
                    self.last_bar[sym] = last_ts
            if warm:
                agent.warm_start(warm)
            self.last_cycle = now
            self.last_cycle_day = now.date()
            if not new_bars:
                self.event("idle", "no new completed candles")
                self._save_state()
                return []
            ts = max(b.timestamp for b in new_bars.values())
            broker = self.exec_broker
            if isinstance(broker, PaperBroker):
                broker.mark({k: b.close for k, b in new_bars.items()})
            entries = agent.on_bar(ts, new_bars)
            if isinstance(broker, PaperBroker):
                # Paper orders execute now at the latest price (+ slippage).
                broker.fill_pending({k: b.close for k, b in new_bars.items()}, pd.Timestamp(now))
                entries += [{"time": f.timestamp, "symbol": f.symbol, "action": "fill",
                             "qty": f.qty, "price": round(f.price, 2), "reason": f.reason}
                            for f in broker.drain_fills()]
            if isinstance(broker, ShadowBroker):
                for e in entries:
                    if e.get("action") in ("enter", "exit", "rebalance", "order"):
                        e["note"] = "shadow: not sent"
            for e in entries:
                e.setdefault("mode", s.mode)
            self.store.append_journal(entries)
            if agent.halted:
                self.running = False
                self.event("halted", f"circuit breaker: {agent.halt_reason}", level="error")
            self._record(agent)
            return entries

    def _record(self, agent: TradingAgent) -> None:
        try:
            acct = agent.broker.account()
            eq = min(acct.equity, acct.capital) if acct.capital and self.settings.mode != "paper" \
                else acct.equity
            self.equity_history.append({"time": self.clock().isoformat(timespec="seconds"),
                                        "equity": round(eq, 2)})
            self.equity_history = self.equity_history[-2000:]
            self.store.write_json("equity.json", self.equity_history)
        except Exception as exc:
            self.event("account_error", str(exc), level="error")
        self._save_state()

    # --------------------------------------------------------- reporting
    def event(self, action: str, message: str, level: str = "info") -> None:
        entry = {"time": self.clock().isoformat(timespec="seconds"), "action": action,
                 "reason": message, "level": level, "mode": self.settings.mode}
        if level == "error":
            self.last_error = message
        getattr(log, "error" if level == "error" else "info")("%s: %s", action, message)
        self.store.append_journal([entry])

    def status(self) -> dict[str, Any]:
        s = self.settings
        out: dict[str, Any] = {
            "running": self.running, "mode": s.mode, "broker": s.broker,
            "connected": self.broker_connected(), "live_confirmed": s.live_confirmed,
            "market_open": MarketCalendar.from_strings(s.holidays).is_open(self.clock()),
            "now": self.clock().isoformat(timespec="seconds"),
            "last_cycle": self.last_cycle.isoformat(timespec="seconds") if self.last_cycle else None,
            "last_error": self.last_error, "halted": bool(self.agent and self.agent.halted),
            "equity_history": self.equity_history[-500:],
        }
        broker = self.exec_broker
        try:
            if broker is None and s.mode == "paper":
                broker = self._build_exec_broker()
            if broker is not None:
                acct = broker.account()
                out["account"] = {"cash": round(acct.cash, 2), "equity": round(acct.equity, 2),
                                  "capital": s.capital}
                out["positions"] = [
                    {"symbol": sym, "qty": q, "price": acct.prices.get(sym),
                     "value": round(q * acct.prices.get(sym, 0.0), 2),
                     "managed": sym in s.symbols,
                     "stop": (round(self.agent.state[sym].stop, 2)
                              if self.agent and sym in self.agent.state and self.agent.state[sym].stop
                              else None)}
                    for sym, q in sorted(acct.positions.items())]
        except Exception as exc:
            out["account_error"] = str(exc)
        if self.agent:
            out["signals"] = {sym: st.last_signal for sym, st in self.agent.state.items()}
        return out

    # ------------------------------------------------------- persistence
    def _save_state(self) -> None:
        state = {"last_bar": {k: v.isoformat() for k, v in self.last_bar.items()},
                 "last_cycle": self.last_cycle.isoformat() if self.last_cycle else None,
                 "paper": self.paper.to_dict() if self.paper else None,
                 "running": self.running}
        self.store.write_json("state.json", state)

    def _load_state(self) -> None:
        st = self.store.read_json("state.json", {}) or {}
        if st.get("paper") and self.settings.mode == "paper":
            self.paper = PaperBroker(cash=self.settings.capital, commission_bps=3.0, slippage_bps=5.0)
            self.paper.load_dict(st["paper"])
        if st.get("last_cycle"):
            self.last_cycle = dt.datetime.fromisoformat(st["last_cycle"])
            self.last_cycle_day = self.last_cycle.astimezone(IST).date()
        self.resume_requested = bool(st.get("running"))


def demo_bars(symbol: str, interval: str) -> pd.DataFrame:
    """Synthetic candles for trying the app offline (ALGO_AGENT_DEMO_DATA=1)."""
    from ..data import synthetic_ohlcv
    seed = zlib.crc32(symbol.encode()) % 10_000
    if interval == "day":
        end = now_ist().date()
        df = synthetic_ohlcv(n_bars=600, seed=seed, start_price=1000 + seed % 2000)
        idx = pd.bdate_range(end=end - dt.timedelta(days=1), periods=len(df))
        df.index = idx.tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        return df
    minutes = INTERVAL_MINUTES[interval]
    df = synthetic_ohlcv(n_bars=600, seed=seed, start_price=1000 + seed % 2000, annual_vol=0.05)
    end = pd.Timestamp(now_ist()).floor(f"{minutes}min")
    df.index = pd.date_range(end=end, periods=len(df), freq=f"{minutes}min")
    return df


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        return f"unavailable ({exc})"


def _profile_name(p: Any) -> str:
    if isinstance(p, dict):
        for k in ("user_name", "user_shortname", "name", "email"):
            if p.get(k):
                return str(p[k])
        data = p.get("data")
        if isinstance(data, dict):
            return _profile_name(data)
    return "connected"
