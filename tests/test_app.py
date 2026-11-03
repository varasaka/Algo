import datetime as dt
import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from algo_agent.app.engine import Engine
from algo_agent.app.server import serve
from algo_agent.app.settings import Settings, Store
from algo_agent.broker import Account, Broker, Order
from algo_agent.data import synthetic_ohlcv
from algo_agent.india.market import IST

TUE_0920 = dt.datetime(2026, 11, 3, 9, 20, 30, tzinfo=IST)


def fetcher(symbol, interval):
    df = synthetic_ohlcv(400, seed=abs(hash(symbol)) % 1000)
    idx = pd.bdate_range(end="2026-11-02", periods=len(df))
    df.index = idx.tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
    return df


class RecordingBroker(Broker):
    def __init__(self):
        self.sent = []
        self.access_token = None

    def account(self):
        return Account(cash=100_000, equity=100_000, positions={"NSE:INFY": 10, "NSE:OTHER": 99},
                       capital=50_000, prices={"NSE:INFY": 1000})

    def submit_order(self, symbol, qty, reason="", timestamp=None, ref_price=None):
        self.sent.append((symbol, qty, reason))
        return Order("x", symbol, qty, reason)

    def cancel_all(self):
        pass

    def login_url(self):
        return "https://broker.example/login"

    def generate_session(self, token):
        self.access_token = "TOKEN-" + token
        return {}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def make_engine(store, clock=lambda: TUE_0920, broker=None):
    rb = broker or RecordingBroker()
    return Engine(store, fetcher=fetcher, clock=clock, broker_factory=lambda s, t: rb), rb


def test_settings_validation_and_masking(store):
    s = Settings()
    with pytest.raises(ValueError):
        s.update({"broker": "paper", "mode": "live"})
    new = s.update({"symbols": "infy, tcs\nNSE:SBIN", "kite_api_secret": "shh", "broker": "kite",
                    "mode": "shadow"})
    assert new.symbols == ["NSE:INFY", "NSE:TCS", "NSE:SBIN"]
    assert new.public()["kite_api_secret"] == "********"
    assert new.update({"kite_api_secret": "********"}).kite_api_secret == "shh"
    with pytest.raises(ValueError):
        s.update({"risk": {"bogus": 1}})


def test_token_expires_at_six_am(store, monkeypatch):
    import algo_agent.india.market as m
    monkeypatch.setattr(m, "now_ist", lambda: dt.datetime(2026, 11, 3, 10, 0, tzinfo=IST))
    store.save_token("kite", "abc")
    assert store.load_token("kite") == "abc"
    monkeypatch.setattr(m, "now_ist", lambda: dt.datetime(2026, 11, 4, 5, 0, tzinfo=IST))
    assert store.load_token("kite") == "abc"          # still the same session day
    monkeypatch.setattr(m, "now_ist", lambda: dt.datetime(2026, 11, 4, 6, 1, tzinfo=IST))
    assert store.load_token("kite") is None


def test_daily_schedule(store):
    eng, _ = make_engine(store)
    eng.settings.holidays = ["2026-11-10"]
    assert not eng.due(dt.datetime(2026, 11, 3, 9, 10, tzinfo=IST))   # before run_at
    assert eng.due(dt.datetime(2026, 11, 3, 9, 25, tzinfo=IST))
    assert not eng.due(dt.datetime(2026, 11, 7, 10, 0, tzinfo=IST))   # Saturday
    assert not eng.due(dt.datetime(2026, 11, 10, 10, 0, tzinfo=IST))  # holiday
    eng.last_cycle_day = dt.date(2026, 11, 3)
    assert not eng.due(dt.datetime(2026, 11, 3, 11, 0, tzinfo=IST))   # once per day


def test_intraday_schedule(store):
    eng, _ = make_engine(store)
    eng.settings.timeframe = "15minute"
    assert eng.due(dt.datetime(2026, 11, 3, 9, 30, 25, tzinfo=IST))
    eng.last_cycle = dt.datetime(2026, 11, 3, 9, 30, 25, tzinfo=IST)
    assert not eng.due(dt.datetime(2026, 11, 3, 9, 40, tzinfo=IST))
    assert eng.due(dt.datetime(2026, 11, 3, 9, 45, 21, tzinfo=IST))
    assert not eng.due(dt.datetime(2026, 11, 3, 15, 45, tzinfo=IST))  # closed


def test_paper_cycle_trades_and_persists(store):
    eng, _ = make_engine(store)
    eng.update_settings({"strategy": "sma_crossover", "strategy_params": {"fast": 5, "slow": 20},
                         "symbols": ["NSE:AAA", "NSE:BBB"]})
    eng.start()
    entries = eng.run_cycle(TUE_0920)
    assert any(e["action"] == "fill" for e in entries) or all(
        abs(s) == 0 for s in eng.status()["signals"].values())
    assert eng.run_cycle(TUE_0920) == []  # same candle twice -> nothing new
    st = eng.status()
    assert st["account"]["equity"] > 0 and st["running"]
    eng.stop()
    eng2, _ = make_engine(store)
    assert eng2.paper is not None and eng2.paper.cash == pytest.approx(eng.paper.cash)
    assert store.tail_journal()


def test_shadow_mode_never_sends(store):
    eng, rb = make_engine(store)
    eng.update_settings({"broker": "kite", "mode": "shadow", "kite_api_key": "k",
                         "symbols": ["NSE:INFY"], "strategy": "sma_crossover",
                         "strategy_params": {"fast": 5, "slow": 20}})
    store.save_token("kite", "tok")
    eng.start()
    eng.run_cycle(TUE_0920)
    eng.kill_switch()
    assert rb.sent == []


def test_live_requires_confirmation_and_only_manages_own_symbols(store):
    eng, rb = make_engine(store)
    eng.update_settings({"broker": "kite", "mode": "live", "kite_api_key": "k",
                         "symbols": ["NSE:INFY"]})
    store.save_token("kite", "tok")
    with pytest.raises(RuntimeError, match="confirmed"):
        eng.start()
    with pytest.raises(ValueError):
        eng.confirm_live("yes")
    eng.confirm_live("I understand the risk")
    eng.start()
    eng.kill_switch()
    assert rb.sent == [("NSE:INFY", -10, "kill_switch")]   # NSE:OTHER untouched
    # changing settings revokes the confirmation
    eng.update_settings({"mode": "shadow"})
    assert not eng.settings.live_confirmed


def test_requires_login_for_real_broker(store):
    eng, _ = make_engine(store)
    eng.update_settings({"broker": "kite", "mode": "shadow", "kite_api_key": "k"})
    with pytest.raises(RuntimeError, match="log in"):
        eng.start()
    eng.complete_login("REQ")
    assert store.load_token("kite") == "TOKEN-REQ"
    eng.start()


# ----------------------------------------------------------------- server
def call(base, path, body=None, cookie=None, origin=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if cookie:
        req.add_header("Cookie", cookie)
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}"), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


@pytest.fixture
def server(store):
    eng, _ = make_engine(store)
    httpd = serve("127.0.0.1", 0, str(store.root), open_browser=False, engine=eng)
    httpd.app.require_pin = True  # behave like a LAN-exposed server
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd
    httpd.shutdown()


def test_server_pin_flow_and_api(server):
    base, httpd = server
    assert call(base, "/api/status")[0] == 401
    code, body, _ = call(base, "/api/session")
    assert body == {"authed": False, "pin_set": False, "pin_required": True}
    code, _, headers = call(base, "/api/pin", {"pin": "1234"})
    assert code == 200
    cookie = headers["Set-Cookie"].split(";")[0]
    assert call(base, "/api/unlock", {"pin": "0000"})[0] == 401
    assert call(base, "/api/unlock", {"pin": "1234"})[0] == 200
    code, st, _ = call(base, "/api/status", cookie=cookie)
    assert code == 200 and st["mode"] == "paper"
    code, s, _ = call(base, "/api/settings", {"kite_api_secret": "topsecret"}, cookie=cookie)
    assert s["kite_api_secret"] == "********" and "pin_hash" not in s
    assert call(base, "/api/strategies", cookie=cookie)[1][0]["name"]
    code, bt, _ = call(base, "/api/backtest", {}, cookie=cookie)
    assert code == 200 and "sharpe" in bt["metrics"] and len(bt["equity"]) > 10
    # cross-site POST blocked
    assert call(base, "/api/start", {}, cookie=cookie, origin="https://evil.example")[0] == 403
    # wrong PIN brute force is throttled
    for _ in range(5):
        call(base, "/api/unlock", {"pin": "9999"})
    assert call(base, "/api/unlock", {"pin": "1234"})[0] == 429


def test_static_files_served(server):
    base, _ = server
    with urllib.request.urlopen(base + "/") as r:
        assert b"Algo Agent" in r.read()
    with urllib.request.urlopen(base + "/manifest.webmanifest") as r:
        assert json.loads(r.read())["display"] == "standalone"
    with urllib.request.urlopen(base + "/../../settings.py") as r:   # traversal -> index
        assert b"<!doctype html>" in r.read()
