import hashlib

import pytest

from algo_agent.india.kite import KiteBroker, KiteError
from algo_agent.india.market import (MarketCalendar, IST, completed_bars, parse_symbol,
                                     parse_yahoo_chart, round_to_tick)
from algo_agent.india.paytm import PaytmMoneyBroker
from algo_agent.india.shadow import ShadowBroker

from .fakes import FakeSession, Resp

import datetime as dt
import pandas as pd


def ok(data):
    return Resp(200, {"status": "success", "data": data})


# ------------------------------------------------------------------ market
def test_symbols_ticks_calendar():
    assert parse_symbol("infy") == ("NSE", "INFY")
    assert parse_symbol("BSE:500325") == ("BSE", "500325")
    assert round_to_tick(100.02, side=1) == 100.05
    assert round_to_tick(100.04, side=-1) == 100.0
    cal = MarketCalendar.from_strings(["2026-11-09"])
    mon = dt.datetime(2026, 11, 9, 10, 0, tzinfo=IST)
    assert not cal.is_open(mon)                                # holiday
    assert cal.is_open(mon + dt.timedelta(days=1))             # Tuesday 10:00
    assert not cal.is_open(dt.datetime(2026, 11, 10, 15, 31, tzinfo=IST))
    assert not cal.is_trading_day(dt.date(2026, 11, 7))        # Saturday


def test_completed_bars_drops_forming_candle():
    idx = pd.date_range("2026-11-03 09:15", periods=4, freq="15min", tz=IST)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=idx)
    now = dt.datetime(2026, 11, 3, 10, 5, tzinfo=IST)  # 10:00 candle still forming
    assert len(completed_bars(df, "15minute", now)) == 3
    days = pd.DatetimeIndex(["2026-11-02 09:15", "2026-11-03 09:15"]).tz_localize(IST)
    d = df.iloc[:2].set_axis(days)
    assert len(completed_bars(d, "day", dt.datetime(2026, 11, 3, 9, 20, tzinfo=IST))) == 1
    assert len(completed_bars(d, "day", dt.datetime(2026, 11, 3, 16, 0, tzinfo=IST))) == 2


def test_parse_yahoo_chart_adjusts_and_indexes_by_day():
    payload = {"chart": {"result": [{
        "meta": {"dataGranularity": "1d"},
        "timestamp": [1762140600, 1762227000],
        "indicators": {"quote": [{"open": [100, 102], "high": [105, 106], "low": [99, 101],
                                  "close": [104, 103], "volume": [1000, None]}],
                       "adjclose": [{"adjclose": [52, 51.5]}]}}]}}
    df = parse_yahoo_chart(payload)
    assert len(df) == 2 and df["close"].iloc[0] == pytest.approx(52)
    assert df["open"].iloc[0] == pytest.approx(50)    # same 0.5 adjustment factor
    assert df.index[0].hour == 9 and df["volume"].iloc[1] == 0


# -------------------------------------------------------------------- kite
def kite_session():
    return FakeSession({
        ("POST", "/session/token"): ok({"access_token": "AT123", "user_id": "AB1234"}),
        ("GET", "/portfolio/holdings"): ok([
            {"exchange": "NSE", "tradingsymbol": "INFY", "quantity": 10, "t1_quantity": 2, "last_price": 1500},
            {"exchange": "NSE", "tradingsymbol": "HDFCBANK", "quantity": 5, "t1_quantity": 0, "last_price": 1700}]),
        ("GET", "/portfolio/positions"): ok({"net": [
            {"exchange": "NSE", "tradingsymbol": "INFY", "quantity": -4, "last_price": 1510, "product": "CNC"},
            {"exchange": "NSE", "tradingsymbol": "TCS", "quantity": 3, "last_price": 4000, "product": "CNC"}]}),
        ("GET", "/user/margins"): ok({"equity": {"net": 50000.0}}),
        ("POST", "/orders/regular"): ok({"order_id": "250101000001"}),
        ("GET", "/quote/ltp"): Resp(403, {"status": "error", "error_type": "PermissionException",
                                          "message": "no market data subscription"}),
        ("GET", "/orders"): ok([
            {"order_id": "1", "status": "OPEN", "tag": "algoagent", "variety": "regular"},
            {"order_id": "2", "status": "OPEN", "tag": None, "variety": "regular"},     # user's own
            {"order_id": "3", "status": "COMPLETE", "tag": "algoagent", "variety": "regular"}]),
        ("DELETE", "/orders/regular/1"): ok({"order_id": "1"}),
    })


def test_kite_login_checksum_and_auth_header():
    s = kite_session()
    k = KiteBroker("key", "secret", session=s)
    assert "api_key=key" in k.login_url() and "v=3" in k.login_url()
    k.generate_session("REQ")
    sent = s.find("POST", "/session/token")[0]["data"]
    assert sent["checksum"] == hashlib.sha256(b"keyREQsecret").hexdigest()
    k.holdings()
    assert s.calls[-1]["headers"]["Authorization"] == "token key:AT123"
    assert s.calls[-1]["headers"]["X-Kite-Version"] == "3"


def test_kite_account_combines_holdings_and_positions():
    k = KiteBroker("key", access_token="t", session=kite_session(),
                   managed_symbols=["NSE:INFY", "NSE:TCS"], capital=40000)
    a = k.account()
    assert a.positions == {"NSE:INFY": 8, "NSE:HDFCBANK": 5, "NSE:TCS": 3}
    assert a.cash == 50000 and a.capital == 40000
    # only managed symbols count towards the bot's equity
    assert a.equity == pytest.approx(50000 + 8 * 1510 + 3 * 4000)


def test_kite_market_order_uses_protection_and_tag():
    s = kite_session()
    k = KiteBroker("key", access_token="t", session=s)
    o = k.submit_order("NSE:INFY", -7.9)
    sent = s.find("POST", "/orders/regular")[0]["data"]
    assert sent["transaction_type"] == "SELL" and sent["quantity"] == 7
    assert sent["order_type"] == "MARKET" and sent["market_protection"] == -1
    assert sent["tag"] == "algoagent" and sent["product"] == "CNC"
    assert o.id == "250101000001"


def test_kite_limit_order_falls_back_to_ref_price():
    s = kite_session()
    k = KiteBroker("key", access_token="t", session=s, order_style="limit", limit_buffer_pct=1.0)
    k.submit_order("INFY", 3, ref_price=1000.0)
    sent = s.find("POST", "/orders/regular")[0]["data"]
    assert sent["order_type"] == "LIMIT" and sent["price"] == "1010.00"
    with pytest.raises(KiteError):
        k.submit_order("INFY", 3)  # no quote and no reference price


def test_kite_cancel_all_only_touches_own_open_orders():
    s = kite_session()
    KiteBroker("key", access_token="t", session=s).cancel_all()
    deletes = [c["url"] for c in s.calls if c["method"] == "DELETE"]
    assert deletes == ["https://api.kite.trade/orders/regular/1"]


def test_kite_error_surfaces_token_problems():
    s = FakeSession({("GET", "/portfolio/holdings"): Resp(403, {
        "status": "error", "error_type": "TokenException", "message": "Incorrect api_key or access_token."})})
    with pytest.raises(KiteError) as e:
        KiteBroker("key", access_token="old", session=s).holdings()
    assert e.value.is_token_error


def test_kite_historical_parses_candles():
    csv = "instrument_token,exchange_token,tradingsymbol,name,exchange\n408065,1594,INFY,INFOSYS,NSE\n"
    s = FakeSession({
        ("GET", "/instruments/NSE"): Resp(200, None, ctype="text/csv", text=csv),
        ("GET", "/instruments/historical/408065/day"): ok({"candles": [
            ["2026-10-30T00:00:00+0530", 1500, 1520, 1490, 1510, 100000],
            ["2026-11-02T00:00:00+0530", 1510, 1530, 1500, 1525, 120000]]}),
    })
    df = KiteBroker("key", access_token="t", session=s).historical("NSE:INFY", "day")
    assert list(df["close"]) == [1510, 1525] and df.index[0].hour == 9


# ------------------------------------------------------------------- paytm
MASTER = ("security_id,symbol,exchange,segment,instrument_type,name\n"
          "1594,INFY,NSE,E,ES,Infosys\n11536,TCS,NSE,E,ES,TCS\n9999,NIFTY24NOV,NSE,D,FUTIDX,x\n")


def paytm_session():
    return FakeSession({
        ("POST", "/accounts/v2/gettoken"): Resp(200, {"access_token": "JWT", "public_access_token": "P"}),
        ("GET", "/data/v1/scrips/security_master.csv"): Resp(200, None, ctype="text/csv", text=MASTER),
        ("GET", "/holdings/v1/get-user-holdings-data"): Resp(200, {"data": {"results": [
            {"security_id": "1594", "exchange": "NSE", "remaining_quantity": "10", "last_traded_price": "1500"}]}}),
        ("GET", "/orders/v1/position"): Resp(200, {"data": [
            {"security_id": "11536", "exchange": "NSE", "net_qty": 2, "last_traded_price": 4000}]}),
        ("GET", "/accounts/v1/funds/summary"): Resp(200, {"data": {"funds_summary": {"available_cash": 25000.5}}}),
        ("POST", "/orders/v1/place/regular"): Resp(200, {"status": "success", "data": [{"order_no": "PM1"}]}),
        ("GET", "/orders/v1/order-book"): Resp(200, {"data": [
            {"order_no": "PM1", "status": "Pending", "txn_type": "B", "security_id": "1594"},
            {"order_no": "MANUAL", "status": "Pending"}]}),
        ("POST", "/orders/v1/cancel/regular"): Resp(200, {"status": "success"}),
    })


def test_paytm_session_account_and_orders():
    s = paytm_session()
    p = PaytmMoneyBroker("key", "secret", session=s, managed_symbols=["NSE:INFY", "NSE:TCS"])
    assert "apiKey=key" in p.login_url()
    p.generate_session("RT")
    assert s.find("POST", "/accounts/v2/gettoken")[0]["json"] == {
        "api_key": "key", "api_secret_key": "secret", "request_token": "RT"}
    a = p.account()
    assert a.positions == {"NSE:INFY": 10, "NSE:TCS": 2}
    assert a.cash == 25000.5 and a.equity == pytest.approx(25000.5 + 15000 + 8000)
    assert s.calls[-1]["headers"]["x-jwt-token"] == "JWT"

    o = p.submit_order("NSE:INFY", 5)
    body = s.find("POST", "/orders/v1/place/regular")[0]["json"]
    assert body["txn_type"] == "B" and body["security_id"] == "1594" and body["order_type"] == "MKT"
    assert body["segment"] == "E" and body["product"] == "C" and o.id == "PM1"

    p.cancel_all()
    cancels = s.find("POST", "/orders/v1/cancel/regular")
    assert len(cancels) == 1 and cancels[0]["json"]["order_no"] == "PM1"


def test_shadow_broker_never_sends():
    s = kite_session()
    real = KiteBroker("key", access_token="t", session=s)
    sh = ShadowBroker(real)
    sh.submit_order("NSE:INFY", 5, reason="test")
    sh.cancel_all()
    assert sh.account().cash == 50000
    assert not s.find("POST", "/orders/regular") and not [c for c in s.calls if c["method"] == "DELETE"]
    assert len(sh.would_send) == 1
