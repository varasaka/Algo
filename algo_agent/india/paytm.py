"""Paytm Money Open API broker (https://developer.paytmmoney.com). EXPERIMENTAL.

Endpoints, the ``x-jwt-token`` header, login URL and order fields match
Paytm Money's official Python SDK (``paytmmoney/pyPMClient``). That SDK does
not document every *response* field, so parsing here is deliberately
tolerant. Run in **shadow mode** first and check that positions and cash
shown in the app match the Paytm Money app before letting it place orders.

Like Zerodha, Paytm Money requires a daily login on their own page, which
redirects back with a ``requestToken``. Your password never reaches this app.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any, Iterable

import pandas as pd

from ..broker import Account, Broker, Order
from .market import normalize_symbol, parse_symbol, round_to_tick

log = logging.getLogger(__name__)

HOST = "https://developer.paytmmoney.com"
LOGIN_URL = "https://login.paytmmoney.com/merchant-login"
OPEN_WORDS = ("open", "pending", "trigger", "received", "modified")


class PaytmError(RuntimeError):
    pass


def _find(obj: Any, keys: Iterable[str]) -> Any:
    """Depth-first search for the first present key (case-insensitive)."""
    wanted = [k.lower() for k in keys]
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            low = {str(k).lower(): v for k, v in cur.items()}
            for k in wanted:
                if k in low and low[k] not in (None, ""):
                    return low[k]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _records(payload: Any) -> list[dict[str, Any]]:
    """Pull the list of row dicts out of a Paytm response, wherever it is."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "result", "positions", "holdings", "orders"):
            if key in payload:
                rows = _records(payload[key])
                if rows:
                    return rows
    return []


def _num(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    v = _find(row, keys)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class PaytmMoneyBroker(Broker):
    name = "paytm"

    def __init__(self, api_key: str, api_secret: str | None = None,
                 access_token: str | None = None, *, product: str = "C",
                 order_style: str = "market", limit_buffer_pct: float = 0.5,
                 source: str = "W", managed_symbols: list[str] | None = None,
                 capital: float | None = None, session=None, timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("Paytm Money API key is required")
        import requests

        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.product = product          # C = delivery, I/T = intraday (see your API docs)
        self.order_style = order_style
        self.limit_buffer_pct = limit_buffer_pct
        self.source = source
        self.managed = [normalize_symbol(s) for s in (managed_symbols or [])]
        self.capital = capital
        self.timeout = timeout
        self.session = session or requests.Session()
        self._sec_ids: dict[str, str] = {}
        self._sec_syms: dict[str, str] = {}
        self.placed: list[str] = []

    # ------------------------------------------------------------- session
    def login_url(self, state: str = "algoagent") -> str:
        return f"{LOGIN_URL}?apiKey={self.api_key}&state={state}"

    def generate_session(self, request_token: str) -> dict[str, Any]:
        if not self.api_secret:
            raise ValueError("Paytm Money API secret is required to create a session")
        data = self._request("POST", "/accounts/v2/gettoken", json={
            "api_key": self.api_key, "api_secret_key": self.api_secret,
            "request_token": request_token})
        token = _find(data, ["access_token"])
        if not token:
            raise PaytmError(f"no access_token in response: {str(data)[:200]}")
        self.access_token = token
        return data

    def profile(self) -> dict[str, Any]:
        return self._request("GET", "/accounts/v1/user/details")

    # ----------------------------------------------------------- transport
    def _request(self, method: str, path: str, params=None, json=None, raw: bool = False) -> Any:
        headers = {"Content-Type": "application/json", "User-Agent": "algo-agent"}
        if self.access_token:
            headers["x-jwt-token"] = self.access_token
        resp = self.session.request(method, HOST + path, params=params, json=json,
                                    headers=headers, timeout=self.timeout)
        if resp.status_code >= 400:
            raise PaytmError(f"Paytm {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        if raw:
            return resp.text
        body = resp.json()
        if isinstance(body, dict) and str(body.get("status", "")).lower() in ("error", "failure"):
            raise PaytmError(str(body.get("message") or body)[:300])
        return body

    # ----------------------------------------------------- security master
    def _load_master(self) -> None:
        text = self._request("GET", "/data/v1/scrips/security_master.csv", raw=True)
        for row in csv.DictReader(io.StringIO(text)):
            r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            seg = r.get("segment", "E").upper()
            itype = r.get("instrument_type", r.get("instrument", "")).upper()
            if seg not in ("E", "") or (itype and itype not in ("ES", "EQ", "EQUITY", "ETF")):
                continue
            sec_id = r.get("security_id") or r.get("securityid")
            sym = r.get("symbol") or r.get("trading_symbol") or r.get("tradingsymbol")
            ex = (r.get("exchange") or "NSE").upper()
            if sec_id and sym:
                key = f"{ex}:{sym.upper()}"
                self._sec_ids.setdefault(key, sec_id)
                self._sec_syms.setdefault(f"{ex}:{sec_id}", key)

    def security_id(self, symbol: str) -> str:
        symbol = normalize_symbol(symbol)
        if symbol not in self._sec_ids:
            self._load_master()
        if symbol not in self._sec_ids:
            raise PaytmError(f"{symbol} not found in Paytm security master")
        return self._sec_ids[symbol]

    def _symbol_for(self, row: dict[str, Any]) -> str | None:
        ex = str(_find(row, ["exchange"]) or "NSE").upper()
        sec_id = _find(row, ["security_id", "securityId", "scrip_id"])
        if sec_id is not None:
            key = f"{ex}:{sec_id}"
            if key not in self._sec_syms and not self._sec_syms:
                self._load_master()
            if key in self._sec_syms:
                return self._sec_syms[key]
        name = _find(row, ["symbol", "trading_symbol", "tradingsymbol", "display_name"])
        return f"{ex}:{str(name).upper()}" if name else None

    # ------------------------------------------------------------ account
    def account(self) -> Account:
        qty: dict[str, float] = {}
        prices: dict[str, float] = {}
        for h in _records(self._request("GET", "/holdings/v1/get-user-holdings-data")):
            sym = self._symbol_for(h)
            if sym:
                qty[sym] = qty.get(sym, 0.0) + _num(h, "remaining_quantity", "quantity", "qty")
                prices[sym] = _num(h, "last_traded_price", "ltp", "last_price", "close_price")
        for p in _records(self._request("GET", "/orders/v1/position")):
            sym = self._symbol_for(p)
            if sym:
                qty[sym] = qty.get(sym, 0.0) + _num(p, "net_qty", "netqty", "net_quantity", "quantity")
                prices[sym] = _num(p, "last_traded_price", "ltp", "last_price",
                                   default=prices.get(sym, 0.0))
        funds = self._request("GET", "/accounts/v1/funds/summary", params={"config": "true"})
        cash = float(_find(funds, ["available_cash", "availableCash", "available_balance",
                                   "net_balance", "trade_balance"]) or 0.0)
        managed = set(self.managed) if self.managed else set(qty)
        value = sum(q * prices.get(s, 0.0) for s, q in qty.items() if s in managed)
        return Account(cash=cash, equity=cash + value, positions={s: q for s, q in qty.items() if q},
                       capital=self.capital, prices=prices)

    # ------------------------------------------------------------- orders
    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None,
                     ref_price: float | None = None) -> Order:
        qty = float(int(qty))
        if qty == 0:
            return Order(id="noop", symbol=symbol, qty=0.0, reason=reason, created=timestamp)
        exchange, _ = parse_symbol(symbol)
        side = 1.0 if qty > 0 else -1.0
        body: dict[str, Any] = {
            "txn_type": "B" if side > 0 else "S", "exchange": exchange, "segment": "E",
            "product": self.product, "security_id": self.security_id(symbol),
            "quantity": int(abs(qty)), "validity": "DAY", "source": self.source,
            "off_mkt_flag": False,
        }
        if self.order_style == "market":
            body.update(order_type="MKT", price=0)
        else:
            if not ref_price:
                raise PaytmError(f"limit order for {symbol} needs a reference price")
            body.update(order_type="LMT", price=round_to_tick(
                ref_price * (1 + side * self.limit_buffer_pct / 100), side=side))
        log.info("paytm order %s", body)
        resp = self._request("POST", "/orders/v1/place/regular", json=body)
        order_no = str(_find(resp, ["order_no", "orderNo", "order_id"]) or "")
        if order_no:
            self.placed.append(order_no)
        return Order(id=order_no or "unknown", symbol=normalize_symbol(symbol), qty=qty,
                     reason=reason, created=timestamp)

    def orders(self) -> list[dict[str, Any]]:
        return _records(self._request("GET", "/orders/v1/order-book"))

    def cancel_all(self) -> None:
        """Cancel open orders that this app placed in this session."""
        mine = set(self.placed)
        for o in self.orders():
            order_no = str(_find(o, ["order_no", "orderNo"]) or "")
            status = str(_find(o, ["status", "display_status"]) or "").lower()
            if order_no not in mine or not any(w in status for w in OPEN_WORDS):
                continue
            body = {k: _find(o, [k]) for k in (
                "txn_type", "exchange", "segment", "product", "security_id", "quantity",
                "validity", "order_type", "price", "mkt_type", "order_no", "serial_no",
                "group_id")}
            body.update(source=self.source, off_mkt_flag=False)
            try:
                self._request("POST", "/orders/v1/cancel/regular", json=body)
            except PaytmError as exc:
                log.warning("could not cancel %s: %s", order_no, exc)
