"""Zerodha Kite Connect v3 broker (https://kite.trade).

Endpoints, headers and the login checksum match Zerodha's official
``kiteconnect`` Python SDK. Key facts about Kite Connect:

* **Daily login.** An access token is valid for one trading day. Each
  morning you log in on Zerodha's own page (password + 2FA), and Zerodha
  redirects back to this app with a ``request_token``. This app never sees or
  stores your Zerodha password, and does not try to automate that login.
* **Static IP.** Under SEBI's retail-algo rules Zerodha only accepts *order*
  requests from the IP you registered in the Kite developer console. Data and
  portfolio calls work from anywhere.
* **Market protection.** API market orders need ``market_protection``; this
  module sends ``-1`` (automatic) for market orders, or you can use
  marketable limit orders instead (``order_style="limit"``).
* **Market data** (quotes and historical candles) needs the paid Kite Connect
  plan. Without it, use the free Yahoo data source; orders still work.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
from typing import Any

import pandas as pd

from ..broker import Account, Broker, Order
from .market import IST, normalize_symbol, parse_symbol, round_to_tick

log = logging.getLogger(__name__)

ROOT = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"
OPEN_STATUSES = {"OPEN", "TRIGGER PENDING", "OPEN PENDING", "VALIDATION PENDING",
                 "PUT ORDER REQ RECEIVED", "MODIFY PENDING", "AMO REQ RECEIVED"}


class KiteError(RuntimeError):
    def __init__(self, message: str, error_type: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status = status

    @property
    def is_token_error(self) -> bool:
        return self.error_type == "TokenException"


class KiteBroker(Broker):
    name = "kite"

    def __init__(self, api_key: str, api_secret: str | None = None,
                 access_token: str | None = None, *, product: str = "CNC",
                 order_style: str = "market", limit_buffer_pct: float = 0.5,
                 tag: str = "algoagent", managed_symbols: list[str] | None = None,
                 capital: float | None = None, session=None, timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("Kite API key is required")
        if order_style not in ("market", "limit"):
            raise ValueError("order_style must be 'market' or 'limit'")
        import requests

        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.product = product
        self.order_style = order_style
        self.limit_buffer_pct = limit_buffer_pct
        self.tag = tag[:20]
        self.managed = [normalize_symbol(s) for s in (managed_symbols or [])]
        self.capital = capital
        self.timeout = timeout
        self.session = session or requests.Session()
        self._tokens: dict[str, int] = {}
        self.placed: list[str] = []

    # ------------------------------------------------------------- session
    def login_url(self) -> str:
        return f"{LOGIN_URL}?api_key={self.api_key}&v=3"

    def generate_session(self, request_token: str) -> dict[str, Any]:
        """Exchange the ``request_token`` from the login redirect for an access token."""
        if not self.api_secret:
            raise ValueError("Kite API secret is required to create a session")
        checksum = hashlib.sha256(
            (self.api_key + request_token + self.api_secret).encode()).hexdigest()
        data = self._request("POST", "/session/token", data={
            "api_key": self.api_key, "request_token": request_token, "checksum": checksum})
        self.access_token = data["access_token"]
        return data

    def profile(self) -> dict[str, Any]:
        return self._request("GET", "/user/profile")

    # ----------------------------------------------------------- transport
    def _request(self, method: str, path: str, params=None, data=None) -> Any:
        headers = {"X-Kite-Version": "3", "User-Agent": "algo-agent"}
        if self.access_token:
            headers["Authorization"] = f"token {self.api_key}:{self.access_token}"
        resp = self.session.request(method, ROOT + path, params=params, data=data,
                                    headers=headers, timeout=self.timeout)
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype:
            body = resp.json()
            if body.get("status") == "error" or body.get("error_type") or resp.status_code >= 400:
                raise KiteError(body.get("message", resp.text[:200]),
                                body.get("error_type", ""), resp.status_code)
            return body.get("data")
        if resp.status_code >= 400:
            raise KiteError(resp.text[:200], status=resp.status_code)
        return resp.content

    # ------------------------------------------------------------ account
    def holdings(self) -> list[dict[str, Any]]:
        return self._request("GET", "/portfolio/holdings") or []

    def positions(self) -> list[dict[str, Any]]:
        return (self._request("GET", "/portfolio/positions") or {}).get("net", [])

    def margins(self) -> dict[str, Any]:
        return self._request("GET", "/user/margins") or {}

    def account(self) -> Account:
        qty: dict[str, float] = {}
        prices: dict[str, float] = {}
        # Delivery holdings (settled + T1) ...
        for h in self.holdings():
            sym = f"{h['exchange']}:{h['tradingsymbol']}"
            qty[sym] = qty.get(sym, 0.0) + float(h.get("quantity", 0)) + float(h.get("t1_quantity", 0))
            prices[sym] = float(h.get("last_price") or 0)
        # ... plus today's net positions (today's CNC buys/sells, MIS trades).
        for p in self.positions():
            sym = f"{p['exchange']}:{p['tradingsymbol']}"
            qty[sym] = qty.get(sym, 0.0) + float(p.get("quantity", 0))
            prices[sym] = float(p.get("last_price") or prices.get(sym, 0))
        equity_seg = self.margins().get("equity", {})
        cash = float(equity_seg.get("net", 0.0))
        managed = set(self.managed) if self.managed else set(qty)
        value = sum(q * prices.get(s, 0.0) for s, q in qty.items() if s in managed)
        return Account(cash=cash, equity=cash + value,
                       positions={s: q for s, q in qty.items() if q},
                       capital=self.capital, prices=prices)

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        """Last traded prices (needs the paid market-data plan)."""
        data = self._request("GET", "/quote/ltp", params=[("i", normalize_symbol(s)) for s in symbols])
        return {k: float(v["last_price"]) for k, v in (data or {}).items()}

    # ------------------------------------------------------------- orders
    def submit_order(self, symbol: str, qty: float, reason: str = "",
                     timestamp: pd.Timestamp | None = None,
                     ref_price: float | None = None) -> Order:
        qty = float(int(qty))
        if qty == 0:
            return Order(id="noop", symbol=symbol, qty=0.0, reason=reason, created=timestamp)
        exchange, tradingsymbol = parse_symbol(symbol)
        side = 1.0 if qty > 0 else -1.0
        params: dict[str, Any] = {
            "exchange": exchange, "tradingsymbol": tradingsymbol,
            "transaction_type": "BUY" if side > 0 else "SELL",
            "quantity": int(abs(qty)), "product": self.product, "validity": "DAY",
            "tag": self.tag,
        }
        if self.order_style == "market":
            params.update(order_type="MARKET", market_protection=-1)
        else:
            price = self._limit_price(symbol, side, ref_price)
            params.update(order_type="LIMIT", price=f"{price:.2f}")
        log.info("kite order %s", params)
        data = self._request("POST", "/orders/regular", data=params)
        order_id = str(data["order_id"])
        self.placed.append(order_id)
        return Order(id=order_id, symbol=normalize_symbol(symbol), qty=qty, reason=reason,
                     created=timestamp)

    def _limit_price(self, symbol: str, side: float, ref_price: float | None) -> float:
        price = None
        try:
            price = self.ltp([symbol]).get(normalize_symbol(symbol))
        except KiteError as exc:  # no market-data subscription
            log.debug("ltp unavailable (%s); using reference price", exc)
        price = price or ref_price
        if not price:
            raise KiteError(f"no price available to place a limit order for {symbol}")
        return round_to_tick(price * (1 + side * self.limit_buffer_pct / 100), side=side)

    def orders(self) -> list[dict[str, Any]]:
        return self._request("GET", "/orders") or []

    def cancel_all(self) -> None:
        """Cancel *this app's* open orders only (matched by tag), never yours."""
        for o in self.orders():
            if o.get("status") in OPEN_STATUSES and o.get("tag") == self.tag:
                try:
                    self._request("DELETE", f"/orders/{o.get('variety', 'regular')}/{o['order_id']}")
                except KiteError as exc:
                    log.warning("could not cancel %s: %s", o["order_id"], exc)

    # ---------------------------------------------------------------- data
    def instrument_token(self, symbol: str) -> int:
        symbol = normalize_symbol(symbol)
        if symbol not in self._tokens:
            exchange, _ = parse_symbol(symbol)
            raw = self._request("GET", f"/instruments/{exchange}")
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            for row in csv.DictReader(io.StringIO(text)):
                self._tokens[f"{row['exchange']}:{row['tradingsymbol']}"] = int(row["instrument_token"])
        if symbol not in self._tokens:
            raise KiteError(f"unknown instrument {symbol}")
        return self._tokens[symbol]

    def historical(self, symbol: str, interval: str = "day", days: int = 400) -> pd.DataFrame:
        """Candles via Kite's historical API (needs the paid plan)."""
        token = self.instrument_token(symbol)
        to = pd.Timestamp.now(tz=IST)
        # Kite limits each request's span per interval; stay within it.
        max_days = {"minute": 60, "5minute": 100, "15minute": 200, "30minute": 200,
                    "60minute": 400, "day": 2000}[interval]
        frm = to - pd.Timedelta(days=min(days, max_days))
        data = self._request("GET", f"/instruments/historical/{token}/{interval}", params={
            "from": frm.strftime("%Y-%m-%d %H:%M:%S"), "to": to.strftime("%Y-%m-%d %H:%M:%S")})
        candles = (data or {}).get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"][:len(candles[0])])
        idx = pd.to_datetime(df.pop("date"))
        idx = idx.dt.tz_convert(IST) if idx.dt.tz is not None else idx.dt.tz_localize(IST)
        df.index = pd.DatetimeIndex(idx)
        if interval == "day":
            df.index = pd.DatetimeIndex(df.index.date).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        return df[["open", "high", "low", "close", "volume"]].astype(float)
