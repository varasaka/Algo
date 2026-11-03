"""Web app server (standard library only, so it also runs on Android/Termux).

    python -m algo_agent app                 # http://127.0.0.1:8000 on this machine
    python -m algo_agent app --host 0.0.0.0  # reachable from your tablet on the LAN

When listening beyond localhost a PIN is mandatory: the app can place real
trades, so nobody on your Wi-Fi should be able to open it without one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import secrets
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..backtest import run_backtest
from ..risk import RiskConfig
from ..strategies import REGISTRY, create_strategy
from .engine import Engine
from .settings import Store, hash_pin

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
COOKIE = "algo_session"
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")


class App:
    def __init__(self, store: Store, engine: Engine, require_pin: bool) -> None:
        self.store = store
        self.engine = engine
        self.require_pin = require_pin
        self.key = store.secret_key()
        self._fails: dict[str, list[float]] = {}

    # ------------------------------------------------------------ auth
    def pin_set(self) -> bool:
        return bool(self.engine.settings.pin_hash)

    def session_value(self) -> str:
        return hmac.new(self.key, self.engine.settings.pin_hash.encode(), hashlib.sha256).hexdigest()

    def authed(self, cookies: dict[str, str]) -> bool:
        if not self.pin_set():
            return not self.require_pin
        return hmac.compare_digest(cookies.get(COOKIE, ""), self.session_value())

    def throttled(self, ip: str) -> bool:
        now = time.time()
        recent = [t for t in self._fails.get(ip, []) if now - t < 300]
        self._fails[ip] = recent
        return len(recent) >= 5

    def check_pin(self, pin: str, ip: str) -> bool:
        if self.throttled(ip):
            return False
        ok = hmac.compare_digest(hash_pin(pin, self.key), self.engine.settings.pin_hash)
        if not ok:
            self._fails.setdefault(ip, []).append(time.time())
        return ok

    def set_pin(self, pin: str) -> None:
        if not (pin.isdigit() and 4 <= len(pin) <= 12):
            raise ValueError("PIN must be 4-12 digits")
        s = self.engine.settings
        s.pin_hash = hash_pin(pin, self.key)
        self.store.save_settings(s)

    # ------------------------------------------------------------ api
    def strategies(self) -> list[dict[str, Any]]:
        out = []
        for name, cls in sorted(REGISTRY.items()):
            out.append({"name": name, "doc": (cls.__doc__ or "").strip().split("\n")[0],
                        "params": cls.default_params()})
        return out

    def backtest(self, body: dict[str, Any]) -> dict[str, Any]:
        s = self.engine.settings
        symbols = body.get("symbols") or s.symbols
        if isinstance(symbols, str):
            symbols = [x.strip() for x in symbols.split(",") if x.strip()]
        strategy = body.get("strategy") or s.strategy
        params = body.get("strategy_params", s.strategy_params if strategy == s.strategy else {})
        risk = RiskConfig.from_dict(body.get("risk") or s.risk)
        timeframe = body.get("timeframe") or s.timeframe
        data = {}
        for sym in symbols:
            df = self.engine.fetch(sym, timeframe)
            data[sym] = df.tz_localize(None) if df.index.tz is not None else df
        res = run_backtest(data, create_strategy(strategy, **params), risk,
                           initial_cash=float(body.get("capital") or s.capital),
                           commission_bps=3.0, slippage_bps=5.0)
        eq = res.equity
        step = max(1, len(eq) // 400)
        trades = res.trades_frame().tail(50)
        return {
            "metrics": {k: (None if v != v or v in (float("inf"), float("-inf")) else v)
                        for k, v in res.metrics.items()},
            "equity": [{"time": t.isoformat(), "equity": round(v, 2)}
                       for t, v in eq.iloc[::step].items()],
            "trades": json.loads(trades.to_json(orient="records", date_format="iso")) if len(trades) else [],
            "range": [eq.index[0].isoformat(), eq.index[-1].isoformat()],
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AlgoAgent/1.0"

        def log_message(self, fmt, *args):  # route through logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ------------------------------------------------- helpers
        def _cookies(self) -> dict[str, str]:
            out = {}
            for part in self.headers.get("Cookie", "").split(";"):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    out[k] = v
            return out

        def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200, headers: dict | None = None):
            self._send(status, json.dumps(data, default=str).encode(), "application/json",
                       {"Cache-Control": "no-store", **(headers or {})})

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ValueError("request too large")
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw or b"{}")

        def _same_origin(self) -> bool:
            """Reject cross-site POSTs (CSRF): browsers send Origin on POST."""
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return urlparse(origin).netloc == self.headers.get("Host")

        # ------------------------------------------------- routing
        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path.startswith("/api/"):
                return self._api("GET", path, parse_qs(url.query))
            if path.startswith("/broker/") and path.endswith("/callback"):
                return self._callback(parse_qs(url.query))
            return self._static(path)

        def do_POST(self):
            url = urlparse(self.path)
            if not self._same_origin():
                return self._json({"error": "cross-origin request blocked"}, 403)
            return self._api("POST", url.path, parse_qs(url.query))

        def _static(self, path: str):
            if path in ("/", ""):
                path = "/index.html"
            target = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                target = STATIC / "index.html"  # SPA fallback
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            headers = {"Cache-Control": "no-cache"}
            if target.name == "sw.js":
                headers["Service-Worker-Allowed"] = "/"
            self._send(200, target.read_bytes(), ctype, headers)

        def _callback(self, q: dict[str, list[str]]):
            """Broker login redirect: ?request_token=... (Kite) / ?requestToken=... (Paytm)."""
            if not app.authed(self._cookies()):
                return self._redirect("/?login_error=Unlock+the+app+then+log+in+again")
            token = (q.get("request_token") or q.get("requestToken") or [""])[0]
            if not token:
                msg = (q.get("message") or q.get("status") or ["login cancelled"])[0]
                return self._redirect(f"/?login_error={msg}")
            try:
                app.engine.complete_login(token)
                return self._redirect("/?login=ok#broker")
            except Exception as exc:
                return self._redirect(f"/?login_error={str(exc)[:120]}#broker")

        def _redirect(self, where: str):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _api(self, method: str, path: str, q: dict[str, list[str]]):
            try:
                cookies = self._cookies()
                eng = app.engine
                # -- unauthenticated endpoints
                if path == "/api/session" and method == "GET":
                    return self._json({"authed": app.authed(cookies), "pin_set": app.pin_set(),
                                       "pin_required": app.require_pin})
                if path == "/api/unlock" and method == "POST":
                    pin = str(self._body().get("pin", ""))
                    ip = self.client_address[0]
                    if app.throttled(ip):
                        return self._json({"error": "too many attempts; wait 5 minutes"}, 429)
                    if app.pin_set() and app.check_pin(pin, ip):
                        cookie = f"{COOKIE}={app.session_value()}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000"
                        return self._json({"ok": True}, headers={"Set-Cookie": cookie})
                    return self._json({"error": "wrong PIN"}, 401)
                if path == "/api/pin" and method == "POST":
                    # First-time PIN setup is allowed without auth; changing it needs auth.
                    if app.pin_set() and not app.authed(cookies):
                        return self._json({"error": "unlock first"}, 401)
                    if not app.pin_set() and app.require_pin and \
                            self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json({"error": "set the first PIN on the computer running "
                                                    "the app (or start it with --pin)"}, 403)
                    app.set_pin(str(self._body().get("pin", "")))
                    cookie = f"{COOKIE}={app.session_value()}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000"
                    return self._json({"ok": True}, headers={"Set-Cookie": cookie})
                if not app.authed(cookies):
                    return self._json({"error": "locked"}, 401)

                # -- authenticated endpoints
                if method == "GET":
                    if path == "/api/status":
                        return self._json(eng.status())
                    if path == "/api/settings":
                        return self._json(eng.settings.public())
                    if path == "/api/strategies":
                        return self._json(app.strategies())
                    if path == "/api/journal":
                        n = int((q.get("n") or ["200"])[0])
                        return self._json(app.store.tail_journal(min(n, 2000)))
                    if path == "/api/broker/login-url":
                        return self._json({"url": eng.login_url()})
                    return self._json({"error": "not found"}, 404)

                body = self._body()
                if path == "/api/settings":
                    return self._json(eng.update_settings(body).public())
                if path == "/api/start":
                    eng.start()
                    return self._json({"ok": True})
                if path == "/api/stop":
                    eng.stop()
                    return self._json({"ok": True})
                if path == "/api/run-now":
                    if not eng.running:
                        eng.start()
                    return self._json({"entries": eng.run_cycle()})
                if path == "/api/kill":
                    return self._json(eng.kill_switch())
                if path == "/api/confirm-live":
                    eng.confirm_live(str(body.get("phrase", "")))
                    return self._json({"ok": True})
                if path == "/api/broker/token":
                    return self._json(eng.complete_login(str(body.get("request_token", ""))))
                if path == "/api/broker/logout":
                    eng.logout()
                    return self._json({"ok": True})
                if path == "/api/broker/test":
                    return self._json(eng.test_connection())
                if path == "/api/backtest":
                    return self._json(app.backtest(body))
                return self._json({"error": "not found"}, 404)
            except (ValueError, RuntimeError) as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:
                log.error("API error on %s: %s\n%s", path, exc, traceback.format_exc())
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, data_dir: str | None = None,
          open_browser: bool = True, engine: Engine | None = None,
          pin: str | None = None) -> ThreadingHTTPServer:
    store = Store(data_dir)
    engine = engine or Engine(store)
    local_only = host in ("127.0.0.1", "localhost", "::1")
    app = App(store, engine, require_pin=not local_only)
    if pin:
        app.set_pin(pin)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.app = app  # type: ignore[attr-defined]
    if getattr(engine, "resume_requested", False):
        try:
            engine.start()
            log.info("resumed agent after restart")
        except Exception as exc:
            engine.last_error = f"could not resume: {exc}"
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"\nAlgo Agent is running at {url}")
    if not local_only:
        print("Listening on the network: open http://<this-computer's-IP>:%d on your tablet." % port)
        print("A PIN is required for access from other devices.")
    print(f"Data folder: {store.root}\nPress Ctrl+C to stop.\n")
    if open_browser:
        threading.Timer(1.0, _open, args=(url,)).start()
    return httpd


def _open(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def run(host: str = "127.0.0.1", port: int = 8000, data_dir: str | None = None,
        open_browser: bool = True, pin: str | None = None) -> None:
    httpd = serve(host, port, data_dir, open_browser, pin=pin)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        # Save state *with* the running flag so a restart resumes the agent.
        httpd.app.engine._save_state()  # type: ignore[attr-defined]
        httpd.app.engine.shutdown()  # type: ignore[attr-defined]
        httpd.server_close()
