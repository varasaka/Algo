"""Persistent app settings and secrets, stored as JSON in the data directory."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..risk import RiskConfig

BROKERS = ("paper", "kite", "paytm")
MODES = ("paper", "shadow", "live")
TIMEFRAMES = ("day", "60minute", "30minute", "15minute", "5minute")


def default_data_dir() -> Path:
    return Path(os.environ.get("ALGO_AGENT_HOME", Path.home() / ".algo_agent"))


@dataclass
class Settings:
    broker: str = "paper"                 # paper | kite | paytm
    mode: str = "paper"                   # paper | shadow | live
    symbols: list[str] = field(default_factory=lambda: ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"])
    strategy: str = "momentum"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=lambda: {
        "max_position_pct": 0.34, "target_vol": 0.20, "risk_per_trade": 0.01,
        "stop_atr_mult": 3.0, "max_drawdown": 0.15, "daily_loss_limit": 0.03})
    timeframe: str = "day"
    run_at: str = "09:20"                 # daily timeframe: when to act (IST)
    square_off_at: str = "15:10"          # intraday (MIS): flatten before this (IST)
    capital: float = 100_000.0            # money the bot may use (₹)
    product: str = "CNC"                  # CNC (delivery) | MIS (intraday)
    order_style: str = "market"           # market (with protection) | limit
    limit_buffer_pct: float = 0.5
    data_source: str = "yahoo"            # yahoo (free) | broker (paid Kite data)
    holidays: list[str] = field(default_factory=list)
    kite_api_key: str = ""
    kite_api_secret: str = ""
    paytm_api_key: str = ""
    paytm_api_secret: str = ""
    paytm_product: str = "C"
    paytm_source: str = "W"
    live_confirmed: bool = False          # set only via the explicit confirmation flow
    pin_hash: str = ""

    SECRET_FIELDS = ("kite_api_secret", "paytm_api_secret", "pin_hash")

    def validate(self) -> None:
        if self.broker not in BROKERS:
            raise ValueError(f"broker must be one of {BROKERS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.broker == "paper" and self.mode != "paper":
            raise ValueError("choose Zerodha or Paytm Money to use shadow/live mode")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        if self.timeframe != "day" and self.product == "CNC":
            pass  # allowed (swing on intraday bars), just unusual
        if self.product not in ("CNC", "MIS"):
            raise ValueError("product must be CNC or MIS")
        if self.order_style not in ("market", "limit"):
            raise ValueError("order_style must be market or limit")
        if self.data_source not in ("yahoo", "broker"):
            raise ValueError("data_source must be yahoo or broker")
        if not self.symbols:
            raise ValueError("add at least one symbol")
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        for t in (self.run_at, self.square_off_at):
            dt.time.fromisoformat(t)
        RiskConfig.from_dict(self.risk)

    def public(self) -> dict[str, Any]:
        """Settings safe to send to the browser (secrets masked)."""
        d = asdict(self)
        for k in self.SECRET_FIELDS:
            d[k] = "********" if d[k] else ""
        d["pin_set"] = bool(self.pin_hash)
        d.pop("pin_hash")
        return d

    def update(self, changes: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(self)}
        merged = asdict(self)
        for k, v in changes.items():
            if k not in known or k in ("pin_hash", "live_confirmed"):
                continue
            if k in self.SECRET_FIELDS and v in ("********", None):
                continue  # untouched masked secret
            merged[k] = v
        if isinstance(merged.get("symbols"), str):
            merged["symbols"] = [s for s in merged["symbols"].replace("\n", ",").split(",")]
        from ..india.market import normalize_symbol
        merged["symbols"] = [normalize_symbol(s) for s in merged["symbols"] if str(s).strip()]
        merged["capital"] = float(merged["capital"])
        merged["limit_buffer_pct"] = float(merged["limit_buffer_pct"])
        new = Settings(**merged)
        # Any change of broker or mode away from live revokes the live confirmation.
        new.live_confirmed = self.live_confirmed and new.mode == "live" and new.broker == self.broker
        new.validate()
        return new


class Store:
    """Files in the data directory: settings, broker tokens, state, journal."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else default_data_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def _path(self, name: str) -> Path:
        return self.root / name

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self._path(name)
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return default

    def write_json(self, name: str, data: Any) -> None:
        p = self._path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(p)

    def load_settings(self) -> Settings:
        raw = self.read_json("settings.json", {}) or {}
        known = {f.name for f in fields(Settings)}
        s = Settings(**{k: v for k, v in raw.items() if k in known})
        try:
            s.validate()
        except ValueError:
            s = Settings()
        return s

    def save_settings(self, s: Settings) -> None:
        self.write_json("settings.json", asdict(s))

    # tokens are valid for one trading day; store the IST date with them
    def save_token(self, broker: str, token: str) -> None:
        from ..india.market import now_ist
        tokens = self.read_json("tokens.json", {}) or {}
        tokens[broker] = {"access_token": token, "date": now_ist().date().isoformat()}
        self.write_json("tokens.json", tokens)

    def load_token(self, broker: str) -> str | None:
        """Today's token, treating Kite/Paytm's ~06:00 IST expiry as a day boundary."""
        from ..india.market import now_ist
        entry = (self.read_json("tokens.json", {}) or {}).get(broker)
        if not entry:
            return None
        now = now_ist()
        issued = dt.date.fromisoformat(entry["date"])
        session_day = now.date() if now.time() >= dt.time(6, 0) else now.date() - dt.timedelta(days=1)
        return entry["access_token"] if issued >= session_day else None

    def clear_token(self, broker: str) -> None:
        tokens = self.read_json("tokens.json", {}) or {}
        tokens.pop(broker, None)
        self.write_json("tokens.json", tokens)

    def secret_key(self) -> bytes:
        p = self._path("server.key")
        if not p.exists():
            p.write_bytes(secrets.token_bytes(32))
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
        return p.read_bytes()

    def append_journal(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        with self._path("journal.jsonl").open("a") as fh:
            for e in entries:
                fh.write(json.dumps(e, default=str) + "\n")

    def tail_journal(self, n: int = 200) -> list[dict[str, Any]]:
        p = self._path("journal.jsonl")
        if not p.exists():
            return []
        lines = p.read_text().splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


def hash_pin(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 200_000).hex()
