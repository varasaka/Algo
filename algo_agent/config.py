"""TOML configuration (see ``configs/*.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk import RiskConfig


@dataclass
class AgentConfig:
    symbols: list[str] = field(default_factory=lambda: ["SYN"])
    data_source: str = "synthetic"         # synthetic | csv | yahoo
    data_dir: str | None = None
    start: str | None = None
    end: str | None = None
    n_bars: int = 1500
    seed: int = 7
    strategy: str = "sma_crossover"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    broker: str = "paper"                  # paper | alpaca
    poll_seconds: float = 60.0
    journal_path: str | None = "state/journal.jsonl"

    @classmethod
    def load(cls, path: str | Path) -> "AgentConfig":
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentConfig":
        agent = dict(raw.get("agent", {}))
        data = raw.get("data", {})
        strategy = raw.get("strategy", {})
        execution = raw.get("execution", {})
        cfg = cls()
        mapping = {
            **{k: agent[k] for k in ("symbols", "initial_cash", "broker", "poll_seconds",
                                     "journal_path") if k in agent},
            **{k: data[k] for k in ("data_dir", "start", "end", "n_bars", "seed") if k in data},
            **{k: execution[k] for k in ("commission_bps", "slippage_bps") if k in execution},
        }
        if "source" in data:
            mapping["data_source"] = data["source"]
        if "name" in strategy:
            mapping["strategy"] = strategy["name"]
        params = {k: v for k, v in strategy.items() if k not in ("name", "params")}
        params.update(strategy.get("params", {}))
        if params:
            mapping["strategy_params"] = params
        for k, v in mapping.items():
            setattr(cfg, k, v)
        cfg.risk = RiskConfig.from_dict(raw.get("risk"))
        return cfg
