"""Command line interface: ``python -m algo_agent <command> ...``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .agent import TradingAgent
from .backtest import run_backtest
from .broker import AlpacaBroker, PaperBroker
from .config import AgentConfig
from .data import load_data
from .live import LiveRunner, PollingFeed, ReplayFeed
from .metrics import format_metrics, infer_periods_per_year
from .optimize import grid_search, parse_grid
from .risk import RiskManager
from .strategies import REGISTRY, create_strategy


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="TOML config file")
    p.add_argument("-s", "--symbols", help="comma-separated symbols (overrides config)")
    p.add_argument("--source", choices=["synthetic", "csv", "yahoo"], help="data source")
    p.add_argument("--data-dir", help="directory of <SYMBOL>.csv files")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--strategy", help=f"one of {sorted(REGISTRY)}")
    p.add_argument("-p", "--param", action="append", default=[],
                   help="strategy param key=value (repeatable)")
    p.add_argument("--cash", type=float)
    p.add_argument("--allow-short", action="store_true")


def _config(args: argparse.Namespace) -> AgentConfig:
    cfg = AgentConfig.load(args.config) if args.config else AgentConfig()
    if args.symbols:
        cfg.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for attr, name in (("source", "data_source"), ("data_dir", "data_dir"), ("start", "start"),
                       ("end", "end"), ("cash", "initial_cash")):
        if getattr(args, attr, None) is not None:
            setattr(cfg, name, getattr(args, attr))
    if args.strategy:
        if args.strategy != cfg.strategy:
            cfg.strategy_params = {}
        cfg.strategy = args.strategy
    from .optimize import _coerce
    for kv in args.param:
        k, _, v = kv.partition("=")
        cfg.strategy_params[k] = _coerce(v)
    if args.allow_short:
        cfg.risk.allow_short = True
        if "allow_short" in create_strategy(cfg.strategy).params:
            cfg.strategy_params["allow_short"] = True
    return cfg


def _data(cfg: AgentConfig) -> dict[str, pd.DataFrame]:
    return load_data(cfg.data_source, cfg.symbols, start=cfg.start, end=cfg.end,
                     data_dir=cfg.data_dir, n_bars=cfg.n_bars, seed=cfg.seed)


def _bt_kwargs(cfg: AgentConfig) -> dict:
    return {"initial_cash": cfg.initial_cash, "commission_bps": cfg.commission_bps,
            "slippage_bps": cfg.slippage_bps}


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    data = _data(cfg)
    strategy = create_strategy(cfg.strategy, **dict(cfg.strategy_params))
    res = run_backtest(data, strategy, cfg.risk, **_bt_kwargs(cfg))
    first, last = res.equity.index[0].date(), res.equity.index[-1].date()
    print(f"\n{strategy!r}\nsymbols={cfg.symbols}  {first} -> {last}  bars={len(res.equity)}")
    print(f"final equity: {res.equity.iloc[-1]:,.2f} (start {cfg.initial_cash:,.2f})\n")
    print(format_metrics(res.metrics))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res.equity.to_csv(out / "equity.csv")
        res.trades_frame().to_csv(out / "trades.csv", index=False)
        res.journal_frame().to_csv(out / "journal.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(res.metrics, indent=2))
        print(f"\nresults written to {out}/")
    if args.plot:
        _plot(res, args.plot)
    return 0


def _plot(res, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    res.equity.plot(ax=a1, title="Equity")
    (res.equity / res.equity.cummax() - 1).plot(ax=a2, color="tab:red", title="Drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot saved to {path}")


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = _config(args)
    data = _data(cfg)
    rows = []
    for name in sorted(REGISTRY):
        params = {"allow_short": True} if args.allow_short and "allow_short" in \
            REGISTRY[name].default_params() else {}
        res = run_backtest(data, create_strategy(name, **params), cfg.risk, **_bt_kwargs(cfg))
        rows.append({"strategy": name, **res.metrics})
    table = pd.DataFrame(rows).set_index("strategy")
    cols = [c for c in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "trades",
                        "win_rate", "exposure", "benchmark_return") if c in table]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(table[cols].sort_values("sharpe", ascending=False).to_string())
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    cfg = _config(args)
    data = _data(cfg)
    grid = parse_grid(args.grid)
    table = grid_search(data, cfg.strategy, grid, cfg.risk, train_frac=args.train_frac,
                        metric=args.metric, top=args.top, base_params=cfg.strategy_params,
                        **_bt_kwargs(cfg))
    keep = list(grid) + [c for c in table.columns if c in (
        "is_total_return", "is_sharpe", "is_max_drawdown", "oos_total_return", "oos_sharpe",
        "oos_max_drawdown", "oos_trades")]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"train fraction {args.train_frac:.0%}; ranked by in-sample {args.metric}\n")
        print(table[keep].to_string(index=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    strategy = create_strategy(cfg.strategy, **dict(cfg.strategy_params))
    broker_name = args.broker or cfg.broker
    if broker_name == "alpaca":
        broker = AlpacaBroker(paper=not args.live)
    else:
        broker = PaperBroker(cash=cfg.initial_cash, commission_bps=cfg.commission_bps,
                             slippage_bps=cfg.slippage_bps)

    if args.feed == "replay":
        data = _data(cfg)
        feed = ReplayFeed(data, warmup_bars=args.warmup)
        poll = args.poll if args.poll is not None else 0.0
        ppy = infer_periods_per_year(feed.index)
    else:
        source = broker if isinstance(broker, AlpacaBroker) else AlpacaBroker(paper=True)
        timeframe = args.timeframe
        duration = {"1Day": pd.Timedelta(days=1), "1Hour": pd.Timedelta(hours=1),
                    "15Min": pd.Timedelta(minutes=15), "5Min": pd.Timedelta(minutes=5),
                    "1Min": pd.Timedelta(minutes=1)}[timeframe]
        feed = PollingFeed(lambda s: source.latest_bars(s, timeframe), cfg.symbols, duration)
        poll = args.poll if args.poll is not None else cfg.poll_seconds
        ppy = 252 if timeframe == "1Day" else int(252 * pd.Timedelta(hours=6.5) / duration)

    agent = TradingAgent(strategy, broker, cfg.symbols, RiskManager(cfg.risk),
                         periods_per_year=ppy)
    runner = LiveRunner(agent, feed, poll_seconds=poll, journal_path=args.journal or cfg.journal_path)
    print(f"running {strategy!r} on {cfg.symbols} via {broker_name} broker / {args.feed} feed "
          f"(Ctrl+C to stop)")
    runner.run(max_steps=args.max_steps)
    acct = broker.account()
    print(f"\nfinal equity {acct.equity:,.2f}  cash {acct.cash:,.2f}  positions {acct.positions}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="algo-agent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="backtest one strategy")
    _add_common(p)
    p.add_argument("-o", "--out", help="directory to write equity/trades/journal CSVs")
    p.add_argument("--plot", help="save an equity/drawdown PNG (needs matplotlib)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("compare", help="backtest every built-in strategy on the same data")
    _add_common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("optimize", help="grid-search params with an out-of-sample check")
    _add_common(p)
    p.add_argument("-g", "--grid", required=True, help='e.g. "fast=10,20,30 slow=50,100,200"')
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--metric", default="sharpe")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("run", help="run the agent loop (paper by default)")
    _add_common(p)
    p.add_argument("--broker", choices=["paper", "alpaca"])
    p.add_argument("--feed", choices=["replay", "alpaca"], default="replay")
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Day", "1Hour", "15Min", "5Min", "1Min"])
    p.add_argument("--poll", type=float, help="seconds between polls")
    p.add_argument("--warmup", type=int, default=250, help="replay: bars used for warm-up")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--journal", help="JSONL file for agent decisions")
    p.add_argument("--live", action="store_true",
                   help="alpaca: use the LIVE account (also needs ALGO_AGENT_ALLOW_LIVE=1)")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
