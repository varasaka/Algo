# algo-agent

A modular algorithmic trading agent in Python. One agent drives
**backtests**, **paper trading** and **live trading (Alpaca)**. Only the broker
and the bar feed change between them, so what you backtest is what you run.

```
 bars ──► TradingAgent ──────────────────────────────► Broker
          │ 1. strategy  → target exposure in [-1, 1]   PaperBroker  (simulated fills)
          │ 2. risk      → position size, stops, caps   AlpacaBroker (paper / live)
          │ 3. orders    → target − current position
          └ 4. journal   → every decision + its reason
```

Core dependencies are only `numpy` and `pandas`.

## Quick start

```bash
pip install -e ".[dev]"            # or just: pip install numpy pandas

python -m algo_agent backtest                         # SMA crossover on synthetic data
python -m algo_agent backtest -c configs/default.toml -o results/
python -m algo_agent compare -s AAA,BBB,CCC           # every strategy, same data
python -m algo_agent optimize --strategy sma -g "fast=10,20,30 slow=50,100,150"
python -m algo_agent run --strategy momentum --max-steps 50   # replay the live loop
pytest
```

Use real data from CSV (`data/<SYMBOL>.csv` with date/open/high/low/close/volume
columns) or Yahoo Finance (`pip install yfinance`):

```bash
python -m algo_agent backtest --source csv --data-dir data -s SPY
python -m algo_agent backtest --source yahoo -s SPY,QQQ --start 2015-01-01 --strategy momentum
```

## Strategies

| name (alias) | idea |
|---|---|
| `sma_crossover` (`sma`) | trend following: long while fast SMA > slow SMA |
| `momentum` (`macd`) | MACD histogram + long-term trend filter, scaled 0 / 0.5 / 1 |
| `rsi_mean_reversion` (`rsi`) | buy RSI dips inside an uptrend, exit on recovery |
| `bollinger_reversion` (`bollinger`) | fade closes outside the bands, exit at the mid band |
| `ensemble` | weighted blend of any of the above |

Pass params with `-p key=value` (e.g. `-p fast=10 -p slow=40`) or in a config
file. `--allow-short` enables short signals and short positions.

**Adding a strategy:** subclass `Strategy`, implement
`generate_signals(df) -> Series` in [-1, 1] using only data up to each bar,
and add it to `REGISTRY` in `algo_agent/strategies/__init__.py`. The parametrised
`test_no_lookahead` test then checks it automatically for look-ahead bias.

## Risk management (`[risk]` in config)

- **Volatility targeting.** Each symbol gets `max_gross_exposure / n_symbols`
  of equity, scaled by `target_vol / realised_vol`, capped at `max_position_pct`.
- **Risk-per-trade cap.** Size is limited so that hitting the stop loses at most
  `risk_per_trade` of equity.
- **ATR trailing stop.** `stop_atr_mult` ATRs from price. It only ever
  tightens. After a stop-out the symbol sits out `cooldown_bars`.
- **Gross exposure cap** across all symbols.
- **Circuit breakers.** Flatten and halt at a `max_drawdown` from peak; flatten
  and block entries for the day after `daily_loss_limit`.
- **Churn control.** Rebalances smaller than `rebalance_threshold` × equity are skipped.

## Execution model and honesty checks

- Decisions use a bar's *close*. Orders fill at the *next bar's open*, plus
  `slippage_bps` against you and `commission_bps` of notional.
- The backtester precomputes indicators for speed. A test checks that this gives
  the same equity curve as the bar-by-bar path that live trading uses.
- Another test checks that the live loop (`LiveRunner` + `ReplayFeed`) ends with
  the same equity as the backtest.
- `optimize` fits on the first `--train-frac` of the data and reports
  out-of-sample results for the top parameter sets. Trust the OOS columns.

## Paper and live trading with Alpaca

```bash
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
pip install requests
python -m algo_agent run -c configs/alpaca_paper.toml --broker alpaca --feed alpaca
```

- The agent polls for **completed** bars (`--timeframe 1Day|1Hour|15Min|5Min|1Min`),
  warms up on recent history, and logs each decision to a JSONL journal.
- Real-money trading needs **both** `--live` and `ALGO_AGENT_ALLOW_LIVE=1`.
- `--broker paper --feed alpaca` trades live Alpaca data against the local simulator.

## Layout

```
algo_agent/
  agent.py        TradingAgent: the decide/act loop
  strategies/     Strategy base + built-ins + registry
  risk.py         RiskConfig / RiskManager
  broker.py       PaperBroker, AlpacaBroker, Order/Fill/Trade
  backtest.py     event-driven backtester
  live.py         LiveRunner, ReplayFeed, PollingFeed
  optimize.py     grid search with out-of-sample check
  metrics.py      Sharpe, Sortino, drawdown, win rate, ...
  data.py         synthetic / CSV / Yahoo loaders
  indicators.py   SMA, EMA, RSI, ATR, MACD, Bollinger, vol
  config.py, cli.py
configs/          example TOML configs
tests/            pytest suite
```

## Disclaimer

This is educational software. Backtest results, and results on synthetic data
in particular, say nothing about future returns. Paper trade first and use it
at your own risk.
