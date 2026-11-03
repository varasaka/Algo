# algo-agent

A modular algorithmic trading agent in Python, with an **app for Indian
brokers (Zerodha Kite, Paytm Money)** that trades automatically. See
[Auto-trading app](#auto-trading-app-zerodha--paytm-money) below.

The core engine: one agent drives **backtests**, **paper trading** and **live trading (Alpaca)**. Only the broker
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

## Auto-trading app (Zerodha / Paytm Money)

A web app with a dashboard, broker login, strategy and risk settings,
backtesting and an activity log. It runs on a PC, Mac, Linux server or Android
tablet (via Termux). Your phone or tablet browser can open it and install it
like an app.

### Install and start

| Device | How |
|---|---|
| **Windows** | Install Python 3.10+ from python.org (tick "Add to PATH"), then double-click `install\start-windows.bat` |
| **macOS / Linux** | `./install/start.sh` |
| **Android tablet (on the tablet itself)** | Install **Termux** from F-Droid, then `pkg install python git && git clone <this repo> && cd Algo && ./install/start.sh`, then open `http://127.0.0.1:8000` in Chrome and choose **Add to Home screen / Install app** |
| **Tablet as a screen for a PC/server** | On the PC: `./install/start.sh --host 0.0.0.0 --pin 482193`. On the tablet: open `http://<PC-IP>:8000` and add it to the home screen |
| **VPS with a static IP (recommended for Zerodha live)** | `docker build -t algo-agent . && docker run -d --restart unless-stopped -p 8000:8000 -v algo-data:/data -e ALGO_AGENT_PIN=482193 algo-agent`, behind HTTPS (e.g. Caddy) |

Or just run `pip install -e .` and then `python -m algo_agent app`.
Settings, tokens and logs are stored in `~/.algo_agent`. Set `ALGO_AGENT_DEMO_DATA=1`
to try the whole app offline on synthetic prices.

### Daily workflow

1. **Broker tab:** pick Zerodha or Paytm Money and enter your API key and secret
   from the developer console. Set the app's redirect URL there to the address
   shown in the app.
2. **Every trading day,** tap **Log in to broker**. You log in on the
   broker's own page and get redirected back. Brokers require this daily, and
   the app never sees your password.
3. **Strategy tab:** choose symbols (`NSE:RELIANCE, NSE:TCS`), the capital the bot
   may use, the strategy and risk limits, then **Backtest** it.
4. **Dashboard:** press **Start auto-trading**. On daily candles the bot acts once
   a day at the run time (default 09:20 IST). On intraday candles it acts after
   every candle and squares off MIS positions at 15:10.

### Modes: always go Paper → Shadow → Live

| Mode | What happens |
|---|---|
| **Paper** | Fake money, real prices. No broker needed. |
| **Shadow** | Reads your **real** holdings and cash, and logs the orders it *would* place, but sends nothing. Use it for a few weeks and compare with your broker app. |
| **Live** | Sends real orders automatically. Needs typing `I UNDERSTAND THE RISK`. Changing broker or mode turns it off again. |

### Safety built in

- **Capital cap:** the bot sizes positions from the capital you allocate, not your whole account.
- **Hands off your own trades:** the bot only trades its own symbols, and only
  cancels orders it placed (tag `algoagent` on Kite).
- **Kill switch:** closes all positions the bot manages and stops it.
- **Risk limits:** stop-losses, a daily loss limit and a max-drawdown halt (see Risk management above).
- **PIN lock:** required whenever the app is reachable from other devices. Wrong
  PINs are rate-limited, and cross-site requests are blocked.
- **Resumes after restart:** if the app restarts mid-day it carries on where it left off.

### Broker-specific notes (check current rules before going live)

- **Zerodha Kite Connect.** Endpoints and the login checksum match Zerodha's official SDK.
  - **Static IP:** SEBI's retail-algo rules mean Kite only accepts *order* calls
    from the static IP registered in your developer console. Home broadband and
    mobile data usually don't have one, so run the app on a VPS with a static
    IP, or get one from your ISP.
  - **Market orders:** sent with `market_protection=-1` (required for API market
    orders). Or choose *Limit* orders in the app.
  - **Price data:** quotes and historical candles need the paid Kite Connect plan.
    The default free source is Yahoo (`.NS`/`.BO`), which is fine for daily
    candles but may be delayed for intraday.
- **Paytm Money (experimental).** Endpoints and order fields follow Paytm Money's
  official SDK, but some response field names are undocumented, so parsing is
  tolerant. **Use Shadow mode first** and check that positions and cash match.
- **Order volume:** SEBI rules for self-built retail algos have order-rate
  thresholds above which registration is needed. This bot places a few orders per
  day, but stay within your broker's limits.
- **Holidays:** add NSE holidays in the Strategy tab each year. On a holiday there
  are no new candles, so the bot simply does nothing.

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
  india/          Zerodha Kite + Paytm Money brokers, shadow broker, IST market hours, Yahoo data
  app/            web app: engine (scheduler), server (stdlib), static/ (UI + PWA)
configs/          example TOML configs
install/          one-click launchers (Windows, macOS/Linux/Termux)
tests/            pytest suite
```

## Disclaimer

This is educational software. Backtest results, and results on synthetic data
in particular, say nothing about future returns. Paper trade first and use it
at your own risk.
