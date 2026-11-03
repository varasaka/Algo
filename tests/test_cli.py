from pathlib import Path

from algo_agent.cli import main
from algo_agent.config import AgentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_load_example_configs():
    cfg = AgentConfig.load(ROOT / "configs" / "default.toml")
    assert cfg.symbols == ["AAA", "BBB", "CCC"]
    assert cfg.strategy == "ensemble" and len(cfg.strategy_params["members"]) == 3
    assert cfg.risk.max_position_pct == 0.5
    alp = AgentConfig.load(ROOT / "configs" / "alpaca_paper.toml")
    assert alp.strategy_params == {"trend_window": 100}


def test_cli_commands(tmp_path, capsys):
    assert main(["backtest", "-c", str(ROOT / "configs" / "default.toml"),
                 "-o", str(tmp_path)]) == 0
    assert (tmp_path / "equity.csv").exists() and (tmp_path / "metrics.json").exists()
    assert main(["backtest", "--strategy", "rsi", "-p", "oversold=25"]) == 0
    assert main(["optimize", "--strategy", "sma", "-g", "fast=10,20 slow=50"]) == 0
    assert main(["run", "--strategy", "sma", "--max-steps", "5",
                 "--journal", str(tmp_path / "j.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "sharpe" in out and "final equity" in out


def test_csv_source(tmp_path):
    from algo_agent.data import synthetic_ohlcv
    df = synthetic_ohlcv(300)
    df.index.name = "date"
    df.to_csv(tmp_path / "ZZZ.csv")
    assert main(["backtest", "--source", "csv", "--data-dir", str(tmp_path), "-s", "ZZZ"]) == 0
