"""algo_agent: a modular algorithmic trading agent."""

from .agent import TradingAgent
from .backtest import BacktestResult, run_backtest
from .broker import AlpacaBroker, PaperBroker
from .config import AgentConfig
from .risk import RiskConfig, RiskManager
from .strategies import create_strategy

__all__ = ["TradingAgent", "run_backtest", "BacktestResult", "PaperBroker", "AlpacaBroker",
           "AgentConfig", "RiskConfig", "RiskManager", "create_strategy"]
__version__ = "0.1.0"
