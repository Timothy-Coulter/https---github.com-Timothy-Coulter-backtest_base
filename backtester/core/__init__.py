"""Core layer for backtest engine, performance metrics, configuration, and logging."""

from .backtest_engine import BacktestEngine
from .config import (
    BacktesterConfig,
    DataRetrievalConfig,
    ExecutionConfig,
    PerformanceConfig,
    PortfolioConfig,
    ComprehensiveRiskConfig,
    StrategyConfig,
    get_config,
    reset_config,
    set_config,
)
from .logger import get_backtester_logger
from .performance import PerformanceAnalyzer

__all__ = [
    'BacktestEngine',
    'BacktesterConfig',
    'DataRetrievalConfig',
    'StrategyConfig',
    'PortfolioConfig',
    'ExecutionConfig',
    'RiskConfig',
    'PerformanceConfig',
    'BacktestConfig',
    'ConfigValidator',
    'get_config',
    'set_config',
    'reset_config',
    'get_backtester_logger',
    'PerformanceAnalyzer',
]
