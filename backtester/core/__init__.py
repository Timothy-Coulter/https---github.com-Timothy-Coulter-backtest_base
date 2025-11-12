"""Core layer for backtest engine, performance metrics, configuration, and logging."""

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import (
    BacktesterConfig,
    ComprehensiveRiskConfig,
    DataRetrievalConfig,
    ExecutionConfig,
    PerformanceConfig,
    PortfolioConfig,
    StrategyConfig,
    get_config,
    reset_config,
    set_config,
)
from backtester.core.logger import get_backtester_logger
from backtester.core.performance import PerformanceAnalyzer

__all__ = [
    'BacktestEngine',
    'BacktesterConfig',
    'ComprehensiveRiskConfig',
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
