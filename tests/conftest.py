"""
Pytest configuration and shared fixtures for the backtester test suite.

This module provides pytest configuration, fixtures, and shared test utilities
that are used across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock
import sys
import os

# Add the parent directory to the path to import backtester modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session")
def test_data():
    """Generate sample test data for use in tests."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic OHLCV data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = initial_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    
    # Strategy config
    config.strategy = Mock()
    config.strategy.strategy_name = "DualPoolMA"
    config.strategy.ma_short = 20
    config.strategy.ma_long = 50
    config.strategy.leverage_base = 2.0
    config.strategy.leverage_alpha = 3.0
    config.strategy.base_to_alpha_split = 0.2
    config.strategy.alpha_to_base_split = 0.2
    config.strategy.stop_loss_base = 0.025
    config.strategy.stop_loss_alpha = 0.025
    config.strategy.take_profit_target = 0.10
    
    # Portfolio config
    config.portfolio = Mock()
    config.portfolio.initial_capital = 1000.0
    config.portfolio.commission_rate = 0.001
    config.portfolio.maintenance_margin = 0.25
    config.portfolio.interest_rate_daily = 0.0001
    config.portfolio.spread_rate = 0.0005
    config.portfolio.slippage_std = 0.001
    config.portfolio.funding_enabled = True
    config.portfolio.tax_rate = 0.15
    config.portfolio.max_positions = 5
    
    # Data config
    config.data = Mock()
    config.data.default_ticker = "SPY"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-01-01"
    config.data.interval = "1d"
    config.data.use_technical_indicators = False
    
    # Performance config with real values (not Mocks)
    config.performance = Mock()
    config.performance.risk_free_rate = 0.02  # Real float value
    config.performance.benchmark_enabled = False
    config.performance.benchmark_symbol = "SPY"
    
    # Execution config
    config.execution = Mock()
    config.execution.commission_rate = 0.001
    config.execution.min_commission = 1.0
    config.execution.spread = 0.0001
    config.execution.slippage_model = "normal"
    config.execution.slippage_std = 0.0005
    config.execution.latency_ms = 0.0
    
    # Risk config
    config.risk = Mock()
    config.risk.max_portfolio_risk = 0.02
    config.risk.max_position_size = 0.10
    config.risk.max_leverage = 5.0
    config.risk.max_drawdown = 0.20
    
    return config


@pytest.fixture
def sample_portfolio_state():
    """Sample portfolio state for testing."""
    state = Mock()
    state.total_value = 1100.0
    state.cash = 100.0
    state.positions = {}
    state.leverage_base = 2.0
    state.leverage_alpha = 3.0
    state.base_pool_value = 600.0
    state.alpha_pool_value = 400.0
    return state


@pytest.fixture
def mock_broker():
    """Mock broker for testing order execution."""
    broker = Mock()
    broker.execute_order = Mock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'fill_price': 100.0,
        'fill_quantity': 10
    })
    broker.get_account_info = Mock(return_value={
        'cash': 1000.0,
        'buying_power': 2000.0,
        'positions': {}
    })
    return broker


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.25,
        'max_drawdown': -0.08,
        'volatility': 0.18,
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'total_trades': 50,
        'winning_trades': 32,
        'losing_trades': 18
    }


@pytest.fixture
def strategy_signals():
    """Sample strategy signals for testing."""
    return pd.DataFrame({
        'signal': [0, 1, -1, 0, 1, 0, -1, 1],
        'confidence': [0.5, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6, 0.85],
        'strength': [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0]
    }, index=pd.date_range('2020-01-01', periods=8, freq='D'))


# Utility functions for tests
def assert_valid_dataframe(df, expected_columns=None):
    """Assert that a DataFrame is valid and optionally check columns."""
    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    if expected_columns:
        assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
    assert df.index.is_monotonic_increasing, "DataFrame index should be sorted"


def create_test_order(ticker='TEST', side='buy', quantity=10, order_type='market'):
    """Create a test order for use in tests."""
    return {
        'ticker': ticker,
        'side': side,
        'quantity': quantity,
        'order_type': order_type,
        'timestamp': datetime.now(),
        'status': 'pending'
    }


# Mark slow tests
pytestmark = pytest.mark.slow