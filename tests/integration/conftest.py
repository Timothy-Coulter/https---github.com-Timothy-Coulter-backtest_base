"""
Integration Test Configuration and Shared Fixtures.

This module provides pytest configuration, fixtures, and shared utilities
specifically for integration testing of the QuantBench backtester system.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import gc
from unittest.mock import Mock
from typing import Dict, Any, List
import logging

# Import backtester modules
from backtester.core.backtest_engine import BacktestEngine
from backtester.portfolio.portfolio import DualPoolPortfolio
from backtester.strategy.moving_average import DualPoolMovingAverageStrategy
from backtester.execution.broker import SimulatedBroker
from backtester.portfolio.risk_manager import RiskManager
from backtester.data.data_handler import DataHandler
from backtester.core.performance import PerformanceAnalyzer


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def integration_test_config():
    """Enhanced configuration for integration tests with comprehensive settings."""
    config = Mock()
    
    # Test mode settings
    config.test_mode = True
    config.data_validation_strict = True
    config.performance_logging = True
    config.error_logging_detailed = True
    
    # Strategy configuration
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
    
    # Portfolio configuration
    config.portfolio = Mock()
    config.portfolio.initial_capital = 10000.0
    config.portfolio.commission_rate = 0.001
    config.portfolio.maintenance_margin = 0.25
    config.portfolio.interest_rate_daily = 0.0001
    config.portfolio.spread_rate = 0.0005
    config.portfolio.slippage_std = 0.001
    config.portfolio.funding_enabled = True
    config.portfolio.tax_rate = 0.15
    config.portfolio.max_positions = 10
    
    # Data configuration
    config.data = Mock()
    config.data.default_ticker = "SPY"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-01-01"
    config.data.interval = "1d"
    config.data.use_technical_indicators = True
    
    # Performance configuration
    config.performance = Mock()
    config.performance.risk_free_rate = 0.02
    config.performance.benchmark_enabled = False
    config.performance.benchmark_symbol = "SPY"
    
    # Execution configuration
    config.execution = Mock()
    config.execution.commission_rate = 0.001
    config.execution.min_commission = 1.0
    config.execution.spread = 0.0001
    config.execution.slippage_model = "normal"
    config.execution.slippage_std = 0.0005
    config.execution.latency_ms = 0.0
    
    # Risk configuration
    config.risk = Mock()
    config.risk.max_portfolio_risk = 0.02
    config.risk.max_position_size = 0.10
    config.risk.max_leverage = 5.0
    config.risk.max_drawdown = 0.20
    
    return config


@pytest.fixture
def sample_market_data():
    """Generate sample market data for integration testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 2 years of daily data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    # Generate realistic price movement
    initial_price = 100.0
    returns = np.random.normal(0.0008, 0.02, n_periods)  # ~20% annual vol, 20% annual return
    prices = initial_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_periods)
    }, index=dates)
    
    # Ensure OHLC integrity
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def bull_market_data():
    """Generate bull market scenario data."""
    np.random.seed(123)
    
    dates = pd.date_range('2020-03-01', '2021-12-31', freq='D')
    n_periods = len(dates)
    
    # Strong upward trend with moderate volatility
    initial_price = 100.0
    returns = np.random.normal(0.0012, 0.018, n_periods)  # ~30% annual return, 18% vol
    prices = initial_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.003, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.008, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(2000000, 15000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def bear_market_data():
    """Generate bear market scenario data."""
    np.random.seed(456)
    
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    n_periods = len(dates)
    
    # Downward trend with higher volatility
    initial_price = 100.0
    returns = np.random.normal(-0.0008, 0.025, n_periods)  # -20% annual return, 25% vol
    prices = initial_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.008, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.015, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(500000, 8000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def crisis_market_data():
    """Generate crisis scenario data with fat tails."""
    np.random.seed(789)
    
    dates = pd.date_range('2008-01-01', '2009-12-31', freq='D')
    n_periods = len(dates)
    
    # Mix of normal and crisis returns
    normal_periods = int(n_periods * 0.7)
    crisis_periods = n_periods - normal_periods
    
    # Normal market conditions
    normal_returns = np.random.normal(0, 0.015, normal_periods)
    
    # Crisis conditions with fat tails
    crisis_returns = np.random.normal(0, 0.08, crisis_periods)
    
    # Interleave the returns
    returns = np.zeros(n_periods)
    returns[::2] = normal_returns[:len(returns[::2])]
    returns[1::2] = crisis_returns[:len(returns[1::2])]
    
    np.random.shuffle(returns)
    
    initial_price = 100.0
    prices = initial_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.01, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(500000, 20000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def sideways_market_data():
    """Generate sideways/ranging market data."""
    np.random.seed(321)
    
    dates = pd.date_range('2015-01-01', '2016-12-31', freq='D')
    n_periods = len(dates)
    
    # Low volatility, mean-reverting
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(n_periods - 1):
        # Mean-reverting random walk
        drift = -0.0001 * (prices[-1] - initial_price) / initial_price  # Pull back to mean
        noise = np.random.normal(0, 0.01)  # Low volatility
        new_price = prices[-1] * (1 + drift + noise)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    data = pd.DataFrame({
        'Open': np.array(prices[:-1]) * (1 + np.random.normal(0, 0.002, n_periods)),
        'High': np.array(prices[:-1]) * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'Low': np.array(prices[:-1]) * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'Close': np.array(prices[:-1]),
        'Volume': np.random.randint(800000, 5000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def high_volatility_data():
    """Generate high volatility market data."""
    np.random.seed(654)
    
    dates = pd.date_range('2020-03-01', '2020-06-30', freq='D')
    n_periods = len(dates)
    
    # High volatility period
    initial_price = 100.0
    returns = np.random.normal(0, 0.05, n_periods)  # 50% daily vol
    prices = initial_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.02, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.03, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.03, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    np.random.seed(999)
    
    # 5 years of daily data
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    initial_price = 100.0
    returns = np.random.normal(0.0008, 0.02, n_periods)
    prices = initial_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


@pytest.fixture
def backtester_components(integration_test_config):
    """Initialize all backtester components for integration testing."""
    # Create a logger for integration tests
    logger = logging.getLogger('integration_test')
    logger.setLevel(logging.DEBUG)
    
    components = {}
    
    # Initialize components
    try:
        components['engine'] = BacktestEngine(integration_test_config, logger)
        components['portfolio'] = DualPoolPortfolio(
            initial_capital=integration_test_config.portfolio.initial_capital,
            leverage_base=integration_test_config.strategy.leverage_base,
            leverage_alpha=integration_test_config.strategy.leverage_alpha,
            base_to_alpha_split=integration_test_config.strategy.base_to_alpha_split,
            alpha_to_base_split=integration_test_config.strategy.alpha_to_base_split,
            stop_loss_base=integration_test_config.strategy.stop_loss_base,
            stop_loss_alpha=integration_test_config.strategy.stop_loss_alpha,
            take_profit_target=integration_test_config.strategy.take_profit_target,
            maintenance_margin=integration_test_config.portfolio.maintenance_margin,
            commission_rate=integration_test_config.portfolio.commission_rate,
            interest_rate_daily=integration_test_config.portfolio.interest_rate_daily,
            spread_rate=integration_test_config.portfolio.spread_rate,
            slippage_std=integration_test_config.portfolio.slippage_std,
            funding_enabled=integration_test_config.portfolio.funding_enabled,
            tax_rate=integration_test_config.portfolio.tax_rate,
            logger=logger
        )
        
        components['strategy'] = DualPoolMovingAverageStrategy(
            name=integration_test_config.strategy.strategy_name,
            ma_short=integration_test_config.strategy.ma_short,
            ma_long=integration_test_config.strategy.ma_long,
            leverage_base=integration_test_config.strategy.leverage_base,
            leverage_alpha=integration_test_config.strategy.leverage_alpha,
            base_to_alpha_split=integration_test_config.strategy.base_to_alpha_split,
            alpha_to_base_split=integration_test_config.strategy.alpha_to_base_split,
            logger=logger
        )
        
        components['broker'] = SimulatedBroker(
            commission_rate=integration_test_config.execution.commission_rate,
            min_commission=integration_test_config.execution.min_commission,
            spread=integration_test_config.execution.spread,
            slippage_model=integration_test_config.execution.slippage_model,
            slippage_std=integration_test_config.execution.slippage_std,
            latency_ms=integration_test_config.execution.latency_ms,
            logger=logger
        )
        
        components['risk_manager'] = RiskManager(
            max_portfolio_var=integration_test_config.risk.max_portfolio_risk,
            max_single_position=integration_test_config.risk.max_position_size,
            max_leverage=integration_test_config.risk.max_leverage,
            max_drawdown=integration_test_config.risk.max_drawdown,
            rebalance_frequency='weekly',
            alert_thresholds={
                'var_threshold': integration_test_config.risk.max_portfolio_risk * 1.2,
                'drawdown_threshold': integration_test_config.risk.max_drawdown * 1.1,
                'correlation_threshold': 0.8
            },
            logger=logger
        )
        
        components['performance_analyzer'] = PerformanceAnalyzer(
            risk_free_rate=integration_test_config.performance.risk_free_rate,
            logger=logger
        )
        
        components['data_handler'] = DataHandler(
            config=integration_test_config.__dict__ if hasattr(integration_test_config, '__dict__') else None,
            logger=logger
        )
        
    except Exception as e:
        pytest.fail(f"Failed to initialize backtester components: {e}")
    
    return components


@pytest.fixture
def performance_monitor():
    """Performance monitoring utility for integration tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start(self):
            """Start performance monitoring."""
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            gc.collect()  # Clean up before measuring
        
        def stop(self):
            """Stop performance monitoring and collect metrics."""
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.metrics.update({
                'execution_time': end_time - self.start_time,
                'memory_used': end_memory - self.start_memory,
                'peak_memory': end_memory
            })
            
            return self.metrics
        
        def assert_performance(self, max_time=300, max_memory=1024):
            """Assert that performance is within acceptable limits."""
            assert self.metrics['execution_time'] <= max_time, \
                f"Execution time {self.metrics['execution_time']:.2f}s exceeds limit {max_time}s"
            assert self.metrics['memory_used'] <= max_memory, \
                f"Memory usage {self.metrics['memory_used']:.2f}MB exceeds limit {max_memory}MB"
    
    return PerformanceMonitor()


@pytest.fixture
def corrupted_data():
    """Generate corrupted market data for error handling tests."""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n_periods = len(dates)
    
    # Create data with various corruption types
    data = pd.DataFrame({
        'Open': np.random.normal(100, 10, n_periods),
        'High': np.random.normal(105, 10, n_periods),
        'Low': np.random.normal(95, 10, n_periods),
        'Close': np.random.normal(100, 10, n_periods),
        'Volume': np.random.randint(1000000, 10000000, n_periods)
    }, index=dates)
    
    # Introduce various data quality issues
    data.loc[100:105, 'Close'] = np.nan  # Missing values
    data.loc[200:205, 'Volume'] = -1  # Invalid volume
    data.loc[300:305, 'High'] = data.loc[300:305, 'Low']  # Invalid OHLC relationship
    
    return data


@pytest.fixture
def empty_data():
    """Generate empty/invalid data for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_point_data():
    """Generate single data point for edge case testing."""
    return pd.DataFrame({
        'Open': [100.0],
        'High': [105.0],
        'Low': [95.0],
        'Close': [100.0],
        'Volume': [1000000]
    }, index=[pd.Timestamp('2020-01-01')])


# Utility functions for integration testing
class IntegrationTestHelpers:
    """Helper methods for integration testing."""
    
    @staticmethod
    def validate_data_flow(data_flow: List[tuple], expected_components: List[str]) -> bool:
        """Validate data flow between components."""
        for source, target, data in data_flow:
            assert data is not None, f"No data passed from {source} to {target}"
            assert len(data) > 0, f"Empty data from {source} to {target}"
            if isinstance(data, pd.DataFrame):
                assert not data.isnull().all().all(), f"All NaN data from {source} to {target}"
        return True
    
    @staticmethod
    def validate_component_integration(result: Dict[str, Any], 
                                     expected_components: List[str]) -> bool:
        """Validate that all expected components participated in integration."""
        integrated_components = result.get('integrated_components', [])
        return all(comp in integrated_components for comp in expected_components)
    
    @staticmethod
    def validate_signal_consistency(signals: List[Dict[str, Any]]) -> bool:
        """Validate signal format and consistency."""
        required_fields = ['signal_type', 'price', 'timestamp']
        
        for signal in signals:
            for field in required_fields:
                assert field in signal, f"Missing field '{field}' in signal"
            
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD'], \
                f"Invalid signal type: {signal['signal_type']}"
            
            assert isinstance(signal['price'], (int, float)), \
                f"Invalid price type: {type(signal['price'])}"
            assert signal['price'] > 0, f"Invalid price value: {signal['price']}"
        
        return True
    
    @staticmethod
    def validate_portfolio_state(portfolio: DualPoolPortfolio) -> bool:
        """Validate portfolio state integrity."""
        # Check pool states
        assert portfolio.base_pool.capital >= 0, "Base pool capital cannot be negative"
        assert portfolio.alpha_pool.capital >= 0, "Alpha pool capital cannot be negative"
        
        # Check leverage limits
        base_leverage = portfolio.base_pool.get_current_leverage()
        alpha_leverage = portfolio.alpha_pool.get_current_leverage()
        
        assert base_leverage <= portfolio.leverage_base * 1.1, \
            f"Base pool leverage {base_leverage} exceeds limit {portfolio.leverage_base}"
        assert alpha_leverage <= portfolio.leverage_alpha * 1.1, \
            f"Alpha pool leverage {alpha_leverage} exceeds limit {portfolio.leverage_alpha}"
        
        return True
    
    @staticmethod
    def generate_stress_market_condition(base_data: pd.DataFrame, 
                                       stress_type: str) -> pd.DataFrame:
        """Generate stress market conditions from base data."""
        data = base_data.copy()
        
        if stress_type == "flash_crash":
            # 20% drop in single day
            crash_idx = len(data) // 2
            data.iloc[crash_idx, data.columns.get_loc('Close')] *= 0.8
            
        elif stress_type == "volatility_spike":
            # Double volatility
            returns = data['Close'].pct_change().dropna()
            high_vol_returns = returns * 2
            prices = data['Close'].iloc[0]
            new_prices = [prices]
            for ret in high_vol_returns:
                new_prices.append(new_prices[-1] * (1 + ret))
            data['Close'] = new_prices[1:]
            
        elif stress_type == "extended_decline":
            # 6-month decline
            decline_start = len(data) // 3
            decline_length = len(data) // 6
            for i in range(decline_length):
                idx = decline_start + i
                if idx < len(data):
                    data.iloc[idx, data.columns.get_loc('Close')] *= 0.99
        
        # Update OHLC based on new Close prices
        data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.005, len(data)))
        data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
        data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
        
        return data


# Make helpers available
@pytest.fixture
def integration_helpers():
    """Provide integration test helpers."""
    return IntegrationTestHelpers()


# Setup and teardown for integration test sessions
@pytest.fixture(scope="session", autouse=True)
def setup_integration_test_environment():
    """Setup integration test environment."""
    # Configure logging for integration tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
    )
    
    yield
    
    # Cleanup after all integration tests
    gc.collect()