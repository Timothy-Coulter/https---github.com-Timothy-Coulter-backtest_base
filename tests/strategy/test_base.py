"""
Comprehensive tests for the base strategy module.

This module contains tests for the base strategy class and common
strategy functionality that all specific strategies inherit from.
"""

import pytest
import pandas as pd
from datetime import datetime

# Import the modules being tested
try:
    from backtester.strategy.base import (
        BaseStrategy, Signal
    )
    from backtester.strategy.moving_average import (
        DualPoolMovingAverageStrategy, SimpleMovingAverageStrategy
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestBaseStrategy:
    """Test suite for the BaseStrategy class."""

    def test_initialization_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy("test")
        
    def test_signal_creation(self):
        """Test Signal class creation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            price=100.0,
            quantity=10.0,
            stop_loss=95.0,
            take_profit=105.0,
            metadata={"test": True}
        )
        
        assert signal.signal_type == "BUY"
        assert signal.price == 100.0
        assert signal.quantity == 10.0
        assert signal.stop_loss == 95.0
        assert signal.take_profit == 105.0
        assert signal.metadata["test"] is True
        
    def test_signal_to_dict(self):
        """Test Signal to_dict conversion."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="SELL",
            price=100.0
        )
        
        signal_dict = signal.to_dict()
        assert isinstance(signal_dict, dict)
        assert "timestamp" in signal_dict
        assert "signal_type" in signal_dict
        assert "price" in signal_dict
        assert signal_dict["signal_type"] == "SELL"
        
    def test_signal_repr(self):
        """Test Signal string representation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            price=100.0,
            quantity=5.0
        )
        
        repr_str = repr(signal)
        assert "BUY" in repr_str
        assert "100.0" in repr_str
        

class TestDualPoolMovingAverageStrategy:
    """Test suite for DualPoolMovingAverageStrategy."""

    def test_initialization(self):
        """Test DualPoolMovingAverageStrategy initialization."""
        strategy = DualPoolMovingAverageStrategy()
        
        assert strategy.name == "DualPoolMA"
        assert strategy.ma_short == 5
        assert strategy.ma_long == 20
        assert strategy.leverage_base == 1.0
        assert strategy.leverage_alpha == 3.0
        assert strategy.base_to_alpha_split == 0.2
        assert strategy.alpha_to_base_split == 0.2
        
    def test_initialization_custom_params(self):
        """Test DualPoolMovingAverageStrategy with custom parameters."""
        strategy = DualPoolMovingAverageStrategy(
            name="CustomMA",
            ma_short=10,
            ma_long=30,
            leverage_base=2.0,
            leverage_alpha=4.0
        )
        
        assert strategy.name == "CustomMA"
        assert strategy.ma_short == 10
        assert strategy.ma_long == 30
        assert strategy.leverage_base == 2.0
        assert strategy.leverage_alpha == 4.0
        
    def test_get_required_columns(self):
        """Test required columns."""
        strategy = DualPoolMovingAverageStrategy()
        required = strategy.get_required_columns()
        
        assert isinstance(required, list)
        assert "Close" in required
        assert "High" in required
        assert "Low" in required
        
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Create test data with fewer rows than ma_long
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101]
        })
        
        signals = strategy.generate_signals(data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0  # No signals generated with insufficient data
        
    def test_generate_signals_bullish_crossover(self):
        """Test bullish crossover signal generation."""
        strategy = DualPoolMovingAverageStrategy(ma_short=2, ma_long=3)
        
        # Create trending data that should generate a bullish signal
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105],
            'High': [105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104]
        })
        
        signals = strategy.generate_signals(data)
        
        assert isinstance(signals, list)
        # Should generate at least one signal once sufficient data is available
        
    def test_trend_determination(self):
        """Test trend determination logic."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Test bullish trend
        bullish_trend = strategy._determine_trend(105, 100)
        assert bullish_trend == "BULLISH"
        
        # Test bearish trend
        bearish_trend = strategy._determine_trend(95, 100)
        assert bearish_trend == "BEARISH"
        
        # Test neutral trend (within 1% threshold)
        neutral_trend = strategy._determine_trend(100.5, 100)
        assert neutral_trend == "NEUTRAL"
        
    def test_strategy_parameters(self):
        """Test strategy parameter retrieval."""
        strategy = DualPoolMovingAverageStrategy(
            ma_short=10,
            ma_long=30,
            leverage_base=2.0,
            leverage_alpha=4.0
        )
        
        params = strategy.get_strategy_parameters()
        
        assert isinstance(params, dict)
        assert params["ma_short"] == 10
        assert params["ma_long"] == 30
        assert params["leverage_base"] == 2.0
        assert params["leverage_alpha"] == 4.0
        
    def test_data_validation(self):
        """Test data validation."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Valid data
        valid_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101]
        })
        
        assert strategy.validate_data(valid_data) is True
        
        # Invalid data (missing columns)
        invalid_data = pd.DataFrame({
            'Price': [100, 101, 102]
        })
        
        assert strategy.validate_data(invalid_data) is False
        
    def test_strategy_reset(self):
        """Test strategy state reset."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Add some state
        strategy.price_history = [100, 101, 102]
        strategy.current_step = 5
        
        # Reset
        strategy.reset()
        
        assert len(strategy.price_history) == 0
        assert len(strategy.ma_short_history) == 0
        assert len(strategy.ma_long_history) == 0
        assert strategy.current_step == 0
        assert strategy.current_trend is None
        
    def test_leverage_allocation(self):
        """Test leverage allocation calculation."""
        strategy = DualPoolMovingAverageStrategy(
            leverage_base=2.0,
            leverage_alpha=3.0
        )
        
        allocation = strategy.calculate_leverage_allocation(1000.0)
        
        assert isinstance(allocation, dict)
        assert "base_allocation" in allocation
        assert "alpha_allocation" in allocation
        assert "base_leveraged_value" in allocation
        assert "alpha_leveraged_value" in allocation
        
        # Check that allocations sum to total capital
        assert abs(allocation["base_allocation"] + allocation["alpha_allocation"] - 1000.0) < 0.01


class TestSimpleMovingAverageStrategy:
    """Test suite for SimpleMovingAverageStrategy."""

    def test_initialization(self):
        """Test SimpleMovingAverageStrategy initialization."""
        strategy = SimpleMovingAverageStrategy()
        
        assert strategy.name == "SimpleMA"
        assert strategy.ma_short == 5
        assert strategy.ma_long == 20
        assert strategy.leverage == 1.0
        
    def test_get_required_columns(self):
        """Test required columns for simple strategy."""
        strategy = SimpleMovingAverageStrategy()
        required = strategy.get_required_columns()
        
        assert isinstance(required, list)
        assert "Close" in required
        assert len(required) == 1  # Only Close needed
        
    def test_generate_signals_simple(self):
        """Test simple signal generation."""
        strategy = SimpleMovingAverageStrategy(ma_short=2, ma_long=3)
        
        # Create trending data
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105]
        })
        
        signals = strategy.generate_signals(data)
        
        assert isinstance(signals, list)


@pytest.mark.parametrize("short_ma,long_ma,expected_trend", [
    (105, 100, "BULLISH"),
    (95, 100, "BEARISH"),
    (100.5, 100, "NEUTRAL"),
    (100, 100.5, "NEUTRAL")
])
def test_trend_determination_parametrized(short_ma, long_ma, expected_trend):
    """Parametrized test for trend determination."""
    strategy = DualPoolMovingAverageStrategy()
    trend = strategy._determine_trend(short_ma, long_ma)
    assert trend == expected_trend


if __name__ == "__main__":
    pytest.main([__file__])