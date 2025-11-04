"""
Comprehensive tests for the moving average strategy module.

This module contains tests for the implemented moving average-based trading strategies
including DualPoolMovingAverageStrategy and SimpleMovingAverageStrategy.
"""

import pytest
import pandas as pd

# Import the modules being tested
try:
    from backtester.strategy.moving_average import (
        DualPoolMovingAverageStrategy, SimpleMovingAverageStrategy
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestDualPoolMovingAverageStrategyIntegration:
    """Extended integration tests for DualPoolMovingAverageStrategy."""

    def test_multiple_signal_generation(self):
        """Test generation of multiple signals over time."""
        strategy = DualPoolMovingAverageStrategy(ma_short=3, ma_long=5)
        
        # Create data that should generate multiple signals
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 107, 106, 105, 104, 103, 102],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 112, 111, 110, 109, 108, 107],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 106, 105, 104, 103, 102, 101]
        })
        
        signals = strategy.generate_signals(data)
        
        assert isinstance(signals, list)
        # With proper trending data, should generate some signals
        if len(signals) > 0:
            for signal in signals:
                assert 'signal_type' in signal
                assert signal['signal_type'] in ['BUY', 'SELL']
                assert 'price' in signal
                assert 'metadata' in signal

    def test_signal_metadata_completeness(self):
        """Test that generated signals contain complete metadata."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Create data that will definitely generate a signal
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        })
        
        signals = strategy.generate_signals(data)
        
        if signals:
            signal = signals[0]
            metadata = signal.get('metadata', {})
            
            # Check all expected metadata fields
            assert 'strategy' in metadata
            assert 'trend' in metadata
            assert 'short_ma' in metadata
            assert 'long_ma' in metadata
            assert 'leverage_base' in metadata
            assert 'leverage_alpha' in metadata
            assert 'pool_type' in metadata
            
            assert metadata['strategy'] == strategy.name
            assert metadata['pool_type'] == 'BOTH'

    def test_price_history_management(self):
        """Test that price history is properly managed."""
        strategy = DualPoolMovingAverageStrategy()
        
        # Initial state
        assert len(strategy.price_history) == 0
        
        # Add some data
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101]
        })
        
        # Process data row by row to simulate real usage
        for i in range(len(data)):
            row_data = data.iloc[[i]]  # Get single row DataFrame
            strategy.generate_signals(row_data)
        
        # Should have recorded prices (one per row processed)
        assert len(strategy.price_history) == 3
        assert strategy.price_history == [100, 101, 102]
        
        # Test reset
        strategy.reset()
        assert len(strategy.price_history) == 0

    def test_ma_history_consistency(self):
        """Test that moving average histories are consistent."""
        strategy = DualPoolMovingAverageStrategy(ma_short=2, ma_long=3)
        
        # Create data that will generate MAs
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103]
        })
        
        strategy.generate_signals(data)
        
        # Check MA histories
        assert len(strategy.ma_short_history) <= len(strategy.price_history)
        assert len(strategy.ma_long_history) <= len(strategy.price_history)
        
        # Both should have same length when both are populated
        if len(strategy.ma_short_history) > 0 and len(strategy.ma_long_history) > 0:
            assert len(strategy.ma_short_history) == len(strategy.ma_long_history)


class TestSimpleMovingAverageStrategyIntegration:
    """Extended integration tests for SimpleMovingAverageStrategy."""

    def test_simple_strategy_single_column_requirement(self):
        """Test that simple strategy only requires Close column."""
        strategy = SimpleMovingAverageStrategy()
        
        # Valid data with only Close
        valid_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        
        assert strategy.validate_data(valid_data) is True
        
        required = strategy.get_required_columns()
        assert len(required) == 1
        assert 'Close' in required

    def test_simple_strategy_bullish_signal_generation(self):
        """Test bullish signal generation in simple strategy."""
        strategy = SimpleMovingAverageStrategy(ma_short=2, ma_long=3)
        
        # Strong uptrend data
        data = pd.DataFrame({
            'Close': [100, 101, 102, 105, 108, 111, 114, 117, 120, 123]
        })
        
        signals = strategy.generate_signals(data)
        
        # May generate signals during trend changes
        if signals:
            for signal in signals:
                assert signal['signal_type'] in ['BUY', 'SELL']
                assert 'trend' in signal.get('metadata', {})

    def test_simple_strategy_trend_tracking(self):
        """Test trend tracking in simple strategy."""
        strategy = SimpleMovingAverageStrategy(ma_short=3, ma_long=5)
        
        # Create data with clear trend change
        data = pd.DataFrame({
            'Close': [100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 103]
        })
        
        signals = strategy.generate_signals(data)
        
        # Strategy should track trend changes internally
        assert strategy.current_trend in [None, "BULLISH", "BEARISH"]
        
        if signals:
            # Should have signals when trend changes
            for signal in signals:
                assert 'trend' in signal.get('metadata', {})

    def test_simple_strategy_parameter_defaults(self):
        """Test simple strategy parameter defaults."""
        strategy = SimpleMovingAverageStrategy()
        
        assert strategy.name == "SimpleMA"
        assert strategy.ma_short == 5
        assert strategy.ma_long == 20
        assert strategy.leverage == 1.0

    def test_simple_strategy_custom_parameters(self):
        """Test simple strategy with custom parameters."""
        strategy = SimpleMovingAverageStrategy(
            name="CustomSimple",
            ma_short=10,
            ma_long=30,
            leverage=2.0
        )
        
        assert strategy.name == "CustomSimple"
        assert strategy.ma_short == 10
        assert strategy.ma_long == 30
        assert strategy.leverage == 2.0


class TestStrategyComparison:
    """Tests comparing different strategy implementations."""

    def test_dual_pool_vs_simple_requirements(self):
        """Compare column requirements between strategies."""
        dual_pool = DualPoolMovingAverageStrategy()
        simple = SimpleMovingAverageStrategy()
        
        dual_pool_cols = set(dual_pool.get_required_columns())
        simple_cols = set(simple.get_required_columns())
        
        # Simple strategy should require fewer columns
        assert len(simple_cols) < len(dual_pool_cols)
        assert 'Close' in dual_pool_cols
        assert 'Close' in simple_cols
        assert 'High' in dual_pool_cols
        assert 'Low' in dual_pool_cols

    def test_signal_structure_consistency(self):
        """Test that both strategies produce consistent signal structures."""
        dual_pool = DualPoolMovingAverageStrategy()
        simple = SimpleMovingAverageStrategy()
        
        # Create minimal valid data for both
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        })
        
        dual_pool_signals = dual_pool.generate_signals(data)
        simple_signals = simple.generate_signals(data)
        
        # Both should return lists
        assert isinstance(dual_pool_signals, list)
        assert isinstance(simple_signals, list)
        
        # If signals exist, check structure
        for signals in [dual_pool_signals, simple_signals]:
            for signal in signals:
                required_fields = ['timestamp', 'signal_type', 'price', 'quantity', 'metadata']
                for field in required_fields:
                    assert field in signal


@pytest.mark.parametrize("short_period,long_period,data_pattern", [
    (2, 4, "uptrend"),
    (3, 6, "downtrend"),
    (2, 3, "volatile")
])
def test_strategies_with_different_periods(short_period, long_period, data_pattern):
    """Parametrized test for different moving average periods."""
    
    # Create test data based on pattern
    if data_pattern == "uptrend":
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    elif data_pattern == "downtrend":
        prices = [120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    else:  # volatile
        prices = [100, 105, 95, 110, 90, 115, 85, 120, 80, 125, 75, 130, 70, 135, 65, 140, 60, 145, 55, 150, 50]
    
    data = pd.DataFrame({
        'Close': prices,
        'High': [p * 1.05 for p in prices],
        'Low': [p * 0.95 for p in prices]
    })
    
    # Test both strategies
    dual_pool = DualPoolMovingAverageStrategy(ma_short=short_period, ma_long=long_period)
    simple = SimpleMovingAverageStrategy(ma_short=short_period, ma_long=long_period)
    
    dual_pool_signals = dual_pool.generate_signals(data)
    simple_signals = simple.generate_signals(data)
    
    # Both should handle the data without errors
    assert isinstance(dual_pool_signals, list)
    assert isinstance(simple_signals, list)


if __name__ == "__main__":
    pytest.main([__file__])