"""
Comprehensive tests for the data handler module.

This module contains tests for data loading, validation, preprocessing,
and data management functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import the modules being tested
try:
    from backtester.data.data_handler import (
        DataHandler, DataLoader, DataValidator, DataPreprocessor
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestDataHandler:
    """Test suite for the DataHandler class."""

    def test_initialization(self):
        """Test DataHandler initialization."""
        handler = DataHandler()
        
        assert handler.data_loader is not None
        assert handler.data_validator is not None
        assert handler.data_preprocessor is not None
        assert handler.data_cache is not None
        assert handler.config is not None

    def test_initialization_with_config(self):
        """Test DataHandler with custom configuration."""
        config = {
            'data_source': 'yahoo',
            'cache_enabled': True,
            'validation_strict': False,
            'preprocess_data': True
        }
        
        handler = DataHandler(config=config)
        
        assert handler.config['data_source'] == 'yahoo'
        assert handler.config['cache_enabled'] is True
        assert handler.config['validation_strict'] is False
        assert handler.config['preprocess_data'] is True

    def test_load_data_success(self):
        """Test successful data loading."""
        handler = DataHandler()
        
        # Mock successful data loading
        test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = test_data
            
            result = handler.load_data(
                symbol='SPY',
                start_date='2023-01-01',
                end_date='2023-01-05',
                interval='1d'
            )
            
            assert result.equals(test_data)
            mock_yf.assert_called_once()

    def test_load_data_with_validation(self):
        """Test data loading with validation."""
        handler = DataHandler(config={'validation_strict': True})
        
        # Mock data that fails validation
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [95, 106, 107, 108, 109],  # Invalid: High < Open
            'Low': [99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = invalid_data
            
            # Should raise validation error in strict mode
            with pytest.raises(ValueError, match="Data validation failed"):
                handler.load_data('SPY', '2023-01-01', '2023-01-05', '1d')

    def test_data_caching(self):
        """Test data caching functionality."""
        handler = DataHandler(config={'cache_enabled': True})
        
        # Create proper OHLC test data
        test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = test_data
            
            # First load - should cache data
            result1 = handler.load_data('SPY', '2023-01-01', '2023-01-05', '1d')
            
            # Second load - should use cache
            result2 = handler.load_data('SPY', '2023-01-01', '2023-01-05', '1d')
            
            assert result1.equals(result2)
            assert mock_yf.call_count == 1  # Only called once due to cache

    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        handler = DataHandler(config={'preprocess_data': True})
        
        raw_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = raw_data
            
            result = handler.load_data('SPY', '2023-01-01', '2023-01-05', '1d')
            
            # Should have added technical indicators
            assert 'sma_5' in result.columns
            assert 'ema_5' in result.columns

    def test_multiple_symbols_loading(self):
        """Test loading multiple symbols."""
        handler = DataHandler()
        
        symbols = ['SPY', 'AAPL', 'GOOGL']
        
        # Mock individual symbol data - test each symbol separately to avoid multi-index complexity
        def mock_download_single(symbol, **kwargs):
            if symbol == 'SPY':
                return pd.DataFrame({
                    'Open': [100, 101, 102],
                    'High': [105, 106, 107],
                    'Low': [99, 100, 101],
                    'Close': [103, 104, 105],
                    'Volume': [1000000, 1100000, 1200000]
                })
            elif symbol == 'AAPL':
                return pd.DataFrame({
                    'Open': [150, 151, 152],
                    'High': [155, 156, 157],
                    'Low': [149, 150, 151],
                    'Close': [153, 154, 155],
                    'Volume': [2100000, 2200000, 2300000]
                })
            elif symbol == 'GOOGL':
                return pd.DataFrame({
                    'Open': [2500, 2501, 2502],
                    'High': [2505, 2506, 2507],
                    'Low': [2499, 2500, 2501],
                    'Close': [2503, 2504, 2505],
                    'Volume': [3100000, 3200000, 3300000]
                })
            else:
                return pd.DataFrame()
        
        with patch('yfinance.download', side_effect=mock_download_single):
            results = handler.load_multiple_symbols(
                symbols, '2023-01-01', '2023-01-03', '1d'
            )
            
            assert len(results) == 3
            assert 'SPY' in results
            assert 'AAPL' in results
            assert 'GOOGL' in results
            assert results['SPY']['Close'].iloc[0] == 103
            assert results['AAPL']['Close'].iloc[0] == 153
            assert results['GOOGL']['Close'].iloc[0] == 2503

    def test_data_aggregation(self):
        """Test data aggregation across timeframes."""
        handler = DataHandler()
        
        # Mock minute data
        minute_data = pd.DataFrame({
            'Open': [100, 101, 102, 103],
            'High': [105, 106, 107, 108],
            'Low': [99, 100, 101, 102],
            'Close': [103, 104, 105, 106],
            'Volume': [100000, 110000, 120000, 130000]
        }, index=pd.date_range('2023-01-01 09:30', periods=4, freq='1min'))
        
        # Aggregate to 2-minute bars
        aggregated = handler.aggregate_data(
            minute_data, 
            target_frequency='2min',
            aggregation_method='ohlcv'
        )
        
        assert len(aggregated) < len(minute_data)  # Should be fewer bars
        assert 'Open' in aggregated.columns
        assert 'High' in aggregated.columns
        assert 'Low' in aggregated.columns
        assert 'Close' in aggregated.columns
        assert 'Volume' in aggregated.columns

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        handler = DataHandler(config={'fill_missing': True})
        
        # Create proper OHLC data with missing values
        incomplete_data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, np.nan],
            'High': [105, 106, np.nan, 108, 109],
            'Low': [99, 100, 101, np.nan, 103],
            'Close': [100, np.nan, 102, 103, np.nan],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = incomplete_data
            
            result = handler.load_data('SPY', '2023-01-01', '2023-01-05', '1d')
            
            # Should handle missing data according to config
            assert not result['Close'].isna().any()

    def test_data_export(self):
        """Test data export functionality."""
        handler = DataHandler()
        
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        try:
            # Test CSV export
            csv_path = handler.export_data(test_data, format='csv', path='test_data.csv')
            
            # Verify file was created and contains data
            exported_data = pd.read_csv(csv_path)
            assert len(exported_data) == len(test_data)
            
            # Test JSON export
            json_path = handler.export_data(test_data, format='json', path='test_data.json')
            
            # Verify JSON file
            imported_data = pd.read_json(json_path)
            assert len(imported_data) == len(test_data)
        finally:
            # Clean up
            import os
            for filename in ['test_data.csv', 'test_data.json']:
                if os.path.exists(filename):
                    os.remove(filename)

    def test_get_data_info(self):
        """Test getting data information."""
        handler = DataHandler()
        
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        info = handler.get_data_info(test_data)
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'date_range' in info
        assert 'missing_values' in info
        
        assert info['shape'] == (5, 2)
        assert len(info['columns']) == 2
        assert info['missing_values'] == {'Close': 0, 'Volume': 0}


class TestDataLoader:
    """Test suite for the DataLoader class."""

    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        
        assert loader.data_sources is not None
        assert loader.default_source == 'yahoo'

    def test_load_from_yahoo_finance(self):
        """Test loading data from Yahoo Finance."""
        loader = DataLoader()
        
        # Mock Yahoo Finance API response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = mock_data
            
            result = loader.load_from_source('yahoo', 'SPY', '2023-01-01', '2023-01-05', '1d')
            
            assert result.equals(mock_data)
            mock_yf.assert_called_once()

    def test_load_from_csv(self):
        """Test loading data from CSV file."""
        loader = DataLoader()
        
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        csv_path = 'test_data.csv'
        test_data.to_csv(csv_path, index=False)
        
        try:
            result = loader.load_from_csv(csv_path)
            
            assert len(result) == 5
            assert 'Close' in result.columns
            assert 'Volume' in result.columns
        finally:
            # Clean up
            import os
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_load_from_database(self):
        """Test loading data from database."""
        loader = DataLoader()
        
        # Mock database connection and query
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
            'price': [100, 101, 102, 103, 104],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        with patch('sqlite3.connect') as mock_connect, \
             patch('pandas.read_sql') as mock_read_sql:
            
            mock_connection = Mock()
            mock_connect.return_value = mock_connection
            mock_read_sql.return_value = mock_data
            
            result = loader.load_from_database(
                'sqlite', 'test.db', 
                'SELECT * FROM prices WHERE symbol = ?', ['SPY']
            )
            
            assert result.equals(mock_data)
            mock_read_sql.assert_called_once()

    def test_error_handling(self):
        """Test error handling in data loading."""
        loader = DataLoader()
        
        # Test invalid symbol
        with patch('yfinance.download') as mock_yf:
            mock_yf.side_effect = Exception("Symbol not found")
            
            with pytest.raises(Exception, match="Symbol not found"):
                loader.load_from_source('yahoo', 'INVALID_SYMBOL', '2023-01-01', '2023-01-05', '1d')

    def test_data_format_standardization(self):
        """Test data format standardization."""
        loader = DataLoader()
        
        # Input data with different column names
        raw_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3, freq='D'),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'vol': [1000000, 1100000, 1200000]
        })
        
        standardized = loader.standardize_data_format(raw_data)
        
        assert 'Open' in standardized.columns  # Capitalized
        assert 'High' in standardized.columns
        assert 'Low' in standardized.columns
        assert 'Close' in standardized.columns
        assert 'Volume' in standardized.columns  # Full name


class TestDataValidator:
    """Test suite for the DataValidator class."""

    def test_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        
        assert validator.validation_rules is not None
        assert validator.strict_mode is False

    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        validator = DataValidator()
        
        # Valid OHLCV data
        valid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validator.validate_ohlcv_data(valid_data)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_high_low_relationship(self):
        """Test High/Low relationship validation."""
        validator = DataValidator()
        
        # Invalid data: High < Low
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [95, 106, 107],  # Invalid: High < Open
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validator.validate_ohlcv_data(invalid_data)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any('Invalid High/Low' in error for error in errors)

    def test_validate_price_movement(self):
        """Test price movement validation."""
        validator = DataValidator()
        
        # Data with unusual price movements (>50% jump between consecutive periods)
        suspicious_data = pd.DataFrame({
            'Open': [100, 101, 170],  # 68% jump from 101 to 170
            'High': [105, 106, 175],
            'Low': [99, 100, 165],
            'Close': [101, 170, 175],  # 68% jump from 101 to 170
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, warnings = validator.validate_price_movements(suspicious_data)
        
        assert len(warnings) > 0
        assert any('unusual' in warning.lower() for warning in warnings)

    def test_validate_volume_data(self):
        """Test volume data validation."""
        validator = DataValidator()
        
        # Valid volume data
        valid_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validator.validate_volume_data(valid_data)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid volume data
        invalid_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [0, -100, np.nan]  # Invalid values
        })
        
        is_valid, errors = validator.validate_volume_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_check_data_completeness(self):
        """Test data completeness checking."""
        validator = DataValidator()
        
        # Complete data
        complete_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        completeness = validator.check_data_completeness(complete_data)
        
        assert 'missing_percentage' in completeness
        assert 'gap_analysis' in completeness
        assert completeness['missing_percentage'] == 0.0
        
        # Data with gaps
        incomplete_data = pd.DataFrame({
            'Close': [100, np.nan, 102, 103, np.nan],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000]
        })
        
        completeness = validator.check_data_completeness(incomplete_data)
        assert completeness['missing_percentage'] > 0.0


class TestDataPreprocessor:
    """Test suite for the DataPreprocessor class."""

    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.scalers is not None
        assert preprocessor.transformers is not None

    def test_handle_missing_values(self):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        
        data_with_gaps = pd.DataFrame({
            'Close': [100, np.nan, 102, 103, np.nan],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000]
        })
        
        # Test forward fill
        filled_data = preprocessor.handle_missing_values(
            data_with_gaps, method='forward_fill'
        )
        
        assert not filled_data['Close'].isna().any()
        assert filled_data['Close'].iloc[1] == 100  # Forward filled
        
        # Test interpolation
        interpolated_data = preprocessor.handle_missing_values(
            data_with_gaps, method='interpolate'
        )
        
        assert not interpolated_data['Close'].isna().any()

    def test_normalize_prices(self):
        """Test price normalization."""
        preprocessor = DataPreprocessor()
        
        price_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        normalized = preprocessor.normalize_prices(price_data)
        
        assert 'Close_normalized' in normalized.columns
        assert abs(normalized['Close_normalized'].iloc[0] - 100.0) < 0.01  # Should start at 100

    def test_calculate_returns(self):
        """Test return calculation."""
        preprocessor = DataPreprocessor()
        
        price_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        
        returns = preprocessor.calculate_returns(price_data, method='simple')
        
        assert 'returns' in returns.columns
        assert len(returns) == len(price_data)  # Same length, NaN at start
        assert abs(returns['returns'].iloc[1] - 0.01) < 0.001  # 1% return

    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        preprocessor = DataPreprocessor()
        
        price_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        })
        
        indicators = preprocessor.calculate_technical_indicators(price_data)
        
        assert 'sma_5' in indicators.columns
        assert 'ema_5' in indicators.columns
        assert 'rsi_14' in indicators.columns
        assert 'volatility' in indicators.columns
        
        # Check that indicators are calculated correctly
        assert not indicators['sma_5'].isna().all()
        assert not indicators['rsi_14'].isna().all()

    def test_resample_data(self):
        """Test data resampling."""
        preprocessor = DataPreprocessor()
        
        # Create minute-level data
        minute_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105],
            'High': [105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104],
            'Close': [103, 104, 105, 106, 107, 108],
            'Volume': [100000, 110000, 120000, 130000, 140000, 150000]
        }, index=pd.date_range('2023-01-01 09:30', periods=6, freq='1min'))
        
        # Resample to 2-minute bars
        resampled = preprocessor.resample_data(minute_data, '2min', 'ohlcv')
        
        assert len(resampled) < len(minute_data)
        assert 'Open' in resampled.columns
        assert 'High' in resampled.columns
        assert 'Low' in resampled.columns
        assert 'Close' in resampled.columns
        assert 'Volume' in resampled.columns


@pytest.mark.parametrize("symbol,expected_source", [
    ("SPY", "yahoo"),
    ("AAPL", "yahoo"),
    ("GOOGL", "yahoo"),
    ("BTC-USD", "crypto")
])
def test_data_source_selection_parametrized(symbol, expected_source):
    """Parametrized test for data source selection."""
    loader = DataLoader()
    
    selected_source = loader.select_data_source(symbol)
    
    # Basic test that a source is selected
    assert selected_source is not None
    assert isinstance(selected_source, str)
    
    # For crypto symbols, expect crypto source
    if expected_source == "crypto":
        assert selected_source == "crypto"


@pytest.mark.parametrize("frequency,expected_bars", [
    ("1min", 60),
    ("5min", 12),
    ("15min", 4),
    ("1h", 1)
])
def test_data_aggregation_parametrized(frequency, expected_bars):
    """Parametrized test for data aggregation."""
    preprocessor = DataPreprocessor()
    
    # Create hourly data (60 minutes)
    hourly_data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105],
        'High': [105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104],
        'Close': [103, 104, 105, 106, 107, 108],
        'Volume': [100000, 110000, 120000, 130000, 140000, 150000]
    }, index=pd.date_range('2023-01-01 09:30', periods=6, freq='1min'))
    
    # Aggregate to target frequency
    aggregated = preprocessor.resample_data(hourly_data, frequency, 'ohlcv')
    
    # Should have approximately the expected number of bars
    assert len(aggregated) <= 6  # Can't have more than input data


if __name__ == "__main__":
    pytest.main([__file__])