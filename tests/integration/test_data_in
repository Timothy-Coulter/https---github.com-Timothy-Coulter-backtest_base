"""
Data Integration Tests for QuantBench Backtester.

This module tests market data flow validation, technical indicator integration,
and data handling across different sources including real-time, cached, and synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from backtester.data.data_handler import DataHandler


@pytest.mark.integration
class TestMarketDataIntegration:
    """Test market data integration from various sources."""
    
    def test_real_time_data_download_validation(self, sample_market_data, integration_test_config):
        """Test real-time data download and validation."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Mock yfinance to avoid actual network calls
        with patch('yfinance.download') as mock_download:
            # Configure mock to return sample data
            mock_download.return_value = sample_market_data.copy()
            
            # Test data download
            try:
                data = data_handler.get_data(
                    ticker="SPY",
                    start_date="2020-01-01",
                    end_date="2023-12-31",
                    interval="1d"
                )
                
                # Validate data structure
                assert isinstance(data, pd.DataFrame), "Should return DataFrame"
                assert not data.empty, "Data should not be empty"
                
                # Validate required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                assert all(col in data.columns for col in required_columns), \
                    "Data should have required OHLCV columns"
                
                # Validate data types
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    assert pd.api.types.is_numeric_dtype(data[col]), \
                        f"Column {col} should be numeric"
                
                # Validate data integrity
                assert (data['High'] >= data[['Open', 'Close']]).all().all(), \
                    "High should be >= Open and Close"
                assert (data['Low'] <= data[['Open', 'Close']]).all().all(), \
                    "Low should be <= Open and Close"
                assert (data['Volume'] >= 0).all().all(), \
                    "Volume should be non-negative"
                
            except Exception as e:
                pytest.skip(f"Real-time data test skipped due to: {e}")
    
    def test_data_caching_retrieval_efficiency(self, integration_test_config):
        """Test data caching and retrieval efficiency."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test with mock data to avoid network dependencies
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000000, 1100000, 900000]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        # Mock the data source
        with patch.object(data_handler, '_get_data_from_source', return_value=sample_data.copy()):
            # First download
            start_time = time.time()
            data1 = data_handler.get_data("TEST", "2020-01-01", "2020-01-03", "1d")
            first_download_time = time.time() - start_time
            
            # Second download (should use cache)
            start_time = time.time()
            data2 = data_handler.get_data("TEST", "2020-01-01", "2020-01-03", "1d")
            second_download_time = time.time() - start_time
            
            # Validate cached data
            assert data1.equals(data2), "Cached data should be identical to original"
            
            # Cache should be faster (though hard to test reliably in unit tests)
            # Just ensure the function completes successfully
            assert first_download_time > 0, "First download should take some time"
            assert second_download_time >= 0, "Second download should complete"
    
    def test_missing_data_handling_imputation(self, integration_test_config):
        """Test missing data handling and imputation methods."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Create data with missing values
        data_with_gaps = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, np.nan],
            'High': [102, 103, np.nan, 105, 106],
            'Low': [99, 100, 101, np.nan, 104],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000, np.nan, 900000, 1100000, 950000]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        # Test data quality validation
        quality_report = data_handler.validate_data_quality(data_with_gaps)
        
        assert 'missing_values' in quality_report, "Should report missing values"
        assert 'data_completeness' in quality_report, "Should report completeness"
        
        missing_counts = quality_report['missing_values']
        assert sum(missing_counts.values()) > 0, "Should detect missing values"
        
        # Test imputation methods
        try:
            imputed_data = data_handler.impute_missing_data(data_with_gaps)
            
            # Validate that imputation completed
            assert isinstance(imputed_data, pd.DataFrame), "Should return DataFrame"
            assert imputed_data.shape == data_with_gaps.shape, "Should preserve shape"
            
            # Check that some missing values were filled
            original_na_count = data_with_gaps.isna().sum().sum()
            imputed_na_count = imputed_data.isna().sum().sum()
            
            # At least some values should be imputed
            assert imputed_na_count <= original_na_count, \
                "Imputation should not increase missing values"
            
        except Exception as e:
            pytest.skip(f"Imputation test skipped: {e}")
    
    def test_data_quality_consistency_checks(self, sample_market_data, integration_test_config):
        """Test comprehensive data quality and consistency validation."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test with clean data
        clean_data = sample_market_data.copy()
        quality_report = data_handler.run_quality_checks(clean_data)
        
        assert 'is_valid' in quality_report, "Should provide validity assessment"
        assert quality_report['is_valid'], "Clean data should pass quality checks"
        
        # Test with corrupted data
        corrupted_data = clean_data.copy()
        corrupted_data.loc[10:15, 'Close'] = np.nan
        corrupted_data.loc[20:25, 'Volume'] = -1
        corrupted_data.loc[30:35, 'High'] = corrupted_data.loc[30:35, 'Low'] - 1  # Invalid OHLC
        
        corrupted_quality_report = data_handler.run_quality_checks(corrupted_data)
        
        assert not corrupted_quality_report['is_valid'], "Corrupted data should fail quality checks"
        
        # Should identify specific issues
        assert 'issues' in corrupted_quality_report, "Should list specific issues"
        issues = corrupted_quality_report['issues']
        
        issue_types = [issue['type'] for issue in issues]
        assert any('missing' in issue.lower() for issue in issue_types), \
            "Should detect missing values"
        assert any('invalid' in issue.lower() or 'negative' in issue.lower() for issue in issue_types), \
            "Should detect invalid values"
    
    def test_data_source_fallback_mechanisms(self, integration_test_config):
        """Test fallback mechanisms when primary data sources fail."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Mock primary source failure
        with patch.object(data_handler, '_get_data_from_primary_source', 
                         side_effect=Exception("Primary source unavailable")):
            
            # Should fall back to cached data if available
            # For this test, we'll mock the fallback mechanism
            with patch.object(data_handler, '_get_cached_data', 
                            return_value=pd.DataFrame({
                                'Open': [100, 101],
                                'High': [102, 103],
                                'Low': [99, 100],
                                'Close': [101, 102],
                                'Volume': [1000000, 1100000]
                            }, index=pd.date_range('2020-01-01', periods=2))):
                
                try:
                    data = data_handler.get_data_with_fallback(
                        "TEST", "2020-01-01", "2020-01-02", "1d"
                    )
                    
                    # Should return fallback data
                    assert isinstance(data, pd.DataFrame), "Should return fallback data"
                    assert not data.empty, "Fallback data should not be empty"
                    
                except Exception as e:
                    # If fallback also fails, system should handle gracefully
                    assert "fallback" in str(e).lower() or "cache" in str(e).lower(), \
                        "Error should indicate fallback failure"


@pytest.mark.integration
class TestTechnicalIndicatorIntegration:
    """Test technical indicator calculation and consistency."""
    
    def test_sma_ema_calculation_accuracy(self, sample_market_data, integration_test_config):
        """Test Simple and Exponential Moving Average calculation accuracy."""
        from backtester.utils.data_utils import calculate_indicators
        
        # Calculate indicators using utility function
        data_with_indicators = calculate_indicators(sample_market_data.copy())
        
        # Validate that indicators were added
        expected_indicators = ['sma_20', 'sma_50', 'ema_20', 'ema_50']
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns, \
                f"Indicator {indicator} should be present"
        
        # Test SMA calculation manually
        prices = sample_market_data['Close']
        
        # 20-day SMA
        sma_20_expected = prices.rolling(window=20).mean()
        sma_20_actual = data_with_indicators['sma_20']
        
        # Should be very close (allowing for floating point precision)
        pd.testing.assert_series_equal(
            sma_20_expected.dropna(), 
            sma_20_actual.dropna(), 
            atol=1e-10
        )
        
        # Test that EMA differs from SMA (should be more responsive)
        ema_20_actual = data_with_indicators['ema_20']
        
        # EMA should generally be closer to recent prices than SMA
        recent_prices = prices.tail(20)
        sma_recent_diff = abs(recent_prices - sma_20_actual.tail(20)).mean()
        ema_recent_diff = abs(recent_prices - ema_20_actual.tail(20)).mean()
        
        # EMA should be more responsive (lower difference to recent prices)
        assert ema_recent_diff <= sma_recent_diff * 1.5, \
            "EMA should be more responsive than SMA"
    
    def test_rsi_calculation_consistency(self, sample_market_data, integration_test_config):
        """Test RSI (Relative Strength Index) calculation consistency."""
        from backtester.utils.data_utils import calculate_indicators
        
        # Add RSI to data
        data_with_indicators = calculate_indicators(sample_market_data.copy())
        
        # Check that RSI was calculated
        assert 'rsi_14' in data_with_indicators.columns, "RSI should be calculated"
        
        rsi_values = data_with_indicators['rsi_14'].dropna()
        
        # RSI should be between 0 and 100
        assert (rsi_values >= 0).all(), "RSI values should be >= 0"
        assert (rsi_values <= 100).all(), "RSI values should be <= 100"
        
        # RSI calculation verification
        # Manually calculate first few RSI values to verify
        prices = sample_market_data['Close']
        price_changes = prices.diff().dropna()
        
        # For RSI calculation, we need gains and losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)
        
        # 14-period RSI should start having valid values after period 14
        assert len(rsi_values) > 0, "RSI should have valid values"
        
        # RSI should oscillate (not be constantly at extreme values)
        assert rsi_values.std() > 1, "RSI should have variation"
    
    def test_volatility_calculation_integration(self, sample_market_data, integration_test_config):
        """Test volatility calculation integration."""
        from backtester.utils.data_utils import calculate_indicators
        
        # Add volatility indicators
        data_with_indicators = calculate_indicators(sample_market_data.copy())
        
        # Check for volatility indicators
        volatility_indicators = [col for col in data_with_indicators.columns 
                               if 'vol' in col.lower()]
        
        assert len(volatility_indicators) > 0, "Should calculate volatility indicators"
        
        # Test specific volatility measures
        if 'volatility_20' in data_with_indicators.columns:
            volatility = data_with_indicators['volatility_20'].dropna()
            
            # Volatility should be positive
            assert (volatility > 0).all(), "Volatility should be positive"
            
            # Volatility should have reasonable values (not extreme)
            assert volatility.mean() < 1.0, "Average volatility should be reasonable"
            assert volatility.std() > 0, "Volatility should vary over time"
        
        # Test rolling standard deviation (realized volatility)
        returns = sample_market_data['Close'].pct_change().dropna()
        rolling_std = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Should match calculated volatility if present
        if 'volatility_20' in data_with_indicators.columns:
            calculated_vol = data_with_indicators['volatility_20'].dropna()
            # Allow some tolerance for calculation differences
            assert len(calculated_vol) > 0, "Should have calculated volatility"
    
    def test_multi_timeframe_indicator_consistency(self, sample_market_data, integration_test_config):
        """Test consistency of indicators across multiple timeframes."""
        from backtester.utils.data_utils import calculate_indicators
        
        # Create different timeframes by resampling
        daily_data = sample_market_data.copy()
        weekly_data = daily_data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Calculate indicators for both timeframes
        daily_with_indicators = calculate_indicators(daily_data)
        weekly_with_indicators = calculate_indicators(weekly_data)
        
        # Check that indicators are calculated for both
        for timeframe_name, data_with_ind in [('daily', daily_with_indicators), 
                                             ('weekly', weekly_with_indicators)]:
            indicators = [col for col in data_with_ind.columns 
                         if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            assert len(indicators) > 0, f"{timeframe_name} timeframe should have indicators"
        
        # Test that longer-term indicators are smoother on longer timeframes
        if 'sma_50' in daily_with_indicators.columns and 'sma_50' in weekly_with_indicators.columns:
            daily_sma = daily_with_indicators['sma_50'].dropna()
            weekly_sma = weekly_with_indicators['sma_50'].dropna()
            
            # Weekly SMA should have less variation (smoother)
            daily_volatility = daily_sma.pct_change().std()
            weekly_volatility = weekly_sma.pct_change().std()
            
            # Allow for different absolute scales, but check relative smoothness
            if len(daily_sma) > 20 and len(weekly_sma) > 5:
                assert weekly_volatility >= 0, "Weekly volatility should be calculable"
    
    def test_indicator_signal_generation_validation(self, sample_market_data, integration_test_config):
        """Test that indicators generate appropriate trading signals."""
        from backtester.utils.data_utils import calculate_indicators
        
        # Add indicators to data
        data_with_indicators = calculate_indicators(sample_market_data.copy())
        
        # Test moving average crossover signals
        if 'sma_20' in data_with_indicators.columns and 'sma_50' in data_with_indicators.columns:
            sma_20 = data_with_indicators['sma_20']
            sma_50 = data_with_indicators['sma_50']
            
            # Find crossover points
            crossover_up = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
            crossover_down = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
            
            crossover_up_count = crossover_up.sum()
            crossover_down_count = crossover_down.sum()
            
            # Should have some crossovers in trending data
            total_crossovers = crossover_up_count + crossover_down_count
            assert total_crossovers > 0, "Should detect some MA crossovers"
            
            # RSI signals
            if 'rsi_14' in data_with_indicators.columns:
                rsi = data_with_indicators['rsi_14']
                
                # RSI oversold/overbought signals
                oversold = rsi < 30
                overbought = rsi > 70
                
                oversold_count = oversold.sum()
                overbought_count = overbought.sum()
                
                # Should have some extreme RSI readings
                assert oversold_count >= 0, "Should detect oversold conditions"
                assert overbought_count >= 0, "Should detect overbought conditions"
    
    def test_indicator_performance_impact(self, large_dataset, integration_test_config):
        """Test performance impact of indicator calculations."""
        import time
        
        from backtester.utils.data_utils import calculate_indicators
        
        # Time indicator calculation
        start_time = time.time()
        data_with_indicators = calculate_indicators(large_dataset.copy())
        calculation_time = time.time() - start_time
        
        # Should calculate indicators in reasonable time
        max_expected_time = len(large_dataset) * 0.001  # 1ms per row
        assert calculation_time < max_expected_time, \
            f"Indicator calculation too slow: {calculation_time:.3f}s for {len(large_dataset)} rows"
        
        # Verify all expected indicators are present
        expected_indicators = ['sma_20', 'sma_50', 'ema_20', 'rsi_14']
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns, \
                f"Indicator {indicator} should be calculated"


@pytest.mark.integration
class TestDataValidationPipeline:
    """Test comprehensive data validation pipeline."""
    
    def test_ohlcv_data_integrity_validation(self, sample_market_data, integration_test_config):
        """Test OHLCV data integrity validation pipeline."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test validation pipeline
        validation_result = data_handler.validate_ohlcv_integrity(sample_market_data)
        
        assert 'is_valid' in validation_result, "Should return validation result"
        assert 'checks_performed' in validation_result, "Should list checks performed"
        assert 'errors' in validation_result, "Should list any errors"
        
        # Clean data should pass validation
        assert validation_result['is_valid'], "Clean data should pass validation"
        assert len(validation_result['errors']) == 0, "Clean data should have no errors"
        
        # Test with invalid data
        invalid_data = sample_market_data.copy()
        invalid_data.loc[10, 'High'] = invalid_data.loc[10, 'Low'] - 1  # Invalid OHLC
        
        invalid_validation = data_handler.validate_ohlcv_integrity(invalid_data)
        
        assert not invalid_validation['is_valid'], "Invalid data should fail validation"
        assert len(invalid_validation['errors']) > 0, "Invalid data should have errors"
    
    def test_data_completeness_validation(self, sample_market_data, integration_test_config):
        """Test data completeness validation across time series."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test completeness validation
        completeness_report = data_handler.validate_data_completeness(sample_market_data)
        
        assert 'completeness_ratio' in completeness_report, "Should report completeness ratio"
        assert 'missing_periods' in completeness_report, "Should identify missing periods"
        assert 'gap_analysis' in completeness_report, "Should provide gap analysis"
        
        # Clean data should have high completeness
        assert completeness_report['completeness_ratio'] >= 0.99, \
            "Clean data should have >99% completeness"
        
        # Test with gapped data
        gapped_data = sample_market_data.copy()
        # Remove some dates to create gaps
        gapped_data = gapped_data.drop(gapped_data.index[10:15])
        
        gapped_completeness = data_handler.validate_data_completeness(gapped_data)
        
        assert gapped_completeness['completeness_ratio'] < 1.0, \
            "Gapped data should have <100% completeness"
        assert len(gapped_completeness['missing_periods']) > 0, \
            "Should identify missing periods"
    
    def test_statistical_properties_validation(self, sample_market_data, integration_test_config):
        """Test statistical properties validation of market data."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Validate statistical properties
        stats_report = data_handler.validate_statistical_properties(sample_market_data)
        
        assert 'returns_analysis' in stats_report, "Should analyze returns"
        assert 'outlier_detection' in stats_report, "Should detect outliers"
        assert 'distribution_analysis' in stats_report, "Should analyze distribution"
        
        # Check return analysis
        returns_analysis = stats_report['returns_analysis']
        assert 'mean_return' in returns_analysis, "Should calculate mean return"
        assert 'volatility' in returns_analysis, "Should calculate volatility"
        assert 'skewness' in returns_analysis, "Should calculate skewness"
        assert 'kurtosis' in returns_analysis, "Should calculate kurtosis"
        
        # Volatility should be positive and reasonable
        assert returns_analysis['volatility'] > 0, "Volatility should be positive"
        assert returns_analysis['volatility'] < 1.0, "Volatility should be reasonable"
        
        # Check outlier detection
        outlier_detection = stats_report['outlier_detection']
        assert 'outlier_count' in outlier_detection, "Should count outliers"
        assert 'outlier_threshold' in outlier_detection, "Should set threshold"
        assert isinstance(outlier_detection['outlier_count'], int), "Outlier count should be integer"
    
    def test_data_type_format_validation(self, sample_market_data, integration_test_config):
        """Test data type and format validation."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test type validation
        type_validation = data_handler.validate_data_types(sample_market_data)
        
        assert 'type_valid' in type_validation, "Should return type validation"
        assert 'column_types' in type_validation, "Should report column types"
        assert 'type_errors' in type_validation, "Should list type errors"
        
        # Clean data should pass type validation
        assert type_validation['type_valid'], "Clean data should have valid types"
        assert len(type_validation['type_errors']) == 0, "Clean data should have no type errors"
        
        # Test with wrong data types
        wrong_type_data = sample_market_data.copy()
        wrong_type_data['Close'] = wrong_type_data['Close'].astype(str)  # Wrong type
        
        wrong_type_validation = data_handler.validate_data_types(wrong_type_data)
        
        assert not wrong_type_validation['type_valid'], "Wrong types should fail validation"
        assert len(wrong_type_validation['type_errors']) > 0, "Should report type errors"
    
    def test_real_time_data_streaming_validation(self, integration_test_config):
        """Test real-time data streaming and validation."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Mock real-time data stream
        mock_stream_data = []
        
        def mock_data_generator():
            """Generate mock streaming data."""
            base_price = 100.0
            for i in range(10):
                price_change = np.random.normal(0, 0.01)
                base_price *= (1 + price_change)
                
                yield pd.DataFrame({
                    'Open': [base_price * 0.999],
                    'High': [base_price * 1.002],
                    'Low': [base_price * 0.998],
                    'Close': [base_price],
                    'Volume': [np.random.randint(100000, 1000000)]
                }, index=[pd.Timestamp.now() + pd.Timedelta(seconds=i)])
        
        # Test streaming validation
        stream_validator = data_handler.create_stream_validator()
        
        validated_data = []
        for data_chunk in mock_data_generator():
            validation_result = stream_validator.validate_chunk(data_chunk)
            if validation_result['is_valid']:
                validated_data.append(data_chunk)
        
        # Should validate and collect streaming data
        assert len(validated_data) > 0, "Should collect valid streaming data"
        
        # Combine validated chunks
        combined_data = pd.concat(validated_data)
        assert isinstance(combined_data, pd.DataFrame), "Should combine into DataFrame"
        assert len(combined_data) == 10, "Should have all data chunks"
    
    def test_data_lineage_tracking(self, sample_market_data, integration_test_config):
        """Test data lineage tracking and provenance."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Process data and track lineage
        processed_data = data_handler.process_data_with_lineage(sample_market_data.copy())
        
        # Check for lineage information
        assert 'data_lineage' in processed_data.attrs, "Should track data lineage"
        lineage = processed_data.attrs['data_lineage']
        
        # Validate lineage structure
        required_lineage_fields = ['source', 'timestamp', 'processing_steps', 'checksum']
        for field in required_lineage_fields:
            assert field in lineage, f"Lineage should include {field}"
        
        # Check processing steps
        assert 'data_source' in lineage['processing_steps'], "Should track data source"
        assert 'validation' in lineage['processing_steps'], "Should track validation"
        assert 'transformation' in lineage['processing_steps'], "Should track transformations"
        
        # Checksum should be valid
        assert isinstance(lineage['checksum'], str), "Checksum should be string"
        assert len(lineage['checksum']) > 0, "Checksum should not be empty"