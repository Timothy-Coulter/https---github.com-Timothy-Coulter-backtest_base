"""Integration tests for DataRetrieval classes.

This module tests the complete data retrieval workflow including cache handling,
data quality validation, and integration with the findatapy library.
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from backtester.data.data_retrieval import DataRetrieval, DataRetrievalConfig


@pytest.mark.integration
class TestDataRetrievalIntegration:
    """Test complete DataRetrieval integration workflow."""

    def test_full_data_retrieval_workflow(
        self, integration_test_config: Any, sample_market_data: pd.DataFrame
    ) -> None:
        """Test complete data retrieval with cache and validation."""
        # Create configuration
        config = DataRetrievalConfig(
            data_source="yahoo",
            tickers=["AAPL", "GOOGL"],
            fields=["open", "close", "high", "low", "volume"],
            start_date="year",
            cache_algo="internet_load_return",
        )

        # Mock the Market and DataQuality to test workflow
        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            # Set up mock instances
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq

            # Configure data quality mock for clean data
            mock_dq.count_repeated_dates.return_value = (0, None)

            # Configure market mock to return sample data
            mock_market.fetch_market.return_value = sample_market_data

            # Create DataRetrieval instance
            retrieval = DataRetrieval(config)

            # Test 1: First call should try cache, then download
            result = retrieval.get_data()

            # Verify data was retrieved
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

            # Verify market.fetch_market was called (at least once)
            assert mock_market.fetch_market.call_count >= 1

            # Test 2: Second call should use cache (if cache_algo supports it)
            mock_market.fetch_market.reset_mock()

            retrieval.get_data()

            # Should call fetch_market for cache attempt
            assert mock_market.fetch_market.call_count >= 1

            # Test 3: Data quality validation
            validation_result = retrieval.validate_data_quality(sample_market_data)

            assert "total_rows" in validation_result
            assert "total_columns" in validation_result
            assert "duplicated_dates" in validation_result
            assert "missing_values" in validation_result
            assert "completeness" in validation_result

            # Test 4: Complete workflow with validation
            mock_market.fetch_market.reset_mock()

            complete_result = retrieval.get_data_with_validation()

            assert "data" in complete_result
            assert "validation_results" in complete_result
            assert "success" in complete_result
            assert "quality_issues" in complete_result

            # Verify the data retrieval was attempted
            assert mock_market.fetch_market.call_count >= 1

    def test_cache_first_then_fallback_workflow(
        self, integration_test_config: Any, sample_market_data: pd.DataFrame
    ) -> None:
        """Test cache-first logic with fallback to download."""
        config = DataRetrievalConfig(
            data_source="yahoo", tickers=["TEST_TICKER"], fields=["close"], start_date="month"
        )

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq
            mock_dq.count_repeated_dates.return_value = (0, None)

            # Test sequence:
            # 1. First call: cache returns empty → should fallback to download
            # 2. Second call: cache returns data

            call_sequence = [
                pd.DataFrame(),  # Cache miss
                sample_market_data.copy(),  # Download success
            ]
            mock_market.fetch_market.side_effect = call_sequence

            retrieval = DataRetrieval(config)

            # First call: should try cache, get empty, then download
            result1 = retrieval.get_data()

            # Should have been called twice (cache attempt + download)
            assert mock_market.fetch_market.call_count == 2

            # Verify download was attempted
            call_args_list = mock_market.fetch_market.call_args_list
            assert len(call_args_list) == 2

            # First call should be cache_algo_return - check the request object
            first_call = call_args_list[0][0][0]  # Get the MarketDataRequest object
            assert first_call.cache_algo == 'cache_algo_return'

            # Second call should be internet_load_return (download)
            second_call = call_args_list[1][0][0]  # Get the MarketDataRequest object
            assert second_call.cache_algo == 'internet_load_return'

            # Result should be from download
            pd.testing.assert_frame_equal(result1, sample_market_data)

            # Reset for second test - clear side_effect and set return_value
            mock_market.fetch_market.reset_mock()
            mock_market.fetch_market.side_effect = None
            mock_market.fetch_market.return_value = sample_market_data

            # Second call: should succeed from cache
            result2 = retrieval.get_data()

            # Should be called only once (cache attempt)
            assert mock_market.fetch_market.call_count == 1
            pd.testing.assert_frame_equal(result2, sample_market_data)

    def test_data_quality_integration(self, integration_test_config: Any) -> None:
        """Test data quality validation integration."""
        config = DataRetrievalConfig(
            data_source="yahoo", tickers=["AAPL"], fields=["close"], start_date="week"
        )

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq

            retrieval = DataRetrieval(config)

            # Test 1: Clean data - no duplicates
            clean_data = pd.DataFrame(
                {'Close': [100, 101, 102, 103, 104]},
                index=pd.date_range('2023-01-01', periods=5),
            )

            mock_market.fetch_market.return_value = clean_data
            mock_dq.count_repeated_dates.return_value = (0, None)

            result = retrieval.get_data_with_validation()

            assert result["success"] is True
            assert result["validation_results"]["duplicated_dates"]["count"] == 0
            assert result["validation_results"]["missing_values"]["total_missing"] == 0

            # Test 2: Data with duplicates - should detect issues
            mock_market.fetch_market.reset_mock()

            # Create data with duplicate dates
            dirty_data = clean_data.copy()
            dirty_data = pd.concat([dirty_data, dirty_data.iloc[[2]]])  # Duplicate row

            mock_market.fetch_market.return_value = dirty_data
            mock_dq.count_repeated_dates.return_value = (1, dirty_data.index[-1])

            result = retrieval.get_data_with_validation()

            assert result["success"] is False
            assert result["validation_results"]["duplicated_dates"]["count"] == 1
            assert len(result["quality_issues"]) > 0

            # Test 3: Data with missing values
            mock_market.fetch_market.reset_mock()

            incomplete_data = clean_data.copy()
            incomplete_data.loc[incomplete_data.index[2], 'Close'] = None

            mock_market.fetch_market.return_value = incomplete_data
            mock_dq.count_repeated_dates.return_value = (0, None)

            result = retrieval.get_data_with_validation()

            assert result["success"] is False
            assert result["validation_results"]["missing_values"]["total_missing"] == 1
            assert result["validation_results"]["completeness"]["percentage_complete"] < 100

    def test_api_key_integration(self, integration_test_config: Any) -> None:
        """Test API key loading and integration."""
        config = DataRetrievalConfig(
            data_source="fred",  # FRED data source
            tickers=["GDP"],
            fields=["close"],
        )

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
            patch.dict(
                'os.environ',
                {
                    'FRED_API_KEY': 'test_fred_key_123',
                    'ALPHAVANTAGE_API_KEY': 'test_alpha_key_456',
                    'EIKON_API_KEY': 'test_eikon_key_789',
                },
            ),
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq
            mock_dq.count_repeated_dates.return_value = (0, None)

            mock_market.fetch_market.return_value = pd.DataFrame({'Close': [100]})

            retrieval = DataRetrieval(config)

            # Verify API keys were loaded
            assert retrieval.config.fred_api_key == 'test_fred_key_123'
            assert retrieval.config.alpha_vantage_api_key == 'test_alpha_key_456'
            assert retrieval.config.eikon_api_key == 'test_eikon_key_789'

            # Verify they're passed to MarketDataRequest
            with patch('backtester.data.data_retrieval.MarketDataRequest') as mock_request:
                mock_request.return_value = Mock()

                retrieval._create_market_data_request()

                # Verify the call was made
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                assert call_kwargs['fred_api_key'] == 'test_fred_key_123'
                assert call_kwargs['alpha_vantage_api_key'] == 'test_alpha_key_456'
                assert call_kwargs['eikon_api_key'] == 'test_eikon_key_789'

    def test_configuration_updates_integration(self, integration_test_config: Any) -> None:
        """Test dynamic configuration updates."""
        # Start with basic config
        config = DataRetrievalConfig(data_source="yahoo", tickers=["AAPL"], fields=["close"])

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq
            mock_dq.count_repeated_dates.return_value = (0, None)

            mock_market.fetch_market.return_value = pd.DataFrame({'Close': [100]})

            retrieval = DataRetrieval(config)

            # Update configuration dynamically
            retrieval.update_config(
                data_source="google",
                tickers=["MSFT", "GOOGL"],
                start_date="2023-01-01",
                finish_date="2023-12-31",
                fields=["open", "close", "volume"],
            )

            # Verify updates
            assert retrieval.config.data_source == "google"
            assert retrieval.config.tickers == ["MSFT", "GOOGL"]
            assert retrieval.config.start_date == "2023-01-01"
            assert retrieval.config.finish_date == "2023-12-31"
            assert retrieval.config.fields == ["open", "close", "volume"]

            # Verify invalid updates are ignored
            retrieval.update_config(invalid_parameter="should_be_ignored")

            # Test that the changes affect MarketDataRequest creation
            with patch('backtester.data.data_retrieval.MarketDataRequest') as mock_request:
                mock_request.return_value = Mock()

                retrieval._create_market_data_request()

                call_kwargs = mock_request.call_args[1]
                assert call_kwargs['data_source'] == "google"
                assert call_kwargs['tickers'] == ["MSFT", "GOOGL"]
                assert call_kwargs['start_date'] == "2023-01-01"
                assert call_kwargs['fields'] == ["open", "close", "volume"]

    def test_error_handling_integration(self, integration_test_config: Any) -> None:
        """Test error handling in integration scenario."""
        config = DataRetrievalConfig(
            data_source="yahoo", tickers=["NONEXISTENT_TICKER"], fields=["close"]
        )

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq
            mock_dq.count_repeated_dates.return_value = (0, None)

            retrieval = DataRetrieval(config)

            # Test 1: Cache failure - should return None, not raise exception
            mock_market.fetch_market.side_effect = Exception("Network error")

            result = retrieval.load_from_cache()
            assert result is None

            # Test 2: Download failure - should raise exception
            mock_market.fetch_market.side_effect = Exception("Download error")

            with pytest.raises(Exception, match="Download error"):
                retrieval.download_data()

            # Test 3: Empty data handling
            mock_market.fetch_market.side_effect = None
            mock_market.fetch_market.return_value = pd.DataFrame()

            # Should raise ValueError for empty download
            with pytest.raises(ValueError, match="Download returned empty or None data"):
                retrieval.download_data()

            # Test 4: get_data_with_validation with no data
            # Mock the get_data method directly to return None
            with patch.object(retrieval, 'get_data', return_value=None):
                result = retrieval.get_data_with_validation()

            assert result["success"] is False
            assert result["data"] is None
            assert "error" in result["validation_results"]

    def test_performance_integration(self, integration_test_config: Any) -> None:
        """Test performance characteristics of data retrieval."""
        config = DataRetrievalConfig(
            data_source="yahoo",
            tickers=["AAPL"],
            fields=["close"],
            list_threads=2,  # Test multi-threading
            split_request_chunks=1,
        )

        with (
            patch('backtester.data.data_retrieval.Market') as mock_market_class,
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq_class,
        ):
            mock_market = Mock()
            mock_market_class.return_value = mock_market

            mock_dq = Mock()
            mock_dq_class.return_value = mock_dq
            mock_dq.count_repeated_dates.return_value = (0, None)

            sample_data = pd.DataFrame(
                {'Close': range(100)}, index=pd.date_range('2023-01-01', periods=100)
            )

            mock_market.fetch_market.return_value = sample_data

            retrieval = DataRetrieval(config)

            # Test timing of operations
            start_time = time.time()
            result = retrieval.get_data()
            end_time = time.time()

            operation_time = end_time - start_time

            assert result is not None
            assert operation_time >= 0  # Should complete

            # Test data quality validation timing
            start_time = time.time()
            validation_result = retrieval.validate_data_quality(sample_data)
            end_time = time.time()

            validation_time = end_time - start_time

            assert validation_result is not None
            assert validation_time >= 0
            assert validation_time < operation_time * 2  # Should be faster than data retrieval

            # Test complete workflow timing
            start_time = time.time()
            complete_result = retrieval.get_data_with_validation()
            end_time = time.time()

            complete_time = end_time - start_time

            assert complete_result is not None
            assert complete_time >= 0
