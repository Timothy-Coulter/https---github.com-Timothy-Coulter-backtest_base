"""Unit tests for DataRetrieval classes."""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.data.data_retrieval import DataRetrieval, DataRetrievalConfig


class TestDataRetrievalConfig:
    """Test DataRetrievalConfig class methods."""

    def test_init_default_config(self) -> None:
        """Test DataRetrievalConfig initialization with defaults."""
        config = DataRetrievalConfig()

        assert config.data_source == "yahoo"
        assert config.start_date == "year"
        assert config.fields == ["close"]
        assert config.freq == "daily"
        assert config.cache_algo == "internet_load_return"
        assert config.push_to_cache is True
        assert config.list_threads == 1
        assert config.trade_side == "trade"
        assert config.resample_how == "last"

    def test_init_custom_config(self) -> None:
        """Test DataRetrievalConfig initialization with custom values."""
        config = DataRetrievalConfig(
            data_source="test_source",
            start_date="2023-01-01",
            finish_date="2023-12-31",
            tickers=["AAPL", "GOOGL"],
            fields=["open", "close", "volume"],
            cache_algo="cache_algo_return",
            fred_api_key="test_fred_key",
            push_to_cache=False,
        )

        assert config.data_source == "test_source"
        assert config.start_date == "2023-01-01"
        assert config.finish_date == "2023-12-31"
        assert config.tickers == ["AAPL", "GOOGL"]
        assert config.fields == ["open", "close", "volume"]
        assert config.cache_algo == "cache_algo_return"
        assert config.fred_api_key == "test_fred_key"
        assert config.push_to_cache is False

    def test_config_validation(self) -> None:
        """Test that pydantic validation works."""
        # Should work with string list
        config = DataRetrievalConfig(tickers="AAPL")
        assert config.tickers == ["AAPL"]

        # Should work with list
        config = DataRetrievalConfig(tickers=["AAPL", "GOOGL"])
        assert config.tickers == ["AAPL", "GOOGL"]

    def test_config_serialization(self) -> None:
        """Test config serialization methods."""
        config = DataRetrievalConfig(
            data_source="yahoo", tickers=["AAPL", "GOOGL"], fields=["open", "close"]
        )

        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["data_source"] == "yahoo"
        assert config_dict["tickers"] == ["AAPL", "GOOGL"]

        # Test JSON serialization
        config_json = config.model_dump_json()
        assert isinstance(config_json, str)


class TestDataRetrieval:
    """Test DataRetrieval class methods."""

    @pytest.fixture
    def sample_config(self) -> DataRetrievalConfig:
        """Create sample DataRetrievalConfig for testing."""
        return DataRetrievalConfig(
            data_source="yahoo",
            tickers=["AAPL", "GOOGL"],
            fields=["close"],
            start_date="year",
            cache_algo="internet_load_return",
        )

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        # Create realistic market data
        np.random.seed(42)
        base_price = 100

        data = pd.DataFrame(index=dates)
        data['Close'] = base_price + np.random.randn(50).cumsum() * 0.5
        data['Open'] = data['Close'] + np.random.randn(50) * 0.1
        data['High'] = np.maximum(data['Open'], data['Close']) + np.abs(np.random.randn(50)) * 0.1
        data['Low'] = np.minimum(data['Open'], data['Close']) - np.abs(np.random.randn(50)) * 0.1

        return data

    @pytest.fixture
    def mock_market(self) -> Mock:
        """Create mock market object."""
        return Mock()

    def test_init(self, sample_config: DataRetrievalConfig, mock_market: Mock) -> None:
        """Test DataRetrieval initialization."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality', return_value=Mock()),
        ):
            retrieval = DataRetrieval(sample_config)

            assert retrieval.config == sample_config
            assert retrieval.market == mock_market
            assert retrieval.logger is not None
            assert isinstance(retrieval.logger, logging.Logger)

    def test_load_api_keys(self, sample_config: DataRetrievalConfig) -> None:
        """Test API key loading from environment."""
        with (
            patch('backtester.data.data_retrieval.Market'),
            patch('backtester.data.data_retrieval.DataQuality'),
            patch.dict(
                'os.environ',
                {
                    'FRED_API_KEY': 'test_fred',
                    'ALPHAVANTAGE_API_KEY': 'test_alpha',
                    'EIKON_API_KEY': 'test_eikon',
                },
            ),
        ):
            retrieval = DataRetrieval(sample_config)

            assert retrieval.config.fred_api_key == 'test_fred'
            assert retrieval.config.alpha_vantage_api_key == 'test_alpha'
            assert retrieval.config.eikon_api_key == 'test_eikon'

    def test_create_market_data_request(self, sample_config: DataRetrievalConfig) -> None:
        """Test MarketDataRequest creation."""
        with (
            patch('backtester.data.data_retrieval.Market'),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            with patch('backtester.data.data_retrieval.MarketDataRequest') as mock_request:
                # Configure the mock to return a mock MarketDataRequest
                mock_md_request = Mock()
                mock_request.return_value = mock_md_request

                retrieval._create_market_data_request("cache_algo_return")

                # Verify MarketDataRequest was called with correct parameters
                mock_request.assert_called_once()
                call_args = mock_request.call_args[1]  # Get keyword arguments

                assert call_args['data_source'] == 'yahoo'
                assert call_args['tickers'] == ['AAPL', 'GOOGL']
                assert call_args['fields'] == ['close']
                assert call_args['start_date'] == 'year'
                assert call_args['cache_algo'] == 'cache_algo_return'

    def test_load_from_cache_success(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test successful cache loading."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.return_value = sample_market_data

            result = retrieval.load_from_cache()

            mock_market.fetch_market.assert_called_once()
            pd.testing.assert_frame_equal(result, sample_market_data)

    def test_load_from_cache_empty(
        self, sample_config: DataRetrievalConfig, mock_market: Mock
    ) -> None:
        """Test cache loading with empty result."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.return_value = pd.DataFrame()

            result = retrieval.load_from_cache()

            assert result is None

    def test_load_from_cache_exception(
        self, sample_config: DataRetrievalConfig, mock_market: Mock
    ) -> None:
        """Test cache loading with exception."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.side_effect = Exception("Cache error")

            result = retrieval.load_from_cache()

            assert result is None

    def test_download_data_success(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test successful data download."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.return_value = sample_market_data

            result = retrieval.download_data()

            mock_market.fetch_market.assert_called_once()
            pd.testing.assert_frame_equal(result, sample_market_data)

    def test_download_data_empty(
        self, sample_config: DataRetrievalConfig, mock_market: Mock
    ) -> None:
        """Test download with empty result."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.return_value = pd.DataFrame()

            with pytest.raises(ValueError, match="Download returned empty or None data"):
                retrieval.download_data()

    def test_download_data_exception(
        self, sample_config: DataRetrievalConfig, mock_market: Mock
    ) -> None:
        """Test download with exception."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.side_effect = Exception("Download error")

            with pytest.raises(Exception, match="Download error"):
                retrieval.download_data()

    def test_get_data_cache_first(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test get_data uses cache first."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)
            mock_market.fetch_market.return_value = sample_market_data

            result = retrieval.get_data()

            # Should only be called once for cache attempt
            assert mock_market.fetch_market.call_count == 1
            pd.testing.assert_frame_equal(result, sample_market_data)

    def test_get_data_fallback_to_download(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test get_data falls back to download when cache fails."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            # First call (cache) returns empty, second call (download) returns data
            mock_market.fetch_market.side_effect = [pd.DataFrame(), sample_market_data]

            result = retrieval.get_data()

            # Should be called twice (cache attempt + download)
            assert mock_market.fetch_market.call_count == 2
            pd.testing.assert_frame_equal(result, sample_market_data)

    def test_validate_data_quality_success(
        self, sample_config: DataRetrievalConfig, sample_market_data: pd.DataFrame
    ) -> None:
        """Test data quality validation with clean data."""
        with (patch('backtester.data.data_retrieval.Market'),):
            retrieval = DataRetrieval(sample_config)

            # Configure mock for no duplicates
            with patch.object(
                retrieval.data_quality, 'count_repeated_dates', return_value=(0, None)
            ):
                result = retrieval.validate_data_quality(sample_market_data)

            assert result["total_rows"] == len(sample_market_data)
            assert result["total_columns"] == len(sample_market_data.columns)
            assert "duplicated_dates" in result
            assert result["duplicated_dates"]["count"] == 0
            assert "missing_values" in result
            assert result["missing_values"]["total_missing"] == 0
            assert "completeness" in result
            assert result["completeness"]["percentage_complete"] == 100.0

    def test_validate_data_quality_with_duplicates(
        self, sample_config: DataRetrievalConfig
    ) -> None:
        """Test data quality validation with duplicated dates."""
        with (patch('backtester.data.data_retrieval.Market'),):
            retrieval = DataRetrieval(sample_config)

            # Create data with duplicates
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
            # Add duplicate
            data = pd.concat([data, data.iloc[[2]]])  # Duplicate the 3rd row

            # Mock the count_repeated_dates method on the retrieval's data_quality instance
            with patch.object(
                retrieval.data_quality, 'count_repeated_dates', return_value=(1, data.index[5])
            ):
                result = retrieval.validate_data_quality(data)

            assert result["duplicated_dates"]["count"] == 1

    def test_validate_data_quality_with_missing_values(
        self, sample_config: DataRetrievalConfig
    ) -> None:
        """Test data quality validation with missing values."""
        with (patch('backtester.data.data_retrieval.Market'),):
            retrieval = DataRetrieval(sample_config)

            # Create data with missing values
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            data = pd.DataFrame({'Close': [100, np.nan, 102, 103, 104]}, index=dates)

            with patch.object(
                retrieval.data_quality, 'count_repeated_dates', return_value=(0, None)
            ):
                result = retrieval.validate_data_quality(data)

            assert result["missing_values"]["total_missing"] == 1
            assert result["missing_values"]["by_column"]["Close"] == 1
            assert result["completeness"]["percentage_complete"] < 100.0

    def test_validate_data_quality_empty_data(self, sample_config: DataRetrievalConfig) -> None:
        """Test data quality validation with empty data."""
        with (
            patch('backtester.data.data_retrieval.Market'),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            result = retrieval.validate_data_quality(pd.DataFrame())

            assert "error" in result
            assert "No data to validate" in result["error"]

    def test_get_data_with_validation_success(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test get_data_with_validation with clean data."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality') as mock_dq,
        ):
            retrieval = DataRetrieval(sample_config)

            # Mock successful cache hit
            mock_market.fetch_market.return_value = sample_market_data

            # Mock no data quality issues
            mock_dq_instance = Mock()
            mock_dq_instance.count_repeated_dates.return_value = (0, None)
            mock_dq.return_value = mock_dq_instance

            result = retrieval.get_data_with_validation()

            assert result["success"] is True
            assert result["data"] is not None
            assert "validation_results" in result
            assert len(result["quality_issues"]) == 0

    def test_get_data_with_validation_with_issues(
        self,
        sample_config: DataRetrievalConfig,
        sample_market_data: pd.DataFrame,
        mock_market: Mock,
    ) -> None:
        """Test get_data_with_validation with data quality issues."""
        with (patch('backtester.data.data_retrieval.Market', return_value=mock_market),):
            retrieval = DataRetrieval(sample_config)

            # Mock successful cache hit
            mock_market.fetch_market.return_value = sample_market_data

            # Mock data quality issues (duplicates)
            with patch.object(
                retrieval.data_quality,
                'count_repeated_dates',
                return_value=(5, ["2023-01-05", "2023-01-10"]),
            ):
                result = retrieval.get_data_with_validation()

            assert result["success"] is False
            assert result["data"] is not None
            assert "validation_results" in result
            assert len(result["quality_issues"]) > 0
            # Check that the quality issues contain information about duplicates
            quality_issues_text = " ".join(result["quality_issues"]).lower()
            assert "duplicated" in quality_issues_text or "completeness" in quality_issues_text

    def test_update_config(self, sample_config: DataRetrievalConfig) -> None:
        """Test configuration updates."""
        with (
            patch('backtester.data.data_retrieval.Market'),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            # Update some config values
            retrieval.update_config(
                data_source="test_source", tickers=["MSFT", "AMZN"], start_date="2023-01-01"
            )

            assert retrieval.config.data_source == "test_source"
            assert retrieval.config.tickers == ["MSFT", "AMZN"]
            assert retrieval.config.start_date == "2023-01-01"

    def test_string_representations(self, sample_config: DataRetrievalConfig) -> None:
        """Test string and repr methods."""
        with (
            patch('backtester.data.data_retrieval.Market'),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            str_repr = str(retrieval)
            repr_repr = repr(retrieval)

            assert "DataRetrieval" in str_repr
            assert "DataRetrieval" in repr_repr
            assert "yahoo" in repr_repr
            assert "AAPL" in repr_repr

    def test_get_data_with_validation_empty_data(
        self, sample_config: DataRetrievalConfig, mock_market: Mock
    ) -> None:
        """Test get_data_with_validation with no data retrieved."""
        with (
            patch('backtester.data.data_retrieval.Market', return_value=mock_market),
            patch('backtester.data.data_retrieval.DataQuality'),
        ):
            retrieval = DataRetrieval(sample_config)

            # Mock empty return from cache and download
            mock_market.fetch_market.return_value = None

            # The get_data method should raise ValueError when download returns None
            with pytest.raises(ValueError, match="Download returned empty or None data"):
                retrieval.get_data()

    def test_validate_data_quality_outliers(self, sample_config: DataRetrievalConfig) -> None:
        """Test outlier detection in data quality validation."""
        with (patch('backtester.data.data_retrieval.Market'),):
            retrieval = DataRetrieval(sample_config)

            # Create data with outliers
            dates = pd.date_range('2023-01-01', periods=20, freq='D')
            data = pd.DataFrame({'Close': [100] * 15 + [200, 201, 202, 203, 204]}, index=dates)

            with patch.object(
                retrieval.data_quality, 'count_repeated_dates', return_value=(0, None)
            ):
                result = retrieval.validate_data_quality(data, check_outliers=True)

            assert "outliers" in result
            outlier_info = result["outliers"]["Close"]
            assert outlier_info["count"] > 0
            assert outlier_info["percentage"] > 0
            assert "bounds" in outlier_info
