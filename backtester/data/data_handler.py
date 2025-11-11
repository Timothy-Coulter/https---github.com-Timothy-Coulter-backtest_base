"""Data handler module for backtester compatibility.

This module provides the DataHandler class and get_data function that are expected
by the backtester.main module, while using the existing DataRetrieval infrastructure.
"""

from typing import Any

import pandas as pd

from backtester.core.config import DataRetrievalConfig
from backtester.data.data_retrieval import DataRetrieval


class DataHandler:
    """Data handler class for backtester compatibility.

    This class wraps the DataRetrieval functionality to provide the expected
    interface for the backtester system.
    """

    def __init__(self, config: Any = None):
        """Initialize DataHandler with optional configuration.

        Args:
            config: Optional configuration object. If None, uses default config.
        """
        if config is None:
            # Create default configuration
            self.config = DataRetrievalConfig()
        else:
            # Use provided configuration
            self.config = config

        # Initialize DataRetrieval with the configuration
        self.data_retrieval = DataRetrieval(self.config)

    def get_data(self) -> pd.DataFrame:
        """Get data using the configured data retrieval.

        Returns:
            pd.DataFrame: Market data
        """
        return self.data_retrieval.get_data()

    def load_data(
        self,
        tickers: list[str],
        start_date: str,
        finish_date: str,
        freq: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data with specified parameters.

        Args:
            tickers: List of ticker symbols
            start_date: Start date string
            finish_date: End date string
            freq: Data frequency
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Market data
        """
        # Update configuration with new parameters
        self.config.tickers = tickers
        self.config.start_date = start_date
        self.config.finish_date = finish_date
        self.config.freq = freq

        # Update any additional parameters
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        return self.get_data()


def get_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    **kwargs: Any,
) -> pd.DataFrame:
    """Get data for a single ticker with specified parameters.

    This is a convenience function that creates a temporary DataHandler
    and returns the requested data.

    Args:
        ticker: Ticker symbol
        start_date: Start date string
        end_date: End date string
        interval: Data interval (e.g., '1d', '1mo', '1h')
        **kwargs: Additional parameters for data retrieval

    Returns:
        pd.DataFrame: Market data for the specified ticker
    """
    # Create default configuration
    config = DataRetrievalConfig()

    # Set the basic parameters
    config.tickers = [ticker]
    config.start_date = start_date
    config.finish_date = end_date
    config.freq = interval

    # Update with any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create temporary data handler and get data
    data_handler = DataHandler(config)
    return data_handler.get_data()


# Create a default global instance for backward compatibility
default_data_handler = DataHandler()
