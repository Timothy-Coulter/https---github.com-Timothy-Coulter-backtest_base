"""
Market data handler for loading and managing financial data.

This module provides functionality to load, validate, and manage market data
for backtesting purposes. It handles data from yfinance and other sources.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
import logging


class DataHandler:
    """Handles market data loading, validation, and preprocessing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the data handler.
        
        Args:
            config: Optional configuration dictionary
            logger: Optional logger instance for logging operations.
        """
        self.config = config or {
            'data_source': 'yahoo',
            'cache_enabled': True,
            'validation_strict': False,
            'preprocess_data': False,
            'fill_missing': False
        }
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # Data cache for performance
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Compatibility attributes for tests
        self.data_validator = DataValidator()
        self.data_preprocessor = DataPreprocessor()
        
    @property
    def data_loader(self) -> 'DataHandler':
        """Compatibility property for tests."""
        return self
        
    def get_data(
        self,
        ticker: str,
        start_date: str = "1990-01-01",
        end_date: str = "2025-01-01",
        interval: str = "1mo"
    ) -> pd.DataFrame:
        """Load market data for a given ticker.
        
        Args:
            ticker: Stock/asset ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data frequency ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLC data (Open, High, Low, Close)
            
        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        try:
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
            
            # Check cache first
            cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
            if self.config.get('cache_enabled', True) and cache_key in self.data_cache:
                self.logger.info(f"Using cached data for {ticker}")
                return self.data_cache[cache_key]
            
            # Download data from yfinance
            data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Handle multi-index columns (when downloading multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                data = pd.DataFrame({
                    "Open": data["Open"][ticker],
                    "High": data["High"][ticker],
                    "Low": data["Low"][ticker],
                    "Close": data["Close"][ticker],
                    "Volume": data["Volume"][ticker] if "Volume" in data.columns else 0
                })
            else:
                # Ensure we have the required columns
                required_columns = ["Open", "High", "Low", "Close"]
                if not all(col in data.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in data.columns]
                    raise ValueError(f"Missing required columns: {missing}")
                    
                # Select only OHLC columns and add volume if available
                columns = ["Open", "High", "Low", "Close"]
                if "Volume" in data.columns:
                    columns.append("Volume")
                data = data[columns]
            
            # Clean and validate data
            data = self._clean_data(data)
            data = self._validate_data(data)
            
            # Process data if configured
            if self.config.get('preprocess_data', False):
                data = self.process(data)
            
            # Cache the data
            if self.config.get('cache_enabled', True):
                self.data_cache[cache_key] = data
            
            self.logger.info(f"Successfully loaded {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {e}")
            raise ValueError(f"Failed to load data for {ticker}: {e}")
    
    def load_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Load data for a single symbol (alias for get_data)."""
        return self.get_data(symbol, start_date, end_date, interval)
    
    def load_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_data(symbol, start_date, end_date, interval)
        return results
    
    def aggregate_data(self, data: pd.DataFrame, target_frequency: str, aggregation_method: str = "ohlcv") -> pd.DataFrame:
        """Aggregate data to target frequency."""
        if aggregation_method == "ohlcv":
            # Simple aggregation by resampling
            aggregated = data.resample(target_frequency).agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            return aggregated
        else:
            return data.resample(target_frequency).last().dropna()
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data according to configuration."""
        processed = data.copy()
        
        if self.config.get('fill_missing', False):
            processed = processed.ffill()
        
        if self.config.get('preprocess_data', False):
            processed = self.data_preprocessor.calculate_technical_indicators(processed)
        
        return processed
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the loaded data.
        
        Args:
            data: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Convert numeric columns to float if needed
        numeric_columns = ["Open", "High", "Low", "Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the cleaned data for common issues.
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If data fails validation checks
        """
        if len(data) == 0:
            raise ValueError("Data is empty after cleaning")
        
        # Check for reasonable price ranges
        if "Close" in data.columns:
            if (data["Close"] <= 0).any():
                self.logger.warning("Found non-positive prices in data")
                data = data[data["Close"] > 0]
        
        # Check for logical OHLC relationships
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            invalid_ohlc = (
                (data["High"] < data["Low"]) |
                (data["High"] < data["Open"]) |
                (data["High"] < data["Close"]) |
                (data["Low"] > data["Open"]) |
                (data["Low"] > data["Close"])
            )
            
            if invalid_ohlc.any():
                self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC records")
                if self.config.get('validation_strict', False):
                    raise ValueError("Data validation failed: invalid OHLC relationships")
                data = data[~invalid_ohlc]
        
        return data
    
    def compute_returns(self, data: pd.DataFrame, column: str = "Close") -> pd.Series:
        """Compute returns for a given price column.
        
        Args:
            data: DataFrame with price data
            column: Column name to compute returns for
            
        Returns:
            Series of returns
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        return data[column].pct_change().dropna()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with added technical indicators
        """
        data = data.copy()
        
        # Simple Moving Averages
        data["SMA_5"] = data["Close"].rolling(window=5).mean()
        data["SMA_10"] = data["Close"].rolling(window=10).mean()
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        
        # Exponential Moving Averages
        data["EMA_5"] = data["Close"].ewm(span=5).mean()
        data["EMA_10"] = data["Close"].ewm(span=10).mean()
        data["EMA_20"] = data["Close"].ewm(span=20).mean()
        
        # Price ratios
        data["Price_vs_SMA5"] = data["Close"] / data["SMA_5"]
        data["Price_vs_SMA20"] = data["Close"] / data["SMA_20"]
        
        # Volatility (rolling standard deviation)
        data["Volatility_20"] = data["Close"].rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        
        return data
    
    def export_data(self, data: pd.DataFrame, format: str = "csv", path: Optional[str] = None) -> str:
        """Export data in specified format."""
        if format.lower() == "csv":
            filename = path or "data_export.csv"
            data.to_csv(filename)
            return filename
        elif format.lower() == "json":
            filename = path or "data_export.json"
            data.to_json(filename)
            return filename
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the data."""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "date_range": {
                "start": data.index[0] if len(data) > 0 else None,
                "end": data.index[-1] if len(data) > 0 else None
            },
            "missing_values": data.isnull().sum().to_dict()
        }
    
    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the loaded data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "start_date": data.index[0],
            "end_date": data.index[-1],
            "total_periods": len(data),
            "columns": list(data.columns)
        }
        
        if "Close" in data.columns:
            stats.update({
                "initial_price": data["Close"].iloc[0],
                "final_price": data["Close"].iloc[-1],
                "min_price": data["Close"].min(),
                "max_price": data["Close"].max(),
                "total_return": (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
            })
            
            returns = self.compute_returns(data["Close"])
            stats.update({
                "mean_return": returns.mean(),
                "volatility": returns.std(),
                "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0
            })
        
        return stats


# Compatibility function for existing code
def get_data(ticker: str, start_date: str = "1990-01-01", end_date: str = "2025-01-01", interval: str = "1mo") -> pd.DataFrame:
    """Compatibility function for existing code.
    
    This function maintains compatibility with the existing utils.py get_data function.
    """
    handler = DataHandler()
    return handler.get_data(ticker, start_date, end_date, interval)


# Legacy compatibility classes for tests
class DataLoader:
    """Legacy DataLoader class for test compatibility."""
    
    def __init__(self) -> None:
        """Initialize the data loader."""
        self.data_sources = {'yahoo': True}
        self.default_source = 'yahoo'
    
    def load_from_source(self, source: str, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Load data from specified source."""
        if source == 'yahoo':
            handler = DataHandler()
            return handler.get_data(ticker, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(csv_path)
    
    def load_from_database(self, db_type: str, db_path: str, query: str, params: List[str]) -> pd.DataFrame:
        """Load data from database."""
        if db_type == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(db_path)
            data = pd.read_sql(query, conn, params=params)
            conn.close()
            return data
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def standardize_data_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format."""
        standardized = data.copy()
        
        # Rename columns to standard format
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume',
            'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in standardized.columns:
                standardized = standardized.rename(columns={old_col: new_col})
        
        return standardized
    
    def select_data_source(self, symbol: str) -> str:
        """Select appropriate data source for symbol."""
        # Simple logic - could be enhanced
        if 'BTC' in symbol or 'ETH' in symbol:
            return 'crypto'
        return 'yahoo'


class DataValidator:
    """Legacy DataValidator class for test compatibility."""
    
    def __init__(self) -> None:
        """Initialize the data validator."""
        self.validation_rules = {'ohlcv': True}
        self.strict_mode = False
    
    def validate_ohlcv_data(self, data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate OHLCV data."""
        required_columns = ["Open", "High", "Low", "Close"]
        errors = []
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Check High/Low relationships
        invalid_hl = ((data["High"] < data["Low"]) | 
                     (data["High"] < data["Open"]) |
                     (data["High"] < data["Close"])).any()
        
        if invalid_hl:
            errors.append("Invalid High/Low price relationships found")
        
        return len(errors) == 0, errors
    
    def validate_high_low_relationship(self, data: pd.DataFrame) -> bool:
        """Validate high/low price relationships."""
        return ((data["High"] >= data["Low"]).all() and 
                (data["High"] >= data["Open"]).all() and
                (data["High"] >= data["Close"]).all())
    
    def validate_price_movements(self, data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate price movement data."""
        warnings = []
        
        # Check for unusual price movements (gaps > 50%)
        if "Close" in data.columns:
            returns = data["Close"].pct_change().dropna()
            unusual_moves = (returns.abs() > 0.5).any()
            if unusual_moves:
                warnings.append("Unusual price movements detected (>50% gap)")
        
        return len(warnings) == 0, warnings
    
    def validate_price_movement(self, data: pd.DataFrame) -> bool:
        """Validate price movement data."""
        # Basic validation - check for positive prices
        return (data[["Open", "High", "Low", "Close"]] > 0).all().all()
    
    def validate_volume_data(self, data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate volume data."""
        errors = []
        
        if "Volume" in data.columns:
            negative_volume = (data["Volume"] < 0).any()
            if negative_volume:
                errors.append("Negative volume values found")
            
            zero_volume = (data["Volume"] == 0).all()
            if zero_volume:
                errors.append("All volume values are zero")
        
        return len(errors) == 0, errors
    
    def check_data_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness."""
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        # Simple gap analysis - could be enhanced
        gap_analysis = {
            "total_missing": int(data.isnull().sum().sum()),
            "columns_with_gaps": list(data.columns[data.isnull().any()].values)
        }
        
        return {
            "missing_percentage": missing_percentage,
            "gap_analysis": gap_analysis
        }


class DataPreprocessor:
    """Legacy DataPreprocessor class for test compatibility."""
    
    def __init__(self) -> None:
        """Initialize the data preprocessor."""
        self.scalers: Dict[str, Any] = {}
        self.transformers: Dict[str, Any] = {}
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """Handle missing values in data."""
        if method == "forward_fill":
            return data.ffill()
        elif method == "drop":
            return data.dropna()
        elif method == "interpolate":
            return data.interpolate()
        else:
            return data.fillna(0)
    
    def normalize_prices(self, data: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
        """Normalize prices to start at 100."""
        result = data.copy()
        if column in data.columns:
            result[f'{column}_normalized'] = (data[column] / data[column].iloc[0]) * 100
        return result
    
    def calculate_returns(self, data: pd.DataFrame, column: str = "Close", method: str = "simple") -> pd.DataFrame:
        """Calculate returns."""
        result = data.copy()
        if column in data.columns:
            if method == "simple":
                result['returns'] = data[column].pct_change()
            else:
                result['returns'] = data[column].pct_change()
        return result
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        result = data.copy()
        
        if "Close" in data.columns:
            # Simple Moving Averages
            result['sma_5'] = data["Close"].rolling(window=5).mean()
            result['ema_5'] = data["Close"].ewm(span=5).mean()
            
            # RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Volatility
            result['volatility'] = data["Close"].rolling(window=20).std()
        
        return result
    
    def resample_data(self, data: pd.DataFrame, frequency: str, method: str = "ohlcv") -> pd.DataFrame:
        """Resample data to different frequency."""
        if method == "ohlcv":
            return data.resample(frequency).agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            return data.resample(frequency).last().dropna()