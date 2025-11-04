"""
Data processing and validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


class DataUtils:
    """Utility class for data processing and validation."""
    
    def validate_ohlcv(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data format.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return False, errors
        
        # Check for negative values
        for col in required_cols:
            if (data[col] < 0).any():
                errors.append(f"Negative values found in {col} column")
        
        # Check OHLC relationships
        if len(data) > 0:
            if (data['High'] < data['Open']).any():
                errors.append("High price is less than Open price in some rows")
            if (data['High'] < data['Close']).any():
                errors.append("High price is less than Close price in some rows")
            if (data['Low'] > data['Open']).any():
                errors.append("Low price is greater than Open price in some rows")
            if (data['Low'] > data['Close']).any():
                errors.append("Low price is greater than Close price in some rows")
            if (data['High'] < data['Low']).any():
                errors.append("High price is less than Low price in some rows")
        
        return len(errors) == 0, errors
    
    def analyze_price_movements(self, prices: pd.Series) -> Dict[str, Any]:
        """Analyze price movements and trends.
        
        Args:
            prices: Series of prices
            
        Returns:
            Dictionary with analysis results
        """
        returns = prices.pct_change().dropna()
        
        # Calculate trend
        recent_returns = returns.tail(20)
        avg_return = recent_returns.mean()
        if avg_return > 0.001:
            trend = 'uptrend'
        elif avg_return < -0.001:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate recovery factor
        total_return = cumulative.iloc[-1] - 1
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'trend': trend,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'recovery_factor': recovery_factor
        }
    
    def fill_missing_values(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Fill missing values in data.
        
        Args:
            data: DataFrame with missing values
            method: Method for filling ('forward_fill', 'backward_fill', 'linear', 'mean')
            
        Returns:
            DataFrame with filled missing values
        """
        if method == 'forward_fill':
            return data.ffill()
        elif method == 'backward_fill':
            return data.bfill()
        elif method == 'linear':
            return data.interpolate(method='linear')
        elif method == 'mean':
            return data.fillna(data.mean())
        else:
            raise ValueError(f"Unknown fill method: {method}")
    
    def remove_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from data.
        
        Args:
            data: DataFrame with data
            columns: List of columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            DataFrame with outliers removed
        """
        data_clean = data.copy()
        
        for column in columns:
            if column not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                data_clean = data_clean[mask]
                
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                mask = z_scores < 3
                data_clean = data_clean[mask]
        
        return data_clean
    
    def align_to_frequency(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Align data to specified frequency.
        
        Args:
            data: DataFrame with timestamp index
            frequency: Target frequency ('D', 'H', 'M', etc.)
            
        Returns:
            Aligned DataFrame
        """
        return data.resample(frequency).last()
    
    def resample_data(self, data: pd.DataFrame, frequency: str, method: str = 'ohlcv') -> pd.DataFrame:
        """Resample data to different frequency.
        
        Args:
            data: DataFrame with OHLCV data
            frequency: Target frequency
            method: Resampling method
            
        Returns:
            Resampled DataFrame
        """
        if method == 'ohlcv':
            ohlc_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }
            
            if 'Volume' in data.columns:
                ohlc_dict['Volume'] = 'sum'
                
            return data.resample(frequency).agg(ohlc_dict)
        else:
            return data.resample(frequency).agg(method)
    
    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Correlation matrix
        """
        returns = data.pct_change().dropna()
        return returns.corr()
    
    def synchronize_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Synchronize multiple dataframes to common date range.
        
        Args:
            dataframes: List of DataFrames
            
        Returns:
            Synchronized DataFrame
        """
        if not dataframes:
            return pd.DataFrame()
        
        # Find common index
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)
        
        # Filter all dataframes to common index
        synchronized = pd.concat([df.loc[common_index] for df in dataframes], axis=1)
        return synchronized