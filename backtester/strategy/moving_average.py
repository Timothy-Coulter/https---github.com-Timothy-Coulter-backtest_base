"""
Dual-Pool Moving Average Strategy Implementation.

This module implements a sophisticated moving average strategy based on the dual-pool
leverage approach from the existing codebase. It uses two pools (base and alpha) with
different leverage levels and rebalancing logic.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.strategy.base import BaseStrategy, Signal


class DualPoolMovingAverageStrategy(BaseStrategy):
    """Dual-pool moving average strategy with leverage and rebalancing."""
    
    def __init__(self,
                 name: str = "DualPoolMA",
                 ma_short: int = 5,
                 ma_long: int = 20,
                 leverage_base: float = 1.0,
                 leverage_alpha: float = 3.0,
                 base_to_alpha_split: float = 0.2,
                 alpha_to_base_split: float = 0.2,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize the dual-pool moving average strategy.
        
        Args:
            name: Strategy name
            ma_short: Short-term moving average period
            ma_long: Long-term moving average period
            leverage_base: Base pool leverage factor
            leverage_alpha: Alpha pool leverage factor
            base_to_alpha_split: Fraction of positive base gains to transfer to alpha
            alpha_to_base_split: Fraction of positive alpha gains to transfer to base
            logger: Optional logger instance
        """
        super().__init__(name, logger)
        
        # Strategy parameters
        self.ma_short: int = ma_short
        self.ma_long: int = ma_long
        self.leverage_base: float = leverage_base
        self.leverage_alpha: float = leverage_alpha
        self.base_to_alpha_split: float = base_to_alpha_split
        self.alpha_to_base_split: float = alpha_to_base_split
        
        # Strategy state
        self.price_history: List[float] = []
        self.ma_short_history: List[float] = []
        self.ma_long_history: List[float] = []
        self.current_trend: Optional[str] = None
        
    def get_required_columns(self) -> List[str]:
        """Get required data columns for this strategy.
        
        Returns:
            List of required column names
        """
        return ["Close", "High", "Low"]
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on moving average crossover.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of signal dictionaries
        """
        if not self.validate_data(data):
            return []
            
        signals = []
        current_price = data["Close"].iloc[-1]
        current_time = data.index[-1]
        
        # Update price history
        self.price_history.append(current_price)
        
        # Calculate moving averages if we have enough data
        if len(self.price_history) >= self.ma_long:
            short_ma = float(np.mean(self.price_history[-self.ma_short:]))
            long_ma = float(np.mean(self.price_history[-self.ma_long:]))
            
            self.ma_short_history.append(short_ma)
            self.ma_long_history.append(long_ma)
            
            # Generate signals based on crossover
            new_trend = self._determine_trend(short_ma, long_ma)
            
            if new_trend != self.current_trend and self.current_trend is not None:
                # Trend change detected - generate signals
                if new_trend == "BULLISH":
                    signal = Signal(
                        timestamp=current_time,
                        signal_type="BUY",
                        price=current_price,
                        metadata={
                            "strategy": self.name,
                            "trend": new_trend,
                            "short_ma": short_ma,
                            "long_ma": long_ma,
                            "leverage_base": self.leverage_base,
                            "leverage_alpha": self.leverage_alpha,
                            "pool_type": "BOTH"
                        }
                    )
                    signals.append(signal.to_dict())
                    
                elif new_trend == "BEARISH":
                    signal = Signal(
                        timestamp=current_time,
                        signal_type="SELL", 
                        price=current_price,
                        metadata={
                            "strategy": self.name,
                            "trend": new_trend,
                            "short_ma": short_ma,
                            "long_ma": long_ma,
                            "leverage_base": self.leverage_base,
                            "leverage_alpha": self.leverage_alpha,
                            "pool_type": "BOTH"
                        }
                    )
                    signals.append(signal.to_dict())
            
            self.current_trend = new_trend
            
            self.logger.debug(f"Generated signals: {len(signals)} signals at {current_time}")
            
        return signals
    
    def _determine_trend(self, short_ma: float, long_ma: float) -> str:
        """Determine market trend based on moving averages.
        
        Args:
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            
        Returns:
            Trend string ('BULLISH', 'BEARISH', or 'NEUTRAL')
        """
        if short_ma > long_ma * 1.01:  # 1% threshold for bullish
            return "BULLISH"
        elif short_ma < long_ma * 0.99:  # 1% threshold for bearish  
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            "ma_short": self.ma_short,
            "ma_long": self.ma_long,
            "leverage_base": self.leverage_base,
            "leverage_alpha": self.leverage_alpha,
            "base_to_alpha_split": self.base_to_alpha_split,
            "alpha_to_base_split": self.alpha_to_base_split
        }
    
    def get_current_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get current signals for a given data point.
        
        Args:
            data: Single row DataFrame with current market data
            
        Returns:
            List of current signals
        """
        # For compatibility with backtesting engine
        return self.generate_signals(data)
    
    def calculate_leverage_allocation(self, total_capital: float) -> Dict[str, float]:
        """Calculate optimal allocation between base and alpha pools.
        
        Args:
            total_capital: Total available capital
            
        Returns:
            Dictionary with pool allocations
        """
        # Simple equal split between pools (can be optimized)
        base_allocation = total_capital * 0.5
        alpha_allocation = total_capital * 0.5
        
        return {
            "base_allocation": base_allocation,
            "alpha_allocation": alpha_allocation,
            "base_leveraged_value": base_allocation * self.leverage_base,
            "alpha_leveraged_value": alpha_allocation * self.leverage_alpha
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.price_history.clear()
        self.ma_short_history.clear()
        self.ma_long_history.clear()
        self.current_trend = None
        self.logger.info(f"Strategy {self.name} reset")


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy (single pool)."""
    
    def __init__(self,
                 name: str = "SimpleMA",
                 ma_short: int = 5,
                 ma_long: int = 20,
                 leverage: float = 1.0,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize the simple moving average strategy.
        
        Args:
            name: Strategy name
            ma_short: Short-term moving average period
            ma_long: Long-term moving average period
            leverage: Leverage factor for positions
            logger: Optional logger instance
        """
        super().__init__(name, logger)
        
        self.ma_short: int = ma_short
        self.ma_long: int = ma_long
        self.leverage: float = leverage
        self.current_trend: Optional[str] = None
        
    def get_required_columns(self) -> List[str]:
        """Get required data columns for this strategy.
        
        Returns:
            List of required column names
        """
        return ["Close"]
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on simple moving average crossover.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of signal dictionaries
        """
        if not self.validate_data(data):
            return []
            
        signals = []
        current_price = data["Close"].iloc[-1]
        current_time = data.index[-1]
        
        # Calculate moving averages
        if len(data) >= self.ma_long:
            short_ma = data["Close"].rolling(window=self.ma_short).mean().iloc[-1]
            long_ma = data["Close"].rolling(window=self.ma_long).mean().iloc[-1]
            
            # Determine trend
            new_trend = "BULLISH" if short_ma > long_ma else "BEARISH"
            
            # Generate signal on trend change
            if new_trend != self.current_trend and self.current_trend is not None:
                signal_type = "BUY" if new_trend == "BULLISH" else "SELL"
                
                signal = Signal(
                    timestamp=current_time,
                    signal_type=signal_type,
                    price=current_price,
                    metadata={
                        "strategy": self.name,
                        "trend": new_trend,
                        "short_ma": short_ma,
                        "long_ma": long_ma,
                        "leverage": self.leverage
                    }
                )
                signals.append(signal.to_dict())
            
            self.current_trend = new_trend
            
        return signals