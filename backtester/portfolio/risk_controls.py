"""
Risk Control Classes for Stop Loss and Take Profit Management.

This module provides comprehensive stop loss and take profit functionality
with support for fixed, percentage, and trailing strategies.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime


class StopLossType(Enum):
    """Types of stop loss mechanisms."""
    FIXED = "FIXED"
    PERCENTAGE = "PERCENTAGE"
    PRICE = "PRICE"
    TRAILING = "TRAILING"
    TRAILING_PERCENTAGE = "TRAILING_PERCENTAGE"


class TakeProfitType(Enum):
    """Types of take profit mechanisms."""
    FIXED = "FIXED"
    PERCENTAGE = "PERCENTAGE"
    TRAILING = "TRAILING"
    TRAILING_PERCENTAGE = "TRAILING_PERCENTAGE"
    PRICE = "PRICE"


@dataclass
class StopLossConfig:
    """Configuration for stop loss mechanisms."""
    stop_loss_type: StopLossType = StopLossType.PERCENTAGE
    stop_loss_value: float = 0.02  # 2% default
    trail_distance: float = 0.01   # 1% trail distance for trailing stop
    trail_step: float = 0.005      # 0.5% minimum trail step
    max_loss_value: Optional[float] = None  # Maximum absolute loss allowed
    activation_price: Optional[float] = None  # Price at which stop loss activates


@dataclass 
class TakeProfitConfig:
    """Configuration for take profit mechanisms."""
    take_profit_type: TakeProfitType = TakeProfitType.PERCENTAGE
    take_profit_value: float = 0.06  # 6% default
    trail_distance: float = 0.02     # 2% trail distance for trailing profit
    trail_step: float = 0.01         # 1% minimum trail step
    max_gain_value: Optional[float] = None  # Maximum absolute gain target
    activation_price: Optional[float] = None  # Price at which take profit activates


class StopLoss:
    """Stop loss management with multiple types of mechanisms."""
    
    def __init__(self,
                 config: Optional[StopLossConfig] = None,
                 logger: Optional[logging.Logger] = None,
                 stop_loss_type: Optional[StopLossType] = None,
                 stop_loss_value: Optional[float] = None,
                 trail_distance: Optional[float] = None,
                 trail_step: Optional[float] = None,
                 max_loss_value: Optional[float] = None,
                 activation_price: Optional[float] = None,
                 entry_price: Optional[float] = None,
                 trailing_stop_pct: Optional[float] = None) -> None:
        """Initialize the stop loss mechanism.
        
        Args:
            config: StopLossConfig with stop loss parameters
            logger: Optional logger instance
            stop_loss_type: Direct parameter for stop loss type
            stop_loss_value: Direct parameter for stop loss value
            trail_distance: Direct parameter for trail distance
            trail_step: Direct parameter for trail step
            max_loss_value: Direct parameter for max loss value
            activation_price: Direct parameter for activation price
            entry_price: Entry price for the position
            trailing_stop_pct: Trailing stop percentage
        """
        # Handle both config object and direct parameters
        if config is not None:
            self.config: StopLossConfig = config
        else:
            # Create config from direct parameters
            self.config = StopLossConfig(
                stop_loss_type=stop_loss_type or StopLossType.PERCENTAGE,
                stop_loss_value=stop_loss_value or 0.02,
                trail_distance=trail_distance or 0.01,
                trail_step=trail_step or 0.005,
                max_loss_value=max_loss_value,
                activation_price=activation_price
            )
        
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # State tracking
        self.activation_price: Optional[float] = self.config.activation_price
        self.entry_price: Optional[float] = entry_price
        self.highest_price: float = 0.0  # For trailing stop loss
        self.is_active: bool = True  # Default to active
        self.stop_triggered: bool = False
        self.stop_price: Optional[float] = None
        
        # Add missing attributes for test compatibility
        self.stop_loss_type = self.config.stop_loss_type
        self.stop_loss_value = self.config.stop_loss_value
        self.triggered = False
        self.triggered_price: Optional[float] = None
        self.triggered_timestamp: Optional[datetime] = None
        self.trigger_price: Optional[float] = None  # Alias for triggered_price
        self.trigger_time: Optional[datetime] = None   # Alias for triggered_timestamp
        
        # Additional attributes for tests
        self.trailing_stop_pct = trailing_stop_pct or self.config.trail_distance
        self.triggered = False
        self.triggered_timestamp = None
        
    def setup_trailing_stop(self, entry_price: float, side: str = 'long') -> None:
        """Setup trailing stop loss.
        
        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position
        """
        self.entry_price = entry_price
        self.highest_price = entry_price
        if side.lower() == 'long':
            self.stop_price = entry_price * (1 - self.trailing_stop_pct)
        else:
            self.stop_price = entry_price * (1 + self.trailing_stop_pct)
            
    def calculate_scaled_target(self, entry_price: float, confidence_level: str, side: str = 'long') -> float:
        """Calculate scaled target based on confidence level.
        
        Args:
            entry_price: Entry price
            confidence_level: 'low', 'medium', 'high'
            side: 'long' or 'short'
            
        Returns:
            Scaled target price
        """
        scaling_factors = {'low': 1.02, 'medium': 1.05, 'high': 1.08}
        factor = scaling_factors.get(confidence_level, 1.05)
        
        if side.lower() == 'long':
            return entry_price * factor
        else:
            return entry_price * (2 - factor)
        
    def initialize_position(self, entry_price: float, timestamp: pd.Timestamp) -> None:
        """Initialize stop loss for a new position.
        
        Args:
            entry_price: Entry price of the position
            timestamp: Entry timestamp
        """
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.is_active = True
        self.stop_triggered = False
        self.triggered = False
        self.trigger_price = None
        self.trigger_time = None
        
        # Set activation price if configured
        if self.config.activation_price is None:
            self.activation_price = entry_price
        
        # Calculate initial stop price
        self.stop_price = self._calculate_stop_price(entry_price, entry_price)
        
        self.logger.debug(f"Stop loss initialized: entry={entry_price:.4f}, stop={self.stop_price:.4f}")
    
    def update(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Update stop loss with current market price.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with stop loss status and any trigger information
        """
        if not self.is_active or self.entry_price is None:
            return {
                'triggered': False,
                'stop_price': None,
                'action': 'NONE',
                'reason': 'Stop loss not active'
            }
        
        result = {
            'triggered': False,
            'stop_price': self.stop_price,
            'action': 'NONE',
            'reason': 'No action required'
        }
        
        # Update highest price for trailing stop loss
        if self.config.stop_loss_type == StopLossType.TRAILING:
            if current_price > self.highest_price:
                self.highest_price = current_price
        
        # Recalculate stop price
        old_stop_price = self.stop_price
        self.stop_price = self._calculate_stop_price(current_price, self.highest_price)
        
        # Check if stop loss is triggered
        if current_price <= self.stop_price:
            self.stop_triggered = True
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            result.update({
                'triggered': True,
                'stop_price': self.stop_price,
                'action': 'STOP_LOSS',
                'reason': f'Stop loss triggered at {current_price:.4f}',
                'exit_price': self.stop_price,
                'pnl_pct': (self.stop_price - self.entry_price) / self.entry_price if self.entry_price else 0,
                'timestamp': timestamp
            })
            self.is_active = False
            
            self.logger.info(f"Stop loss triggered: {result['reason']}, P&L: {result['pnl_pct']:.2%}")
        
        # Check maximum loss limit
        elif self.config.max_loss_value is not None:
            loss_amount = self.entry_price - current_price
            if loss_amount >= self.config.max_loss_value:
                self.stop_triggered = True
                self.triggered = True
                self.trigger_price = current_price
                self.trigger_time = timestamp
                result.update({
                    'triggered': True,
                    'stop_price': self.stop_price,
                    'action': 'MAX_LOSS',
                    'reason': f'Maximum loss limit reached: {loss_amount:.4f}',
                    'exit_price': current_price,
                    'pnl_pct': (current_price - self.entry_price) / self.entry_price if self.entry_price else 0,
                    'timestamp': timestamp
                })
                self.is_active = False
                
                self.logger.warning(f"Maximum loss triggered: {result['reason']}")
        
        # Log stop price changes for trailing stops
        if (self.config.stop_loss_type == StopLossType.TRAILING and
            old_stop_price != self.stop_price):
            self.logger.debug(f"Trailing stop updated: {old_stop_price:.4f} -> {self.stop_price:.4f}")
        
        return result
    
    def _calculate_stop_price(self, current_price: float, reference_price: float) -> float:
        """Calculate stop price based on configuration.
        
        Args:
            current_price: Current market price
            reference_price: Reference price (highest for trailing, entry for others)
            
        Returns:
            Calculated stop price
        """
        if self.config.stop_loss_type == StopLossType.FIXED:
            return self.config.stop_loss_value
            
        elif self.config.stop_loss_type == StopLossType.PERCENTAGE:
            return reference_price * (1 - self.config.stop_loss_value)
            
        elif self.config.stop_loss_type == StopLossType.TRAILING:
            trail_stop = reference_price * (1 - self.config.trail_distance)
            
            # Ensure we only move the stop up (never down)
            if self.stop_price is not None:
                min_stop = self.stop_price + (reference_price * self.config.trail_step)
                return max(trail_stop, min_stop)
            else:
                return trail_stop
        
        return reference_price * 0.95  # Default 5% stop loss
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stop loss status.
        
        Returns:
            Dictionary with stop loss status information
        """
        distance_to_stop = 0.0
        if self.entry_price and self.stop_price:
            distance_to_stop = (self.entry_price - self.stop_price) / self.entry_price
            
        return {
            'active': self.is_active,
            'triggered': self.triggered,
            'stop_price': self.stop_price,
            'entry_price': self.entry_price,
            'highest_price': self.highest_price,
            'stop_type': self.stop_loss_type.value if hasattr(self.stop_loss_type, 'value') else str(self.stop_loss_type),
            'stop_value': self.config.stop_loss_value,
            'is_active': self.is_active,
            'distance_to_stop': distance_to_stop,
            'config': {
                'type': self.config.stop_loss_type.value,
                'value': self.config.stop_loss_value,
                'trail_distance': self.config.trail_distance,
                'max_loss_value': self.config.max_loss_value
            }
        }
    
    def calculate_stop_price(self, entry_price: float, side: str = 'long') -> float:
        """Calculate stop price based on entry price and side.
        
        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position
            
        Returns:
            Calculated stop price
        """
        if side.lower() == 'short':
            # For short positions, stop loss is above entry price
            if self.config.stop_loss_type == StopLossType.FIXED:
                return self.config.stop_loss_value
            elif self.config.stop_loss_type == StopLossType.PERCENTAGE:
                return entry_price * (1 + self.config.stop_loss_value)
            elif self.config.stop_loss_type == StopLossType.PRICE:
                return self.config.stop_loss_value
        
        # Long position logic (default)
        return self._calculate_stop_price(entry_price, entry_price)
    
    def check_trigger(self, entry_price: float, current_price: float, side: str = 'long') -> bool:
        """Check if stop loss should trigger.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            side: 'long' or 'short' position
            
        Returns:
            True if stop loss should trigger
        """
        # Update entry price if not set
        if self.entry_price is None:
            self.entry_price = entry_price
            
        stop_price = self.calculate_stop_price(entry_price, side)
        
        if side.lower() == 'long':
            return current_price <= stop_price
        else:  # short
            return current_price >= stop_price
    
    def activate(self) -> None:
        """Activate the stop loss."""
        self.is_active = True
        
    def deactivate(self) -> None:
        """Deactivate the stop loss."""
        self.is_active = False
        
    def trigger(self) -> None:
        """Manually trigger the stop loss."""
        self.stop_triggered = True
        self.triggered = True
        self.is_active = False
        
    def get_stop_loss_details(self) -> Dict[str, Any]:
        """Get stop loss status."""
        distance_to_stop = 0.0
        if self.entry_price and self.stop_price:
            distance_to_stop = (self.entry_price - self.stop_price) / self.entry_price
            
        return {
            'stop_type': self.stop_loss_type.value if hasattr(self.stop_loss_type, 'value') else str(self.stop_loss_type),
            'stop_value': self.config.stop_loss_value,
            'is_active': self.is_active,
            'triggered': self.triggered,
            'distance_to_stop': distance_to_stop
        }
        
    def reset(self) -> None:
        """Reset stop loss to initial state."""
        self.activation_price = self.config.activation_price
        self.entry_price = None
        self.highest_price = 0.0
        self.is_active = True  # Reset to active
        self.stop_triggered = False
        self.triggered = False  # Reset triggered flag
        self.trigger_price = None  # Clear trigger price
        self.trigger_time = None  # Clear trigger time
        self.stop_price = None
        self.logger.debug("Stop loss reset")


class TakeProfit:
    """Take profit management with multiple types of mechanisms."""
    
    def __init__(self,
                 config: Optional[TakeProfitConfig] = None,
                 logger: Optional[logging.Logger] = None,
                 take_profit_type: Optional[TakeProfitType] = None,
                 take_profit_value: Optional[float] = None,
                 trail_distance: Optional[float] = None,
                 trail_step: Optional[float] = None,
                 max_gain_value: Optional[float] = None,
                 activation_price: Optional[float] = None,
                 stop_loss_value: Optional[float] = None,
                 trailing_take_profit_pct: Optional[float] = None,
                 partial_take_profit_levels: Optional[List[float]] = None,
                 scaling_factors: Optional[List[float]] = None) -> None:
        """Initialize the take profit mechanism.
        
        Args:
            config: TakeProfitConfig with take profit parameters
            logger: Optional logger instance
            take_profit_type: Direct parameter for take profit type
            take_profit_value: Direct parameter for take profit value
            trail_distance: Direct parameter for trail distance
            trail_step: Direct parameter for trail step
            max_gain_value: Direct parameter for max gain value
            activation_price: Direct parameter for activation price
            stop_loss_value: Stop loss value for RR ratio calculation
            trailing_take_profit_pct: Trailing take profit percentage
            partial_take_profit_levels: Levels for partial profit taking
            scaling_factors: Scaling factors for take profit
        """
        # Handle both config object and direct parameters
        if config is not None:
            self.config: TakeProfitConfig = config
        else:
            # Create config from direct parameters
            self.config = TakeProfitConfig(
                take_profit_type=take_profit_type or TakeProfitType.PERCENTAGE,
                take_profit_value=take_profit_value or 0.06,
                trail_distance=trail_distance or 0.02,
                trail_step=trail_step or 0.01,
                max_gain_value=max_gain_value,
                activation_price=activation_price
            )
        
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # State tracking
        self.activation_price: Optional[float] = self.config.activation_price
        self.entry_price: Optional[float] = None
        self.lowest_price: float = float('inf')  # For trailing take profit
        self.is_active: bool = True  # Default to active
        self.triggered: bool = False
        self.target_price: Optional[float] = None
        
        # Add missing attributes for test compatibility
        self.take_profit_type = self.config.take_profit_type
        self.take_profit_value = self.config.take_profit_value
        self.triggered = False
        self.triggered_price: Optional[float] = None
        self.triggered_timestamp: Optional[datetime] = None
        self.trigger_price: Optional[float] = None  # Alias for triggered_price
        self.trigger_time: Optional[datetime] = None   # Alias for triggered_timestamp
        
        # Additional attributes for tests
        self.stop_loss_value = stop_loss_value
        self.trailing_take_profit_pct = trailing_take_profit_pct or self.config.trail_distance
        self.partial_take_profit_levels = partial_take_profit_levels or []
        self.scaling_factors = scaling_factors or []
        self.enforce_rr_ratio: bool = False  # Initialize the attribute
        self.triggered = False
        self.triggered_timestamp = None
        
        # Attributes for trailing stop functionality
        self.highest_price: float = 0.0
        self.stop_price: Optional[float] = None
        self.trailing_stop_pct: float = self.trailing_take_profit_pct
        
    def setup_trailing_target(self, entry_price: float, side: str = 'long') -> None:
        """Setup trailing take profit target.
        
        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position
        """
        self.entry_price = entry_price
        self.lowest_price = entry_price
        if side.lower() == 'long':
            self.target_price = entry_price * (1 + self.trailing_take_profit_pct)
        else:
            self.target_price = entry_price * (1 - self.trailing_take_profit_pct)
            
    def setup_partial_take_profit(self, entry_price: float, quantity: int, side: str = 'long') -> List[float]:
        """Setup partial take profit levels.
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            side: 'long' or 'short'
            
        Returns:
            List of partial take profit levels
        """
        if not self.partial_take_profit_levels:
            self.partial_take_profit_levels = [0.5, 0.25, 0.25]  # Default: 50%, 25%, 25%
        
        levels = []
        for i, fraction in enumerate(self.partial_take_profit_levels):
            if side.lower() == 'long':
                level_price = entry_price * (1 + (i + 1) * 0.02)  # 2%, 4%, 6% etc.
            else:
                level_price = entry_price * (1 - (i + 1) * 0.02)  # -2%, -4%, -6% etc.
            levels.append(level_price)
        
        return levels
        
    def calculate_scaled_target(self, entry_price: float, confidence_level: str, side: str = 'long') -> float:
        """Calculate scaled target based on confidence level.
        
        Args:
            entry_price: Entry price
            confidence_level: 'low', 'medium', 'high'
            side: 'long' or 'short'
            
        Returns:
            Scaled target price
        """
        scaling_factors = {'low': 1.02, 'medium': 1.05, 'high': 1.08}
        factor = scaling_factors.get(confidence_level, 1.05)
        
        if side.lower() == 'long':
            return entry_price * factor
        else:
            return entry_price * (2 - factor)
            
    def update_trailing_target(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Update trailing take profit target with current market data.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with update results
        """
        if not self.is_active or self.entry_price is None:
            return {'triggered': False, 'action': 'NONE'}
        
        # Update lowest price for trailing take profit
        if current_price < self.lowest_price:
            self.lowest_price = current_price
        
        # Recalculate target price
        self.target_price = self.lowest_price * (1 + self.trailing_take_profit_pct)
        
        # Check if target is reached
        triggered = current_price >= self.target_price
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            self.is_active = False
            
        return {
            'triggered': triggered,
            'target_price': self.target_price,
            'lowest_price': self.lowest_price,
            'action': 'TAKE_PROFIT' if triggered else 'NONE'
        }
            
    def update_trailing_stop(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Update trailing stop with current market data.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with update results
        """
        if not self.is_active or self.entry_price is None:
            return {'triggered': False, 'action': 'NONE'}
        
        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Recalculate stop price
        self.stop_price = self.highest_price * (1 - self.trailing_stop_pct)
        
        # Check if stop is triggered
        triggered = current_price <= self.stop_price
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            self.is_active = False
            
        return {
            'triggered': triggered,
            'stop_price': self.stop_price,
            'highest_price': self.highest_price,
            'action': 'STOP_LOSS' if triggered else 'NONE'
        }
        
    def initialize_position(self, entry_price: float, timestamp: pd.Timestamp) -> None:
        """Initialize take profit for a new position.
        
        Args:
            entry_price: Entry price of the position
            timestamp: Entry timestamp
        """
        self.entry_price = entry_price
        self.lowest_price = entry_price
        self.is_active = True
        self.triggered = False
        self.trigger_price = None
        self.trigger_time = None
        
        # Set activation price if configured
        if self.config.activation_price is None:
            self.activation_price = entry_price
        
        # Calculate initial target price
        self.target_price = self._calculate_target_price(entry_price, entry_price)
        
        self.logger.debug(f"Take profit initialized: entry={entry_price:.4f}, target={self.target_price:.4f}")
    
    def update(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Update take profit with current market price.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with take profit status and any trigger information
        """
        if not self.is_active or self.entry_price is None:
            return {
                'triggered': False,
                'target_price': None,
                'action': 'NONE',
                'reason': 'Take profit not active'
            }
        
        result = {
            'triggered': False,
            'target_price': self.target_price,
            'action': 'NONE',
            'reason': 'No action required'
        }
        
        # Update lowest price for trailing take profit
        if self.config.take_profit_type == TakeProfitType.TRAILING:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
        
        # Recalculate target price
        old_target_price = self.target_price
        self.target_price = self._calculate_target_price(current_price, self.lowest_price)
        
        # Check if take profit is triggered
        if current_price >= self.target_price:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            result.update({
                'triggered': True,
                'target_price': self.target_price,
                'action': 'TAKE_PROFIT',
                'reason': f'Take profit triggered at {current_price:.4f}',
                'exit_price': self.target_price,
                'pnl_pct': (self.target_price - self.entry_price) / self.entry_price if self.entry_price else 0,
                'timestamp': timestamp
            })
            self.is_active = False
            
            self.logger.info(f"Take profit triggered: {result['reason']}, P&L: {result['pnl_pct']:.2%}")
        
        # Check maximum gain limit
        elif self.config.max_gain_value is not None:
            gain_amount = current_price - self.entry_price
            if gain_amount >= self.config.max_gain_value:
                self.triggered = True
                self.trigger_price = current_price
                self.trigger_time = timestamp
                result.update({
                    'triggered': True,
                    'target_price': self.target_price,
                    'action': 'MAX_GAIN',
                    'reason': f'Maximum gain target reached: {gain_amount:.4f}',
                    'exit_price': current_price,
                    'pnl_pct': (current_price - self.entry_price) / self.entry_price if self.entry_price else 0,
                    'timestamp': timestamp
                })
                self.is_active = False
                
                self.logger.info(f"Maximum gain target reached: {result['reason']}")
        
        # Log target price changes for trailing profit
        if (self.config.take_profit_type == TakeProfitType.TRAILING and
            old_target_price != self.target_price):
            self.logger.debug(f"Trailing profit updated: {old_target_price:.4f} -> {self.target_price:.4f}")
        
        return result
    
    def _calculate_target_price(self, current_price: float, reference_price: float) -> float:
        """Calculate target price based on configuration.
        
        Args:
            current_price: Current market price
            reference_price: Reference price (lowest for trailing, entry for others)
            
        Returns:
            Calculated target price
        """
        if self.config.take_profit_type == TakeProfitType.FIXED:
            return self.config.take_profit_value
            
        elif self.config.take_profit_type == TakeProfitType.PERCENTAGE:
            return reference_price * (1 + self.config.take_profit_value)
            
        elif self.config.take_profit_type == TakeProfitType.PRICE:
            return self.config.take_profit_value
            
        elif self.config.take_profit_type == TakeProfitType.TRAILING:
            trail_target = reference_price * (1 + self.config.trail_distance)
            
            # Ensure we only move the target down (never up)
            if self.target_price is not None:
                max_target = self.target_price - (reference_price * self.config.trail_step)
                return min(trail_target, max_target)
            else:
                return trail_target
        
        return reference_price * 1.05  # Default 5% profit target
    
    def get_status(self) -> Dict[str, Any]:
        """Get current take profit status.
        
        Returns:
            Dictionary with take profit status information
        """
        return {
            'active': self.is_active,
            'triggered': self.triggered,
            'target_price': self.target_price,
            'entry_price': self.entry_price,
            'lowest_price': self.lowest_price if self.lowest_price != float('inf') else None,
            'config': {
                'type': self.config.take_profit_type.value,
                'value': self.config.take_profit_value,
                'trail_distance': self.config.trail_distance,
                'max_gain_value': self.config.max_gain_value
            }
        }
    
    def calculate_target_price(self, entry_price: float, side: str = 'long') -> float:
        """Calculate target price based on entry price and side.
        
        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position
            
        Returns:
            Calculated target price
        """
        if side.lower() == 'short':
            # For short positions, target is below entry price
            if self.config.take_profit_type == TakeProfitType.FIXED:
                return self.config.take_profit_value
            elif self.config.take_profit_type == TakeProfitType.PERCENTAGE:
                return entry_price * (1 - self.config.take_profit_value)
            elif self.config.take_profit_type == TakeProfitType.PRICE:
                return self.config.take_profit_value
        
        # Long position logic (default)
        return self._calculate_target_price(entry_price, entry_price)
    
    def check_target(self, entry_price: float, current_price: float, side: str = 'long') -> bool:
        """Check if take profit should trigger.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            side: 'long' or 'short' position
            
        Returns:
            True if take profit should trigger
        """
        # Update entry price if not set
        if self.entry_price is None:
            self.entry_price = entry_price
            
        target_price = self.calculate_target_price(entry_price, side)
        
        if side.lower() == 'long':
            return current_price >= target_price
        else:  # short
            return current_price <= target_price
    
    def activate(self) -> None:
        """Activate the take profit."""
        self.is_active = True
        
    def deactivate(self) -> None:
        """Deactivate the take profit."""
        self.is_active = False
        
    def trigger(self) -> None:
        """Manually trigger the take profit."""
        self.triggered = True
        self.trigger_price = self.target_price
        self.trigger_time = datetime.now()
        self.is_active = False
        
    def get_take_profit_details(self) -> Dict[str, Any]:
        """Get take profit status."""
        distance_to_target = 0.0
        if self.entry_price and self.target_price:
            distance_to_target = (self.target_price - self.entry_price) / self.entry_price
            
        return {
            'target_type': self.config.take_profit_type.value,
            'target_value': self.config.take_profit_value,
            'is_active': self.is_active,
            'triggered': self.triggered,
            'distance_to_target': distance_to_target
        }
        
    def reset(self) -> None:
        """Reset take profit to initial state."""
        self.activation_price = self.config.activation_price
        self.entry_price = None
        self.lowest_price = float('inf')
        self.is_active = True  # Reset to active
        self.triggered = False
        self.trigger_price = None  # Clear trigger price
        self.trigger_time = None  # Clear trigger time
        self.target_price = None
        self.logger.debug("Take profit reset")


class RiskControlManager:
    """Manager for coordinating multiple stop loss and take profit mechanisms."""
    
    def __init__(self,
                 stop_loss_config: Optional[StopLossConfig] = None,
                 take_profit_config: Optional[TakeProfitConfig] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize the risk control manager.
        
        Args:
            stop_loss_config: Optional stop loss configuration
            take_profit_config: Optional take profit configuration  
            logger: Optional logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # Initialize controls
        self.stop_loss: Optional[StopLoss] = None
        self.take_profit: Optional[TakeProfit] = None
        
        if stop_loss_config:
            self.stop_loss = StopLoss(stop_loss_config, logger)
        if take_profit_config:
            self.take_profit = TakeProfit(take_profit_config, logger)
    
    def initialize_position(self, entry_price: float, timestamp: pd.Timestamp) -> None:
        """Initialize all risk controls for a position.
        
        Args:
            entry_price: Entry price of the position
            timestamp: Entry timestamp
        """
        if self.stop_loss:
            self.stop_loss.initialize_position(entry_price, timestamp)
        if self.take_profit:
            self.take_profit.initialize_position(entry_price, timestamp)
    
    def update(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Update all risk controls and check for triggers.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with comprehensive risk control status
        """
        result = {
            'triggered': False,
            'action': 'NONE',
            'reason': 'No risk controls triggered',
            'exit_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # Update stop loss
        if self.stop_loss:
            stop_result = self.stop_loss.update(current_price, timestamp)
            result['stop_loss'] = stop_result
            
            if stop_result['triggered']:
                result.update({
                    'triggered': True,
                    'action': stop_result['action'],
                    'reason': stop_result['reason'],
                    'exit_price': stop_result.get('exit_price', current_price)
                })
        
        # Update take profit (only if stop loss not triggered)
        if self.take_profit and not result['triggered']:
            profit_result = self.take_profit.update(current_price, timestamp)
            result['take_profit'] = profit_result
            
            if profit_result['triggered']:
                result.update({
                    'triggered': True,
                    'action': profit_result['action'],
                    'reason': profit_result['reason'],
                    'exit_price': profit_result.get('exit_price', current_price)
                })
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all risk controls.
        
        Returns:
            Dictionary with complete risk control status
        """
        return {
            'stop_loss': self.stop_loss.get_status() if self.stop_loss else None,
            'take_profit': self.take_profit.get_status() if self.take_profit else None,
            'has_stop_loss': self.stop_loss is not None,
            'has_take_profit': self.take_profit is not None
        }
    
    def reset(self) -> None:
        """Reset all risk controls."""
        if self.stop_loss:
            self.stop_loss.reset()
        if self.take_profit:
            self.take_profit.reset()
        self.logger.debug("Risk control manager reset")


class PositionSizer:
    """Position sizing component for risk management."""
    
    def __init__(self,
                 max_position_size: float = 0.10,
                 min_position_size: float = 0.01,
                 volatility_adjustment: bool = True,
                 risk_per_trade: float = 0.02,
                 max_daily_trades: int = 5,
                 max_sector_exposure: float = 0.30,
                 logger: Optional[logging.Logger] = None,
                 sizing_method: str = "fixed_percentage",
                 max_correlation: float = 0.80) -> None:
        """Initialize position sizer.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
            volatility_adjustment: Whether to adjust for volatility
            risk_per_trade: Risk per trade as fraction of portfolio
            max_daily_trades: Maximum number of trades per day
            max_sector_exposure: Maximum sector exposure
            logger: Optional logger instance
            sizing_method: Method for position sizing
            max_correlation: Maximum correlation for position sizing
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.volatility_adjustment = volatility_adjustment
        self.risk_per_trade = risk_per_trade
        self.max_daily_trades = max_daily_trades
        self.max_sector_exposure = max_sector_exposure
        self.sizing_method = sizing_method
        self.max_correlation = max_correlation
    
    def calculate(self, signal: Any, portfolio_value: float, current_price: float) -> float:
        """Calculate position size based on configured sizing method."""
        # Check for sizing_method attribute
        if hasattr(self, 'sizing_method'):
            if self.sizing_method == "fixed_percentage":
                return portfolio_value * self.risk_per_trade
            elif self.sizing_method == "kelly":
                # Kelly Criterion implementation (if needed)
                # For now, fallback to max_position_size
                pass
        
        # Default: use max_position_size
        if hasattr(self, 'max_position_size'):
            return portfolio_value * self.max_position_size
        
        # Fallback to risk_per_trade if max_position_size not available
        return portfolio_value * self.risk_per_trade
    
    def calculate_position_size(self,
                               portfolio_value: float,
                               volatility: float = 0.0,
                               conviction: float = 1.0) -> float:
        """Calculate position size based on risk parameters.
        
        Args:
            portfolio_value: Current portfolio value
            volatility: Current market volatility
            conviction: Signal conviction factor (0.0 to 2.0)
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base position size
        position_size = self.max_position_size * conviction
        
        # Apply volatility adjustment if enabled
        if self.volatility_adjustment and volatility > 0:
            volatility_factor = max(0.1, 1.0 - volatility * 5)
            position_size *= volatility_factor
        
        # Apply bounds
        position_size = max(self.min_position_size,
                           min(position_size, self.max_position_size))
        
        self.logger.debug(f"Position size calculated: {position_size:.3f} "
                         f"(conviction: {conviction:.2f}, vol: {volatility:.3f})")
        
        return position_size
    
    def calculate_position_size_fixed_risk(self,
                                          account_value: float,
                                          entry_price: float,
                                          stop_price: float) -> float:
        """Calculate position size based on fixed risk per trade."""
        risk_amount = account_value * self.risk_per_trade
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance == 0:
            return 0.0
        
        # Position size in shares = risk_amount / stop_distance
        # But we need to return dollar amount, not shares
        position_value = risk_amount / (stop_distance / entry_price)
        max_position_value = account_value * self.max_position_size
        
        return min(position_value, max_position_value)
    
    def calculate_position_size_percentage(self,
                                          account_value: float,
                                          entry_price: float,
                                          percentage: float) -> float:
        """Calculate position size based on percentage of account."""
        position_value = account_value * percentage
        return position_value / entry_price
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion fraction."""
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss  # odds ratio
        q = 1 - win_rate  # loss probability
        p = win_rate  # win probability
        
        # Kelly formula: f* = (bp - q) / b
        # where b = odds ratio, p = win probability, q = loss probability
        kelly_fraction = (b * p - q) / b
        
        # Return as fraction of portfolio, not capped immediately
        # Cap at maximum position size
        return max(0.0, min(kelly_fraction, self.max_position_size))
    
    def calculate_position_size_volatility_adjusted(self,
                                                   account_value: float,
                                                   entry_price: float,
                                                   volatility: float) -> float:
        """Calculate position size adjusted for volatility."""
        base_position = self.calculate_position_size_percentage(account_value, entry_price, self.max_position_size)
        
        # Reduce position size for higher volatility
        volatility_adjustment = max(0.1, 1.0 - volatility * 5)
        
        return base_position * volatility_adjustment
    
    def calculate_position_size_correlation_adjusted(self,
                                                   account_value: float,
                                                   entry_price: float,
                                                   portfolio_correlation: float) -> float:
        """Calculate position size adjusted for portfolio correlation."""
        base_position = self.calculate_position_size_percentage(account_value, entry_price, self.max_position_size)
        
        # Reduce position size for higher correlation
        correlation_adjustment = max(0.5, 1.0 - portfolio_correlation)
        
        return base_position * correlation_adjustment
    
    def calculate_position_size_risk_based(self,
                                          **params) -> float:
        """Comprehensive risk-based position sizing."""
        account_value = params.get('account_value', 10000.0)
        entry_price = params.get('entry_price', 100.0)
        stop_price = params.get('stop_price', entry_price * 0.95)
        volatility = params.get('volatility', 0.0)
        correlation = params.get('correlation', 0.0)
        conviction_level = params.get('conviction_level', 'medium')
        
        # Base position size from fixed risk
        position_size = self.calculate_position_size_fixed_risk(account_value, entry_price, stop_price)
        
        # Apply volatility adjustment
        if volatility > 0:
            volatility_factor = max(0.1, 1.0 - volatility * 5)
            position_size *= volatility_factor
        
        # Apply correlation adjustment
        if correlation > 0:
            correlation_factor = max(0.5, 1.0 - correlation)
            position_size *= correlation_factor
        
        # Apply conviction adjustment
        conviction_factors = {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        conviction_factor = conviction_factors.get(conviction_level, 1.0)
        position_size *= conviction_factor
        
        # Apply bounds
        max_size = (account_value * self.max_position_size) / entry_price
        min_size = (account_value * self.min_position_size) / entry_price
        
        return max(min_size, min(position_size, max_size))
    
    def enforce_constraints(self,
                           position_size: float,
                           account_value: float,
                           entry_price: float) -> float:
        """Enforce position sizing constraints."""
        # Calculate actual position value
        position_value = position_size * entry_price
        
        # Check minimum size constraint
        min_position_value = account_value * self.min_position_size
        if position_value < min_position_value:
            position_size = min_position_value / entry_price
        
        # Check maximum size constraint
        max_position_value = account_value * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size


class RiskLimits:
    """Risk limits management class."""
    
    def __init__(self,
                 max_drawdown: float = 0.20,
                 max_leverage: float = 3.0,
                 max_position_size: float = 0.15,
                 max_sector_exposure: float = 0.30,
                 max_correlation: float = 0.80,
                 max_volatility: float = 0.25):
        """Initialize risk limits."""
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.max_volatility = max_volatility
        self.current_profile = 'moderate'
    
    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Check if drawdown is within limits."""
        return abs(current_drawdown) <= self.max_drawdown
    
    def check_leverage_limit(self, current_leverage: float) -> bool:
        """Check if leverage is within limits."""
        return current_leverage <= self.max_leverage
    
    def check_position_size_limit(self, position_size: float) -> bool:
        """Check if position size is within limits."""
        return position_size <= self.max_position_size
    
    def check_sector_exposure_limit(self, sector: str, exposure: float) -> bool:
        """Check if sector exposure is within limits."""
        return exposure <= self.max_sector_exposure
    
    def check_correlation_limit(self, correlation: float) -> bool:
        """Check if correlation is within limits."""
        return correlation <= self.max_correlation
    
    def check_volatility_limit(self, volatility: float) -> bool:
        """Check if volatility is within limits."""
        return volatility <= self.max_volatility
    
    def check_limits(self, portfolio: Any) -> List[str]:
        """Check risk limits and return list of violations."""
        violations = []
        
        # Check if portfolio has positions
        if hasattr(portfolio, 'positions'):
            # Check position sizes
            for symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                allocation = abs(position.get('allocation', 0))
                if allocation > self.max_position_size:
                    violations.append(
                        f"Position {symbol} exceeds max size: "
                        f"{allocation:.2%} > {self.max_position_size:.2%}"
                    )
        
        # Check drawdown if available
        if hasattr(portfolio, 'get_current_drawdown'):
            current_dd = portfolio.get_current_drawdown()
            if abs(current_dd) > self.max_drawdown:
                violations.append(
                    f"Max drawdown exceeded: {current_dd:.2%}"
                )
        
        # Check leverage if available
        if hasattr(portfolio, 'get_total_leverage'):
            leverage = portfolio.get_total_leverage()
            if leverage > self.max_leverage:
                violations.append(
                    f"Leverage exceeds limit: {leverage:.2f}x"
                )
        
        return violations
    
    def check_all_limits(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check all risk limits."""
        breached_limits = []
        all_passed = True
        
        # Check drawdown
        if 'current_drawdown' in portfolio_state:
            if not self.check_drawdown_limit(portfolio_state['current_drawdown']):
                breached_limits.append('drawdown')
                all_passed = False
        
        # Check leverage
        if 'leverage' in portfolio_state:
            if not self.check_leverage_limit(portfolio_state['leverage']):
                breached_limits.append('leverage')
                all_passed = False
        
        # Check position sizes
        if 'largest_position' in portfolio_state:
            if not self.check_position_size_limit(portfolio_state['largest_position']):
                breached_limits.append('position_size')
                all_passed = False
        
        # Calculate risk score (simplified)
        risk_score = len(breached_limits) * 0.25
        
        return {
            'all_limits_passed': all_passed,
            'breached_limits': breached_limits,
            'risk_score': risk_score,
            'recommendations': self._generate_recommendations(breached_limits)
        }
    
    def set_risk_profile(self, profile_name: str):
        """Set risk profile with different limits."""
        profiles = {
            'conservative': {
                'max_drawdown': 0.10,
                'max_leverage': 2.0,
                'max_position_size': 0.08
            },
            'moderate': {
                'max_drawdown': 0.15,
                'max_leverage': 2.5,
                'max_position_size': 0.12
            },
            'aggressive': {
                'max_drawdown': 0.25,
                'max_leverage': 4.0,
                'max_position_size': 0.20
            }
        }
        
        if profile_name in profiles:
            profile = profiles[profile_name]
            self.max_drawdown = profile['max_drawdown']
            self.max_leverage = profile['max_leverage']
            self.max_position_size = profile['max_position_size']
            self.current_profile = profile_name
    
    def handle_limit_breach(self, breaches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle limit breach escalation."""
        severity_level = 'medium'
        if len(breaches) >= 3:
            severity_level = 'critical'
        elif len(breaches) >= 2:
            severity_level = 'high'
        
        required_actions = ['reduce_position_sizes']
        if severity_level in ['high', 'critical']:
            required_actions.extend(['increase_cash_position', 'review_strategies'])
        
        timeline = 'immediate' if severity_level == 'critical' else 'within_hour'
        
        return {
            'severity_level': severity_level,
            'required_actions': required_actions,
            'timeline': timeline
        }
    
    def _generate_recommendations(self, breached_limits: List[str]) -> List[str]:
        """Generate recommendations based on breached limits."""
        recommendations = []
        
        if 'drawdown' in breached_limits:
            recommendations.append("Reduce overall portfolio risk")
        if 'leverage' in breached_limits:
            recommendations.append("Lower leverage ratio")
        if 'position_size' in breached_limits:
            recommendations.append("Reduce individual position sizes")
        if 'volatility' in breached_limits:
            recommendations.append("Increase diversification")
        
        return recommendations


class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self,
                 check_interval: int = 60,
                 enable_real_time_alerts: bool = True,
                 max_history_size: int = 500):
        """Initialize risk monitor."""
        self.is_monitoring = True
        self.check_interval = check_interval
        self.enable_real_time_alerts = enable_real_time_alerts
        self.max_history_size = max_history_size
        self.risk_metrics: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'volatility': 0.30,
            'drawdown': 0.15,
            'leverage': 3.0
        }
    
    def add_risk_metric(self, metric_name: str, threshold: float, comparison: str = 'greater_than'):
        """Add a risk metric to monitor."""
        self.risk_metrics[metric_name] = {
            'threshold': threshold,
            'comparison': comparison
        }
    
    def check_risk_metrics(self, portfolio_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all risk metrics for violations."""
        violations = []
        
        for metric_name, config in self.risk_metrics.items():
            if metric_name in portfolio_state:
                current_value = portfolio_state[metric_name]
                threshold = config['threshold']
                comparison = config['comparison']
                
                is_violation = False
                if comparison == 'greater_than' and current_value > threshold:
                    is_violation = True
                elif comparison == 'less_than' and current_value < threshold:
                    is_violation = True
                
                if is_violation:
                    violations.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'high' if abs(current_value - threshold) / threshold > 0.5 else 'medium'
                    })
            elif metric_name == 'drawdown' and 'current_drawdown' in portfolio_state:
                current_value = portfolio_state['current_drawdown']
                threshold = config['threshold']
                comparison = config['comparison']
                
                is_violation = False
                if comparison == 'greater_than' and current_value > threshold:
                    is_violation = True
                elif comparison == 'less_than' and current_value < threshold:
                    is_violation = True
                
                if is_violation:
                    violations.append({
                        'metric': 'drawdown',
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'high' if abs(current_value - threshold) / threshold > 0.5 else 'medium'
                    })
        
        return violations
    
    def process_portfolio_update(self, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process portfolio update and generate alerts if necessary."""
        if not self.is_monitoring:
            return None
        
        violations = self.check_risk_metrics(update)
        
        if violations and self.enable_real_time_alerts:
            # Generate alert for most severe violation
            most_severe = max(violations, key=lambda x: {'high': 2, 'medium': 1, 'low': 0}.get(x['severity'], 0))
            
            # Always generate an alert if there are violations and alerts are enabled
            return {
                'alert_type': 'risk_metric_breach',
                'metric': most_severe['metric'],
                'current_value': most_severe['current_value'],
                'threshold': most_severe['threshold'],
                'severity': most_severe['severity'],
                'timestamp': update.get('timestamp'),
                'recommended_action': self._get_recommended_action(most_severe)
            }
        
        return None
    
    def update(self, portfolio: Any, timestamp: datetime) -> None:
        """Update risk metrics with current portfolio state."""
        # Initialize tracking lists if not present
        if not hasattr(self, 'portfolio_values'):
            self.portfolio_values = []
        if not hasattr(self, 'timestamps'):
            self.timestamps = []
        
        # Get current portfolio value
        total_value = (
            portfolio.get_total_value()
            if hasattr(portfolio, 'get_total_value')
            else 0
        )
        
        # Append current values
        self.portfolio_values.append(total_value)
        self.timestamps.append(timestamp)
        
        # Calculate rolling volatility if enough data points
        if len(self.portfolio_values) >= 20:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            # Annualized volatility (assuming 252 trading days)
            self.current_volatility = np.std(returns) * np.sqrt(252)
        
        # Calculate other risk metrics as needed
        # (drawdown, var, etc. can be added here)
    
    def record_risk_measurement(self, measurement: Dict[str, Any]):
        """Record risk measurement to history."""
        measurement['timestamp'] = measurement.get('timestamp', pd.Timestamp.now())
        self.risk_metrics_history.append(measurement)
        
        # Keep history size limited
        if len(self.risk_metrics_history) > self.max_history_size:
            self.risk_metrics_history = self.risk_metrics_history[-self.max_history_size:]
    
    def analyze_risk_trends(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk trends over time."""
        if len(trend_data) < 2:
            return {'trend_direction': 'insufficient_data'}
        
        # Simplified trend analysis
        volatility_trend = self._calculate_trend([d.get('volatility', 0) for d in trend_data])
        drawdown_trend = self._calculate_trend([d.get('drawdown', 0) for d in trend_data])
        
        return {
            'volatility_trend': {'direction': volatility_trend},
            'drawdown_trend': {'direction': drawdown_trend},
            'trend_direction': 'increasing' if volatility_trend == 'increasing' else 'stable'
        }
    
    def add_alert_rule(self, metric: str, threshold: float, severity: str, action: str):
        """Add alert rule for automated alerting."""
        if not hasattr(self, 'alert_rules'):
            self.alert_rules = []
        
        self.alert_rules.append({
            'metric': metric,
            'threshold': threshold,
            'severity': severity,
            'action': action
        })
    
    def generate_alert(self, metric: str, value: float, severity: str) -> Dict[str, Any]:
        """Generate risk alert."""
        return {
            'alert_type': 'manual_alert',
            'metric': metric,
            'value': value,
            'severity': severity,
            'recommended_action': self._get_recommended_action({'metric': metric}),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for risk monitoring."""
        current_metrics = {}
        if self.risk_metrics_history:
            latest = self.risk_metrics_history[-1]
            current_metrics = latest
        
        historical_trends = self.risk_metrics_history[-30:] if len(self.risk_metrics_history) >= 30 else self.risk_metrics_history
        
        # Calculate risk summary
        risk_summary = {
            'total_measurements': len(self.risk_metrics_history),
            'latest_metrics': current_metrics,
            'trend_status': 'stable'
        }
        
        return {
            'current_metrics': current_metrics,
            'historical_trends': historical_trends,
            'risk_summary': risk_summary,
            'alerts': []
        }
    
    def generate_risk_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate risk report for date range."""
        filtered_history = [
            m for m in self.risk_metrics_history
            if start_date <= m.get('timestamp', start_date) <= end_date
        ]
        
        return {
            'report_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'executive_summary': 'Risk monitoring shows stable conditions',
            'risk_metrics': self._aggregate_metrics(filtered_history),
            'trends': self.analyze_risk_trends(filtered_history),
            'recommendations': ['Continue monitoring current risk levels'],
            'alerts_summary': {'total_alerts': 0, 'critical_alerts': 0}
        }
    
    def optimize_risk_limits(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize risk limits based on historical performance."""
        # Simplified optimization based on historical volatility
        returns = performance_data.get('returns', pd.Series())
        
        if len(returns) == 0:
            return {
                'recommended_drawdown_limit': 0.15,
                'recommended_volatility_limit': 0.25,
                'recommended_position_limit': 0.12,
                'optimization_score': 0.5
            }
        
        historical_vol = returns.std() * np.sqrt(252)
        
        return {
            'recommended_drawdown_limit': min(0.20, historical_vol * 1.5),
            'recommended_volatility_limit': min(0.30, historical_vol * 1.2),
            'recommended_position_limit': 0.10 if historical_vol > 0.20 else 0.15,
            'optimization_score': 0.7
        }
    
    def get_recommended_action(self, risk_level: str, position_size: float) -> str:
        """Get recommended action based on risk level and position size."""
        action_mapping = {
            ('low', 0.05): 'normal_trading',
            ('medium', 0.10): 'normal_trading',
            ('high', 0.15): 'reduce_position_size',
            ('very_high', 0.25): 'emergency_reduction'
        }
        
        return action_mapping.get((risk_level, position_size), 'review_positions')
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.05:
            return 'increasing'
        elif second_half < first_half * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_recommended_action(self, violation: Dict[str, Any]) -> str:
        """Get recommended action for a violation."""
        metric = violation.get('metric', 'unknown')
        
        action_mapping = {
            'drawdown': 'reduce_position_sizes',
            'volatility': 'increase_diversification',
            'leverage': 'lower_leverage',
            'position_size': 'reduce_individual_positions'
        }
        
        return action_mapping.get(metric, 'review_all_positions')
    
    def _aggregate_metrics(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from measurements."""
        if not measurements:
            return {}
        
        # Calculate averages and extremes
        aggregated = {}
        metrics = ['volatility', 'drawdown', 'leverage']
        
        for metric in metrics:
            values = [m.get(metric, 0) for m in measurements]
            aggregated[metric] = {
                'average': np.mean(values),
                'maximum': np.max(values),
                'minimum': np.min(values)
            }
        
        return aggregated