"""
Dual-Pool Portfolio Management System.

This module implements a sophisticated portfolio management system with dual pools
(base and alpha) with different leverage levels, based on the existing run_sim.py logic.
"""

from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass


@dataclass
class PoolState:
    """State information for a single pool."""
    # Constructor arguments expected by tests: (pool_type, leverage, max_allocation)
    pool_type: str = 'base'
    leverage: float = 1.0
    max_allocation: float = 1.0
    # Original attributes with defaults
    capital: float = 0.0
    active: bool = False
    entry_price: float = 0.0
    position_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    available_capital: float = 0.0  # Add missing attribute
    used_capital: float = 0.0  # Add missing attribute
    positions: Optional[Dict[str, Any]] = None  # Add missing attribute
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
    
    def allocate_capital(self, amount: float) -> None:
        """Allocate capital to this pool."""
        self.available_capital += amount
    
    def use_capital(self, amount: float) -> None:
        """Use capital from this pool."""
        if self.available_capital >= amount:
            self.available_capital -= amount
            self.used_capital += amount
    
    def get_current_leverage(self) -> float:
        """Get current leverage ratio."""
        if self.used_capital == 0:
            return 1.0
        return (self.used_capital + self.available_capital) / self.used_capital
    
    def get_max_leverage(self) -> float:
        """Get maximum leverage for this pool."""
        return self.leverage
    
    def check_health(self) -> Dict[str, Any]:
        """Check pool health metrics."""
        current_leverage = self.get_current_leverage()
        utilization_rate = self.used_capital / (self.used_capital + self.available_capital) if (self.used_capital + self.available_capital) > 0 else 0
        
        health_status = "healthy"
        if current_leverage > self.leverage * 0.9:
            health_status = "high_risk"
        elif current_leverage > self.leverage * 0.7:
            health_status = "moderate"
        
        return {
            'leverage_ratio': current_leverage,
            'utilization_rate': utilization_rate,
            'health_status': health_status
        }

@dataclass
class Position:
    """State information for a single position in GeneralPortfolio."""
    symbol: str
    quantity: float
    avg_price: float  # Changed from entry_price to avg_price for test compatibility
    timestamp: Any = None  # Accept timestamp parameter
    current_price: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    total_cost: Optional[float] = None  # Will be auto-calculated in __post_init__
    total_commission: float = 0.0
    entry_timestamp: Any = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-calculate total_cost if not provided
        if self.total_cost is None:
            self.total_cost = self.quantity * self.avg_price
        # Handle timestamp -> entry_timestamp mapping
        if self.entry_timestamp is None and self.timestamp is not None:
            self.entry_timestamp = self.timestamp


    def update_quantity(self, additional_quantity: float, price: float) -> None:
        """Update position quantity and recalculate average price."""
        total_quantity = self.quantity + additional_quantity
        total_cost = (self.quantity * self.avg_price) + (additional_quantity * price)
        new_avg_price = total_cost / total_quantity
        
        self.quantity = total_quantity
        self.avg_price = new_avg_price
    
    def get_current_value(self, current_price: float) -> float:
        """Get current market value of position."""
        return self.quantity * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L."""
        return (current_price - self.avg_price) * self.quantity
    
    def close_position(self, exit_price: float, quantity: float, timestamp: Any) -> float:
        """Close position and return realized P&L."""
        if quantity > self.quantity:
            quantity = self.quantity
        
        realized_pnl = (exit_price - self.avg_price) * quantity
        self.quantity -= quantity
        self.realized_pnl += realized_pnl
        
        return realized_pnl
    
    def get_weight(self, portfolio_value: float, current_price: Optional[float] = None) -> float:
        """Get position weight in portfolio."""
        if portfolio_value <= 0:
            return 0.0
        price = current_price if current_price is not None else self.current_price
        if price <= 0:
            # Use avg_price as fallback for weight calculation
            price = self.avg_price if self.avg_price > 0 else 0
        if price <= 0:
            return 0.0
        return (self.quantity * price) / portfolio_value


class DualPoolPortfolio:
    """Portfolio with dual-pool leverage strategy."""
    
    def __init__(self,
                 initial_capital: float = 100.0,
                 leverage_base: float = 1.0,
                 leverage_alpha: float = 3.0,
                 base_to_alpha_split: float = 0.2,
                 alpha_to_base_split: float = 0.2,
                 stop_loss_base: float = 0.025,
                 stop_loss_alpha: float = 0.025,
                 take_profit_target: float = 0.10,
                 maintenance_margin: float = 0.5,
                 commission_rate: float = 0.001,
                 interest_rate_daily: float = 0.00025,
                 spread_rate: float = 0.0002,
                 slippage_std: float = 0.0005,
                 funding_enabled: bool = True,
                 max_total_leverage: float = 4.0,
                 cash: float = 0.0,
                 tax_rate: float = 0.45,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize the dual-pool portfolio.
        
        Args:
            initial_capital: Starting capital for the portfolio
            leverage_base: Leverage factor for base pool
            leverage_alpha: Leverage factor for alpha pool
            base_to_alpha_split: Fraction of positive base gains to transfer to alpha
            alpha_to_base_split: Fraction of positive alpha gains to transfer to base
            stop_loss_base: Stop loss threshold for base pool (as decimal)
            stop_loss_alpha: Stop loss threshold for alpha pool (as decimal)
            take_profit_target: Take profit threshold (as decimal)
            maintenance_margin: Margin requirement for leveraged positions
            commission_rate: Commission rate for trades
            interest_rate_daily: Daily interest rate for borrowed funds
            spread_rate: Spread cost per trade
            slippage_std: Standard deviation for slippage simulation
            funding_enabled: Whether to charge interest on borrowed funds
            tax_rate: Tax rate on capital gains
            logger: Optional logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # Portfolio parameters
        self.initial_capital: float = initial_capital
        self.leverage_base: float = leverage_base
        self.leverage_alpha: float = leverage_alpha
        self.base_to_alpha_split: float = base_to_alpha_split
        self.alpha_to_base_split: float = alpha_to_base_split
        self.stop_loss_base: float = stop_loss_base
        self.stop_loss_alpha: float = stop_loss_alpha
        self.take_profit_target: float = take_profit_target
        self.maintenance_margin: float = maintenance_margin
        self.commission_rate: float = commission_rate
        self.interest_rate_daily: float = interest_rate_daily
        self.spread_rate: float = spread_rate
        self.slippage_std: float = slippage_std
        self.funding_enabled: bool = funding_enabled
        self.max_total_leverage: float = max_total_leverage
        self.cash: float = cash
        self.tax_rate: float = tax_rate
        
        # Initialize pool states
        self.base_pool: PoolState = PoolState(
            pool_type='base',
            leverage=leverage_base,
            max_allocation=1.0 - base_to_alpha_split,
            capital=initial_capital * 0.5,
            active=False,
            entry_price=0.0,
            available_capital=initial_capital * 0.5
        )
        
        self.alpha_pool: PoolState = PoolState(
            pool_type='alpha',
            leverage=leverage_alpha,
            max_allocation=base_to_alpha_split,
            capital=initial_capital * 0.5,
            active=False,
            entry_price=0.0,
            available_capital=initial_capital * 0.5
        )
        
        # Portfolio tracking
        self.portfolio_values: List[float] = [initial_capital]
        self.base_values: List[float] = [self.base_pool.capital]
        self.alpha_values: List[float] = [self.alpha_pool.capital]
        self.trade_log: List[Dict[str, Any]] = []
        self.cumulative_tax: float = 0.0
        self.tax_loss_carryforward: float = 0.0
        self.yearly_gains: Dict[str, float] = {"base": 0.0, "alpha": 0.0}
        self.current_year: Optional[int] = None
        
        self.logger.info(f"Initialized portfolio with ${initial_capital:.2f} capital")
    
    def process_tick(self, 
                     timestamp: Any,
                     current_price: float,
                     day_high: float,
                     day_low: float) -> Dict[str, Any]:
        """Process a market tick and update portfolio state.
        
        Args:
            timestamp: Current timestamp
            current_price: Current market price
            day_high: High price for the day
            day_low: Low price for the day
            
        Returns:
            Dictionary with updated portfolio information
        """
        # Initialize entry prices if first time
        if not self.base_pool.active and self.base_pool.capital > 0:
            self.base_pool.active = True
            self.base_pool.entry_price = current_price
            
        if not self.alpha_pool.active and self.alpha_pool.capital > 0:
            self.alpha_pool.active = True
            self.alpha_pool.entry_price = current_price
        
        # Calculate returns and apply risk management
        base_pnl, base_exit_reason = self._process_pool_pnl(
            self.base_pool, current_price, day_high, day_low
        )
        
        alpha_pnl, alpha_exit_reason = self._process_pool_pnl(
            self.alpha_pool, current_price, day_high, day_low
        )
        
        # Calculate fees and costs
        base_fee = self._calculate_pool_fee(self.base_pool)
        alpha_fee = self._calculate_pool_fee(self.alpha_pool)
        
        # Redistribute gains between pools
        alpha_to_base, base_to_alpha = self._redistribute_gains(
            base_pnl, alpha_pnl, current_price
        )
        
        # Update pool capitals
        old_base_capital = self.base_pool.capital
        old_alpha_capital = self.alpha_pool.capital
        
        self.base_pool.capital += base_pnl + alpha_to_base - base_to_alpha - base_fee
        self.alpha_pool.capital += alpha_pnl + base_to_alpha - alpha_to_base - alpha_fee
        
        # Handle bankruptcy
        self._handle_bankruptcy()
        
        # Apply tax calculations at year end
        self._handle_tax_calculation(timestamp)
        
        # Log trade if exit occurred
        if base_exit_reason or alpha_exit_reason:
            trade_info = {
                'timestamp': timestamp,
                'price': current_price,
                'day_high': day_high,
                'day_low': day_low,
                'base_pool': self.base_pool.capital,
                'alpha_pool': self.alpha_pool.capital,
                'base_exit': base_exit_reason,
                'alpha_exit': alpha_exit_reason,
                'base_capital_change': self.base_pool.capital - old_base_capital,
                'alpha_capital_change': self.alpha_pool.capital - old_alpha_capital
            }
            self.trade_log.append(trade_info)
            self.logger.debug(f"Trade logged: {trade_info}")
        
        # Record portfolio values
        total_value = self.base_pool.capital + self.alpha_pool.capital
        self.portfolio_values.append(total_value)
        self.base_values.append(self.base_pool.capital)
        self.alpha_values.append(self.alpha_pool.capital)
        
        return {
            'timestamp': timestamp,
            'total_value': total_value,
            'base_pool': self.base_pool.capital,
            'alpha_pool': self.alpha_pool.capital,
            'base_active': self.base_pool.active,
            'alpha_active': self.alpha_pool.active,
            'base_exit': base_exit_reason,
            'alpha_exit': alpha_exit_reason,
            'cumulative_tax': self.cumulative_tax
        }
    
    def _process_pool_pnl(self, pool: PoolState, current_price: float, 
                         day_high: float, day_low: float) -> Tuple[float, Optional[str]]:
        """Process P&L calculation for a single pool with risk management.
        
        Args:
            pool: Pool state to process
            current_price: Current market price
            day_high: High price for the day
            day_low: Low price for the day
            
        Returns:
            Tuple of (pnl, exit_reason)
        """
        if not pool.active or pool.capital <= 0:
            return 0.0, None
        
        pnl = 0.0
        exit_reason = None
        
        # Calculate position value
        position_value = pool.capital * pool.leverage
        
        # Check stop loss using intraday low
        stop_price = pool.entry_price * (1 - (self.stop_loss_alpha if pool == self.alpha_pool else self.stop_loss_base))
        if day_low <= stop_price:
            pnl = (stop_price - pool.entry_price) / pool.entry_price * position_value
            pnl -= self.spread_rate * position_value
            pool.active = False
            exit_reason = "STOP_LOSS"
            
        # Check take profit using intraday high
        elif day_high >= pool.entry_price * (1 + self.take_profit_target):
            tp_price = pool.entry_price * (1 + self.take_profit_target)
            pnl = (tp_price - pool.entry_price) / pool.entry_price * position_value
            pnl -= self.spread_rate * position_value
            pool.active = False
            exit_reason = "TAKE_PROFIT"
            
        # Normal P&L calculation
        else:
            price_return = (current_price - pool.entry_price) / pool.entry_price
            pnl = price_return * position_value
            
            # Apply spread and slippage
            spread_cost = self.spread_rate
            slippage = np.random.normal(0, self.slippage_std)
            pnl -= spread_cost * position_value + slippage * position_value
            
            # Check maintenance margin
            if pool.leverage > 1:
                equity_ratio = (pool.capital + pnl) / position_value
                if equity_ratio < self.maintenance_margin:
                    pnl = -pool.capital  # Total loss
                    pool.active = False
                    exit_reason = "LIQUIDATION"
        
        return pnl, exit_reason
    
    def _calculate_pool_fee(self, pool: PoolState) -> float:
        """Calculate fees for a pool.
        
        Args:
            pool: Pool to calculate fees for
            
        Returns:
            Total fee amount
        """
        if not pool.active or pool.capital <= 0:
            return 0.0
        
        fee = 0.0
        
        # Commission
        position_value = pool.capital * pool.leverage
        commission = self.commission_rate * position_value
        fee += commission
        
        # Interest on borrowed funds
        if self.funding_enabled and pool.leverage > 1:
            borrowed_amount = pool.capital * (pool.leverage - 1)
            interest = self.interest_rate_daily * borrowed_amount
            fee += interest
        
        return fee
    
    def _redistribute_gains(self, base_pnl: float, alpha_pnl: float, 
                          current_price: float) -> Tuple[float, float]:
        """Redistribute gains between pools.
        
        Args:
            base_pnl: Base pool P&L
            alpha_pnl: Alpha pool P&L  
            current_price: Current market price
            
        Returns:
            Tuple of (alpha_to_base, base_to_alpha)
        """
        alpha_to_base = 0.0
        base_to_alpha = 0.0
        
        # Only redistribute positive gains when both pools are active
        if self.base_pool.active and base_pnl > 0:
            alpha_to_base = self.alpha_to_base_split * base_pnl
            
        if self.alpha_pool.active and alpha_pnl > 0:
            base_to_alpha = self.base_to_alpha_split * alpha_pnl
        
        return alpha_to_base, base_to_alpha
    
    def _handle_bankruptcy(self):
        """Handle bankruptcy scenarios for both pools."""
        if self.base_pool.capital < 0:
            self.base_pool.capital = 0
            self.base_pool.active = False
            
        if self.alpha_pool.capital < 0:
            self.alpha_pool.capital = 0
            self.alpha_pool.active = False
    
    def _handle_tax_calculation(self, timestamp: Any):
        """Handle tax calculations at year end.
        
        Args:
            timestamp: Current timestamp
        """
        if self.current_year is None:
            self.current_year = timestamp.year
            return
        
        current_year = timestamp.year
        
        # If new year, process previous year's taxes
        if current_year != self.current_year:
            yearly_total_gain = self.yearly_gains["base"] + self.yearly_gains["alpha"]
            taxable_gain = yearly_total_gain + self.tax_loss_carryforward
            
            if taxable_gain > 0:
                tax = taxable_gain * self.tax_rate
                self.cumulative_tax += tax
                
                # Deduct tax proportionally from pools
                base_positive = max(self.yearly_gains["base"], 0)
                alpha_positive = max(self.yearly_gains["alpha"], 0)
                total_positive = base_positive + alpha_positive
                
                if total_positive > 0:
                    base_tax_share = (base_positive / total_positive) * tax
                    alpha_tax_share = (alpha_positive / total_positive) * tax
                    self.base_pool.capital -= base_tax_share
                    self.alpha_pool.capital -= alpha_tax_share
                
                self.tax_loss_carryforward = 0
            else:
                # Carry forward losses
                self.tax_loss_carryforward = taxable_gain
            
            # Reset for new year
            self.yearly_gains = {"base": 0.0, "alpha": 0.0}
            self.current_year = current_year
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)
        
        # Basic metrics
        total_return = ((portfolio_values[-1] / self.initial_capital) - 1) * 100
        
        # Calculate returns for risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Risk metrics
        sharpe_ratio = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trade_log 
                             if t.get('base_capital_change', 0) > 0 or t.get('alpha_capital_change', 0) > 0])
        total_trades = len(self.trade_log)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'cumulative_tax': self.cumulative_tax,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'base_pool_final': self.base_pool.capital,
            'alpha_pool_final': self.alpha_pool.capital,
            'portfolio_values': portfolio_values,
            'base_values': np.array(self.base_values),
            'alpha_values': np.array(self.alpha_values),
            'trade_log': self.trade_log
        }
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.base_pool = PoolState(
            pool_type='base',
            leverage=self.leverage_base,
            max_allocation=1.0,
            capital=self.initial_capital * 0.5,
            active=False,
            entry_price=0.0
        )
        
        self.alpha_pool = PoolState(
            pool_type='alpha',
            leverage=self.leverage_alpha,
            max_allocation=1.0,
            capital=self.initial_capital * 0.5,
            active=False,
            entry_price=0.0
        )
        
        self.portfolio_values = [self.initial_capital]
        self.base_values = [self.initial_capital * 0.5]
        self.alpha_values = [self.initial_capital * 0.5]
        self.trade_log.clear()
        self.cumulative_tax = 0.0
        self.tax_loss_carryforward = 0.0
        self.yearly_gains = {"base": 0.0, "alpha": 0.0}
        self.current_year = None
        
        self.logger.info("Portfolio reset to initial state")
    
    def allocate_to_pool(self, pool_type: str, amount: float) -> None:
        """Allocate capital to a specific pool.
        
        Args:
            pool_type: Type of pool ('base' or 'alpha')
            amount: Amount of capital to allocate
        """
        if pool_type == 'base':
            self.base_pool.available_capital = amount
            self.base_pool.used_capital = 0  # Start with no used capital
        elif pool_type == 'alpha':
            self.alpha_pool.available_capital = amount
            self.alpha_pool.used_capital = 0  # Start with no used capital
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")
    
    def add_base_position(self, symbol: str, quantity: float, price: float, timestamp: Any) -> bool:
        """Add a position to the base pool.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            
        Returns:
            True if position was added successfully
        """
        # Calculate margin required for base pool
        total_position_value = quantity * price
        margin_required = total_position_value / self.base_pool.leverage
        
        if self.base_pool.available_capital >= margin_required:
            self.base_pool.available_capital -= margin_required
            self.base_pool.used_capital += margin_required
            
            # Add to pool positions (simplified)
            if self.base_pool.positions is None:
                self.base_pool.positions = {}
            if symbol not in self.base_pool.positions:
                self.base_pool.positions[symbol] = Position(symbol, quantity, price, timestamp)
            else:
                # Update existing position
                existing = self.base_pool.positions[symbol]
                existing.update_quantity(quantity, price)
                
            self.logger.debug(f"Added {symbol} position to base pool: {quantity}@{price:.4f}, margin: {margin_required:.2f}")
            return True
        return False
    
    def add_alpha_position(self, symbol: str, quantity: float, price: float, timestamp: Any) -> bool:
        """Add a position to the alpha pool.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            
        Returns:
            True if position was added successfully
        """
        # Calculate margin required for alpha pool
        total_position_value = quantity * price
        margin_required = total_position_value / self.alpha_pool.leverage
        
        if self.alpha_pool.available_capital >= margin_required:
            self.alpha_pool.available_capital -= margin_required
            self.alpha_pool.used_capital += margin_required
            
            # Add to pool positions (simplified)
            if self.alpha_pool.positions is None:
                self.alpha_pool.positions = {}
            if symbol not in self.alpha_pool.positions:
                self.alpha_pool.positions[symbol] = Position(symbol, quantity, price, timestamp)
            else:
                # Update existing position
                existing = self.alpha_pool.positions[symbol]
                existing.update_quantity(quantity, price)
                
            self.logger.debug(f"Added {symbol} position to alpha pool: {quantity}@{price:.4f}, margin: {margin_required:.2f}")
            return True
        return False
    
    def rebalance_pools(self, base_target: float, alpha_target: float) -> None:
        """Rebalance capital between pools.
        
        Args:
            base_target: Target allocation for base pool (0.0 to 1.0)
            alpha_target: Target allocation for alpha pool (0.0 to 1.0)
        """
        total_capital = self.initial_capital
        target_base = total_capital * base_target
        
        # Adjust base pool available_capital
        base_diff = target_base - self.base_pool.available_capital
        if base_diff > 0:
            # Move from alpha to base
            transfer_amount = min(base_diff, self.alpha_pool.available_capital)
            self.alpha_pool.available_capital -= transfer_amount
            self.base_pool.available_capital += transfer_amount
        elif base_diff < 0:
            # Move from base to alpha
            transfer_amount = min(-base_diff, self.base_pool.available_capital)
            self.base_pool.available_capital -= transfer_amount
            self.alpha_pool.available_capital += transfer_amount
        
        # Set max allocation for tracking
        self.base_pool.max_allocation = base_target
        self.alpha_pool.max_allocation = alpha_target
    
    def get_total_leverage(self) -> float:
        """Get total portfolio leverage."""
        base_value = self.base_pool.capital * self.base_pool.leverage if self.base_pool.active else 0
        alpha_value = self.alpha_pool.capital * self.alpha_pool.leverage if self.alpha_pool.active else 0
        total_value = self.base_pool.capital + self.alpha_pool.capital
        
        if total_value <= 0:
            return 1.0
        
        return (base_value + alpha_value) / total_value
    
    def check_risk_limits(self) -> bool:
        """Check if portfolio is within risk limits.
        
        Returns:
            True if within limits, False otherwise
        """
        total_leverage = self.get_total_leverage()
        return total_leverage <= getattr(self, 'max_total_leverage', 4.0)
    
    def get_pool_performance(self, pool_type: str) -> Dict[str, Any]:
        """Get performance metrics for a specific pool.
        
        Args:
            pool_type: Type of pool ('base' or 'alpha')
            
        Returns:
            Dictionary with performance metrics
        """
        if pool_type == 'base':
            values = self.base_values
        elif pool_type == 'alpha':
            values = self.alpha_values
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")
        
        if len(values) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0}
        
        initial_value = values[0]
        final_value = values[-1]
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_value,
            'initial_value': initial_value
        }
    
    def get_pool_value(self, pool_type: str) -> float:
        """Get current value of a specific pool.
        
        Args:
            pool_type: Type of pool ('base' or 'alpha')
            
        Returns:
            Current pool value
        """
        if pool_type == 'base':
            return self.base_pool.capital
        elif pool_type == 'alpha':
            return self.alpha_pool.capital
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")
    
    def get_total_value(self) -> float:
        """Get total portfolio value including cash.
        
        Returns:
            Total portfolio value
        """
        total_value = self.base_pool.capital + self.alpha_pool.capital
        if hasattr(self, 'cash'):
            total_value += self.cash
        return total_value


class GeneralPortfolio:
    """Multi-asset portfolio for handling 1-N tickers."""
    
    def __init__(self,
                 initial_capital: float = 100.0,
                 commission_rate: float = 0.001,
                 interest_rate_daily: float = 0.00025,
                 spread_rate: float = 0.0002,
                 slippage_std: float = 0.0005,
                 funding_enabled: bool = True,
                 tax_rate: float = 0.45,
                 max_positions: int = 10,
                 logger: Optional[logging.Logger] = None) -> None:
        """Initialize the general portfolio.
        
        Args:
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate for trades
            interest_rate_daily: Daily interest rate for borrowed funds
            spread_rate: Spread cost per trade
            slippage_std: Standard deviation for slippage simulation
            funding_enabled: Whether to charge interest on borrowed funds
            tax_rate: Tax rate on capital gains
            max_positions: Maximum number of concurrent positions
            logger: Optional logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        
        # Portfolio parameters
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.commission_rate: float = commission_rate
        self.interest_rate_daily: float = interest_rate_daily
        self.spread_rate: float = spread_rate
        self.slippage_std: float = slippage_std
        self.funding_enabled: bool = funding_enabled
        self.tax_rate: float = tax_rate
        self.max_positions: int = max_positions
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.portfolio_values: List[float] = [initial_capital]
        self.trade_log: List[Dict[str, Any]] = []
        self.cumulative_tax: float = 0.0
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.current_year: Optional[int] = None
        self.yearly_gains: Dict[str, float] = {}
        self.logger.info(f"Initialized general portfolio with ${initial_capital:.2f} capital, max {max_positions} positions")
    
    @property
    def total_value(self) -> float:
        """Get total portfolio value including cash and positions."""
        total = self.cash
        for position in self.positions.values():
            total += position.quantity * position.current_price
        return total

    def add_position(self, symbol: str, quantity: float, price: float, timestamp: Any) -> bool:
        """Add a new position (alias for open_position)."""
        return self.open_position(symbol, quantity, price, timestamp)

    def update_position(self, symbol: str, quantity: float, price: float, timestamp: Any) -> bool:
        """Update an existing position by adding to it."""
        if symbol not in self.positions:
            return False
        
        old_position = self.positions[symbol]
        
        # Calculate new average price
        total_quantity = old_position.quantity + quantity
        total_cost = (old_position.quantity * old_position.avg_price) + (quantity * price)
        new_avg_price = total_cost / total_quantity
        
        # Update position
        old_position.quantity = total_quantity
        old_position.avg_price = new_avg_price
        old_position.current_price = price
        
        # Calculate and update costs
        commission = quantity * price * self.commission_rate
        old_position.total_commission += commission
        self.total_commission += commission
        
        # Deduct additional cash for the new quantity
        additional_cost = quantity * price + commission
        self.cash -= additional_cost
        
        # Log the trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'UPDATE',
            'quantity': quantity,
            'price': price,
            'new_total_quantity': total_quantity,
            'new_avg_price': new_avg_price,
            'commission': commission
        }
        self.trade_log.append(trade_record)
        
        return True

    def close_position(self, symbol: str, price: float, timestamp: Any, quantity: Optional[float] = None) -> bool:
        """Close a position (alias for _close_position)."""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        close_quantity = quantity or position.quantity
        
        if close_quantity > position.quantity:
            return False
        
        # Calculate proportional P&L
        proportion = close_quantity / position.quantity
        realized_pnl = proportion * ((price - position.avg_price) * position.quantity)
        
        # Calculate costs
        commission = close_quantity * price * self.commission_rate
        spread_cost = close_quantity * price * self.spread_rate
        slippage = np.random.normal(0, self.slippage_std) * close_quantity * price
        
        # Update position or remove it
        if close_quantity == position.quantity:
            # Close entire position
            position.realized_pnl += realized_pnl - commission - spread_cost - slippage
            self.cash += close_quantity * price - commission - spread_cost - slippage
            del self.positions[symbol]
        else:
            # Partial close
            position.quantity -= close_quantity
            position.realized_pnl += realized_pnl - commission - spread_cost - slippage
            self.cash += close_quantity * price - commission - spread_cost - slippage
        
        # Update tracking
        self.total_commission += commission
        self.total_slippage += slippage
        
        # Log the trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'quantity': close_quantity,
            'price': price,
            'realized_pnl': realized_pnl,
            'commission': commission,
            'spread_cost': spread_cost,
            'slippage': slippage
        }
        self.trade_log.append(trade_record)
        
        return True

    def get_position_value(self, symbol: str, price: float) -> float:
        """Get current value of a position with explicit price."""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol].quantity * price

    def get_position_value_current(self, symbol: str) -> float:
        """Get current market value of position for the given symbol.
        
        Args:
            symbol: The ticker symbol to get position value for
            
        Returns:
            float: Current market value of the position (quantity * current_price)
                   Returns 0.0 if symbol has no position
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Defensive check for required attributes
        if not hasattr(position, 'quantity') or not hasattr(position, 'current_price'):
            return 0.0
        
        return position.quantity * position.current_price

    def get_position_allocation(self, symbol: str) -> float:
        """Get position as percentage of total portfolio value.
        
        Args:
            symbol: The ticker symbol to get allocation for
            
        Returns:
            float: Position value as percentage of total portfolio (0.0 to 1.0)
                   Returns 0.0 if symbol has no position or portfolio is empty
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        
        position_value = self.get_position_value_current(symbol)
        return position_value / total_value

    def process_dividend(self, symbol: str, dividend_per_share: float) -> None:
        """Process dividend payment for the given symbol.
        
        Args:
            symbol: The ticker symbol receiving the dividend
            dividend_per_share: Dividend amount per share
            
        Note:
            - Increases cash by dividend amount
            - Logs the transaction in trade_history
            - Does nothing if symbol has no position
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Defensive check for quantity attribute
        if not hasattr(position, 'quantity'):
            return
        
        dividend_amount = position.quantity * dividend_per_share
        
        # Add to cash
        self.cash += dividend_amount
        
        # Initialize trade_history if it doesn't exist
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        
        # Log the dividend
        dividend_record = {
            'timestamp': datetime.now(),
            'type': 'dividend',
            'symbol': symbol,
            'quantity': position.quantity,
            'dividend_per_share': dividend_per_share,
            'total_amount': dividend_amount
        }
        
        self.trade_history.append(dividend_record)

    def process_split(self, symbol: str, split_ratio: float) -> None:
        """Process stock split for the given symbol.
        
        Args:
            symbol: The ticker symbol being split
            split_ratio: Split ratio (e.g., 2.0 for 2-for-1 split, 0.5 for 1-for-2)
            
        Note:
            - Adjusts position quantity by split ratio
            - Adjusts average price inversely by split ratio
            - Adjusts current price inversely by split ratio
            - Logs the transaction in trade_history
            - Does nothing if symbol has no position
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Defensive checks for required attributes
        if not all(hasattr(position, attr) for attr in ['quantity', 'avg_price', 'current_price']):
            return
        
        # Store old values for logging
        old_quantity = position.quantity
        old_avg_price = position.avg_price
        old_current_price = position.current_price
        
        # Apply split
        position.quantity = int(position.quantity * split_ratio)
        position.avg_price = old_avg_price / split_ratio
        position.current_price = old_current_price / split_ratio
        
        # Initialize trade_history if it doesn't exist
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        
        # Log the split
        split_record = {
            'timestamp': datetime.now(),
            'type': 'stock_split',
            'symbol': symbol,
            'old_quantity': old_quantity,
            'new_quantity': position.quantity,
            'split_ratio': split_ratio,
            'old_avg_price': old_avg_price,
            'new_avg_price': position.avg_price
        }
        
        self.trade_history.append(split_record)

    def get_total_value(self) -> float:
        """Get total portfolio value including cash and positions.
        
        Returns:
            float: Total portfolio value
        """
        return self.total_value

    def calculate_total_value(self) -> float:
        """Calculate total portfolio value."""
        return self.total_value

    def can_add_position(self, symbol: str) -> bool:
        """Check if we can add a position for the given symbol."""
        return (len(self.positions) < self.max_positions and
                symbol not in self.positions)

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.quantity * pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.positions.items()},
            'total_return': ((self.total_value / self.initial_capital) - 1) * 100,
            'total_commission': self.total_commission,
            'total_trades': len(self.trade_log)
        }

    def rebalance(self, target_allocation: Dict[str, float], tolerance: float = 0.02) -> Dict[str, Any]:
        """Rebalance portfolio to target allocation."""
        # This is a simplified rebalance - in practice would be more complex
        trades = []
        current_allocation = self.get_allocation()
        
        for symbol, target_weight in target_allocation.items():
            current_weight = current_allocation.get(symbol, 0.0)
            diff = target_weight - current_weight
            
            if abs(diff) > tolerance:
                # Calculate trade amount
                trade_value = diff * self.total_value
                if symbol in self.positions:
                    current_price = self.positions[symbol].current_price
                    trade_quantity = trade_value / current_price
                    
                    if trade_quantity > 0:
                        # Buy
                        if self.can_add_position(symbol):
                            self.add_position(symbol, trade_quantity, current_price, pd.Timestamp.now())
                            trades.append({'action': 'BUY', 'symbol': symbol, 'quantity': trade_quantity})
                    else:
                        # Sell
                        self.close_position(symbol, current_price, pd.Timestamp.now(), abs(trade_quantity))
                        trades.append({'action': 'SELL', 'symbol': symbol, 'quantity': abs(trade_quantity)})
        
        return {
            'trades': trades,
            'new_allocation': self.get_allocation()
        }

    def get_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation."""
        total_value = self.total_value
        allocation = {}
        
        # Add positions
        for symbol, position in self.positions.items():
            weight = (position.quantity * position.current_price) / total_value
            allocation[symbol] = weight
        
        # Add cash
        allocation['CASH'] = self.cash / total_value
        
        return allocation

    def pay_dividend(self, symbol: str, dividend_per_share: float) -> None:
        """Pay dividend for a position."""
        if symbol in self.positions:
            position = self.positions[symbol]
            dividend_amount = position.quantity * dividend_per_share
            self.cash += dividend_amount
            
            # Log the dividend
            trade_record = {
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol,
                'action': 'DIVIDEND',
                'amount': dividend_amount,
                'dividend_per_share': dividend_per_share
            }
            self.trade_log.append(trade_record)

    def handle_stock_split(self, symbol: str, split_ratio: int, reverse_ratio: int = 1) -> None:
        """Handle stock split."""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Update quantity and price
            total_ratio = split_ratio / reverse_ratio
            position.quantity *= total_ratio
            position.avg_price /= total_ratio
            position.current_price /= total_ratio


    def process_tick(self,
                     timestamp: Any,
                     market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process a market tick and update portfolio state.
        
        Args:
            timestamp: Current timestamp
            market_data: Dictionary with symbol -> DataFrame mapping
            
        Returns:
            Dictionary with updated portfolio information
        """
        total_portfolio_value = self.cash
        position_updates: List[Dict[str, Any]] = []
        
        # Update all positions
        for symbol, position in list(self.positions.items()):
            if symbol in market_data:
                current_data = market_data[symbol]
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    day_high = current_data['High'].iloc[-1]
                    day_low = current_data['Low'].iloc[-1]
                    
                    # Update position
                    update_result = self._update_position(
                        position, current_price, day_high, day_low, timestamp
                    )
                    position_updates.append(update_result)
                    
                    # Check if position should be closed
                    if update_result.get('should_close', False):
                        self._close_position(symbol, update_result['close_reason'], current_price, timestamp)
                        continue
                    
                    # Add position value to portfolio
                    total_portfolio_value += position.quantity * current_price
        
        # Calculate financing costs
        financing_cost = self._calculate_financing_costs()
        self.cash -= financing_cost
        
        # Update portfolio value
        self.portfolio_values.append(total_portfolio_value)
        
        # Handle tax calculations
        self._handle_tax_calculation(timestamp)
        
        return {
            'timestamp': timestamp,
            'total_value': total_portfolio_value,
            'cash': self.cash,
            'position_count': len(self.positions),
            'position_updates': position_updates,
            'financing_cost': financing_cost,
            'cumulative_tax': self.cumulative_tax
        }
    
    def _update_position(self,
                         position: Position,
                         current_price: float,
                         day_high: float,
                         day_low: float,
                         timestamp: Any) -> Dict[str, Any]:
        """Update a single position.
        
        Args:
            position: Position to update
            current_price: Current market price
            day_high: High price for the day
            day_low: Low price for the day
            timestamp: Current timestamp
            
        Returns:
            Dictionary with position update information
        """
        old_price = position.current_price
        position.current_price = current_price
        
        # Calculate unrealized P&L
        position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
        
        # Check stop loss
        should_close = False
        close_reason = None
        exit_price = current_price
        
        if position.stop_loss_price and day_low <= position.stop_loss_price:
            should_close = True
            close_reason = "STOP_LOSS"
            exit_price = position.stop_loss_price
            position.unrealized_pnl = (exit_price - position.avg_price) * position.quantity
            
        # Check take profit
        elif position.take_profit_price and day_high >= position.take_profit_price:
            should_close = True
            close_reason = "TAKE_PROFIT"
            exit_price = position.take_profit_price
            position.unrealized_pnl = (exit_price - position.avg_price) * position.quantity
        
        # Calculate daily P&L change
        if old_price != current_price:
            daily_pnl_change = (current_price - old_price) * position.quantity
        else:
            daily_pnl_change = 0.0
        
        return {
            'symbol': position.symbol,
            'quantity': position.quantity,
            'entry_price': position.avg_price,
            'current_price': current_price,
            'unrealized_pnl': position.unrealized_pnl,
            'daily_pnl_change': daily_pnl_change,
            'should_close': should_close,
            'close_reason': close_reason,
            'exit_price': exit_price
        }
    
    def _calculate_financing_costs(self) -> float:
        """Calculate financing costs for all positions.
        
        Returns:
            Total financing cost for the period
        """
        if not self.funding_enabled:
            return 0.0
        
        total_cost = 0.0
        for position in self.positions.values():
            # Assume long positions, so financing cost is on full position value
            position_value = position.quantity * position.current_price
            daily_cost = position_value * self.interest_rate_daily
            total_cost += daily_cost
        
        return total_cost
    
    def _close_position(self, symbol: str, reason: str, exit_price: float, timestamp: Any) -> None:
        """Close a position.
        
        Args:
            symbol: Symbol to close
            reason: Reason for closing
            exit_price: Exit price
            timestamp: Exit timestamp
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        realized_pnl = (exit_price - position.avg_price) * position.quantity
        
        # Calculate total costs
        commission = position.quantity * exit_price * self.commission_rate
        spread_cost = position.quantity * exit_price * self.spread_rate
        slippage = np.random.normal(0, self.slippage_std) * position.quantity * exit_price
        
        # Update cash
        position.realized_pnl += realized_pnl - commission - spread_cost - slippage
        self.cash += position.quantity * exit_price - commission - spread_cost - slippage
        
        # Update tracking
        self.total_commission += commission
        self.total_slippage += slippage
        
        # Log the trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'quantity': position.quantity,
            'entry_price': position.avg_price,
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
            'commission': commission,
            'spread_cost': spread_cost,
            'slippage': slippage,
            'reason': reason
        }
        self.trade_log.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.debug(f"Closed {symbol} position: {reason}, P&L: {realized_pnl:.2f}")
    
    def open_position(self,
                      symbol: str,
                      quantity: float,
                      price: float,
                      timestamp: Any,
                      stop_loss_price: Optional[float] = None,
                      take_profit_price: Optional[float] = None,
                      leverage: float = 1.0) -> bool:
        """Open a new position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            leverage: Leverage factor (1.0 = no leverage)
            
        Returns:
            True if position was opened successfully
        """
        # Check if we can open another position
        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"Cannot open position for {symbol}: max positions reached")
            return False
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            self.logger.warning(f"Cannot open position for {symbol}: already have position")
            return False
        
        # Calculate costs - leverage reduces required cash
        total_position_value = quantity * price
        margin_required = total_position_value / leverage  # This is the cash needed
        commission = margin_required * self.commission_rate  # Commission on margin, not full position
        spread_cost = 0.0
        slippage = 0.0
        
        # Total required cash
        total_required = margin_required + commission + spread_cost + slippage
        
        # Check if we have enough cash
        if total_required > self.cash:
            self.logger.warning(f"Insufficient cash for {symbol} position: need {total_required:.2f}, have {self.cash:.2f}")
            return False
        
        # Deduct from cash
        self.cash -= total_required
        
        # Create position with full quantity (leverage affects margin, not quantity)
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            current_price=price,
            entry_timestamp=timestamp,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            commission_paid=commission + spread_cost + slippage,
            total_cost=margin_required,  # Store margin cost, not full position cost
            total_commission=commission
        )

        self.positions[symbol] = position
        
        # Log the trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'OPEN',
            'quantity': position.quantity,
            'price': price,
            'leverage': leverage,
            'margin_required': margin_required,
            'commission': commission,
            'spread_cost': spread_cost,
            'slippage': slippage,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price
        }
        self.trade_log.append(trade_record)
        
        self.total_commission += commission
        self.total_slippage += slippage
        
        self.logger.info(f"Opened {symbol} position: {quantity}@{price:.4f}, leverage: {leverage:.1f}x, margin: {margin_required:.2f}")
        return True
    
    def _handle_tax_calculation(self, timestamp: Any) -> None:
        """Handle tax calculations at year end.
        
        Args:
            timestamp: Current timestamp
        """
        if self.current_year is None:
            self.current_year = timestamp.year
            return
        
        current_year = timestamp.year
        
        # If new year, process previous year's taxes
        if current_year != self.current_year:
            # Add gains from yearly tracking
            yearly_total = sum(self.yearly_gains.values())
            
            # Apply tax if positive
            if yearly_total > 0:
                tax = yearly_total * self.tax_rate
                self.cumulative_tax += tax
                self.cash -= tax
                
                self.logger.info(f"Applied yearly taxes: ${tax:.2f}")
            
            # Reset for new year
            self.yearly_gains.clear()
            self.current_year = current_year
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)
        
        # Basic metrics
        total_return = ((portfolio_values[-1] / self.initial_capital) - 1) * 100
        
        # Calculate returns for risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Risk metrics
        sharpe_ratio = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trade_log
                             if t.get('action') == 'CLOSE' and t.get('realized_pnl', 0) > 0])
        closed_trades = len([t for t in self.trade_log if t.get('action') == 'CLOSE'])
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'cumulative_tax': self.cumulative_tax,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_trades': len(self.trade_log),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'final_cash': self.cash,
            'current_positions': len(self.positions),
            'portfolio_values': portfolio_values,
            'trade_log': self.trade_log
        }
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.portfolio_values = [self.initial_capital]
        self.trade_log.clear()
        self.cumulative_tax = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.current_year = None
        self.yearly_gains.clear()
        
        self.logger.info("General portfolio reset to initial state")