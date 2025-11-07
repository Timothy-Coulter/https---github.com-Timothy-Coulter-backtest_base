"""Simulated Broker for Order Execution.

This module provides a simulated broker that handles order execution, market data,
commission calculations, and trade reporting.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from backtester.core.config import SimulatedBrokerConfig

from .order import Order, OrderManager, OrderType


class SimulatedBroker:
    """Simulated broker for order execution and trade reporting."""

    def __init__(
        self,
        config: SimulatedBrokerConfig | None = None,
        commission_rate: float | None = None,
        min_commission: float | None = None,
        spread: float | None = None,
        slippage_model: str | None = None,
        slippage_std: float | None = None,
        latency_ms: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the simulated broker.

        Args:
            config: SimulatedBrokerConfig instance. If provided, other parameters are ignored.
            commission_rate: Commission rate for trades (as decimal) - deprecated, use config
            min_commission: Minimum commission per trade - deprecated, use config
            spread: Bid-ask spread (as decimal) - deprecated, use config
            slippage_model: Type of slippage model - deprecated, use config
            slippage_std: Standard deviation for slippage simulation - deprecated, use config
            latency_ms: Simulated latency in milliseconds - deprecated, use config
            logger: Optional logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            # Use config values - define attributes once
            self.commission_rate = config.commission_rate
            self.min_commission = config.min_commission
            self.spread = config.spread
            self.slippage_model = config.slippage_model
            self.slippage_std = config.slippage_std
            self.latency_ms = config.latency_ms
        else:
            # Backward compatibility - use individual parameters with defaults
            self.commission_rate = commission_rate or 0.001
            self.min_commission = min_commission or 1.0
            self.spread = spread or 0.0001
            self.slippage_model = slippage_model or "normal"
            self.slippage_std = slippage_std or 0.0005
            self.latency_ms = latency_ms or 0.0

        # State
        self.order_manager = OrderManager(logger)
        self.market_data: dict[str, pd.DataFrame] = {}
        self.current_prices: dict[str, float] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.cash_balance: float = 0.0
        self.positions: dict[str, float] = {}
        self.portfolio_value: float = 0.0

        self.logger.info("Simulated broker initialized")

    def set_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Set market data for a symbol.

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
        """
        self.market_data[symbol] = data.copy()
        if 'Close' in data.columns:
            self.current_prices[symbol] = data['Close'].iloc[-1]
        self.logger.info(f"Set market data for {symbol}: {len(data)} records")

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        return self.current_prices.get(symbol, 0.0)

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Get bid and ask prices for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (bid, ask) prices
        """
        price = self.get_current_price(symbol)
        spread_pct = self.spread / 2
        bid = price * (1 - spread_pct)
        ask = price * (1 + spread_pct)
        return bid, ask

    def execute_order(self, order: Order, market_data: pd.Series | None = None) -> bool:
        """Execute an order based on current market conditions.

        Args:
            order: Order to execute
            market_data: Optional current market data

        Returns:
            True if order was executed, False otherwise
        """
        if not order.is_active:
            return False

        # Simulate latency
        if self.latency_ms > 0:
            # In a real implementation, this would involve async operations
            pass

        # Get current market data
        current_price = self.get_current_price(order.symbol)
        if current_price == 0.0:
            order.reject("No market data available")
            return False

        bid, ask = self.get_bid_ask(order.symbol)

        # Execute based on order type
        execution_price = self._determine_execution_price(order, current_price, bid, ask)
        if execution_price is None:
            return False

        # Calculate quantity to fill
        fill_quantity = min(
            order.remaining_quantity, self._calculate_max_quantity(order, execution_price)
        )
        if fill_quantity <= 0:
            order.reject("Insufficient funds or position limits")
            return False

        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)

        # Update order with fill
        order.update_fill(fill_quantity, execution_price, commission)

        # Record trade
        trade_record = {
            'timestamp': order.timestamp,
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': fill_quantity,
            'price': execution_price,
            'commission': commission,
            'notional': fill_quantity * execution_price,
            'order_type': order.order_type.value,
        }
        self.trade_history.append(trade_record)

        # Update broker positions and cash
        self._update_positions_and_cash(order, fill_quantity, execution_price, commission)

        self.logger.info(
            f"Executed {order}: {fill_quantity}@{execution_price:.4f}, commission: ${commission:.2f}"
        )
        return True

    def _determine_execution_price(
        self, order: Order, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Determine execution price based on order type and market conditions.

        Args:
            order: Order to execute
            market_price: Current market price
            bid: Current bid price
            ask: Current ask price

        Returns:
            Execution price or None if not executable
        """
        # Handle market orders
        if order.order_type == OrderType.MARKET:
            return self._handle_market_order(order, bid, ask)

        # Handle limit orders
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order, market_price, bid, ask)

        # Handle stop orders
        elif order.order_type == OrderType.STOP:
            return self._handle_stop_order(order, market_price, bid, ask)

        # Handle stop-limit orders
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._handle_stop_limit_order(order, market_price, bid, ask)

    def _handle_market_order(self, order: Order, bid: float, ask: float) -> float:
        """Handle market order execution price determination."""
        if order.is_buy:
            return ask + self._calculate_slippage(order, ask)
        else:
            return bid + self._calculate_slippage(order, bid)

    def _handle_limit_order(
        self, order: Order, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle limit order execution price determination."""
        if order.is_buy and market_price <= (order.price or 0):
            assert order.price is not None
            return min(order.price, ask) + self._calculate_slippage(order, ask)
        elif order.is_sell and market_price >= (order.price or 0):
            assert order.price is not None
            return max(order.price, bid) + self._calculate_slippage(order, bid)
        else:
            return None  # Price not favorable

    def _handle_stop_order(
        self, order: Order, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle stop order execution price determination."""
        assert order.stop_price is not None
        if (order.is_buy and market_price >= order.stop_price) or (
            order.is_sell and market_price <= order.stop_price
        ):
            if order.is_buy:
                return ask + self._calculate_slippage(order, ask)
            else:
                return bid + self._calculate_slippage(order, bid)
        else:
            return None  # Stop not triggered

    def _handle_stop_limit_order(
        self, order: Order, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle stop-limit order execution price determination."""
        assert order.stop_price is not None
        assert order.price is not None
        if (order.is_buy and market_price >= order.stop_price) or (
            order.is_sell and market_price <= order.stop_price
        ):
            if order.is_buy and market_price <= order.price:
                return min(order.price, ask) + self._calculate_slippage(order, ask)
            elif order.is_sell and market_price >= order.price:
                return max(order.price, bid) + self._calculate_slippage(order, bid)
            else:
                return None  # Price not favorable for limit
        else:
            return None  # Stop not triggered

    def _calculate_slippage(self, order: Order, reference_price: float) -> float:
        """Calculate slippage for an order.

        Args:
            order: Order being executed
            reference_price: Reference price for slippage calculation

        Returns:
            Slippage amount
        """
        if self.slippage_model == "none":
            return 0.0
        elif self.slippage_model == "fixed":
            return reference_price * self.slippage_std
        elif self.slippage_model == "normal":
            slippage = np.random.normal(0, self.slippage_std)
            return reference_price * slippage

        return 0.0

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade.

        Args:
            quantity: Trade quantity
            price: Trade price

        Returns:
            Commission amount
        """
        notional_value = quantity * price
        commission = notional_value * self.commission_rate
        return max(commission, self.min_commission)

    def _calculate_max_quantity(self, order: Order, price: float) -> float:
        """Calculate maximum executable quantity based on available capital and position limits.

        Args:
            order: Order to assess
            price: Execution price

        Returns:
            Maximum executable quantity
        """
        # In a real implementation, this would check available cash and position limits
        # For simulation, we'll use a simple approach
        available_cash = max(0, self.cash_balance)
        max_quantity_by_cash = available_cash / price if price > 0 else 0

        # Position limits could be implemented here
        max_position_size = 1000000  # Placeholder limit

        current_position = self.positions.get(order.symbol, 0)
        max_additional = max_position_size - current_position if order.is_buy else current_position

        return min(order.remaining_quantity, max_quantity_by_cash, max_additional)

    def _update_positions_and_cash(
        self, order: Order, fill_quantity: float, execution_price: float, commission: float
    ) -> None:
        """Update broker positions and cash balance.

        Args:
            order: Executed order
            fill_quantity: Quantity filled
            execution_price: Execution price
            commission: Commission amount
        """
        # Update positions
        if order.is_buy:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + fill_quantity
            cash_change = -(fill_quantity * execution_price + commission)
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - fill_quantity
            cash_change = fill_quantity * execution_price - commission

        self.cash_balance += cash_change

        # Update portfolio value
        self.portfolio_value = self.cash_balance
        for symbol, position in self.positions.items():
            if position != 0:
                current_price = self.get_current_price(symbol)
                self.portfolio_value += position * current_price

    def process_market_data_update(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0,
    ) -> None:
        """Process market data update and execute pending orders.

        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Trading volume
        """
        # Update current price
        self.current_prices[symbol] = close_price

        # Execute pending orders for this symbol
        pending_orders = self.order_manager.get_active_orders(symbol)
        for order in pending_orders:
            self.execute_order(order)

        # Expire stale orders
        self._expire_stale_orders(symbol, timestamp)

    def _expire_stale_orders(self, symbol: str, timestamp: pd.Timestamp) -> None:
        """Expire orders that have been pending too long.

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
        """
        # In a real implementation, orders would expire based on time-in-force
        # For simulation, we'll skip this for now
        pass

    def get_account_summary(self) -> dict[str, Any]:
        """Get comprehensive account summary.

        Returns:
            Dictionary with account information
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if position != 0:
                current_price = self.get_current_price(symbol)
                # This is a simplified unrealized P&L calculation
                # In reality, you'd track the cost basis for each position
                unrealized_pnl += position * current_price * 0.01  # Placeholder

        return {
            'cash_balance': self.cash_balance,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'unrealized_pnl': unrealized_pnl,
            'total_commission': sum(trade['commission'] for trade in self.trade_history),
            'total_trades': len(self.trade_history),
            'order_summary': self.order_manager.get_order_summary(),
        }

    def get_trade_history(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get trade history.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of trade records
        """
        if symbol is None:
            return self.trade_history.copy()

        return [trade for trade in self.trade_history if trade['symbol'] == symbol]

    def reset(self) -> None:
        """Reset broker to initial state."""
        self.order_manager.reset()
        self.trade_history.clear()
        self.current_prices.clear()
        self.cash_balance = 0.0
        self.positions.clear()
        self.portfolio_value = 0.0
        self.logger.info("Broker reset to initial state")
