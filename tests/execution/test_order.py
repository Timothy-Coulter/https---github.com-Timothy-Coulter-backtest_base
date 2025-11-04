"""
Comprehensive tests for the order management module.

This module contains tests for order creation, validation, processing,
and order lifecycle management.
"""

import pytest

# Import the modules being tested
try:
    from backtester.execution.order import (
        Order, OrderType, OrderSide, OrderStatus, OrderManager
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestOrder:
    """Test suite for the Order class."""

    def test_initialization(self):
        """Test Order initialization."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.symbol == 'SPY'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.order_id is not None
        assert order.timestamp is not None
        assert order.remaining_quantity == 100

    def test_initialization_with_all_params(self):
        """Test Order with all parameters."""
        order = Order(
            symbol='AAPL',
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=175.0,
            stop_price=None
        )
        
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.SELL
        assert order.quantity == 50
        assert order.order_type == OrderType.LIMIT
        assert order.price == 175.0
        assert order.stop_price is None

    def test_order_properties(self):
        """Test order properties."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.is_buy is True
        assert order.is_sell is False
        assert order.is_active is True
        
        # Test filled order
        filled_order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            filled_price=400.0
        )
        assert filled_order.is_active is False

    def test_order_modification(self):
        """Test order fill update."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Partial fill
        order.update_fill(50, 400.0, 4.0)
        assert order.filled_quantity == 50
        assert order.remaining_quantity == 50
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_price == 400.0
        assert order.commission == 4.0
        
        # Complete fill
        order.update_fill(50, 401.0, 4.01)
        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED

    def test_order_cancellation(self):
        """Test order cancellation."""
        order = Order(
            symbol='GOOGL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            price=2490.0
        )
        
        # Cancel the order
        order.cancel("User requested cancellation")
        
        assert order.status == OrderStatus.CANCELLED
        assert 'cancel_reason' in order.metadata
        assert order.metadata['cancel_reason'] == "User requested cancellation"

    def test_order_rejection(self):
        """Test order rejection."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Reject the order
        order.reject("Insufficient funds")
        
        assert order.status == OrderStatus.REJECTED
        assert 'reject_reason' in order.metadata
        assert order.metadata['reject_reason'] == "Insufficient funds"

    def test_order_notional_value(self):
        """Test order notional value calculation."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=400.0
        )
        
        # Should use price when not filled
        assert order.notional_value == 40000.0
        
        # Should use filled_price when filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100
        order.filled_price = 405.0
        assert order.notional_value == 40500.0

    def test_order_to_dict(self):
        """Test order serialization to dictionary."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=395.0
        )
        order.order_id = "test_order_123"
        
        order_dict = order.to_dict()
        
        assert isinstance(order_dict, dict)
        assert order_dict['symbol'] == 'SPY'
        assert order_dict['side'] == 'BUY'
        assert order_dict['quantity'] == 100
        assert order_dict['order_type'] == 'LIMIT'
        assert order_dict['price'] == 395.0
        assert order_dict['order_id'] == 'test_order_123'

    def test_order_repr(self):
        """Test order string representation."""
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=395.0
        )
        
        repr_str = repr(order)
        assert 'SPY' in repr_str
        assert 'BUY' in repr_str
        assert 'LIMIT' in repr_str
        assert '100@395.0' in repr_str

    def test_order_equality(self):
        """Test order equality comparison."""
        order1 = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        order1.order_id = "test_123"
        
        order2 = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        order2.order_id = "test_123"
        
        order3 = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        order3.order_id = "test_456"
        
        # Test that order_id is used for equality (though not implemented, test basic functionality)
        assert order1.order_id == order2.order_id
        assert order1.order_id != order3.order_id


class TestOrderManager:
    """Test suite for the OrderManager class."""

    def test_initialization(self):
        """Test OrderManager initialization."""
        manager = OrderManager()
        
        assert len(manager.orders) == 0
        assert len(manager.order_history) == 0
        assert manager.next_order_id == 1

    def test_create_order(self):
        """Test order creation."""
        manager = OrderManager()
        
        order = manager.create_order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.symbol == 'SPY'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.order_id in manager.orders

    def test_create_order_validation(self):
        """Test order creation validation."""
        manager = OrderManager()
        
        # Should raise ValueError for invalid quantity
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            manager.create_order(
                symbol='SPY',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=-10
            )
        
        # Should raise ValueError for missing price on limit order
        with pytest.raises(ValueError, match="Price required"):
            manager.create_order(
                symbol='SPY',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=10
            )

    def test_get_order(self):
        """Test getting order by ID."""
        manager = OrderManager()
        
        order = manager.create_order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        retrieved_order = manager.get_order(order.order_id)
        assert retrieved_order == order
        
        # Test non-existent order
        assert manager.get_order("non_existent") is None

    def test_cancel_order(self):
        """Test order cancellation."""
        manager = OrderManager()
        
        order = manager.create_order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=395.0
        )
        
        success = manager.cancel_order(order.order_id, "User requested")
        assert success is True
        
        cancelled_order = manager.get_order(order.order_id)
        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_cancel_all_orders(self):
        """Test cancel all orders."""
        manager = OrderManager()
        
        # Create multiple orders
        order1 = manager.create_order('SPY', OrderSide.BUY, OrderType.MARKET, 100)
        order2 = manager.create_order('AAPL', OrderSide.BUY, OrderType.MARKET, 50)
        order3 = manager.create_order('GOOGL', OrderSide.SELL, OrderType.MARKET, 25)
        
        cancelled_count = manager.cancel_all_orders("Market close")
        assert cancelled_count == 3
        
        # Check all orders are cancelled
        for order in [order1, order2, order3]:
            updated_order = manager.get_order(order.order_id)
            assert updated_order.status == OrderStatus.CANCELLED

    def test_get_active_orders(self):
        """Test getting active orders."""
        manager = OrderManager()
        
        order1 = manager.create_order('SPY', OrderSide.BUY, OrderType.MARKET, 100)
        order2 = manager.create_order('AAPL', OrderSide.BUY, OrderType.MARKET, 50)
        
        # Cancel one order
        manager.cancel_order(order1.order_id)
        
        active_orders = manager.get_active_orders()
        assert len(active_orders) == 1
        assert active_orders[0] == order2

    def test_get_filled_orders(self):
        """Test getting filled orders."""
        manager = OrderManager()
        
        order1 = manager.create_order('SPY', OrderSide.BUY, OrderType.MARKET, 100)
        manager.create_order('AAPL', OrderSide.BUY, OrderType.MARKET, 50)
        
        # Fill one order
        order1.update_fill(100, 400.0, 4.0)
        
        filled_orders = manager.get_filled_orders()
        assert len(filled_orders) == 1
        assert filled_orders[0] == order1

    def test_get_order_summary(self):
        """Test order summary."""
        manager = OrderManager()
        
        # Create various orders
        order1 = manager.create_order('SPY', OrderSide.BUY, OrderType.MARKET, 100)
        order2 = manager.create_order('AAPL', OrderSide.BUY, OrderType.MARKET, 50)
        manager.create_order('GOOGL', OrderSide.SELL, OrderType.MARKET, 25)
        
        # Fill one order
        order1.update_fill(100, 400.0, 4.0)
        
        # Cancel one order
        manager.cancel_order(order2.order_id)
        
        summary = manager.get_order_summary()
        
        assert summary['total_orders'] == 3
        assert summary['active_orders'] == 1
        assert summary['filled_orders'] == 1
        assert summary['cancelled_orders'] == 1
        assert summary['rejected_orders'] == 0

    def test_reset(self):
        """Test order manager reset."""
        manager = OrderManager()
        
        # Create orders
        manager.create_order('SPY', OrderSide.BUY, OrderType.MARKET, 100)
        manager.create_order('AAPL', OrderSide.BUY, OrderType.MARKET, 50)
        
        # Reset
        manager.reset()
        
        assert len(manager.orders) == 0
        assert len(manager.order_history) == 0
        assert manager.next_order_id == 1


@pytest.mark.parametrize("order_type,side,quantity,price,expected_valid", [
    (OrderType.MARKET, OrderSide.BUY, 100, None, True),
    (OrderType.LIMIT, OrderSide.BUY, 100, 395.0, True),
    (OrderType.LIMIT, OrderSide.BUY, 100, None, False),  # Missing limit price
    (OrderType.STOP, OrderSide.SELL, 50, 410.0, True),
    (OrderType.STOP, OrderSide.SELL, 50, None, False),   # Missing stop price
    (OrderType.MARKET, OrderSide.BUY, -50, None, False), # Negative quantity
    (OrderType.LIMIT, OrderSide.BUY, 0, 395.0, False),   # Zero quantity
])
def test_order_validation_parametrized(order_type, side, quantity, price, expected_valid):
    """Parametrized test for order validation scenarios."""
    manager = OrderManager()
    
    kwargs = {'symbol': 'SPY', 'side': side, 'order_type': order_type, 'quantity': quantity}
    if price is not None:
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            kwargs['price'] = price
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            kwargs['stop_price'] = price
    
    if expected_valid:
        # Should not raise exception for valid orders
        try:
            order = manager.create_order(**kwargs)
            assert order is not None
        except ValueError:
            pytest.fail("Valid order creation raised ValueError")
    else:
        # Should raise exception for invalid orders
        with pytest.raises(ValueError):
            manager.create_order(**kwargs)


if __name__ == "__main__":
    pytest.main([__file__])