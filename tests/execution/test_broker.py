"""
Comprehensive tests for the broker execution module.

This module contains tests for order execution, broker interactions,
trade routing, and order management functionality.
"""

import pytest
import pandas as pd
import numpy as np

# Import the modules being tested
try:
    from backtester.execution.broker import SimulatedBroker
    from backtester.execution.order import Order, OrderType, OrderSide
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestSimulatedBroker:
    """Test suite for the SimulatedBroker class."""

    def test_initialization(self):
        """Test SimulatedBroker initialization."""
        broker = SimulatedBroker(
            commission_rate=0.001,
            min_commission=1.0,
            spread=0.0001,
            slippage_model="normal",
            slippage_std=0.0005,
            latency_ms=100.0
        )
        
        assert broker.commission_rate == 0.001
        assert broker.min_commission == 1.0
        assert broker.spread == 0.0001
        assert broker.slippage_model == "normal"
        assert broker.slippage_std == 0.0005
        assert broker.latency_ms == 100.0
        assert broker.order_manager is not None
        assert broker.cash_balance == 0.0
        assert len(broker.positions) == 0

    def test_set_market_data(self):
        """Test setting market data."""
        broker = SimulatedBroker()
        
        # Create test market data
        data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [101.5, 102.5, 103.0],
            'Low': [99.5, 100.5, 101.5],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000, 1100, 1200]
        })
        
        broker.set_market_data('SPY', data)
        
        assert 'SPY' in broker.market_data
        assert len(broker.market_data['SPY']) == 3
        assert broker.current_prices['SPY'] == 103.0

    def test_get_current_price(self):
        """Test getting current price."""
        broker = SimulatedBroker()
        
        broker.current_prices['SPY'] = 400.0
        
        assert broker.get_current_price('SPY') == 400.0
        assert broker.get_current_price('NONEXISTENT') == 0.0

    def test_get_bid_ask(self):
        """Test getting bid-ask spread."""
        broker = SimulatedBroker(spread=0.001)
        
        broker.current_prices['SPY'] = 400.0
        
        bid, ask = broker.get_bid_ask('SPY')
        
        expected_bid = 400.0 * (1 - 0.0005)  # spread/2
        expected_ask = 400.0 * (1 + 0.0005)
        
        assert abs(bid - expected_bid) < 0.0001
        assert abs(ask - expected_ask) < 0.0001

    def test_execute_market_order(self):
        """Test market order execution."""
        broker = SimulatedBroker(commission_rate=0.001, min_commission=1.0)
        
        # Set up market data and cash
        data = pd.DataFrame({
            'Open': [400.0], 'High': [401.0], 'Low': [399.0],
            'Close': [400.0], 'Volume': [1000]
        })
        broker.set_market_data('SPY', data)
        broker.cash_balance = 10000.0
        
        # Create and execute market order
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        success = broker.execute_order(order)
        
        assert success is True
        assert order.status.value == 'FILLED'
        assert order.filled_quantity == 10
        assert order.filled_price is not None
        assert order.commission > 0

    def test_execute_limit_order_favorable(self):
        """Test limit order execution when price is favorable."""
        broker = SimulatedBroker(commission_rate=0.001)
        
        # For buy limit: executes when market price <= limit price
        # Set market price below limit and provide sufficient cash
        broker.current_prices['SPY'] = 395.0  # Market price
        broker.cash_balance = 10000.0
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            price=400.0  # Limit price (market 395 <= limit 400, so favorable)
        )
        
        success = broker.execute_order(order)
        
        assert success is True
        assert order.status.value == 'FILLED'

    def test_execute_limit_order_unfavorable(self):
        """Test limit order execution when price is not favorable."""
        broker = SimulatedBroker(commission_rate=0.001)
        
        # For buy limit: does NOT execute when market price > limit price
        # Set market price above limit
        broker.current_prices['SPY'] = 405.0  # Market price
        broker.cash_balance = 10000.0
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            price=400.0  # Limit price (market 405 > limit 400, so unfavorable)
        )
        
        success = broker.execute_order(order)
        
        assert success is False
        # Price not favorable, should remain pending
        assert order.status.value == 'PENDING'

    def test_execute_stop_order(self):
        """Test stop order execution."""
        broker = SimulatedBroker(commission_rate=0.001)
        
        # For buy stop: executes when market price >= stop price
        # Set market price above stop and provide sufficient cash
        broker.current_prices['SPY'] = 405.0  # Market price
        broker.cash_balance = 10000.0
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=10,
            stop_price=400.0  # Stop trigger (market 405 >= stop 400, so triggered)
        )
        
        success = broker.execute_order(order)
        
        assert success is True
        assert order.status.value == 'FILLED'

    def test_commission_calculation(self):
        """Test commission calculation."""
        broker = SimulatedBroker(commission_rate=0.001, min_commission=1.0)
        
        # Test small order (should use minimum)
        commission = broker._calculate_commission(1, 100.0)
        assert commission == 1.0  # Min commission
        
        # Test large order
        commission = broker._calculate_commission(1000, 100.0)
        expected = max(1000 * 100 * 0.001, 1.0)  # = 100, which is > 1
        assert commission == expected

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        broker = SimulatedBroker(slippage_model="fixed", slippage_std=0.001)
        
        price = 400.0
        
        # Set seed for reproducible results
        np.random.seed(42)
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        slippage = broker._calculate_slippage(order, price)
        
        # For fixed slippage model, slippage should be price * slippage_std
        expected_slippage = price * 0.001
        assert abs(slippage - expected_slippage) < 0.0001

    def test_max_quantity_calculation(self):
        """Test maximum executable quantity calculation."""
        broker = SimulatedBroker()
        
        broker.cash_balance = 5000.0
        broker.current_prices['SPY'] = 100.0
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100  # More than cash allows
        )
        
        max_qty = broker._calculate_max_quantity(order, 100.0)
        
        # Should be limited by cash: 5000 / 100 = 50
        assert max_qty == 50

    def test_positions_and_cash_update(self):
        """Test positions and cash balance updates."""
        broker = SimulatedBroker(commission_rate=0.001)
        
        # Set up
        broker.current_prices['SPY'] = 400.0
        broker.cash_balance = 10000.0
        
        # Create buy order
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        # Execute order (simulate fill)
        order.update_fill(10, 400.0, 4.0)
        broker._update_positions_and_cash(order, 10, 400.0, 4.0)
        
        # Check updates
        assert broker.positions['SPY'] == 10
        assert broker.cash_balance == 10000.0 - (10 * 400.0 + 4.0)

    def test_process_market_data_update(self):
        """Test market data update processing."""
        broker = SimulatedBroker()
        
        # Set up market data and sufficient cash
        data = pd.DataFrame({
            'Open': [400.0], 'High': [401.0], 'Low': [399.0],
            'Close': [400.5], 'Volume': [1000]
        })
        broker.set_market_data('SPY', data)
        broker.cash_balance = 10000.0
        
        # Create pending order
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        broker.order_manager.orders[order.order_id] = order
        
        # Process market update
        timestamp = pd.Timestamp.now()
        broker.process_market_data_update('SPY', timestamp, 400.0, 401.0, 399.0, 400.5, 1000)
        
        # Order should be executed
        assert order.status.value == 'FILLED'
        assert order.filled_quantity == 10

    def test_get_account_summary(self):
        """Test account summary retrieval."""
        broker = SimulatedBroker()
        
        # Set up some positions
        broker.positions['SPY'] = 10
        broker.positions['AAPL'] = -5
        broker.cash_balance = 8000.0
        broker.current_prices['SPY'] = 400.0
        broker.current_prices['AAPL'] = 150.0
        
        summary = broker.get_account_summary()
        
        assert summary['cash_balance'] == 8000.0
        assert summary['positions']['SPY'] == 10
        assert summary['positions']['AAPL'] == -5
        assert summary['total_trades'] == 0
        assert 'portfolio_value' in summary

    def test_get_trade_history(self):
        """Test trade history retrieval."""
        broker = SimulatedBroker()
        
        # Simulate some trades
        broker.trade_history = [
            {'symbol': 'SPY', 'quantity': 10, 'price': 400.0},
            {'symbol': 'AAPL', 'quantity': 5, 'price': 150.0}
        ]
        
        # Test all trades
        all_trades = broker.get_trade_history()
        assert len(all_trades) == 2
        
        # Test filtered trades
        spy_trades = broker.get_trade_history('SPY')
        assert len(spy_trades) == 1
        assert spy_trades[0]['symbol'] == 'SPY'

    def test_reset(self):
        """Test broker reset functionality."""
        broker = SimulatedBroker()
        
        # Set up some state
        broker.positions['SPY'] = 10
        broker.cash_balance = 5000.0
        broker.trade_history.append({'symbol': 'SPY', 'quantity': 10})
        
        # Reset
        broker.reset()
        
        # Check state is cleared
        assert len(broker.positions) == 0
        assert broker.cash_balance == 0.0
        assert len(broker.trade_history) == 0

    def test_sell_order_execution(self):
        """Test sell order execution."""
        broker = SimulatedBroker(commission_rate=0.001)
        
        # Set up
        broker.current_prices['SPY'] = 400.0
        broker.cash_balance = 10000.0
        broker.positions['SPY'] = 10  # Existing position
        
        # Create sell order
        order = Order(
            symbol='SPY',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5
        )
        
        # Execute order
        broker._update_positions_and_cash(order, 5, 400.0, 2.0)
        
        # Check results
        assert broker.positions['SPY'] == 5  # Reduced position
        assert broker.cash_balance == 10000.0 + (5 * 400.0 - 2.0)

    def test_insufficient_funds_rejection(self):
        """Test order rejection due to insufficient funds."""
        broker = SimulatedBroker()
        
        # Set up market data but insufficient cash
        broker.current_prices['SPY'] = 400.0
        broker.cash_balance = 1000.0  # Not enough for 10 shares at 400
        
        order = Order(
            symbol='SPY',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        # Execute order (should be limited by cash)
        broker.execute_order(order)
        
        # Order might still execute with reduced quantity
        assert order.status.value in ['FILLED', 'PARTIALLY_FILLED', 'REJECTED']


if __name__ == "__main__":
    pytest.main([__file__])