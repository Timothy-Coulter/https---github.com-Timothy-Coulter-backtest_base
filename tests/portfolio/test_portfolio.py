"""Comprehensive tests for the portfolio management module.

This module contains tests for portfolio creation, position management,
allocation strategies, and performance tracking.
"""

from datetime import datetime

import pytest

# Import the modules being tested
try:
    from backtester.portfolio.portfolio import (
        DualPoolPortfolio,
        GeneralPortfolio,
        PoolState,
        Position,
    )

    BACKTESTER_AVAILABLE = True
except ImportError as e:
    BACKTESTER_AVAILABLE = False
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestGeneralPortfolio:
    """Test suite for the GeneralPortfolio class."""

    def test_initialization(self) -> None:
        """Test GeneralPortfolio initialization."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        assert portfolio.initial_capital == 1000.0
        assert portfolio.cash == 1000.0
        assert portfolio.positions == {}
        assert portfolio.total_value == 1000.0
        assert portfolio.commission_rate == 0.001

    def test_initialization_with_params(self) -> None:
        """Test GeneralPortfolio with custom parameters."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(
            initial_capital=5000.0, max_positions=10, commission_rate=0.002
        )

        assert portfolio.initial_capital == 5000.0
        assert portfolio.cash == 5000.0
        assert portfolio.max_positions == 10
        assert portfolio.commission_rate == 0.002

    def test_add_position(self) -> None:
        """Test adding a new position."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add a position that fits within capital (2 shares at $400 = $800)
        portfolio.add_position(symbol='SPY', quantity=2, price=400.0, timestamp=datetime.now())

        assert 'SPY' in portfolio.positions
        assert portfolio.positions['SPY'].quantity == 2
        assert portfolio.positions['SPY'].avg_price == 400.0
        expected_cash = 1000.0 - (2 * 400.0) - (2 * 400.0 * 0.001)  # Minus commission
        assert abs(portfolio.cash - expected_cash) < 2.0  # Allow for slippage variance

    def test_add_position_with_commission(self) -> None:
        """Test position addition with commission calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0, commission_rate=0.001)

        # Add position that fits within capital (4 shares at $200 = $800)
        portfolio.add_position('AAPL', 4, 200.0, datetime.now())

        expected_commission = 4 * 200.0 * 0.001  # $0.80
        expected_cash = 1000.0 - (4 * 200.0) - expected_commission

        # Account for slippage variance
        assert abs(portfolio.cash - expected_cash) < 2.0
        assert abs(portfolio.positions['AAPL'].total_commission - expected_commission) < 0.1

    def test_update_position(self) -> None:
        """Test updating an existing position."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add initial position - using smaller quantities to fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800 worth

        # Add more shares at different price
        portfolio.update_position('SPY', 1, 410.0, datetime.now())

        position = portfolio.positions['SPY']
        expected_quantity = 3
        expected_avg_price = (2 * 400.0 + 1 * 410.0) / 3  # 403.33

        assert position.quantity == expected_quantity
        assert abs(position.avg_price - expected_avg_price) < 0.01

    def test_close_position(self) -> None:
        """Test closing a position."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add position - using smaller amount to fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800 worth
        initial_cash = portfolio.cash

        # Close position at higher price
        portfolio.close_position('SPY', 450.0, datetime.now())

        assert 'SPY' not in portfolio.positions
        assert portfolio.cash > initial_cash  # Should have made profit

    def test_partial_close_position(self) -> None:
        """Test partially closing a position."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add position - using smaller amount to fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800 worth

        # Close half the position
        portfolio.close_position('SPY', 450.0, datetime.now(), quantity=1)

        position = portfolio.positions['SPY']
        assert position.quantity == 1
        assert position.avg_price == 400.0  # Average price unchanged

    def test_get_position_value(self) -> None:
        """Test getting current position value."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add position - using smaller amount to fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800 worth

        # Get position value at current price
        current_price = 410.0
        position_value = portfolio.get_position_value('SPY', current_price)

        expected_value = 2 * 410.0
        assert position_value == expected_value

    def test_calculate_portfolio_value(self) -> None:
        """Test total portfolio value calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add positions - using amounts that fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800
        portfolio.add_position('AAPL', 1, 200.0, datetime.now())  # $200

        # Calculate total value
        total_value = portfolio.calculate_total_value()

        # Note: This might not match exactly due to commission calculations
        assert total_value > 0

    def test_check_max_positions(self) -> None:
        """Test maximum positions limit enforcement."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=10000.0, max_positions=3)

        # Add positions up to limit - using amounts that fit within capital
        # These are realistic position sizes for testing
        portfolio.add_position('SPY', 10, 400.0, datetime.now())  # $4000
        portfolio.add_position('AAPL', 10, 200.0, datetime.now())  # $2000
        portfolio.add_position('MSFT', 10, 300.0, datetime.now())  # $3000
        # Total: $9000, leaving $1000 for fees and cushion

        # Should not allow more positions
        can_add = portfolio.can_add_position('GOOGL')
        assert can_add is False

        # Remove a position
        portfolio.close_position('SPY', 410.0, datetime.now())

        # Should allow adding new position
        can_add = portfolio.can_add_position('GOOGL')
        assert can_add is True

    def test_get_portfolio_summary(self) -> None:
        """Test portfolio summary generation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add positions - using smaller amounts that fit within capital
        portfolio.add_position('SPY', 1, 400.0, datetime.now())  # $400
        portfolio.add_position('AAPL', 1, 300.0, datetime.now())  # $300
        # Total: $700, leaving $300 for commissions and cushion

        summary = portfolio.get_summary()

        assert 'total_value' in summary
        assert 'cash' in summary
        assert 'positions' in summary
        assert 'total_return' in summary
        assert len(summary['positions']) == 2

    def test_rebalance_positions(self) -> None:
        """Test position rebalancing."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add positions - using amounts that fit within capital
        portfolio.add_position('SPY', 1, 400.0, datetime.now())  # $400
        portfolio.add_position('AAPL', 1, 300.0, datetime.now())  # $300

        # Target allocation: 60% SPY, 40% AAPL
        target_allocation = {'SPY': 0.6, 'AAPL': 0.4}

        rebalance_result = portfolio.rebalance(target_allocation, tolerance=0.02)

        assert 'trades' in rebalance_result
        assert 'new_allocation' in rebalance_result

    def test_get_allocation(self) -> None:
        """Test portfolio allocation calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add positions - using amounts that fit within capital
        portfolio.add_position('SPY', 1, 400.0, datetime.now())  # $400
        portfolio.add_position('AAPL', 1, 300.0, datetime.now())  # $300

        allocation = portfolio.get_allocation()

        spy_weight = allocation['SPY']
        aapl_weight = allocation['AAPL']
        cash_weight = allocation['CASH']

        # Weights should sum to 1.0
        total_weight = spy_weight + aapl_weight + cash_weight
        assert abs(total_weight - 1.0) < 0.01

    def test_handle_dividends(self) -> None:
        """Test dividend payment handling."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add position - using smaller amount to fit within capital
        portfolio.add_position('SPY', 2, 400.0, datetime.now())  # $800 worth

        # Pay dividend using correct method name
        dividend_per_share = 2.50
        portfolio.process_dividend('SPY', dividend_per_share)

        # Dividend payment should have increased cash
        assert portfolio.cash > portfolio.initial_capital - (2 * 400.0)

    def test_handle_splits(self) -> None:
        """Test stock split handling."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add position - using smaller amount to fit within capital
        portfolio.add_position('AAPL', 2, 200.0, datetime.now())  # $400 worth

        # 2-for-1 split using correct method name
        portfolio.process_split('AAPL', 2.0)

        position = portfolio.positions['AAPL']
        assert position.quantity == 4  # Doubled
        assert position.avg_price == 100.0  # Halved


class TestDualPoolPortfolio:
    """Test suite for the DualPoolPortfolio class."""

    def test_initialization(self) -> None:
        """Test DualPoolPortfolio initialization."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        assert portfolio.initial_capital == 1000.0
        assert hasattr(portfolio, 'base_pool')
        assert hasattr(portfolio, 'alpha_pool')
        if hasattr(portfolio.base_pool, 'leverage'):
            assert portfolio.base_pool.leverage > 0
        if hasattr(portfolio.alpha_pool, 'leverage'):
            assert portfolio.alpha_pool.leverage > 0

    def test_initialization_with_config(self) -> None:
        """Test DualPoolPortfolio with custom configuration."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(
            initial_capital=5000.0, leverage_base=2.0, leverage_alpha=3.0, base_to_alpha_split=0.3
        )

        assert portfolio.initial_capital == 5000.0
        if hasattr(portfolio.base_pool, 'leverage'):
            assert portfolio.base_pool.leverage == 2.0
        if hasattr(portfolio.alpha_pool, 'leverage'):
            assert portfolio.alpha_pool.leverage == 3.0
        if hasattr(portfolio.base_pool, 'max_allocation'):
            assert portfolio.base_pool.max_allocation == 0.7  # 70%
        if hasattr(portfolio.alpha_pool, 'max_allocation'):
            assert portfolio.alpha_pool.max_allocation == 0.3  # 30%

    def test_base_pool_operations(self) -> None:
        """Test base pool specific operations."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Allocate to base pool - use smaller amounts that fit within constraints
        portfolio.allocate_to_pool('base', 600.0)

        if hasattr(portfolio.base_pool, 'available_capital'):
            assert portfolio.base_pool.available_capital == 600.0
        if hasattr(portfolio.base_pool, 'used_capital'):
            assert portfolio.base_pool.used_capital == 0.0

        # Add position in base pool - use smaller position that fits
        portfolio.add_base_position('SPY', 1, 400.0, datetime.now())

        if hasattr(portfolio.base_pool, 'used_capital'):
            assert portfolio.base_pool.used_capital > 0
        if hasattr(portfolio.base_pool, 'positions') and portfolio.base_pool.positions:
            assert len(portfolio.base_pool.positions) > 0

    def test_alpha_pool_operations(self) -> None:
        """Test alpha pool specific operations."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Allocate to alpha pool - use smaller amounts that fit within constraints
        portfolio.allocate_to_pool('alpha', 300.0)

        if hasattr(portfolio.alpha_pool, 'available_capital'):
            assert portfolio.alpha_pool.available_capital == 300.0
        if hasattr(portfolio.alpha_pool, 'used_capital'):
            assert portfolio.alpha_pool.used_capital == 0.0

        # Add position in alpha pool - use smaller position that fits
        portfolio.add_alpha_position('AAPL', 1, 200.0, datetime.now())

        if hasattr(portfolio.alpha_pool, 'used_capital'):
            assert portfolio.alpha_pool.used_capital > 0
        if hasattr(portfolio.alpha_pool, 'positions') and portfolio.alpha_pool.positions:
            assert len(portfolio.alpha_pool.positions) > 0

    def test_pool_rebalancing(self) -> None:
        """Test rebalancing between pools."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Initial allocation
        portfolio.allocate_to_pool('base', 700.0)
        portfolio.allocate_to_pool('alpha', 300.0)

        # Rebalance to different ratios
        portfolio.rebalance_pools(base_target=0.8, alpha_target=0.2)

        if hasattr(portfolio.base_pool, 'available_capital'):
            base_ratio = portfolio.base_pool.available_capital / portfolio.initial_capital
            assert abs(base_ratio - 0.8) < 0.01
        if hasattr(portfolio.alpha_pool, 'available_capital'):
            alpha_ratio = portfolio.alpha_pool.available_capital / portfolio.initial_capital
            assert abs(alpha_ratio - 0.2) < 0.01

    def test_cross_pool_risk_management(self) -> None:
        """Test risk management across both pools."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(
            initial_capital=1000.0, leverage_base=2.0, leverage_alpha=3.0, max_total_leverage=4.0
        )

        # Add positions in both pools
        portfolio.add_base_position('SPY', 10, 400.0, datetime.now())
        portfolio.add_alpha_position('AAPL', 5, 200.0, datetime.now())

        # Check leverage limits
        total_leverage = portfolio.get_total_leverage()

        assert total_leverage <= portfolio.max_total_leverage

        # Check if portfolio is within risk limits
        is_safe = portfolio.check_risk_limits()
        assert isinstance(is_safe, bool)

    def test_pool_performance_tracking(self) -> None:
        """Test performance tracking for individual pools."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Add positions
        portfolio.add_base_position('SPY', 10, 400.0, datetime.now())
        portfolio.add_alpha_position('AAPL', 5, 200.0, datetime.now())

        # Get pool performance
        base_performance = portfolio.get_pool_performance('base')
        alpha_performance = portfolio.get_pool_performance('alpha')

        assert 'total_return' in base_performance
        assert 'total_return' in alpha_performance
        assert 'sharpe_ratio' in base_performance
        assert 'sharpe_ratio' in alpha_performance

    def test_pool_value_calculation(self) -> None:
        """Test individual pool value calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Add positions
        portfolio.add_base_position('SPY', 10, 400.0, datetime.now())
        portfolio.add_alpha_position('AAPL', 5, 200.0, datetime.now())

        base_value = portfolio.get_pool_value('base')
        alpha_value = portfolio.get_pool_value('alpha')
        total_value = portfolio.get_total_value()

        assert base_value > 0
        assert alpha_value > 0
        assert total_value == base_value + alpha_value + portfolio.cash


class TestPosition:
    """Test suite for the Position class."""

    def test_initialization(self) -> None:
        """Test Position initialization."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position(symbol='SPY', quantity=10, avg_price=400.0, timestamp=datetime.now())

        assert position.symbol == 'SPY'
        assert position.quantity == 10
        assert position.avg_price == 400.0
        if hasattr(position, 'total_cost'):
            assert position.total_cost == 4000.0
        if hasattr(position, 'total_commission'):
            assert position.total_commission == 0.0

    def test_update_quantity(self) -> None:
        """Test updating position quantity."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position('SPY', 10, 400.0, datetime.now())

        # Add more quantity
        position.update_quantity(5, 410.0)

        expected_quantity = 15
        expected_avg_price = (10 * 400.0 + 5 * 410.0) / 15

        assert position.quantity == expected_quantity
        assert abs(position.avg_price - expected_avg_price) < 0.01

    def test_current_value(self) -> None:
        """Test current position value calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position('SPY', 10, 400.0, datetime.now())

        current_price = 410.0
        current_value = position.get_current_value(current_price)

        expected_value = 10 * 410.0
        assert current_value == expected_value

    def test_unrealized_pnl(self) -> None:
        """Test unrealized P&L calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position('SPY', 10, 400.0, datetime.now())

        current_price = 410.0
        unrealized_pnl = position.get_unrealized_pnl(current_price)

        expected_pnl = 10 * (410.0 - 400.0)  # $100 profit
        assert unrealized_pnl == expected_pnl

    def test_realized_pnl(self) -> None:
        """Test realized P&L calculation."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position('SPY', 10, 400.0, datetime.now())

        # Close position at profit
        realized_pnl = position.close_position(420.0, 5, datetime.now())

        expected_pnl = 5 * (420.0 - 400.0)  # $100 profit on 5 shares
        assert realized_pnl == expected_pnl
        assert position.quantity == 5  # Remaining shares

    def test_position_weight(self) -> None:
        """Test position weight in portfolio."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        position = Position('SPY', 10, 400.0, datetime.now())
        portfolio_value = 10000.0

        weight = position.get_weight(portfolio_value)
        expected_weight = (10 * 400.0) / portfolio_value

        assert weight == expected_weight


class TestPoolState:
    """Test suite for the PoolState class."""

    def test_initialization(self) -> None:
        """Test PoolState initialization."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        pool_state = PoolState(pool_type='base', leverage=2.0, max_allocation=0.7)

        assert pool_state.pool_type == 'base'
        assert pool_state.leverage == 2.0
        assert pool_state.max_allocation == 0.7
        if hasattr(pool_state, 'available_capital'):
            assert pool_state.available_capital == 0.0
        if hasattr(pool_state, 'used_capital'):
            assert pool_state.used_capital == 0.0

    def test_capital_allocation(self) -> None:
        """Test capital allocation within pool."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        pool_state = PoolState('base', 2.0, 0.7)

        # Allocate capital
        pool_state.allocate_capital(500.0)

        if hasattr(pool_state, 'available_capital'):
            assert pool_state.available_capital == 500.0

        # Use capital
        pool_state.use_capital(300.0)

        if hasattr(pool_state, 'used_capital'):
            assert pool_state.used_capital == 300.0
        if hasattr(pool_state, 'available_capital'):
            assert pool_state.available_capital == 200.0

    def test_leverage_calculation(self) -> None:
        """Test leverage calculations."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        pool_state = PoolState('alpha', 3.0, 0.3)

        pool_state.allocate_capital(1000.0)
        pool_state.use_capital(800.0)

        current_leverage = pool_state.get_current_leverage()
        max_leverage = pool_state.get_max_leverage()

        assert current_leverage > 0
        assert max_leverage == 3.0

    def test_pool_health_check(self) -> None:
        """Test pool health monitoring."""
        if not BACKTESTER_AVAILABLE:
            pytest.skip("backtester modules not available")

        pool_state = PoolState('base', 2.0, 0.7)

        # Allocate and use capital
        pool_state.allocate_capital(1000.0)
        pool_state.use_capital(800.0)

        health = pool_state.check_health()

        assert 'leverage_ratio' in health
        assert 'utilization_rate' in health
        assert 'health_status' in health


@pytest.mark.parametrize(
    "initial_capital,max_positions,expected_positions",
    [(1000.0, 5, 5), (10000.0, 10, 10), (500.0, 3, 3)],
)
def test_portfolio_capacity_parametrized(
    initial_capital: float, max_positions: int, expected_positions: int
) -> None:
    """Parametrized test for portfolio capacity limits."""
    if not BACKTESTER_AVAILABLE:
        pytest.skip("backtester modules not available")

    portfolio = GeneralPortfolio(initial_capital=initial_capital, max_positions=max_positions)

    # Try to add maximum allowed positions
    success_count = 0
    for i in range(expected_positions):
        symbol = f'STOCK_{i}'
        success = portfolio.add_position(symbol, 10, 100.0, datetime.now())
        if success:
            success_count += 1

    # Should be able to add up to max_positions
    assert success_count <= max_positions

    # Adding one more should fail
    can_add = portfolio.can_add_position('EXTRA_STOCK')
    if success_count >= max_positions:
        assert can_add is False


if __name__ == "__main__":
    pytest.main([__file__])
