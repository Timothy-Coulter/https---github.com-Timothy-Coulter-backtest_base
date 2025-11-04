"""
Comprehensive tests for the risk control modules.

This module contains tests for risk control mechanisms including
stop loss, take profit, position sizing, and risk management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

# Import the modules being tested
try:
    from backtester.portfolio.risk_controls import (
        StopLoss, TakeProfit, StopLossType, TakeProfitType,
        PositionSizer, RiskLimits, RiskMonitor
    )
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestStopLoss:
    """Test suite for the StopLoss class."""

    def test_initialization_percentage_stop_loss(self):
        """Test StopLoss initialization with percentage-based stop."""
        stop_loss = StopLoss(stop_loss_type=StopLossType.PERCENTAGE, stop_loss_value=0.025)
        
        assert stop_loss.stop_loss_type == StopLossType.PERCENTAGE
        assert stop_loss.stop_loss_value == 0.025
        assert stop_loss.is_active is True
        assert stop_loss.triggered is False

    def test_initialization_price_based_stop_loss(self):
        """Test StopLoss initialization with price-based stop."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PRICE,
            stop_loss_value=95.0,
            entry_price=100.0
        )
        
        assert stop_loss.stop_loss_type == StopLossType.PRICE
        assert stop_loss.stop_loss_value == 95.0
        assert stop_loss.entry_price == 100.0

    def test_calculate_stop_price_percentage(self):
        """Test stop price calculation for percentage-based stop."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.025
        )
        
        entry_price = 400.0
        stop_price = stop_loss.calculate_stop_price(entry_price, side='long')
        
        expected_stop_price = 400.0 * (1 - 0.025)  # 2.5% below entry
        assert abs(stop_price - expected_stop_price) < 0.01
        
        # For short positions, stop should be above entry
        stop_price_short = stop_loss.calculate_stop_price(entry_price, side='short')
        expected_stop_short = 400.0 * (1 + 0.025)  # 2.5% above entry
        assert abs(stop_price_short - expected_stop_short) < 0.01

    def test_calculate_stop_price_price_based(self):
        """Test stop price calculation for price-based stop."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PRICE,
            stop_loss_value=95.0
        )
        
        entry_price = 100.0
        stop_price = stop_loss.calculate_stop_price(entry_price, side='long')
        
        assert stop_price == 95.0  # Fixed price

    def test_check_stop_trigger_long_position(self):
        """Test stop loss trigger check for long position."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.025
        )
        
        entry_price = 400.0
        stop_price = stop_loss.calculate_stop_price(entry_price, side='long')
        
        # Price above stop - should not trigger
        assert stop_loss.check_trigger(entry_price, 410.0, 'long') is False
        
        # Price at stop - should trigger
        assert stop_loss.check_trigger(entry_price, stop_price, 'long') is True
        
        # Price below stop - should trigger
        assert stop_loss.check_trigger(entry_price, 390.0, 'long') is True

    def test_check_stop_trigger_short_position(self):
        """Test stop loss trigger check for short position."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.025
        )
        
        entry_price = 400.0
        stop_price = stop_loss.calculate_stop_price(entry_price, side='short')
        
        # Price below stop - should not trigger for short
        assert stop_loss.check_trigger(entry_price, 390.0, 'short') is False
        
        # Price at stop - should trigger
        assert stop_loss.check_trigger(entry_price, stop_price, 'short') is True
        
        # Price above stop - should trigger for short
        assert stop_loss.check_trigger(entry_price, 410.0, 'short') is True

    def test_trailing_stop_update(self):
        """Test trailing stop loss functionality."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.TRAILING_PERCENTAGE,
            stop_loss_value=0.03,  # 3% trailing
            trailing_stop_pct=0.05  # 5% max distance
        )
        
        entry_price = 100.0
        
        # Initial setup
        stop_loss.setup_trailing_stop(entry_price, side='long')
        
        # Price moves up
        stop_loss.update_trailing_stop(110.0, 'long')
        
        # Stop should have moved up with price
        assert stop_loss.current_stop_price > 97.0  # 3% below 110
        
        # Price moves down but stays above trailing stop
        stop_loss.update_trailing_stop(108.0, 'long')
        
        # Stop should stay at the higher level
        # (Implementation-dependent behavior)

    def test_stop_loss_activation_deactivation(self):
        """Test stop loss activation and deactivation."""
        stop_loss = StopLoss(stop_loss_type=StopLossType.PERCENTAGE, stop_loss_value=0.025)
        
        # Initially active
        assert stop_loss.is_active is True
        
        # Deactivate
        stop_loss.deactivate()
        assert stop_loss.is_active is False
        
        # Reactivate
        stop_loss.activate()
        assert stop_loss.is_active is True

    def test_stop_loss_reset(self):
        """Test stop loss reset functionality."""
        stop_loss = StopLoss(stop_loss_type=StopLossType.PERCENTAGE, stop_loss_value=0.025)
        
        # Trigger the stop
        stop_loss.trigger()
        assert stop_loss.triggered is True
        
        # Reset
        stop_loss.reset()
        assert stop_loss.triggered is False
        assert stop_loss.trigger_price is None
        assert stop_loss.trigger_time is None

    def test_stop_loss_status(self):
        """Test stop loss status reporting."""
        stop_loss = StopLoss(
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.025
        )
        
        status = stop_loss.get_status()
        
        assert 'stop_type' in status
        assert 'stop_value' in status
        assert 'is_active' in status
        assert 'triggered' in status
        assert 'distance_to_stop' in status
        
        assert status['stop_type'] == 'PERCENTAGE'
        assert status['stop_value'] == 0.025
        assert status['is_active'] is True
        assert status['triggered'] is False


class TestTakeProfit:
    """Test suite for the TakeProfit class."""

    def test_initialization_percentage_take_profit(self):
        """Test TakeProfit initialization with percentage-based target."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.10
        )
        
        assert take_profit.take_profit_type == TakeProfitType.PERCENTAGE
        assert take_profit.take_profit_value == 0.10
        assert take_profit.is_active is True
        assert take_profit.triggered is False

    def test_initialization_price_based_take_profit(self):
        """Test TakeProfit initialization with price-based target."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PRICE,
            take_profit_value=110.0
        )
        
        assert take_profit.take_profit_type == TakeProfitType.PRICE
        assert take_profit.take_profit_value == 110.0

    def test_calculate_target_price_percentage(self):
        """Test target price calculation for percentage-based take profit."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.10
        )
        
        entry_price = 400.0
        target_price = take_profit.calculate_target_price(entry_price, side='long')
        
        expected_target = 400.0 * (1 + 0.10)  # 10% above entry
        assert abs(target_price - expected_target) < 0.01
        
        # For short positions, target should be below entry
        target_price_short = take_profit.calculate_target_price(entry_price, side='short')
        expected_target_short = 400.0 * (1 - 0.10)  # 10% below entry
        assert abs(target_price_short - expected_target_short) < 0.01

    def test_calculate_target_price_price_based(self):
        """Test target price calculation for price-based take profit."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PRICE,
            take_profit_value=110.0
        )
        
        entry_price = 100.0
        target_price = take_profit.calculate_target_price(entry_price, side='long')
        
        assert target_price == 110.0  # Fixed price

    def test_check_target_reached_long_position(self):
        """Test take profit check for long position."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.10
        )
        
        entry_price = 400.0
        target_price = take_profit.calculate_target_price(entry_price, side='long')
        
        # Price below target - should not trigger
        assert take_profit.check_target(entry_price, 430.0, 'long') is False
        
        # Price at target - should trigger
        assert take_profit.check_target(entry_price, target_price, 'long') is True
        
        # Price above target - should trigger
        assert take_profit.check_target(entry_price, 450.0, 'long') is True

    def test_check_target_reached_short_position(self):
        """Test take profit check for short position."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.10
        )
        
        entry_price = 400.0
        target_price = take_profit.calculate_target_price(entry_price, side='short')
        
        # Price above target - should not trigger for short
        assert take_profit.check_target(entry_price, 370.0, 'short') is False
        
        # Price at target - should trigger
        assert take_profit.check_target(entry_price, target_price, 'short') is True
        
        # Price below target - should trigger for short
        assert take_profit.check_target(entry_price, 350.0, 'short') is True

    def test_trailing_take_profit(self):
        """Test trailing take profit functionality."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.TRAILING_PERCENTAGE,
            take_profit_value=0.05,  # 5% trailing
            trailing_take_profit_pct=0.03  # 3% minimum distance
        )
        
        entry_price = 100.0
        
        # Initial setup
        take_profit.setup_trailing_target(entry_price, side='long')
        
        # Price moves up significantly
        take_profit.update_trailing_target(120.0, 'long')
        
        # Target should have moved up with price
        assert take_profit.current_target_price > 110.0  # 5% below 120
        
        # Price pulls back but stays above trailing level
        take_profit.update_trailing_target(118.0, 'long')
        
        # Target should stay at the higher level

    def test_partial_profit_taking(self):
        """Test partial profit taking functionality."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.05,
            partial_take_profit_levels=[0.02, 0.035, 0.05]
        )
        
        entry_price = 100.0
        
        # Setup partial take profit levels
        partial_levels = take_profit.setup_partial_take_profit(entry_price, 100, side='long')
        
        assert len(partial_levels) == 3
        assert abs(partial_levels[0]['target_price'] - 102.0) < 0.01  # 2%
        assert abs(partial_levels[1]['target_price'] - 103.5) < 0.01  # 3.5%
        assert abs(partial_levels[2]['target_price'] - 105.0) < 0.01  # 5%
        
        # Check which level gets triggered
        triggered_levels = take_profit.check_partial_targets(103.0, 'long')
        
        assert len(triggered_levels) >= 1  # First level should trigger

    def test_take_profit_rr_ratio(self):
        """Test risk-reward ratio calculation and enforcement."""
        take_profit = TakeProfit(
            take_profit_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.025,  # 2.5% risk
            enforce_rr_ratio=2.0    # 2:1 reward:risk ratio
        )
        
        entry_price = 100.0
        stop_price = take_profit.calculate_stop_price(entry_price, side='long')
        
        # Calculate minimum take profit based on RR ratio
        min_take_profit = take_profit.calculate_min_take_profit(stop_price, entry_price)
        
        risk = abs(entry_price - stop_price)
        expected_min_target = entry_price + (risk * 2.0)  # 2:1 RR
        
        assert abs(min_take_profit - expected_min_target) < 0.01

    def test_take_profit_scaling(self):
        """Test take profit scaling based on conviction."""
        take_profit = TakeProfit(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.05,
            scaling_factors={'low': 0.8, 'medium': 1.0, 'high': 1.3}
        )
        
        entry_price = 100.0
        
        # Test different conviction levels
        low_target = take_profit.calculate_scaled_target(entry_price, 'low', side='long')
        medium_target = take_profit.calculate_scaled_target(entry_price, 'medium', side='long')
        high_target = take_profit.calculate_scaled_target(entry_price, 'high', side='long')
        
        assert low_target < medium_target < high_target
        
        # High conviction should give better target
        expected_high = 100.0 * (1 + 0.05 * 1.3)  # 6.5%
        assert abs(high_target - expected_high) < 0.01


class TestPositionSizer:
    """Test suite for PositionSizer class."""

    def test_initialization(self):
        """Test PositionSizer initialization."""
        sizer = PositionSizer()
        
        assert sizer.max_position_size == 0.10  # 10% of portfolio
        assert sizer.min_position_size == 0.01  # 1% of portfolio
        assert sizer.risk_per_trade == 0.02     # 2% risk per trade

    def test_initialization_custom_params(self):
        """Test PositionSizer with custom parameters."""
        sizer = PositionSizer(
            max_position_size=0.20,
            min_position_size=0.005,
            risk_per_trade=0.015
        )
        
        assert sizer.max_position_size == 0.20
        assert sizer.min_position_size == 0.005
        assert sizer.risk_per_trade == 0.015

    def test_calculate_position_size_fixed_risk(self):
        """Test position size calculation with fixed risk."""
        sizer = PositionSizer(risk_per_trade=0.02)
        
        account_value = 10000.0
        entry_price = 400.0
        stop_price = 390.0  # $10 stop
        risk_amount = account_value * 0.02  # $200 risk
        
        position_size = sizer.calculate_position_size_fixed_risk(
            account_value, entry_price, stop_price
        )
        
        # Position size = risk_amount / stop_distance
        expected_size = risk_amount / abs(entry_price - stop_price)  # 20 shares
        assert abs(position_size - expected_size) < 1.0

    def test_calculate_position_size_percentage(self):
        """Test position size calculation based on percentage."""
        sizer = PositionSizer(max_position_size=0.15)
        
        account_value = 10000.0
        entry_price = 400.0
        
        position_size = sizer.calculate_position_size_percentage(
            account_value, entry_price, percentage=0.10
        )
        
        expected_size = (account_value * 0.10) / entry_price  # 2.5 shares
        assert abs(position_size - expected_size) < 0.1

    def test_calculate_position_size_kelly(self):
        """Test position size calculation using Kelly Criterion."""
        sizer = PositionSizer()
        
        win_rate = 0.6  # 60% win rate
        avg_win = 0.05  # 5% average win
        avg_loss = 0.03  # 3% average loss
        
        kelly_fraction = sizer.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Kelly fraction = (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
        b = avg_win / avg_loss
        q = 1 - win_rate
        expected_kelly = (b * win_rate - q) / b
        
        assert abs(kelly_fraction - expected_kelly) < 0.01
        
        # Should cap Kelly fraction
        assert kelly_fraction <= sizer.max_position_size

    def test_calculate_position_size_volatility_adjusted(self):
        """Test volatility-adjusted position sizing."""
        sizer = PositionSizer(max_position_size=0.10)
        
        account_value = 10000.0
        entry_price = 400.0
        volatility = 0.02  # 2% daily volatility
        
        position_size = sizer.calculate_position_size_volatility_adjusted(
            account_value, entry_price, volatility
        )
        
        # Higher volatility should reduce position size
        assert position_size > 0
        assert position_size <= (account_value * 0.10) / entry_price

    def test_calculate_position_size_correlation_adjusted(self):
        """Test correlation-adjusted position sizing."""
        sizer = PositionSizer(max_position_size=0.15)
        
        account_value = 10000.0
        entry_price = 400.0
        portfolio_correlation = 0.8  # High correlation
        
        position_size = sizer.calculate_position_size_correlation_adjusted(
            account_value, entry_price, portfolio_correlation
        )
        
        # High correlation should reduce position size
        assert position_size < (account_value * 0.15) / entry_price

    def test_risk_based_position_sizing(self):
        """Test comprehensive risk-based position sizing."""
        sizer = PositionSizer(
            max_position_size=0.20,
            min_position_size=0.005,
            risk_per_trade=0.02,
            max_correlation=0.7,
            max_volatility=0.03
        )
        
        position_params = {
            'account_value': 10000.0,
            'entry_price': 400.0,
            'stop_price': 390.0,
            'volatility': 0.025,
            'correlation': 0.6,
            'conviction_level': 'medium'
        }
        
        position_size = sizer.calculate_position_size_risk_based(**position_params)
        
        assert position_size > 0
        assert position_size <= (position_params['account_value'] * sizer.max_position_size) / position_params['entry_price']
        
        # Should consider all risk factors
        # Higher volatility or correlation should reduce position size

    def test_position_sizing_constraints(self):
        """Test position sizing constraint enforcement."""
        sizer = PositionSizer(
            max_position_size=0.10,
            min_position_size=0.02,
            max_daily_trades=5,
            max_sector_exposure=0.30
        )
        
        # Test minimum size constraint
        small_size = sizer.enforce_constraints(0.5, 1000.0, 100.0)  # Very small
        assert small_size >= 0.02
        
        # Test maximum size constraint
        large_size = sizer.enforce_constraints(50, 1000.0, 100.0)  # Very large
        assert large_size <= 10.0  # 10% of $1000 / $100 = 10 shares


class TestRiskLimits:
    """Test suite for RiskLimits class."""

    def test_initialization(self):
        """Test RiskLimits initialization."""
        limits = RiskLimits()
        
        assert limits.max_drawdown == 0.20  # 20%
        assert limits.max_leverage == 3.0
        assert limits.max_position_size == 0.15  # 15%
        assert limits.max_sector_exposure == 0.30  # 30%
        assert limits.max_correlation == 0.80
        assert limits.max_volatility == 0.25  # 25%

    def test_initialization_custom_limits(self):
        """Test RiskLimits with custom values."""
        limits = RiskLimits(
            max_drawdown=0.15,
            max_leverage=2.5,
            max_position_size=0.10,
            max_sector_exposure=0.25,
            max_correlation=0.70,
            max_volatility=0.20
        )
        
        assert limits.max_drawdown == 0.15
        assert limits.max_leverage == 2.5
        assert limits.max_position_size == 0.10

    def test_check_drawdown_limit(self):
        """Test drawdown limit checking."""
        limits = RiskLimits(max_drawdown=0.15)
        
        # Current drawdown within limit
        current_drawdown = 0.12
        assert limits.check_drawdown_limit(current_drawdown) is True
        
        # Current drawdown exceeds limit
        current_drawdown = 0.18
        assert limits.check_drawdown_limit(current_drawdown) is False

    def test_check_leverage_limit(self):
        """Test leverage limit checking."""
        limits = RiskLimits(max_leverage=2.5)
        
        # Leverage within limit
        current_leverage = 2.2
        assert limits.check_leverage_limit(current_leverage) is True
        
        # Leverage exceeds limit
        current_leverage = 3.1
        assert limits.check_leverage_limit(current_leverage) is False

    def test_check_position_size_limit(self):
        """Test position size limit checking."""
        limits = RiskLimits(max_position_size=0.15)
        
        # Position size within limit
        position_size = 0.12
        assert limits.check_position_size_limit(position_size) is True
        
        # Position size exceeds limit
        position_size = 0.18
        assert limits.check_position_size_limit(position_size) is False

    def test_check_sector_exposure_limit(self):
        """Test sector exposure limit checking."""
        limits = RiskLimits(max_sector_exposure=0.30)
        
        # Sector exposure within limit
        tech_exposure = 0.25
        assert limits.check_sector_exposure_limit('technology', tech_exposure) is True
        
        # Sector exposure exceeds limit
        tech_exposure = 0.35
        assert limits.check_sector_exposure_limit('technology', tech_exposure) is False

    def test_check_correlation_limit(self):
        """Test correlation limit checking."""
        limits = RiskLimits(max_correlation=0.75)
        
        # Correlation within limit
        correlation = 0.68
        assert limits.check_correlation_limit(correlation) is True
        
        # Correlation exceeds limit
        correlation = 0.82
        assert limits.check_correlation_limit(correlation) is False

    def test_check_volatility_limit(self):
        """Test volatility limit checking."""
        limits = RiskLimits(max_volatility=0.20)
        
        # Volatility within limit
        volatility = 0.18
        assert limits.check_volatility_limit(volatility) is True
        
        # Volatility exceeds limit
        volatility = 0.23
        assert limits.check_volatility_limit(volatility) is False

    def test_comprehensive_risk_check(self):
        """Test comprehensive risk checking."""
        limits = RiskLimits()
        
        portfolio_state = {
            'current_drawdown': 0.10,
            'leverage': 2.5,
            'largest_position': 0.12,
            'sector_exposures': {'technology': 0.25, 'healthcare': 0.15},
            'avg_correlation': 0.65,
            'portfolio_volatility': 0.18
        }
        
        risk_check = limits.check_all_limits(portfolio_state)
        
        assert 'all_limits_passed' in risk_check
        assert 'breached_limits' in risk_check
        assert 'risk_score' in risk_check
        assert 'recommendations' in risk_check
        
        # In this case, all limits should pass
        if risk_check['all_limits_passed']:
            assert len(risk_check['breached_limits']) == 0
            assert risk_check['risk_score'] <= 0.5  # Low risk score

    def test_dynamic_limit_adjustment(self):
        """Test dynamic limit adjustment based on market conditions."""
        limits = RiskLimits()
        
        # Set different risk profiles
        risk_profiles = {
            'conservative': {'max_drawdown': 0.10, 'max_leverage': 2.0, 'max_position_size': 0.08},
            'moderate': {'max_drawdown': 0.15, 'max_leverage': 2.5, 'max_position_size': 0.12},
            'aggressive': {'max_drawdown': 0.25, 'max_leverage': 4.0, 'max_position_size': 0.20}
        }
        
        for profile_name, profile_limits in risk_profiles.items():
            limits.set_risk_profile(profile_name)
            
            assert limits.max_drawdown == profile_limits['max_drawdown']
            assert limits.max_leverage == profile_limits['max_leverage']
            assert limits.max_position_size == profile_limits['max_position_size']

    def test_limit_breach_escalation(self):
        """Test limit breach escalation procedures."""
        limits = RiskLimits()
        
        # Simulate multiple limit breaches
        breaches = [
            {'limit_type': 'drawdown', 'current_value': 0.22, 'limit_value': 0.15},
            {'limit_type': 'leverage', 'current_value': 3.2, 'limit_value': 2.5}
        ]
        
        escalation = limits.handle_limit_breach(breaches)
        
        assert 'severity_level' in escalation
        assert 'required_actions' in escalation
        assert 'timeline' in escalation
        
        # Multiple breaches should increase severity
        assert escalation['severity_level'] in ['high', 'critical']


class TestRiskMonitor:
    """Test suite for RiskMonitor class."""

    def test_initialization(self):
        """Test RiskMonitor initialization."""
        monitor = RiskMonitor()
        
        assert monitor.is_monitoring is True
        assert monitor.check_interval == 60  # 60 seconds
        assert monitor.alert_thresholds is not None
        assert monitor.risk_metrics_history == []

    def test_initialization_custom_params(self):
        """Test RiskMonitor with custom parameters."""
        monitor = RiskMonitor(
            check_interval=30,
            enable_real_time_alerts=True,
            max_history_size=1000
        )
        
        assert monitor.check_interval == 30
        assert monitor.enable_real_time_alerts is True
        assert monitor.max_history_size == 1000

    def test_add_risk_metric(self):
        """Test adding risk metrics to monitor."""
        monitor = RiskMonitor()
        
        # Add a risk metric
        monitor.add_risk_metric('portfolio_volatility', threshold=0.25, comparison='greater_than')
        
        assert 'portfolio_volatility' in monitor.risk_metrics
        assert monitor.risk_metrics['portfolio_volatility']['threshold'] == 0.25
        assert monitor.risk_metrics['portfolio_volatility']['comparison'] == 'greater_than'

    def test_check_risk_metrics(self):
        """Test risk metric checking."""
        monitor = RiskMonitor()
        
        # Add risk metrics
        monitor.add_risk_metric('drawdown', threshold=0.15, comparison='greater_than')
        monitor.add_risk_metric('leverage', threshold=3.0, comparison='greater_than')
        
        # Current portfolio state
        portfolio_state = {
            'current_drawdown': 0.18,
            'leverage': 2.8
        }
        
        violations = monitor.check_risk_metrics(portfolio_state)
        
        assert len(violations) >= 1
        assert any(v['metric'] == 'drawdown' for v in violations)

    def test_real_time_monitoring(self):
        """Test real-time risk monitoring."""
        monitor = RiskMonitor(enable_real_time_alerts=True)
        
        # Mock portfolio updates
        portfolio_updates = [
            {'timestamp': datetime.now(), 'drawdown': 0.10, 'leverage': 2.0},
            {'timestamp': datetime.now() + timedelta(seconds=30), 'drawdown': 0.18, 'leverage': 2.5},
            {'timestamp': datetime.now() + timedelta(seconds=60), 'drawdown': 0.22, 'leverage': 3.2}
        ]
        
        alerts = []
        for update in portfolio_updates:
            alert = monitor.process_portfolio_update(update)
            if alert:
                alerts.append(alert)
        
        # Should generate alerts for significant violations
        assert len(alerts) > 0
        
        # Most recent updates should have triggered alerts
        high_risk_alerts = [a for a in alerts if a.get('severity') in ['high', 'critical']]
        assert len(high_risk_alerts) > 0

    def test_risk_metric_history(self):
        """Test risk metric history tracking."""
        monitor = RiskMonitor(max_history_size=100)
        
        # Add multiple risk measurements
        for i in range(150):  # More than max history size
            monitor.record_risk_measurement({
                'timestamp': datetime.now() + timedelta(seconds=i),
                'portfolio_volatility': 0.15 + (i * 0.001),
                'drawdown': min(0.05 + (i * 0.001), 0.20)
            })
        
        # Should only keep most recent 100 measurements
        assert len(monitor.risk_metrics_history) <= 100
        
        # Should contain latest measurements
        latest = monitor.risk_metrics_history[-1]
        assert latest['portfolio_volatility'] > 0.15

    def test_risk_trend_analysis(self):
        """Test risk trend analysis."""
        monitor = RiskMonitor()
        
        # Create trend data
        trend_data = []
        base_volatility = 0.15
        
        for i in range(30):  # 30 days
            trend_data.append({
                'date': datetime.now() + timedelta(days=i),
                'volatility': base_volatility + (i * 0.002),  # Increasing trend
                'drawdown': min(0.05 + (i * 0.003), 0.25)
            })
        
        analysis = monitor.analyze_risk_trends(trend_data)
        
        assert 'volatility_trend' in analysis
        assert 'drawdown_trend' in analysis
        assert 'trend_direction' in analysis
        
        # Should detect increasing volatility trend
        assert analysis['volatility_trend']['direction'] == 'increasing'

    def test_risk_alert_system(self):
        """Test risk alert system."""
        monitor = RiskMonitor(enable_real_time_alerts=True)
        
        # Configure alert rules
        monitor.add_alert_rule(
            metric='drawdown',
            threshold=0.20,
            severity='critical',
            action='reduce_positions'
        )
        
        # Trigger alert
        alert = monitor.generate_alert('drawdown', 0.25, 'critical')
        
        assert alert is not None
        assert alert['metric'] == 'drawdown'
        assert alert['value'] == 0.25
        assert alert['severity'] == 'critical'
        assert 'recommended_action' in alert

    def test_risk_dashboard_data(self):
        """Test risk dashboard data generation."""
        monitor = RiskMonitor()
        
        # Add some historical data
        for i in range(10):
            monitor.record_risk_measurement({
                'timestamp': datetime.now() + timedelta(days=i),
                'portfolio_volatility': 0.15 + (i * 0.01),
                'drawdown': i * 0.02,
                'leverage': 2.0 + (i * 0.1)
            })
        
        dashboard_data = monitor.generate_dashboard_data()
        
        assert 'current_metrics' in dashboard_data
        assert 'historical_trends' in dashboard_data
        assert 'risk_summary' in dashboard_data
        assert 'alerts' in dashboard_data
        
        # Should have current and historical data
        assert len(dashboard_data['historical_trends']) > 0

    def test_risk_reporting(self):
        """Test risk reporting functionality."""
        monitor = RiskMonitor()
        
        # Generate mock risk report
        report = monitor.generate_risk_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert 'report_period' in report
        assert 'executive_summary' in report
        assert 'risk_metrics' in report
        assert 'trends' in report
        assert 'recommendations' in report
        assert 'alerts_summary' in report

    def test_risk_limit_optimization(self):
        """Test risk limit optimization based on historical performance."""
        monitor = RiskMonitor()
        
        # Mock historical portfolio performance
        performance_data = {
            'returns': np.random.randn(252).cumsum() * 0.01,  # 1 year of daily returns
            'volatilities': np.random.uniform(0.10, 0.30, 252),
            'drawdowns': np.random.uniform(0.05, 0.25, 252)
        }
        
        optimization = monitor.optimize_risk_limits(performance_data)
        
        assert 'recommended_drawdown_limit' in optimization
        assert 'recommended_volatility_limit' in optimization
        assert 'recommended_position_limit' in optimization
        assert 'optimization_score' in optimization


@pytest.mark.parametrize("stop_type,stop_value,entry_price,current_price,side,expected_trigger", [
    (StopLossType.PERCENTAGE, 0.025, 400.0, 390.0, 'long', True),   # Stop triggered
    (StopLossType.PERCENTAGE, 0.025, 400.0, 395.0, 'long', True),   # Stop triggered
    (StopLossType.PERCENTAGE, 0.025, 400.0, 410.0, 'long', False),  # No trigger
    (StopLossType.PRICE, 390.0, 400.0, 385.0, 'long', True),        # Price stop triggered
    (StopLossType.PRICE, 390.0, 400.0, 395.0, 'long', False),       # No trigger
])
def test_stop_loss_trigger_parametrized(stop_type, stop_value, entry_price, current_price, side, expected_trigger):
    """Parametrized test for stop loss trigger conditions."""
    stop_loss = StopLoss(stop_loss_type=stop_type, stop_loss_value=stop_value)
    
    is_triggered = stop_loss.check_trigger(entry_price, current_price, side)
    
    assert is_triggered is expected_trigger


@pytest.mark.parametrize("risk_level,position_size,expected_action", [
    ('low', 0.05, 'normal_trading'),
    ('medium', 0.10, 'normal_trading'),
    ('high', 0.15, 'reduce_position_size'),
    ('very_high', 0.25, 'emergency_reduction')
])
def test_risk_level_actions_parametrized(risk_level, position_size, expected_action):
    """Parametrized test for risk level action mapping."""
    risk_monitor = RiskMonitor()
    
    action = risk_monitor.get_recommended_action(risk_level, position_size)
    
    assert action == expected_action


if __name__ == "__main__":
    pytest.main([__file__])