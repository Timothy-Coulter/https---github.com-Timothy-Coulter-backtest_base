"""Risk Control Manager.

This module provides a manager for coordinating multiple stop loss and take profit mechanisms,
position sizing, and risk limits into a unified risk control system.
"""

import logging
from typing import Any

import pandas as pd

from .component_configs.comprehensive_risk_config import ComprehensiveRiskConfig
from .position_sizing import PositionSizer
from .risk_limits import RiskLimits
from .risk_monitor import RiskMonitor
from .stop_loss import StopLoss
from .take_profit import TakeProfit


class RiskControlManager:
    """Manager for coordinating multiple risk control mechanisms."""

    def __init__(
        self,
        config: ComprehensiveRiskConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the risk control manager.

        Args:
            config: ComprehensiveRiskConfig with risk control parameters
            logger: Optional logger instance
        """
        self.config: ComprehensiveRiskConfig = config or ComprehensiveRiskConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # Initialize risk control components
        self.stop_loss: StopLoss | None = None
        self.take_profit: TakeProfit | None = None
        self.position_sizer: PositionSizer | None = None
        self.risk_limits: RiskLimits | None = None
        self.risk_monitor: RiskMonitor | None = None

        # Initialize components based on config
        self._initialize_components()

        # State tracking
        self.current_positions: dict[str, Any] = {}
        self.risk_signals_history: list[dict[str, Any]] = []

    def _initialize_components(self) -> None:
        """Initialize all risk control components based on config."""
        # Initialize stop loss
        if self.config.stop_loss_config:
            self.stop_loss = StopLoss(self.config.stop_loss_config, self.logger)

        # Initialize take profit
        if self.config.take_profit_config:
            self.take_profit = TakeProfit(self.config.take_profit_config, self.logger)

        # Initialize position sizer
        if self.config.position_sizing_config:
            self.position_sizer = PositionSizer(self.config.position_sizing_config, self.logger)

        # Initialize risk limits
        if self.config.risk_limits_config:
            self.risk_limits = RiskLimits(self.config.risk_limits_config, self.logger)

        # Initialize risk monitor
        if self.config.risk_monitoring_config:
            self.risk_monitor = RiskMonitor(self.config.risk_monitoring_config, self.logger)

    def initialize_position(
        self, symbol: str, entry_price: float, quantity: float, timestamp: pd.Timestamp
    ) -> None:
        """Initialize all risk controls for a position.

        Args:
            symbol: Position symbol
            entry_price: Entry price
            quantity: Position quantity
            timestamp: Entry timestamp
        """
        # Store position information
        self.current_positions[symbol] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_timestamp': timestamp,
        }

        # Initialize stop loss
        if self.stop_loss:
            self.stop_loss.initialize_position(entry_price, timestamp)

        # Initialize take profit
        if self.take_profit:
            self.take_profit.initialize_position(entry_price, timestamp)

        self.logger.info(
            f"Risk controls initialized for {symbol}: "
            f"entry={entry_price:.4f}, quantity={quantity:.2f}"
        )

    def update_position(
        self, symbol: str, current_price: float, timestamp: pd.Timestamp
    ) -> dict[str, Any]:
        """Update all risk controls and check for triggers.

        Args:
            symbol: Position symbol
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            Dictionary with comprehensive risk control status
        """
        if symbol not in self.current_positions:
            return {
                'triggered': False,
                'action': 'NONE',
                'reason': 'Position not found',
            }

        result = {
            'symbol': symbol,
            'triggered': False,
            'action': 'NONE',
            'reason': 'No risk controls triggered',
            'exit_price': None,
            'stop_loss': None,
            'take_profit': None,
        }

        # Update stop loss
        if self.stop_loss:
            stop_result = self.stop_loss.update(current_price, timestamp)
            result['stop_loss'] = stop_result

            if stop_result['triggered']:
                result.update(
                    {
                        'triggered': True,
                        'action': stop_result['action'],
                        'reason': stop_result['reason'],
                        'exit_price': stop_result.get('exit_price', current_price),
                    }
                )
                return result

        # Update take profit (only if stop loss not triggered)
        if self.take_profit and not result['triggered']:
            profit_result = self.take_profit.update(current_price, timestamp)
            result['take_profit'] = profit_result

            if profit_result['triggered']:
                result.update(
                    {
                        'triggered': True,
                        'action': profit_result['action'],
                        'reason': profit_result['reason'],
                        'exit_price': profit_result.get('exit_price', current_price),
                    }
                )
                return result

        return result

    def calculate_position_size(
        self,
        symbol: str,
        account_value: float,
        entry_price: float,
        stop_price: float | None = None,
        volatility: float = 0.0,
        conviction: float = 1.0,
    ) -> tuple[float, float]:
        """Calculate optimal position size based on risk parameters.

        Args:
            symbol: Asset symbol
            account_value: Current account value
            entry_price: Planned entry price
            stop_price: Optional stop loss price
            volatility: Current market volatility
            conviction: Signal conviction factor

        Returns:
            Tuple of (position_size, risk_amount)
        """
        if not self.position_sizer:
            # Fallback calculation if position sizer not configured
            max_position_value = account_value * self.config.max_position_size
            position_size = max_position_value / entry_price
            return position_size, 0.0

        # Use position sizer for calculation
        if stop_price:
            position_size = self.position_sizer.calculate_position_size_fixed_risk(
                account_value, entry_price, stop_price
            )
            risk_amount = (
                account_value * self.config.position_sizing_config.risk_per_trade
                if self.config.position_sizing_config
                else account_value * 0.02
            )
        else:
            position_size = self.position_sizer.calculate_position_size(
                account_value, volatility, conviction, account_value, entry_price
            )
            risk_amount = position_size * entry_price

        return position_size, risk_amount

    def check_portfolio_risk(
        self, portfolio_value: float, positions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Check overall portfolio risk levels.

        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of current positions

        Returns:
            Dictionary with portfolio risk analysis
        """
        result: dict[str, Any] = {
            'portfolio_value': portfolio_value,
            'risk_level': 'LOW',
            'violations': [],
            'recommendations': [],
        }

        # Check position concentration
        if positions:
            position_sizes = [
                abs(pos.get('market_value', 0) / portfolio_value)
                for pos in positions.values()
                if pos.get('active', False)
            ]

            max_position_size = max(position_sizes) if position_sizes else 0
            if max_position_size > self.config.max_position_size:
                result['violations'].append(f"Max position size exceeded: {max_position_size:.2%}")
                result['risk_level'] = 'HIGH'

        # Check leverage if available
        if hasattr(self, 'current_leverage') and self.current_leverage > self.config.max_leverage:
            result['violations'].append(f"Max leverage exceeded: {self.current_leverage:.2f}x")
            result['risk_level'] = 'HIGH'

        # Generate recommendations
        if result['violations']:
            result['recommendations'].extend(
                [
                    'Reduce position sizes',
                    'Review risk limits',
                    'Consider increasing cash position',
                ]
            )
        else:
            result['recommendations'].append('Risk levels within acceptable range')

        return result

    def get_position_risk_status(self, symbol: str) -> dict[str, Any]:
        """Get comprehensive risk status for a position.

        Args:
            symbol: Position symbol

        Returns:
            Dictionary with position risk status
        """
        status = {
            'symbol': symbol,
            'has_stop_loss': self.stop_loss is not None,
            'has_take_profit': self.take_profit is not None,
            'stop_loss_status': None,
            'take_profit_status': None,
        }

        if self.stop_loss:
            status['stop_loss_status'] = self.stop_loss.get_status()

        if self.take_profit:
            status['take_profit_status'] = self.take_profit.get_status()

        return status

    def get_portfolio_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio risk summary.

        Returns:
            Dictionary with portfolio risk summary
        """
        summary: dict[str, Any] = {
            'total_positions': len(self.current_positions),
            'positions_with_stop_loss': 0,
            'positions_with_take_profit': 0,
            'risk_monitor_active': self.risk_monitor is not None,
            'risk_limits_configured': self.risk_limits is not None,
        }

        # Count positions with risk controls
        for _symbol in self.current_positions:
            if self.stop_loss:
                summary['positions_with_stop_loss'] += 1
            if self.take_profit:
                summary['positions_with_take_profit'] += 1

        # Add risk monitor status
        if self.risk_monitor:
            summary['monitoring_status'] = self.risk_monitor.get_monitoring_status()

        # Add risk limits summary
        if self.risk_limits:
            summary['limits_summary'] = self.risk_limits.get_limits_summary()

        return summary

    def reset_position(self, symbol: str) -> None:
        """Reset risk controls for a position.

        Args:
            symbol: Position symbol
        """
        if symbol in self.current_positions:
            del self.current_positions[symbol]

        if self.stop_loss:
            self.stop_loss.reset()

        if self.take_profit:
            self.take_profit.reset()

        self.logger.info(f"Risk controls reset for position {symbol}")

    def reset_all_positions(self) -> None:
        """Reset all risk controls."""
        self.current_positions.clear()

        if self.stop_loss:
            self.stop_loss.reset()

        if self.take_profit:
            self.take_profit.reset()

        self.logger.info("All risk controls reset")

    def update_risk_limits(
        self, positions: dict[str, dict[str, Any]], sector_mapping: dict[str, str] | None = None
    ) -> None:
        """Update risk limits based on current positions.

        Args:
            positions: Dictionary of current positions
            sector_mapping: Optional mapping of symbols to sectors
        """
        if self.risk_limits:
            self.risk_limits.update_exposures(positions, sector_mapping)

    def process_portfolio_update(self, update: dict[str, Any]) -> dict[str, Any] | None:
        """Process portfolio update for risk monitoring.

        Args:
            update: Portfolio update information

        Returns:
            Alert information if triggered
        """
        if self.risk_monitor:
            return self.risk_monitor.process_portfolio_update(update)
        return None

    def add_risk_signal(self, signal: dict[str, Any]) -> None:
        """Add a risk signal to history.

        Args:
            signal: Risk signal information
        """
        self.risk_signals_history.append(
            {
                'timestamp': pd.Timestamp.now(),
                'signal': signal,
            }
        )

        self.logger.info(
            f"Risk signal: {signal.get('action', 'UNKNOWN')} - {signal.get('reason', 'No reason')}"
        )

    def get_risk_signals_summary(self) -> dict[str, Any]:
        """Get summary of risk signals history.

        Returns:
            Dictionary with risk signals summary
        """
        return {
            'total_signals': len(self.risk_signals_history),
            'recent_signals': self.risk_signals_history[-10:] if self.risk_signals_history else [],
        }
