"""Risk Signals and Actions.

This module provides enums and data structures for risk signals and actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class RiskAction(str, Enum):
    """Risk management action types."""

    HOLD = "HOLD"
    REDUCE_POSITION = "REDUCE_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    INCREASE_POSITION = "INCREASE_POSITION"
    EMERGENCY_HALT = "EMERGENCY_HALT"
    REVIEW_POSITIONS = "REVIEW_POSITIONS"


@dataclass
class RiskSignal:
    """Risk management signal."""

    action: RiskAction
    reason: str
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())


@dataclass
class RiskMetric:
    """Risk metric with threshold and status tracking."""

    name: str
    value: float
    unit: str
    threshold: float
    status: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())

    def is_within_threshold(self) -> bool:
        """Check if metric is within acceptable threshold."""
        if self.unit == 'percentage':
            return self.value <= self.threshold
        else:
            return abs(self.value) <= abs(self.threshold)

    def __lt__(self, other: 'RiskMetric') -> bool:
        """For percentage metrics, less negative is better."""
        if self.unit == 'percentage' and other.unit == 'percentage':
            return self.value > other.value  # Less negative is "less than"
        return self.value < other.value

    def __gt__(self, other: 'RiskMetric') -> bool:
        """Greater than comparison for RiskMetric.

        For percentage thresholds, more negative values are considered "greater than"
        because they represent higher risk limits.

        Args:
            other: Another RiskMetric to compare with

        Returns:
            bool: True if this threshold is greater than other
        """
        if self.unit == 'percentage' and other.unit == 'percentage':
            return self.value < other.value  # More negative is "greater than"
        return self.value > other.value


@dataclass
class RiskLimit:
    """Risk limit configuration."""

    limit_type: str
    threshold: float
    severity: str
    description: str | None = None
    is_active: bool = True

    def is_breached(self, current_value: float) -> bool:
        """Check if limit is breached.

        Args:
            current_value: Current metric value

        Returns:
            True if limit is breached
        """
        if not self.is_active:
            return False

        if self.limit_type in ['position_size', 'leverage', 'drawdown']:
            return current_value > self.threshold
        elif self.limit_type in ['var', 'cvar']:
            return abs(current_value) > abs(self.threshold)
        else:
            return current_value > self.threshold


@dataclass
class RiskAlert:
    """Risk alert for limit breaches."""

    alert_type: str
    severity: str
    message: str
    affected_symbol: str | None = None
    current_value: float | None = None
    limit_value: float | None = None
    timestamp: str | None = None
    escalated: bool = False

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())

    def escalate(self, new_severity: str) -> None:
        """Escalate alert severity.

        Args:
            new_severity: New severity level
        """
        self.severity = new_severity
        self.escalated = True
