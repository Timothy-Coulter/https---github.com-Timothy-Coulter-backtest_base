"""Strategy layer for trading strategy implementations."""

from backtester.strategy.base import BaseStrategy, Signal
from backtester.strategy.moving_average import (
    DualPoolMovingAverageStrategy,
    SimpleMovingAverageStrategy,
)

__all__ = ['BaseStrategy', 'Signal', 'DualPoolMovingAverageStrategy', 'SimpleMovingAverageStrategy']
