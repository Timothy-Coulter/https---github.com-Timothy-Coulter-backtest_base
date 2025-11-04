"""Strategy layer for trading strategy implementations."""

from .base import BaseStrategy, Signal
from .moving_average import DualPoolMovingAverageStrategy, SimpleMovingAverageStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'DualPoolMovingAverageStrategy',
    'SimpleMovingAverageStrategy'
]