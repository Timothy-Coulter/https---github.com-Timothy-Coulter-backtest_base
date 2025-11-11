"""Portfolio layer for portfolio management and risk handling."""

from .base_portfolio import BasePortfolio
from .dual_pool_portfolio import DualPoolPortfolio
from .general_portfolio import GeneralPortfolio
from .pool_state import PoolState
from .position import Position

__all__ = [
    'BasePortfolio',
    'Position',
    'PoolState',
    'GeneralPortfolio',
    'DualPoolPortfolio',
]

# For backward compatibility, also provide the original class names
DualPoolPortfolio.__name__ = 'DualPoolPortfolio'
GeneralPortfolio.__name__ = 'GeneralPortfolio'

# Type aliases for convenience
PortfolioType = BasePortfolio
