"""Quick test of the modular backtester.

This script provides a simple test of the modular backtester functionality.
"""

import sys
from typing import Any

sys.path.append('..')
from backtester.main import get_data, run_modular_backtest

# Load test data
print('Loading test data...')
data = get_data('SPY', '2020-01-01', '2024-01-01', '1mo')
print(f'Loaded {len(data)} records')

# Test modular system
print('Testing modular backtester...')
result: dict[str, Any] = run_modular_backtest(
    data=data,
    leverage_base=2.0,
    leverage_alpha=3.0,
    base_to_alpha_split=0.2,
    alpha_to_base_split=0.2,
    stop_loss_base=0.025,
    stop_loss_alpha=0.025,
    take_profit_target=0.10,
    initial_capital=100.0,
)

print('✅ Modular backtester test completed successfully!')
print(f'Total Return: {result["performance"]["total_return"]:.2%}')
print(f'Sharpe Ratio: {result["performance"]["sharpe_ratio"]:.3f}')
print(f'Max Drawdown: {result["performance"]["max_drawdown"]:.2%}')
