import numpy as np
import pandas as pd
import yfinance as yf

def get_data(ticker, start_date="1990-01-01", end_date="2025-01-01", interval="1mo"):
    
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = pd.DataFrame({
                "Close": data["Close"][ticker],
                "Low": data["Low"][ticker],
                "High": data["High"][ticker]
            })
    else:
        data = data[["Close", "Low", "High"]]
    data = data.dropna()

    return data

def run_portfolio_simulation(
    data,
    initial_capital=100,
    leverage_base=1,
    leverage_alpha=3,
    base_to_alpha_split=0.2,
    alpha_to_base_split=0.2,
    stop_loss_base=0.025,
    stop_loss_alpha=0.025,
    take_profit_target=0.10,
    enable_take_profit=True,
    commission_rate=0.001,
    interest_rate_daily=0.00025,
    spread_rate=0.0002,
    slippage_std=0.0005,
    maintenance_margin=0.5,
    min_trade_size=0.0,
    funding_enabled=True,
    tax_rate=0.45,
    seed=42
):
    """
    Run a portfolio simulation with two pools: base and alpha with different leverage.
    """
    np.random.seed(seed)

    # Validate data
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be None or empty")
    required_columns = ["Close", "Low", "High"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Data must contain '{col}' column")

    # Initialize portfolio
    portfolio = {
        "base_pool": [initial_capital * 0.5],
        "alpha_pool": [initial_capital * 0.5],
        "base_active": True,
        "alpha_active": True,
        "base_entry_price": data["Close"].iloc[0],
        "alpha_entry_price": data["Close"].iloc[0],
        "cumulative_tax": 0.0,
        "portfolio_values": [initial_capital],
        "tax_loss_carryforward": 0.0
    }
    trade_log = []

    price = data["Close"].values
    lows = data["Low"].values
    highs = data["High"].values

    yearly_base_gain = 0.0
    yearly_alpha_gain = 0.0
    current_year = data.index[0].year

    for i in range(len(price) - 1):
        current_price = price[i]
        next_price = price[i + 1]
        day_low = lows[i + 1]
        day_high = highs[i + 1]

        price_return = (next_price - current_price) / current_price

        # Time delta in days for interest
        time_diff = (data.index[i + 1] - data.index[i]).days

        # Check if trade is significant enough
        trade_significant = abs(price_return) >= min_trade_size

        # Apply spread (cost on both sides) and slippage (random execution cost)
        spread_cost = spread_rate  # Fixed cost per trade
        slippage = np.random.normal(0, slippage_std)

        # Effective return after costs (spread always reduces gains/increases losses)
        price_return_after_costs = price_return - spread_cost - slippage if trade_significant else 0

        # Initialize gains and exit reasons
        base_gain = 0
        alpha_gain = 0
        base_exit_reason = None
        alpha_exit_reason = None
        base_exit_price = None
        alpha_exit_price = None

        # BASE POOL LOGIC
        if portfolio["base_active"] and portfolio["base_pool"][-1] > 0:
            # Calculate position value
            base_position_value = portfolio["base_pool"][-1] * leverage_base

            # Check stop-loss using intraday low
            stop_price_base = portfolio["base_entry_price"] * (1 - stop_loss_base)
            if day_low <= stop_price_base:
                # Exit at stop price
                actual_return = (stop_price_base - portfolio["base_entry_price"]) / portfolio["base_entry_price"]
                base_gain = actual_return * base_position_value - spread_cost * base_position_value
                portfolio["base_active"] = False
                base_exit_reason = "STOP_LOSS"
                base_exit_price = stop_price_base
            # Check take-profit using intraday high
            elif enable_take_profit:
                tp_price_base = portfolio["base_entry_price"] * (1 + take_profit_target)
                if day_high >= tp_price_base:
                    # Exit at take-profit price
                    actual_return = (tp_price_base - portfolio["base_entry_price"]) / portfolio["base_entry_price"]
                    base_gain = actual_return * base_position_value - spread_cost * base_position_value
                    portfolio["base_active"] = False
                    base_exit_reason = "TAKE_PROFIT"
                    base_exit_price = tp_price_base

            # If still active, calculate normal P&L
            if portfolio["base_active"]:
                base_gain = price_return_after_costs * base_position_value

                # Check maintenance margin (equity / position_value)
                temp_base_equity = portfolio["base_pool"][-1] + base_gain
                if leverage_base > 1:
                    equity_ratio = temp_base_equity / base_position_value
                    if equity_ratio < maintenance_margin:
                        base_gain = -portfolio["base_pool"][-1]  # Total loss
                        portfolio["base_active"] = False
                        base_exit_reason = "LIQUIDATION"
                        base_exit_price = current_price

        # ALPHA POOL LOGIC
        if portfolio["alpha_active"] and portfolio["alpha_pool"][-1] > 0:
            # Calculate position value
            alpha_position_value = portfolio["alpha_pool"][-1] * leverage_alpha

            # Check stop-loss using intraday low
            stop_price_alpha = portfolio["alpha_entry_price"] * (1 - stop_loss_alpha)
            if day_low <= stop_price_alpha:
                # Exit at stop price
                actual_return = (stop_price_alpha - portfolio["alpha_entry_price"]) / portfolio["alpha_entry_price"]
                alpha_gain = actual_return * alpha_position_value - spread_cost * alpha_position_value
                portfolio["alpha_active"] = False
                alpha_exit_reason = "STOP_LOSS"
                alpha_exit_price = stop_price_alpha
            # Check take-profit using intraday high
            elif enable_take_profit:
                tp_price_alpha = portfolio["alpha_entry_price"] * (1 + take_profit_target)
                if day_high >= tp_price_alpha:
                    # Exit at take-profit price
                    actual_return = (tp_price_alpha - portfolio["alpha_entry_price"]) / portfolio["alpha_entry_price"]
                    alpha_gain = actual_return * alpha_position_value - spread_cost * alpha_position_value
                    portfolio["alpha_active"] = False
                    alpha_exit_reason = "TAKE_PROFIT"
                    alpha_exit_price = tp_price_alpha

            # If still active, calculate normal P&L
            if portfolio["alpha_active"]:
                alpha_gain = price_return_after_costs * alpha_position_value

                # Check maintenance margin
                temp_alpha_equity = portfolio["alpha_pool"][-1] + alpha_gain
                if leverage_alpha > 1:
                    equity_ratio = temp_alpha_equity / alpha_position_value
                    if equity_ratio < maintenance_margin:
                        alpha_gain = -portfolio["alpha_pool"][-1]  # Total loss
                        portfolio["alpha_active"] = False
                        alpha_exit_reason = "LIQUIDATION"
                        alpha_exit_price = current_price

        # Calculate fees
        base_fee = 0
        alpha_fee = 0

        if portfolio["base_active"] and portfolio["base_pool"][-1] > 0:
            # Commission on position value
            base_commission = commission_rate * portfolio["base_pool"][-1] * leverage_base if trade_significant else 0
            # Interest on borrowed amount
            base_interest = 0
            if funding_enabled and leverage_base > 1:
                borrowed_amount = portfolio["base_pool"][-1] * (leverage_base - 1)
                base_interest = interest_rate_daily * borrowed_amount * time_diff
            base_fee = base_commission + base_interest

        if portfolio["alpha_active"] and portfolio["alpha_pool"][-1] > 0:
            # Commission on position value
            alpha_commission = commission_rate * portfolio["alpha_pool"][-1] * leverage_alpha if trade_significant else 0
            # Interest on borrowed amount
            alpha_interest = 0
            if funding_enabled and leverage_alpha > 1:
                borrowed_amount = portfolio["alpha_pool"][-1] * (leverage_alpha - 1)
                alpha_interest = interest_rate_daily * borrowed_amount * time_diff
            alpha_fee = alpha_commission + alpha_interest

        # Redistribute gains (only positive gains, only when both pools are active)
        alpha_to_base = 0
        base_to_alpha = 0
        if price_return_after_costs > 0 and trade_significant:
            if portfolio["alpha_active"] and alpha_gain > 0:
                alpha_to_base = alpha_to_base_split * alpha_gain
            if portfolio["base_active"] and base_gain > 0:
                base_to_alpha = base_to_alpha_split * base_gain

        # Update pools
        new_base = portfolio["base_pool"][-1] + base_gain + alpha_to_base - base_to_alpha - base_fee
        new_alpha = portfolio["alpha_pool"][-1] + alpha_gain + base_to_alpha - alpha_to_base - alpha_fee

        # Handle bankruptcy
        if new_base < 0:
            new_base = 0
            portfolio["base_active"] = False
            if not base_exit_reason:
                base_exit_reason = "BANKRUPTCY"

        if new_alpha < 0:
            new_alpha = 0
            portfolio["alpha_active"] = False
            if not alpha_exit_reason:
                alpha_exit_reason = "BANKRUPTCY"

        # Track yearly gains for tax
        yearly_base_gain += base_gain + alpha_to_base - base_to_alpha - base_fee
        yearly_alpha_gain += alpha_gain + base_to_alpha - alpha_to_base - alpha_fee

        # Calculate and apply tax at year end
        step_year = data.index[i + 1].year
        if step_year != current_year:
            yearly_total_gain = yearly_base_gain + yearly_alpha_gain

            # Apply loss carryforward
            taxable_gain = yearly_total_gain + portfolio["tax_loss_carryforward"]

            if taxable_gain > 0:
                tax = taxable_gain * tax_rate
                portfolio["cumulative_tax"] += tax

                # Deduct tax proportionally from pools based on their contribution to positive gains
                base_positive = max(yearly_base_gain, 0)
                alpha_positive = max(yearly_alpha_gain, 0)
                total_positive = base_positive + alpha_positive

                if total_positive > 0:
                    base_tax_share = (base_positive / total_positive) * tax
                    alpha_tax_share = (alpha_positive / total_positive) * tax
                    new_base -= base_tax_share
                    new_alpha -= alpha_tax_share

                portfolio["tax_loss_carryforward"] = 0
            else:
                # Carry forward losses
                portfolio["tax_loss_carryforward"] = taxable_gain

            yearly_base_gain = 0.0
            yearly_alpha_gain = 0.0
            current_year = step_year

        # Log trades if exit occurred
        if base_exit_reason or alpha_exit_reason:
            step_info = {
                'step': i + 1,
                'date': data.index[i + 1],
                'price': next_price,
                'low': day_low,
                'high': day_high,
                'base_pool': new_base,
                'alpha_pool': new_alpha,
                'base_active': portfolio["base_active"],
                'alpha_active': portfolio["alpha_active"],
                'base_exit': base_exit_reason,
                'alpha_exit': alpha_exit_reason,
                'base_exit_price': base_exit_price,
                'alpha_exit_price': alpha_exit_price
            }
            trade_log.append(step_info)

        # Re-entry logic (wait one period after exit, don't re-enter immediately)
        # Only re-enter if we have capital and didn't just exit this period
        if not portfolio["base_active"] and new_base > 0 and not base_exit_reason:
            portfolio["base_active"] = True
            portfolio["base_entry_price"] = next_price

        if not portfolio["alpha_active"] and new_alpha > 0 and not alpha_exit_reason:
            portfolio["alpha_active"] = True
            portfolio["alpha_entry_price"] = next_price

        portfolio["base_pool"].append(new_base)
        portfolio["alpha_pool"].append(new_alpha)
        portfolio["portfolio_values"].append(new_base + new_alpha)

    # Handle final partial year tax
    yearly_total_gain = yearly_base_gain + yearly_alpha_gain
    taxable_gain = yearly_total_gain + portfolio["tax_loss_carryforward"]

    if taxable_gain > 0:
        tax = taxable_gain * tax_rate
        portfolio["cumulative_tax"] += tax

        base_positive = max(yearly_base_gain, 0)
        alpha_positive = max(yearly_alpha_gain, 0)
        total_positive = base_positive + alpha_positive

        if total_positive > 0:
            base_tax_share = (base_positive / total_positive) * tax
            alpha_tax_share = (alpha_positive / total_positive) * tax
            portfolio["base_pool"][-1] -= base_tax_share
            portfolio["alpha_pool"][-1] -= alpha_tax_share
            portfolio["portfolio_values"][-1] = portfolio["base_pool"][-1] + portfolio["alpha_pool"][-1]

    # Performance metrics
    portfolio_values = np.array(portfolio["portfolio_values"])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    sharpe_ratio = 0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

    total_return = ((portfolio_values[-1] / initial_capital) - 1) * 100
    buy_and_hold_value = initial_capital * price[-1]/price[0]
    buy_and_hold_return = (price[-1]/price[0]-1)*100

    return {
        "portfolio_values": portfolio_values,
        "base_pool": np.array(portfolio["base_pool"]),
        "alpha_pool": np.array(portfolio["alpha_pool"]),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "cumulative_tax": portfolio["cumulative_tax"],
        "trade_log": trade_log,
        "final_base": portfolio["base_pool"][-1],
        "final_alpha": portfolio["alpha_pool"][-1],
        "buy_and_hold_value": buy_and_hold_value,
        "buy_and_hold_return": buy_and_hold_return
    }