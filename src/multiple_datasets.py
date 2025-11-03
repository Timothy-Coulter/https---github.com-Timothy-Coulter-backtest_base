import optuna
from run_sim import get_data, run_portfolio_simulation

# List of datasets (tickers)
datasets = ["SPY", "QQQ", "VTI"]

# Pre-load all data to avoid repeated downloads
all_data = {ticker: get_data(ticker=ticker, start_date="1990-01-01", end_date="2025-11-01", interval="1mo")
            for ticker in datasets}

def objective(trial):
    total_score = 0.0

    # Suggest parameters once — same for all datasets
    leverage_base = trial.suggest_float("leverage_base", 1.0, 10.0)
    leverage_alpha = trial.suggest_float("leverage_alpha", 1.0, 10.0)
    base_to_alpha_split = trial.suggest_float("base_to_alpha_split", 0.01, 0.99)
    alpha_to_base_split = trial.suggest_float("alpha_to_base_split", 0.01, 0.99)
    stop_loss_base = trial.suggest_float("stop_loss_base", 0.01, 0.05)
    stop_loss_alpha = trial.suggest_float("stop_loss_alpha", 0.01, 0.10)
    take_profit_target = trial.suggest_float("take_profit_target", 0.05, 0.20)

    penalty = 50

    for ticker, data in all_data.items():
        result = run_portfolio_simulation(
            data=data,
            leverage_base=leverage_base,
            leverage_alpha=leverage_alpha,
            base_to_alpha_split=base_to_alpha_split,
            alpha_to_base_split=alpha_to_base_split,
            stop_loss_base=stop_loss_base,
            stop_loss_alpha=stop_loss_alpha,
            take_profit_target=take_profit_target
        )
        # Combine scores (e.g., sum of return - penalty * drawdown)
        score = result["total_return"] - penalty * abs(result["max_drawdown"]*100)
        total_score += score

    # Optionally, take average instead of sum
    return total_score / len(all_data)

# Create and run study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best parameters overall for all datasets:")
print(study.best_params)
print("Best aggregated score:", study.best_value)
