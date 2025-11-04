import numpy as np
import optuna
import matplotlib.pyplot as plt
import multiprocessing
import os
from functools import partial
from dotenv import load_dotenv
from datetime import datetime

from utils import get_data  # type: ignore[import-not-found]
from run_sim import run_portfolio_simulation  # type: ignore[import-not-found]

# Load environment variables from .env file
load_dotenv()


# Note, need to install postgressql if not installed: https://www.postgresql.org/download/windows/


def objective(trial, data):
    result = run_portfolio_simulation(
        data=data,
        leverage_base=trial.suggest_float("leverage_base", 1.0, 10.0),
        leverage_alpha=trial.suggest_float("leverage_alpha", 1.0, 10.0),
        base_to_alpha_split=trial.suggest_float("base_to_alpha_split", 0.01, 0.99),
        alpha_to_base_split=trial.suggest_float("alpha_to_base_split", 0.01, 0.99),
        stop_loss_base=trial.suggest_float("stop_loss_base", 0.01, 0.05),
        stop_loss_alpha=trial.suggest_float("stop_loss_alpha", 0.01, 0.10),
        take_profit_target=trial.suggest_float("take_profit_target", 0.05, 0.20),
    )
    penalty = 50
    score = result["total_return"] - penalty * abs(result["max_drawdown"] * 100)
    return score


def worker(study_name, storage_name, data):
    """Single worker function for parallel Optuna grid search"""
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(partial(objective, data=data))


def run_parallel_grid_search(data):
    N1 = 5
    N2 = 2

    search_space = {
        "leverage_base": np.linspace(1.0, 10.0, N1).tolist(),
        "leverage_alpha": np.linspace(1.0, 10.0, N1).tolist(),
        "base_to_alpha_split": np.linspace(0.01, 0.99, N1).tolist(),
        "alpha_to_base_split": np.linspace(0.01, 0.99, N1).tolist(),
        "stop_loss_base": np.linspace(0.01, 0.05, N2).tolist(),
        "stop_loss_alpha": np.linspace(0.01, 0.10, N2).tolist(),
        "take_profit_target": np.linspace(0.05, 0.20, N2).tolist(),
    }

    # Get PostgreSQL credentials from environment variables
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "optuna_db")
    
    if not db_password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is not set. "
            "Please create a .env file with your PostgreSQL credentials."
        )
    
    storage_name = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    sampler = optuna.samplers.GridSampler(search_space)
    
    # Create unique study name with timestamp (YYMMDD_HHMM)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    study_name = f"portfolio_grid_search_{timestamp}"
    
    print(f"Creating new study: {study_name}")

    # Create a new study (load_if_exists=False ensures fresh study each run)
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=False,
    )

    num_workers = os.cpu_count()
    print(f"Launching {num_workers} parallel workers...")

    # Spawn processes with target=worker (no lambdas!)
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(worker, [(study_name, storage_name, data)] * num_workers)

    print("✅ Grid search completed.")
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    return study


if __name__ == "__main__":
    ticker = "SPY"
    start_date = "1990-01-01"
    end_date = "2025-11-01"
    interval = "1mo"

    data = get_data(ticker=ticker, start_date=start_date, end_date=end_date, interval=interval)

    # Run random optimization first
    def wrapped_objective(trial):
        return objective(trial, data)

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=100)

    print("Best parameters (random search):")
    print(study.best_params)
    print("Best score:", study.best_value)

    # Run grid search in parallel
    grid_study = run_parallel_grid_search(data)

    # Final evaluation
    best_result = run_portfolio_simulation(data=data, **grid_study.best_params)
    print(f"\nFinal portfolio value: {best_result['portfolio_values'][-1]:.2f}")
    print(f"Total return: {best_result['total_return']:.2f}%")
    print(f"Max drawdown: {best_result['max_drawdown']*100:.2f}%")
    print(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Cumulative tax paid: {best_result['cumulative_tax']:.2f}")
    print(f"Buy and Hold Return: {best_result['buy_and_hold_return']:.2f}%")

    # Plot
    X = np.linspace(0, 1, len(data["Close"]))
    initial_equity_price = data["Close"].iloc[0]
    initial_price = 100
    close_normalised = data["Close"] * initial_price / initial_equity_price

    plt.figure()
    plt.plot(X, close_normalised)
    plt.plot(X, best_result["base_pool"])
    plt.plot(X, best_result["alpha_pool"])
    plt.legend(["Close", "Base", "Alpha"])
    plt.show()
