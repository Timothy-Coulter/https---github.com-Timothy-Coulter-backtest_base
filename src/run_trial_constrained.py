# ====================================
# MULTI-OBJECTIVE OPTIMIZATION
# ====================================
import optuna
import numpy as np
import matplotlib.pyplot as plt
from run_sim import get_data, run_portfolio_simulation

if __name__ == '__main__':

    MAX_DRAWDOWN_LIMIT = -0.5  # constraint: drawdown must not exceed -30%

    ticker = "SPY"
    start_date = "1990-01-01"
    end_date = "2025-11-01"
    interval = "1mo"

    data = get_data(ticker=ticker, start_date=start_date, end_date=end_date, interval=interval)

    def objective(trial):
        result = run_portfolio_simulation(
            data = data,
            leverage_base = trial.suggest_float("leverage_base", 1, 2),
            leverage_alpha = trial.suggest_float("leverage_alpha", 1.0, 10.0),
            base_to_alpha_split = trial.suggest_float("base_to_alpha_split", 0.01, 0.99),
            alpha_to_base_split = trial.suggest_float("alpha_to_base_split", 0.01, 0.99),
            stop_loss_base = trial.suggest_float("stop_loss_base", 0.01, 0.1),
            stop_loss_alpha = trial.suggest_float("stop_loss_alpha", 0.01, 0.1),
            take_profit_target = trial.suggest_float("take_profit_target", 0.01, 0.99)        )

        # We want to maximize total_return and minimize drawdown (more positive = better)
        total_return = result["total_return"]
        drawdown = result["max_drawdown"]
        print(drawdown)
        # constraint: penalize if drawdown < limit
        if abs(drawdown) > abs(MAX_DRAWDOWN_LIMIT):
            constraint_violation = 9999
        else:
            constraint_violation = -1
        trial.set_user_attr("result", result)
        return total_return, abs(drawdown), constraint_violation

    study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # ====================================
    # FILTER FEASIBLE TRIALS (within drawdown constraint)
    # ====================================
    feasible_trials = [t for t in study.trials if t.values and t.values[2] <= 0]
    if feasible_trials:
        best_feasible = max(feasible_trials, key=lambda t: t.values[0])
    else:
        print("No feasible trials within drawdown constraint, using best overall trial.")
        best_feasible = max(study.trials, key=lambda t: t.values[0])

    best_result = best_feasible.user_attrs["result"]

    print("\n✅ BEST FEASIBLE PARAMETERS:")
    print(best_feasible.params)
    print(f"Total return: {best_result['total_return']:.2f}%")
    print(f"Max drawdown: {best_result['max_drawdown']*100:.2f}%")
    print(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Cumulative tax: {best_result['cumulative_tax']:.2f}")

    # ====================================
    # PARETO FRONTIER PLOT
    # ====================================

    # Extract results
    returns = []
    drawdowns = []
    feasible_mask = []

    for t in study.trials:
        if not t.values:
            continue
        total_return, neg_drawdown, constraint_violation = t.values
        returns.append(total_return)
        drawdowns.append(-neg_drawdown)  # back to actual drawdown (negative)
        feasible_mask.append(constraint_violation <= 0)

    returns = np.array(returns)
    drawdowns = np.array(drawdowns)
    feasible_mask = np.array(feasible_mask)

    # Compute Pareto front (feasible or all)
    points = np.column_stack([returns, drawdowns])
    pareto_idx = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    pareto_points = points[pareto_idx]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 1])]

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(drawdowns[~feasible_mask]*100, returns[~feasible_mask], c="gray", alpha=0.5, label="Infeasible")
    plt.scatter(drawdowns[feasible_mask]*100, returns[feasible_mask], c="blue", alpha=0.7, label="Feasible")
    plt.plot(pareto_points[:,1]*100, pareto_points[:,0], "r--", linewidth=2, label="Pareto Frontier")
    plt.scatter(best_result["max_drawdown"]*100, best_result["total_return"], c="gold", s=100, edgecolor="black", label="Best Selected")

    plt.xlabel("Max Drawdown (%)")
    plt.ylabel("Total Return (%)")
    plt.title("Pareto Frontier: Return vs. Drawdown")
    plt.legend()
    plt.grid(True)
    plt.show()


