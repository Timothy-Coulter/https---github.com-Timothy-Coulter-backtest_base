"""Integration tests for the event-driven backtest engine pipeline."""

from __future__ import annotations

import types
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.core.event_bus import EventFilter
from backtester.core.events import SignalEvent, create_portfolio_update_event
from backtester.strategy.orchestration import StrategyKind
from backtester.strategy.signal.base_signal_strategy import BaseSignalStrategy
from backtester.strategy.signal.momentum_strategy import MomentumStrategy


def _sample_market_data(rows: int = 5) -> pd.DataFrame:
    """Build deterministic OHLCV data for the integration scenario."""
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "Open": [100 + i for i in range(rows)],
        "High": [102 + i for i in range(rows)],
        "Low": [98 + i for i in range(rows)],
        "Close": [101 + i for i in range(rows)],
        "Volume": [1000.0 for _ in range(rows)],
    }
    return pd.DataFrame(data, index=index)


@dataclass
class StubPortfolio:
    """Minimal portfolio stub that tracks portfolio values."""

    portfolio_values: list[float] = field(default_factory=list)
    initial_capital: float = 100.0
    trade_log: list[dict[str, float]] = field(default_factory=list)
    commission_rate: float = 0.0
    max_positions: int | None = None
    before_run_calls: int = 0
    after_run_calls: int = 0
    before_tick_calls: int = 0
    after_tick_calls: int = 0

    def reset(self) -> None:
        """Reset accrued trade history and valuation."""
        self.portfolio_values.clear()
        self.trade_log.clear()

    @property
    def total_value(self) -> float:
        """Return the latest portfolio valuation."""
        if self.portfolio_values:
            return self.portfolio_values[-1]
        return self.initial_capital

    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Lifecycle hook before the simulation."""
        self.before_run_calls += 1

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Lifecycle hook after the simulation."""
        self.after_run_calls += 1

    def before_tick(self, context: dict[str, Any]) -> None:
        """Lifecycle hook before each tick."""
        self.before_tick_calls += 1

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        """Lifecycle hook after each tick."""
        self.after_tick_calls += 1


class StubBroker:
    """Simulated broker capturing orders without external effects."""

    def __init__(self) -> None:
        """Initialise the broker stub with empty state."""
        self.positions: dict[str, float] = {}
        self.order_manager = MagicMock()
        self.before_run_calls = 0
        self.after_run_calls = 0
        self.before_tick_calls = 0
        self.after_tick_calls = 0

    def execute_order(self, order: MagicMock) -> None:
        """Mark an order as executed."""
        order.is_active = False

    def reset(self) -> None:
        """Reset broker positions."""
        self.positions.clear()

    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Lifecycle hook before the simulation."""
        self.before_run_calls += 1

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Lifecycle hook after the simulation."""
        self.after_run_calls += 1

    def before_tick(self, context: dict[str, Any]) -> None:
        """Lifecycle hook before each tick."""
        self.before_tick_calls += 1

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        """Lifecycle hook after each tick."""
        self.after_tick_calls += 1


class StubRiskManager:
    """Basic risk manager stub reporting low risk."""

    def check_portfolio_risk(self, value: float, positions: dict[str, float]) -> dict[str, object]:
        """Always return a low-risk assessment for the test."""
        return {"risk_level": "LOW", "violations": []}

    def add_risk_signal(self, signal: dict[str, object]) -> None:  # noqa: D401
        """No-op risk signal recorder for the stub."""

    def reset(self) -> None:  # noqa: D401
        """No state to reset for the stub."""


def _install_stubs(
    engine: BacktestEngine,
    stub_portfolio: StubPortfolio,
    stub_broker: StubBroker,
    stub_risk: StubRiskManager,
    processed_signals: list[list[dict[str, object]]],
) -> None:
    """Attach stubbed components and signal capture callbacks to the engine."""

    def fake_create_portfolio(
        self: Any, portfolio_params: dict[str, object] | None = None
    ) -> StubPortfolio:
        """Return the prepared stub portfolio."""
        self.current_portfolio = stub_portfolio
        return stub_portfolio

    def fake_create_broker(self: Any) -> StubBroker:
        """Return the prepared stub broker."""
        self.current_broker = stub_broker
        return stub_broker

    def fake_create_risk_manager(self: Any) -> StubRiskManager:
        """Return the prepared stub risk manager."""
        self.current_risk_manager = stub_risk
        return stub_risk

    def fake_process(
        self: BacktestEngine,
        signals: list[dict[str, object]],
        current_price: float,
        day_high: float,
        day_low: float,
        timestamp: object,
        symbol: str,
        historical_data: pd.DataFrame,
        *,
        allow_execution: bool = True,
    ) -> dict[str, float]:
        """Record processed signals and update stub portfolio values."""
        processed_signals.append(signals)
        next_value = 100.0 + len(processed_signals)
        stub_portfolio.portfolio_values.append(next_value)
        portfolio_event = create_portfolio_update_event(
            portfolio_id="stub_portfolio",
            total_value=next_value,
            cash_balance=stub_portfolio.initial_capital,
            positions_value=0.0,
            source="stub_portfolio",
            metadata={
                'timestamp': timestamp,
                'position_updates': [],
            },
        )
        self.event_bus.publish(portfolio_event, immediate=True)
        return {"total_value": next_value}

    engine.create_portfolio = types.MethodType(fake_create_portfolio, engine)
    engine.create_broker = types.MethodType(fake_create_broker, engine)
    engine.create_risk_manager = types.MethodType(fake_create_risk_manager, engine)
    engine._process_signals_and_update_portfolio = types.MethodType(fake_process, engine)
    engine._calculate_performance_metrics = types.MethodType(
        lambda self: {
            "final_portfolio_value": (
                stub_portfolio.portfolio_values[-1] if stub_portfolio.portfolio_values else 0.0
            )
        },
        engine,
    )


def test_backtest_engine_event_driven_flow() -> None:
    """Ensure the engine propagates events through the orchestrator to portfolio stubs."""
    config = BacktesterConfig()
    engine = BacktestEngine(config=config)

    market_data = _sample_market_data(rows=6)
    engine.current_data = market_data

    captured_signals: list[SignalEvent] = []
    captured_portfolio_events = []
    engine.event_bus.subscribe(
        lambda event: captured_signals.append(event),
        EventFilter(event_types={"SIGNAL"}),
    )
    engine.event_bus.subscribe(
        lambda event: captured_portfolio_events.append(event),
        EventFilter(event_types={"PORTFOLIO_UPDATE"}),
    )

    stub_portfolio = StubPortfolio()
    stub_broker = StubBroker()
    stub_risk = StubRiskManager()
    processed_signals: list[list[dict[str, object]]] = []

    _install_stubs(engine, stub_portfolio, stub_broker, stub_risk, processed_signals)

    original_create_strategy = engine.create_strategy

    def dual_strategy_factory(
        self: BacktestEngine, strategy_params: dict[str, Any] | None = None
    ) -> BaseSignalStrategy:
        """Create the default strategy and register a secondary copy for multi-strategy testing."""
        primary = original_create_strategy(strategy_params)
        secondary_config = self._build_momentum_config(strategy_params or {}, "secondary_strategy")
        secondary_strategy = MomentumStrategy(secondary_config, self.event_bus)
        self.strategy_orchestrator.register_strategy(
            identifier="secondary_strategy",
            strategy=secondary_strategy,
            kind=StrategyKind.SIGNAL,
            priority=1,
        )
        return primary

    engine.create_strategy = types.MethodType(dual_strategy_factory, engine)

    with (
        patch.object(
            MomentumStrategy,
            "generate_signals",
            lambda self, data, symbol: [
                {
                    "signal_type": (
                        "SELL" if "secondary" in getattr(self, "name", "").lower() else "BUY"
                    ),
                    "confidence": 0.9,
                    "metadata": {
                        "symbol": symbol,
                        "timestamp": (
                            data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
                        ),
                    },
                }
            ],
        ),
        ExitStack() as stack,
    ):
        mock_before_tick = stack.enter_context(
            patch.object(MomentumStrategy, "before_tick", autospec=True, return_value=None)
        )
        mock_after_tick = stack.enter_context(
            patch.object(MomentumStrategy, "after_tick", autospec=True, return_value=None)
        )
        mock_before_run = stack.enter_context(
            patch.object(MomentumStrategy, "before_run", autospec=True, return_value=None)
        )
        mock_after_run = stack.enter_context(
            patch.object(MomentumStrategy, "after_run", autospec=True, return_value=None)
        )
        results = engine.run_backtest()

    assert processed_signals, "The orchestrator should forward signals to the portfolio pipeline."
    expected_iterations = len(market_data) - 1
    assert len(processed_signals) == expected_iterations
    strategy_count = 2
    assert len(captured_signals) >= expected_iterations * strategy_count
    assert all(batch for batch in processed_signals)
    assert len(captured_portfolio_events) == expected_iterations
    assert captured_portfolio_events[-1].total_value == pytest.approx(100.0 + expected_iterations)
    assert results["performance"]["final_portfolio_value"] == pytest.approx(
        100.0 + expected_iterations
    )
    assert stub_portfolio.before_run_calls == 1
    assert stub_portfolio.after_run_calls == 1
    assert stub_portfolio.before_tick_calls == expected_iterations
    assert stub_portfolio.after_tick_calls == expected_iterations
    assert stub_broker.before_run_calls == 1
    assert stub_broker.after_run_calls == 1
    assert stub_broker.before_tick_calls == expected_iterations
    assert stub_broker.after_tick_calls == expected_iterations
    assert mock_before_tick.call_count == expected_iterations
    assert mock_after_tick.call_count == expected_iterations
    assert mock_before_run.call_count == 1
    assert mock_after_run.call_count == 1
