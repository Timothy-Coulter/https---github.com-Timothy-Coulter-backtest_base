"""Integration tests for the event-driven backtest engine pipeline."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.core.event_bus import EventFilter
from backtester.core.events import SignalEvent
from backtester.strategy.moving_average import DualPoolMovingAverageStrategy


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

    def reset(self) -> None:
        """Reset accrued trade history and valuation."""
        self.portfolio_values.clear()
        self.trade_log.clear()


class StubBroker:
    """Simulated broker capturing orders without external effects."""

    def __init__(self) -> None:
        """Initialise the broker stub with empty state."""
        self.positions: dict[str, float] = {}
        self.order_manager = MagicMock()

    def execute_order(self, order: MagicMock) -> None:
        """Mark an order as executed."""
        order.is_active = False

    def reset(self) -> None:
        """Reset broker positions."""
        self.positions.clear()


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
    ) -> dict[str, float]:
        """Record processed signals and update stub portfolio values."""
        processed_signals.append(signals)
        next_value = 100.0 + len(processed_signals)
        stub_portfolio.portfolio_values.append(next_value)
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
    engine.event_bus.subscribe(
        lambda event: captured_signals.append(event),
        EventFilter(event_types={"SIGNAL"}),
    )

    stub_portfolio = StubPortfolio()
    stub_broker = StubBroker()
    stub_risk = StubRiskManager()
    processed_signals: list[list[dict[str, object]]] = []

    _install_stubs(engine, stub_portfolio, stub_broker, stub_risk, processed_signals)

    with patch.object(
        DualPoolMovingAverageStrategy,
        "generate_signals",
        lambda self, data: [{"signal_type": "BUY", "confidence": 0.9}],
    ):
        results = engine.run_backtest()

    assert processed_signals, "The orchestrator should forward signals to the portfolio pipeline."
    expected_iterations = len(market_data) - 1
    assert len(processed_signals) == expected_iterations
    assert len(captured_signals) == expected_iterations
    assert results["performance"]["final_portfolio_value"] == pytest.approx(
        100.0 + expected_iterations
    )
