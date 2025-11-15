# quant-bench

A quantitative backtesting framework for financial analysis.

## Features

- Market data handling and validation
- Portfolio management and backtesting
- Strategy development and optimization
- Performance analysis and reporting

## Installation

```bash
uv sync
```

## Usage

```python
from backtester.main import run_backtest

# Run a basic backtest
results = run_backtest()
```

## Event Payload Contract

All components communicate through the event bus using a shared metadata contract. The keys
below are guaranteed to exist (or be explicitly empty) so subscribers can build filters without
having to reverse‑engineer publisher internals.

- **MarketDataEvent**
  - `metadata.symbol` and `metadata.symbols` (universe slice for the payload)
  - `metadata.bar` containing `open`, `high`, `low`, `close`, `volume`, `timestamp`
  - `metadata.provenance.source` and `metadata.provenance.ingested_at`
  - `metadata.data_frame` when the upstream producer supplied a pandas window
- **SignalEvent**
  - `metadata.symbol`, `metadata.symbols`, and `metadata.signal_type`
  - `metadata.source_strategy` and any raw indicator payload under `metadata.raw_signal`
  - Priority defaults to HIGH so downstream order routers can rely on delivery order
- **OrderEvent**
  - `metadata.symbol`, `metadata.side`, `metadata.order_type`
  - Execution context (`fill_quantity`, `fill_price`, `commission`) plus `metadata.message`
    when an order is rejected or cancelled
- **PortfolioUpdateEvent**
  - `metadata.portfolio_id`, `metadata.positions`, and `metadata.position_updates`
  - Mirrors the monetary snapshot via `metadata.total_value`, `metadata.cash_balance`,
    and `metadata.positions_value`
- **RiskAlertEvent**
  - `metadata.component`, `metadata.portfolio_id` (when applicable), and `metadata.violations`
  - `metadata.recommendations` summarising the suggested remediation steps

When subscribing, prefer the metadata keys (`symbol`/`symbols`) rather than bespoke attributes,
as they are populated uniformly regardless of which module produced the event.

## Engine Workflow

The canonical simulation loop—from market data ingestion through strategy signals,
risk evaluation, order routing, broker fills, and portfolio/performance updates—is
documented in detail (with a Mermaid diagram) in `docs/engine_workflow.md`.
Reference it when implementing new components or investigating event ordering.

## Development Commands

### Code Formatting and Linting

**Windows one-liners:**
```cmd
rem Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

rem Lint code (ruff check)
uv run ruff check .

rem Lint and fix (ruff check --fix)
uv run ruff check --fix .

rem Type checking (mypy)
uv run mypy .

rem combined command
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check --fix . && uv run mypy .

```

**Linux/macOS one-liners:**
```bash
# Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

# Lint code (ruff check)
uv run ruff check .

# Lint and fix (ruff check --fix)
uv run ruff check --fix .

# Type checking (mypy)
uv run mypy .
```

### Testing

```bash
# Run tests (uses pyproject addopts: -n auto --reruns 3)
uv run pytest

# Run tests with coverage
uv run pytest --cov=backtester --cov-report=term-missing --cov-report=html
```

### Complete Development Workflow

**Windows:**
```cmd
rem Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```

**Linux/macOS:**
```bash
# Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```
