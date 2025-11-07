"""Configuration System for the Backtester.

This module provides a centralized configuration system that can be used
globally throughout the backtesting framework.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtester.risk_management.component_configs.comprehensive_risk_config import ComprehensiveRiskConfig


class DataRetrievalConfig(BaseModel):
    """Configuration class for data retrieval parameters using pydantic BaseModel.

    This class defines all the parameters needed to configure a data retrieval request,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Data source configuration
    data_source: str = Field(
        default="yahoo", description="Data source (e.g., yahoo, bloomberg, fred)"
    )
    start_date: str | Any = Field(default="year", description="Start date for data retrieval")
    finish_date: str | Any | None = Field(
        default=None, description="Finish date for data retrieval"
    )

    # Ticker and field configuration
    tickers: str | list[str] | None = Field(default=None, description="List of ticker symbols")
    fields: list[str] = Field(default=["close"], description="List of fields to retrieve")
    vendor_tickers: list[str] | None = Field(
        default=None, description="Vendor-specific ticker symbols"
    )
    vendor_fields: list[str] | None = Field(default=None, description="Vendor-specific field names")

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v: str | list[str] | None) -> str | list[str] | None:
        """Convert single string to list for tickers."""
        if isinstance(v, str):
            return [v]
        return v

    # Frequency and granularity
    freq: str = Field(default="daily", description="Data frequency (daily, intraday, etc.)")
    gran_freq: str | None = Field(default=None, description="Granular frequency")
    freq_mult: int = Field(default=1, description="Frequency multiplier")

    # Cache and environment configuration
    cache_algo: str = Field(default="internet_load_return", description="Cache algorithm to use")
    environment: str | None = Field(default=None, description="Data environment (prod, backtest)")
    cut: str = Field(default="NYC", description="Cut time for data")

    # API keys and authentication
    fred_api_key: str | None = Field(default=None, description="FRED API key")
    alpha_vantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")
    eikon_api_key: str | None = Field(default=None, description="Eikon API key")

    # Additional parameters
    category: str | None = Field(default=None, description="Data category")
    dataset: str | None = Field(default=None, description="Dataset name")
    trade_side: str = Field(default="trade", description="Trade side (trade, bid, ask)")
    resample: str | None = Field(default=None, description="Resample frequency")
    resample_how: str = Field(default="last", description="Resample method")

    # Threading and performance
    split_request_chunks: int = Field(default=0, description="Split request into chunks")
    list_threads: int = Field(default=1, description="Number of threads for data loading")

    # Cache behavior
    push_to_cache: bool = Field(default=True, description="Whether to push data to cache")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Data overrides")


class StrategyConfig(BaseModel):
    """Strategy-related configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    strategy_name: str = Field(default="DualPoolMA", description="Name of the strategy")
    ma_short: int = Field(default=5, description="Short moving average period")
    ma_long: int = Field(default=20, description="Long moving average period")
    leverage_base: float = Field(default=1.0, description="Base leverage")
    leverage_alpha: float = Field(default=3.0, description="Alpha leverage")
    base_to_alpha_split: float = Field(default=0.2, description="Base to alpha split ratio")
    alpha_to_base_split: float = Field(default=0.2, description="Alpha to base split ratio")
    stop_loss_base: float = Field(default=0.025, description="Base stop loss percentage")
    stop_loss_alpha: float = Field(default=0.025, description="Alpha stop loss percentage")
    take_profit_target: float = Field(default=0.10, description="Take profit target percentage")


class PortfolioConfig(BaseModel):
    """Portfolio-related configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    initial_capital: float = Field(default=100.0, description="Initial capital")
    maintenance_margin: float = Field(default=0.5, description="Maintenance margin")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    interest_rate_daily: float = Field(default=0.00025, description="Daily interest rate")
    spread_rate: float = Field(default=0.0002, description="Spread rate")
    slippage_std: float = Field(default=0.0005, description="Slippage standard deviation")
    funding_enabled: bool = Field(default=True, description="Whether funding is enabled")
    tax_rate: float = Field(default=0.45, description="Tax rate")


class ExecutionConfig(BaseModel):
    """Execution-related configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    commission_rate: float = Field(default=0.001, description="Commission rate")
    min_commission: float = Field(default=1.0, description="Minimum commission")
    spread: float = Field(default=0.0001, description="Spread")
    slippage_model: str = Field(default="normal", description="Slippage model")
    slippage_std: float = Field(default=0.0005, description="Slippage standard deviation")
    latency_ms: float = Field(default=0.0, description="Latency in milliseconds")


class SimulatedBrokerConfig(BaseModel):
    """Configuration for SimulatedBroker parameters."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    commission_rate: float = Field(
        default=0.001, description="Commission rate for trades (as decimal)"
    )
    min_commission: float = Field(default=1.0, description="Minimum commission per trade")
    spread: float = Field(default=0.0001, description="Bid-ask spread (as decimal)")
    slippage_model: str = Field(
        default="normal", description="Type of slippage model ('normal', 'fixed', 'none')"
    )
    slippage_std: float = Field(
        default=0.0005, description="Standard deviation for slippage simulation"
    )
    latency_ms: float = Field(default=0.0, description="Simulated latency in milliseconds")

    @field_validator('slippage_model')
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        """Validate slippage model is one of the allowed values."""
        valid_models = ['normal', 'fixed', 'none']
        if v not in valid_models:
            raise ValueError(f"slippage_model must be one of {valid_models}")
        return v

class PerformanceConfig(BaseModel):
    """Performance analysis configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")
    benchmark_enabled: bool = Field(default=False, description="Whether benchmark is enabled")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")

class BacktesterConfig(BaseModel):
    """Main configuration class for the backtester."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    data: DataRetrievalConfig | None = Field(
        default=None, description="Data retrieval configuration"
    )
    strategy: StrategyConfig | None = Field(default=None, description="Strategy configuration")
    portfolio: PortfolioConfig | None = Field(default=None, description="Portfolio configuration")
    execution: ExecutionConfig | None = Field(default=None, description="Execution configuration")
    risk: ComprehensiveRiskConfig | None = Field(default=None, description="Risk configuration")
    performance: PerformanceConfig | None = Field(
        default=None, description="Performance configuration"
    )

    # Data period and trading behavior settings
    data_period_days: int = Field(
        default=1, description="Time interval between data points in days"
    )
    maximum_period_between_trade: int = Field(
        default=30, description="Maximum periods to wait between trades"
    )
    trade_immediately_after_stop: bool = Field(
        default=True, description="Whether to trade immediately after stop"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize BacktesterConfig and set default configurations if not provided."""
        super().__init__(**data)
        # Initialize default configurations if not provided
        if self.data is None:
            self.data = DataRetrievalConfig()
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.portfolio is None:
            self.portfolio = PortfolioConfig()
        if self.execution is None:
            self.execution = ExecutionConfig()
        if self.risk is None:
            self.risk = ComprehensiveRiskConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()


# Global configuration instance
_global_config: BacktesterConfig | None = None


def get_config() -> BacktesterConfig:
    """Get the global configuration instance.

    Returns:
        Global BacktesterConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = BacktesterConfig()
    assert _global_config is not None
    return _global_config


def set_config(config: BacktesterConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: BacktesterConfig instance to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = BacktesterConfig()


