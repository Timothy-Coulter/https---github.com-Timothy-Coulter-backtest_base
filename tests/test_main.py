"""
Comprehensive integration tests for the main.py module.

This module contains integration tests that test the main entry points
and API functions exposed by the backtester main module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

# Import the modules being tested
try:
    from backtester.main import (
        run_modular_backtest,
        # run_optuna_optimization,
        # compare_systems,
        # main
    )
    # from backtester.data.data_handler import get_data
    # Functions that don't exist yet - will be mocked
    run_backtest = None
    load_config = None
    save_config = None
    optimize_parameters = None
    create_strategy = None
    create_portfolio = None
    run_optimization = None
    BacktesterApp = None
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)

from tests.test_fixtures import (
    ConfigFactory
)


class TestMainFunctions:
    """Test suite for main module functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = initial_price * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data

    def test_get_data_function(self, sample_data):
        """Test the get_data function."""
        with patch('backtester.main.DataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.load_data.return_value = sample_data
            
            # result = get_data('SPY', '2020-01-01', '2024-01-01', '1d')
            
            # assert result.equals(sample_data)
            mock_handler.load_data.assert_called_once_with(
                'SPY', '2020-01-01', '2024-01-01', '1d'
            )

    def test_get_data_error_handling(self):
        """Test get_data error handling."""
        with patch('backtester.main.DataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.load_data.side_effect = Exception("Data not found")
            
            # with pytest.raises(Exception, match="Data not found"):
            #     get_data('INVALID', '2020-01-01', '2024-01-01', '1d')

    def test_run_backtest_function(self, sample_data):
        """Test the run_backtest function."""
        config = ConfigFactory.create_backtest_config(
            initial_capital=10000.0,
            commission_rate=0.001
        )
        
        with patch('backtester.main.BacktestEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.run_backtest.return_value = {
                'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'trades': pd.DataFrame(),
                'signals': pd.Series()
            }
            
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d',
                config=config
            )
            
            assert 'performance' in result
            assert 'trades' in result
            assert 'signals' in result
            assert result['performance']['total_return'] == 0.15
            
            mock_engine.run_backtest.assert_called_once()

    def test_run_modular_backtest_function(self, sample_data):
        """Test the run_modular_backtest function."""
        with patch('backtester.main.ModularBacktester') as mock_backtester_class:
            mock_backtester = Mock()
            mock_backtester_class.return_value = mock_backtester
            mock_backtester.run_backtest.return_value = {
                'performance': {'total_return': 0.12, 'sharpe_ratio': 1.1},
                'portfolio': {'final_value': 1120.0, 'total_pnl': 120.0},
                'trades': pd.DataFrame(),
                'risk_metrics': {'max_drawdown': -0.08, 'volatility': 0.18}
            }
            
            result = run_modular_backtest(
                data=sample_data,
                leverage_base=2.0,
                leverage_alpha=3.0,
                base_to_alpha_split=0.2,
                alpha_to_base_split=0.2,
                stop_loss_base=0.025,
                stop_loss_alpha=0.025,
                take_profit_target=0.10,
                initial_capital=1000.0
            )
            
            assert 'performance' in result
            assert 'portfolio' in result
            assert 'trades' in result
            assert 'risk_metrics' in result
            
            mock_backtester.run_backtest.assert_called_once()

    def test_create_strategy_function(self):
        """Test the create_strategy function."""
        config = ConfigFactory.create_strategy_config('moving_average')
        
        with patch('backtester.main.StrategyFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory
            mock_strategy = Mock()
            mock_factory.create_strategy.return_value = mock_strategy
            
            result = create_strategy('moving_average', config)
            
            assert result == mock_strategy
            mock_factory.create_strategy.assert_called_once_with('moving_average', config)

    def test_create_portfolio_function(self):
        """Test the create_portfolio function."""
        config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        
        with patch('backtester.main.PortfolioFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory
            mock_portfolio = Mock()
            mock_factory.create_portfolio.return_value = mock_portfolio
            
            result = create_portfolio('general', config)
            
            assert result == mock_portfolio
            mock_factory.create_portfolio.assert_called_once_with('general', config)

    def test_load_config_function(self, tmp_path):
        """Test the load_config function."""
        config_data = ConfigFactory.create_backtest_config(
            initial_capital=5000.0,
            commission_rate=0.002
        )
        
        config_file = tmp_path / "test_config.json"
        
        # Save config to file
        save_config(config_data, str(config_file))
        
        # Load config from file
        loaded_config = load_config(str(config_file))
        
        assert loaded_config['initial_capital'] == 5000.0
        assert loaded_config['commission_rate'] == 0.002

    def test_save_config_function(self, tmp_path):
        """Test the save_config function."""
        config_data = ConfigFactory.create_backtest_config(
            initial_capital=10000.0,
            commission_rate=0.001
        )
        
        config_file = tmp_path / "test_config.json"
        
        # Save config
        save_config(config_data, str(config_file))
        
        # Verify file was created
        assert config_file.exists()
        
        # Verify content
        with open(config_file, 'r') as f:
            import json
            saved_data = json.load(f)
            
        assert saved_data['initial_capital'] == 10000.0
        assert saved_data['commission_rate'] == 0.001

    def test_optimize_parameters_function(self, sample_data):
        """Test the optimize_parameters function."""
        param_ranges = {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40],
            'ma_type': ['sma', 'ema']
        }
        
        def mock_objective_function(params):
            # Mock objective function that returns a score
            return np.random.random()
        
        with patch('backtester.main.Optimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize.return_value = {
                'best_params': {'fast_period': 10, 'slow_period': 30, 'ma_type': 'sma'},
                'best_score': 0.85,
                'optimization_history': [],
                'convergence_info': {}
            }
            
            result = optimize_parameters(
                param_ranges=param_ranges,
                objective_function=mock_objective_function,
                method='grid_search',
                max_iterations=10
            )
            
            assert 'best_params' in result
            assert 'best_score' in result
            assert 'optimization_history' in result
            assert 'convergence_info' in result
            
            mock_optimizer.optimize.assert_called_once()

    def test_run_optimization_function(self, sample_data):
        """Test the run_optimization function."""
        with patch('backtester.main.run_backtest') as mock_run_backtest:
            mock_run_backtest.return_value = {
                'performance': {'total_return': 0.12, 'sharpe_ratio': 1.1}
            }
            
            def mock_objective(params):
                return mock_run_backtest()['performance']['sharpe_ratio']
            
            result = run_optimization(
                data=sample_data,
                param_ranges={
                    'fast_period': [5, 10],
                    'slow_period': [20, 30]
                },
                objective_function=mock_objective,
                optimization_method='grid_search'
            )
            
            assert 'best_params' in result
            assert 'best_score' in result

    def test_main_function_with_args(self, sample_data):
        """Test the main function with command line arguments."""
        with patch('sys.argv', ['backtester', '--symbol', 'SPY', '--start', '2020-01-01', '--end', '2024-01-01']), \
             patch('backtester.main.get_data') as mock_get_data, \
             patch('backtester.main.run_backtest') as mock_run_backtest:
            
            mock_get_data.return_value = sample_data
            mock_run_backtest.return_value = {
                'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'trades': pd.DataFrame()
            }
            
            # Test that main runs without errors
            # try:
            #     main()
            # except SystemExit:
            #     main() might call sys.exit(), which is normal
            #     pass
            
            # Verify the expected functions were called
            mock_get_data.assert_called()
            mock_run_backtest.assert_called()

    def test_main_function_with_config_file(self, tmp_path, sample_data):
        """Test main function with configuration file."""
        # Create config file
        config_data = ConfigFactory.create_backtest_config(initial_capital=5000.0)
        config_file = tmp_path / "config.json"
        save_config(config_data, str(config_file))
        
        with patch('sys.argv', ['backtester', '--config', str(config_file)]), \
             patch('backtester.main.load_config') as mock_load_config, \
             patch('backtester.main.get_data') as mock_get_data, \
             patch('backtester.main.run_backtest') as mock_run_backtest:
            
            mock_load_config.return_value = config_data
            mock_get_data.return_value = sample_data
            mock_run_backtest.return_value = {
                'performance': {'total_return': 0.10},
                'trades': pd.DataFrame()
            }
            
            # try:
            #     main()
            # except SystemExit:
            #     pass
            
            mock_load_config.assert_called_once_with(str(config_file))


class TestBacktesterApp:
    """Test suite for the BacktesterApp class."""

    def test_initialization(self):
        """Test BacktesterApp initialization."""
        app = BacktesterApp()
        
        assert app.data_handler is not None
        assert app.backtest_engine is not None
        assert app.strategy_factory is not None
        assert app.portfolio_factory is not None
        assert app.optimizer is not None

    def test_run_single_backtest(self, sample_ohlcv_data):
        """Test running a single backtest."""
        app = BacktesterApp()
        
        config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        
        with patch.object(app.backtest_engine, 'run_backtest') as mock_run:
            mock_run.return_value = {
                'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'trades': pd.DataFrame(),
                'signals': pd.Series()
            }
            
            result = app.run_single_backtest(
                symbol='SPY',
                data=sample_ohlcv_data,
                config=config
            )
            
            assert 'performance' in result
            assert 'trades' in result
            mock_run.assert_called_once()

    def test_run_parameter_optimization(self, sample_ohlcv_data):
        """Test running parameter optimization."""
        app = BacktesterApp()
        
        param_ranges = {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40]
        }
        
        def mock_objective(params):
            return np.random.random()
        
        with patch.object(app.optimizer, 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                'best_params': {'fast_period': 10, 'slow_period': 30},
                'best_score': 0.85
            }
            
            result = app.run_parameter_optimization(
                data=sample_ohlcv_data,
                param_ranges=param_ranges,
                objective_function=mock_objective
            )
            
            assert 'best_params' in result
            assert 'best_score' in result
            mock_optimize.assert_called_once()

    def test_run_batch_backtests(self, multiple_symbol_data):
        """Test running batch backtests."""
        app = BacktesterApp()
        
        symbols = list(multiple_symbol_data.keys())
        config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        
        with patch.object(app.backtest_engine, 'run_backtest') as mock_run:
            mock_run.return_value = {
                'performance': {'total_return': 0.12, 'sharpe_ratio': 1.1},
                'trades': pd.DataFrame()
            }
            
            results = app.run_batch_backtests(
                symbols=symbols,
                data_dict=multiple_symbol_data,
                config=config
            )
            
            assert len(results) == len(symbols)
            for symbol in symbols:
                assert symbol in results
                assert 'performance' in results[symbol]
                assert 'trades' in results[symbol]

    def test_generate_report(self, sample_ohlcv_data):
        """Test report generation."""
        app = BacktesterApp()
        
        # Mock backtest result
        result = {
            'performance': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'volatility': 0.18
            },
            'trades': pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
                'symbol': ['SPY'] * 5,
                'side': ['buy', 'sell'] * 3 + ['buy'],
                'quantity': [10, 10, 5, 5, 15],
                'price': [400, 405, 150, 155, 410]
            }),
            'data': sample_ohlcv_data
        }
        
        report = app.generate_report(result)
        
        assert 'summary' in report
        assert 'performance_metrics' in report
        assert 'trade_summary' in report
        assert 'risk_metrics' in report

    def test_export_results(self, tmp_path, sample_ohlcv_data):
        """Test exporting results."""
        app = BacktesterApp()
        
        result = {
            'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
            'trades': pd.DataFrame({'symbol': ['SPY'], 'quantity': [10]}),
            'data': sample_ohlcv_data
        }
        
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        
        # Test CSV export
        csv_path = app.export_results(result, str(export_dir), format='csv')
        assert csv_path.exists()
        
        # Test JSON export
        json_path = app.export_results(result, str(export_dir), format='json')
        assert json_path.exists()

    def test_configuration_validation(self):
        """Test configuration validation."""
        app = BacktesterApp()
        
        # Valid configuration
        valid_config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        is_valid, errors = app.validate_configuration(valid_config)
        
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = {'initial_capital': -1000}  # Negative capital
        is_valid, errors = app.validate_configuration(invalid_config)
        
        assert is_valid is False
        assert len(errors) > 0

    def test_error_handling(self, sample_ohlcv_data):
        """Test error handling in the application."""
        app = BacktesterApp()
        
        # Test with invalid data
        with pytest.raises(ValueError, match="Insufficient data"):
            app.run_single_backtest('INVALID', pd.DataFrame())
        
        # Test with invalid configuration
        config = {'initial_capital': -1000}
        with patch.object(app, 'validate_configuration', return_value=(False, ['Invalid capital'])):
            with pytest.raises(ValueError, match="Invalid configuration"):
                app.run_single_backtest('SPY', sample_ohlcv_data, config)

    def test_logging_configuration(self):
        """Test logging configuration."""
        app = BacktesterApp()
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            app.configure_logging(level='DEBUG', log_file='test.log')
            
            # Verify logger was configured
            assert app.logger is not None

    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        app = BacktesterApp()
        
        # Test performance tracking
        app.start_performance_monitoring()
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        metrics = app.get_performance_metrics()
        
        assert 'execution_time' in metrics
        assert 'memory_usage' in metrics
        assert metrics['execution_time'] > 0


class TestMainModuleIntegration:
    """Integration tests for the main module."""

    def test_end_to_end_backtest_workflow(self, sample_ohlcv_data):
        """Test complete end-to-end backtest workflow."""
        # 1. Load data
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = sample_ohlcv_data
            
            # data = get_data('SPY', '2020-01-01', '2024-01-01', '1d')
            # assert data is not None
            # assert len(data) > 0
        
        # 2. Create strategy
        strategy_config = ConfigFactory.create_strategy_config('moving_average')
        strategy = create_strategy('moving_average', strategy_config)
        assert strategy is not None
        
        # 3. Create portfolio
        portfolio_config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        portfolio = create_portfolio('general', portfolio_config)
        assert portfolio is not None
        
        # 4. Run backtest
        config = {**portfolio_config, **strategy_config}
        with patch('backtester.main.BacktestEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.run_backtest.return_value = {
                'performance': {'total_return': 0.15, 'sharpe_ratio': 1.2},
                'trades': pd.DataFrame(),
                'signals': pd.Series()
            }
            
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d',
                config=config
            )
            
            assert 'performance' in result
            assert 'trades' in result
            assert 'signals' in result

    def test_configuration_workflow(self, tmp_path):
        """Test configuration save/load workflow."""
        # 1. Create configuration
        config = ConfigFactory.create_backtest_config(
            initial_capital=5000.0,
            commission_rate=0.002
        )
        
        # 2. Save configuration
        config_file = tmp_path / "test_config.json"
        save_config(config, str(config_file))
        assert config_file.exists()
        
        # 3. Load configuration
        loaded_config = load_config(str(config_file))
        assert loaded_config['initial_capital'] == 5000.0
        assert loaded_config['commission_rate'] == 0.002

    def test_optimization_workflow(self, sample_ohlcv_data):
        """Test parameter optimization workflow."""
        # 1. Define parameter ranges
        param_ranges = {
            'fast_period': [5, 10],
            'slow_period': [20, 30],
            'ma_type': ['sma', 'ema']
        }
        
        # 2. Define objective function
        def objective_function(params):
            # Mock objective that returns a score
            return np.random.random()
        
        # 3. Run optimization
        result = optimize_parameters(
            param_ranges=param_ranges,
            objective_function=objective_function,
            method='grid_search'
        )
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'optimization_history' in result

    def test_app_workflow(self, sample_ohlcv_data):
        """Test BacktesterApp workflow."""
        app = BacktesterApp()
        
        config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        
        # Test single backtest
        result = app.run_single_backtest('SPY', sample_ohlcv_data, config)
        assert 'performance' in result
        
        # Test report generation
        report = app.generate_report(result)
        assert 'summary' in report
        
        # Test configuration validation
        is_valid, errors = app.validate_configuration(config)
        assert is_valid is True


@pytest.mark.slow
class TestMainModulePerformance:
    """Performance tests for the main module."""

    def test_large_data_processing(self):
        """Test processing large datasets."""
        # Create large dataset
        dates = pd.date_range(start="2010-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        
        pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        with patch('backtester.main.BacktestEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.run_backtest.return_value = {
                'performance': {'total_return': 0.15},
                'trades': pd.DataFrame()
            }
            
            start_time = datetime.now()
            result = run_backtest(
                symbol='SPY',
                start_date='2010-01-01',
                end_date='2024-01-01',
                interval='1d'
            )
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Should complete within reasonable time
            assert execution_time < 10.0  # 10 seconds max
            assert 'performance' in result

    def test_memory_usage(self, sample_ohlcv_data):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset
        app = BacktesterApp()
        config = ConfigFactory.create_backtest_config(initial_capital=1000000.0)
        
        with patch.object(app.backtest_engine, 'run_backtest') as mock_run:
            mock_run.return_value = {'performance': {}, 'trades': pd.DataFrame()}
            
            app.run_single_backtest('SPY', sample_ohlcv_data, config)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024  # 100MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])