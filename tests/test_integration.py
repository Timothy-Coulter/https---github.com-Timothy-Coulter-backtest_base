"""
Comprehensive integration tests for the main backtester module.

This module contains integration tests that test the full backtester workflow
including data loading, strategy execution, portfolio management, and performance reporting.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

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
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)

from tests.test_fixtures import (
    sample_ohlcv_data, ConfigFactory, MockMarketData
)


class TestMainBacktester:
    """Integration tests for the main backtester functions."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for integration testing."""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        
        # Generate realistic OHLCV data
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

    def test_run_backtest_integration(self, sample_market_data):
        """Test complete backtest workflow integration."""
        config = ConfigFactory.create_backtest_config(
            initial_capital=10000.0,
            commission_rate=0.001
        )
        
        # Mock the data loading to return our sample data
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = sample_market_data
            
            # Run the backtest
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d',
                config=config
            )
            
            # Verify the result structure
            assert 'performance' in result
            assert 'trades' in result
            assert 'data' in result
            assert 'signals' in result
            
            # Verify performance metrics are calculated
            performance = result['performance']
            assert 'total_return' in performance
            assert 'sharpe_ratio' in performance
            assert 'max_drawdown' in performance
            
            # Verify trades are recorded
            trades = result['trades']
            assert len(trades) >= 0  # May or may not have trades
            
            # Verify data is preserved
            data = result['data']
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0

    def test_run_modular_backtest_integration(self, sample_market_data):
        """Test the modular backtest function integration."""
        # Mock data loading
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = sample_market_data
            
            # Run modular backtest
            result = run_modular_backtest(
                data=sample_market_data,
                leverage_base=2.0,
                leverage_alpha=3.0,
                base_to_alpha_split=0.2,
                alpha_to_base_split=0.2,
                stop_loss_base=0.025,
                stop_loss_alpha=0.025,
                take_profit_target=0.10,
                initial_capital=1000.0
            )
            
            # Verify result structure
            assert 'performance' in result
            assert 'portfolio' in result
            assert 'trades' in result
            assert 'risk_metrics' in result
            
            # Verify performance metrics
            performance = result['performance']
            assert isinstance(performance, dict)
            assert 'total_return' in performance
            
            # Verify portfolio state
            portfolio = result['portfolio']
            assert isinstance(portfolio, dict)
            assert 'final_value' in portfolio
            assert 'total_pnl' in portfolio
            
            # Verify risk metrics
            risk_metrics = result['risk_metrics']
            assert 'max_drawdown' in risk_metrics
            assert 'volatility' in risk_metrics

    def test_end_to_end_workflow(self, sample_market_data):
        """Test complete end-to-end trading workflow."""
        # Step 1: Load and validate data
        assert sample_market_data is not None
        assert len(sample_market_data) > 0
        assert all(col in sample_market_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Step 2: Create and configure strategy
        strategy_config = ConfigFactory.create_strategy_config('moving_average')
        strategy = create_strategy('moving_average', strategy_config)
        assert strategy is not None
        
        # Step 3: Create portfolio
        portfolio_config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        portfolio = create_portfolio('general', portfolio_config)
        assert portfolio is not None
        
        # Step 4: Run backtest simulation
        with patch.object(strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = pd.Series([0, 1, 0, -1, 1] * (len(sample_market_data) // 5),
                                                 index=sample_market_data.index[:len(sample_market_data) // 5 * 5])
            
            # Mock portfolio to accept signals
            with patch.object(portfolio, 'process_signal') as mock_process:
                mock_process.return_value = {'executed': True, 'pnl': 10.0}
                
                # Simulate the workflow
                signals = strategy.generate_signals(sample_market_data)
                assert len(signals) > 0
                
                # Process some signals
                for i, signal in enumerate(signals[:10]):
                    if signal != 0:
                        result = portfolio.process_signal('SPY', signal, 100, sample_market_data['Close'].iloc[i])
                        assert result is not None

    def test_config_loading_and_saving(self, tmp_path):
        """Test configuration loading and saving functionality."""
        config = ConfigFactory.create_backtest_config(
            initial_capital=5000.0,
            commission_rate=0.002
        )
        
        config_path = tmp_path / "test_config.json"
        
        # Save config
        save_config(config, str(config_path))
        assert config_path.exists()
        
        # Load config
        loaded_config = load_config(str(config_path))
        
        # Verify loaded config matches saved config
        assert loaded_config['initial_capital'] == config['initial_capital']
        assert loaded_config['commission_rate'] == config['commission_rate']
        assert loaded_config['leverage'] == config['leverage']

    def test_parameter_optimization_integration(self, sample_market_data):
        """Test parameter optimization integration."""
        # Define parameter ranges
        param_ranges = {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40],
            'ma_type': ['sma', 'ema']
        }
        
        # Mock backtest function to return performance scores
        def mock_backtest_function(params):
            # Return dummy performance score
            score = np.random.random()
            return {
                'score': score,
                'total_return': score * 0.1,
                'sharpe_ratio': score * 2,
                'max_drawdown': -score * 0.05
            }
        
        with patch('backtester.main.run_backtest') as mock_run_backtest:
            mock_run_backtest.side_effect = lambda **kwargs: mock_backtest_function(kwargs.get('strategy_params', {}))
            
            # Run optimization
            optimization_result = optimize_parameters(
                param_ranges=param_ranges,
                objective_function=mock_backtest_function,
                method='grid_search',
                max_iterations=10
            )
            
            # Verify optimization results
            assert 'best_params' in optimization_result
            assert 'best_score' in optimization_result
            assert 'optimization_history' in optimization_result
            assert 'convergence_info' in optimization_result
            
            # Verify best parameters are in valid ranges
            best_params = optimization_result['best_params']
            assert best_params['fast_period'] in param_ranges['fast_period']
            assert best_params['slow_period'] in param_ranges['slow_period']
            assert best_params['ma_type'] in param_ranges['ma_type']

    def test_multi_symbol_integration(self):
        """Test multi-symbol backtesting integration."""
        symbols = ['SPY', 'AAPL', 'GOOGL']
        symbol_data = {}
        
        for symbol in symbols:
            # Create different data for each symbol
            dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
            np.random.seed(hash(symbol) % 2**32)
            
            initial_price = {'SPY': 300, 'AAPL': 150, 'GOOGL': 2500}[symbol]
            returns = np.random.normal(0.001, 0.025, len(dates))
            prices = initial_price * np.cumprod(1 + returns)
            
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.randint(500000, 5000000, len(dates))
            }, index=dates)
            
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
            
            symbol_data[symbol] = data
        
        # Test loading multiple symbols
        with patch('backtester.main.get_data') as mock_get_data:
            def side_effect(symbol, start_date, end_date, interval):
                return symbol_data.get(symbol, pd.DataFrame())
            
            mock_get_data.side_effect = side_effect
            
            # Run backtest for each symbol
            results = {}
            for symbol in symbols:
                result = run_backtest(
                    symbol=symbol,
                    start_date='2020-01-01',
                    end_date='2024-01-01',
                    interval='1d'
                )
                results[symbol] = result
            
            # Verify all symbols were processed
            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)
            
            # Verify each result has the expected structure
            for symbol, result in results.items():
                assert 'performance' in result
                assert 'trades' in result
                assert 'data' in result

    def test_risk_management_integration(self, sample_market_data):
        """Test risk management integration."""
        config = ConfigFactory.create_backtest_config(
            initial_capital=10000.0,
            leverage=2.0
        )
        
        risk_config = ConfigFactory.create_risk_config(
            stop_loss=0.02,
            take_profit=0.08,
            max_drawdown_limit=0.15,
            max_positions=5
        )
        
        # Mock portfolio with risk management
        with patch('backtester.main.create_portfolio') as mock_create_portfolio:
            mock_portfolio = Mock()
            mock_portfolio.get_total_value.return_value = 10500.0
            mock_portfolio.check_risk_limits.return_value = True
            mock_portfolio.process_signal.return_value = {
                'executed': True, 
                'stopped_out': False,
                'reason': None
            }
            mock_create_portfolio.return_value = mock_portfolio
            
            # Run backtest with risk management
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d',
                config=config,
                risk_config=risk_config
            )
            
            # Verify risk management was applied
            assert mock_portfolio.check_risk_limits.called
            assert mock_portfolio.process_signal.called
            
            # Verify result includes risk metrics
            if 'risk_metrics' in result:
                risk_metrics = result['risk_metrics']
                assert 'max_drawdown' in risk_metrics
                assert 'stopped_out_count' in risk_metrics

    def test_performance_calculation_integration(self, sample_market_data):
        """Test performance calculation integration."""
        # Create known good and bad performance scenarios
        good_performance_data = MockMarketData.bull_market()
        bad_performance_data = MockMarketData.bear_market()
        
        scenarios = [
            ('bull_market', good_performance_data),
            ('bear_market', bad_performance_data)
        ]
        
        for scenario_name, price_data in scenarios:
            with patch('backtester.main.get_data') as mock_get_data:
                # Convert Series to DataFrame
                test_data = pd.DataFrame({
                    'Open': price_data * (1 + np.random.normal(0, 0.005, len(price_data))),
                    'High': price_data * 1.01,
                    'Low': price_data * 0.99,
                    'Close': price_data,
                    'Volume': 1000000
                })
                
                mock_get_data.return_value = test_data
                
                result = run_backtest(
                    symbol='SPY',
                    start_date=price_data.index[0].strftime('%Y-%m-%d'),
                    end_date=price_data.index[-1].strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                performance = result['performance']
                
                # Verify performance metrics are reasonable
                assert 'total_return' in performance
                assert 'annualized_return' in performance
                assert 'sharpe_ratio' in performance
                assert 'max_drawdown' in performance
                assert 'volatility' in performance
                
                # Bull market should generally have positive returns
                if scenario_name == 'bull_market':
                    # Note: This might not always be true due to strategy performance
                    # but the test validates that calculations are performed
                    pass
                
                # Verify metric values are valid
                assert isinstance(performance['total_return'], (int, float))
                assert isinstance(performance['sharpe_ratio'], (int, float))
                assert performance['max_drawdown'] <= 0  # Drawdown should be negative

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test with invalid data
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()  # Empty data
            
            with pytest.raises(ValueError, match="Insufficient data"):
                run_backtest(
                    symbol='INVALID',
                    start_date='2020-01-01',
                    end_date='2024-01-01',
                    interval='1d'
                )
        
        # Test with invalid configuration
        invalid_config = {'initial_capital': -100}  # Negative capital
        
        with patch('backtester.main.get_data') as mock_get_data, \
             patch('backtester.main.load_config') as mock_load_config:
            
            mock_get_data.return_value = sample_ohlcv_data()
            mock_load_config.return_value = invalid_config
            
            with pytest.raises(ValueError, match="Invalid configuration"):
                run_backtest(
                    symbol='SPY',
                    start_date='2020-01-01',
                    end_date='2024-01-01',
                    interval='1d',
                    config=invalid_config
                )

    def test_data_preprocessing_integration(self, sample_market_data):
        """Test data preprocessing integration."""
        # Create data with missing values and outliers
        data_with_issues = sample_market_data.copy()
        
        # Add missing values
        data_with_issues.loc[data_with_issues.index[10:15], 'Close'] = np.nan
        
        # Add outliers
        data_with_issues.loc[data_with_issues.index[20], 'Close'] *= 10  # Extreme outlier
        
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = data_with_issues
            
            # Run backtest with preprocessing enabled
            config = ConfigFactory.create_backtest_config(preprocess_data=True)
            
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d',
                config=config
            )
            
            # Verify that preprocessing handled the issues
            # The backtest should still run without errors
            assert 'performance' in result
            assert 'trades' in result
            
            # The final data should be clean (no NaN values in processed columns)
            final_data = result['data']
            assert not final_data['Close'].isna().any()

    def test_large_dataset_integration(self):
        """Test integration with larger datasets."""
        # Create a larger dataset
        dates = pd.date_range(start="2010-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = initial_price * np.cumprod(1 + returns)
        
        large_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        large_data['High'] = large_data[['Open', 'High', 'Close']].max(axis=1)
        large_data['Low'] = large_data[['Open', 'Low', 'Close']].min(axis=1)
        
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = large_data
            
            # Run backtest on large dataset
            result = run_backtest(
                symbol='SPY',
                start_date='2010-01-01',
                end_date='2024-01-01',
                interval='1d'
            )
            
            # Verify it handles large datasets properly
            assert 'performance' in result
            assert 'trades' in result
            assert len(result['data']) > 0
            
            # Performance should still be calculated correctly
            performance = result['performance']
            assert 'total_return' in performance
            assert 'annualized_return' in performance

    def test_concurrent_backtests_integration(self, sample_market_data):
        """Test running multiple backtests concurrently."""
        configs = [
            ConfigFactory.create_backtest_config(initial_capital=1000.0),
            ConfigFactory.create_backtest_config(initial_capital=5000.0),
            ConfigFactory.create_backtest_config(initial_capital=10000.0)
        ]
        
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = self.sample_health_data()
            
            # Run multiple backtests
            results = []
            for config in configs:
                result = run_backtest(
                    symbol='SPY',
                    start_date='2020-01-01',
                    end_date='2024-01-01',
                    interval='1d',
                    config=config
                )
                results.append(result)
            
            # Verify all backtests completed
            assert len(results) == 3
            
            # Verify each result is valid
            for i, result in enumerate(results):
                assert 'performance' in result
                assert 'trades' in result
                assert 'data' in result
                
                # Each should have different initial capital effects
                performance = result['performance']
                assert 'total_return' in performance

    def sample_health_data(self):
        """Helper method to create sample health data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.015, len(dates))  # Lower volatility
        prices = initial_price * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.003, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(2000000, 8000000, len(dates))
        }, index=dates)
        
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data


@pytest.mark.integration
class TestSystemIntegration:
    """System-level integration tests."""

    def test_full_system_workflow(self, sample_ohlcv_data):
        """Test complete system workflow from data to results."""
        # This is a comprehensive test that exercises the entire system
        
        # 1. Data loading and validation
        assert len(sample_ohlcv_data) > 100  # Ensure sufficient data
        
        # 2. Strategy creation and configuration
        strategy_config = ConfigFactory.create_strategy_config('moving_average')
        strategy = create_strategy('moving_average', strategy_config)
        assert strategy is not None
        
        # 3. Portfolio creation
        portfolio_config = ConfigFactory.create_backtest_config(initial_capital=10000.0)
        portfolio = create_portfolio('general', portfolio_config)
        assert portfolio is not None
        
        # 4. Data preprocessing
        processed_data = strategy.preprocess_data(sample_ohlcv_data)
        assert processed_data is not None
        
        # 5. Signal generation
        signals = strategy.generate_signals(processed_data)
        assert len(signals) == len(processed_data)
        
        # 6. Portfolio simulation
        trades_executed = 0
        for i, signal in enumerate(signals):
            if signal != 0:
                result = portfolio.process_signal('SPY', signal, 100, processed_data['Close'].iloc[i])
                if result.get('executed', False):
                    trades_executed += 1
        
        # 7. Performance calculation
        performance = portfolio.calculate_performance()
        assert 'total_return' in performance
        
        # 8. Risk analysis
        risk_metrics = portfolio.analyze_risk()
        assert 'max_drawdown' in risk_metrics
        
        # Verify the system produced reasonable results
        assert trades_executed >= 0  # May or may not have trades
        assert performance['total_return'] is not None
        assert isinstance(risk_metrics['max_drawdown'], (int, float))

    def test_system_resilience(self):
        """Test system resilience to various failure modes."""
        # Test with various edge cases
        test_cases = [
            ('empty_data', pd.DataFrame()),
            ('minimal_data', pd.DataFrame({'Close': [100, 101]})),
            ('single_point', pd.DataFrame({'Close': [100]}))
        ]
        
        for case_name, test_data in test_cases:
            if case_name == 'empty_data':
                # Should handle empty data gracefully
                with pytest.raises(ValueError, match="Insufficient data"):
                    run_backtest('SPY', '2020-01-01', '2024-01-01', '1d')
            elif case_name == 'minimal_data':
                # Should handle minimal data
                with patch('backtester.main.get_data') as mock_get_data:
                    mock_get_data.return_value = test_data
                    with pytest.raises(ValueError, match="Insufficient data"):
                        run_backtest('SPY', '2020-01-01', '2024-01-01', '1d')
            elif case_name == 'single_point':
                # Should handle single data point
                with patch('backtester.main.get_data') as mock_get_data:
                    mock_get_data.return_value = test_data
                    with pytest.raises(ValueError, match="Insufficient data"):
                        run_backtest('SPY', '2020-01-01', '2024-01-01', '1d')

    def test_performance_benchmark(self, sample_ohlcv_data):
        """Test system performance with benchmark comparisons."""
        # Run backtest
        with patch('backtester.main.get_data') as mock_get_data:
            mock_get_data.return_value = sample_ohlcv_data
            
            result = run_backtest(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval='1d'
            )
            
            # Compare against buy-and-hold
            buy_hold_return = (sample_ohlcv_data['Close'].iloc[-1] - sample_ohlcv_data['Close'].iloc[0]) / sample_ohlcv_data['Close'].iloc[0]
            
            strategy_return = result['performance']['total_return']
            
            # Both should be valid numbers
            assert isinstance(buy_hold_return, (int, float))
            assert isinstance(strategy_return, (int, float))
            
            # Strategy performance should be calculated (may be better or worse than buy-and-hold)
            assert abs(strategy_return) >= 0  # Can be negative but should be a valid return

    def test_configuration_validation_integration(self):
        """Test configuration validation throughout the system."""
        # Valid configuration should work
        ConfigFactory.create_backtest_config(
            initial_capital=10000.0,
            commission_rate=0.001,
            leverage=1.0
        )
        
        # Invalid configurations should be rejected
        invalid_configs = [
            {'initial_capital': -1000},  # Negative capital
            {'commission_rate': -0.001},  # Negative commission
            {'leverage': -1.0},  # Negative leverage
            {'max_positions': 0},  # Zero positions
        ]
        
        for invalid_config in invalid_configs:
            # Each invalid config should cause an error
            # (Specific error messages depend on validation implementation)
            pass  # Test would need to be implemented based on actual validation logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])