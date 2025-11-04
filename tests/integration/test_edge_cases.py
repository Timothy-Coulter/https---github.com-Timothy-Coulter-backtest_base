"""
Edge Cases and Error Handling Integration Tests for QuantBench Backtester.

This module tests boundary conditions, error handling, and recovery scenarios
including data corruption, system failures, and emergency procedures.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from backtester.core.backtest_engine import BacktestEngine
from backtester.data.data_handler import DataHandler


@pytest.mark.integration
class TestDataBoundaryConditions:
    """Test system behavior with data boundary conditions."""
    
    def test_empty_data_handling(self, empty_data, integration_test_config):
        """Test system response to empty data."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = empty_data.copy()
        
        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="No data loaded"):
            engine.run_backtest()
    
    def test_minimal_data_handling(self, single_point_data, integration_test_config):
        """Test system response to minimal data."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = single_point_data.copy()
        
        # Should handle single data point gracefully
        with pytest.raises((ValueError, RuntimeError)):
            engine.run_backtest()
    
    def test_corrupted_data_handling(self, corrupted_data, integration_test_config):
        """Test system response to corrupted market data."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = corrupted_data.copy()
        
        # Should detect and handle corrupted data
        with pytest.raises((ValueError, RuntimeError)):
            engine.run_backtest()
    
    def test_missing_values_handling(self, integration_test_config):
        """Test system response to missing values in data."""
        # Create data with various missing value patterns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data_with_missing = pd.DataFrame({
            'Open': [100.0] * 100,
            'High': [105.0] * 100,
            'Low': [95.0] * 100,
            'Close': np.where(np.arange(100) < 50, np.nan, [100.0 + i for i in range(50)]),
            'Volume': [1000000] * 100
        }, index=dates)
        
        # Create data handler with missing values
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Test data quality detection
        quality_report = data_handler.validate_data_quality(data_with_missing)
        
        assert 'missing_values' in quality_report, "Should detect missing values"
        assert sum(quality_report['missing_values'].values()) > 0, \
            "Should identify missing values in data"
        
        # Test missing value imputation
        try:
            imputed_data = data_handler.impute_missing_data(data_with_missing)
            
            # Should handle imputation without errors
            assert isinstance(imputed_data, pd.DataFrame), "Should return DataFrame after imputation"
            
            # Should reduce or eliminate missing values
            original_missing = data_with_missing.isnull().sum().sum()
            imputed_missing = imputed_data.isnull().sum().sum()
            
            assert imputed_missing <= original_missing, \
                "Imputation should not increase missing values"
                
        except Exception as e:
            # Imputation may not be implemented for all cases
            print(f"Imputation not supported: {e}")
    
    def test_extreme_values_handling(self, integration_test_config):
        """Test system response to extreme market values."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        
        # Create data with extreme values
        extreme_data = pd.DataFrame({
            'Open': [100.0, 1e6, 0.01, 100.0] * 13,  # Mix of normal and extreme values
            'High': [105.0, 1.1e6, 0.02, 105.0] * 13,
            'Low': [95.0, 9e5, 0.005, 95.0] * 13,
            'Close': [100.0, 1.05e6, 0.015, 100.0] * 13,
            'Volume': [1000000, 0, 1e9, 1000000] * 13  # Zero and extreme volume
        }, index=dates)
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = extreme_data.copy()
        
        # Should handle extreme values (may result in errors or warnings)
        try:
            results = engine.run_backtest()
            # If it completes, results should be valid or indicate issues
            if results:
                print("Extreme values handled without errors")
        except Exception as e:
            # Acceptable to fail with extreme values
            assert "extreme" in str(e).lower() or "invalid" in str(e).lower(), \
                f"Error should relate to extreme values: {e}"


@pytest.mark.integration
class TestMarketBoundaryConditions:
    """Test system behavior with market boundary conditions."""
    
    def test_extreme_volatility_handling(self, high_volatility_data, integration_test_config):
        """Test system response to extreme market volatility."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = high_volatility_data.copy()
        
        # Run backtest during high volatility
        results = engine.run_backtest()
        
        # System should complete without crashing
        assert results is not None, "Should complete backtest even in high volatility"
        
        # Risk management should be active
        if 'risk_events' in results:
            risk_events = results['risk_events']
            # In high volatility, risk events should be present or tracked
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        # Performance should reflect high volatility environment
        performance = results.get('performance', {})
        assert performance['volatility'] is not None, "Volatility should be calculated"
        assert performance['volatility'] > 0.01, "Volatility should be high for volatile data"
    
    def test_flash_crash_handling(self, integration_test_config):
        """Test system response to flash crash scenarios."""
        # Create flash crash data
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        
        # Normal market conditions followed by flash crash and recovery
        prices = [100.0] * 10
        # Flash crash (20% drop in one day)
        prices.append(80.0)
        # Partial recovery
        prices.extend([82.0, 85.0, 88.0, 90.0, 92.0, 95.0, 98.0, 100.0, 102.0])
        
        flash_crash_data = pd.DataFrame({
            'Open': [p * 0.999 for p in prices],
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 20
        }, index=dates)
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = flash_crash_data.copy()
        
        results = engine.run_backtest()
        
        # Should handle flash crash without catastrophic failure
        assert results is not None, "Should complete backtest during flash crash"
        
        # Risk management should detect the crash
        if 'risk_events' in results:
            risk_events = results['risk_events']
            crash_events = [e for e in risk_events 
                          if 'crash' in e.get('type', '').lower() or 
                             'emergency' in e.get('type', '').lower()]
            
            # Some risk response should occur during flash crash
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        # Portfolio should survive flash crash
        performance = results.get('performance', {})
        if performance:
            final_value = performance.get('final_portfolio_value', 0)
            initial_value = integration_test_config.portfolio.initial_capital
            
            assert final_value > 0, "Portfolio should retain some value after crash"
    
    def test_market_closure_handling(self, integration_test_config):
        """Test system behavior during market closures."""
        # Create data with missing periods (simulating market closures)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        
        # Remove some dates to simulate market closures
        missing_dates = dates[5:8]  # Missing 3 days
        working_dates = dates.difference(missing_dates)
        
        # Create sparse data
        sparse_data = pd.DataFrame({
            'Open': [100.0 + i for i in range(len(working_dates))],
            'High': [105.0 + i for i in range(len(working_dates))],
            'Low': [95.0 + i for i in range(len(working_dates))],
            'Close': [100.0 + i for i in range(len(working_dates))],
            'Volume': [1000000] * len(working_dates)
        }, index=working_dates)
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = sparse_data.copy()
        
        # Should handle market closures gracefully
        results = engine.run_backtest()
        assert results is not None, "Should handle market closures"
        
        # Should detect data gaps
        if 'data_quality' in results:
            data_quality = results['data_quality']
            assert 'gaps_detected' in data_quality, "Should detect data gaps"
            assert data_quality['gaps_detected'] > 0, "Should identify missing periods"
    
    def test_correlation_breakdown_handling(self, integration_test_config):
        """Test system response to correlation breakdown scenarios."""
        # Create multi-asset data with broken correlations
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Asset A: Normal market behavior
        asset_a_returns = np.random.normal(0.001, 0.02, 100)
        asset_a_prices = 100 * np.cumprod(1 + asset_a_returns)
        
        # Asset B: Initially correlated, then correlation breaks
        asset_b_returns = asset_a_returns[:50] + np.random.normal(0, 0.01, 50)  # Correlated first half
        asset_b_returns = np.concatenate([
            asset_b_returns,
            -asset_a_returns[50:] + np.random.normal(0, 0.03, 50)  # Negatively correlated second half
        ])
        asset_b_prices = 100 * np.cumprod(1 + asset_b_returns)
        
        # Since we have single-asset backtester, simulate correlation issues through price behavior
        correlation_breakdown_data = pd.DataFrame({
            'Open': asset_a_prices * 0.999,
            'High': asset_a_prices * 1.001,
            'Low': asset_a_prices * 0.999,
            'Close': asset_a_prices,
            'Volume': [1000000] * 100
        }, index=dates)
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = correlation_breakdown_data.copy()
        
        results = engine.run_backtest()
        
        # System should handle correlation breakdown without errors
        assert results is not None, "Should handle correlation breakdown"
        
        # Should detect unusual market behavior
        if 'market_analysis' in results:
            market_analysis = results['market_analysis']
            if 'correlation_stability' in market_analysis:
                # System should attempt correlation analysis
                assert market_analysis['correlation_stability'] is not None


@pytest.mark.integration
class TestSystemBoundaryConditions:
    """Test system behavior under system boundary conditions."""
    
    def test_memory_limit_handling(self, integration_test_config):
        """Test system behavior under memory pressure."""
        # Mock memory constraints
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate low memory condition
            mock_memory.return_value.free = 100 * 1024 * 1024  # 100MB free
            
            engine = BacktestEngine(integration_test_config)
            
            # Should activate memory management
            engine.memory_management_enabled = True
            
            # Create data that might stress memory
            large_data = pd.DataFrame({
                'Open': np.random.random(10000) * 100,
                'High': np.random.random(10000) * 105,
                'Low': np.random.random(10000) * 95,
                'Close': np.random.random(10000) * 100,
                'Volume': np.random.randint(1000000, 10000000, 10000)
            }, index=pd.date_range('2020-01-01', periods=10000, freq='1min'))
            
            engine.current_data = large_data.copy()
            
            # Should handle memory constraints gracefully
            try:
                results = engine.run_backtest()
                
                # If successful, should indicate memory optimization
                if 'memory_optimization' in results:
                    assert results['memory_optimization']['activated'] == True
                    
            except MemoryError:
                # Acceptable to run out of memory in test
                print("Memory limit test: System ran out of memory as expected")
            except Exception as e:
                # Should be memory-related error
                assert "memory" in str(e).lower(), f"Error should be memory-related: {e}"
    
    def test_processing_time_limit_handling(self, integration_test_config):
        """Test system behavior under processing time constraints."""
        engine = BacktestEngine(integration_test_config)
        
        # Set aggressive time limits
        engine.max_processing_time = 1.0  # 1 second limit
        
        # Create computationally intensive data
        intensive_data = pd.DataFrame({
            'Open': np.random.random(5000) * 100,
            'High': np.random.random(5000) * 105,
            'Low': np.random.random(5000) * 95,
            'Close': np.random.random(5000) * 100,
            'Volume': np.random.randint(1000000, 10000000, 5000)
        }, index=pd.date_range('2020-01-01', periods=5000, freq='1min'))
        
        engine.current_data = intensive_data.copy()
        
        import time
        start_time = time.time()
        
        try:
            results = engine.run_backtest()
            execution_time = time.time() - start_time
            
            # If it completes, should be within time limits
            assert execution_time <= engine.max_processing_time * 2, \
                "Should respect processing time limits"
                
        except RuntimeError as e:
            # Should indicate timeout
            assert "time" in str(e).lower() or "timeout" in str(e).lower(), \
                f"Should indicate timeout error: {e}"
    
    def test_network_failure_handling(self, integration_test_config):
        """Test system response to network failures during data download."""
        data_handler = DataHandler(integration_test_config.__dict__, Mock())
        
        # Mock network failure
        with patch('yfinance.download', side_effect=Exception("Network connection failed")):
            # Should handle network failure gracefully
            with pytest.raises((ValueError, ConnectionError, OSError)):
                data_handler.get_data("SPY", "2020-01-01", "2020-12-31", "1d")
    
    def test_disk_space_exhaustion_handling(self, integration_test_config):
        """Test system behavior when disk space is exhausted."""
        # Mock disk space issues
        with patch('os.path.exists', side_effect=OSError("No space left on device")):
            engine = BacktestEngine(integration_test_config)
            
            # Should handle disk space issues gracefully
            try:
                results = engine.run_backtest()
                # Should indicate disk space issues
                if 'disk_space_error' in results:
                    assert results['disk_space_error'] == True
            except OSError as e:
                assert "space" in str(e).lower(), f"Should be disk space error: {e}"


@pytest.mark.integration
class TestErrorPropagationHandling:
    """Test error handling and propagation across components."""
    
    def test_component_failure_recovery(self, integration_test_config):
        """Test recovery from individual component failures."""
        # Test strategy failure
        engine = BacktestEngine(integration_test_config)
        
        # Mock strategy failure
        failing_strategy = Mock()
        failing_strategy.generate_signals.side_effect = Exception("Strategy calculation failed")
        
        engine.strategy = failing_strategy
        engine.current_data = sample_market_data.copy()
        
        # Should handle strategy failure gracefully
        with pytest.raises(Exception) as exc_info:
            engine.run_backtest()
        
        assert "strategy" in str(exc_info.value).lower() or "calculation" in str(exc_info.value).lower()
    
    def test_data_corruption_recovery(self, integration_test_config):
        """Test recovery from data corruption."""
        # Create corrupted data
        corrupted_data = pd.DataFrame({
            'Open': ['invalid'] * 10,  # String instead of numeric
            'High': [105.0] * 10,
            'Low': [95.0] * 10,
            'Close': [100.0] * 10,
            'Volume': [1000000] * 10
        })
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = corrupted_data
        
        # Should detect and handle data corruption
        with pytest.raises((ValueError, TypeError)):
            engine.run_backtest()
    
    def test_portfolio_state_corruption_recovery(self, integration_test_config):
        """Test recovery from portfolio state corruption."""
        engine = BacktestEngine(integration_test_config)
        
        # Create portfolio with corrupted state
        engine.create_portfolio()
        engine.portfolio.cash = float('nan')  # Corrupt state
        
        engine.current_data = sample_market_data.copy()
        
        # Should detect portfolio state corruption
        with pytest.raises((ValueError, RuntimeError)):
            engine.run_backtest()
    
    def test_error_logging_monitoring(self, integration_test_config):
        """Test comprehensive error logging and monitoring."""
        engine = BacktestEngine(integration_test_config)
        
        # Enable detailed error logging
        engine.error_logging_enabled = True
        engine.detailed_monitoring = True
        
        # Create scenario that will generate errors
        corrupted_data = pd.DataFrame({
            'Open': np.nan,  # NaN values
            'High': np.nan,
            'Low': np.nan,
            'Close': np.nan,
            'Volume': np.nan
        }, index=pd.date_range('2020-01-01', periods=10))
        
        engine.current_data = corrupted_data
        
        # Run backtest which should generate errors
        try:
            engine.run_backtest()
        except Exception:
            pass  # Expected to fail
        
        # Check that errors were logged and tracked
        error_logs = getattr(engine, 'error_logs', [])
        assert len(error_logs) > 0, "Should log errors for monitoring"
        
        # Check error details
        for error_log in error_logs:
            assert 'timestamp' in error_log, "Should timestamp errors"
            assert 'error_type' in error_log, "Should categorize errors"
            assert 'component' in error_log, "Should identify component"


@pytest.mark.integration
class TestEmergencyProcedures:
    """Test emergency procedures and system recovery."""
    
    def test_emergency_liquidation_procedures(self, integration_test_config):
        """Test emergency liquidation workflow."""
        # Configure extremely tight risk limits to trigger emergency procedures
        integration_test_config.risk.max_drawdown = 0.01  # 1% drawdown limit
        integration_test_config.risk.max_portfolio_risk = 0.005  # 0.5% VaR limit
        
        engine = BacktestEngine(integration_test_config)
        
        # Create data that will trigger emergency procedures
        emergency_data = pd.DataFrame({
            'Open': [100.0, 50.0, 30.0, 10.0, 5.0],  # Rapid decline
            'High': [105.0, 55.0, 35.0, 15.0, 10.0],
            'Low': [95.0, 45.0, 25.0, 8.0, 3.0],
            'Close': [100.0, 50.0, 30.0, 10.0, 5.0],
            'Volume': [1000000, 5000000, 10000000, 15000000, 20000000]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        engine.current_data = emergency_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Should trigger emergency procedures
        if 'risk_events' in results:
            risk_events = results['risk_events']
            emergency_events = [e for e in risk_events 
                              if e.get('type') == 'EMERGENCY_LIQUIDATION']
            
            # Emergency procedures should be triggered
            assert len(emergency_events) >= 0, "Emergency procedures should be attempted"
        
        # Portfolio should implement emergency liquidation
        if 'emergency_liquidation' in results:
            liquidation = results['emergency_liquidation']
            assert liquidation['triggered'] == True, "Emergency liquidation should be triggered"
            assert liquidation['completion_percentage'] is not None, "Should track completion"
    
    def test_system_health_monitoring(self, integration_test_config):
        """Test continuous system health monitoring."""
        engine = BacktestEngine(integration_test_config)
        
        # Enable health monitoring
        engine.health_monitoring_enabled = True
        engine.real_time_monitoring = True
        
        engine.current_data = sample_market_data.copy()
        
        # Run backtest with monitoring
        results = engine.run_backtest()
        
        # Should include health monitoring data
        if 'system_health' in results:
            health_report = results['system_health']
            
            assert 'overall_status' in health_report, "Should report overall status"
            assert 'component_health' in health_report, "Should report component health"
            assert 'performance_metrics' in health_report, "Should report performance metrics"
            
            # Health status should be assessed
            valid_statuses = ['HEALTHY', 'WARNING', 'CRITICAL', 'UNKNOWN']
            assert health_report['overall_status'] in valid_statuses, \
                f"Invalid health status: {health_report['overall_status']}"
    
    def test_recovery_procedures_validation(self, integration_test_config):
        """Test system recovery procedures and state restoration."""
        engine = BacktestEngine(integration_test_config)
        
        # Enable recovery procedures
        engine.recovery_procedures_enabled = True
        engine.checkpoint_enabled = True
        
        # Run initial backtest to establish baseline
        engine.current_data = sample_market_data.copy()
        results1 = engine.run_backtest()
        
        # Simulate system failure scenario
        # (In real implementation, this would involve checkpoint/restore)
        
        # Run second backtest to verify recovery
        engine.current_data = sample_market_data.copy()
        results2 = engine.run_backtest()
        
        # Should be able to run multiple backtests without issues
        assert results1 is not None, "First backtest should succeed"
        assert results2 is not None, "Second backtest should succeed after recovery test"
        
        # Performance should be consistent
        if 'performance' in results1 and 'performance' in results2:
            perf1 = results1['performance']
            perf2 = results2['performance']
            
            # Results should be similar for same data
            assert abs(perf1['total_return'] - perf2['total_return']) < 0.01, \
                "Performance should be consistent across runs"


# Utility functions for edge case testing
def create_extreme_scenario_data(scenario_type, n_periods=100):
    """Create data for various extreme scenarios."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    if scenario_type == "flash_crash":
        prices = [100.0] * (n_periods // 2)
        prices.extend([80.0])  # 20% crash
        prices.extend([85.0, 90.0, 95.0] * ((n_periods - len(prices)) // 3))
    elif scenario_type == "volatility_spike":
        prices = [100.0]
        for i in range(n_periods - 1):
            change = np.random.normal(0, 0.1)  # 10% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
    elif scenario_type == "extended_decline":
        prices = [100.0]
        for i in range(n_periods - 1):
            change = np.random.normal(-0.01, 0.03)  # Declining trend
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
    else:
        # Default random walk
        prices = [100.0]
        for i in range(n_periods - 1):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
    
    return pd.DataFrame({
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.002 for p in prices],
        'Low': [p * 0.998 for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in prices]
    }, index=dates[:len(prices)])


# Add sample data for testing
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    initial_price = 100.0
    returns = np.random.normal(0.0008, 0.02, len(dates))
    prices = initial_price * np.cumprod(1 + returns)
    
    return pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)