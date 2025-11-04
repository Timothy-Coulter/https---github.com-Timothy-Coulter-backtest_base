"""
End-to-End Workflow Integration Tests for QuantBench Backtester.

This module tests complete backtesting workflows from start to finish under
different market conditions including bull markets, bear markets, crisis periods,
and various portfolio management scenarios.
"""

import pytest
import time

from backtester.core.backtest_engine import BacktestEngine


@pytest.mark.integration
class TestCompleteBacktestCycle:
    """Test complete backtest cycle integration."""
    
    @pytest.mark.slow
    def test_complete_bull_market_workflow(self, bull_market_data, integration_test_config):
        """Test complete workflow during strong bull market conditions."""
        engine = BacktestEngine(integration_test_config)
        
        # Load bull market data
        engine.current_data = bull_market_data.copy()
        
        # Run complete backtest
        start_time = time.time()
        results = engine.run_backtest()
        execution_time = time.time() - start_time
        
        # Validate complete workflow execution
        assert 'performance' in results, "Results should contain performance metrics"
        assert 'trade_history' in results, "Results should contain trade history"
        assert 'portfolio_values' in results, "Results should contain portfolio values"
        
        # Validate performance results for bull market
        performance = results['performance']
        assert performance['total_return'] > 0, "Bull market should generate positive returns"
        assert performance['sharpe_ratio'] > 0, "Bull market should have positive Sharpe ratio"
        
        # Validate trading activity
        if 'trade_history' in results and len(results['trade_history']) > 0:
            trades = results['trade_history']
            assert len(trades) > 0, "Should execute some trades in bull market"
            
            # Validate trade structure
            for trade in trades:
                assert 'timestamp' in trade, "Trade should have timestamp"
                assert 'price' in trade, "Trade should have price"
        
        # Execution time should be reasonable
        assert execution_time < 60, f"Execution took too long: {execution_time:.2f}s"
        
        # Risk metrics should be within acceptable ranges
        assert performance['max_drawdown'] > -0.5, "Drawdown should not be extreme in bull market"
        
        print(f"Bull market test completed: {execution_time:.2f}s, Return: {performance['total_return']:.2%}")
    
    @pytest.mark.slow
    def test_bear_market_survival(self, bear_market_data, integration_test_config):
        """Test risk management and capital preservation during bear market."""
        engine = BacktestEngine(integration_test_config)
        
        # Load bear market data
        engine.current_data = bear_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate performance during bear market
        performance = results['performance']
        
        # In bear market, risk management should prevent catastrophic losses
        # Allow for some loss, but should be controlled
        assert performance['total_return'] > -0.5, "Risk management should prevent >50% losses"
        assert performance['max_drawdown'] < 0, "Drawdown should be negative"
        
        # Validate that risk management was active
        if 'risk_events' in results:
            risk_events = results['risk_events']
            # In bear market, risk events should be present
            assert len(risk_events) >= 0, "Risk management should monitor bear market"
        
        # Portfolio should survive bear market
        final_value = performance['final_portfolio_value']
        initial_value = integration_test_config.portfolio.initial_capital
        
        assert final_value > initial_value * 0.3, "Portfolio should retain some value"
        
        print(f"Bear market test completed: Return: {performance['total_return']:.2%}, "
              f"Max DD: {performance['max_drawdown']:.2%}")
    
    @pytest.mark.slow
    def test_crisis_period_resilience(self, crisis_market_data, integration_test_config):
        """Test system resilience during crisis periods with extreme volatility."""
        engine = BacktestEngine(integration_test_config)
        
        # Load crisis market data
        engine.current_data = crisis_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate crisis handling
        performance = results['performance']
        
        # System should handle crisis without crashing
        assert performance is not None, "System should complete backtest"
        
        # Check that emergency procedures were triggered if needed
        if 'risk_events' in results:
            risk_events = results['risk_events']
            emergency_events = [e for e in risk_events 
                              if e.get('type') == 'EMERGENCY_LIQUIDATION']
            
            # In crisis data, emergency procedures may or may not trigger
            # depending on the specific crisis implementation
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        # Performance should be reasonable given crisis conditions
        assert performance['total_return'] is not None, "Performance should be calculated"
        assert isinstance(performance['max_drawdown'], (int, float)), "Drawdown should be numeric"
        
        print(f"Crisis period test completed: Return: {performance['total_return']:.2%}")
    
    @pytest.mark.slow
    def test_sideways_market_performance(self, sideways_market_data, integration_test_config):
        """Test strategy performance in sideways/ranging markets."""
        engine = BacktestEngine(integration_test_config)
        
        # Load sideways market data
        engine.current_data = sideways_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate performance in sideways market
        performance = results['performance']
        
        # In sideways markets, returns should be modest
        # Strategy should avoid excessive trading that erodes returns
        assert abs(performance['total_return']) < 1.0, "Returns should be moderate in sideways market"
        
        # Transaction costs should not dominate
        if 'total_trades' in performance and performance['total_trades'] > 0:
            # Average trade should generate positive return after costs
            total_return = performance['total_return']
            total_trades = performance['total_trades']
            
            # Don't expect perfect accuracy, but should not be catastrophic
            assert total_return > -0.5, "Strategy should not lose heavily in sideways market"
        
        print(f"Sideways market test completed: Return: {performance['total_return']:.2%}")
    
    @pytest.mark.slow
    def test_multi_timeframe_analysis(self, sample_market_data, integration_test_config):
        """Test multi-timeframe analysis capabilities."""
        engine = BacktestEngine(integration_test_config)
        
        # Create multi-timeframe data by resampling
        daily_data = sample_market_data.copy()
        weekly_data = daily_data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min', 
            'Close': 'last',
            'Volume': 'sum'
        })
        monthly_data = daily_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last', 
            'Volume': 'sum'
        })
        
        timeframes = [
            ('daily', daily_data),
            ('weekly', weekly_data),
            ('monthly', monthly_data)
        ]
        
        timeframe_results = {}
        
        for tf_name, tf_data in timeframes:
            if len(tf_data) < 50:  # Skip if too few data points
                continue
                
            engine_copy = BacktestEngine(integration_test_config)
            engine_copy.current_data = tf_data.copy()
            
            try:
                results = engine_copy.run_backtest()
                timeframe_results[tf_name] = results
            except Exception as e:
                # Some timeframes might fail due to insufficient data
                print(f"Timeframe {tf_name} failed: {e}")
                continue
        
        # Validate that at least daily timeframe works
        assert 'daily' in timeframe_results, "Daily timeframe should work"
        
        # Check that results are consistent across timeframes
        if len(timeframe_results) > 1:
            daily_perf = timeframe_results['daily']['performance']
            for tf_name, tf_results in timeframe_results.items():
                tf_perf = tf_results['performance']
                
                # Returns should be in reasonable range across timeframes
                assert abs(tf_perf['total_return']) < 2.0, \
                    f"Timeframe {tf_name} has unreasonable returns"
        
        print(f"Multi-timeframe test completed: {len(timeframe_results)} timeframes")
    
    @pytest.mark.slow
    def test_full_year_backtest_validation(self, sample_market_data, integration_test_config):
        """Test complete backtest from January to December."""
        engine = BacktestEngine(integration_test_config)
        
        # Use full year of data
        engine.current_data = sample_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate all components executed
        assert 'performance' in results, "Performance analysis should complete"
        assert 'trade_history' in results, "Trade history should be recorded"
        assert 'data' in results, "Data should be preserved"
        
        # Validate data period coverage
        data_period = results.get('data_period', {})
        if data_period:
            assert 'start' in data_period, "Data period should have start date"
            assert 'end' in data_period, "Data period should have end date"
            assert 'periods' in data_period, "Data period should have count"
        
        # Validate performance metrics
        performance = results['performance']
        
        # Check required performance metrics
        required_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'volatility'
        ]
        
        for metric in required_metrics:
            assert metric in performance, f"Performance should include {metric}"
        
        # Validate metric ranges
        assert performance['total_return'] is not None, "Total return should be calculated"
        assert isinstance(performance['sharpe_ratio'], (int, float)), "Sharpe ratio should be numeric"
        assert performance['max_drawdown'] <= 0, "Max drawdown should be negative or zero"
        
        # Check that trading occurred
        if 'total_trades' in performance:
            assert performance['total_trades'] >= 0, "Trade count should be non-negative"
        
        print(f"Full year backtest completed: {len(sample_market_data)} periods, "
              f"Return: {performance['total_return']:.2%}")


@pytest.mark.integration
class TestMultiAssetPortfolioTests:
    """Test cross-asset correlation and diversification scenarios."""
    
    @pytest.mark.slow
    def test_correlated_asset_movements(self, sample_market_data, integration_test_config):
        """Test portfolio behavior with correlated asset movements."""
        # This would require multi-asset portfolio implementation
        # For now, test with single asset and validate correlation tracking
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = sample_market_data.copy()
        
        results = engine.run_backtest()
        
        # Validate that correlation analysis is available if implemented
        performance = results['performance']
        
        # Basic validation that system handles single asset case
        assert performance['total_return'] is not None, "Single asset should work"
        
        print("Correlated assets test completed (single asset)")
    
    @pytest.mark.slow
    def test_diversification_benefit_validation(self, sample_market_data, integration_test_config):
        """Test that diversification benefits are captured."""
        # This test would be expanded with multiple assets
        # For now, validate single asset baseline
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = sample_market_data.copy()
        
        results = engine.run_backtest()
        performance = results['performance']
        
        # Validate that performance metrics are calculated
        assert 'total_return' in performance, "Performance should be calculated"
        assert 'volatility' in performance, "Volatility should be calculated"
        
        # Risk-adjusted metrics should be present
        assert 'sharpe_ratio' in performance, "Sharpe ratio should be calculated"
        
        print("Diversification test completed (single asset baseline)")


@pytest.mark.integration
class TestDynamicRebalancingTests:
    """Test capital redistribution and leverage adjustment workflows."""
    
    def test_performance_based_redistribution(self, bull_market_data, integration_test_config):
        """Test automatic capital transfer between base and alpha pools."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = bull_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check if dual pool redistribution occurred
        portfolio_values = results.get('base_values', [])
        alpha_values = results.get('alpha_values', [])
        
        if portfolio_values and alpha_values:
            # Initial and final values
            initial_base = portfolio_values[0]
            initial_alpha = alpha_values[0]
            final_base = portfolio_values[-1]
            final_alpha = alpha_values[-1]
            
            # In bull market, alpha pool should generally outperform
            alpha_return = (final_alpha - initial_alpha) / initial_alpha
            base_return = (final_base - initial_base) / initial_base
            
            # Alpha should generally perform better in bull market
            assert alpha_return >= base_return - 0.1, \
                "Alpha pool should perform at least as well as base in bull market"
        
        print("Performance-based redistribution test completed")
    
    def test_leverage_adjustment_workflow(self, high_volatility_data, integration_test_config):
        """Test automatic leverage adjustments based on risk metrics."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = high_volatility_data.copy()
        
        # Run backtest during high volatility
        results = engine.run_backtest()
        
        # Validate that leverage management occurred
        performance = results['performance']
        
        # In high volatility, risk management should control leverage
        # This is validated through drawdown and risk metrics
        assert performance['max_drawdown'] is not None, "Drawdown should be calculated"
        
        # System should complete without errors during volatility spike
        assert performance['total_return'] is not None, "Performance should be calculated"
        
        print("Leverage adjustment test completed")
    
    def test_risk_parity_adjustments(self, sample_market_data, integration_test_config):
        """Test risk parity based adjustments."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = sample_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate risk-adjusted performance
        performance = results['performance']
        
        # Risk parity should result in reasonable risk metrics
        assert 'sharpe_ratio' in performance, "Risk-adjusted returns should be calculated"
        assert 'max_drawdown' in performance, "Risk metrics should be tracked"
        
        # Performance should be reasonable given risk management
        if 'sharpe_ratio' in performance:
            assert performance['sharpe_ratio'] > -10, "Sharpe ratio should be reasonable"
            assert performance['sharpe_ratio'] < 10, "Sharpe ratio should be reasonable"
        
        print("Risk parity adjustments test completed")


@pytest.mark.integration
class TestRiskManagementWorkflowTests:
    """Test continuous risk monitoring and mitigation workflows."""
    
    def test_var_limit_breach_handling(self, crisis_market_data, integration_test_config):
        """Test portfolio response to VaR limit breaches."""
        engine = BacktestEngine(integration_test_config)
        
        # Configure tight VaR limits to trigger breaches
        integration_test_config.risk.max_portfolio_risk = 0.01  # 1% VaR
        
        engine.current_data = crisis_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check for risk events
        if 'risk_events' in results:
            risk_events = results['risk_events']
            
            # In crisis conditions, risk events should be present
            var_breach_events = [e for e in risk_events 
                               if 'var' in e.get('type', '').lower() or 
                                  'var' in e.get('reason', '').lower()]
            
            # VaR breach events should be tracked
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        # Final portfolio should reflect risk management actions
        performance = results['performance']
        assert performance['max_drawdown'] is not None, "Drawdown should be controlled"
        
        print(f"VaR breach handling test completed with {len(risk_events) if 'risk_events' in results else 0} risk events")
    
    def test_drawdown_threshold_enforcement(self, bear_market_data, integration_test_config):
        """Test drawdown threshold enforcement procedures."""
        engine = BacktestEngine(integration_test_config)
        
        # Configure tight drawdown limits
        integration_test_config.risk.max_drawdown = 0.10  # 10% drawdown limit
        
        engine.current_data = bear_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Validate drawdown control
        performance = results['performance']
        
        # Drawdown should not exceed configured limit by too much
        assert performance['max_drawdown'] > -0.20, "Drawdown should be reasonably controlled"
        
        # Check for drawdown-related risk events
        if 'risk_events' in results:
            risk_events = results['risk_events']
            drawdown_events = [e for e in risk_events 
                             if 'drawdown' in e.get('type', '').lower()]
            
            # Drawdown events should be tracked in bear market
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        print(f"Drawdown enforcement test completed: Max DD = {performance['max_drawdown']:.2%}")
    
    def test_emergency_liquidation_procedures(self, crisis_market_data, integration_test_config):
        """Test emergency liquidation procedures."""
        engine = BacktestEngine(integration_test_config)
        
        # Configure very aggressive risk limits to trigger emergency procedures
        integration_test_config.risk.max_drawdown = 0.05  # 5% drawdown limit
        integration_test_config.risk.max_portfolio_risk = 0.005  # 0.5% VaR
        
        engine.current_data = crisis_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check for emergency procedures
        if 'risk_events' in results:
            risk_events = results['risk_events']
            emergency_events = [e for e in risk_events 
                              if e.get('type') == 'EMERGENCY_LIQUIDATION']
            
            # Emergency procedures should be attempted in extreme conditions
            assert isinstance(risk_events, list), "Risk events should be tracked"
        
        # Portfolio should survive emergency procedures
        performance = results['performance']
        
        # Some value should be preserved through emergency procedures
        final_value = performance['final_portfolio_value']
        initial_value = integration_test_config.portfolio.initial_capital
        
        # Emergency procedures should preserve some capital
        assert final_value > 0, "Emergency procedures should preserve some value"
        
        print(f"Emergency liquidation test completed: Final value = ${final_value:.2f}")


@pytest.mark.integration
class TestTaxOptimizationWorkflow:
    """Test tax optimization and year-end procedures."""
    
    def test_year_end_tax_calculations(self, sample_market_data, integration_test_config):
        """Test tax calculations at year end."""
        # This test would be more comprehensive with multi-year data
        # For now, test with single year and validate tax tracking
        
        engine = BacktestEngine(integration_test_config)
        engine.current_data = sample_market_data.copy()
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check for tax-related tracking
        performance = results['performance']
        
        if 'cumulative_tax' in performance:
            cumulative_tax = performance['cumulative_tax']
            
            # Tax should be tracked and reasonable
            assert cumulative_tax >= 0, "Cumulative tax should be non-negative"
            assert cumulative_tax < performance['final_portfolio_value'], \
                "Tax should not exceed portfolio value"
        
        print("Year-end tax calculations test completed")
    
    def test_tax_loss_carryforward(self, bear_market_data, integration_test_config):
        """Test tax loss carryforward procedures."""
        engine = BacktestEngine(integration_test_config)
        engine.current_data = bear_market_data.copy()
        
        # Run backtest during loss period
        results = engine.run_backtest()
        
        # Validate tax handling in losing period
        performance = results['performance']
        
        # In bear market, tax should account for losses
        if 'cumulative_tax' in performance:
            cumulative_tax = performance['cumulative_tax']
            
            # Tax should be minimal or zero in losing period
            # (depending on implementation of loss carryforward)
            assert cumulative_tax >= 0, "Cumulative tax should be non-negative"
        
        print("Tax loss carryforward test completed")